# -*- coding: utf-8 -*-
"""
test_signal_injection.py — E0.3 focal-anomaly signal-preservation test.

The gate's REAL job (docs/cig_vss.md, manuscript §4.4) is lesion / anomaly
PRESERVATION: at a genuine perfusion anomaly the ASL feature departs from its
partial-volume tissue expectation, the evidence for the PV-tissue model falls,
and the gate should ATTENUATE the anatomical modulation so the anomaly is kept
rather than reshaped toward the "normal tissue" prior.

This script measures that directly. For each subject it injects a synthetic
focal perfusion anomaly (a Gaussian hyper-perfusion blob of controlled contrast
`--inject_amp` × GM-reference, hence "E0.3" at the default 0.3) into ALL ASL
difference frames at a GM location where the anatomy says "normal tissue", runs
inference on the clean and injected inputs (SAME frame sampling + SAME output
normalisation), and reports, per injected slice:

  • retention        — recovered / injected contrast in the blob core (1 = kept,
                       0 = smoothed away, >1 = over-sharpened). THE endpoint.
  • spillover        — |Δ| leaked into a surrounding ring / injected peak (blur).
  • gamma_roi_clean / gamma_roi_inj / dgamma
                     — the gate value in the ROI (NaN for gate-less models);
                       for the Bayesian/cosine gate dgamma should be NEGATIVE
                       (the gate CLOSES at the anomaly), the mechanistic witness.

Run ONE model per invocation (like test_mismatched_t1.py); compare main vs −EC
vs baseline by running it per checkpoint and aggregating. V=ASL is untouched —
the anomaly is injected into the ASL stream, never into T1/PV.

Usage
-----
python scripts/test_signal_injection.py \
    --checkpoint $EXP/logs/cig_vss_ec_csem_bayes_seed42/best_cnr.pth \
    --config     env/hpc/configs/server_v37.yml \
    --runner_args "<same arch flags used to train>" \
    --out_dir    $EXP/e03_injection/bayes --inject_amp 0.3 --max_subjects 12
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/ (for test_mismatched_t1)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config.conf_data import Config
from dataio.data_generator import DatasetGenerator
from dataio.data_classes import get_pre_asl_transform, build_cached_subjects, _affine_from_frames
from monai.data import partition_dataset

# Reuse the EXACT loader + reference used by the mismatched-T1 gate (same harness).
from test_mismatched_t1 import load_model, direct_mean_volume


def parse_args():
    p = argparse.ArgumentParser("E0.3 focal-anomaly signal-preservation test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--out_dir",    default="./exp/e03_injection")
    p.add_argument("--runner_args", default="",
                   help="Training-style arch flags as one string (must match the ckpt).")
    p.add_argument("--n_frames",   type=int, default=6)
    p.add_argument("--split",      default="val", choices=["train", "val", "test"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--cache_rate", type=float, default=0.0)
    p.add_argument("--label",      default="", help="Model label written into the CSV (defaults to ckpt dir).")
    # injection knobs
    p.add_argument("--inject_amp",   type=float, default=0.3,
                   help="Blob peak = inject_amp × GM-reference level (RAW ΔM units). 0.3 = the E0.3 point.")
    p.add_argument("--inject_sigma", type=float, default=3.0, help="Blob Gaussian sigma (voxels).")
    p.add_argument("--n_slices",     type=int,   default=3, help="Injected z-slices per subject (largest-brain first).")
    p.add_argument("--gm_pct",       type=float, default=60.0, help="GM ref-proxy band: ref ≥ this percentile (legacy GM path).")
    p.add_argument("--inject_tissue", default="gm", choices=["gm", "wm"],
                   help="Tissue the blob is placed in: gm (high-perfusion band, DEFAULT) or "
                        "wm (low-perfusion band — HARDER: the anatomical prior says 'no perfusion "
                        "here', so erasing the anomaly toward the prior is the failure mode).")
    p.add_argument("--tissue_pct",   type=float, default=70.0,
                   help="Tissue-band percentile for the WM / FSL-PV-map placement "
                        "(FSL PV ≥ pct-percentile, or ref-proxy ≤ (100−pct)-percentile for wm).")
    # amplitude relative to the TARGET TISSUE's own reasonable range (physiologically-grounded)
    p.add_argument("--amp_mode", default="gm_ref", choices=["gm_ref", "tissue_sigma"],
                   help="How the blob peak is set. gm_ref (legacy) = inject_amp x brain-median ΔM "
                        "(NOT tissue-aware). tissue_sigma = n_sigma x the TARGET tissue's own "
                        "robust sigma, so the blob core lands OUTSIDE that tissue's reasonable "
                        "range (mu_t +/- n_sigma*sigma_t); n_sigma>=3 = a genuine anomaly for THAT tissue.")
    p.add_argument("--n_sigma", type=float, default=4.0,
                   help="tissue_sigma mode: peak = n_sigma x sigma_target_tissue above/below its mean.")
    p.add_argument("--inject_dir", default="hyper", choices=["hyper", "hypo"],
                   help="hyper = hyper-perfusion anomaly (e.g. tumor/AVM), value ABOVE the tissue "
                        "range; hypo = hypo-perfusion (e.g. ischemia), value BELOW the tissue range.")
    p.add_argument("--pure_thr", type=float, default=0.9,
                   help="tissue_sigma mode: FSL PV >= this = 'pure' tissue voxel for the normal-range stats.")
    p.add_argument("--device",     default="")
    return p.parse_args()


def _tissue_stats(pv_vol, ref: np.ndarray, brain: np.ndarray, thr: float):
    """Robust (median mu_t, MAD-sigma sigma_t) of RAW ΔM over PURE-tissue voxels
    (FSL PV >= thr AND brain) — the tissue's 'reasonable range'. Returns None if
    there is no PV map or too few pure voxels. Used to place an injected anomaly
    OUTSIDE the target tissue's normal distribution (mu_t +/- n_sigma*sigma_t)."""
    if pv_vol is None:
        return None
    m = (pv_vol[0].numpy() >= thr) & brain
    if m.sum() < 50:
        m = (pv_vol[0].numpy() >= max(0.5, thr - 0.2)) & brain   # relax if too few
    if m.sum() < 50:
        return None
    v = ref[m].astype(np.float64)
    mu = float(np.median(v))
    sd = float(1.4826 * np.median(np.abs(v - mu)) + 1e-8)        # robust sigma (MAD)
    return mu, sd


def _gauss2d(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))).astype(np.float32)


def _pick_roi_center(bz: np.ndarray, ref_z: np.ndarray, tissue: str, gm_pct: float,
                     tissue_pct: float, tissue_map: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """A reproducible voxel in the requested TISSUE band near the brain centroid.

    Placement is MODEL-INDEPENDENT (so the same anomaly is scored across every arm):
    prefers the FSL PV map (`tissue_map`, same input for all models) when supplied,
    else a ref-proxy — GM = high-perfusion band, WM = low-perfusion band. A WM blob is
    the harder test: the anatomical prior expects ~no perfusion there, so any erasure
    toward the prior shows up as low retention."""
    ys, xs = np.where(bz)
    cy0, cx0 = ys.mean(), xs.mean()
    if tissue_map is not None:                                   # FSL PV map (preferred)
        thr = max(float(np.percentile(tissue_map[bz], tissue_pct)), 0.5)
        sel = bz & (tissue_map >= thr)
    elif tissue == "wm":                                         # ref-proxy: low-perfusion band
        thr = float(np.percentile(ref_z[bz], 100.0 - tissue_pct))
        sel = bz & (ref_z <= thr)
    else:                                                        # gm ref-proxy (legacy, gm_pct=60)
        thr = float(np.percentile(ref_z[bz], gm_pct))
        sel = bz & (ref_z >= thr)
    if not sel.any():
        sel = bz
    gy, gx = np.where(sel)
    j = np.argmin((gy - cy0) ** 2 + (gx - cx0) ** 2)
    return int(gy[j]), int(gx[j])


@torch.no_grad()
def infer_volume_g(model, asl_vol: torch.Tensor, t1_vol: torch.Tensor, n_frames: int,
                   device: torch.device, seed: int,
                   fix_affine: Optional[Tuple[float, float]] = None,
                   gm: Optional[torch.Tensor] = None, wm: Optional[torch.Tensor] = None,
                   csf: Optional[torch.Tensor] = None):
    """Like test_mismatched_t1.infer_volume but (a) optionally reuses a FIXED
    (lo, scale) so clean vs injected land in the SAME output space, (b) also
    returns the gate γ upsampled to [H,W,Z] (NaN volume for gate-less models), and
    (c) for cond_src='fsl' supplies the brain-masked FSL GM/WM/CSF conditioning PV
    per z-slice via set_cond_pv (V=ASL: the PV only modulates SSM B/Δ, never Value).
    The anomaly is injected into the ASL stream only; gm/wm/csf are the clean priors."""
    _cond_src = str(getattr(model, "lrda_cond_src", "pv"))
    _is_fsl = _cond_src in ("fsl", "fsl_t1")
    T, H, W, Z = asl_vol.shape
    rng = np.random.default_rng(seed)
    n = min(n_frames, T)
    idx = rng.choice(T, size=n, replace=False).tolist()
    frames = asl_vol[idx]                                   # [n,H,W,Z] RAW
    lo, scale = fix_affine if fix_affine is not None else _affine_from_frames(frames)
    frames = (frames - lo) / scale

    ctx = int(getattr(model, "slice_context", 0))
    pwi = np.zeros((H, W, Z), dtype=np.float32)
    gam = np.full((H, W, Z), np.nan, dtype=np.float32)
    pvcb = np.full((H, W, Z), np.nan, dtype=np.float32)   # PVC base E (NaN if --pvc_residual off)
    pvcw = np.full((H, W, Z), np.nan, dtype=np.float32)   # PVC reproducibility gate w
    lengths = torch.tensor([n], dtype=torch.long, device=device)
    for z in range(Z):
        if ctx > 0:
            zc = [min(max(z + dz, 0), Z - 1) for dz in range(-ctx, ctx + 1)]
            sl = frames[:, :, :, zc].permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=torch.float32)
        else:
            sl = frames[:, :, :, z].unsqueeze(1).unsqueeze(0).to(device, dtype=torch.float32)
        t1 = t1_vol[:, :, :, z].unsqueeze(0).to(device, dtype=torch.float32)
        if _is_fsl and gm is not None:
            from models.asl_t1_model import build_fsl_cond_pv
            model.set_cond_pv(build_fsl_cond_pv(
                gm[:, :, :, z].unsqueeze(0).to(device, dtype=torch.float32),
                wm[:, :, :, z].unsqueeze(0).to(device, dtype=torch.float32),
                csf[:, :, :, z].unsqueeze(0).to(device, dtype=torch.float32), t1=t1,
                with_t1=(_cond_src == "fsl_t1")))
        out = model.infer_from_subset(sl, t1, lengths=lengths)
        pwi[:, :, z] = out["asl_recon"][0, 0].cpu().numpy()
        g = out.get("ec_gamma")
        if g is not None:
            g = F.interpolate(g.float(), size=(H, W), mode="bilinear", align_corners=False)
            gam[:, :, z] = g[0, 0].cpu().numpy()
        eb, ew = out.get("pvc_base"), out.get("pvc_w")   # image-space full-res (None unless --pvc_residual)
        if eb is not None:
            pvcb[:, :, z] = eb[0, 0].cpu().numpy()
        if ew is not None:
            pvcw[:, :, z] = ew[0, 0].cpu().numpy()
    return pwi, gam, pvcb, pvcw, float(lo), float(scale)


def _save_injection_panels(out_dir, sid, z, label, args, blob, bmask, ref_slice, scale,
                           rc, ri, gc, gi, eb, ew, retention):
    """One comprehensive qualitative figure per (subject, slice): inputs, recon
    clean/injected, the injected-signal GT vs the recovered Δ-recon (SAME colour
    scale ⇒ the erased fraction is visible by eye), the γ gate (clean/inj/Δ), the
    PVC base E + reproducibility gate w (when --pvc_residual), and a zoomed ROI on
    the blob. Panels that don't apply to a model are skipped automatically."""
    g, amp_raw, (cy, cx) = blob
    H, W = ref_slice.shape
    bm = bmask.astype(np.float32)
    ref_z = ref_slice.astype(np.float32)
    blob_out = (amp_raw * g / scale).astype(np.float32)          # injected signal, OUTPUT units
    dif = (ri - rc) * bm                                          # recovered change (output units)
    has_gate = np.isfinite(gc).any() or np.isfinite(gi).any()
    has_pvc = np.isfinite(eb).any()

    # (image, title, cmap, symmetric, shared-scale-key)
    P = [(ref_z * bm,              "input mean (clean)",  "gray",   False, "in"),
         ((ref_z + amp_raw * g) * bm, "input mean (+blob)", "gray", False, "in"),
         (rc * bm,                 "recon (clean)",       "gray",   False, "rec"),
         (ri * bm,                 "recon (injected)",    "gray",   False, "rec"),
         (blob_out * bm,           "injected signal (GT)", "magma", False, "sig"),
         (dif,                     "Δ recon (i−c)",       "magma",  False, "sig")]
    if has_gate:
        P += [(np.nan_to_num(gc) * bm, "γ (clean)",    "viridis", False, "gam"),
              (np.nan_to_num(gi) * bm, "γ (injected)", "viridis", False, "gam"),
              ((np.nan_to_num(gi) - np.nan_to_num(gc)) * bm, "Δγ (i−c)", "RdBu_r", True, None)]
    if has_pvc:
        P += [(np.nan_to_num(eb) * bm, "PVC base E=P·m̄", "gray",   False, "rec"),
              (np.nan_to_num(ew) * bm, "repro gate w",   "viridis", False, None)]
    # ROI zoom on the blob (recovered vs GT, output units) — the money panels
    R = int(max(6, round(4 * args.inject_sigma)))
    y0, y1 = max(0, cy - R), min(H, cy + R)
    x0, x1 = max(0, cx - R), min(W, cx + R)
    P += [(ri[y0:y1, x0:x1],       "recon inj [ROI]", "gray",  False, None),
          (dif[y0:y1, x0:x1],      "Δ recon [ROI]",   "magma", False, "sig"),
          (blob_out[y0:y1, x0:x1], "injected [ROI]",  "magma", False, "sig")]

    # shared vmin/vmax per key (1..99 pct over the pooled finite values)
    keys = {}
    for im, _t, _c, _s, k in P:
        if k is None:
            continue
        v = im[np.isfinite(im)]
        if v.size == 0:
            continue
        lo_k, hi_k = np.percentile(v, 1), np.percentile(v, 99)
        keys[k] = (min(keys[k][0], lo_k), max(keys[k][1], hi_k)) if k in keys else (lo_k, hi_k)

    ncol = 6
    nrow = int(np.ceil(len(P) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.3 * nrow))
    ax = np.array(ax).reshape(-1)
    for a, (im, t, cm, sym, k) in zip(ax, P):
        if sym:
            fin = im[np.isfinite(im)]
            m = float(np.percentile(np.abs(fin), 99)) if fin.size else 1.0
            m = m if m > 1e-9 else 1.0
            kw = dict(cmap=cm, vmin=-m, vmax=m)
        elif k in keys:
            kw = dict(cmap=cm, vmin=keys[k][0], vmax=keys[k][1])
        else:
            kw = dict(cmap=cm)
        h = a.imshow(im, **kw); a.set_title(t, fontsize=8); a.axis("off")
        if "ROI" not in t:
            a.plot(cx, cy, "r+", ms=7)
        fig.colorbar(h, ax=a, fraction=0.046, pad=0.04)
    for a in ax[len(P):]:
        a.axis("off")
    _at = (f"{args.inject_dir} {args.n_sigma}σ({args.inject_tissue})" if args.amp_mode == "tissue_sigma"
           else f"amp={args.inject_amp}·GMref")
    fig.suptitle(f"{sid}  z={z}  ({label})   inject={_at}  retention={retention:.3f}  "
                 f"blobσ={args.inject_sigma}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sid}_z{z}_{args.inject_tissue}_injection.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.checkpoint).parent.parent.name
    _adesc = (f"amp_mode={args.amp_mode} n_sigma={args.n_sigma} dir={args.inject_dir}"
              if args.amp_mode == "tissue_sigma" else f"amp_mode=gm_ref inject_amp={args.inject_amp}")
    print(f"[INFO] Device: {device} | out: {out_dir} | label: {label} | tissue={args.inject_tissue} | {_adesc}")

    cfg = Config(args.config)
    model = load_model(args, cfg, device)

    tp = cfg.asl_denoiser_train_params
    pre_tf = get_pre_asl_transform(asl_hw=tp.asl_hw, asl_z=tp.asl_z, t1_hw=tp.t1_hw, t1_z=tp.t1_z)
    subjects = DatasetGenerator(cfg).generate_dataset()
    cached = build_cached_subjects(subjects, pre_tf, cache_rate=float(args.cache_rate), cache_workers=4)
    splits = partition_dataset(
        data=cached,
        ratios=[cfg.data_loading.train_ratio, cfg.data_loading.val_ratio, cfg.data_loading.test_ratio],
        shuffle=cfg.data_loading.shuffle)
    subset = {"train": splits[0], "val": splits[1], "test": splits[2]}[args.split]
    n_subjects = len(subset)
    if args.max_subjects:
        n_subjects = min(n_subjects, args.max_subjects)
    print(f"[INFO] Injecting into {n_subjects} subjects from '{args.split}'")

    rows: List[Dict] = []
    for si in range(n_subjects):
        item = subset[si]
        asl_vol = item["asl_diff"].cpu()                    # [T,H,W,Z] RAW
        t1_vol  = item["t1"].cpu()                          # [1,H,W,Z]
        gm  = item["gm"].cpu()  if item.get("gm")  is not None else None   # [1,H,W,Z] FSL PV
        wm  = item["wm"].cpu()  if item.get("wm")  is not None else None
        csf = item["csf"].cpu() if item.get("csf") is not None else None
        # model-independent tissue map for placement (same prior for every arm)
        tvol = None
        if args.inject_tissue == "gm" and gm is not None:   tvol = gm[0].numpy()
        elif args.inject_tissue == "wm" and wm is not None: tvol = wm[0].numpy()
        sid = item.get("subject_id", f"sub_{si:03d}")
        T, H, W, Z = asl_vol.shape
        brain = (t1_vol[0].numpy() > 0.05)                  # [H,W,Z]
        ref = direct_mean_volume(asl_vol)                   # [H,W,Z] RAW mean (GM>WM, positive)

        # per-subject tissue normal-range stats (raw ΔM) from pure FSL PV voxels — the
        # 'reasonable range' each tissue's injected anomaly must land OUTSIDE of.
        gm_stat = _tissue_stats(gm, ref, brain, args.pure_thr)
        wm_stat = _tissue_stats(wm, ref, brain, args.pure_thr)
        tgt_stat = gm_stat if args.inject_tissue == "gm" else wm_stat
        if args.amp_mode == "tissue_sigma" and tgt_stat is None:
            print(f"[WARN] {sid}: no pure-{args.inject_tissue} PV stats -> falling back to gm_ref amp")

        # target slices = the largest-brain ones
        areas = brain.reshape(-1, Z).sum(0)
        z_list = [int(z) for z in np.argsort(-areas)[:args.n_slices] if areas[z] > 50]
        if not z_list:
            print(f"[{si+1}/{n_subjects}] {sid}: no slice with enough brain — skip"); continue

        # build the injected volume: a Gaussian hyper-perfusion blob in ALL frames at each z
        asl_inj = asl_vol.clone()
        blobs: Dict[int, Tuple[np.ndarray, float, Tuple[int, int]]] = {}
        for z in z_list:
            bz = brain[:, :, z]
            cy, cx = _pick_roi_center(
                bz, ref[:, :, z], args.inject_tissue, args.gm_pct, args.tissue_pct,
                tissue_map=(tvol[:, :, z] if tvol is not None else None))
            gm_ref_raw = float(np.median(ref[:, :, z][bz]))          # brain-median reference (raw)
            if args.amp_mode == "tissue_sigma" and tgt_stat is not None:
                mu_t, sd_t = tgt_stat
                sign = 1.0 if args.inject_dir == "hyper" else -1.0
                amp_raw = sign * args.n_sigma * sd_t                 # peak = n_sigma x tissue-sigma => core outside range
            else:
                amp_raw = args.inject_amp * max(gm_ref_raw, 1e-6)    # legacy: fraction of brain-median
            g = _gauss2d(H, W, cy, cx, args.inject_sigma)           # [H,W] peak 1
            blob_raw = (amp_raw * g).astype(np.float32)
            asl_inj[:, :, :, z] += torch.from_numpy(blob_raw)[None]  # broadcast over T
            blobs[z] = (g, amp_raw, (cy, cx))

        seed_si = args.seed + si
        pwi_c, gam_c, pvcb_c, pvcw_c, lo, scale = infer_volume_g(
            model, asl_vol, t1_vol, args.n_frames, device, seed_si, gm=gm, wm=wm, csf=csf)
        pwi_i, gam_i, pvcb_i, pvcw_i, _, _ = infer_volume_g(
            model, asl_inj, t1_vol, args.n_frames, device, seed_si, fix_affine=(lo, scale),
            gm=gm, wm=wm, csf=csf)

        for z in z_list:
            g, amp_raw, (cy, cx) = blobs[z]
            diff = (pwi_i[:, :, z] - pwi_c[:, :, z]).astype(np.float64)   # normalised units
            core = g >= np.exp(-0.5 * 1.5 ** 2)                          # within 1.5σ
            ring = (g >= np.exp(-0.5 * 3.0 ** 2)) & (g < np.exp(-0.5 * 2.0 ** 2))  # 2σ..3σ shell
            wcore = g[core].astype(np.float64)
            inj_norm_core = (amp_raw / scale) * g[core].astype(np.float64)  # injected signal in output units
            denom = float((wcore * inj_norm_core).sum())
            retention = float((wcore * diff[core]).sum() / denom) if denom > 1e-12 else float("nan")
            peak_norm = amp_raw / scale
            spillover = float(np.abs(diff[ring]).mean() / peak_norm) if (ring.any() and peak_norm > 1e-12) else float("nan")
            gc = float(np.nanmean(gam_c[:, :, z][core])); gi = float(np.nanmean(gam_i[:, :, z][core]))
            # interpretability: where the injected core lands vs the tissue's normal range and
            # on the WM->GM perfusion scale (frac 0=WM level, 1=GM level; >1 = supra-GM).
            core_base = float(np.median(ref[:, :, z][core]))
            core_target = core_base + amp_raw
            mu_t = tgt_stat[0] if tgt_stat else float("nan")
            sd_t = tgt_stat[1] if tgt_stat else float("nan")
            n_sig_actual = ((core_target - mu_t) / sd_t) if (tgt_stat and sd_t > 1e-9) else float("nan")
            gmwm_frac = float("nan")
            if gm_stat and wm_stat and (gm_stat[0] - wm_stat[0]) > 1e-9:
                gmwm_frac = (core_target - wm_stat[0]) / (gm_stat[0] - wm_stat[0])
            rows.append({
                "subject": sid, "z": int(z), "cy": int(cy), "cx": int(cx),
                "tissue": args.inject_tissue, "inject_dir": args.inject_dir, "amp_mode": args.amp_mode,
                "n_sigma": args.n_sigma, "inject_amp": args.inject_amp,
                "mu_t": mu_t, "sigma_t": sd_t, "amp_raw": amp_raw,
                "core_target": core_target, "n_sigma_actual": n_sig_actual, "gmwm_frac": gmwm_frac,
                "retention": retention, "spillover": spillover,
                "gamma_roi_clean": gc, "gamma_roi_inj": gi, "dgamma": (gi - gc),
            })
        print(f"[{si+1}/{n_subjects}] {sid}: "
              f"retention={np.nanmean([r['retention'] for r in rows if r['subject']==sid]):.3f}")

        if si < 3:   # comprehensive qualitative panels for the first few subjects, EVERY target slice
            for z in z_list:
                _save_injection_panels(
                    out_dir, sid, int(z), label, args,
                    blobs[z], brain[:, :, z], ref[:, :, z], scale,
                    pwi_c[:, :, z], pwi_i[:, :, z], gam_c[:, :, z], gam_i[:, :, z],
                    pvcb_i[:, :, z], pvcw_i[:, :, z],
                    next((r["retention"] for r in rows
                          if r["subject"] == sid and r["z"] == int(z)), float("nan")))

    csv_path = out_dir / "results.csv"
    if rows:
        for r in rows:
            r["model"] = label
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        print("\n" + "=" * 60 + f"\nSUMMARY  ({label}, tissue={args.inject_tissue}, {_adesc})\n" + "=" * 60)
        for k in ("retention", "spillover", "dgamma", "n_sigma_actual", "gmwm_frac"):
            v = np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
            print(f"  {k:<16s}: {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")
        print("  retention→1 = anomaly preserved; n_sigma_actual = how far outside the tissue's normal "
              "range the core lands; gmwm_frac: 0=WM level, 1=GM level, >1 supra-GM.")
        print(f"\nFull results → {csv_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
