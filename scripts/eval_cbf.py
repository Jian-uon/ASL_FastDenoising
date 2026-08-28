#!/usr/bin/env python
"""Paper-1 (IEEE TMI) E1 + E4: accelerated vs full-acquisition CBF agreement and the
2->12-frame degradation curve, on the training-format data (per-subject raw M0).

Pipeline (per subject, per n_frames):
  1. run the trained denoiser on n randomly-drawn frames  (reuse scripts/infer_osipi)
  2. de-normalise its output to raw-intensity dM  (pwi * scale + lo)
  3. resample that dM back to the NATIVE grid (real affine) so it aligns with M0/masks
  4. dM -> CBF via utils.cbf (real M0 + nominal 7T single-PLD pCASL params)
  5. record regional GM / WM / whole-brain mean CBF

Reference = the model at n=12 (isolates the effect of FEWER frames only). Across
subjects we then report ICC(2,1)/Bland-Altman of accel-vs-full regional CBF and the
degradation curve. Absolute params are nominal (accel-vs-full agreement is invariant
to them); CBF also reported as rCBF (normalized to whole-brain GM+WM mean).

Run in WSL asl-mamba (CUDA + mamba_ssm). Example:
  python scripts/eval_cbf.py \
    --checkpoint /mnt/g/training/ASL_dmvae/logs/cig_vss_25d_seg_seed42/checkpoints/latest.pth \
    --config env/local/configs/wsl_asl_2d_home_v37.yml \
    --data_root /mnt/d/data/ASL/ASL_denoising_dataset/data \
    --runner_args "--use_vmamba_encoder --vmamba_n_directions 2 --vmamba_convffn --vmamba_overlap_merge --vmamba_lrda_inscan --base_ch 32 --depth 4 --t1_task seg" \
    --slice_context 2 --n_frames 2 4 6 8 10 12 --max_subjects 20
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import nibabel as nib
from nibabel.processing import resample_from_to

from scripts.infer_osipi import build_model, load_ema, preproc, infer_volume
from utils.cbf import dm_to_cbf, net_alpha, qc_cbf_ranges
from utils.cbf_metrics import icc_agreement, bland_altman


def _load(path):
    img = nib.load(path)
    return np.asarray(img.get_fdata(), dtype=np.float64), img


def regional_cbf(dm_native, m0, brain, gm, wm, params):
    """dM (native) + M0 -> CBF, return GM/WM/whole-brain mean CBF + rCBF (norm to GM+WM)."""
    cbf = dm_to_cbf(dm_native, m0, mask=brain, m0_floor=1.0, **params)["cbf"]
    gmask, wmask = gm > 0.5, wm > 0.5
    ref = cbf[gmask | wmask].mean() if (gmask | wmask).any() else np.nan   # rCBF reference
    out = {}
    for name, mk in (("gm", gmask), ("wm", wmask), ("brain", brain > 0.5)):
        v = float(cbf[mk].mean()) if mk.any() else float("nan")
        out[f"cbf_{name}"] = v
        out[f"rcbf_{name}"] = float(v / ref) if ref else float("nan")
    return out, cbf


def recon_fidelity(dm_n, dm_ref, brain):
    """Recon-space agreement of accelerated vs reference dM within brain."""
    b = brain > 0.5
    a, r = dm_n[b], dm_ref[b]
    corr = float(np.corrcoef(a, r)[0, 1]) if a.size > 2 else float("nan")
    nrmse = float(np.sqrt(np.mean((a - r) ** 2)) / (r.mean() + 1e-6))
    return {"recon_corr": corr, "recon_nrmse": nrmse}


def _rcbf_png(sid, out_png, t1, dm_native, rcbf_maps, brain, ns, ref_frames, ref_cbf=None,
              cmap="turbo", vmax=2.0, dpi=200, erode=1):
    """rCBF montage: rows are slices, columns are T1, the denoised dM, then rCBF at every k.

    All rCBF panels share one window (0 .. vmax x the whole-brain mean) and one colorbar, so a
    column is read against the other columns instead of against its own private scale. Voxels
    outside the brain are set to NaN rather than 0: on a blue-to-red map 0 is dark blue, which
    would paint the background as if it carried the lowest perfusion.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except Exception as e:
        print(f"  [qc] matplotlib unavailable ({e}); skip PNG")
        return
    accel_n = min([n for n in ns if n != ref_frames], default=ref_frames)
    b = brain > 0.5
    if erode:
        # CBF is dM/M0, and M0 collapses in the partial-volume shell at the brain edge, so the
        # outermost voxels carry a bright rim that is quantification, not perfusion. It is
        # hidden here for display only; the saved NIfTI and every reported number are untouched.
        try:
            from scipy.ndimage import binary_erosion
            b = binary_erosion(b, iterations=int(erode))
        except Exception:
            pass
    Z = t1.shape[2]
    zs = [int(Z * f) for f in (0.4, 0.5, 0.6)]

    cols = [("T1", t1, "gray", None),
            (rf"denoised $\Delta$M, {accel_n} rep", dm_native[accel_n], "gray", None)]
    for n in sorted(rcbf_maps):
        lab = f"rCBF, {n} rep" + (" (full)" if n == ref_frames else "")
        cols.append((lab, rcbf_maps[n], cmap, (0.0, float(vmax))))
    if ref_cbf is not None:
        cols.append(("reference CBF", ref_cbf, cmap, (0.0, float(vmax))))

    fig, ax = plt.subplots(len(zs), len(cols),
                           figsize=(2.2 * len(cols) + 0.8, 2.2 * len(zs)))
    ax = np.atleast_2d(ax)
    cbar_axes = []
    for ri, z in enumerate(zs):
        for ci, (title, vol, cm, win) in enumerate(cols):
            axx = ax[ri, ci]
            img = vol[:, :, z]
            if win is not None:
                img = np.where(b[:, :, z], img, np.nan)      # background stays unpainted
                vmin, vhi = win
                cbar_axes.append(axx)
            else:
                vv = img[img != 0]
                vmin, vhi = ((float(np.percentile(vv, 2)), float(np.percentile(vv, 98)))
                             if vv.size else (0.0, 1.0))
            axx.imshow(np.rot90(img), cmap=cm, vmin=vmin, vmax=max(vhi, vmin + 1e-6),
                       interpolation="nearest")
            axx.set_title(title if ri == 0 else "", fontsize=9)
            axx.axis("off")

    if cbar_axes:
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=float(vmax)),
                            cmap=plt.get_cmap(cmap))
        cb = fig.colorbar(sm, ax=cbar_axes, fraction=0.020, pad=0.012)
        cb.set_label("rCBF (fraction of whole-brain mean)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(sid, fontsize=11)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"  [qc] {out_png}")


def _degradation_plot(summary, out_png):
    """E4 degradation figure: auto-discover ALL numeric metric columns, plot each vs n_frames."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable ({e}); skip"); return
    if not summary:
        return
    ns = [r["n_frames"] for r in summary]
    metrics = [k for k in summary[0]
               if k not in ("n_frames", "n_subjects") and isinstance(summary[0][k], (int, float))]
    ncol = 3; nrow = int(np.ceil(len(metrics) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for i, m in enumerate(metrics):
        ax[i].plot(ns, [r.get(m, float("nan")) for r in summary], "o-")
        ax[i].set_title(m, fontsize=9); ax[i].set_xlabel("n_frames"); ax[i].grid(alpha=0.3)
    for j in range(len(metrics), len(ax)):
        ax[j].axis("off")
    fig.suptitle("E4 degradation — all metrics vs n_frames", fontsize=11); fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  [plot] {out_png}")


def main():
    ap = argparse.ArgumentParser("Paper-1 E1/E4 CBF-space evaluation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True, help="root with sub-*/raw/{asldata_diff,m0,...}")
    ap.add_argument("--runner_args", default="")
    ap.add_argument("--slice_context", type=int, default=0)
    ap.add_argument("--n_frames", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--ref_frames", type=int, default=12, help="reference = model at this n")
    ap.add_argument("--max_subjects", type=int, default=20)
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relu_input", action="store_true", default=True)
    ap.add_argument("--out_dir", default="./exp/eval_cbf")
    ap.add_argument("--save_maps", action="store_true",
                    help="save per-subject CBF/rCBF NIfTI + a QC montage PNG")
    ap.add_argument("--rcbf_cmap", default="turbo",
                    help="colormap for the rCBF panels; 'turbo' is the blue-to-red ramp with "
                         "jet's appearance but without its perceptual false edges.")
    ap.add_argument("--rcbf_vmax", type=float, default=2.0,
                    help="top of the shared rCBF window, in units of the whole-brain mean.")
    ap.add_argument("--qc_dpi", type=int, default=200)
    ap.add_argument("--show_ref_cbf", action="store_true",
                    help="add the dataset's own cbf.nii.gz as a column. Off by default: it is "
                         "an externally computed map whose left/right GM+WM ratio ranges from "
                         "0.48 to 1.72 across these subjects, against 0.87-1.12 for the "
                         "reconstructions, so it carries artefacts this study did not produce "
                         "and does not serve as its reference. The reference here is the "
                         "12-repetition reconstruction, already a column.")
    ap.add_argument("--rcbf_erode", type=int, default=1,
                    help="voxels of brain-edge erosion for the montage only. The edge shell "
                         "carries a bright CBF rim from the M0 partial volume; 0 shows it.")
    # nominal 7T single-PLD pCASL (accel-vs-full agreement is invariant to these)
    ap.add_argument("--pld", type=float, default=1.8)
    ap.add_argument("--ld", type=float, default=1.5)
    ap.add_argument("--t1_blood", type=float, default=2.2)
    ap.add_argument("--alpha_label", type=float, default=0.85)
    ap.add_argument("--n_bs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    params = dict(pld=args.pld, ld=args.ld, t1_blood=args.t1_blood, lam=0.9,
                  alpha=net_alpha(args.alpha_label, 0.93, args.n_bs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  params={params}")
    model, cfg = build_model(args.runner_args, args.config, device, slice_context=args.slice_context)
    ok = load_ema(model, args.checkpoint, device)
    if not ok:
        print("[FATAL] arch mismatch -> fix --runner_args to match the ckpt arch; aborting.")
        sys.exit(2)
    tp = cfg.asl_denoiser_train_params

    subs = sorted(glob.glob(os.path.join(args.data_root, "sub-*")))
    if args.subjects:
        want = set(args.subjects); subs = [s for s in subs if os.path.basename(s) in want]
    subs = subs[: args.max_subjects]
    ns = sorted(set(args.n_frames) | {args.ref_frames})
    print(f"[data] {len(subs)} subjects; n_frames={ns}; ref=n{args.ref_frames}")

    rows = []            # one dict per (subject, n)
    for sdir in subs:
        sid = os.path.basename(sdir); raw = os.path.join(sdir, "raw")
        ap_ = os.path.join(raw, "asldata_diff.nii.gz"); t1p = os.path.join(raw, "t1_in_asl.nii.gz")
        m0p = os.path.join(raw, "m0.nii.gz")
        if not (os.path.isfile(ap_) and os.path.isfile(m0p)):
            print(f"  [skip] {sid}: missing asldata_diff/m0"); continue
        m0, _ = _load(m0p)
        brain, _ = _load(os.path.join(raw, "brain_mask_asl.nii.gz"))
        gm, _ = _load(os.path.join(raw, "gm_asl.nii.gz")); wm, _ = _load(os.path.join(raw, "wm_asl.nii.gz"))
        nat_img = nib.load(ap_); nat_shape, nat_aff = nat_img.shape[:3], nat_img.affine
        T = nat_img.shape[-1]
        asl_vol, t1_vol, _, affine = preproc(ap_, t1p, tp, raw_dir=raw, relu_input=args.relu_input)

        dm_native = {}
        for n in ns:
            nf = min(n, T)
            pwi, nused, lo, sc = infer_volume(model, asl_vol, t1_vol, nf, device,
                                              seed=args.seed, slice_context=args.slice_context)
            deltam_raw = pwi * sc + lo                                   # de-normalise -> raw dM (model grid)
            res = resample_from_to(nib.Nifti1Image(deltam_raw.astype(np.float32), affine),
                                   (nat_shape, nat_aff), order=1, cval=0.0)
            dm_native[n] = np.clip(np.asarray(res.dataobj, dtype=np.float64), 0, None) * (brain > 0.5)

        ref = dm_native[args.ref_frames]
        rcbf_maps = {}
        for n in ns:
            reg, cbf = regional_cbf(dm_native[n], m0, brain, gm, wm, params)
            fid = recon_fidelity(dm_native[n], ref, brain)
            rows.append({"subject": sid, "n_frames": n, **reg, **fid})
            if args.save_maps:
                bm = brain > 0.5
                # Same reference as regional_cbf: the GM+WM mean, not the whole-brain mean.
                # The whole-brain mean is inflated by CSF and by the partial-volume rim at the
                # brain edge, which would put the maps on a different scale from the rcbf_gm /
                # rcbf_wm numbers reported beside them.
                tis = (gm > 0.5) | (wm > 0.5)
                rb = float(cbf[tis].mean()) if tis.any() else 1.0
                rcbf = cbf / (rb + 1e-6) * bm
                rcbf_maps[n] = rcbf
                odir = os.path.join(args.out_dir, sid); os.makedirs(odir, exist_ok=True)
                nib.save(nib.Nifti1Image(cbf.astype(np.float32), nat_aff), os.path.join(odir, f"cbf_n{n}.nii.gz"))
                nib.save(nib.Nifti1Image(rcbf.astype(np.float32), nat_aff), os.path.join(odir, f"rcbf_n{n}.nii.gz"))
        if args.save_maps:
            t1n = _load(t1p)[0]
            refc_p = os.path.join(raw, "cbf.nii.gz")
            refc = _load(refc_p)[0] if (args.show_ref_cbf and os.path.isfile(refc_p)) else None
            if refc is not None:
                # cbf.nii.gz is absolute ml/100g/min (in-brain mean ~40); drawn on the shared
                # 0-2 rCBF window it saturates to a solid block. Normalise it the same way.
                _t = (gm > 0.5) | (wm > 0.5)
                _r = float(refc[_t].mean()) if _t.any() else 0.0
                refc = (refc / _r) if _r > 0 else None
            _rcbf_png(sid, os.path.join(args.out_dir, "qc", f"{sid}.png"),
                      t1n, dm_native, rcbf_maps, brain, ns, args.ref_frames, refc,
                      cmap=args.rcbf_cmap, vmax=args.rcbf_vmax, dpi=args.qc_dpi,
                      erode=args.rcbf_erode)
        g = rows[-1]["cbf_gm"]
        print(f"  [ok] {sid}: n{args.ref_frames} GM-CBF={g:.1f}  (rows so far {len(rows)})")

    # ---- aggregate: E1 agreement (accel vs ref) + E4 degradation (full metric suite) ----
    by_n = {n: [r for r in rows if r["n_frames"] == n] for n in ns}
    ref_n = args.ref_frames
    ref_by_sid = {r["subject"]: r for r in by_n[ref_n]}
    col = lambda d, sids, k: np.array([d[s][k] for s in sids], dtype=float)
    summary = []
    for n in ns:
        cur = {r["subject"]: r for r in by_n[n]}
        sids = [s for s in cur if s in ref_by_sid]
        row = {"n_frames": n, "n_subjects": len(cur)}
        for k in ("cbf_gm", "cbf_wm", "rcbf_gm", "rcbf_wm"):
            row[k] = float(np.nanmean([cur[s][k] for s in cur]))
        row["gm_wm_ratio"] = row["cbf_gm"] / row["cbf_wm"] if row["cbf_wm"] else float("nan")
        row["recon_corr"] = float(np.nanmean([cur[s]["recon_corr"] for s in cur]))
        row["recon_nrmse"] = float(np.nanmean([cur[s]["recon_nrmse"] for s in cur]))
        if n == ref_n:
            row["icc_rcbf_gm"] = row["icc_rcbf_wm"] = 1.0; row["ba_bias_rcbf_gm"] = 0.0
        elif len(sids) >= 2:
            row["icc_rcbf_gm"] = icc_agreement(col(cur, sids, "rcbf_gm"), col(ref_by_sid, sids, "rcbf_gm"))
            row["icc_rcbf_wm"] = icc_agreement(col(cur, sids, "rcbf_wm"), col(ref_by_sid, sids, "rcbf_wm"))
            row["ba_bias_rcbf_gm"] = bland_altman(col(cur, sids, "rcbf_gm"), col(ref_by_sid, sids, "rcbf_gm"))["bias"]
        else:
            row["icc_rcbf_gm"] = row["icc_rcbf_wm"] = row["ba_bias_rcbf_gm"] = float("nan")
        summary.append(row)

    print("\n=== E4 degradation + E1 agreement (ref = model @ n%d, %d subjects) ===" % (ref_n, len(ref_by_sid)))
    hdr = ["n_frames", "cbf_gm", "cbf_wm", "gm_wm_ratio", "rcbf_gm",
           "icc_rcbf_gm", "icc_rcbf_wm", "ba_bias_rcbf_gm", "recon_corr", "recon_nrmse"]
    print(" | ".join(f"{h:>13}" for h in hdr))
    for r in summary:
        print(" | ".join((f"{r.get(h):>13}" if isinstance(r.get(h), int)
                          else f"{r.get(h, float('nan')):>13.3f}") for h in hdr))

    with open(os.path.join(args.out_dir, "cbf_eval.json"), "w") as f:
        json.dump({"params": params, "ref_frames": ref_n, "summary": summary, "rows": rows}, f, indent=2)
    _degradation_plot(summary, os.path.join(args.out_dir, "degradation_all_metrics.png"))
    print(f"[done] wrote {os.path.join(args.out_dir, 'cbf_eval.json')}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
