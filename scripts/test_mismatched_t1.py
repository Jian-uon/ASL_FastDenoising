# -*- coding: utf-8 -*-
"""
test_mismatched_t1.py — Mismatched-T1 sanity check.

Loads a trained model and evaluates two conditions for every subject in the
validation set:
  (a) matched_t1   — correct T1 from the same subject (normal inference)
  (b) mismatched_t1 — T1 replaced with a randomly shuffled other subject's T1

Compares PWI output between conditions. A well-behaved model should show:
  • Small structural layout change (acceptable — low-freq anatomical prior)
  • No direct T1 texture printed onto PWI (hallucination indicator)

Metrics reported per-subject:
  • L1 / PSNR / SSIM between matched and mismatched PWI outputs
  • L1 between each output and the direct mean of all frames (reference PWI)

Outputs a CSV summary and per-subject PNG comparisons.

Usage
-----
python scripts/test_mismatched_t1.py \
    --checkpoint exp/logs/run_full/checkpoints/best.pth \
    --config     env/local/configs/win_asl_2d_home.yml \
    --out_dir    exp/mismatched_t1_test \
    --n_frames   6 \
    --seed       42
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from models.asl_t1_model import ASLT1Denoiser
from monai.data import partition_dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Mismatched-T1 structure-leakage test")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--config",         required=True)
    p.add_argument("--out_dir",        default="./exp/mismatched_t1_test")
    p.add_argument("--n_frames",       type=int, default=6,
                   help="Number of ASL frames to use per inference")
    p.add_argument("--split",          default="val", choices=["train", "val", "test"],
                   help="Which data split to evaluate on")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--max_subjects",   type=int, default=None,
                   help="Limit number of subjects (for quick testing)")
    p.add_argument("--cache_rate",     type=float, default=1.0,
                   help="MONAI cache fraction for preprocessed subjects. 1.0 caches ALL "
                        "subjects up front (wasteful when only --max_subjects are scored -- "
                        "it preloads the whole cohort); 0.0 lazy-loads on access so only the "
                        "few evaluated subjects are read. Use 0 for a small-subset gate.")
    # Model architecture — must match checkpoint
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth",   type=int, default=4)
    p.add_argument("--device",  default="")
    # Build the model via the training arg-parser/Runner so MoSSM/CADA/CADA-LR/
    # hybrid checkpoints load strictly. Pass the SAME arch flags used in training
    # as one string. When empty, falls back to the legacy default-arch loader.
    p.add_argument("--runner_args", default="",
                   help="Training-style arch flags as one string (e.g. "
                        "'--use_mossm_encoder --use_cada --cada_lr ...'). "
                        "Required for MoSSM/CADA-LR checkpoints.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loader (same as infer_pwi.py)
# ---------------------------------------------------------------------------

def _build_model_from_runner_args(runner_args_str: str, device) -> ASLT1Denoiser:
    """Build the model via the training arg-parser + Runner, so the exact
    MoSSM/CADA/CADA-LR/hybrid architecture is reproduced (keys match for a
    strict EMA load). Mirrors scripts/eval_select_ckpt.py:_build_runner."""
    import shlex
    from runners.asl_t1_guided_runner_dmvae_n2n import parse_args as _rparse, Runner
    argv = shlex.split(runner_args_str)
    if "--max_steps" not in argv:
        argv.extend(["--max_steps", "1"])
    orig = sys.argv
    try:
        sys.argv = ["test_mismatched_t1.py"] + argv
        rargs = _rparse()
    finally:
        sys.argv = orig
    runner = Runner(rargs)
    return runner.model.to(device)


def load_model(args, cfg: Config, device: torch.device) -> ASLT1Denoiser:
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = state.get("arch")
    if arch:
        # Rebuild the EXACT training architecture from ckpt metadata so a wrong /
        # missing --runner_args flag can't silently load a shape-matched PARTIAL model
        # (which would give a meaningless mismatched-T1 result).
        model = ASLT1Denoiser(**arch).to(device)
    elif getattr(args, "runner_args", ""):
        model = _build_model_from_runner_args(args.runner_args, device)
    else:
        tp = cfg.asl_denoiser_train_params
        model = ASLT1Denoiser(
            asl_hw=int(tp.asl_hw), t1_hw=int(tp.t1_hw), in_ch=1,
            base_ch=args.base_ch, depth=args.depth,
        ).to(device)
    raw = model.state_dict()
    src = state["ema"] if "ema" in state else state.get("model", state)
    tag = "EMA" if "ema" in state else "model"
    filt = {k: v.to(raw[k].device) for k, v in src.items()
            if k in raw and v.shape == raw[k].shape}
    frac = len(filt) / max(len(raw), 1)
    if frac < 0.98:
        missing = [k for k in raw if k not in filt][:6]
        raise RuntimeError(
            f"Arch mismatch: only {len(filt)}/{len(raw)} {tag} weights matched the "
            f"checkpoint (frac={frac:.2f}). Missing/shape-mismatched e.g.: {missing}. "
            + ("The ckpt has no 'arch' metadata; " if not arch else "")
            + "fix --runner_args so the mismatched-T1 test runs on the TRAINED arch "
              "(a partial load silently invalidates the leakage measurement).")
    model.load_state_dict(filt, strict=False)
    print(f"[INFO] Loaded {len(filt)}/{len(raw)} {tag} weights"
          + ("  (strict from ckpt arch)" if arch else ""))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, ref: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Both inputs are [H, W] float32 in [0, 1]. If `mask` ([H, W]) is given,
    every metric is computed over the masked region ONLY.

    For the mismatched-T1 test this MUST be the ORIGINAL subject's brain mask
    (t1>0.05), never the mismatched-T1's mask (CLAUDE.md §2): otherwise the
    background and wrong-anatomy pixels dominate the leakage L1 and the
    match-vs-mismatch comparison is meaningless."""
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if int(m.sum()) < 1:
            return {"l1": float("nan"), "psnr": float("nan"), "ssim": float("nan")}
        p = pred[m].astype(np.float64); r = ref[m].astype(np.float64)
    else:
        p = pred.astype(np.float64).ravel(); r = ref.astype(np.float64).ravel()
    l1 = float(np.abs(p - r).mean())
    mse = float(((p - r) ** 2).mean())
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    # Simple global-SSIM approximation over the (masked) region.
    mu_p, mu_r = p.mean(), r.mean()
    sig_p = p.std(); sig_r = r.std()
    sig_pr = ((p - mu_p) * (r - mu_r)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2*mu_p*mu_r + c1)*(2*sig_pr + c2)) / \
           ((mu_p**2 + mu_r**2 + c1)*(sig_p**2 + sig_r**2 + c2) + 1e-10)
    return {"l1": l1, "psnr": float(psnr), "ssim": float(ssim)}


# ---------------------------------------------------------------------------
# Slice-level inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_volume(
    model: ASLT1Denoiser,
    asl_vol: torch.Tensor,   # [T, H, W, Z]
    t1_vol: torch.Tensor,    # [1, H, W, Z]
    n_frames: int,
    device: torch.device,
    seed: int,
):                           # -> (pwi [H,W,Z], lo, scale, gate [H,W,Z] or None); gate = Bayesian γ map
    T, H, W, Z = asl_vol.shape
    rng = np.random.default_rng(seed)
    n = min(n_frames, T)
    idx = rng.choice(T, size=n, replace=False).tolist()
    frames = asl_vol[idx]    # [n, H, W, Z]  (RAW — pre-transform no longer normalises ASL)

    # Normalise from the SELECTED input frames only (matches training's set-A affine,
    # data_classes._affine_from_frames). Return (lo, scale) so the caller can map the
    # all-frames reference into the SAME output space for a meaningful L1.
    lo, scale = _affine_from_frames(frames)
    frames = (frames - lo) / scale

    ctx = int(getattr(model, "slice_context", 0))   # 2.5D: read from the model
    pwi = np.zeros((H, W, Z), dtype=np.float32)
    gate = np.full((H, W, Z), np.nan, dtype=np.float32)   # γ = ec_gamma (Bayesian gate); stays NaN if the model has no gate
    has_gate = False
    lengths = torch.tensor([n], dtype=torch.long, device=device)
    for z in range(Z):
        if ctx > 0:                                  # 2.5D: K=2*ctx+1 edge-clamped z-window as channels
            zc = [min(max(z + dz, 0), Z - 1) for dz in range(-ctx, ctx + 1)]
            sl = frames[:, :, :, zc].permute(0, 3, 1, 2).unsqueeze(0).to(device, dtype=torch.float32)  # [1,n,K,H,W]
        else:                                        # 2D: single center slice (K=1)
            sl = frames[:, :, :, z].unsqueeze(1).unsqueeze(0).to(device, dtype=torch.float32)          # [1,n,1,H,W]
        t1 = t1_vol[:, :, :, z].unsqueeze(0).to(device, dtype=torch.float32)                           # T1 stays 2D (center)
        out = model.infer_from_subset(sl, t1, lengths=lengths)
        pwi[:, :, z] = out["asl_recon"][0, 0].cpu().numpy()
        g = out.get("ec_gamma")                            # [1,1,h,w] work-res Bayesian gate γ, or None (ungated model)
        if g is None:
            g = out.get("ec_c_sem")
        if g is not None:
            has_gate = True
            g = torch.nn.functional.interpolate(g.float(), size=(H, W), mode="bilinear", align_corners=False)
            gate[:, :, z] = g[0, 0].cpu().numpy()
    return pwi, float(lo), float(scale), (gate if has_gate else None)


def direct_mean_volume(asl_vol: torch.Tensor) -> np.ndarray:
    """Simple unweighted mean across all T frames → [H, W, Z]."""
    return asl_vol.float().mean(dim=0).numpy()  # [H, W, Z]


# ---------------------------------------------------------------------------
# Per-subject comparison plot
# ---------------------------------------------------------------------------

def save_comparison_png(
    sid: str,
    matched_pwi: np.ndarray,      # [H, W, Z]
    mismatched_pwi: np.ndarray,
    direct_mean: np.ndarray,
    t1_matched: np.ndarray,        # [H, W, Z]  (1-channel squeezed)
    t1_mismatched: np.ndarray,
    out_path: str,
    asl_valid: Optional[np.ndarray] = None,   # [H, W, Z] bool: where ASL has signal
    gate_matched: Optional[np.ndarray] = None,     # [H, W, Z] Bayesian γ map (matched T1), or None
    gate_mismatched: Optional[np.ndarray] = None,  # [H, W, Z] Bayesian γ map (mismatched T1), or None
):
    Z = matched_pwi.shape[2]
    z_mid = Z // 2

    # Mask the recon panels to the ORIGINAL subject's brain (t1>0.05) INTERSECTED with ASL
    # validity, so the preprocessing-zeroed rim/void (no-signal voxels the t1>0.05 mask still
    # covers) recedes into the background instead of rendering as a black hole inside the shown
    # brain. DISPLAY-ONLY — the leakage metrics keep their own t1>0.05 mask (compute_metrics).
    _bm = (t1_matched[:, :, z_mid] > 0.05)
    if asl_valid is not None:
        _bm = _bm & (asl_valid[:, :, z_mid] > 0)
    try:
        from scipy.ndimage import binary_fill_holes
        _bm = binary_fill_holes(_bm)
    except Exception:
        pass
    brain = _bm.astype(matched_pwi.dtype)
    # (img, title, cmap, vlim) — vlim=None => matplotlib autoscale.
    rows = [
        (t1_matched[:, :, z_mid],    "T1 (matched)",          "gray", None),
        (t1_mismatched[:, :, z_mid], "T1 (mismatched)",       "gray", None),
        (direct_mean[:, :, z_mid],   "Direct mean (ref PWI)", "gray", None),
        (matched_pwi[:, :, z_mid] * brain,    "Recon (matched T1)",    "gray", None),
        (mismatched_pwi[:, :, z_mid] * brain, "Recon (mismatched T1)", "gray", None),
        (np.abs(matched_pwi[:, :, z_mid] - mismatched_pwi[:, :, z_mid]) * brain,
         "|matched − mismatched|", "gray", None),
    ]
    # Bayesian gate γ∈(0,1] (lower = guidance attenuated). Δγ shows WHERE the gate closes under
    # the wrong anatomy — the diagnostic for whether the gate (not just V=ASL) rejects mismatch.
    if gate_matched is not None:
        bm = brain > 0
        gm = np.where(bm, gate_matched[:, :, z_mid], np.nan)
        rows.append((gm, "γ (matched T1)", "viridis", (0.0, 1.0)))
        if gate_mismatched is not None:
            gx = np.where(bm, gate_mismatched[:, :, z_mid], np.nan)
            dg = np.where(bm, gate_matched[:, :, z_mid] - gate_mismatched[:, :, z_mid], np.nan)
            amax = float(np.nanmax(np.abs(dg))) if np.isfinite(dg).any() else 1e-6
            amax = amax if amax > 1e-6 else 1e-6
            rows.append((gx, "γ (mismatched T1)", "viridis", (0.0, 1.0)))
            rows.append((dg, "Δγ = matched − mismatched", "RdBu_r", (-amax, amax)))
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4))
    for ax, (img, title, cmap, vlim) in zip(axes, rows):
        if vlim is None:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Subject: {sid}", fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {device}  |  Output: {out_dir}")

    cfg = Config(args.config)
    model = load_model(args, cfg, device)

    # --- Load preprocessed subjects (same split as training) ---
    tp = cfg.asl_denoiser_train_params
    pre_tf = get_pre_asl_transform(
        asl_hw=tp.asl_hw, asl_z=tp.asl_z,
        t1_hw=tp.t1_hw,   t1_z=tp.t1_z,
    )
    subjects = DatasetGenerator(cfg).generate_dataset()
    cached = build_cached_subjects(subjects, pre_tf, cache_rate=float(args.cache_rate), cache_workers=4)

    split_ratios = [cfg.data_loading.train_ratio,
                    cfg.data_loading.val_ratio,
                    cfg.data_loading.test_ratio]
    # MUST match dataio/dataloaders.py's partition_dataset call exactly. It passes
    # config.data_loading.shuffle (True); this used to hardcode shuffle=False and
    # call it "deterministic", which was a misreading: partition_dataset seeds its
    # own RNG, so shuffle=True is ALREADY reproducible. shuffle=False did not buy
    # determinism -- it silently produced a DIFFERENT split, so the mismatched-T1
    # safety gate was scored on subjects the model had trained on.
    splits = partition_dataset(data=cached, ratios=split_ratios,
                               shuffle=cfg.data_loading.shuffle)
    split_map = {"train": splits[0], "val": splits[1], "test": splits[2]}
    subset = split_map[args.split]

    n_subjects = len(subset)
    if args.max_subjects:
        n_subjects = min(n_subjects, args.max_subjects)
    print(f"[INFO] Evaluating {n_subjects} subjects from '{args.split}' split")

    # Build shuffled T1 list (shift by 1 → guaranteed mismatch)
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(n_subjects)
    # Ensure no subject maps to itself
    for i in range(n_subjects):
        if perm[i] == i:
            swap = (i + 1) % n_subjects
            perm[i], perm[swap] = perm[swap], perm[i]

    csv_rows: List[Dict] = []

    for si in range(n_subjects):
        item = subset[si]
        item_mismatched_t1 = subset[perm[si]]

        asl_vol = item["asl_diff"].cpu()         # [T, H, W, Z]
        asl_valid = (asl_vol.abs().mean(dim=0) > 0).numpy()   # [H,W,Z] ASL-signal mask (drops prep-zeroed rim/void)
        t1_matched = item["t1"].cpu()             # [1, H, W, Z]
        t1_mismatched = item_mismatched_t1["t1"].cpu()

        sid = item.get("subject_id", f"sub_{si:03d}")
        mis_sid = item_mismatched_t1.get("subject_id", f"sub_{perm[si]:03d}")
        print(f"[{si+1}/{n_subjects}] {sid} (T1 mismatch: {mis_sid})")

        # SAME frame-selection seed for both so the ONLY difference between the two
        # runs is the T1 input — otherwise matched vs mismatched also differ in which
        # ASL frames were drawn, and the "leakage" L1 conflates T1-sensitivity with
        # frame-sampling variance. (Per-subject variation still comes from asl_vol.)
        seed_si = args.seed + si
        pwi_matched, lo, scale, gate_matched  = infer_volume(model, asl_vol, t1_matched,    args.n_frames, device, seed_si)
        pwi_mismatched, _, _, gate_mismatched = infer_volume(model, asl_vol, t1_mismatched, args.n_frames, device, seed_si)
        # Reference into the SAME (input-frame) normalised space as the outputs, so
        # matched/mismatched-vs-ref L1 is on one scale. matched & mismatched share the
        # seed → share (lo, scale).
        ref_pwi = (direct_mean_volume(asl_vol) - lo) / scale    # [H, W, Z]

        # Aggregate metrics over all z-slices, masked to the ORIGINAL subject's
        # brain (t1_matched > 0.05) — NOT the mismatched T1's mask (CLAUDE.md §2).
        Z = ref_pwi.shape[2]
        brain = (t1_matched[0].numpy() > 0.05)   # [H, W, Z] original-subject mask
        m_vs_mm = []   # matched vs mismatched  (the content-guard leakage metric)
        m_vs_ref = []  # matched vs direct mean
        mm_vs_ref = [] # mismatched vs direct mean

        for z in range(Z):
            bz = brain[:, :, z]
            m_vs_mm.append(compute_metrics(pwi_matched[:,:,z],    pwi_mismatched[:,:,z], bz))
            m_vs_ref.append(compute_metrics(pwi_matched[:,:,z],   ref_pwi[:,:,z],        bz))
            mm_vs_ref.append(compute_metrics(pwi_mismatched[:,:,z], ref_pwi[:,:,z],      bz))

        def mean_metrics(lst):
            keys = lst[0].keys()
            return {k: float(np.nanmean([d[k] for d in lst])) for k in keys}

        row = {"subject": sid, "mismatch_t1": mis_sid}
        for k, v in mean_metrics(m_vs_mm).items():
            row[f"match_vs_mismatch_{k}"] = v
        for k, v in mean_metrics(m_vs_ref).items():
            row[f"match_vs_ref_{k}"] = v
        for k, v in mean_metrics(mm_vs_ref).items():
            row[f"mismatch_vs_ref_{k}"] = v
        # Bayesian-gate response to wrong anatomy: mean γ in brain, matched vs mismatched T1.
        # If the gate actively rejects mismatched anatomy it CLOSES ⇒ gate_drop_mismatch > 0;
        # if γ barely moves, the low leakage is the V=ASL contract's doing, not the gate.
        if gate_matched is not None:
            gmn = float(np.nanmean(gate_matched[brain]))
            gxn = float(np.nanmean(gate_mismatched[brain])) if gate_mismatched is not None else float("nan")
            row["gate_mean_matched"] = gmn
            row["gate_mean_mismatched"] = gxn
            row["gate_drop_mismatch"] = gmn - gxn
        csv_rows.append(row)

        # Save comparison PNG
        save_comparison_png(
            sid=sid,
            matched_pwi=pwi_matched,
            mismatched_pwi=pwi_mismatched,
            direct_mean=ref_pwi,
            t1_matched=t1_matched[0].numpy(),
            t1_mismatched=t1_mismatched[0].numpy(),
            out_path=str(out_dir / f"{sid}_comparison.png"),
            asl_valid=asl_valid,
            gate_matched=gate_matched,
            gate_mismatched=gate_mismatched,
        )

    # --- Write CSV summary ---
    csv_path = out_dir / "results.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        # Print aggregate summary
        print("\n" + "="*60)
        print("SUMMARY (mean over all subjects)")
        print("="*60)
        keys = [k for k in csv_rows[0] if k not in ("subject", "mismatch_t1")]
        for k in keys:
            vals = [r[k] for r in csv_rows]
            print(f"  {k:<40s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print(f"\nFull results → {csv_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
