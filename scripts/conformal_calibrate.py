# -*- coding: utf-8 -*-
"""Innovation U — Conformal Hallucination Detection (calibration).

Construct a calibration set of per-voxel discrepancies between matched-T1
and mismatched-T1 model outputs, then compute the (1-α) quantile threshold
q_α. Used by infer_pwi.py to flag voxels with epistemic uncertainty above
q_α as 'high hallucination risk'.

Coverage guarantee (Vovk 2005, Angelopoulos 2021): at any new test point
drawn iid from the same distribution, P(disc > q_α) ≤ α (distribution-free,
finite-sample).

Usage (after training, before deployment):
    python scripts/conformal_calibrate.py \
        --config env/local/configs/win_asl_2d_home_v37.yml \
        --exp C:/tmp/asl_exp --name run_full_v39 \
        --base_ch 32 --depth 4 --use_t1_cross_fusion --use_svfw \
        --init_t1_from C:/tmp/asl_exp/logs/stage1_t1_300step/checkpoints/latest.pth \
        --freeze_t1 \
        --calibration_alpha 0.10 \
        --calibration_ckpt swa.pth \
        --calibration_out C:/tmp/asl_exp/logs/run_full_v39/conformal.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from runners.asl_t1_guided_runner_dmvae_n2n import Runner, parse_args, prepare_asl_pair_batch


def _strip_calibration_args(argv):
    """Pull our extra args out before the runner's parser sees argv."""
    out = {"alpha": 0.10, "ckpt": "swa.pth", "out": None}
    keep = []
    i = 0
    argv = list(argv)
    while i < len(argv):
        a = argv[i]
        if a == "--calibration_alpha":
            out["alpha"] = float(argv[i + 1]); i += 2
        elif a == "--calibration_ckpt":
            out["ckpt"] = argv[i + 1]; i += 2
        elif a == "--calibration_out":
            out["out"] = argv[i + 1]; i += 2
        else:
            keep.append(a); i += 1
    return keep, out


def main():
    sys.argv[1:], extras = _strip_calibration_args(sys.argv[1:])
    args = parse_args()
    runner = Runner(args)

    # Load the requested ckpt (typically swa.pth) into model + EMA.
    ckpt_path = Path(args.exp) / "logs" / args.name / "checkpoints" / extras["ckpt"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"calibration ckpt not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=runner.device, weights_only=False)
    if "model" in state:
        runner.model.load_state_dict(state["model"], strict=True)
    if "ema" in state:
        runner.ema.ema_state = {k: v.to(runner.device) for k, v in state["ema"].items()}
        runner.ema.optimization_step = state.get("ema_optimization_step", 0)
    runner.model.eval()

    discrepancies = []  # per-batch flat arrays of |pred_match - pred_mm| within brain
    n_batches = 0
    print(f"[INFO] Building calibration set from {ckpt_path.name} on val loader...")

    with torch.no_grad():
        for vb in runner.val_loader:
            pack = prepare_asl_pair_batch(vb, runner.device)
            pack = runner._mask_asl_inputs(pack)
            B = pack["t1"].shape[0]
            if B < 2:
                continue
            shift = max(1, B // 2)
            t1_match = pack["t1"]
            t1_mm = t1_match.roll(shifts=shift, dims=0)
            len_a = pack.get("lenA")
            mask_a = pack.get("maskA")

            pred_match = runner._predict_with_ema(pack["setA"], t1_match, len_a, mask_a)["asl_recon"]
            pred_mm    = runner._predict_with_ema(pack["setA"], t1_mm,    len_a, mask_a)["asl_recon"]

            brain = (t1_match > runner.criterion.weights.mask_threshold).float()
            disc = (pred_match - pred_mm).abs() * brain
            disc_flat = disc[brain > 0.5].detach().cpu().numpy().ravel()
            if disc_flat.size > 0:
                discrepancies.append(disc_flat)
            n_batches += 1

    if not discrepancies:
        raise RuntimeError("no calibration samples collected (val loader empty or B<2)")

    pooled = np.concatenate(discrepancies, axis=0)
    alpha = float(extras["alpha"])
    q_alpha = float(np.quantile(pooled, 1.0 - alpha))
    print(f"[INFO] N voxels = {pooled.size}, batches = {n_batches}")
    print(f"[INFO] Calibration: alpha={alpha:.3f} → q_alpha={q_alpha:.6f} "
          f"(median disc={float(np.median(pooled)):.6f}, "
          f"max disc={float(pooled.max()):.6f})")

    out_path = extras["out"] or str(ckpt_path.parent.parent / "conformal.json")
    out_obj = {
        "alpha": alpha,
        "q_alpha": q_alpha,
        "n_voxels": int(pooled.size),
        "n_batches": n_batches,
        "ckpt": str(ckpt_path),
        "method": "mismatched_t1_calibration",
        "stats": {
            "median": float(np.median(pooled)),
            "p50":    float(np.quantile(pooled, 0.50)),
            "p90":    float(np.quantile(pooled, 0.90)),
            "p95":    float(np.quantile(pooled, 0.95)),
            "p99":    float(np.quantile(pooled, 0.99)),
            "max":    float(pooled.max()),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"[INFO] Saved → {out_path}")


if __name__ == "__main__":
    main()
