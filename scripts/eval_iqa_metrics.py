# -*- coding: utf-8 -*-
"""Quick val-set evaluation of EFC / sCoV-GM (and other no-ref IQA metrics)
across a list of checkpoints — to calibrate criterion targets.

Usage (after the runner's CLI):
    python scripts/eval_iqa_metrics.py \
        --config env/local/configs/win_asl_2d_home_v37.yml \
        --exp C:/tmp/asl_exp \
        --name run_full_v37 \
        --base_ch 32 --depth 4 --seed 42 \
        --use_t1_cross_fusion --t1_attn_max_tokens 1024 --use_svfw \
        --init_t1_from C:/tmp/asl_exp/logs/stage1_t1_300step/checkpoints/best.pth \
        --freeze_t1 \
        --eval_ckpts step000050.pth step000150.pth ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from runners.asl_t1_guided_runner_dmvae_n2n import Runner, parse_args


def main():
    # Reuse the runner's parse_args so all required defaults are set.
    # Add an extra --eval_ckpts arg.
    sys_argv = list(sys.argv)
    eval_ckpts = []
    if "--eval_ckpts" in sys_argv:
        idx = sys_argv.index("--eval_ckpts")
        # Collect all positional values until next flag
        i = idx + 1
        while i < len(sys_argv) and not sys_argv[i].startswith("--"):
            eval_ckpts.append(sys_argv[i])
            i += 1
        # Remove --eval_ckpts and its values from argv before runner parses
        del sys_argv[idx:i]
        sys.argv = sys_argv

    args = parse_args()
    runner = Runner(args)
    runner.model.eval()

    ckpt_dir = Path(args.exp) / "logs" / args.name / "checkpoints"

    print(f"\n{'ckpt':<22s}{'uPSNR':>9s}{'cnr_p':>8s}{'cnr_r':>8s}{'cyc':>8s}"
          f"{'psnr_ref':>10s}{'psnr_b':>9s}{'sGMr':>8s}{'sWMr':>8s}{'lapvar':>8s}")
    print("-" * 100)

    for ck in eval_ckpts:
        ck_path = ckpt_dir / ck
        if not ck_path.exists():
            print(f"[SKIP] not found: {ck_path}")
            continue
        state = torch.load(str(ck_path), map_location=runner.device, weights_only=False)
        # Load BOTH model state and EMA state — validate() uses _predict_with_ema
        # which copies self.ema → model at inference, so the EMA dict is what
        # actually drives val predictions.
        if "model" in state:
            runner.model.load_state_dict(state["model"], strict=True)
        if "ema" in state:
            runner.ema.ema_state = {k: v.to(runner.device) for k, v in state["ema"].items()}
            runner.ema.optimization_step = state.get("ema_optimization_step", 0)

        l1_b, psnr_b, ssim_b, l1_ref, psnr_ref, ssim_ref = runner.validate()
        extras = runner._last_val_extras
        upsnr = extras.get("upsnr", float("nan"))
        cnr_p = extras.get("cnr_pred", float("nan"))
        cnr_r = extras.get("cnr_ref", float("nan"))
        cyc   = extras.get("subset_consistency", float("nan"))
        sgm_ratio = extras.get("scov_gm_pred", 0.0) / max(extras.get("scov_gm_ref", 1.0), 1e-8)
        swm_ratio = extras.get("scov_wm_pred", 0.0) / max(extras.get("scov_wm_ref", 1.0), 1e-8)
        lapv = extras.get("lapvar_ratio", 0.0)

        print(f"{ck:<22s}{upsnr:>9.3f}{cnr_p:>8.3f}{cnr_r:>8.3f}{cyc:>8.4f}"
              f"{psnr_ref:>10.3f}{psnr_b:>9.3f}{sgm_ratio:>8.3f}{swm_ratio:>8.3f}{lapv:>8.3f}")


if __name__ == "__main__":
    main()
