# -*- coding: utf-8 -*-
"""Global best-CNR checkpoint selection for the PlainUNet2D reference baselines.

`eval_select_ckpt.py` rebuilds the MoSSM runner (`runner._predict_with_ema`) and so
can only select the 3 core models. The PlainUNet baselines (n2n / n2self / sup) were
previously left at their psnr_ref-tagged `best.pth`, which made the headline table an
unfair fight: ours/aslonly/naive were global-max-CNR while the baselines were not
(CLAUDE.md §4 selection asymmetry). This applies the SAME rule to the baselines —
evaluate every checkpoint's pooled GM-WM CNR on val and copy the global argmax to
`best_cnr.pth` — reusing eval_baselines' load_unet / run_unet and the project CNR/uMSE
so the numbers are directly comparable to the MoSSM selection.

Usage:
  python scripts/select_cnr_plainunet.py \
    --ckpts $RUN/checkpoints/step*.pth $RUN/checkpoints/best.pth \
    --config env/hpc/configs/server_v37.yml --base_ch 32 --depth 4 \
    --output $RUN/selection_cnr_plainunet.json --save_selected $RUN/best_cnr.pth
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from config.conf_data import Config
from runners.asl_t1_guided_runner_dmvae_n2n import set_seed
from runners.eval_baselines import materialise_split, load_unet, run_unet
from utils.metrics import cnr as _cnr, scov as _scov, upsnr_components as _upsnr


def parse_args():
    p = argparse.ArgumentParser("global best-CNR selection for PlainUNet baselines")
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoints to score (step*.pth best.pth)")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=0, help="0 = all val batches; >0 = smoke subset")
    p.add_argument("--output", required=True, help="per-ckpt metric JSON")
    p.add_argument("--save_selected", default=None, help="copy the selected ckpt verbatim here")
    p.add_argument("--metric", choices=["umse", "cnr"], default="umse",
                   help="selection objective: 'umse' = global argmin pooled uMSE (2026-08-11 "
                        "uMSE-lead default; matches the runner's best_umse.pth operating point); "
                        "'cnr' = legacy global argmax GM-WM CNR.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[sel-unet] materialising '{args.split}' split (seed={args.seed}) ...")
    val = materialise_split(cfg, device, split=args.split, max_samples=args.max_samples)
    if not (len(val) > 0 and "gm" in val[0] and "wm" in val[0]):
        raise SystemExit("[sel-unet] no GM/WM PV maps in the split -> cannot compute CNR.")

    results = []
    for ck in args.ckpts:
        if not os.path.isfile(ck):
            print(f"[sel-unet] skip missing {ck}")
            continue
        m = load_unet(ck, args, device)
        cnr_sum = scov_sum = 0.0
        ssq = svc = npix = 0.0
        n = 0
        for pack in val:
            gm = pack["gm"].to(device)
            wm = pack["wm"].to(device)
            brain = (pack["t1"].to(device) > 0.05).float()
            pred = run_unet(m, pack, 0, device)            # n_frames=0 -> full setA mean (operating point)
            cnr_sum += float(_cnr(pred, gm, wm, threshold=0.5))
            scov_sum += float(_scov(pred, gm, threshold=0.5))
            a, b, c, _ = _upsnr(pred, pack["setB"].to(device), pack["lenB"].to(device), mask=brain)
            ssq += a; svc += b; npix += c
            n += 1
        pooled_cnr = cnr_sum / max(n, 1)
        pooled_scov_gm = scov_sum / max(n, 1)
        # Not clamped: an unbiased risk may be negative, and clamping collapses every such
        # checkpoint onto one value for the argmin below to choose between arbitrarily.
        pooled_umse = ((ssq - svc) / npix if npix > 0 else float("nan"))
        results.append(dict(ckpt=os.path.basename(ck), ckpt_path=ck,
                            pooled_cnr=pooled_cnr, pooled_scov_gm=pooled_scov_gm, pooled_umse=pooled_umse))
        print(f"[sel-unet] {os.path.basename(ck):16s} CNR={pooled_cnr:.4f} uMSE={pooled_umse:.5f} sCoVgm={pooled_scov_gm:.4f}")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _key = "pooled_umse" if args.metric == "umse" else "pooled_cnr"
    finite = [r for r in results if r[_key] == r[_key]]
    if not finite:
        raise SystemExit(f"[sel-unet] no ckpt has a finite {args.metric}.")
    # uMSE-lead default: global argmin pooled uMSE (= argmax uPSNR); cnr = legacy argmax.
    sel = (min(finite, key=lambda r: r["pooled_umse"]) if args.metric == "umse"
           else max(finite, key=lambda r: r["pooled_cnr"]))

    out = dict(metric=args.metric,
               objective=f"{args.metric}_global_{'min' if args.metric == 'umse' else 'max'}",
               selected=sel["ckpt"], selected_cnr=sel["pooled_cnr"], selected_umse=sel["pooled_umse"],
               selected_scov_gm=sel["pooled_scov_gm"], results=results)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sel-unet] selected {sel['ckpt']} (metric={args.metric}: uMSE={sel['pooled_umse']:.5f} "
          f"CNR={sel['pooled_cnr']:.4f}) -> {args.output}")
    if args.save_selected:
        shutil.copyfile(sel["ckpt_path"], args.save_selected)
        print(f"[sel-unet] copied -> {args.save_selected}")


if __name__ == "__main__":
    main()
