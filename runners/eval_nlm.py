# -*- coding: utf-8 -*-
"""Standalone NLM baseline (no training).

Per validation slice:
  input  = mean(setA[:n_frames])
  output = NLM(input)  with sigma estimated from input itself
  metrics computed vs reference = mean(setA U setB)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm

from config.conf_data import Config
from dataio.dataloaders import get_asl_2d_loaders
from runners.asl_t1_guided_runner_dmvae_n2n import (
    _compute_psnr_ssim,
    direct_mean_from_frames,
    prepare_asl_pair_batch,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser("NLM baseline evaluation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_frames", type=int, default=0,
                   help="If >0, take first N frames of setA; else use full setA.")
    p.add_argument("--patch_size", type=int, default=5)
    p.add_argument("--patch_distance", type=int, default=6)
    p.add_argument("--h_factor", type=float, default=0.6)
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--max_subjects", type=int, default=0,
                   help="If >0, evaluate only first N samples (for smoke test).")
    return p.parse_args()


def nlm_denoise(img: np.ndarray, patch_size: int, patch_distance: int, h_factor: float) -> np.ndarray:
    sigma = float(estimate_sigma(img))
    if sigma <= 0:
        sigma = 1e-3
    return denoise_nl_means(
        img, h=h_factor * sigma, sigma=sigma,
        patch_size=patch_size, patch_distance=patch_distance,
        fast_mode=True, channel_axis=None,
    ).astype(np.float32)


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = Config(args.config)
    tp = cfg.asl_denoiser_train_params

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "figures")
    if args.save_images:
        os.makedirs(img_dir, exist_ok=True)

    loaders = get_asl_2d_loaders(
        cfg, modes=["val"],
        asl_hw=tp.asl_hw, asl_z=tp.asl_z, t1_hw=tp.t1_hw, t1_z=tp.t1_z,
    )
    val_loader = loaders["val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    saved = 0
    with torch.no_grad():
        for k, vb in enumerate(tqdm(val_loader, desc="NLM eval")):
            if args.max_subjects and k >= args.max_subjects:
                break
            pack = prepare_asl_pair_batch(vb, device)
            setA = pack["setA"]
            lenA = pack.get("lenA")
            if args.n_frames > 0:
                setA = setA[:, :args.n_frames]
                lenA = torch.full((setA.size(0),), args.n_frames, device=device, dtype=torch.long)
            meanA = direct_mean_from_frames(setA, lenA)
            union = direct_mean_from_frames(
                torch.cat([pack["setA"], pack["setB"]], dim=1),
                pack["lenA"] + pack["lenB"],
            )
            B = meanA.size(0)
            out = torch.zeros_like(meanA)
            for b in range(B):
                arr = meanA[b, 0].cpu().numpy()
                out[b, 0] = torch.from_numpy(
                    nlm_denoise(arr, args.patch_size, args.patch_distance, args.h_factor)
                ).to(device)

            l1_ref = F.l1_loss(out, union).item()
            psnr_ref, ssim_ref = _compute_psnr_ssim(out, union)
            naive_l1 = F.l1_loss(meanA, union).item()
            naive_psnr, naive_ssim = _compute_psnr_ssim(meanA, union)

            for b in range(B):
                rows.append({
                    "sample": f"{k}_{b}",
                    "n_frames": args.n_frames if args.n_frames > 0 else int(pack["lenA"][b]),
                    "method": "NLM",
                    "l1_ref": l1_ref, "psnr_ref": psnr_ref, "ssim_ref": ssim_ref,
                    "naive_l1": naive_l1, "naive_psnr": naive_psnr, "naive_ssim": naive_ssim,
                })

            if args.save_images and saved < 8:
                try:
                    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                    for ax, im, t in zip(
                        axes,
                        [pack["t1"][0, 0].cpu().numpy(), union[0, 0].cpu().numpy(),
                         meanA[0, 0].cpu().numpy(), out[0, 0].cpu().numpy()],
                        ["T1w", "12-NEX ref", "mean(A) input", "NLM output"],
                    ):
                        ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=8); ax.axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(img_dir, f"sample{saved:03d}.png"),
                                dpi=80, bbox_inches="tight")
                    plt.close(fig)
                    saved += 1
                except Exception:
                    plt.close("all")

    csv_path = os.path.join(args.out_dir, "nlm_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    arr = np.array([(r["l1_ref"], r["psnr_ref"], r["ssim_ref"]) for r in rows])
    print(f"NLM   : L1={arr[:,0].mean():.4f}±{arr[:,0].std():.4f}  "
          f"PSNR={arr[:,1].mean():.2f}  SSIM={arr[:,2].mean():.4f}")
    arr_n = np.array([(r["naive_l1"], r["naive_psnr"], r["naive_ssim"]) for r in rows])
    print(f"naive : L1={arr_n[:,0].mean():.4f}±{arr_n[:,0].std():.4f}  "
          f"PSNR={arr_n[:,1].mean():.2f}  SSIM={arr_n[:,2].mean():.4f}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
