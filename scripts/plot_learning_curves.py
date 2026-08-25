# -*- coding: utf-8 -*-
"""Plot learning curves (val L1 / PSNR / SSIM vs epoch) from a runner stdout.txt log.

Parses lines like:
  [BEST] step=N l1_B=0.0354 | psnr_ref=23.40 ssim_ref=0.8932
  [VAL ] step=N l1_B=0.0376 | psnr_ref=23.33 ssim_ref=0.8928 (best 0.0354, ...)

Usage:
  python scripts/plot_learning_curves.py \
    --log C:/tmp/asl_exp/logs/run_full_v17/stdout.txt \
    --out C:/tmp/asl_exp/figures/learning_curves/run_full_v17.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


TAG_RE = re.compile(r"\[(BEST|VAL )\]\s+step=(\d+)")
L1_RE = re.compile(r"l1_B=([\d.]+)")
PSNR_RE = re.compile(r"psnr_ref=([\d.]+)")
SSIM_RE = re.compile(r"ssim_ref=([\d.]+)")


def parse_log(path: str) -> Tuple[List[int], List[float], List[float], List[float], List[int]]:
    """Return (epochs, l1, psnr, ssim, best_epochs)."""
    epochs: List[int] = []
    l1: List[float] = []
    psnr: List[float] = []
    ssim: List[float] = []
    best_epochs: List[int] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tag = TAG_RE.search(line)
            if not tag:
                continue
            ml1 = L1_RE.search(line); mp = PSNR_RE.search(line); ms = SSIM_RE.search(line)
            if not (ml1 and mp and ms):
                continue
            ep = int(tag.group(2))
            epochs.append(ep)
            l1.append(float(ml1.group(1)))
            psnr.append(float(mp.group(1)))
            ssim.append(float(ms.group(1)))
            if tag.group(1) == "BEST":
                best_epochs.append(ep)
    return epochs, l1, psnr, ssim, best_epochs


def plot(log_path: str, out_path: str, title: str = ""):
    epochs, l1, psnr, ssim, best_epochs = parse_log(log_path)
    if not epochs:
        print(f"WARN: no [VAL ]/[BEST] lines found in {log_path}")
        return

    epochs_a = np.array(epochs); l1_a = np.array(l1)
    psnr_a = np.array(psnr); ssim_a = np.array(ssim)

    # Best by PSNR vs 12-NEX reference (matches updated runner selection criterion).
    best_idx = int(np.argmax(psnr_a))
    best_ep, best_l1, best_psnr, best_ssim = (
        epochs_a[best_idx], l1_a[best_idx], psnr_a[best_idx], ssim_a[best_idx])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs_a, l1_a, marker=".", lw=1, ms=3, color="C0")
    axes[0].axvline(best_ep, color="C3", ls="--", lw=1, alpha=0.7,
                    label=f"best @ epoch {best_ep}")
    axes[0].axhline(best_l1, color="C3", ls=":", lw=0.7, alpha=0.6)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val L1 vs set B")
    axes[0].set_title(f"L1 (best={best_l1:.4f})")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_a, psnr_a, marker=".", lw=1, ms=3, color="C2")
    axes[1].axvline(best_ep, color="C3", ls="--", lw=1, alpha=0.7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("PSNR vs 12-NEX ref (dB)")
    axes[1].set_title(f"PSNR (@best epoch={best_psnr:.2f} dB)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_a, ssim_a, marker=".", lw=1, ms=3, color="C4")
    axes[2].axvline(best_ep, color="C3", ls="--", lw=1, alpha=0.7)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("SSIM vs 12-NEX ref")
    axes[2].set_title(f"SSIM (@best epoch={best_ssim:.4f})")
    axes[2].grid(True, alpha=0.3)

    sup = title or os.path.basename(os.path.dirname(log_path))
    fig.suptitle(f"{sup}  |  N evals={len(epochs)}, range=[{epochs_a.min()}, {epochs_a.max()}]",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"  best epoch = {best_ep}  l1_B = {best_l1:.4f}  "
          f"psnr_ref = {best_psnr:.2f}  ssim_ref = {best_ssim:.4f}")
    print(f"  total evals = {len(epochs)}, last epoch = {epochs_a.max()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=str, required=True, help="Path to stdout.txt")
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    p.add_argument("--title", type=str, default="")
    args = p.parse_args()
    plot(args.log, args.out, args.title)


if __name__ == "__main__":
    sys.exit(main())
