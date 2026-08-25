# -*- coding: utf-8 -*-
"""Aggregate the mismatched-T1 safety test across model variants into a paper table + figure.

The mismatched-T1 test feeds each subject's ASL frames with ANOTHER subject's T1, and
measures how much the recon changes (`match_vs_mismatch_l1`). A content-isolated model
(V=ASL invariant) should change very little: swapping the T1 only re-routes attention,
it cannot inject T1 pixel content. The honest yardstick is the ratio to the recon's own
magnitude vs the 12-NEX reference (`match_vs_ref_l1`): a SMALL ratio = the wrong anatomy
barely perturbed the output.

Reads each run's logs/<run>/mismatched_t1/results.csv (produced by test_mismatched_t1.py),
writes:
  <out>/mismatched_t1_safety.md     — mean±std table across the variants
  <out>/mismatched_t1_safety.png    — grouped bar chart (mismatch-L1 vs ref-L1, + ratio)

Usage:
  python scripts/build_mismatched_t1_figure.py --out_dir /mnt/d/tmp/asl_exp/paper_figs
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS_DEFAULT = "/mnt/d/tmp/asl_exp/logs"

# (display label, run dir). Order = method evolution; ours last.
VARIANTS = [
    ("gated-CADA (v1)",      "run_grad06_keeper_500"),
    ("CADA-LR",              "run_cadalr_v2_500"),
    ("CADA-LR+Hybrid (nvOFF)", "run_cadalr_hybrid_nonv_500"),
    ("CIG-Net v2 (final)",   "run_wd1e4_probe"),
]


def _read(run_dir):
    path = os.path.join(run_dir, "mismatched_t1", "results.csv")
    if not os.path.isfile(path):
        return None
    mm, ref = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                mm.append(float(r["match_vs_mismatch_l1"]))
                ref.append(float(r["match_vs_ref_l1"]))
            except (KeyError, ValueError):
                continue
    if not mm:
        return None
    return np.array(mm), np.array(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=LOGS_DEFAULT)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    labels, mm_mean, mm_std, ref_mean, ratio_mean, ratio_std, npb = [], [], [], [], [], [], []
    for label, run in VARIANTS:
        got = _read(os.path.join(args.logs, run))
        if got is None:
            print(f"[skip] {label}: no results.csv under {run}/mismatched_t1/")
            continue
        mm, ref = got
        ratio = mm / np.clip(ref, 1e-8, None)
        labels.append(label)
        mm_mean.append(mm.mean()); mm_std.append(mm.std())
        ref_mean.append(ref.mean())
        ratio_mean.append(ratio.mean()); ratio_std.append(ratio.std())
        npb.append(len(mm))

    if not labels:
        raise SystemExit("No results.csv found for any variant.")

    # ---- markdown table ----
    md = os.path.join(args.out_dir, "mismatched_t1_safety.md")
    with open(md, "w") as f:
        f.write("# Mismatched-T1 safety (content-isolation / anti-hallucination)\n\n")
        f.write("`match_vs_mismatch_l1` = how much the recon changes when the subject's T1 is "
                "replaced by another subject's T1. `match_vs_ref_l1` = recon distance to the "
                "12-NEX reference (output magnitude yardstick). **ratio ≪ 1 = safe** (wrong "
                "anatomy barely perturbs the output).\n\n")
        f.write("| Variant | mismatch-L1 (mean±std) ↓ | ref-L1 | ratio ↓ | N |\n")
        f.write("|---|---|---|---|---|\n")
        for i, lab in enumerate(labels):
            f.write(f"| {lab} | {mm_mean[i]:.4f} ± {mm_std[i]:.4f} | {ref_mean[i]:.4f} | "
                    f"{ratio_mean[i]:.3f} ± {ratio_std[i]:.3f} | {npb[i]} |\n")
    print(f"[ok] wrote {md}")

    # ---- grouped bar figure ----
    x = np.arange(len(labels)); w = 0.38
    fig, ax1 = plt.subplots(figsize=(1.9 * len(labels) + 2, 4.2))
    b1 = ax1.bar(x - w / 2, mm_mean, w, yerr=mm_std, capsize=3, label="mismatch-L1 (Δ from wrong T1)", color="#c0392b")
    b2 = ax1.bar(x + w / 2, ref_mean, w, label="ref-L1 (output magnitude)", color="#bdc3c7")
    ax1.set_ylabel("masked L1")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax1.legend(loc="upper left", fontsize=8)
    # ratio annotations
    for i in range(len(labels)):
        ax1.text(x[i], max(mm_mean[i] + mm_std[i], ref_mean[i]) + 0.001,
                 f"r={ratio_mean[i]:.2f}", ha="center", fontsize=8, color="#2c3e50")
    ax1.set_title("Mismatched-T1 safety: a content-isolated model barely moves\n"
                  "when fed the wrong anatomy (lower red = safer)", fontsize=9)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "mismatched_t1_safety.png")
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] wrote {png}")

    # console summary
    print("\nVariant                       mismatch-L1     ref-L1    ratio   N")
    for i, lab in enumerate(labels):
        print(f"  {lab:28s} {mm_mean[i]:.4f}±{mm_std[i]:.4f}  {ref_mean[i]:.4f}  {ratio_mean[i]:.3f}  {npb[i]}")


if __name__ == "__main__":
    main()
