#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-subject paired differences between one method and each comparator.

Absolute uMSE varies enormously between subjects -- far more than the methods differ from one
another -- so boxes of absolute values overlap even where every subject moves the same way.
The comparison is paired by construction (every method reconstructs the same frames of the
same subject), and this plots that pairing directly: one box per comparator holding the 32
per-subject differences, with the zero line and the fraction of subjects improved.

uMSE is kept on its linear scale and subjects with a negative estimate are kept. uMSE is an
unbiased estimator, so it scatters below zero wherever the true error is small; dropping those
subjects, or taking a logarithm that forces it, biases the comparison towards whichever method
is worse.

Usage:
  python scripts/plot_paired_diff.py --long_csv <dir>/comparison_long_merged.csv \
      --out_dir <dir>/figures --ref proposed --max_k 5
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LOWER_BETTER = {"umse", "scov_gm", "scov_wm", "efc", "hfen", "gmsd", "lapvar", "lapvar_ratio",
                 "l1_ref", "gmwm_contrast_err"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_args():
    p = argparse.ArgumentParser("per-subject paired differences vs each comparator")
    p.add_argument("--long_csv", required=True, help="comparison_long_merged.csv (one row per subject)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ref", default="proposed", help="method whose advantage is plotted")
    p.add_argument("--metrics", nargs="*", default=["umse", "cnr", "snr_gm", "lapvar"])
    p.add_argument("--exclude", nargs="*", default=[],
                   help="comparators to leave out. Repetition averaging differs from every network "
                        "by an order of magnitude, so including it flattens the comparison that "
                        "is actually in question into an unreadable sliver.")
    p.add_argument("--max_k", type=int, default=0,
                   help="drop repetition counts above this. uMSE subtracts two nearly equal "
                        "noise terms and stops being informative once the held-out groups thin "
                        "out; 0 keeps every k.")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.long_csv, encoding="utf-8")))
    if not rows:
        raise SystemExit("empty CSV: %s" % a.long_csv)

    methods = sorted({r["method"] for r in rows})
    if a.ref not in methods:
        raise SystemExit("--ref %r not in %s" % (a.ref, methods))
    others = [m for m in methods if m != a.ref and m not in set(a.exclude)]
    if not others:
        raise SystemExit("no comparators left after --exclude")
    ks = sorted({int(float(r["n_frames"])) for r in rows})
    if a.max_k:
        ks = [k for k in ks if k <= a.max_k]

    cmap = plt.get_cmap("tab10")
    colors = {m: cmap((i + 1) % 10) for i, m in enumerate(others)}
    width = 0.8 / max(len(others), 1)

    for metric in a.metrics:
        if metric not in rows[0]:
            print("  skip %s (not a column)" % metric)
            continue
        lower_better = metric.lower() in _LOWER_BETTER
        vals = {(r["method"], int(float(r["n_frames"])), r["subject_id"]): fnum(r[metric])
                for r in rows}
        subs = sorted({r["subject_id"] for r in rows})

        fig, ax = plt.subplots(figsize=(1.15 * len(ks) + 4.0, 4.4))
        for mi, m in enumerate(others):
            boxes, poss, frac = [], [], []
            for i, k in enumerate(ks):
                d = [vals.get((a.ref, k, s), np.nan) - vals.get((m, k, s), np.nan) for s in subs]
                d = [x for x in d if np.isfinite(x)]
                if not d:
                    continue
                boxes.append(d)
                poss.append(i + (mi - (len(others) - 1) / 2.0) * width)
                better = sum(1 for x in d if (x < 0) == lower_better)
                frac.append((poss[-1], better / len(d), len(d)))
            if not boxes:
                continue
            bp = ax.boxplot(boxes, positions=poss, widths=width * 0.85, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", linewidth=1.2))
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[m]); patch.set_alpha(0.65)
            bp["boxes"][0].set_label("vs %s" % m)
            trans = ax.get_xaxis_transform()
            for x, f, n in frac:
                ax.annotate("%.0f%%" % (100 * f), (x, 0.985), xycoords=trans, ha="center",
                            va="top", fontsize=7, color=colors[m])

        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("repetitions used for the reconstruction")
        gain = "negative = %s better" % a.ref if lower_better else "positive = %s better" % a.ref
        ax.set_ylabel("%s:  %s - comparator" % (metric, a.ref))
        ax.set_title("paired per-subject difference in %s   (%s; %% = subjects improved)"
                     % (metric, gain), fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = out / ("paired_diff_%s.png" % metric)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  wrote %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
