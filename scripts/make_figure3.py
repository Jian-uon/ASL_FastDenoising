#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assemble the manuscript's per-subject distribution figure.

One row of four measures, one box per method, one value per held-out subject. A second row of
paired differences was dropped: the paired test already appears as a p-value under Table 1, and
the boxes of differences said little that the ranking did not.

On the fidelity panel there is a choice with consequences, so both are buildable:

  --fidelity umse   (default) keeps all subjects. uMSE is an unbiased risk estimate, so it
                    scatters below zero wherever the true error is small; those subjects are
                    real measurements, not failures.
  --fidelity upsnr  matches a dB axis but silently drops every subject whose uMSE is <= 0,
                    because the logarithm is undefined there. That loss is not uniform: it
                    removes 25% of subjects for the proposed model and 0% for repetition
                    averaging, i.e. it penalises exactly the methods whose error is smallest.
                    The panel therefore prints the surviving n.

Usage:
  python scripts/make_figure3.py --dir <out>/medphys_eval --k 2 --fidelity umse
"""
from __future__ import annotations

import argparse
import csv
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

PAPER_NAME = {
    "naive_mean": "Repetition\naveraging",
    "vanilla_N2N": "UNet\n-N2N",
    "SwinIR_N2N": "SwinIR\n-N2N",
    "proposed": "Proposed",
}
ORDER = ["naive_mean", "vanilla_N2N", "SwinIR_N2N", "proposed"]
COMPARATORS = ["vanilla_N2N", "SwinIR_N2N"]
COLORS = {"naive_mean": "#b0b0b0", "vanilla_N2N": "#7fb3d5",
          "SwinIR_N2N": "#f0b27a", "proposed": "#7dcea0"}

# (key, axis label, higher-is-better). The fidelity panel's direction follows --fidelity:
# uPSNR is higher-better, uMSE is lower-better.
def panels(fidelity):
    # White matter is reported beside gray: perfusion there is low and the signal sits closest
    # to noise, so its uniformity is the harder test of a reconstruction and the one that
    # exposes smoothing rather than denoising.
    return [
        ("fid",     None,          fidelity == "upsnr"),
        ("cnr",     "CNR",         True),
        ("scov_gm", "sCoV$_{GM}$", False),
        ("scov_wm", "sCoV$_{WM}$", False),
        ("snr_gm",  "SNR$_{GM}$",  True),
    ]


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load(path, ks, fidelity):
    """-> {method: {subject: {metric: value}}}, each measure averaged over `ks`.

    `ks` is the range validation draws set A from, so every panel describes the model at the
    acquisition lengths its checkpoint was selected over. A subject contributes a value only
    if it has one at every length, so no measure is averaged over a different set than
    another. uPSNR is taken from the averaged uMSE, since decibels do not average.
    """
    ks = set(ks)
    cols = (("fid", "umse"), ("cnr", "cnr"), ("scov_gm", "scov_gm"),
            ("scov_wm", "scov_wm"), ("snr_gm", "snr_gm"))
    acc = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if int(float(r["n_frames"])) not in ks:
            continue
        d = acc.setdefault(r["method"], {}).setdefault(r["subject_id"], {})
        for key, col in cols:
            v = fnum(r.get(col, "nan"))
            if v == v:
                d.setdefault(key, []).append(v)

    out = {}
    for m, subs in acc.items():
        for sub, d in subs.items():
            row = {}
            for key, _ in cols:
                vals = d.get(key, [])
                row[key] = sum(vals) / len(vals) if len(vals) == len(ks) else float("nan")
            if fidelity == "upsnr":
                u = row["fid"]
                row["fid"] = -10.0 * math.log10(u) if (u == u and u > 0) else float("nan")
            out.setdefault(m, {})[sub] = row
    return out


def box(ax, series, labels, colors, zero_line=False):
    ok = [[v for v in s if v == v] for s in series]
    bp = ax.boxplot(ok, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#333333")
    for key in ("whiskers", "caps", "medians"):
        for art in bp[key]:
            art.set_color("#333333")
    for i, s in enumerate(ok, start=1):
        ax.plot([i] * len(s), s, ".", ms=3, color="#33333355", zorder=3)
    if zero_line:
        ax.axhline(0.0, color="#c0392b", lw=1.2, ls="--", zorder=1)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return ok


def main() -> int:
    p = argparse.ArgumentParser("assemble the per-subject distribution figure")
    p.add_argument("--dir", required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6],
                   help="acquisition lengths every panel averages over. Default 3 4 5 6 = the "
                        "range validation draws set A from, matching Table 1's uMSE.")
    p.add_argument("--fidelity", choices=["umse", "upsnr"], default="umse")
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()

    src = os.path.join(a.dir, "sweep", "comparison_long_merged.csv")
    if not os.path.isfile(src):
        raise SystemExit("no %s -- run merge_seeds.py first" % src)
    data = load(src, a.ks, a.fidelity)
    methods = [m for m in ORDER if m in data]
    fid_label = "uPSNR (dB)" if a.fidelity == "upsnr" else "uMSE"
    PANELS = panels(a.fidelity)

    subs = sorted(set.intersection(*(set(data[m]) for m in methods)))
    ktxt = ",".join(str(x) for x in sorted(a.ks))
    print("subjects present in every method over k=%s: %d" % (ktxt, len(subs)))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.75 * len(PANELS), 4.1))

    for j, (key, label, _) in enumerate(PANELS):
        ax = axes[j]
        series = [[data[m][s][key] for s in subs] for m in methods]
        ok = box(ax, series, [PAPER_NAME[m] for m in methods],
                 [COLORS[m] for m in methods])
        ax.set_ylabel(label or fid_label)
        n = min(len(s) for s in ok)
        title = (label or fid_label)
        if n < len(subs):
            title += "  (n=%d of %d)" % (n, len(subs))
        ax.set_title(title, fontsize=10)

    lo, hi = min(a.ks), max(a.ks)
    span = ("%d to %d" % (lo, hi)) if sorted(a.ks) == list(range(lo, hi + 1)) else ktxt
    fig.suptitle("Per-subject distributions over %s repetitions (n=%d held-out subjects)"
                 % (span, len(subs)), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = a.out or os.path.join(a.dir, "figures", "Figure3_%s.png" % a.fidelity)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out)

    if a.fidelity == "upsnr":
        for m in methods:
            tot = len(subs)
            ok = sum(1 for s in subs if data[m][s]["fid"] == data[m][s]["fid"])
            print("  %-12s uPSNR defined for %2d/%d subjects (%d dropped)"
                  % (m, ok, tot, tot - ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
