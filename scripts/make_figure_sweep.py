#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance against acquisition length, which Table 1 no longer carries.

Table 1 compares the methods under one condition. The paper's other claim is about the
acquisition axis -- that reconstruction quality barely depends on how many repetitions enter
it, so a short scan is enough -- and that claim needs the axis itself.

The horizontal line is what makes the figure worth drawing: it marks repetition averaging at
the full twelve repetitions, the image today's protocol delivers. Where a reconstruction curve
sits above that line, a shorter scan has bought more contrast than the complete one.

uMSE is not plotted. It is estimated from the repetitions left out of the reconstruction, so
it is undefined past nine and noise-dominated well before, which would put a broken curve next
to three intact ones; Table 1 reports it once instead.

Usage:
  python scripts/make_figure_sweep.py --dir <out>/medphys_eval
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

PAPER_NAME = {
    "naive_mean":  "Repetition averaging",
    "vanilla_N2N": "UNet-N2N",
    "SwinIR_N2N":  "SwinIR-N2N",
    "proposed":    "Proposed",
}
ORDER = ["naive_mean", "vanilla_N2N", "SwinIR_N2N", "proposed"]
COLORS = {"naive_mean": "#b0b0b0", "vanilla_N2N": "#7fb3d5",
          "SwinIR_N2N": "#f0b27a", "proposed": "#c0392b"}
MARKERS = {"naive_mean": "s", "vanilla_N2N": "^", "SwinIR_N2N": "D", "proposed": "o"}

# (column, axis label, better direction)
PANELS = [("cnr", "GM-WM CNR", "higher"),
          ("scov_gm", "sCoV$_{GM}$", "lower"),
          ("snr_gm", "SNR$_{GM}$", "higher")]


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    """-> {method: {k: {metric: (mean over subjects, n)}}}, subject-level."""
    acc = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        k = int(float(r["n_frames"]))
        d = acc.setdefault(r["method"], {}).setdefault(k, {})
        for col, _, _ in PANELS:
            v = fnum(r.get(col, "nan"))
            if v == v:
                d.setdefault(col, []).append(v)
    out = {}
    for m, ks in acc.items():
        for k, d in ks.items():
            out.setdefault(m, {})[k] = {c: (sum(d[c]) / len(d[c]), len(d[c]))
                                        for c in d if d[c]}
    return out


def main() -> int:
    p = argparse.ArgumentParser("performance against acquisition length")
    p.add_argument("--dir", required=True)
    p.add_argument("--full_k", type=int, default=12,
                   help="acquisition length of the reference protocol")
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()

    src = os.path.join(a.dir, "sweep", "comparison_long_merged.csv")
    if not os.path.isfile(src):
        raise SystemExit("no %s -- run merge_seeds.py first" % src)
    data = load(src)
    methods = [m for m in ORDER if m in data]
    ks = sorted(set().union(*(set(data[m]) for m in methods)))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(12.6, 3.9))
    for ax, (col, label, better) in zip(axes, PANELS):
        # the image today's protocol delivers, for the "is a short scan enough" reading
        ref = data.get("naive_mean", {}).get(a.full_k, {}).get(col, (float("nan"), 0))[0]
        if ref == ref:
            ax.axhline(ref, color="#444444", lw=1.0, ls="--", zorder=1)
            ax.annotate("full acquisition\n(%d rep. averaged)" % a.full_k,
                        xy=(ks[-1], ref), xytext=(-4, 4), textcoords="offset points",
                        ha="right", va="bottom", fontsize=7, color="#444444")
        for m in methods:
            xs = [k for k in ks if col in data[m].get(k, {})]
            ys = [data[m][k][col][0] for k in xs]
            ax.plot(xs, ys, marker=MARKERS[m], ms=4.5, lw=1.6,
                    color=COLORS[m], label=PAPER_NAME.get(m, m), zorder=3)
        ax.set_xlabel("Repetitions entering the reconstruction", fontsize=9)
        ax.set_title("%s (%s is better)" % (label, better), fontsize=10)
        ax.set_xticks(ks)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    n_sub = max(data[m][ks[0]][PANELS[0][0]][1] for m in methods)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Performance against acquisition length (n=%d held-out subjects)" % n_sub,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = a.out or os.path.join(a.dir, "figures", "Figure_sweep.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s (%d methods, k=%s)" % (out, len(methods),
                                           ",".join(str(k) for k in ks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
