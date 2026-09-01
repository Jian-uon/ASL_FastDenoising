#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance against acquisition length, which Table 1 no longer carries.

Table 1 compares the methods under one condition. The paper's other claim is about the
acquisition axis -- that reconstruction quality barely depends on how many repetitions enter
it, so a short scan is enough -- and that claim needs the axis itself.

Repetition averaging is drawn as one of the methods, so its value at the full twelve
repetitions -- the image today's protocol delivers -- is the last point of that curve, and a
reconstruction curve sitting above it has bought more from a shorter scan than the complete
acquisition gives. No separate reference line is needed for that reading.

uMSE is plotted only as far as --umse_max_k. It is estimated from the repetitions left out of
the reconstruction, and the estimator averages (12-k)//3 of them into each of its three groups:
three at k=2, two from k=4, and a single frame from k=7, where the variance correction is taken
between two lone noisy frames. Past five the estimate is dominated by its own noise -- it turns
negative at six -- so the curve stops and the caption says why.

Reference-free measures need no hold-out and run the full width. Two of them trend the wrong
way: gray-matter SNR and the coefficient of variation both improve as repetitions are REMOVED,
because both reward smoothing and the network smooths harder when given less. That is the
reason to carry uMSE here at all -- without it only CNR has the right sign.

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

# (column, axis label, arrow marking the direction of improvement)
#
# uMSE comes from the pooled summary rather than the per-subject long table: it subtracts two
# noise terms, so single subjects land below zero and only the pooled value is on the scale
# Table 1 reports. It is also the only panel with a hold-out, hence the k cut.
#
# White matter is not shown. The measure is defined on gray matter by the work it cites, and
# white-matter perfusion at this field and PLD sits close to the noise floor, where a ratio
# with that mean in the denominator mostly reports smoothing. Figure 3 and Table 1 carry it.
UMSE = ("umse", "uMSE", "↓")
PANELS = [UMSE,
          ("cnr", "GM-WM CNR", "↑"),
          ("scov_gm", "sCoV$_{GM}$", "↓"),
          ("snr_gm", "SNR$_{GM}$", "↑")]


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
            if col == UMSE[0]:
                continue          # pooled, loaded separately
            v = fnum(r.get(col, "nan"))
            if v == v:
                d.setdefault(col, []).append(v)
    out = {}
    for m, ks in acc.items():
        for k, d in ks.items():
            out.setdefault(m, {})[k] = {c: (sum(d[c]) / len(d[c]), len(d[c]))
                                        for c in d if d[c]}
    return out


def load_pooled_umse(path, max_k):
    """-> {method: {k: umse_pooled}}, for k <= max_k only.

    Pooled, because uMSE subtracts two noise terms and single subjects land below zero; only
    the sum over the evaluation set is on the scale Table 1 reports. Values past max_k exist
    in the CSV but are dropped here: the estimator's three groups thin to one frame each and
    its spread swamps the differences being drawn.
    """
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        k = int(float(r["n_frames"]))
        v = fnum(r.get("umse_pooled", "nan"))
        if k <= max_k and v == v and v > 0:
            out.setdefault(r["method"], {})[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser("performance against acquisition length")
    p.add_argument("--dir", required=True)
    p.add_argument("--full_k", type=int, default=12,
                   help="acquisition length of the reference protocol")
    p.add_argument("--out", default=None)
    p.add_argument("--umse_max_k", type=int, default=5,
                   help="last k with a usable unbiased risk estimate; see load_pooled_umse")
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()

    src = os.path.join(a.dir, "sweep", "comparison_long_merged.csv")
    if not os.path.isfile(src):
        raise SystemExit("no %s -- run merge_seeds.py first" % src)
    data = load(src)
    pooled = os.path.join(a.dir, "sweep", "comparison_summary_merged.csv")
    umse = load_pooled_umse(pooled, a.umse_max_k) if os.path.isfile(pooled) else {}
    methods = [m for m in ORDER if m in data]
    ks = sorted(set().union(*(set(data[m]) for m in methods)))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 3.9))
    for ax, (col, label, arrow) in zip(axes, PANELS):
        # No reference rule. Repetition averaging at the full count is the last point of the
        # repetition-averaging curve, so a line across the panel would draw the same number
        # twice.
        is_umse = col == UMSE[0]
        for m in methods:
            if is_umse:
                xs = sorted(umse.get(m, {}))
                ys = [umse[m][k] for k in xs]
            else:
                xs = [k for k in ks if col in data[m].get(k, {})]
                ys = [data[m][k][col][0] for k in xs]
            if not xs:
                continue
            ax.plot(xs, ys, marker=MARKERS[m], ms=4.5, lw=1.6,
                    color=COLORS[m], label=PAPER_NAME.get(m, m), zorder=3)
        ax.set_xlabel("Repetitions entering the reconstruction", fontsize=9)
        ax.set_title("%s %s" % (label, arrow), fontsize=10)
        if is_umse:
            # An order of magnitude separates repetition averaging from every reconstruction;
            # on a linear axis that gap flattens the three curves that matter into one line.
            ax.set_yscale("log")
            ax.set_xticks([k for k in ks if k <= a.umse_max_k])
        else:
            ax.set_xticks(ks)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    # Count subjects from a reference-free column: uMSE is pooled and carries no per-subject
    # count, and it is the first panel.
    ref_col = next(c for c, _, _ in PANELS if c != UMSE[0])
    n_sub = max(data[m][ks[0]][ref_col][1] for m in methods)
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
