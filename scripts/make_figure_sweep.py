#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance against repetition count, which Table 1 no longer carries.

Table 1 compares the methods under one condition. The paper's other claim is about the
acquisition axis -- that reconstruction quality barely depends on how many repetitions enter
it, so a short scan is enough -- and that claim needs the axis itself.

Repetition averaging is drawn as one of the methods, so its value at the full twelve
repetitions -- the image today's protocol delivers -- is the last point of that curve, and a
reconstruction curve sitting above it has bought more from a shorter scan than the complete
acquisition gives. No separate reference line is needed for that reading.

Only reference-free measures are drawn, so every panel runs the full width of the axis. uMSE
is not among them: it is estimated from the repetitions left out of the reconstruction, and the
estimator averages (12-k)//3 of them into each of its three groups -- three at k=2, two from
k=4, a single frame from k=7 -- so it is unusable over most of the range this figure covers.
Table 1 and Figure 3 carry it at the selection condition instead.

Two of the panels trend the wrong way: gray-matter SNR and the coefficient of variation both
improve as repetitions are REMOVED, because both reward smoothing and the network smooths
harder when given less. CNR is the panel with the right sign, and the caption says so.

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
    "UNet_T1concat": "UNet-N2N+T1",
    "proposed":    "Proposed",
}
# UNet_T1concat is trained and scored, and its numbers land in the sweep CSVs; it is held
# out of ORDER until there is a decision to report it. Its display name, colour and marker
# are kept above, so putting it back is one word.
ORDER = ["naive_mean", "vanilla_N2N", "SwinIR_N2N", "proposed"]
COLORS = {"naive_mean": "#b0b0b0", "vanilla_N2N": "#7fb3d5",
          "SwinIR_N2N": "#f0b27a", "UNet_T1concat": "#7d3c98", "proposed": "#c0392b"}
MARKERS = {"naive_mean": "s", "vanilla_N2N": "^", "SwinIR_N2N": "D",
           "UNet_T1concat": "v", "proposed": "o"}

# (column, axis label, arrow marking the direction of improvement)
#
# White matter is not shown. The measure is defined on gray matter by the work it cites, and
# white-matter perfusion at this field and PLD sits close to the noise floor, where a ratio
# with that mean in the denominator mostly reports smoothing. Figure 3 and Table 1 carry it.
PANELS = [("cnr_csf", "CNR", "↑"),
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
    p = argparse.ArgumentParser("performance against repetition count")
    p.add_argument("--dir", required=True)
    p.add_argument("--full_k", type=int, default=12,
                   help="repetition count of the reference protocol")
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()

    src = os.path.join(a.dir, "sweep", "comparison_long_merged.csv")
    if not os.path.isfile(src):
        raise SystemExit("no %s -- run merge_seeds.py first" % src)
    data = load(src)
    methods = [m for m in ORDER if m in data]
    ks = sorted(set().union(*(set(data[m]) for m in methods)))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 3.9))
    for ax, (col, label, arrow) in zip(axes, PANELS):
        # No reference rule. Repetition averaging at the full count is the last point of the
        # repetition-averaging curve, so a line across the panel would draw the same number
        # twice.
        # Plotted against position, not against k: the grid skips 11, so a linear axis would
        # leave one gap twice as wide as the others for no reason a reader could use.
        pos = {k: i for i, k in enumerate(ks)}
        for m in methods:
            xs = [k for k in ks if col in data[m].get(k, {})]
            ys = [data[m][k][col][0] for k in xs]
            if not xs:
                continue
            ax.plot([pos[k] for k in xs], ys, marker=MARKERS[m], ms=4.5, lw=1.6,
                    color=COLORS[m], label=PAPER_NAME.get(m, m), zorder=3)
        ax.set_xlabel("Number of repetitions", fontsize=9)
        ax.set_title("%s %s" % (label, arrow), fontsize=10)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlim(-0.4, len(ks) - 0.6)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Performance against repetition count", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    out = a.out or os.path.join(a.dir, "figures", "Figure_sweep.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s (%d methods, k=%s)" % (out, len(methods),
                                           ",".join(str(k) for k in ks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
