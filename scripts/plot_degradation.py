# -*- coding: utf-8 -*-
"""plot_degradation.py — re-draw the n_frames degradation curves from an existing
comparison_summary.csv (no GPU / no re-inference).

Mirrors eval_comparison_table._plot_degradation but reads the already-written summary, and
ALWAYS spans the full swept n_frames range with ticks at the actual values. uMSE (pooled,
unbiased risk) needs a >=3-frame disjoint hold-out, so it is NaN once <3 frames remain
(n_frames >= pool-2, e.g. 10/12 for a 12-NEX pool) and its line ends before the last tick —
by design. CNR / sCoV are reference-free and span every n.

Plots ONE FIGURE PER METRIC for ALL metrics in the summary (user request 2026-07-24):
umse_pooled + upsnr_pooled + every *_mean column (cnr, scov_gm/wm, hfr_tcsf, hfc_corr,
snr_gm/wm, efc, psnr_ref, ssim_ref, hfen, lapvar, gmsd, tenengrad, mi_t1/nmi_t1, ...).

Usage:
  python scripts/plot_degradation.py --dir <comparison dir with comparison_summary.csv>
Writes <dir>/figures/degradation_<metric>.png for every metric.
"""
from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


_UP = {"cnr", "cnr_ref", "upsnr", "psnr_ref", "ssim_ref", "hfr_tcsf", "hfc_corr",
       "hfc_energy", "snr_gm", "snr_wm", "tenengrad"}
_DOWN = {"umse", "scov_gm", "scov_wm", "efc", "l1_ref", "hfen", "lapvar",
         "lapvar_ratio", "gmsd", "gmwm_contrast_err", "mi_t1", "nmi_t1"}


def _ylabel(name):
    if name in _UP:
        return f"{name} (higher=better)"
    if name in _DOWN:
        return f"{name} (lower=better)"
    return name


def _discover_metrics(cols):
    """(summary column, display name) for every plottable metric. Pooled risk cols
    (umse_pooled/upsnr_pooled) keep their short name; *_mean columns drop the suffix;
    *_std / bookkeeping columns are skipped."""
    specs = []
    for c in cols:
        if c in ("method", "n_frames", "eff_frames", "n_batches") or c.endswith("_std"):
            continue
        if c == "umse_pooled":
            name = "umse"
        elif c == "upsnr_pooled":
            name = "upsnr"
        elif c.endswith("_mean"):
            name = c[:-5]
        else:
            continue
        specs.append((c, name))
    # umse first (headline unbiased risk), rest in discovery order
    specs.sort(key=lambda t: (t[1] != "umse",))
    return specs


def main():
    ap = argparse.ArgumentParser("re-plot n_frames degradation from comparison_summary.csv")
    ap.add_argument("--dir", required=True, help="dir containing comparison_summary.csv")
    ap.add_argument("--out", default=None, help="output figures dir (default <dir>/figures)")
    args = ap.parse_args()

    summ = os.path.join(args.dir, "comparison_summary.csv")
    if not os.path.isfile(summ):
        raise SystemExit(f"no comparison_summary.csv under {args.dir}")
    out = args.out or os.path.join(args.dir, "figures")
    os.makedirs(out, exist_ok=True)

    rows = list(csv.DictReader(open(summ)))
    nfs = sorted({fnum(r["n_frames"]) for r in rows if fnum(r["n_frames"]) > 0})
    if len(nfs) < 2:
        raise SystemExit(f"need >=2 swept n_frames>0, got {nfs}")
    methods = sorted({r["method"] for r in rows})

    specs = _discover_metrics(rows[0].keys()) if rows else []
    print(f"[plot_degradation] {len(specs)} metrics x {len(methods)} methods over n_frames={nfs}")
    for col, name in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in methods:
            ys = []
            for nf in nfs:
                vals = [fnum(r[col]) for r in rows
                        if r["method"] == m and fnum(r["n_frames"]) == nf]
                vals = [v for v in vals if v == v]        # drop NaN (uMSE undefined tail)
                ys.append(vals[0] if vals else float("nan"))
            ax.plot(nfs, ys, marker="o", label=m)
        ax.set_xlabel("n_frames (random subset of the 12-NEX pool setA∪setB)")
        ax.set_ylabel(_ylabel(name))
        ax.set_xticks(nfs)
        ax.set_xlim(min(nfs) - 0.3, max(nfs) + 0.3)
        if name == "umse":
            ax.set_title("uMSE undefined where <3 frames remain held-out (line ends early)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(out, f"degradation_{name}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
