#!/usr/bin/env python
"""plot_sweep_boxplots.py — per-metric frame-budget BOXPLOTS (F4), ALL metrics.

Reads a frame-budget sweep's per-(method, subject, n_frames) long CSV
(comparison_long.csv, produced by eval_comparison_table.py PHASE=sweep) and renders
ONE boxplot panel PER METRIC (auto-discovered — every numeric column, not a hand-picked
few; per feedback_plot_all_metrics). Each panel: x = n_frames, a box (distribution over
subjects) per method side-by-side. Emits a combined grid PNG + one PNG per metric.

Boxes = per-subject distribution ⇒ shows spread, not just the mean. NaN-safe: uMSE/uPSNR
go NaN at n_frames >= 10 (the 3-way held-out split is unavailable) — those positions are
simply left empty (annotated), while CNR/sCoV/SSIM stay valid at every n.

Usage:
  python scripts/plot_sweep_boxplots.py \
    --long_csv $EXP/comparison_fsl_seed42/sweep/comparison_long.csv \
    --out_dir  $EXP/comparison_fsl_seed42/sweep/figures
  # optional: --methods "CIG-VSS+EC-LRDA (BASE, FSL)" "CIG-VSS_M0-pv0" naive_mean   (order/filter)
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

# columns that are identifiers / not metrics (case-insensitive); anything else numeric = a metric
_ID_COLS = {"method", "subject", "subject_id", "sid", "n_frames", "nframes", "eff_frames",
            "n_batches", "batch", "z", "cy", "cx", "idx", "index", "seed", "slice", "run"}
# lower-is-better metrics (for the arrow glyph); everything else assumed higher-is-better
_LOWER_BETTER = {"umse", "scov_gm", "scov_wm", "efc", "hfen", "gmsd", "lapvar", "lapvar_ratio",
                 "gmwm_contrast_err", "nrmse", "cyc", "spillover", "bright_tail"}


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_args():
    p = argparse.ArgumentParser("per-metric frame-budget boxplots (all metrics)")
    p.add_argument("--long_csv", required=True, help="sweep comparison_long.csv (per method/subject/n_frames)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--methods", nargs="*", default=None,
                   help="subset/order of methods (default: all present, sorted with 'ours' first)")
    p.add_argument("--exclude_cols", nargs="*", default=[], help="extra columns to treat as non-metric")
    p.add_argument("--ncols", type=int, default=3, help="panels per row in the combined grid")
    p.add_argument("--min_finite", type=int, default=1, help="skip a metric with fewer finite values than this")
    p.add_argument("--umse_max_k", type=int, default=0,
                   help="drop uMSE points above this repetition count and say so in the panel "
                        "title. The estimator subtracts two nearly equal noise terms, so past a "
                        "point its per-subject spread is estimator noise rather than a "
                        "difference between methods; 0 keeps every k.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.long_csv)))
    if not rows:
        raise SystemExit(f"[boxplots] empty CSV: {args.long_csv}")

    cols = list(rows[0].keys())
    excl = _ID_COLS | {c.lower() for c in args.exclude_cols}
    # a metric column = not an id col, not a *_std, and has >=1 finite numeric value
    metrics = []
    for c in cols:
        if c.lower() in excl or c.lower().endswith("_std"):
            continue
        vals = [_fnum(r.get(c)) for r in rows]
        if sum(np.isfinite(vals)) >= args.min_finite and any(v == v for v in vals):
            # exclude columns that are non-numeric text (all NaN after coercion handled above)
            if np.isfinite(vals).any():
                metrics.append(c)
    if not metrics:
        raise SystemExit("[boxplots] no numeric metric columns discovered")

    methods_present = list(dict.fromkeys(r["method"] for r in rows))
    if args.methods:
        methods = [m for m in args.methods if m in methods_present]
    else:  # 'ours'/SAFE/BASE first, then the rest alphabetical
        def _key(m):
            ml = m.lower()
            return (0 if ("ours" in ml or "safe" in ml or "base" in ml) else 1, m)
        methods = sorted(methods_present, key=_key)
    nfs = sorted({_fnum(r.get("n_frames")) for r in rows if np.isfinite(_fnum(r.get("n_frames")))})

    # values[metric][method][nf] -> np.array of per-subject finite values
    def collect(metric):
        d = {m: {nf: [] for nf in nfs} for m in methods}
        for r in rows:
            m = r["method"]; nf = _fnum(r.get("n_frames")); v = _fnum(r.get(metric))
            if m in d and np.isfinite(nf) and nf in d[m] and np.isfinite(v):
                d[m][nf].append(v)
        return d

    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods)}
    width = 0.8 / max(len(methods), 1)

    def draw_panel(ax, metric):
        d = collect(metric)
        for mi, m in enumerate(methods):
            arrs, poss = [], []
            for i, nf in enumerate(nfs):
                a = d[m][nf]
                if metric.lower() == "umse" and args.umse_max_k and nf > args.umse_max_k:
                    a = []
                if a:                                   # skip empty (e.g. uMSE NaN at n>=10)
                    arrs.append(a)
                    poss.append(i + (mi - (len(methods) - 1) / 2.0) * width)
            if not arrs:
                continue
            bp = ax.boxplot(arrs, positions=poss, widths=width * 0.9, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", lw=1.0))
            for box in bp["boxes"]:
                box.set(facecolor=colors[m], alpha=0.65, edgecolor="black", lw=0.5)
            for w in bp["whiskers"] + bp["caps"]:
                w.set(color="black", lw=0.5)
        arrow = "↓" if metric.lower() in _LOWER_BETTER else "↑"
        note = ""
        if metric.lower() == "umse" and args.umse_max_k:
            note = f"  (k<={args.umse_max_k}; the estimator collapses above)"
        ax.set_title(f"{metric} {arrow}{note}", fontsize=10)
        ax.set_xticks(range(len(nfs)))
        ax.set_xticklabels([str(int(nf)) if nf == int(nf) else f"{nf:g}" for nf in nfs])
        ax.set_xlabel("n_frames")
        ax.grid(axis="y", ls=":", alpha=0.4)
        # annotate frame points where this metric is entirely missing (all methods empty)
        for i, nf in enumerate(nfs):
            if all(not d[m][nf] for m in methods):
                ax.text(i, ax.get_ylim()[0], "NaN", ha="center", va="bottom", fontsize=6, color="gray")

    # legend proxies
    handles = [plt.Line2D([0], [0], marker="s", ls="", markerfacecolor=colors[m],
                          markeredgecolor="black", markersize=8, label=m) for m in methods]

    # combined grid
    ncol = max(1, args.ncols)
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow), squeeze=False)
    ax_flat = axes.reshape(-1)
    for a, metric in zip(ax_flat, metrics):
        draw_panel(a, metric)
    for a in ax_flat[len(metrics):]:
        a.axis("off")
    fig.legend(handles=handles, loc="lower center", ncol=min(len(methods), 4), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Frame-budget degradation — per-metric boxplots (per-subject distributions)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    combined = out_dir / "all_metrics_boxplots.png"
    fig.savefig(combined, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[boxplots] wrote {combined}  ({len(metrics)} metrics x {len(methods)} methods x {len(nfs)} frame points)")

    # one PNG per metric
    for metric in metrics:
        f, a = plt.subplots(figsize=(6.2, 3.8))
        draw_panel(a, metric)
        a.legend(handles=handles, fontsize=7, frameon=False, loc="best")
        f.tight_layout()
        f.savefig(out_dir / f"box_{metric}.png", dpi=130, bbox_inches="tight")
        plt.close(f)
    print(f"[boxplots] wrote {len(metrics)} per-metric PNGs -> {out_dir}/box_*.png")
    print(f"[boxplots] metrics: {', '.join(metrics)}")


if __name__ == "__main__":
    main()
