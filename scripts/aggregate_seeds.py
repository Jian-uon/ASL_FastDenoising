# -*- coding: utf-8 -*-
"""Aggregate per-seed CIG-Net v2 comparison tables into mean ± SE across seeds.

phase4_eval.sh / phase2b_eval.sh write ONE self-contained table per seed
(`$EXP/comparison_seed<S>/` and `$EXP/ablation_seed<S>/`, via
scripts/eval_comparison_table.py). This script rolls the 3 seeds up into the
mean ± SE numbers the manuscript reports (§4.4 / §4.5), and combines the per-seed
within-sample Wilcoxon p-values (ours vs each baseline) across seeds via Fisher's
method to give an across-seed significance summary.

What "mean ± SE across seeds" means here
----------------------------------------
Each seed already reduces the whole val split to ONE number per (method, n_frames)
per metric — the pooled uMSE, or the sample-mean of CNR/sCoV/etc. We treat those K
per-seed point estimates as the sample, and report their mean and standard error
SE = std(ddof=1)/sqrt(K). SE needs K>=2 seeds; with one seed it is NaN (the mean is
still the single value). NaN per-seed values (e.g. uMSE at n_frames>=10, where no
disjoint hold-out exists) are dropped, and K is reported per cell.

Wilcoxon across seeds (optional)
--------------------------------
Each seed's `wilcoxon_ours_vs_baselines.csv` holds a within-sample (per val sample)
paired Wilcoxon p for ours vs each baseline, per metric, per n_frames. Combining K
independent per-seed p's via Fisher: X^2 = -2*sum(ln p), df = 2K, p_fisher =
chi2.sf(X^2, 2K). This is a meta-analytic "does the effect hold across seeds" summary;
the per-seed p's are also kept side by side (with K=3 a paired-across-seeds test would
be underpowered, so Fisher over the within-seed tests is the honest choice).

Works for BOTH the main comparison and the ablation table (same CSV schema).

Usage
-----
# auto-locate $EXP/comparison_seed{42,1,2}/ :
python scripts/aggregate_seeds.py --exp /fs1/home/duancaohui/jian/exp \
    --seeds 42 1 2 --prefix comparison --out_dir /fs1/home/duancaohui/jian/exp/comparison_agg

# ablation table:
python scripts/aggregate_seeds.py --exp $EXP --seeds 42 1 2 --prefix ablation \
    --out_dir $EXP/ablation_agg

# or point at explicit per-seed dirs (or the summary CSVs directly):
python scripts/aggregate_seeds.py --in DIR_seed42 DIR_seed1 DIR_seed2 --out_dir OUT
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple

# Per-seed POINT estimate for each metric = this column in comparison_summary.csv.
# (umse uses the pooled value; the rest use the per-seed sample mean.)
#   key, summary_column, arrow(better direction)
METRICS: List[Tuple[str, str, str]] = [
    ("umse",         "umse_pooled",       "down"),
    ("cnr",          "cnr_mean",          "up"),
    ("snr_gm",       "snr_gm_mean",       "up"),
    ("snr_wm",       "snr_wm_mean",       "up"),
    ("scov_gm",      "scov_gm_mean",      "down"),
    ("scov_wm",      "scov_wm_mean",      "down"),
    ("efc",          "efc_mean",          "down"),
    ("psnr_ref",     "psnr_ref_mean",     "up"),
    ("ssim_ref",     "ssim_ref_mean",     "up"),
    ("hfen",         "hfen_mean",         "down"),
    ("lapvar_ratio", "lapvar_ratio_mean", "-"),
]


def _f(x: str) -> float:
    """Parse a CSV cell as float; blanks / 'nan' / 'NA' -> NaN."""
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def _finite(xs: List[float]) -> List[float]:
    return [x for x in xs if x == x and not math.isinf(x)]  # drop NaN/inf


def _mean_se(xs: List[float]) -> Tuple[float, float, int]:
    """Across-seed mean, standard error (std ddof=1 / sqrt(K)), and K used."""
    v = _finite(xs)
    k = len(v)
    if k == 0:
        return float("nan"), float("nan"), 0
    mean = sum(v) / k
    if k < 2:
        return mean, float("nan"), k
    var = sum((x - mean) ** 2 for x in v) / (k - 1)          # sample variance
    se = math.sqrt(var) / math.sqrt(k)
    return mean, se, k


# ------------------------------------------------------------------
# Locate & load per-seed CSVs
# ------------------------------------------------------------------

def resolve_dirs(args) -> List[str]:
    if args.inputs:
        return list(args.inputs)
    if not (args.exp and args.seeds):
        raise SystemExit("Provide either --in DIR... or (--exp EXP --seeds S...).")
    dirs = []
    for s in args.seeds:
        d = os.path.join(args.exp, f"{args.prefix}_seed{s}")
        dirs.append(d)
    return dirs


def summary_csv_path(d: str) -> str:
    """Accept a dir (containing comparison_summary.csv) or the csv itself."""
    if os.path.isdir(d):
        return os.path.join(d, "comparison_summary.csv")
    return d


def load_summary(path: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    """(method, n_frames) -> {summary_column: float}."""
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["method"].strip(), int(float(row["n_frames"])))
            except (KeyError, ValueError):
                continue
            out[key] = {k: _f(v) for k, v in row.items() if k not in ("method", "n_frames")}
    return out


def load_wilcoxon(path: str) -> Dict[Tuple[str, int, str], float]:
    """(baseline, n_frames, metric) -> p_value (NaN if 'NA'/missing)."""
    out: Dict[Tuple[str, int, str], float] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["baseline"].strip(), int(float(row["n_frames"])), row["metric"].strip())
            except (KeyError, ValueError):
                continue
            out[key] = _f(row.get("p_value", "nan"))
    return out


# ------------------------------------------------------------------
# Fisher combination of per-seed p-values
# ------------------------------------------------------------------

def fisher_combine(ps: List[float]) -> Tuple[float, float, int]:
    """Fisher: X^2 = -2 sum ln p, df=2K -> (stat, p_combined, K). p floored at 1e-300."""
    v = _finite(ps)
    v = [min(max(p, 1e-300), 1.0) for p in v]
    k = len(v)
    if k == 0:
        return float("nan"), float("nan"), 0
    stat = -2.0 * sum(math.log(p) for p in v)
    try:
        from scipy.stats import chi2
        p_comb = float(chi2.sf(stat, 2 * k))
    except Exception:
        # No scipy: report the stat; leave combined p NaN rather than guess.
        p_comb = float("nan")
    return stat, p_comb, k


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Aggregate per-seed comparison tables (mean ± SE)")
    ap.add_argument("--in", dest="inputs", nargs="+", default=None,
                    help="Per-seed dirs (each with comparison_summary.csv) or CSV paths.")
    ap.add_argument("--exp", type=str, default=None, help="EXP root ($EXP).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds, e.g. 42 1 2.")
    ap.add_argument("--prefix", type=str, default="comparison",
                    help="Per-seed dir prefix: 'comparison' (main) or 'ablation'. Default comparison.")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    dirs = resolve_dirs(args)
    os.makedirs(args.out_dir, exist_ok=True)

    per_seed: List[Dict[Tuple[str, int], Dict[str, float]]] = []
    wil_per_seed: List[Dict[Tuple[str, int, str], float]] = []
    used = []
    for d in dirs:
        sp = summary_csv_path(d)
        if not os.path.isfile(sp):
            print(f"[agg] SKIP (no summary): {sp}")
            continue
        per_seed.append(load_summary(sp))
        wdir = os.path.dirname(sp)
        wil_per_seed.append(load_wilcoxon(os.path.join(wdir, "wilcoxon_ours_vs_baselines.csv")))
        used.append(sp)
        print(f"[agg] loaded {sp}")
    if not per_seed:
        raise SystemExit("[agg] no per-seed comparison_summary.csv found — nothing to aggregate.")
    print(f"[agg] aggregating {len(per_seed)} seed(s).")

    # union of (method, n_frames) keys across seeds
    keys = sorted({k for ps in per_seed for k in ps.keys()}, key=lambda t: (t[1], t[0]))

    # ---- aggregated_summary.csv ----
    out_csv = os.path.join(args.out_dir, "aggregated_summary.csv")
    header = ["method", "n_frames", "n_seeds_max"]
    for mkey, _, _ in METRICS:
        header += [f"{mkey}_mean", f"{mkey}_se", f"{mkey}_k"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (method, nf) in keys:
            row = [method, nf, len(per_seed)]
            for _mkey, col, _ in METRICS:
                vals = [ps[(method, nf)][col] for ps in per_seed
                        if (method, nf) in ps and col in ps[(method, nf)]]
                mean, se, k = _mean_se(vals)
                row += [f"{mean:.6f}", f"{se:.6f}", k]
            w.writerow(row)
    print(f"[agg] wrote {out_csv}")

    # ---- aggregated_table.md at the operating point (smallest n_frames, = 0) ----
    nf0 = min(nf for (_m, nf) in keys)
    out_md = os.path.join(args.out_dir, "aggregated_table.md")
    show = [("umse", "uMSE↓"), ("cnr", "CNR↑"), ("scov_gm", "sCoV-GM↓"),
            ("scov_wm", "sCoV-WM↓"), ("efc", "EFC↓"), ("psnr_ref", "psnr_ref↑"),
            ("lapvar_ratio", "lapvR")]
    col_for = {mkey: col for mkey, col, _ in METRICS}
    with open(out_md, "w") as f:
        f.write(f"# Cross-seed aggregate @ n_frames={nf0} (0 = operating point)\n\n")
        f.write(f"Seeds aggregated: {len(per_seed)}. Cells are mean ± SE across seeds "
                f"(SE = std/√K; K shown when < n_seeds). Source: {', '.join(used)}\n\n")
        f.write("| Method | " + " | ".join(lbl for _k, lbl in show) + " |\n")
        f.write("|---" * (len(show) + 1) + "|\n")
        for (method, nf) in keys:
            if nf != nf0:
                continue
            cells = [method]
            for mkey, _lbl in show:
                col = col_for[mkey]
                vals = [ps[(method, nf)][col] for ps in per_seed
                        if (method, nf) in ps and col in ps[(method, nf)]]
                mean, se, k = _mean_se(vals)
                if mean != mean:
                    cells.append("—")
                elif se != se:
                    cells.append(f"{mean:.4f} (K=1)")
                else:
                    tag = "" if k == len(per_seed) else f" (K={k})"
                    cells.append(f"{mean:.4f}±{se:.4f}{tag}")
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"[agg] wrote {out_md}")

    # ---- aggregated_wilcoxon.csv (Fisher across seeds) ----
    if any(wil_per_seed):
        wkeys = sorted({k for w in wil_per_seed for k in w.keys()},
                       key=lambda t: (t[1], t[0], t[2]))
        out_w = os.path.join(args.out_dir, "aggregated_wilcoxon.csv")
        with open(out_w, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["baseline", "n_frames", "metric", "k_seeds",
                        "p_per_seed", "fisher_stat", "fisher_p"])
            for (base, nf, metric) in wkeys:
                ps = [wp[(base, nf, metric)] for wp in wil_per_seed if (base, nf, metric) in wp]
                stat, p_comb, k = fisher_combine(ps)
                p_str = ";".join(f"{p:.3e}" if (p == p) else "nan" for p in ps)
                w.writerow([base, nf, metric, k, p_str,
                            f"{stat:.4f}" if stat == stat else "nan",
                            f"{p_comb:.3e}" if p_comb == p_comb else "nan"])
        print(f"[agg] wrote {out_w}")
    else:
        print("[agg] no per-seed wilcoxon_ours_vs_baselines.csv found — skipped Fisher combination.")

    print("[agg] done.")


if __name__ == "__main__":
    main()
