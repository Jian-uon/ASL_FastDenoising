#!/usr/bin/env python
"""Operating-point table: score EVERY best_<metric>.pth of a run on the full suite.

For each checkpoint in --ckpts (typically a run's checkpoints/best_*.pth, saved
in-loop by --save_per_metric_best), rebuild the runner ONCE, load its EMA weights,
and run the SAME self-supervised validation the in-loop selection uses
(scripts.eval_select_ckpt._validate_with_per_unit_umse). Emits a per-run
[ select-by-<metric> ckpt  x  full metric suite ] table.

Answers "if we'd taken the best-<metric> checkpoint as the operating point, what
would ALL the metrics be?" — the operating-point sensitivity table (user 2026-08-15:
eval each per-metric best, not just best_umse). EMA weights, @no_grad, VAL split
(the split the ckpts were selected on).

The Markdown shows a curated, ordered lead set (uMSE/uPSNR lead, CNR/sCoV/EFC/HFEN
supplementary, *_ref biased-vs-12NEX tagged); the CSV/JSON carry the FULL pooled_*
suite (every metric the validator returns) so nothing is silently dropped.

Usage (mirror the run's training args in --runner_args):
  python scripts/score_best_per_metric.py \
    --ckpts $LOGS/cig_vss_ec_csem_bayes_seed42/checkpoints/best_*.pth \
    --runner_args "<RA_BAYES ...> --name opbest_bayes" --label bayes \
    --out_md out/opbest_bayes.md --out_csv out/opbest_bayes.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.eval_select_ckpt import (  # noqa: E402
    _build_runner, _load_model_weights, _validate_with_per_unit_umse,
)

# Curated, ordered LEAD columns for the Markdown (label -> pooled_* key). uMSE/uPSNR
# lead (2026-08-11 uMSE-lead); the rest are supplementary. *_ref = biased-vs-12NEX
# (reported, NEVER selected on). The CSV/JSON additionally carry every other pooled_* key.
_LEAD = [
    ("uMSE",      "pooled_umse"),
    ("uPSNR",     "pooled_upsnr"),
    ("uPSNR_cyc", "pooled_upsnr_cyc"),
    ("CNR",       "pooled_cnr"),
    ("sCoV_gm",   "pooled_scov_gm"),
    ("sCoV_wm",   "pooled_scov_wm"),
    ("EFC",       "pooled_efc"),
    ("HFEN",      "pooled_hfen"),
    ("cyc",       "pooled_cyc"),
    ("SNR_gm",    "pooled_snr_gm"),
    ("SNR_wm",    "pooled_snr_wm"),
    ("CNR_ref*",  "pooled_cnr_ref"),
]
_LEAD_KEYS = [k for _, k in _LEAD]


# better-direction per lead metric (for the heatmap orientation + arrow glyph)
_DIR = {
    "pooled_umse": "min", "pooled_upsnr": "max", "pooled_upsnr_cyc": "max",
    "pooled_cnr": "max", "pooled_scov_gm": "min", "pooled_scov_wm": "min",
    "pooled_efc": "min", "pooled_hfen": "min", "pooled_cyc": "min",
    "pooled_snr_gm": "max", "pooled_snr_wm": "max", "pooled_cnr_ref": "max",
}


def _fmt(v):
    if v is None or v != v:  # None / NaN
        return "—"
    a = abs(v)
    return f"{v:.5f}" if a < 0.1 else (f"{v:.4f}" if a < 100 else f"{v:.2f}")


def _plot(rows, out_fig):
    """Heatmap of the operating-point table: rows = best-<metric> ckpt, cols = lead metrics,
    colour = column-normalised value (green = better within column, direction-aware), cell = raw."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [n for n, _ in rows]
    M = np.array([[(m.get(k) if m.get(k) is not None else np.nan) for _, k in _LEAD]
                  for _, m in rows], dtype=float)
    Z = np.full_like(M, np.nan)
    for j, (_, k) in enumerate(_LEAD):
        col = M[:, j]
        fin = col[np.isfinite(col)]
        if fin.size == 0:
            continue
        lo, hi = float(fin.min()), float(fin.max())
        if hi > lo:
            z = (col - lo) / (hi - lo)
            if _DIR.get(k) == "min":
                z = 1.0 - z
        else:
            z = np.where(np.isfinite(col), 0.5, np.nan)
        Z[:, j] = z

    fig, ax = plt.subplots(figsize=(0.92 * len(_LEAD) + 2.0, 0.52 * len(rows) + 1.8))
    ax.imshow(Z, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(_LEAD)))
    ax.set_xticklabels([f"{lbl}\n{'↓' if _DIR.get(k) == 'min' else '↑'}" for lbl, k in _LEAD], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(_LEAD)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, _fmt(M[i, j]), ha="center", va="center", fontsize=6.3)
    ax.set_title("Operating-point trade-off (green = better within column; full suite in CSV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"[opbest] wrote {out_fig}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="checkpoints to score (globs OK, e.g. .../checkpoints/best_*.pth)")
    p.add_argument("--runner_args", required=True,
                   help="the run's training CLI (same string the eval phases pass as RA)")
    p.add_argument("--label", default="", help="run tag for the table header / json")
    p.add_argument("--out_md", required=True)
    p.add_argument("--out_csv", default="")
    p.add_argument("--out_fig", default="", help="also render the table as a heatmap PNG")
    args = p.parse_args()

    ckpts = []
    for c in args.ckpts:
        g = sorted(glob.glob(c))
        ckpts.extend(g if g else [c])
    ckpts = [c for c in dict.fromkeys(ckpts) if Path(c).is_file()]  # de-dup, keep order
    if not ckpts:
        raise SystemExit(f"[opbest] no checkpoints matched {args.ckpts}")

    runner, _ = _build_runner(args.runner_args)
    rows = []
    for ck in ckpts:
        _load_model_weights(runner, ck)
        m = _validate_with_per_unit_umse(runner)
        name = Path(ck).stem  # best_umse / best_cnr_pred / ...
        rows.append((name, m))
        print(f"[opbest] {name:24s} uMSE={_fmt(m.get('pooled_umse'))} "
              f"uPSNR={_fmt(m.get('pooled_upsnr'))} CNR={_fmt(m.get('pooled_cnr'))}")

    if not rows:
        raise SystemExit("[opbest] no checkpoint scored.")

    # Full metric key union (lead first, then any remaining pooled_* keys sorted).
    all_keys = list(_LEAD_KEYS)
    for _, m in rows:
        for k in m:
            if k.startswith("pooled_") and k not in all_keys:
                all_keys.append(k)
    extra_keys = [k for k in all_keys if k not in _LEAD_KEYS]

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    hdr = args.label or Path(args.out_md).stem
    with open(args.out_md, "w") as f:
        f.write(f"# Operating-point table (val, EMA) — {hdr}\n\n")
        f.write("Each row = this run's best-`<metric>`.pth checkpoint, re-scored on the full "
                "self-supervised suite. uMSE/uPSNR lead (2026-08-11); `*_ref` = biased-vs-12NEX "
                "(reported, never selected on). Full suite in the CSV/JSON siblings.\n\n")
        f.write("| select-by | " + " | ".join(lbl for lbl, _ in _LEAD) + " |\n")
        f.write("|" + "---|" * (len(_LEAD) + 1) + "\n")
        for name, m in rows:
            f.write(f"| {name} | " + " | ".join(_fmt(m.get(k)) for _, k in _LEAD) + " |\n")
        f.write("\nRead: rows should CLUSTER on the lead metrics if selection is robust; a row that "
                "wins its own metric but is far worse on uMSE/uPSNR marks a biased operating point.\n")
    print(f"[opbest] wrote {args.out_md}")

    out_json = args.out_md[:-3] + ".json" if args.out_md.endswith(".md") else args.out_md + ".json"
    json.dump({"label": args.label, "split": "val",
               "rows": [{"ckpt": n, **{k: m.get(k) for k in all_keys}} for n, m in rows]},
              open(out_json, "w"), indent=2)
    print(f"[opbest] wrote {out_json}")

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("select_by," + ",".join(all_keys) + "\n")
            for name, m in rows:
                cells = ["" if (m.get(k) is None or m.get(k) != m.get(k)) else f"{m.get(k)}" for k in all_keys]
                f.write(name + "," + ",".join(cells) + "\n")
        print(f"[opbest] wrote {args.out_csv}  (full suite: {len(_LEAD_KEYS)} lead + {len(extra_keys)} extra)")

    if args.out_fig:
        try:
            _plot(rows, args.out_fig)
        except Exception as e:  # plotting must never sink the (already-written) table
            print(f"[opbest] figure skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
