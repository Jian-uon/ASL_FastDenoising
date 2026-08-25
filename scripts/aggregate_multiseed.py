# -*- coding: utf-8 -*-
"""aggregate_multiseed.py — pool the per-seed eval outputs into the paper's FINAL
mean ± std (across seeds) tables. No GPU; pure stdlib.

For each seed S it reads:
  <exp>/comparison_step400_seed<S>/comparison_summary.csv   (quality, n_frames=0)
  <exp>/comparison_step400_seed<S>/mismatch_<tag>/results.csv (T1-leakage L1)
and reports, per method, the mean ± between-seed std of each metric, plus a per-seed
appendix. "No silent caps": every seed/file that is missing is listed explicitly.

Usage:
  python scripts/aggregate_multiseed.py --exp /path/exp --seeds "42 1 2"
Writes <exp>/final_multiseed_table.md .
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st


# summary column -> (display, lower_is_better)
METRICS = [
    ("umse_pooled", "uMSE↓", True),
    ("upsnr_pooled", "uPSNR↑", False),
    ("cnr_mean", "CNR↑", False),
    ("scov_wm_mean", "sCoV-WM↓", True),
    ("hfc_corr_mean", "hfc_corr↑", False),
    ("gmsd_mean", "GMSD↓", True),
    ("ssim_ref_mean", "SSIM↑", False),
    ("nmi_t1_mean", "NMI(T1)↓", True),
]
# mismatch dir tag -> method label (must match eval_step400.sh / comparison labels)
MM_TAG2LABEL = {
    "zerot1": "CIG-VSS+EC_T1-off",
    "ecfull": "CIG-VSS+EC-LRDA (ours)",
    "a1":     "CIG-VSS+EC_A1",
    "csem":   "CIG-VSS+EC_csem",
    "mec":    "CIG-VSS_-EC",
}


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def mstd(vals):
    """(mean, std, n) over finite values; sample std for n>=2 else 0."""
    v = [x for x in vals if x == x]
    if not v:
        return float("nan"), float("nan"), 0
    m = st.mean(v)
    s = st.stdev(v) if len(v) >= 2 else 0.0
    return m, s, len(v)


def load_seed_quality(exp, seed):
    """{method: {col: value}} for n_frames=0, one seed."""
    p = os.path.join(exp, f"comparison_step400_seed{seed}", "comparison_summary.csv")
    if not os.path.isfile(p):
        return None, p
    out = {}
    for r in csv.DictReader(open(p)):
        if r.get("n_frames") not in ("0", "0.0"):
            continue
        out[r["method"]] = r
    return out, p


def load_seed_mismatch(exp, seed):
    """{label: mean match_vs_mismatch_l1} for one seed."""
    base = os.path.join(exp, f"comparison_step400_seed{seed}")
    out = {}
    for tag, label in MM_TAG2LABEL.items():
        p = os.path.join(base, f"mismatch_{tag}", "results.csv")
        if not os.path.isfile(p):
            continue
        rows = list(csv.DictReader(open(p)))
        vals = [fnum(r.get("match_vs_mismatch_l1")) for r in rows]
        vals = [v for v in vals if v == v]
        if vals:
            out[label] = st.mean(vals)
    return out


def main():
    ap = argparse.ArgumentParser("aggregate multi-seed eval into final tables")
    ap.add_argument("--exp", required=True, help="experiment root ($EXP)")
    ap.add_argument("--seeds", required=True, help="space/comma list, e.g. '42 1 2'")
    ap.add_argument("--out", default=None, help="output md (default <exp>/final_multiseed_table.md)")
    args = ap.parse_args()
    seeds = [s for s in args.seeds.replace(",", " ").split() if s]
    out_md = args.out or os.path.join(args.exp, "final_multiseed_table.md")

    # gather
    qual = {}   # seed -> {method -> row}
    found_seeds, missing = [], []
    for s in seeds:
        q, p = load_seed_quality(args.exp, s)
        if q is None:
            missing.append(f"seed {s}: {p} absent")
            continue
        qual[s] = q
        found_seeds.append(s)
    mm = {s: load_seed_mismatch(args.exp, s) for s in found_seeds}

    if not found_seeds:
        raise SystemExit(f"no comparison_summary.csv found for any of seeds {seeds} under {args.exp}")

    methods = sorted({m for s in found_seeds for m in qual[s]})

    with open(out_md, "w") as f:
        f.write(f"# Final multi-seed results (n_frames=0, test split)\n\n")
        f.write(f"Mean ± between-seed std over seeds **{{{', '.join(found_seeds)}}}** "
                f"({len(found_seeds)} seed(s)). Each per-seed value is that seed's pooled/"
                f"subject-mean metric; the ± is the spread across seeds (init robustness).\n\n")
        if missing:
            f.write("**Missing (not yet aggregated):** " + "; ".join(missing) + "\n\n")

        # ---- main quality table ----
        f.write("## Quality (mean ± std across seeds)\n\n")
        f.write("| Method | " + " | ".join(d for _, d, _ in METRICS) + " |\n")
        f.write("|" + "---|" * (len(METRICS) + 1) + "\n")
        for m in methods:
            cells = []
            for col, _disp, _lo in METRICS:
                vals = [fnum(qual[s][m].get(col)) for s in found_seeds if m in qual[s]]
                mu, sd, n = mstd(vals)
                cells.append("n/a" if mu != mu else (f"{mu:.4f} ± {sd:.4f}" if n >= 2 else f"{mu:.4f}"))
            f.write(f"| {m} | " + " | ".join(cells) + " |\n")

        # ---- T1-leakage mismatch table ----
        f.write("\n## T1 content-isolation — matched-vs-mismatched-T1 L1 (mean ± std across seeds)\n\n")
        f.write("Lower = tighter content isolation (less T1 footprint). zerot1 ≈ 0 is the leakage floor.\n\n")
        f.write("| Model | mismatch L1↓ |\n|---|---|\n")
        mm_labels = sorted({lab for s in found_seeds for lab in mm.get(s, {})})
        for lab in mm_labels:
            vals = [mm[s][lab] for s in found_seeds if lab in mm.get(s, {})]
            mu, sd, n = mstd(vals)
            cell = "n/a" if mu != mu else (f"{mu:.4f} ± {sd:.4f}" if n >= 2 else f"{mu:.4f}")
            f.write(f"| {lab} | {cell} |\n")

        # ---- per-seed appendix ----
        f.write("\n## Per-seed appendix (raw values)\n\n")
        for col, disp, _lo in METRICS:
            f.write(f"\n### {disp}\n\n")
            f.write("| Method | " + " | ".join(f"seed {s}" for s in found_seeds) + " |\n")
            f.write("|" + "---|" * (len(found_seeds) + 1) + "\n")
            for m in methods:
                cells = []
                for s in found_seeds:
                    v = fnum(qual[s][m].get(col)) if m in qual[s] else float("nan")
                    cells.append("—" if v != v else f"{v:.4f}")
                f.write(f"| {m} | " + " | ".join(cells) + " |\n")

    print(f"[aggregate_multiseed] seeds found: {found_seeds}  methods: {len(methods)}")
    if missing:
        print("[aggregate_multiseed] MISSING: " + "; ".join(missing))
    print(f"[aggregate_multiseed] wrote {out_md}")


if __name__ == "__main__":
    main()
