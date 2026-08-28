#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Assemble the manuscript's main comparison table from the evaluation outputs.

Reads the seed-merged sweep written by merge_seeds.py and emits the same table as Markdown and
as a LaTeX tabular, with the paper's method names and fixed significant digits, so the numbers
are never retyped.

Three things the data forces:

* uMSE is pooled over the whole test set -- one estimate per method, not a mean of per-subject
  estimates -- so it carries no standard deviation. The reference-free measures are per-subject
  and are written as mean +- SD. Mixing the two is only honest if the caption says which is
  which, so the emitted caption says it.
* Those standard deviations are recomputed from the per-subject table rather than taken from
  the summary's `*_std` columns, which are spreads over batches. There are 687 batches against
  32 subjects and slices of one subject are not independent, so the batch spread understates
  what the caption claims.
* rCBF agreement is not a column. eval_cbf.py reconstructs with one checkpoint, so an ICC
  exists for the proposed model and for no comparator; a column of blanks would imply the
  comparison was run and lost. Those values print separately, for the rCBF subsection.

Usage:
  python scripts/make_table1.py --dir <out>/medphys_eval --k 2
"""
import argparse
import csv
import json
import math
import os

BS = chr(92)  # the LaTeX escape, kept out of the literals below

PAPER_NAME = {
    "naive_mean":  "Temporal average",
    "vanilla_N2N": "PlainUNet-N2N",
    "SwinIR_N2N":  "SwinIR-N2N",
    "proposed":    "Proposed",
}
ORDER = ["naive_mean", "vanilla_N2N", "SwinIR_N2N", "proposed"]

# (csv column, header, decimals, carries a +- SD)
COLS = [
    ("umse_pooled",  "uMSE",           5, False),
    ("upsnr_pooled", "uPSNR (dB)",     2, False),
    ("snr_gm_mean",  "SNR$_{GM}$",     2, True),
    ("snr_wm_mean",  "SNR$_{WM}$",     2, True),
    ("cnr_mean",     "CNR",            3, True),
    ("lapvar_mean",  "Laplacian var.", 4, True),
]


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def stdev(vals):
    v = [x for x in vals if x == x]
    if len(v) < 2:
        return float("nan")
    mu = sum(v) / len(v)
    return (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def cell(row, col, dec, with_sd, sd_of):
    v = fnum(row.get(col, "nan"))
    if v != v:
        return "--"
    txt = "%.*f" % (dec, v)
    if with_sd:
        sd = sd_of(col.replace("_mean", ""))
        if sd == sd:
            txt += " $" + BS + "pm$ " + "%.*f" % (dec, sd)
    return txt


def main() -> int:
    p = argparse.ArgumentParser("assemble the main comparison table")
    p.add_argument("--dir", required=True, help="medphys_eval directory (holds sweep/ and cbf/)")
    p.add_argument("--k", type=int, default=2, help="repetition count the table reports")
    p.add_argument("--out", default=None, help="default <dir>/table1")
    a = p.parse_args()

    src = os.path.join(a.dir, "sweep", "comparison_summary_merged.csv")
    if not os.path.isfile(src):
        raise SystemExit("no %s -- run merge_seeds.py first" % src)
    rows = {r["method"]: r for r in csv.DictReader(open(src, encoding="utf-8"))
            if int(float(r["n_frames"])) == a.k}
    if not rows:
        raise SystemExit("no rows at k=%d in %s" % (a.k, src))
    methods = [m for m in ORDER if m in rows] + [m for m in rows if m not in ORDER]

    long_src = os.path.join(a.dir, "sweep", "comparison_long_merged.csv")
    per_sub = {}
    if os.path.isfile(long_src):
        for r in csv.DictReader(open(long_src, encoding="utf-8")):
            if int(float(r["n_frames"])) == a.k:
                per_sub.setdefault(r["method"], []).append(r)
    n_sub = max((len(v) for v in per_sub.values()), default=0)

    def sd_for(method):
        return lambda base: stdev(fnum(r.get(base, "nan")) for r in per_sub.get(method, []))

    heads = ["Method"] + [h for _, h, _, _ in COLS]
    body = [[PAPER_NAME.get(m, m)] + [cell(rows[m], c, d, sd, sd_for(m)) for c, _, d, sd in COLS]
            for m in methods]

    # The full-length acquisition as it is performed today, for the short one to be read
    # against. Its uMSE is not defined: the estimator needs held-out repetitions and this
    # image already uses all of them. Its Laplacian variance is recovered from the ratio the
    # sweep reports, which every method agrees on to within 3%.
    ref_src = rows[methods[0]]
    ref_row = ["Full acquisition (12 rep., average)"]
    for col, _, dec, with_sd in COLS:
        if col in ("umse_pooled", "upsnr_pooled"):
            ref_row.append("--")
            continue
        if col == "lapvar_mean":
            lv, rt = fnum(ref_src.get("lapvar_mean")), fnum(ref_src.get("lapvar_ratio_mean"))
            ref_row.append("%.*f" % (dec, lv / rt) if (rt == rt and rt > 0) else "--")
            continue
        rc = {"snr_gm_mean": "snr_gm_ref_mean", "snr_wm_mean": "snr_wm_ref_mean",
              "cnr_mean": "cnr_ref_mean"}.get(col)
        v = fnum(ref_src.get(rc, "nan")) if rc else float("nan")
        if v != v:
            ref_row.append("--")
            continue
        txt = "%.*f" % (dec, v)
        sd = stdev(fnum(r.get(rc.replace("_mean", ""), "nan")) for r in per_sub.get(methods[0], []))
        if with_sd and sd == sd:
            txt += " $" + BS + "pm$ " + "%.*f" % (dec, sd)
        ref_row.append(txt)
    body.append(ref_row)

    wp = {}
    wsrc = os.path.join(a.dir, "sweep", "wilcoxon_ours_vs_baselines.csv")
    if os.path.isfile(wsrc):
        for r in csv.DictReader(open(wsrc, encoding="utf-8")):
            if int(float(r["n_frames"])) == a.k and r["metric"] == "umse":
                wp[r["baseline"]] = float(r["p_value"])

    caption = ("**Table 1. Reconstruction performance from %d repetitions.** uMSE and uPSNR are "
               "pooled over the test set and therefore carry no standard deviation; the "
               "remaining measures are the mean $" + BS + "pm$ SD across the %d test subjects. "
               "Laplacian variance is reported as an over-smoothing guard, not as a quality "
               "score.") % (a.k, n_sub)
    md = [caption,
          "",
          "| " + " | ".join(heads) + " |",
          "|" + "|".join(["---"] * len(heads)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in body]
    if wp:
        md += ["", "Paired Wilcoxon on uMSE against the proposed model: "
               + "; ".join("%s $p$ = %.1e" % (PAPER_NAME.get(k, k), v)
                           for k, v in sorted(wp.items()) if k in PAPER_NAME) + "."]

    tex = [BS + "begin{tabular}{l" + "r" * len(COLS) + "}", BS + "hline",
           " & ".join(heads) + " " + BS * 2, BS + "hline"]
    tex += [" & ".join(r) + " " + BS * 2 for r in body]
    tex += [BS + "hline", BS + "end{tabular}"]

    out = a.out or os.path.join(a.dir, "table1")
    os.makedirs(out, exist_ok=True)
    for name, lines in (("table1.md", md), ("table1.tex", tex)):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print("  wrote %s" % os.path.join(out, name))
    print()
    print("\n".join(md))

    cj = os.path.join(a.dir, "cbf", "cbf_eval.json")
    if os.path.isfile(cj):
        d = json.load(open(cj, encoding="utf-8"))
        print()
        print("rCBF agreement (proposed model only; no comparator was reconstructed for CBF):")
        for r in d["summary"]:
            if r["n_frames"] == d["ref_frames"]:
                continue
            print("  %2d rep  ICC GM %.3f  ICC WM %.3f  BA bias %+.4f  voxelwise r %.3f  (n=%d)"
                  % (r["n_frames"], r["icc_rcbf_gm"], r["icc_rcbf_wm"],
                     r["ba_bias_rcbf_gm"], r["recon_corr"], r["n_subjects"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
