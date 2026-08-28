#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Assemble the manuscript's main comparison table from the evaluation outputs.

Rows are the number of repetitions entering the reconstruction, and within each block the four
methods. Reporting the whole sweep rather than one operating point removes the need to justify
a choice of operating point, and it dissolves what used to be an orphan row: repetition
averaging at the full repetition count IS the acquisition as it is performed today, so it
becomes the last cell of a regular grid instead of a separate row with four empty columns.

Two things the data forces:

* uMSE needs at least three held-out repetitions to form its noise correction, so it is
  undefined for k >= 10 and untrustworthy well before that. Past --umse_max_k the column
  prints "--". That is a property of the estimator, not a missing measurement, and the caption
  says so.
* uMSE is pooled over the test set -- one estimate per method and k, not a mean of per-subject
  estimates -- so it carries no standard deviation. The reference-free measures are per-subject
  and are written as mean +- SD, recomputed from the per-subject table rather than taken from
  the summary's `*_std` columns, which are spreads over batches. Slices of one subject are not
  independent, so the batch spread understates what the caption claims.

rCBF agreement is not here: eval_cbf.py reconstructs with one checkpoint, so an ICC exists for
the proposed model and for no comparator. Those values are Table 2.

Usage:
  python scripts/make_table1.py --dir <out>/medphys_eval
  python scripts/make_table1.py --dir <out>/medphys_eval --ks 2 4 6 8 10 12
"""
import argparse
import csv
import math
import os
import sys

# The emitted files are UTF-8, but the console echo below can land on a code page that cannot
# represent the superscripts and the times sign (GBK on this Windows box, ASCII under some
# batch schedulers). Echo with replacement rather than letting a display detail abort a run
# whose output has already been written.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

BS = chr(92)  # the LaTeX escape, kept out of the literals below

PAPER_NAME = {
    "naive_mean":  "Repetition averaging",
    "vanilla_N2N": "PlainUNet-N2N",
    "SwinIR_N2N":  "SwinIR-N2N",
    "proposed":    "Proposed",
}
ORDER = ["naive_mean", "vanilla_N2N", "SwinIR_N2N", "proposed"]

# (csv column, header, decimals, carries a +- SD) -- the same four measures as Figure 3
COLS = [
    ("umse_pooled",  "uMSE",        5, False),
    ("cnr_mean",     "CNR",         3, True),
    ("scov_gm_mean", "sCoV$_{GM}$", 3, True),
    ("snr_gm_mean",  "SNR$_{GM}$",  2, True),
]


SUP = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
       "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸",
       "9": "⁹"}


def sci(v):
    """4.3e-04 -> 4.3 x 10^-4, in the notation Section 3.1 already uses."""
    m, e = ("%.1e" % v).split("e")
    return "%s × 10%s" % (m, "".join(SUP[c] for c in str(int(e))))


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


def main() -> int:
    p = argparse.ArgumentParser("assemble the main comparison table")
    p.add_argument("--dir", required=True, help="medphys_eval directory (holds sweep/ and cbf/)")
    p.add_argument("--ks", type=int, nargs="+", default=None,
                   help="repetition counts to report (default: every even k in the sweep)")
    p.add_argument("--full_k", type=int, default=12,
                   help="the acquisition's full repetition count, labelled as such")
    p.add_argument("--umse_max_k", type=int, default=5,
                   help="uMSE is printed only up to this k; past it the estimator's noise "
                        "correction is the size of the quantity it estimates")
    p.add_argument("--out", default=None, help="default <dir>/table1")
    a = p.parse_args()

    src = os.path.join(a.dir, "sweep", "comparison_summary_merged.csv")
    if not os.path.isfile(src):
        raise SystemExit("no %s -- run merge_seeds.py first" % src)
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    have = sorted({int(float(r["n_frames"])) for r in rows})
    ks = a.ks or [k for k in have if k % 2 == 0]
    missing = [k for k in ks if k not in have]
    if missing:
        print("NOTE: k=%s absent from the sweep -- rerun with KS covering them"
              % ",".join(str(k) for k in missing))
        ks = [k for k in ks if k in have]
    if not ks:
        raise SystemExit("none of the requested k are in %s" % src)

    idx = {(r["method"], int(float(r["n_frames"]))): r for r in rows}

    long_src = os.path.join(a.dir, "sweep", "comparison_long_merged.csv")
    per_sub = {}
    if os.path.isfile(long_src):
        for r in csv.DictReader(open(long_src, encoding="utf-8")):
            per_sub.setdefault((r["method"], int(float(r["n_frames"]))), []).append(r)
    n_sub = max((len(v) for v in per_sub.values()), default=0)

    def cell(m, k, col, dec, with_sd):
        r = idx.get((m, k))
        if r is None:
            return "--"
        if col == "umse_pooled" and k > a.umse_max_k:
            return "--"
        v = fnum(r.get(col, "nan"))
        if v != v:
            return "--"
        txt = "%.*f" % (dec, v)
        if with_sd:
            sd = stdev(fnum(x.get(col.replace("_mean", ""), "nan"))
                       for x in per_sub.get((m, k), []))
            if sd == sd:
                txt += " $" + BS + "pm$ " + "%.*f" % (dec, sd)
        return txt

    heads = ["Repetitions", "Method"] + [h for _, h, _, _ in COLS]
    body = []
    for k in ks:
        for i, m in enumerate([m for m in ORDER if (m, k) in idx]):
            name = PAPER_NAME.get(m, m)
            if m == "naive_mean" and k == a.full_k:
                name += " (full acquisition)"
            body.append([str(k) if i == 0 else "", name]
                        + [cell(m, k, c, d, sd) for c, _, d, sd in COLS])

    caption = (
        "**Table 1. Reconstruction performance.** Each block reconstructs from the stated "
        "number of acquired repetitions; repetition averaging at the full count is the "
        "acquisition as it is performed today. uMSE is pooled over the test set and so carries "
        "no standard deviation, and is left blank beyond %d repetitions, where too few are held "
        "out for its noise correction to be meaningful. The remaining measures need no held-out "
        "data and are the mean $%spm$ SD across the %d test subjects."
        % (a.umse_max_k, BS, n_sub))
    md = [caption, "",
          "| " + " | ".join(heads) + " |",
          "|" + "|".join(["---"] * len(heads)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in body]

    wp = {}
    wsrc = os.path.join(a.dir, "sweep", "wilcoxon_ours_vs_baselines.csv")
    if os.path.isfile(wsrc):
        for r in csv.DictReader(open(wsrc, encoding="utf-8")):
            if int(float(r["n_frames"])) == min(ks) and r["metric"] == "umse":
                wp[r["baseline"]] = float(r["p_value"])
    if wp:
        ordered = [m for m in ORDER if m in wp and m != "proposed"]
        md += ["", "Paired Wilcoxon on uMSE against the proposed model at %d repetitions: "
               % min(ks)
               + "; ".join("%s $p$ = %s" % (PAPER_NAME.get(m, m), sci(wp[m]))
                           for m in ordered) + "."]

    tex = [BS + "begin{tabular}{ll" + "r" * len(COLS) + "}", BS + "hline",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
