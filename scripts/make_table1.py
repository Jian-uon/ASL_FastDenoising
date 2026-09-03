# -*- coding: utf-8 -*-
r"""Collapse Table 1 to one row per method, all measures over the selection range.

The repetition axis leaves the table. It has to go somewhere, because the paper's central
claim is that two repetitions reconstruct to better contrast than twelve averaged, so the
sweep becomes a figure (make_figure_sweep.py) and Table 1 becomes what it is now good at:
four methods under one condition, the condition their checkpoints were selected under.

uMSE is the mean of the per-length pooled values, exact because every length is scored on the
same slices. The reference-free measures are averaged per subject first, then reported as mean
+- SD over subjects, so the SD is between subjects and not between slices.
"""
import argparse
import csv
import io
import math
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

BS = chr(92)

PAPER_NAME = {
    "naive_mean":  "Repetition averaging",
    "vanilla_N2N": "UNet-N2N",
    "SwinIR_N2N":  "SwinIR-N2N",
    "proposed":    "Proposed",
}
ORDER = ["naive_mean", "vanilla_N2N", "SwinIR_N2N", "proposed"]

# (per-subject column, header, decimals)
# White matter is not a column. The coefficient of variation is defined on gray matter by the
# work it cites, and in white matter its denominator sits near the noise floor, so the ratio
# tracks smoothing rather than quality; Section 3.2 quotes the two values it needs inline.
REF_COLS = [("cnr_csf", "CNR", 3), ("scov_gm", "sCoV$_{GM}$", 3),
            ("snr_gm", "SNR$_{GM}$", 2)]


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def sd(vals):
    v = [x for x in vals if x == x]
    if len(v) < 2:
        return float("nan")
    mu = sum(v) / len(v)
    return (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def main() -> int:
    p = argparse.ArgumentParser("Table 1: methods under the selection condition")
    p.add_argument("--dir", required=True, help="medphys_eval directory (holds sweep/)")
    p.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6],
                   help="acquisition lengths to average over. Default 3 4 5 6 = the range "
                        "validation draws set A from, so the table is the condition the "
                        "checkpoints were selected under.")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    ks = sorted(set(a.ks))

    summ = list(csv.DictReader(
        open(os.path.join(a.dir, "sweep", "comparison_summary_merged.csv"), encoding="utf-8")))
    long_rows = list(csv.DictReader(
        open(os.path.join(a.dir, "sweep", "comparison_long_merged.csv"), encoding="utf-8")))

    # uMSE: mean of the per-length pooled values (exact: equal slice counts per length)
    umse = {}
    for m in ORDER:
        vals = [fnum(r["umse_pooled"]) for r in summ
                if r["method"] == m and int(float(r["n_frames"])) in ks]
        vals = [v for v in vals if v == v]
        umse[m] = sum(vals) / len(vals) if len(vals) == len(ks) else float("nan")

    # reference-free: average within subject over the lengths, then across subjects
    per_sub = {}
    for r in long_rows:
        if int(float(r["n_frames"])) not in ks:
            continue
        d = per_sub.setdefault(r["method"], {}).setdefault(r["subject_id"], {})
        for col, _, _ in REF_COLS:
            v = fnum(r.get(col, "nan"))
            if v == v:
                d.setdefault(col, []).append(v)

    n_sub = 0
    body = []
    for m in ORDER:
        if m not in per_sub:
            continue
        subs = per_sub[m]
        n_sub = max(n_sub, len(subs))
        u = umse[m]
        row = [PAPER_NAME.get(m, m),
               "--" if u != u else "%.5f" % u,
               "--" if not (u == u and u > 0) else "%.2f" % (10.0 * math.log10(1.0 / u))]
        for col, _, dec in REF_COLS:
            vals = [sum(d[col]) / len(d[col]) for d in subs.values()
                    if len(d.get(col, [])) == len(ks)]
            if not vals:
                row.append("--")
                continue
            s = sd(vals)
            txt = "%.*f" % (dec, sum(vals) / len(vals))
            if s == s:
                txt += " $" + BS + "pm$ " + "%.*f" % (dec, s)
            row.append(txt)
        body.append(row)

    span = ("%d to %d" % (ks[0], ks[-1]) if ks == list(range(ks[0], ks[-1] + 1))
            else ", ".join(str(k) for k in ks))
    heads = ["Method", "uMSE", "uPSNR (dB)"] + [h for _, h, _ in REF_COLS]
    caption = (
        "**Table 1. Reconstruction performance.** Every method, repetition averaging "
        "included, reconstructs from the same %s of the twelve acquired repetitions, "
        "drawn uniformly per slice. uMSE is pooled over the test set; the other measures "
        "are the mean $%spm$ SD across the %d test subjects, and over the three training "
        "runs for the proposed model." % (span, BS, n_sub))

    md = [caption, "", "| " + " | ".join(heads) + " |",
          "|" + "|".join(["---"] * len(heads)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in body]

    tex = [BS + "begin{tabular}{l" + "r" * (len(heads) - 1) + "}", BS + "hline",
           " & ".join(heads) + " " + BS * 2, BS + "hline"]
    tex += [" & ".join(r) + " " + BS * 2 for r in body]
    tex += [BS + "hline", BS + "end{tabular}"]

    out_dir = a.out or os.path.join(a.dir, "table1")
    os.makedirs(out_dir, exist_ok=True)
    for name, txt in (("table1.md", "\n".join(md) + "\n"),
                      ("table1.tex", "\n".join(tex) + "\n")):
        io.open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n").write(txt)
    print("wrote %s (%d methods over k=%s, n=%d subjects)"
          % (out_dir, len(body), ",".join(str(k) for k in ks), n_sub))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
