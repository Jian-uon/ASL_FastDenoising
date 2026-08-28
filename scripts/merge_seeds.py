#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collapse a comparison sweep into one row per method, for plotting.

Two aggregations, both of which the raw CSVs get wrong for a figure:

1. **Seeds.** `proposed_seed42/1/2` are three training runs of ONE method, not three
   methods. Drawn as three lines they read as three competitors; the seed spread is
   uncertainty on the method and belongs in a band. Method names are grouped by stripping a
   trailing `_seed<n>`, so single-seed baselines pass through untouched.

2. **Batches to subjects.** `comparison_long.csv` holds one row per *batch* (687 per method
   and repetition count here, against 32 held-out subjects). Slices of one subject are not
   independent, so a box drawn over batches is far tighter than the subject-level spread it
   appears to show. Rows are averaged within subject first.

uPSNR is recomputed from the merged uMSE rather than averaged in dB: the logarithm does not
commute with averaging, and averaging dB across seeds would not equal the dB of the mean
error. Everything else is a plain mean, with the across-seed standard deviation written
alongside as `<metric>_seedsd` for the degradation band.

Usage:
  python scripts/merge_seeds.py --dir $EXP/medphys_eval/sweep
  -> comparison_long_merged.csv, comparison_summary_merged.csv
"""
import argparse
import csv
import math
import os
import re
import statistics as st

ID_LONG = {"batch", "subject_id", "n_frames", "method"}
ID_SUMM = {"method", "n_frames", "eff_frames", "n_batches"}
SEED_RE = re.compile(r"_seed\d+$", re.IGNORECASE)


def group_of(method: str) -> str:
    return SEED_RE.sub("", method)


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def mean_finite(vals):
    v = [x for x in vals if x == x]
    return sum(v) / len(v) if v else float("nan")


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s (%d rows)" % (path, len(rows)))


def merge_long(src, dst):
    rows = read(src)
    if not rows:
        raise SystemExit("empty: %s" % src)
    metrics = [c for c in rows[0] if c not in ID_LONG]

    # batch -> subject, still per seed
    per_sub = {}
    for r in rows:
        key = (r["method"], r.get("subject_id", "?"), r["n_frames"])
        per_sub.setdefault(key, []).append(r)
    stage1 = {k: {m: mean_finite([fnum(x[m]) for x in v]) for m in metrics}
              for k, v in per_sub.items()}

    # seeds -> one method, per subject
    per_grp = {}
    for (meth, sub, nf), vals in stage1.items():
        per_grp.setdefault((group_of(meth), sub, nf), []).append(vals)
    out = []
    for (grp, sub, nf), seeds in sorted(per_grp.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        row = {"method": grp, "subject_id": sub, "n_frames": nf, "n_seeds": len(seeds)}
        for m in metrics:
            row[m] = mean_finite([s[m] for s in seeds])
        out.append(row)
    write(dst, ["method", "subject_id", "n_frames", "n_seeds"] + metrics, out)
    n_grp = len({r["method"] for r in out})
    n_sub = len({r["subject_id"] for r in out})
    print("  %d batches -> %d subjects x %d methods" % (len(rows), n_sub, n_grp))


def merge_summary(src, dst):
    rows = read(src)
    if not rows:
        raise SystemExit("empty: %s" % src)
    metrics = [c for c in rows[0] if c not in ID_SUMM]

    per_grp = {}
    for r in rows:
        per_grp.setdefault((group_of(r["method"]), r["n_frames"]), []).append(r)

    out = []
    for (grp, nf), seeds in sorted(per_grp.items(), key=lambda kv: (kv[0][0], fnum(kv[0][1]))):
        row = {"method": grp, "n_frames": nf, "n_seeds": len(seeds),
               "eff_frames": seeds[0].get("eff_frames", ""),
               "n_batches": seeds[0].get("n_batches", "")}
        for m in metrics:
            vals = [fnum(s[m]) for s in seeds]
            fin = [v for v in vals if v == v]
            row[m] = mean_finite(vals)
            row[m + "_seedsd"] = st.pstdev(fin) if len(fin) > 1 else 0.0
        # uPSNR follows the merged uMSE; averaging decibels would not be the decibels of the
        # mean error.
        u = row.get("umse_pooled", float("nan"))
        if "upsnr_pooled" in row:
            row["upsnr_pooled"] = 10.0 * math.log10(1.0 / u) if (u == u and u > 0) else float("nan")
        out.append(row)

    fields = ["method", "n_frames", "n_seeds", "eff_frames", "n_batches"]
    fields += [c for m in metrics for c in (m, m + "_seedsd")]
    write(dst, fields, out)
    for r in out:
        if r["n_seeds"] > 1:
            print("  %-16s merged %d seeds" % (r["method"], r["n_seeds"]))
            break


def main() -> int:
    p = argparse.ArgumentParser("collapse seeds and batches in a comparison sweep")
    p.add_argument("--dir", required=True, help="directory holding comparison_{long,summary}.csv")
    a = p.parse_args()
    for name, fn in (("long", merge_long), ("summary", merge_summary)):
        src = os.path.join(a.dir, "comparison_%s.csv" % name)
        if not os.path.isfile(src):
            print("  skip: no %s" % src)
            continue
        fn(src, os.path.join(a.dir, "comparison_%s_merged.csv" % name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
