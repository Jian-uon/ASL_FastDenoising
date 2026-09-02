#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collapse a comparison sweep into one row per method, for plotting.

Two aggregations, both of which the raw CSVs get wrong for a figure:

1. **Seeds.** `proposed_seed42/1/2` are three training runs of ONE method, not three
   methods. Drawn as three lines they read as three competitors; the seed spread is
   uncertainty on the method and belongs in a band. Method names are grouped by stripping a
   trailing `_seed<n>`, so single-seed baselines pass through untouched.

   `--seeds` restricts which runs are read before any of that happens. It exists because
   averaging one arm over three seeds while its comparators are single runs is not a fair
   row: the averaged arm has had its training variance reduced by root three and the others
   have not. `--seeds 42` puts every arm on one run each; omitting it uses whatever is on
   disk. Either is defensible, but the choice should be deliberate and recorded, so the
   count that survived is printed and `n_seeds` carries it into the merged file.

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


def seed_of(method: str):
    """The trailing seed number, or None for an arm that carries no seed suffix."""
    m = SEED_RE.search(method)
    return int(m.group(0).rsplit("seed", 1)[1]) if m else None


def keep_seeds(rows, seeds):
    """Drop runs whose seed is not wanted. Arms with no seed suffix always pass."""
    if not seeds:
        return rows
    out = [r for r in rows if seed_of(r["method"]) in (None,) or seed_of(r["method"]) in seeds]
    dropped = sorted({r["method"] for r in rows} - {r["method"] for r in out})
    if dropped:
        print("  --seeds %s: dropped %s" % (
            ",".join(str(s) for s in sorted(seeds)), ", ".join(dropped)))
    if not out:
        raise SystemExit("--seeds kept nothing; check the seed numbers against the CSV")
    return out


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


def warn_sparse(rows, cols=("umse", "cnr", "snr_gm", "scov_gm"), tol=0.5):
    """Shout if a metric is mostly NaN, instead of letting it average away to a plausible number.

    A sweep written with the 2-voxel CSF erosion returned snr_gm on 3.8% of batches, and the
    few that survived took sigma_CSF from a handful of voxels, which underestimates it and put
    SNR near 8.5 where it belongs near 2.9. Averaging that silently produces a table nobody can
    tell is wrong by looking at it.
    """
    if not rows:
        return rows
    for c in cols:
        if c not in rows[0]:
            continue
        n = sum(1 for r in rows if fnum(r[c]) != fnum(r[c]))
        if n > tol * len(rows):
            print("  !! %s is NaN in %d of %d rows (%.0f%%) -- the sweep that wrote this is "
                  "not usable for that column" % (c, n, len(rows), 100.0 * n / len(rows)))
    return rows


def ensure_cnr_csf(rows):
    """Back-fill cnr_csf on a CSV written before the evaluation emitted it.

    The reported gray-white contrast is (mu_GM - mu_WM)/sigma_CSF, which is identically
    snr_gm - snr_wm because both SNRs already carry that sigma_CSF. Deriving it here lets the
    figures and Table 1 be redrawn from an existing sweep without re-running inference.
    """
    if not rows or "cnr_csf" in rows[0]:
        return rows
    for r in rows:
        g, w = fnum(r.get("snr_gm", "nan")), fnum(r.get("snr_wm", "nan"))
        r["cnr_csf"] = "" if (g != g or w != w) else repr(g - w)
    print("  cnr_csf back-filled from snr_gm - snr_wm (%d rows)" % len(rows))
    return rows


def merge_long(src, dst, seeds=None):
    rows = ensure_cnr_csf(warn_sparse(read(src)))
    if not rows:
        raise SystemExit("empty: %s" % src)
    metrics = [c for c in rows[0] if c not in ID_LONG]
    rows = keep_seeds(rows, seeds)

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


def merge_summary(src, dst, seeds=None):
    rows = ensure_cnr_csf(warn_sparse(read(src)))
    if not rows:
        raise SystemExit("empty: %s" % src)
    metrics = [c for c in rows[0] if c not in ID_SUMM]
    rows = keep_seeds(rows, seeds)

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
    p.add_argument("--seeds", type=str, default="",
                   help="keep only these seeds, e.g. '42' or '42,1,2'. Arms with no seed "
                        "suffix always pass. Default: use every run present.")
    a = p.parse_args()
    seeds = {int(s) for s in a.seeds.replace(",", " ").split()} if a.seeds.strip() else None
    for name, fn in (("long", merge_long), ("summary", merge_summary)):
        src = os.path.join(a.dir, "comparison_%s.csv" % name)
        if not os.path.isfile(src):
            print("  skip: no %s" % src)
            continue
        fn(src, os.path.join(a.dir, "comparison_%s_merged.csv" % name), seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
