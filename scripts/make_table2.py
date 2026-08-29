#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Assemble the manuscript's rCBF-agreement table from eval_cbf.py's output.

This table answers a different question from Table 1. Table 1 asks which method reconstructs
the better image from the same short acquisition; this one asks whether the physiological
quantity derived from the image survives the shortening. It therefore has no method column --
eval_cbf.py carries a single checkpoint through the CBF model, so no comparator has an ICC.

Three things the data forces:

* The 12-repetition row is the reference the other rows are compared against, so its ICC, bias
  and correlation are 1, 0 and 1 by construction. They are printed as "reference", not as
  measurements, because a column of 1.000 reads as a result.
* Limits of agreement are recomputed here from the per-subject rows. eval_cbf.py calls
  bland_altman(), which returns them, but stores only the bias.
* Scan time is a fraction of the full acquisition rather than a duration, so the numbers carry
  over to any ASL protocol with a different repetition count or TR. It counts the discarded
  first repetition by default: that frame is dropped as unstable, and if that is a property of
  the sequence reaching steady state then a prospective scan still has to acquire it.
  --warmup_reps 0 reports the plain k/N instead, which moves two repetitions from 23% to 17%,
  so the convention is stated in the caption either way.

Usage:
  python scripts/make_table2.py --dir <out>/medphys_eval
"""
from __future__ import annotations

import argparse
import json
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

BS = chr(92)


def stdev(vals):
    v = [x for x in vals if x == x]
    if len(v) < 2:
        return float("nan")
    mu = sum(v) / len(v)
    return (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def pct(k, full, warmup):
    """Scan time as a fraction of the full acquisition, so it transfers to other protocols."""
    return 100.0 * (k + warmup) / float(full + warmup)


def main() -> int:
    p = argparse.ArgumentParser("assemble the rCBF agreement table")
    p.add_argument("--dir", required=True)
    p.add_argument("--warmup_reps", type=int, default=1,
                   help="repetitions acquired but discarded; counted in scan time")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    src = os.path.join(a.dir, "cbf", "cbf_eval.json")
    if not os.path.isfile(src):
        raise SystemExit("no %s -- run the cbf phase first" % src)
    d = json.load(open(src, encoding="utf-8"))
    ref_k = d["ref_frames"]

    ref_arm = d.get("params", {}).get("ref_arm", "model")
    by = {}
    for r in d["rows"]:
        if r.get("arm", "model") == "model":
            by.setdefault(r["n_frames"], {})[r["subject"]] = r
    n_sub_rows = len(by.get(ref_k, {}))

    heads = ["Repetitions", "Scan time", "rCBF$_{GM}$", "rCBF$_{WM}$",
             "ICC$_{GM}$", "ICC$_{WM}$", "Bias (95 % LoA), GM", "Voxelwise $r$"]
    body, n_sub = [], 0
    for s in d["summary"]:
        k = s["n_frames"]
        row = [str(k), "%.0f %%" % pct(k, ref_k, a.warmup_reps),
               "%.3f" % s["rcbf_gm"], "%.3f" % s["rcbf_wm"]]
        n_sub = max(n_sub, n_sub_rows)
        if k == ref_k:
            row[0] = "%d (full)" % k
        if k == ref_k and ref_arm == "model":
            # the reference compared with itself: one, zero and one by construction
            row += ["--"] * 4
        else:
            row += ["%.3f" % s["icc_rcbf_gm"], "%.3f" % s["icc_rcbf_wm"],
                    "%+.3f (%+.3f, %+.3f)"
                    % (s["ba_bias_rcbf_gm"], s["ba_lo_rcbf_gm"], s["ba_hi_rcbf_gm"]),
                    "%.3f" % s["recon_corr"]]
        body.append(row)

    warm = ("counting" if a.warmup_reps else "excluding")
    caption = (
        "**Table 2. Relative CBF agreement.** Agreement of relative CBF, normalized to the "
        "gray- plus white-matter mean, against the %d-repetition acquisition, which is the "
        "reference. ICC is ICC(2,1), "
        "absolute agreement; limits of agreement are the bias $%spm$ 1.96 SD; voxelwise $r$ "
        "is measured within the brain mask. Scan time is a percentage of the full "
        "acquisition, %s the discarded first repetition. n = %d held-out subjects."
        % (ref_k, BS, warm, n_sub))
    md = [caption, "",
          "| " + " | ".join(heads) + " |",
          "|" + "|".join(["---"] * len(heads)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in body]

    tex = [BS + "begin{tabular}{l" + "r" * (len(heads) - 1) + "}", BS + "hline",
           " & ".join(heads) + " " + BS * 2, BS + "hline"]
    tex += [" & ".join(r) + " " + BS * 2 for r in body]
    tex += [BS + "hline", BS + "end{tabular}"]

    out = a.out or os.path.join(a.dir, "table2")
    os.makedirs(out, exist_ok=True)
    for name, lines in (("table2.md", md), ("table2.tex", tex)):
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print("  wrote %s" % os.path.join(out, name))
    print()
    print("\n".join(md))
    print()
    print("scan time, both conventions (%% of the full acquisition):")
    for s in d["summary"]:
        k = s["n_frames"]
        print("  k=%2d   k/N = %3.0f%%   (k+1)/(N+1) = %3.0f%%"
              % (k, pct(k, ref_k, 0), pct(k, ref_k, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
