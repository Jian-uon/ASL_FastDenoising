#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Rebuild the summary block of cbf_eval.json from its own per-subject rows.

eval_cbf.py writes both, and the summary is derived, so it can be corrected without repeating
the inference. That is worth having because a run of eval_cbf.py takes a GPU and an hour, and
the reference the summary is computed against is a one-line decision that has already been got
wrong once: the voxelwise correlation was measured against the plain full-repetition average
while the ICC and Bland-Altman were still measured against the model's own output at that
count, so the same table compared two different things.

Usage:
  python scripts/recompute_cbf_summary.py --dir <out>/medphys_eval [--ref_arm mean]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.cbf_metrics import icc_agreement, bland_altman


def main() -> int:
    p = argparse.ArgumentParser("rebuild the cbf summary from its rows")
    p.add_argument("--dir", required=True)
    p.add_argument("--ref_arm", choices=["mean", "model"], default="mean",
                   help="what the reference is at the full repetition count")
    p.add_argument("--out", default=None, help="default: rewrite cbf_eval.json in place")
    a = p.parse_args()

    src = os.path.join(a.dir, "cbf", "cbf_eval.json")
    d = json.load(open(src, encoding="utf-8"))
    ref_n = d["ref_frames"]
    rows = d["rows"]

    def pool(arm):
        out = {}
        for r in rows:
            if r.get("arm", "model") == arm:
                out.setdefault(r["n_frames"], {})[r["subject"]] = r
        return out

    model, mean = pool("model"), pool("mean")
    ref_pool = mean if (a.ref_arm == "mean" and mean.get(ref_n)) else model
    if a.ref_arm == "mean" and not mean.get(ref_n):
        print("WARNING: no averaging arm in the rows; falling back to the model reference")
    ref = ref_pool[ref_n]
    print("reference: %s arm at n=%d (%d subjects)"
          % ("mean" if ref_pool is mean else "model", ref_n, len(ref)))

    ns = sorted(model)
    summary = []
    for n in ns:
        cur = model[n]
        sids = [s for s in cur if s in ref]
        row = {"n_frames": n, "n_subjects": len(cur)}
        for k in ("cbf_gm", "cbf_wm", "rcbf_gm", "rcbf_wm"):
            row[k] = float(np.nanmean([cur[s][k] for s in cur]))
        row["gm_wm_ratio"] = row["cbf_gm"] / row["cbf_wm"] if row["cbf_wm"] else float("nan")
        row["recon_corr"] = float(np.nanmean([cur[s]["recon_corr"] for s in cur]))
        row["recon_nrmse"] = float(np.nanmean([cur[s]["recon_nrmse"] for s in cur]))
        for k in ("cbf_gm", "cbf_wm", "rcbf_gm", "rcbf_wm"):
            m = mean.get(n, {})
            row["mean_" + k] = (float(np.nanmean([m[s][k] for s in m])) if m else float("nan"))

        same = (ref_pool is model) and n == ref_n
        if same:
            row["icc_rcbf_gm"] = row["icc_rcbf_wm"] = 1.0
            for t in ("gm", "wm"):
                row["ba_bias_rcbf_%s" % t] = 0.0
                row["ba_sd_rcbf_%s" % t] = 0.0
                row["ba_lo_rcbf_%s" % t] = row["ba_hi_rcbf_%s" % t] = 0.0
        elif len(sids) >= 2:
            col = lambda dd, k: np.array([dd[s][k] for s in sids], dtype=float)
            for t in ("gm", "wm"):
                row["icc_rcbf_%s" % t] = icc_agreement(col(cur, "rcbf_" + t),
                                                       col(ref, "rcbf_" + t))
                ba = bland_altman(col(cur, "rcbf_" + t), col(ref, "rcbf_" + t))
                row["ba_bias_rcbf_%s" % t] = ba["bias"]
                row["ba_sd_rcbf_%s" % t] = ba["sd_diff"]
                row["ba_lo_rcbf_%s" % t] = ba["loa_low"]
                row["ba_hi_rcbf_%s" % t] = ba["loa_high"]
        else:
            for t in ("gm", "wm"):
                row["icc_rcbf_%s" % t] = row["ba_bias_rcbf_%s" % t] = float("nan")
        summary.append(row)

    d["summary"] = summary
    d.setdefault("params", {})["ref_arm"] = a.ref_arm
    out = a.out or src
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=2)
    print("wrote %s" % out)
    print()
    print("%3s %9s %9s %10s %9s" % ("k", "ICC_GM", "ICC_WM", "bias_GM", "r_vox"))
    for r in summary:
        print("%3d %9.3f %9.3f %+10.4f %9.3f"
              % (r["n_frames"], r["icc_rcbf_gm"], r["icc_rcbf_wm"],
                 r["ba_bias_rcbf_gm"], r["recon_corr"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
