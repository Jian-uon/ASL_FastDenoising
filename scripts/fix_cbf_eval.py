#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild a cbf_eval.json written by an eval_cbf.py older than 460b707.

Three things that file gets wrong, all fixed here from data it already contains:

  * ICC and Bland-Altman at n_frames == ref_frames were hard-coded to 1.0 and 0.0 regardless
    of --ref_arm. Against the averaged acquisition the model at twelve repetitions is still a
    reconstruction, and its disagreement there is the number that separates what reconstruction
    costs from what shortening the scan costs.
  * recon_corr was taken on dM, while everything reported beside it is on rCBF. Recomputed
    from the rcbf_*.nii.gz maps --save_maps wrote, so no inference is repeated.
  * The cohort is every test subject, including one that contributes no slice to the
    slice-level comparison. The sweep decides who that is; this script reads it from there.

Idempotent: a file already carrying params.ref_arm came from a fixed eval_cbf and is left
alone unless --force. Delete this script once the server has re-run on current master.

Usage:
  python scripts/fix_cbf_eval.py --dir D:/data/training/medphys_eval_v2 \
                                 --data_root D:/data/ASL/ASL_denoising_dataset/data
"""
import argparse
import copy
import csv
import json
import os
import sys

import numpy as np
import nibabel as nib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

KS = (2, 4, 6, 8, 10, 12)


def icc21(a, b):
    """ICC(2,1), absolute agreement, single measurement."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    M = np.stack([a, b], 1)
    g = M.mean()
    msr = 2 * ((M.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((M.mean(0) - g) ** 2).sum()
    mse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + g) ** 2).sum() / (n - 1)
    return float((msr - mse) / (msr + mse + 2 * (msc - mse) / n))


def sweep_cohort(sweep_csv, order):
    """Subject names kept by the slice-level comparison, in split order.

    The sweep records a subject by its index within the split, so the names come from `order`
    -- the same split, in the same order -- and a subject absent there contributed no slice.
    """
    if not os.path.isfile(sweep_csv):
        return None
    idx = set()
    with open(sweep_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx.add(int(float(r["subject_id"])))
    return [s for i, s in enumerate(order) if i in idx]


def main() -> int:
    p = argparse.ArgumentParser("rebuild a stale cbf_eval.json")
    p.add_argument("--dir", required=True, help="medphys_eval directory (holds cbf/ and sweep/)")
    p.add_argument("--data_root", required=True, help="dataset root, for the brain masks")
    p.add_argument("--force", action="store_true", help="rebuild even if already fixed")
    a = p.parse_args()

    cbf = os.path.join(a.dir, "cbf")
    path = os.path.join(cbf, "cbf_eval.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("params", {}).get("ref_arm") and not a.force:
        print("already fixed (params.ref_arm present) -- nothing to do")
        return 0

    rows = d["rows"]
    ref_n = int(d.get("ref_frames", 12))
    order = [r["subject"] for r in rows if r["arm"] == "model" and int(float(r["n_frames"])) == min(KS)]
    keep = sweep_cohort(os.path.join(a.dir, "sweep", "comparison_long.csv"), order)
    if keep is None:
        keep = order
        print("no sweep CSV -- keeping all %d subjects" % len(keep))
    else:
        print("cohort from the sweep: %d of %d subjects (dropped %s)"
              % (len(keep), len(order), ", ".join(s for s in order if s not in keep) or "none"))
    ref = {r["subject"]: r for r in rows
           if r["arm"] == "mean" and int(float(r["n_frames"])) == ref_n}

    # voxel-wise agreement, on the maps --save_maps already wrote
    fid = {}
    for s in keep:
        R = nib.load(os.path.join(cbf, s, "rcbf_mean_n%d.nii.gz" % ref_n)).get_fdata(dtype=np.float64)
        B = nib.load(os.path.join(a.data_root, s, "raw", "brain_mask_asl.nii.gz")
                     ).get_fdata(dtype=np.float64) > 0.5
        r = R[B]
        for arm, pat in (("model", "rcbf_n%d.nii.gz"), ("mean", "rcbf_mean_n%d.nii.gz")):
            for k in KS:
                f = os.path.join(cbf, s, pat % k)
                if not os.path.isfile(f):
                    continue
                v = nib.load(f).get_fdata(dtype=np.float64)[B]
                fid[(s, arm, k)] = (float(np.corrcoef(v, r)[0, 1]),
                                    float(np.sqrt(np.mean((v - r) ** 2)) / (r.mean() + 1e-6)))

    out = copy.deepcopy(d)
    out["params"]["ref_arm"] = "mean"
    out["rows"] = [r for r in rows if r["subject"] in keep]
    for row in out["rows"]:
        key = (row["subject"], row["arm"], int(float(row["n_frames"])))
        if key in fid:
            row["recon_corr"], row["recon_nrmse"] = fid[key]

    summary = []
    for row in d["summary"]:
        k = int(float(row["n_frames"]))
        R = [r for r in rows if int(float(r["n_frames"])) == k
             and r["arm"] == "model" and r["subject"] in keep]
        sids = [r["subject"] for r in R]
        new = dict(row)
        new["n_subjects"] = len(R)
        for c in ("cbf_gm", "cbf_wm", "rcbf_gm", "rcbf_wm"):
            new[c] = float(np.mean([r[c] for r in R]))
        new["gm_wm_ratio"] = new["cbf_gm"] / new["cbf_wm"]
        v = [fid[(s, "model", k)] for s in sids if (s, "model", k) in fid]
        if v:
            new["recon_corr"] = float(np.mean([x[0] for x in v]))
            new["recon_nrmse"] = float(np.mean([x[1] for x in v]))
        new["icc_rcbf_gm"] = icc21([r["rcbf_gm"] for r in R], [ref[s]["rcbf_gm"] for s in sids])
        new["icc_rcbf_wm"] = icc21([r["rcbf_wm"] for r in R], [ref[s]["rcbf_wm"] for s in sids])
        for t in ("gm", "wm"):
            dif = np.array([r["rcbf_%s" % t] - ref[s]["rcbf_%s" % t] for r, s in zip(R, sids)])
            new["ba_bias_rcbf_%s" % t] = float(dif.mean())
            new["ba_sd_rcbf_%s" % t] = float(dif.std(ddof=1))
            new["ba_lo_rcbf_%s" % t] = float(dif.mean() - 1.96 * dif.std(ddof=1))
            new["ba_hi_rcbf_%s" % t] = float(dif.mean() + 1.96 * dif.std(ddof=1))
        summary.append(new)
    out["summary"] = summary

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("rewrote %s  (n=%d)" % (path, summary[0]["n_subjects"]))
    print("%-4s %9s %9s %9s" % ("k", "ICC_GM", "ICC_WM", "map r"))
    for r in summary:
        print("%-4d %9.4f %9.4f %9.4f"
              % (int(r["n_frames"]), r["icc_rcbf_gm"], r["icc_rcbf_wm"], r["recon_corr"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
