#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-render the rCBF montages from maps eval_cbf.py already saved. No GPU, no inference.

eval_cbf.py writes absolute `cbf_n{k}.nii.gz` per subject, so the figure can be rebuilt
without repeating the reconstruction -- which matters because the reported agreement numbers
were already correct and only the rendering needed fixing.

dM is recovered as cbf x m0 up to a constant: the CBF model is linear in dM at fixed M0, and
that column is windowed on its own percentiles, so the constant cancels.

Usage:
  python scripts/render_rcbf_qc.py --cbf_dir <out>/cbf --data_root <dataset root>
"""
import argparse
import glob
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.eval_cbf import _rcbf_png


def load(p):
    return np.asarray(nib.load(p).dataobj, dtype=np.float64)


def main() -> int:
    p = argparse.ArgumentParser("re-render rCBF montages from saved maps")
    p.add_argument("--cbf_dir", required=True, help="eval_cbf.py --out_dir (holds sub-*/cbf_n*.nii.gz)")
    p.add_argument("--data_root", required=True, help="dataset root with sub-*/raw/")
    p.add_argument("--raw_dir", default="raw")
    p.add_argument("--out_dir", default=None, help="default <cbf_dir>/qc")
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--vmax", type=float, default=2.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--erode", type=int, default=1)
    p.add_argument("--show_ref_cbf", action="store_true",
                   help="add the dataset's own cbf.nii.gz column; see eval_cbf.py for why it "
                        "is off by default.")
    p.add_argument("--subjects", nargs="*", default=None)
    a = p.parse_args()
    out_dir = a.out_dir or os.path.join(a.cbf_dir, "qc")

    subs = sorted(d for d in glob.glob(os.path.join(a.cbf_dir, "sub-*")) if os.path.isdir(d))
    if a.subjects:
        want = set(a.subjects)
        subs = [d for d in subs if os.path.basename(d) in want]
    if not subs:
        raise SystemExit("no sub-* directories under %s" % a.cbf_dir)
    print("[qc] %d subjects -> %s" % (len(subs), out_dir))

    done = 0
    for d in subs:
        sid = os.path.basename(d)
        raw = os.path.join(a.data_root, sid, a.raw_dir)
        try:
            gm, wm = load("%s/gm_asl.nii.gz" % raw), load("%s/wm_asl.nii.gz" % raw)
            brain = load("%s/brain_mask_asl.nii.gz" % raw)
            t1, m0 = load("%s/t1_in_asl.nii.gz" % raw), load("%s/m0.nii.gz" % raw)
        except Exception as e:
            print("  skip %s (%s)" % (sid, e))
            continue
        tis = (gm > 0.5) | (wm > 0.5)
        if not tis.any():
            print("  skip %s (no GM/WM voxels)" % sid)
            continue

        ns, rcbf, dm = [], {}, {}
        for f in sorted(glob.glob(os.path.join(d, "cbf_n*.nii.gz"))):
            n = int(os.path.basename(f)[len("cbf_n"):-len(".nii.gz")])
            cbf = load(f)
            ns.append(n)
            rcbf[n] = cbf / (cbf[tis].mean() + 1e-6) * (brain > 0.5)
            dm[n] = cbf * m0                      # linear in dM at fixed M0; scale cancels
        if not ns:
            print("  skip %s (no cbf_n*.nii.gz)" % sid)
            continue
        ns = sorted(ns)

        refc = None
        rp = "%s/cbf.nii.gz" % raw
        if a.show_ref_cbf and os.path.isfile(rp):
            r = load(rp)
            rm = float(r[tis].mean())
            refc = r / rm if rm > 0 else None

        _rcbf_png(sid, os.path.join(out_dir, "%s.png" % sid), t1, dm, rcbf, brain,
                  ns, max(ns), refc, cmap=a.cmap, vmax=a.vmax, dpi=a.dpi, erode=a.erode)
        done += 1
    print("[qc] re-rendered %d montages" % done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
