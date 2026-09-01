#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Assemble the manuscript's relative-CBF figure.

Top: one slice from each of three subjects, reconstructed from an increasing number of
repetitions and ending at the full acquisition. Three subjects rather than one, because a
single example cannot show whether the result is typical. Every panel shares one colour scale,
so panels are comparable to each other rather than each stretched to its own range.

Bottom: the agreement statistics against acquisition length. The Bland-Altman scatters that
used to sit beside them are gone -- Table 2 reports the bias and the limits of agreement as
numbers, and the plots repeated that without adding anything.

Nothing here is chosen by appearance, and every choice is printed when the script runs:

* subjects -- the ones sitting at the requested percentiles of voxelwise agreement with the
              full acquisition at the shortest reconstruction, so the rows span the cohort
              instead of showing three variations on a favourable case
* slice    -- the same fraction through each subject's brain-bearing range, so the three rows
              are at a comparable anatomical level

Rows are labelled by percentile, never by subject identifier: the directory names are patient
names and must not reach the manuscript.

Usage:
  python scripts/make_figure5.py --dir <out>/medphys_eval
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def main() -> int:
    p = argparse.ArgumentParser("assemble the rCBF figure")
    p.add_argument("--dir", required=True)
    p.add_argument("--subject_pcts", type=float, nargs="+", default=[25, 50, 90],
                   help="percentiles of voxelwise agreement at the shortest reconstruction; "
                        "one subject per value, one row each")
    p.add_argument("--slice_frac", type=float, default=0.38,
                   help="where the displayed slice sits in each subject's brain-bearing range")
    p.add_argument("--data_root", default="D:/data/ASL/ASL_denoising_dataset/data",
                   help="dataset root, for brain_mask_asl.nii.gz")
    p.add_argument("--raw_dir", default="raw")
    p.add_argument("--erode", type=int, default=2,
                   help="voxels to erode off the brain mask before display; the outermost "
                        "voxels carry a partial-volume rim whose rCBF saturates the scale")
    p.add_argument("--ref_arm_prefix", default="rcbf_mean_n",
                   help="file prefix of the averaged reference maps, shown as the last "
                        "column. This is the image the current protocol delivers and what "
                        "the agreement statistics are measured against.")
    p.add_argument("--arm", default="rcbf_n",
                   help="map prefix: rcbf_n (proposed), rcbf_mean_n, rcbf_vanilla_n")
    p.add_argument("--vmax", type=float, default=2.0)
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()

    cbf = os.path.join(a.dir, "cbf")
    d = json.load(open(os.path.join(cbf, "cbf_eval.json"), encoding="utf-8"))
    ref_k = d["ref_frames"]
    ref_arm = d.get("params", {}).get("ref_arm", "model")
    ks = [s["n_frames"] for s in d["summary"]]

    by = {}
    for r in d["rows"]:
        if r.get("arm", "model") == "model":
            by.setdefault(r["n_frames"], {})[r["subject"]] = r
    ref = by[ref_k]

    # -- subjects by rule: percentiles of agreement at the shortest reconstruction ------
    k_lo = min(k for k in by if k != ref_k)
    ranked = sorted((r["recon_corr"], sid) for sid, r in by[k_lo].items()
                    if sid in ref and r["recon_corr"] == r["recon_corr"])
    if not ranked:
        raise SystemExit("no agreement values at %d repetitions" % k_lo)

    REF_COL = "ref"          # sorts after every integer repetition count

    def load(sid):
        v = {}
        for k in ks:
            f = os.path.join(cbf, sid, "%s%d.nii.gz" % (a.arm, k))
            if os.path.isfile(f):
                v[k] = np.asarray(nib.load(f).get_fdata(), dtype=np.float64)
        # the averaged reference: the map the current protocol delivers, and what every
        # agreement number in the lower panel is measured against
        fr = os.path.join(cbf, sid, "%s%d.nii.gz" % (a.ref_arm_prefix, ref_k))
        if os.path.isfile(fr):
            v[REF_COL] = np.asarray(nib.load(fr).get_fdata(), dtype=np.float64)
        if not v:
            return None
        mf = os.path.join(a.data_root, sid, a.raw_dir, "brain_mask_asl.nii.gz")
        if os.path.isfile(mf):
            bm = np.asarray(nib.load(mf).get_fdata(), dtype=np.float64) > 0.5
            if a.erode > 0:
                from scipy.ndimage import binary_erosion
                bm = binary_erosion(bm, iterations=a.erode)
            v = {k: np.where(bm, x, 0.0) for k, x in v.items()}
        return v

    rows_fig = []
    for pc in a.subject_pcts:
        i = min(len(ranked) - 1, max(0, int(round((pc / 100.0) * (len(ranked) - 1)))))
        corr, sid = ranked[i]
        vv = load(sid)
        if vv is None:
            print("  [skip] %s: no %s maps" % (sid, a.arm))
            continue
        any3 = vv[ref_k if ref_k in vv else sorted(vv)[0]]
        zs = [z for z in range(any3.shape[2]) if np.count_nonzero(any3[:, :, z]) > 200]
        z = int(round(zs[0] + a.slice_frac * (zs[-1] - zs[0]))) if zs else any3.shape[2] // 2
        rows_fig.append(("%dth percentile\nagreement" % pc, vv, z))
        # printed, not drawn: the identifier is a patient name
        print("  %3dth pct -> %s  (voxelwise r = %.3f at %d rep., slice z=%d)"
              % (pc, sid, corr, k_lo, z))
    if not rows_fig:
        raise SystemExit("no subject rows could be built")

    vols = rows_fig[0][1]
    kk = sorted(k for k in vols if k != REF_COL)
    if REF_COL in vols:
        kk.append(REF_COL)
    nr = len(rows_fig)
    fig = plt.figure(figsize=(13.5, 2.7 + 2.4 * nr))
    top_lo = 0.36 if nr >= 3 else 0.48
    gs_top = fig.add_gridspec(nr, len(kk), left=0.07, right=0.90,
                              top=0.95, bottom=top_lo, wspace=0.05, hspace=0.08)
    gs_bot = fig.add_gridspec(1, 1, left=0.31, right=0.73,
                              top=top_lo - 0.07, bottom=0.06)

    norm = Normalize(vmin=0.0, vmax=a.vmax)
    cm = plt.get_cmap(a.cmap).copy()
    cm.set_bad("black")
    for i, (lab, vv, z) in enumerate(rows_fig):
        for j, k in enumerate(kk):
            ax = fig.add_subplot(gs_top[i, j])
            if k in vv:
                img = np.rot90(vv[k][:, :, z])
                ax.imshow(np.where(img > 0, img, np.nan), cmap=cm, norm=norm,
                          interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            if i == 0:
                ax.set_title("Full acquisition\n(%d rep. averaged)" % ref_k
                             if k == REF_COL else "%d repetitions" % k, fontsize=10)
    cax = fig.add_axes([0.915, top_lo + 0.03, 0.010, 0.92 - top_lo - 0.05])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=a.cmap), cax=cax)
    cb.set_label("rCBF (fraction of the GM+WM mean)", fontsize=9)

    ax = fig.add_subplot(gs_bot[0, 0])
    srt = sorted(d["summary"], key=lambda r: r["n_frames"])
    xs = [r["n_frames"] for r in srt]
    for key, lab, style, mk, col in (("icc_rcbf_gm", "ICC$_{GM}$", "-", "o", "#2e86c1"),
                                     ("icc_rcbf_wm", "ICC$_{WM}$", "-", "s", "#1e8449"),
                                     ("recon_corr", "Voxelwise $r$ of the maps", "--", "^", "#b9770e")):
        ys = [r[key] for r in srt]
        ax.plot(xs, ys, style, color=col)
        # Filled where the value was measured. Only the model-as-reference arm has a point
        # that is 1 by construction rather than by result, and only that one is drawn hollow
        # below; against the averaged reference every length is a measurement, including the
        # last, which is the one that separates the reconstruction cost from the scan-length
        # cost and so must not be left off.
        skip = {ref_k} if ref_arm == "model" else set()
        ax.plot([x for x in xs if x not in skip], [y for x, y in zip(xs, ys) if x not in skip],
                mk, color=col, label=lab, linestyle="none")
        if ref_arm == "model":
            ax.plot([ref_k], [dict(zip(xs, ys))[ref_k]], mk, color=col, mfc="white",
                    mew=1.4, linestyle="none")
    if ref_arm == "model":
        ax.annotate("reference\n(identity)", (ref_k, 1.0), xytext=(-6, -14),
                    textcoords="offset points", ha="right", va="top", fontsize=8,
                    color="#555555")
    ax.set_xlabel("Repetitions entering the reconstruction")
    ax.set_ylabel("Agreement with the full acquisition")
    ax.set_xticks(xs)
    lo = min(min(r[k] for r in d["summary"])
             for k in ("icc_rcbf_gm", "icc_rcbf_wm", "recon_corr"))
    ax.set_ylim(max(0.0, lo - 0.05), 1.005)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Agreement against acquisition length", fontsize=10)

    out = a.out or os.path.join(a.dir, "figures", "Figure4.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=a.dpi)
    plt.close(fig)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
