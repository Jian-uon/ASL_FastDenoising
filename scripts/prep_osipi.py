#!/usr/bin/env python
"""Convert the OSIPI ASL MRI Challenge (2021, TF6.1) BIDS data into the per-subject
`raw/` layout our loader expects, for EXTERNAL VALIDATION.

Per subject `<out_root>/<sid>/raw/`:
    asldata_diff.nii.gz  [H,W,Z,T]   ASL difference frames (POSITIVE perfusion), BRAIN-ONLY
    t1_in_asl.nii.gz     [H,W,Z]      T1w resampled onto the ASL grid, BRAIN-ONLY (skull-stripped)
    gm_asl / wm_asl / csf_asl.nii.gz [H,W,Z] in [0,1]  tissue partial-volume maps (brain)
    brain_mask_asl.nii.gz [H,W,Z]     BET brain mask on the ASL grid

Pipeline decisions (see the discussion that produced this file):
  * ΔM = control - label  (PHYSICAL positive perfusion; matches training data whose
    brain-mean ΔM is +13, GM>WM). NOT label-control.
  * SKULL-STRIP with FSL BET on the native T1w, resample the mask to the ASL grid,
    and MASK both t1_in_asl and asldata_diff to brain (zero outside). The OSIPI T1w
    is a full head (~4500 mL >thr) while our training T1 (t1_in_asl) is brain-only —
    feeding skull/scalp caused a bright cortical-rim artifact in the recon.
  * Resample everything to `--voxel` mm ISO (default 2.0): our loader pad-or-crops
    (not resamples), so the brain must occupy a training-like pixel footprint
    (training native is 2 mm; OSIPI is 4 mm).
  * T1w & ASL share world coordinates (synthetic DRO) -> resample_from_to is a clean
    affine resample, no registration.
  * gm/wm/csf APPROXIMATED by a dependency-free 1-D 3-means on T1 intensity WITHIN
    the BET brain (CSF<GM<WM in T1w). Replace with ASLDRO GT / FSL FAST for final
    reported metrics.
  * NaN sanitized (ASLDRO writes NaN in air for some DROs).

Run in WSL (has FSL bet; note sklearn is NOT needed):
    /home/jian/anaconda3/envs/asl-mamba/bin/python scripts/prep_osipi.py \
        --osipi_root /mnt/g/data/OSIPI_ASL_Challenge --sets synthetic --voxel 2.0
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from scipy import ndimage


def read_aslcontext(tsv_path):
    lines = [l.strip() for l in open(tsv_path).read().splitlines() if l.strip()]
    body = lines[1:] if lines and lines[0].lower().startswith("volume") else lines
    ctrl = [i for i, v in enumerate(body) if v == "control"]
    lab = [i for i, v in enumerate(body) if v == "label"]
    return ctrl, lab


def compute_deltam(asl_img, ctrl, lab):
    """ΔM = control - label, per pair -> [X,Y,Z,Npair]. Positive perfusion."""
    data = np.nan_to_num(asl_img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = min(len(ctrl), len(lab))
    if len(ctrl) != len(lab):
        print(f"    [warn] control({len(ctrl)}) != label({len(lab)}); using first {n} pairs")
    return (data[..., np.asarray(ctrl[:n])] - data[..., np.asarray(lab[:n])]).astype(np.float32)


def resample_4d_to_target(dm, src_affine, target_img, order=1):
    tgt_shape, tgt_affine = target_img.shape[:3], target_img.affine
    out = np.zeros(tgt_shape + (dm.shape[-1],), dtype=np.float32)
    for t in range(dm.shape[-1]):
        r = resample_from_to(nib.Nifti1Image(dm[..., t], src_affine),
                             (tgt_shape, tgt_affine), order=order, cval=0.0)
        out[..., t] = np.asarray(r.dataobj, dtype=np.float32)
    return out


def bet_mask(t1_path, frac=0.5):
    """FSL BET on the native T1w -> boolean brain mask (nib image, native grid).
    Returns None if bet is unavailable."""
    bet = shutil.which("bet")
    if bet is None:
        return None
    env = os.environ.copy()
    env.setdefault("FSLDIR", "/usr/local/fsl")
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["PATH"] = os.path.join(env["FSLDIR"], "share", "fsl", "bin") + os.pathsep + env.get("PATH", "")
    with tempfile.TemporaryDirectory() as td:
        pref = os.path.join(td, "bet")
        subprocess.run([bet, t1_path, pref, "-m", "-n", "-R", "-f", str(frac)],
                       env=env, check=True, capture_output=True)
        mp = pref + "_mask.nii.gz"
        img = nib.load(mp)
        return nib.Nifti1Image((np.asarray(img.dataobj) > 0.5).astype(np.uint8), img.affine)


def fallback_headmask(t1, frac=0.05):
    m = ndimage.binary_fill_holes(t1 > frac * float(t1.max()))
    lbl, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        m = lbl == (int(np.argmax(sizes)) + 1)
    return m


def kmeans1d(x, k=3, iters=100):
    """Dependency-free 1-D k-means (Lloyd). x: 1-D array. Returns (labels, centers)."""
    x = x.astype(np.float64).reshape(-1)
    c = np.percentile(x, np.linspace(100 / (2 * k), 100 - 100 / (2 * k), k))
    lab = np.zeros_like(x, dtype=np.int64)
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None, :]).argmin(1)
        newc = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(newc, c):
            break
        c = newc
    return lab, c


def tissue_maps(t1, brain):
    """3-means on T1 in brain -> (gm, wm, csf) hard [0,1] maps. T1w: CSF<GM<WM."""
    idx = np.where(brain)
    lab, c = kmeans1d(t1[brain], k=3)
    order = np.argsort(c)                                    # low->high intensity
    remap = {int(order[0]): "csf", int(order[1]): "gm", int(order[2]): "wm"}
    out = {t: np.zeros(t1.shape, dtype=np.float32) for t in ("gm", "wm", "csf")}
    for cluster, tissue in remap.items():
        sel = lab == cluster
        out[tissue][idx[0][sel], idx[1][sel], idx[2][sel]] = 1.0
    return out["gm"], out["wm"], out["csf"]


def process_subject(sid, perf_dir, anat_dir, out_dir, voxel, bet_frac, order=1, smooth_sigma=0.0):
    asl_p = os.path.join(perf_dir, f"{sid}_asl.nii.gz")
    ctx_p = os.path.join(perf_dir, f"{sid}_aslcontext.tsv")
    t1_p = os.path.join(anat_dir, f"{sid}_T1w.nii.gz")
    for p in (asl_p, ctx_p, t1_p):
        if not os.path.isfile(p):
            print(f"  [skip] {sid}: missing {os.path.basename(p)}")
            return False

    asl_img = nib.load(asl_p)
    ctrl, lab = read_aslcontext(ctx_p)
    dm = compute_deltam(asl_img, ctrl, lab)

    mean_img = nib.Nifti1Image(dm.mean(-1), asl_img.affine)
    target = resample_to_output(mean_img, voxel_sizes=voxel, order=order)
    tgt_shape, tgt_affine = target.shape[:3], target.affine

    dm_rs = resample_4d_to_target(dm, asl_img.affine, target, order=order)
    if smooth_sigma > 0:   # suppress the 4mm->2mm upsampling interpolation grid (spatial only, per frame)
        dm_rs = ndimage.gaussian_filter(dm_rs, sigma=(smooth_sigma, smooth_sigma, smooth_sigma, 0.0))
    t1_rs = np.nan_to_num(np.asarray(
        resample_from_to(nib.load(t1_p), (tgt_shape, tgt_affine), order=order, cval=0.0).dataobj,
        dtype=np.float32))
    t1_rs[t1_rs < 0] = 0.0

    # --- skull strip: BET native T1 -> resample mask to ASL grid ---
    m_native = bet_mask(t1_p, frac=bet_frac)
    if m_native is not None:
        brain = np.asarray(resample_from_to(m_native, (tgt_shape, tgt_affine), order=0, cval=0).dataobj) > 0.5
        brain = ndimage.binary_fill_holes(brain)
        how = "BET"
    else:
        brain = fallback_headmask(t1_rs)
        how = "fallback-headmask(NO BET!)"

    # mask inputs to brain (match training's brain-only distribution)
    t1_rs = t1_rs * brain
    dm_rs = np.nan_to_num(dm_rs, nan=0.0) * brain[..., None]
    gm, wm, csf = tissue_maps(t1_rs, brain)

    raw = os.path.join(out_dir, sid, "raw")
    os.makedirs(raw, exist_ok=True)
    for name, arr in [("asldata_diff", dm_rs), ("t1_in_asl", t1_rs),
                      ("gm_asl", gm), ("wm_asl", wm), ("csf_asl", csf),
                      ("brain_mask_asl", brain.astype(np.float32))]:
        nib.save(nib.Nifti1Image(arr, tgt_affine), os.path.join(raw, f"{name}.nii.gz"))

    vox_ml = float(np.prod(np.abs(np.diag(tgt_affine)[:3]))) / 1000.0
    bmean = float(dm_rs.mean(-1)[brain].mean()) if brain.sum() else float("nan")
    print(f"  [ok] {sid}: diff{dm_rs.shape} t1{t1_rs.shape} pairs={dm.shape[-1]} "
          f"brain={int(brain.sum())}vox/{brain.sum()*vox_ml:.0f}mL ({how})  "
          f"brainΔM(mean)={bmean:+.4f}  GM/WM/CSF={int(gm.sum())}/{int(wm.sum())}/{int(csf.sum())}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osipi_root", default=r"G:/data/OSIPI_ASL_Challenge")
    ap.add_argument("--out_root", default=None, help="default <osipi_root>/converted")
    ap.add_argument("--sets", default="synthetic", help="comma list: synthetic,population")
    ap.add_argument("--voxel", type=float, default=2.0, help="iso output voxel (mm); match training 2.0")
    ap.add_argument("--bet_frac", type=float, default=0.5, help="FSL BET -f fractional intensity threshold")
    ap.add_argument("--smooth_sigma", type=float, default=0.0,
                    help="gaussian σ (voxels) to smooth the 4mm->2mm upsampling grid out of ΔM "
                         "frames; 0=off. ~0.5-0.7 removes most of the interpolation grid.")
    args = ap.parse_args()
    out_root = args.out_root or os.path.join(args.osipi_root, "converted")
    if shutil.which("bet") is None:
        print("[WARN] FSL 'bet' NOT on PATH -> falling back to a head mask (INCLUDES SKULL). "
              "Run this in the WSL asl-mamba env where FSL is installed.")

    set_map = {
        "synthetic": ("synthetic", ["sub-DRO%d" % i for i in range(1, 10)]),
        "population": ("Population_based", ["sub-PopulationAverage"]),
    }
    n_ok = 0
    for s in [x.strip() for x in args.sets.split(",") if x.strip()]:
        if s not in set_map:
            print(f"[warn] unknown set '{s}', skip"); continue
        sub_root, sids = set_map[s]
        base = os.path.join(args.osipi_root, sub_root, "rawdata")
        print(f"=== set '{s}'  ({base}) -> {out_root} ===")
        for sid in sids:
            perf, anat = os.path.join(base, sid, "perf"), os.path.join(base, sid, "anat")
            if not os.path.isdir(perf):
                print(f"  [skip] {sid}: no perf dir"); continue
            n_ok += process_subject(sid, perf, anat, out_root, args.voxel, args.bet_frac,
                                    smooth_sigma=args.smooth_sigma)
    print(f"\nDONE: {n_ok} subjects -> {out_root}")
    print("NOTE: gm/wm/csf are APPROXIMATE (3-means on T1 in the BET brain). "
          "Replace with ASLDRO GT / FSL FAST for final reported metrics.")


if __name__ == "__main__":
    main()
