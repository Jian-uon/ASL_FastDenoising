#!/usr/bin/env python
"""Replace the crude 3-means gm/wm/csf in the converted OSIPI subjects with FSL
FAST partial-volume (PVE) probability maps.

`prep_osipi.py` writes gm/wm/csf as a dependency-free 1-D 3-means on T1 intensity
(hard maps) because the WSL env lacks sklearn — it self-flags them as APPROXIMATE.
This runs FSL FAST on the already-BET `t1_in_asl.nii.gz` (same grid the maps live
on) and overwrites gm/wm/csf_asl with the *probabilistic* PVE maps, masked to
brain. These are only QC-metric tissue masks (scov_gm / cnr); the model computes
its own T1 seg internally, so this does NOT change any reconstruction.

PVE classes are assigned to CSF < GM < WM by ascending mean T1 intensity (robust
to FAST's class ordering). The first run keeps a *_3means.nii.gz backup of the old
map so the crude version is not lost.

Run in WSL (FSL present; ~1-3 min/subject):
  /home/jian/anaconda3/envs/asl-mamba/bin/python scripts/osipi_fast_seg.py \
    --root /mnt/g/data/OSIPI_ASL_Challenge/converted
"""
from __future__ import annotations
import argparse, glob, os, shutil, subprocess, tempfile
import numpy as np
import nibabel as nib

FSLDIR = os.environ.get("FSLDIR") or "/usr/local/fsl"


def fsl_env():
    env = dict(os.environ)
    env["FSLDIR"] = FSLDIR
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["PATH"] = f"{FSLDIR}/share/fsl/bin:{FSLDIR}/bin:" + env.get("PATH", "")
    return env


def run_fast(t1_path, workdir, env):
    """FAST 3-class PVE on a brain-extracted T1w. Returns [pve_0, pve_1, pve_2]."""
    pref = os.path.join(workdir, "fast")
    subprocess.run(["fast", "-t", "1", "-n", "3", "-o", pref, t1_path],
                   check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [f"{pref}_pve_{k}.nii.gz" for k in range(3)]


def process(sdir, env):
    sid = os.path.basename(sdir)
    raw = os.path.join(sdir, "raw")
    t1p = os.path.join(raw, "t1_in_asl.nii.gz")
    bmp = os.path.join(raw, "brain_mask_asl.nii.gz")
    if not os.path.isfile(t1p):
        print(f"  [skip] {sid}: no t1_in_asl"); return
    t1_img = nib.load(t1p)
    t1 = np.nan_to_num(t1_img.get_fdata().astype(np.float32))
    brain = (nib.load(bmp).get_fdata() > 0.5) if os.path.isfile(bmp) else (t1 > 0.05)

    with tempfile.TemporaryDirectory() as wd:
        pves = run_fast(t1p, wd, env)
        maps = [np.nan_to_num(nib.load(p).get_fdata().astype(np.float32)) for p in pves]

    # assign PVE class -> tissue by ascending mean-T1 intensity: CSF (dark) < GM < WM (bright)
    mean_int = [float((m * t1).sum() / max(m.sum(), 1e-6)) for m in maps]
    order = sorted(range(3), key=lambda k: mean_int[k])
    tissue = {"csf": maps[order[0]], "gm": maps[order[1]], "wm": maps[order[2]]}

    for name, m in tissue.items():
        m = np.clip(m * brain, 0.0, 1.0).astype(np.float32)
        outp = os.path.join(raw, f"{name}_asl.nii.gz")
        bak = os.path.join(raw, f"{name}_asl_3means.nii.gz")
        if os.path.isfile(outp) and not os.path.isfile(bak):
            shutil.copy2(outp, bak)                       # keep the crude 3-means once
        nib.save(nib.Nifti1Image(m, t1_img.affine), outp)

    fr = {k: float(v[brain].mean()) if brain.sum() else float("nan") for k, v in tissue.items()}
    print(f"  [ok] {sid}: PVE mean-in-brain  GM={fr['gm']:.3f}  WM={fr['wm']:.3f}  CSF={fr['csf']:.3f}"
          f"  (FAST intensity order csf<gm<wm = {mean_int[order[0]]:.1f}<{mean_int[order[1]]:.1f}<{mean_int[order[2]]:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="converted root with sub-*/raw/")
    ap.add_argument("--subjects", nargs="+", default=None, help="subset, e.g. sub-DRO1 sub-DRO2")
    args = ap.parse_args()

    env = fsl_env()
    if not shutil.which("fast", path=env["PATH"]):
        raise SystemExit(f"[fatal] FSL `fast` not found on PATH (FSLDIR={FSLDIR}). Run in the WSL env with FSL.")
    subs = sorted(glob.glob(os.path.join(args.root, "sub-*")))
    if args.subjects:
        want = set(args.subjects)
        subs = [s for s in subs if os.path.basename(s) in want]
    print(f"[fast-seg] {len(subs)} subjects; FSLDIR={FSLDIR}")
    for s in subs:
        process(s, env)
    print("[done] gm/wm/csf_asl overwritten with FAST PVE (old 3-means kept as *_3means.nii.gz)")


if __name__ == "__main__":
    main()
