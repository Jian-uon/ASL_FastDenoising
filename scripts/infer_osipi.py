#!/usr/bin/env python
"""Run a trained ASLT1Denoiser on the converted OSIPI DRO subjects (external test).

Builds the EXACT architecture via the training arg-parser + Runner (so a strict
EMA load matches all keys), but STUBS the data loader so no training dataset / T1
prior file is needed — the checkpoint already contains the full model (incl. the
frozen T1 branch). Runs slice-by-slice inference, saves the denoised PWI per
subject (+ a direct-mean baseline) and an optional QC montage PNG.

MUST run in the WSL `asl-mamba` env (CUDA + mamba_ssm); Windows is CPU-only w/o
mamba_ssm and falls back to a 10-20x slower pure-PyTorch scan.

Example (WSL) — 2D texsetb ckpt:
  /home/jian/anaconda3/envs/asl-mamba/bin/python scripts/infer_osipi.py \
    --checkpoint /mnt/g/training/ASL_dmvae/logs/cig_vmamba_vss_texsetb_seed42/checkpoints/step000200.pth \
    --config env/local/configs/wsl_asl_2d_home_v37.yml \
    --osipi_root /mnt/g/data/OSIPI_ASL_Challenge/converted \
    --out_dir /mnt/g/data/OSIPI_ASL_Challenge/pred_texsetb_step200 \
    --runner_args "--use_vmamba_encoder --vmamba_n_directions 4 --vmamba_convffn --vmamba_overlap_merge --vmamba_lrda_inscan --base_ch 32 --depth 4 --t1_task recon" \
    --relu_input --frame_ratios 0.2 0.4 0.6 0.8 1.0 --qc_png

Example (WSL) — 2.5D RKMR + T1-seg main model (5-slice window):
  /home/jian/anaconda3/envs/asl-mamba/bin/python scripts/infer_osipi.py \
    --checkpoint /mnt/g/training/ASL_dmvae/logs/cig_vss_rkmr_25d_seg_seed42/best_cnr.pth \
    --config env/local/configs/wsl_asl_2d_home_v37.yml \
    --osipi_root /mnt/g/data/OSIPI_ASL_Challenge/converted \
    --out_dir /mnt/g/data/OSIPI_ASL_Challenge/pred_rkmr25dseg \
    --runner_args "--use_vmamba_encoder --vmamba_n_directions 2 --vmamba_convffn --vmamba_overlap_merge --vmamba_lrda_inscan --akmr_repro_key --base_ch 32 --depth 4 --t1_task seg" \
    --slice_context 2 --relu_input --frame_ratios 0.2 0.4 0.6 0.8 1.0 --qc_png
"""
from __future__ import annotations
import argparse, glob, json, os, shlex, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd,
    ResizeWithPadOrCropd, ScaleIntensityRangePercentilesd, Lambdad,
)
from monai.data import NibabelReader


def build_model(runner_args: str, config: str, device, slice_context: int = 0):
    """Build the model through the training Runner with the data loader stubbed out."""
    import runners.asl_t1_guided_runner_dmvae_n2n as R
    R.get_asl_2d_loaders = lambda *a, **k: {"train": [], "val": [], "test": []}   # stub
    argv = shlex.split(runner_args)
    for flag, val in [("--config", config), ("--name", "osipi_infer"), ("--max_steps", "1")]:
        if flag not in argv:
            argv += [flag, val]
    argv += ["--slice_context", str(int(slice_context))]   # authoritative (argparse last-wins) so model K matches the inference window
    orig = sys.argv
    try:
        sys.argv = ["infer_osipi.py"] + argv
        args = R.parse_args()
    finally:
        sys.argv = orig
    runner = R.Runner(args)
    model = runner._unwrap().to(device).eval()
    return model, runner.cfg


def load_ema(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw = model.state_dict()
    src = state["ema"] if "ema" in state else state.get("model", state)
    ok = {k: v.to(raw[k].device) for k, v in src.items() if k in raw and v.shape == raw[k].shape}
    model.load_state_dict(ok, strict=False)
    tag = "EMA" if "ema" in state else "model"
    print(f"[load] {len(ok)}/{len(raw)} {tag} weights from {os.path.basename(ckpt_path)}"
          + ("" if len(ok) == len(raw) else "  <-- ARCH MISMATCH, check --runner_args!"))
    return len(ok) == len(raw)


def preproc(asl_p, t1_p, tp, raw_dir=None, relu_input=False):
    extra = {}
    for k, fn in [("brain", "brain_mask_asl.nii.gz"), ("gm", "gm_asl.nii.gz"), ("wm", "wm_asl.nii.gz")]:
        p = os.path.join(raw_dir, fn) if raw_dir else None
        if p and os.path.isfile(p):
            extra[k] = p
    other = ["t1"] + list(extra.keys())
    keys = ["asl_diff"] + other
    tfs = [
        LoadImaged(keys=keys, reader=NibabelReader(), image_only=False),
        EnsureChannelFirstd(keys=["asl_diff"], channel_dim=-1),   # [T,H,W,Z]
        EnsureChannelFirstd(keys=other),
        EnsureTyped(keys=keys, track_meta=True),
        Orientationd(keys=keys, axcodes="LPS"),
        ResizeWithPadOrCropd(keys=["asl_diff"], spatial_size=(tp.asl_hw, tp.asl_hw, tp.asl_z)),
        ResizeWithPadOrCropd(keys=other, spatial_size=(tp.t1_hw, tp.t1_hw, tp.t1_z)),
    ]
    if relu_input:      # match training's NON-NEGATIVE ΔM frames -> percentile p1 back to 0
        tfs.append(Lambdad(keys=["asl_diff"], func=lambda x: x.clamp_min(0.0)))
    tfs += [
        # ASL diff left RAW here — normalised from the SELECTED input frames in
        # infer_volume (matches training's set-A affine; the old all-frames
        # percentile+clip mismatched the retrained model). T1 percentile only.
        ScaleIntensityRangePercentilesd(keys=["t1"], lower=1, upper=99, b_min=0., b_max=1., clip=True),
    ]
    s = Compose(tfs)({"asl_diff": asl_p, "t1": t1_p, **extra})
    masks = {k: s[k][0].cpu().numpy() for k in extra}    # [H,W,Z] float on the 128 grid
    # Real affine of the post-transform (LPS, pad/cropped) grid — the MetaTensor tracks
    # it through Orientationd + ResizeWithPadOrCropd. Fixes the old np.eye(4) geometry bug
    # so saved ΔM/CBF volumes carry valid world coordinates (needed for territory ROI /
    # test-retest registration downstream).
    aff = np.asarray(s["asl_diff"].affine.detach().cpu().numpy(), dtype=np.float64)
    return s["asl_diff"], s["t1"], masks, aff            # NOTE: recon is NOT clamped downstream


@torch.no_grad()
def infer_volume(model, asl_vol, t1_vol, n_frames, device, seed=42, baseline=False, slice_context=0):
    """2D (slice_context=0) or 2.5D (>0) inference. For 2.5D, each center slice z is
    fed a K=2*ctx+1 edge-clamped z-window as the ASL channel dim (Design A: T1 stays
    2D/center); the model reconstructs the CENTER slice. Matches ASLTwoSetDataset2DFlat."""
    T, H, W, Z = asl_vol.shape
    n = min(n_frames, T) if n_frames else T
    idx = np.random.default_rng(seed).choice(T, size=n, replace=False).tolist()
    frames = asl_vol[idx]
    # Normalise from the selected input frames only (matches training's set-A affine).
    from dataio.data_classes import _affine_from_frames
    _lo, _sc = _affine_from_frames(frames)
    frames = (frames - _lo) / _sc
    ctx = int(slice_context)
    pwi = np.zeros((H, W, Z), dtype=np.float32)
    lengths = torch.tensor([n], dtype=torch.long, device=device)
    for z in range(Z):
        if baseline:                                    # PlainUNet2D: input = mean(frames), no T1 (2D only)
            xin = frames[:, :, :, z].mean(0)[None, None].to(device, torch.float32)   # [1,1,H,W]
            pwi[:, :, z] = model(xin)[0, 0].cpu().numpy()
        else:
            if ctx > 0:                                 # 2.5D: K-slice edge-clamped window -> channel dim
                zc = [min(max(z + dz, 0), Z - 1) for dz in range(-ctx, ctx + 1)]
                sl = frames[:, :, :, zc].permute(0, 3, 1, 2).unsqueeze(0)             # [1,n,K,H,W]
            else:                                       # 2D: single center slice (K=1)
                sl = frames[:, :, :, z].unsqueeze(1).unsqueeze(0)                     # [1,n,1,H,W]
            sl = sl.to(device, torch.float32)
            t1 = t1_vol[:, :, :, z].unsqueeze(0).to(device, torch.float32)            # [1,1,H,W] center (2D T1)
            out = model.infer_from_subset(sl, t1, lengths=lengths)
            pwi[:, :, z] = out["asl_recon"][0, 0].cpu().numpy()                       # center-slice recon
    # Also return the set-A robust affine so the normalisation can be inverted downstream
    # (raw-intensity ΔM = pwi_norm * scale + lo) for post-hoc CBF quantification.
    return pwi, n, float(_lo), float(_sc)


def qc_png(sid, direct, denoised_by_nf, t1, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [qc] matplotlib unavailable ({e}); skip PNG"); return
    Z = t1.shape[2]
    zs = [int(Z*0.4), int(Z*0.5), int(Z*0.6)]
    labels = list(denoised_by_nf.keys())    # insertion order (ratios/counts as given)
    ncol = 2 + len(labels)                   # T1 | direct-mean | denoised(...)
    fig, ax = plt.subplots(len(zs), ncol, figsize=(2.4*ncol, 2.4*len(zs)))
    ax = np.atleast_2d(ax)
    for r, z in enumerate(zs):
        panels = [("T1", t1[:, :, z]), ("direct-mean", direct[:, :, z])]
        panels += [(f"denoised {lab}", denoised_by_nf[lab][:, :, z]) for lab in labels]
        for c, (title, img) in enumerate(panels):
            a = ax[r, c]
            vals = img[img != 0]                          # robust window over the brain (incl. negatives)
            if vals.size:
                vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
            else:
                vmin, vmax = 0, 1
            a.imshow(np.rot90(img), cmap="gray", vmin=vmin, vmax=max(vmax, vmin + 1e-6))
            a.set_title(title if r == 0 else "", fontsize=8); a.axis("off")
    fig.suptitle(sid, fontsize=10); fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  [qc] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--osipi_root", required=True, help="converted root with sub-*/raw/")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--runner_args", default="", help="arch flags matching the ckpt (CIG-VSS); ignored with --baseline")
    ap.add_argument("--baseline", action="store_true",
                    help="ckpt is the PlainUNet2D vanilla-N2N baseline (input=mean(frames), T1 never used)")
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--relu_input", action="store_true",
                    help="clamp input ΔM frames to >=0 before percentile norm, to match the "
                         "training data's non-negative frames (fixes the p1<0 normalization shift)")
    ap.add_argument("--slice_context", type=int, default=0,
                    help="2.5D context: 0 = 2D (single slice), 2 = 5-slice z-window (z+-2) as the "
                         "ASL channel dim. MUST match the trained ckpt's --slice_context. Ignored with --baseline.")
    ap.add_argument("--n_frames", type=int, nargs="+", default=[4, 6, 30],
                    help="absolute frame counts (ignored if --frame_ratios is given)")
    ap.add_argument("--frame_ratios", type=float, nargs="+", default=None,
                    help="fractions of the available frames, e.g. 0.2 0.4 0.6 0.8 1.0. "
                         "Per subject uses round(ratio*T) frames. Overrides --n_frames.")
    ap.add_argument("--subjects", nargs="+", default=None, help="subset, e.g. sub-DRO1 sub-DRO2")
    ap.add_argument("--qc_png", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  cuda={torch.cuda.is_available()}")
    if args.baseline:
        from config.conf_data import Config
        from models.plain_unet import PlainUNet2D
        cfg = Config(args.config)
        model = PlainUNet2D(hw=int(cfg.asl_denoiser_train_params.asl_hw), in_ch=1, out_ch=1,
                            base_ch=args.base_ch, depth=args.depth, skip_dropout=0.3).to(device).eval()
        print("[model] PlainUNet2D vanilla-N2N baseline (input=mean, no T1)")
    else:
        model, cfg = build_model(args.runner_args, args.config, device, slice_context=args.slice_context)
        if args.slice_context:
            print(f"[model] 2.5D inference: slice_context={args.slice_context} (K={2*args.slice_context+1}-slice ASL window)")
    load_ema(model, args.checkpoint, device)
    tp = cfg.asl_denoiser_train_params

    subs = sorted(glob.glob(os.path.join(args.osipi_root, "sub-*")))
    if args.subjects:
        want = set(args.subjects)
        subs = [s for s in subs if os.path.basename(s) in want]
    print(f"[data] {len(subs)} subjects; n_frames={args.n_frames}")

    for sdir in subs:
        sid = os.path.basename(sdir)
        raw = os.path.join(sdir, "raw")
        asl_p = os.path.join(raw, "asldata_diff.nii.gz")
        t1_p = os.path.join(raw, "t1_in_asl.nii.gz")
        if not os.path.isfile(asl_p):
            print(f"  [skip] {sid}: no asldata_diff"); continue
        asl_vol, t1_vol, masks, affine = preproc(asl_p, t1_p, tp, raw_dir=raw, relu_input=args.relu_input)
        t1_np = t1_vol[0].cpu().numpy()
        brain = (masks["brain"] > 0.5) if "brain" in masks else (t1_np > 0.05)
        gmm = masks.get("gm"); wmm = masks.get("wm")
        gmb = (gmm > 0.5) if gmm is not None else None
        wmb = (wmm > 0.5) if wmm is not None else None
        direct = asl_vol.float().mean(0).cpu().numpy() * brain   # baseline, brain-only, RAW (no clamp)
        odir = os.path.join(args.out_dir, sid); os.makedirs(odir, exist_ok=True)
        T = int(asl_vol.shape[0])
        if args.frame_ratios:                                   # % of available frames (per subject)
            plan = [(f"{int(round(r*100))}pct", max(1, int(round(r * T)))) for r in args.frame_ratios]
        else:
            plan = [(f"nf{min(nf, T)}", min(nf, T)) for nf in args.n_frames]
        den = {}
        norm_affine = {}
        for label, nf in plan:
            pwi, nused, lo_, sc_ = infer_volume(model, asl_vol, t1_vol, nf, device, args.seed,
                                                baseline=args.baseline, slice_context=args.slice_context)
            pwi = pwi * brain                                   # brain-only, NORMALISED values (NOT clamped)
            den[label] = pwi
            nib.save(nib.Nifti1Image(pwi, affine), os.path.join(odir, f"pwi_{label}.nii.gz"))
            # De-normalise to raw-intensity ΔM (control-label, positive perfusion) for
            # post-hoc CBF: the set-A affine has a subtractive offset, so invert linearly.
            deltam_raw = (pwi * sc_ + lo_) * brain
            nib.save(nib.Nifti1Image(deltam_raw.astype(np.float32), affine),
                     os.path.join(odir, f"deltam_raw_{label}.nii.gz"))
            norm_affine[label] = {"lo": lo_, "scale": sc_, "n_used": int(nused)}
            bm = float(pwi[brain].mean()) if brain.sum() else float("nan")
            gm_m = float(pwi[gmb].mean()) if gmb is not None and gmb.sum() else float("nan")
            wm_m = float(pwi[wmb].mean()) if wmb is not None and wmb.sum() else float("nan")
            neg = float((pwi[brain] < 0).mean()) if brain.sum() else float("nan")
            print(f"  [ok] {sid} {label}(={nused}/{T}fr): brainMean={bm:+.4f} GM={gm_m:+.4f} WM={wm_m:+.4f} "
                  f"GM-WM={gm_m-wm_m:+.4f} frac<0={neg:.2%} range[{pwi[brain].min():+.3f},{pwi[brain].max():+.3f}]")
        nib.save(nib.Nifti1Image(direct, affine), os.path.join(odir, "direct_mean.nii.gz"))
        with open(os.path.join(odir, "norm_affine.json"), "w") as _f:
            json.dump(norm_affine, _f, indent=2)                # persist (lo,scale) per n_frames for reproducible CBF
        if args.qc_png:
            qc_png(sid, direct, den, t1_np, os.path.join(args.out_dir, "qc", f"{sid}.png"))
    print("[done]")


if __name__ == "__main__":
    main()
