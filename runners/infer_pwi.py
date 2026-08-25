# -*- coding: utf-8 -*-
"""
infer_pwi.py — Standalone PWI reconstruction inference script.

Loads a trained ASLT1Denoiser checkpoint and reconstructs a high-quality
PWI (ΔM, perfusion-weighted image) for one subject from its ASL difference
frames and co-registered T1w image.

Output: 3D NIfTI in the same voxel space as the input ASL data.

Usage
-----
# Single subject, use all available frames:
python runners/infer_pwi.py \
    --checkpoint exp/logs/run_full/checkpoints/best.pth \
    --config    env/local/configs/win_asl_2d_home.yml \
    --asl_diff  /data/sub-Y001/raw/asldata_diff.nii.gz \
    --t1        /data/sub-Y001/raw/t1_in_asl.nii.gz \
    --out       /data/sub-Y001/pred/pwi_recon.nii.gz

# Frame-count ablation (produces one output per count):
python runners/infer_pwi.py \
    --checkpoint exp/logs/run_full/checkpoints/best.pth \
    --config    env/local/configs/win_asl_2d_home.yml \
    --asl_diff  /data/sub-Y001/raw/asldata_diff.nii.gz \
    --t1        /data/sub-Y001/raw/t1_in_asl.nii.gz \
    --out       /data/sub-Y001/pred/pwi_recon.nii.gz \
    --n_frames  2 4 6 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, ResizeWithPadOrCropd, ScaleIntensityRangePercentilesd,
)

from config.conf_data import Config
from models.asl_t1_model import ASLT1Denoiser


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("PWI inference (single subject)")
    p.add_argument("--checkpoint", required=True, help="Path to best.pth or latest.pth")
    p.add_argument("--config",     required=True, help="YAML config path")
    p.add_argument("--asl_diff",   required=True, help="asldata_diff.nii.gz [H,W,Z,T]")
    p.add_argument("--t1",         required=True, help="t1_in_asl.nii.gz [H,W,Z]")
    p.add_argument("--out",        required=True, help="Output NIfTI path for reconstructed PWI")
    p.add_argument("--n_frames",   type=int, nargs="+", default=[None],
                   help="Number of frames to use. None = all. Multiple values = frame sweep.")
    p.add_argument("--seed",       type=int, default=42)
    # Model architecture. NEW checkpoints embed full 'arch' metadata and are
    # rebuilt automatically (no flags needed). The flags below are ONLY a fallback
    # for OLD checkpoints without that metadata — pass the same ones used to train.
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth",   type=int, default=4)
    p.add_argument("--use_vmamba_encoder", action="store_true")
    p.add_argument("--vmamba_n_directions", type=int, default=2)
    p.add_argument("--vmamba_convffn", action="store_true")
    p.add_argument("--vmamba_overlap_merge", action="store_true")
    p.add_argument("--vmamba_lrda_inscan", action="store_true")
    p.add_argument("--lrda_coupling_gate", action="store_true")
    p.add_argument("--akmr_repro_key", action="store_true")
    p.add_argument("--slice_context", type=int, default=0)
    p.add_argument("--t1_task", type=str, default="seg", choices=["seg", "recon"])
    p.add_argument("--device",  type=str, default="")
    # Epistemic uncertainty (MC dropout + optional SWA-ensemble)
    p.add_argument("--mc_n", type=int, default=1,
                   help="Number of stochastic forward passes per slice. >1 enables MC dropout; "
                        "outputs both mean PWI and per-pixel std as a separate _confmap.nii.gz.")
    p.add_argument("--mc_ckpts", type=str, nargs="+", default=None,
                   help="Optional list of additional checkpoints (e.g. swa.pth) to deep-ensemble "
                        "with --checkpoint. Each ckpt contributes mc_n / N_ckpts forwards.")
    # Innovation U — Conformal Hallucination Detection (inference side).
    p.add_argument("--conformal_threshold", type=str, default=None,
                   help="Path to a calibration JSON produced by scripts/conformal_calibrate.py. "
                        "When set together with --mc_n>1, additionally writes a *_riskmap.nii.gz "
                        "marking voxels where MC dropout std exceeds the (1-α)-calibrated threshold.")
    # Innovation T — PID deployment mode.
    p.add_argument("--pid_zero_t1", action="store_true",
                   help="Replace T1 input with zeros at inference (use when the model was trained "
                        "with --use_pid). Hallucination is structurally impossible since the model "
                        "receives no T1 anatomy.")
    # Brain masking of the OUTPUT. Training supervises the model only inside the
    # brain (loss masked to t1 > thr); outside the brain the output is undefined,
    # so we zero it out of the saved PWI by default (matches the eval regime,
    # which is always brain-masked). Set --no_brain_mask to keep the raw output.
    p.add_argument("--no_brain_mask", action="store_true",
                   help="Do NOT mask the reconstructed PWI to the brain. By default the output is "
                        "multiplied by (T1 > --brain_mask_thr): the model is only supervised inside "
                        "the brain, so out-of-brain values are undefined (scalp/vessel/ATA residual, "
                        "amplified by percentile normalisation) and are zeroed for a clean PWI.")
    p.add_argument("--brain_mask_thr", type=float, default=0.05,
                   help="Threshold on the normalised T1 ([0,1] after percentile scaling) for the "
                        "output brain mask. Default 0.05 = the training loss mask threshold.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing (same parameters as training)
# ---------------------------------------------------------------------------

def load_and_preprocess(asl_path: str, t1_path: str, cfg: Config):
    """
    Returns:
        asl_vol:  torch.Tensor [T, H, W, Z]   (normalised, on CPU)
        t1_vol:   torch.Tensor [1, H, W, Z]   (normalised, on CPU)
        ref_nib:  nibabel image of original ASL (for affine/header)
    """
    tp = cfg.asl_denoiser_train_params
    tf = Compose([
        LoadImaged(keys=["asl_diff", "t1"], image_only=False),
        EnsureChannelFirstd(keys=["asl_diff"], channel_dim=-1),  # [T,H,W,Z]
        EnsureChannelFirstd(keys=["t1"]),                         # [1,H,W,Z]
        EnsureTyped(keys=["asl_diff", "t1"], track_meta=True),
        Orientationd(keys=["asl_diff", "t1"], axcodes="LPS"),
        # Pad/crop — NOT trilinear resize — to match training (data_classes.py:39).
        # Native (96,112,52); trilinear→128 injects a periodic interpolation alias
        # that corrupts noisy ASL frames. Zero-pad H/W, center-crop Z.
        ResizeWithPadOrCropd(keys=["asl_diff"], spatial_size=(tp.asl_hw, tp.asl_hw, tp.asl_z)),
        ResizeWithPadOrCropd(keys=["t1"],       spatial_size=(tp.t1_hw,  tp.t1_hw,  tp.t1_z)),
        # T1 percentile only. ASL diff is left RAW here and normalised from the SELECTED
        # input frames in reconstruct_volume (matches training's set-A affine; the old
        # all-frames percentile+clip mismatched the retrained model and used frames the
        # deployment would not have).
        ScaleIntensityRangePercentilesd(keys=["t1"],       lower=1, upper=99, b_min=0., b_max=1., clip=True),
    ])
    sample = tf({"asl_diff": asl_path, "t1": t1_path})
    ref_nib = nib.load(asl_path)
    return sample["asl_diff"], sample["t1"], ref_nib


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _build_model(args, cfg: Config, device: torch.device,
                 arch: Optional[dict] = None) -> ASLT1Denoiser:
    """Build the denoiser. If the checkpoint carried 'arch' metadata (new ckpts),
    rebuild the EXACT training architecture from it — vmamba/conv/mossm backbone,
    2.5D slice_context, LRDA/coupling/RKMR, t1_task head width, everything — with
    no need to re-pass CLI flags and no silent shape mismatch. Otherwise fall back
    to the (limited) CLI architecture flags for old, metadata-less checkpoints."""
    tp = cfg.asl_denoiser_train_params
    if arch:
        kwargs = dict(arch)
        print(f"[INFO] Rebuilding model from ckpt 'arch' metadata "
              f"(t1_task={kwargs.get('t1_task')}, vmamba={kwargs.get('use_vmamba_encoder')}, "
              f"n_dir={kwargs.get('vmamba_n_directions')}, slice_context={kwargs.get('slice_context', 0)}).")
        return ASLT1Denoiser(**kwargs).to(device)
    print("[WARN] Checkpoint has no 'arch' metadata (old ckpt) — building from CLI "
          "flags. Pass the SAME --use_vmamba_encoder/--slice_context/--t1_task/... "
          "used at training, or strict load will fail.")
    return ASLT1Denoiser(
        asl_hw=int(tp.asl_hw), t1_hw=int(tp.t1_hw), in_ch=1,
        base_ch=args.base_ch, depth=args.depth,
        use_vmamba_encoder=bool(getattr(args, "use_vmamba_encoder", False)),
        vmamba_n_directions=int(getattr(args, "vmamba_n_directions", 2)),
        vmamba_convffn=bool(getattr(args, "vmamba_convffn", False)),
        vmamba_overlap_merge=bool(getattr(args, "vmamba_overlap_merge", False)),
        vmamba_lrda_inscan=bool(getattr(args, "vmamba_lrda_inscan", False)),
        lrda_coupling_gate=bool(getattr(args, "lrda_coupling_gate", False)),
        akmr_repro_key=bool(getattr(args, "akmr_repro_key", False)),
        slice_context=int(getattr(args, "slice_context", 0)),
        t1_task=str(getattr(args, "t1_task", "seg")),
    ).to(device)


def _load_state_into(model: ASLT1Denoiser, state: dict, ckpt_path: str) -> None:
    if "ema" in state:
        raw = model.state_dict()
        ema = {k: v.to(raw[k].device) for k, v in state["ema"].items() if k in raw}
        missing = set(raw.keys()) - set(ema.keys())
        if not missing:
            model.load_state_dict(ema, strict=True)
            print(f"[INFO] Loaded EMA weights from {ckpt_path}")
            return
        print(f"[INFO] EMA incomplete ({len(missing)} keys), falling back to raw model weights")
        model.load_state_dict(state["model"], strict=True)
    elif "model" in state:
        model.load_state_dict(state["model"], strict=True)
        print(f"[INFO] Loaded model weights from {ckpt_path}")
    else:
        model.load_state_dict(state, strict=True)
        print(f"[INFO] Loaded state dict directly from {ckpt_path}")


def load_model(args, cfg: Config, device: torch.device,
               ckpt_path: Optional[str] = None) -> ASLT1Denoiser:
    path = ckpt_path or args.checkpoint
    state = torch.load(path, map_location=device)
    model = _build_model(args, cfg, device, arch=state.get("arch") if isinstance(state, dict) else None)
    _load_state_into(model, state, path)
    model.eval()
    return model


def _enable_mc_dropout(model: torch.nn.Module) -> int:
    """Set every Dropout/Dropout2d/Dropout3d to train mode while keeping
    BatchNorm/LayerNorm/etc. in eval. Returns count of activated dropout layers."""
    n = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()
            n += 1
    return n


# ---------------------------------------------------------------------------
# Per-volume reconstruction (slice-by-slice)
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_volume(
    models: List[ASLT1Denoiser],
    asl_vol: torch.Tensor,   # [T, H, W, Z]
    t1_vol: torch.Tensor,    # [1, H, W, Z]
    n_frames: Optional[int],
    device: torch.device,
    seed: int,
    mc_n: int = 1,
    pid_zero_t1: bool = False,   # Innovation T deployment mode
    slice_context: int = 0,      # 2.5D: feed each frame as 2*ctx+1 adjacent z-slices
) -> tuple:
    """
    Reconstruct PWI volume slice-by-slice.

    If `mc_n > 1` or `len(models) > 1`, runs an ensemble of stochastic forward
    passes (MC dropout × deep-ensemble) and returns (mean, std). Std is the
    epistemic-uncertainty confidence map.

    Returns:
        pwi_mean: [H, W, Z] np.float32
        pwi_std:  [H, W, Z] np.float32 (zeros if mc_n=1 and len(models)=1)
    """
    torch.manual_seed(seed)
    T, H, W, Z = asl_vol.shape
    pwi_mean = np.zeros((H, W, Z), dtype=np.float32)
    pwi_std  = np.zeros((H, W, Z), dtype=np.float32)

    if n_frames is None or n_frames >= T:
        frame_idx = list(range(T))
    else:
        rng = np.random.default_rng(seed)
        frame_idx = rng.choice(T, size=n_frames, replace=False).tolist()

    n_used = len(frame_idx)
    frames_subset = asl_vol[frame_idx]  # [n_used, H, W, Z]  (RAW)
    # Normalise from the selected input frames only (matches training's set-A affine).
    from dataio.data_classes import _affine_from_frames
    _lo, _sc = _affine_from_frames(frames_subset)
    frames_subset = (frames_subset - _lo) / _sc

    # Total stochastic samples: mc_n × num_models (MC dropout × deep ensemble)
    n_models = len(models)
    samples_per_ckpt = max(1, mc_n)
    total_samples = samples_per_ckpt * n_models
    stochastic = total_samples > 1

    ctx = int(slice_context)
    for z in range(Z):
        if ctx > 0:
            # 2.5D: edge-clamped z-window as channels (matches ASLTwoSetDataset2DFlat).
            zc = [min(max(z + dz, 0), Z - 1) for dz in range(-ctx, ctx + 1)]
            sl = frames_subset[:, :, :, zc].permute(0, 3, 1, 2).contiguous()  # [n_used,K,H,W]
        else:
            sl = frames_subset[:, :, :, z].unsqueeze(1)                        # [n_used,1,H,W]
        set_x = sl.unsqueeze(0).to(device, dtype=torch.float32)                # [1,n_used,K,H,W]
        # T1 stays 2D (center slice) — the model keys on the center slice only.
        t1_in = t1_vol[:, :, :, z].unsqueeze(0).to(device, dtype=torch.float32)  # [1,1,H,W]
        if pid_zero_t1:
            t1_in = torch.zeros_like(t1_in)
        lengths = torch.tensor([n_used], dtype=torch.long, device=device)

        if not stochastic:
            out = models[0].infer_from_subset(set_x, t1_in, lengths=lengths)
            pwi_mean[:, :, z] = out["asl_recon"][0, 0].cpu().numpy()
            continue

        # Stochastic: collect samples, compute mean + std
        samples = np.empty((total_samples, H, W), dtype=np.float32)
        idx = 0
        for m in models:
            for _ in range(samples_per_ckpt):
                out = m.infer_from_subset(set_x, t1_in, lengths=lengths)
                samples[idx] = out["asl_recon"][0, 0].cpu().numpy()
                idx += 1
        pwi_mean[:, :, z] = samples.mean(axis=0)
        pwi_std[:, :, z]  = samples.std(axis=0)

    return pwi_mean, pwi_std


# ---------------------------------------------------------------------------
# NIfTI output — keep geometry of input ASL (after LPS reorientation + resize)
# ---------------------------------------------------------------------------

def save_nifti(data: np.ndarray, ref_nib: nib.Nifti1Image, out_path: str):
    """Save [H, W, Z] float32 array as NIfTI, reusing the reference image header."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Use simple identity-scaled header; affine from reference
    img = nib.Nifti1Image(data.astype(np.float32), affine=ref_nib.affine, header=ref_nib.header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, out_path)
    print(f"[INFO] Saved PWI → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    cfg = Config(args.config)

    # Build ensemble: primary ckpt + optional extra (e.g., swa.pth) for deep ensemble.
    models: List[ASLT1Denoiser] = [load_model(args, cfg, device)]
    if args.mc_ckpts:
        for extra_ckpt in args.mc_ckpts:
            models.append(load_model(args, cfg, device, ckpt_path=extra_ckpt))
        print(f"[INFO] Deep-ensemble: {len(models)} checkpoints")

    # MC dropout: only relevant when mc_n > 1.
    if args.mc_n > 1:
        for m in models:
            n_drop = _enable_mc_dropout(m)
            print(f"[INFO] MC dropout enabled on {n_drop} layers (mc_n={args.mc_n} per ckpt)")

    print(f"[INFO] Preprocessing {args.asl_diff} ...")
    asl_vol, t1_vol, ref_nib = load_and_preprocess(args.asl_diff, args.t1, cfg)
    T = asl_vol.shape[0]
    print(f"[INFO] ASL shape after preprocessing: {tuple(asl_vol.shape)} (T={T} frames)")

    # Brain mask for the OUTPUT (from the real, non-zeroed T1). The model is only
    # supervised inside t1 > thr; out-of-brain output is undefined, so zero it.
    brain_mask3d = None
    if not args.no_brain_mask:
        brain_mask3d = (t1_vol[0].cpu().numpy() > args.brain_mask_thr).astype(np.float32)  # [H,W,Z]
        print(f"[INFO] Output brain mask: t1 > {args.brain_mask_thr} "
              f"({float(brain_mask3d.mean()):.1%} of voxels kept). Use --no_brain_mask to disable.")

    n_frames_list: List[Optional[int]] = args.n_frames

    out_base = args.out
    stem = out_base.replace(".nii.gz", "").replace(".nii", "")
    is_stochastic = (args.mc_n > 1) or (len(models) > 1)

    for nf in n_frames_list:
        tag = f"_nframes{nf}" if nf is not None else "_allfr"
        out_path = f"{stem}{tag}.nii.gz" if len(n_frames_list) > 1 else out_base

        actual = min(nf, T) if nf is not None else T
        print(f"[INFO] Reconstructing with {actual}/{T} frames"
              + (f" | MC samples = {args.mc_n} × {len(models)} ckpts" if is_stochastic else "")
              + " ...")
        pwi_mean, pwi_std = reconstruct_volume(
            models, asl_vol, t1_vol, nf, device, args.seed, mc_n=args.mc_n,
            pid_zero_t1=bool(getattr(args, "pid_zero_t1", False)),
            slice_context=int(getattr(models[0], "slice_context", 0)),
        )
        if brain_mask3d is not None:
            pwi_mean = pwi_mean * brain_mask3d
            pwi_std  = pwi_std  * brain_mask3d
        save_nifti(pwi_mean, ref_nib, out_path)

        if is_stochastic:
            conf_path = out_path.replace(".nii.gz", "_confmap.nii.gz").replace(".nii", "_confmap.nii")
            if conf_path == out_path:
                conf_path = out_path + "_confmap.nii.gz"
            save_nifti(pwi_std, ref_nib, conf_path)
            print(f"[INFO] Confidence map (epistemic std) → {conf_path}")

            # Innovation U — Conformal Hallucination Detection: emit a binary
            # risk map where epistemic std exceeds the (1-α)-calibrated threshold.
            if args.conformal_threshold:
                import json as _json
                with open(args.conformal_threshold) as _f:
                    cal = _json.load(_f)
                q_alpha = float(cal["q_alpha"])
                alpha   = float(cal.get("alpha", 0.10))
                risk_map = (pwi_std > q_alpha).astype(np.float32)
                risk_path = out_path.replace(".nii.gz", "_riskmap.nii.gz").replace(".nii", "_riskmap.nii")
                if risk_path == out_path:
                    risk_path = out_path + "_riskmap.nii.gz"
                save_nifti(risk_map, ref_nib, risk_path)
                frac = float(risk_map.mean())
                print(f"[INFO] Conformal risk map (α={alpha:.3f}, q_α={q_alpha:.4f}) → "
                      f"{risk_path} (flagged fraction = {frac:.3%})")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
