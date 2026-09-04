# -*- coding: utf-8 -*-
"""Apples-to-apples comparison: CIG-Net v2 (ours) vs trainable + classical baselines.

Every method is scored on the SAME materialised split (fixed seed) with the project's
SELF-SUPERVISED metric suite — the metrics CLAUDE.md actually selects on — plus the
biased-reference supplementary metrics:

  self-supervised (headline):  uMSE (pooled, Marcos-Morales 2023), CNR, sCoV-GM, sCoV-WM, EFC
  supplementary (vs 12-NEX):   psnr_ref, ssim_ref, hfen, lapvar_ratio

Why a separate script from runners/eval_baselines.py: that one reports only the biased
psnr_ref/ssim/l1 family, so its numbers are not comparable to the v2 table. This script
adds the self-supervised suite and loads the v2 model through the SAME runner machinery
(scripts/eval_select_ckpt._build_runner) that produced feasibility_full.json, so the v2
architecture (MoSSM + CADA-LR + Hybrid + no_noise_var) is reconstructed correctly.

Fairness contract
-----------------
* identical materialised (setA,setB,t1,gm,wm) tuples for all methods (seed-fixed)
* identical brain mask (t1>0.05) and identical 12-NEX union reference for all methods
* each method runs in its NATIVE pipeline (ours: seg-head-masked input + EMA weights;
  PlainUNet baselines: mean(setA); naive: mean(setA); NLM: NL-means on mean(setA))
* uMSE target is ALWAYS disjoint from the frames fed as input, so the unbiased risk
  estimator stays unbiased (no input/target leakage): at the operating point (n_frames=0)
  the target is setB; in the sweep it is the held-out remainder of the pool. uMSE needs a
  3-way split, so it is reported as NaN once <3 frames remain (n_frames >= 10).
* the n_frames sweep draws a per-sample RANDOM n-subset of the FULL pool setA∪setB (=12
  NEX; deterministic in (seed,batch,n,sample) so every method sees the same frames). setA
  alone holds only 3–6 frames (TA_range=(3,6)), so 8/10/12-frame inputs must come from the
  union — the sweep therefore spans 2..12, and the reference-free ASL-QC metrics
  (CNR/sCoV/lapvar/EFC) remain valid at every n even where uMSE is unavailable.

Usage (WSL, asl-mamba env)
--------------------------
python scripts/eval_comparison_table.py \
  --config env/local/configs/wsl_asl_2d_home_v37.yml --split val \
  --out_dir /mnt/d/tmp/asl_exp/comparison_v2 \
  --ours /mnt/d/tmp/asl_exp/logs/run_wd1e4_probe/swa_feasible.pth \
  --ours_runner_args "$RA"   # the v2 runner_args (see scripts/run_mismatched_t1.sh) \
  --vanilla /mnt/d/tmp/asl_exp/logs/baseline_n2n/checkpoints/best.pth \
  --sup     /mnt/d/tmp/asl_exp/logs/baseline_sup/checkpoints/best.pth \
  --n2self  /mnt/d/tmp/asl_exp/logs/baseline_n2self/checkpoints/best.pth \
  --include_naive --include_nlm \
  --n_frames 0 2 4 6 8 10 12 --save_figures
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Callable, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config.conf_data import Config
from runners.asl_t1_guided_runner_dmvae_n2n import (
    direct_mean_from_frames, set_seed, _compute_psnr_ssim,
)
# Reuse the proven baseline loaders / runners / extra metrics.
from runners.eval_baselines import (
    materialise_split, load_unet, run_unet, run_naive,
    hfen as _hfen,
)
from utils.metrics import (
    PV_TISSUE, PV_CSF_PURE,
    scov as _scov, cnr as _cnr, entropy_focus_criterion as _efc,
    upsnr_components as _upsnr_components, laplacian_variance as _lapvar, _erode,
    tissue_csf_hf_ratio as _tcsf_ratio, hf_consistency as _hf_consistency,
    # full supplementary suite (2026-07-20: compute EVERYTHING the codebase offers)
    gmsd as _gmsd, gm_wm_contrast_error as _gmwm_ce, tenengrad as _tenengrad,
    image_entropy as _image_entropy, gradient_entropy as _grad_entropy,
    bright_tail_ratio as _bright_tail, mi_nmi as _mi_nmi,
)


def _csf_noise(img, csf, erode_iters=0, min_vox=64):
    """Standard deviation of `img` inside pure CSF, or None if too little survives.

    Purity is the point -- CSF stands in for the noise floor, and a voxel that still holds
    tissue carries perfusion -- but it is enforced by the PV threshold, not by eroding the
    mask; see PV_CSF_PURE for why erosion does not survive this data.
    """
    m = _erode((csf > PV_CSF_PURE).float(), erode_iters)
    if float(m.sum().item()) < min_vox:
        return None
    n = m.sum().clamp_min(1.0)
    mu = (img * m).sum() / n
    return ((((img - mu) ** 2) * m).sum() / n).clamp_min(1e-12).sqrt()


# thr defaults to PV_TISSUE, not 0.5: both callers want a region that is one tissue. The
# numerator is a tissue mean, which a half-and-half voxel dilutes, and the denominator is CSF
# standing in for the noise floor, where leaked tissue would count real perfusion as noise.
def _seed_suffix(path):
    """'_seedN' from a checkpoint path, or '' when the run name carries no seed."""
    m = re.search(r"_seed(\d+)", str(path).replace(chr(92), "/"))
    return "_seed%s" % m.group(1) if m else ""


# Voxels at or below this M0 are dropped from the sCoV masks rather than divided into.
# 1.0 is the floor eval_cbf.py passes to dm_to_cbf, which marks such voxels invalid, so
# sCoV and the reported CBF maps exclude the same voxels. Clamping instead of excluding
# would put pred/epsilon outliers into a std, which is exactly what sCoV is sensitive to.
_M0_FLOOR = 1.0


def _masked_mean(img, mask, thr=PV_TISSUE):
    m = (mask > thr).float()
    return (img * m).sum() / m.sum().clamp_min(1.0)


def _masked_std(img, mask, thr=PV_TISSUE):
    m = (mask > thr).float()
    n = m.sum().clamp_min(1.0)
    mu = (img * m).sum() / n
    return (((img - mu) ** 2) * m).sum().div(n).clamp_min(1e-12).sqrt()


# ------------------------------------------------------------------
# Args
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("CIG-Net v2 comparison table")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--slice_context", type=int, default=0,
                   help="2.5D: 2*ctx+1 z-slices per frame as channels. MUST match the "
                        "evaluated model's --slice_context (2 for the CIG-VSS 2.5D main); "
                        "0 = 2D. Runner-based methods (--ours/--extra_runner) must all "
                        "share it. Classical/UNet baselines (naive/NLM/--vanilla/--n2self/"
                        "--sup) are safe at any ctx: run_naive/run_unet/run_nlm_local "
                        "center-slice [B,2*ctx+1,H,W] -> [B,1,H,W] before use. Getting this "
                        "wrong is SILENT: a 2.5D model fed 1-channel packs raises, and the "
                        "per-method except-handler drops it from the table.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_frames", type=int, nargs="+", default=[0],
                   help="0 = full setA / setB target (table operating point); >0 = per-sample "
                        "random n-subset of the full setA∪setB pool (sweep, up to 12; uMSE NaN for n>=10).")
    p.add_argument("--max_samples", type=int, default=0, help="0 = all; >0 = smoke subset (#batches).")
    p.add_argument("--save_figures", action="store_true")
    # ours (v2) — loaded via runner machinery so the arch is reconstructed correctly
    p.add_argument("--ours", type=str, default=None, help="v2 checkpoint (e.g. swa_feasible.pth)")
    p.add_argument("--ours_runner_args", type=str, default=None,
                   help="Runner args string that rebuilds the v2 arch (see run_mismatched_t1.sh).")
    p.add_argument("--ours_label", type=str, default="CIG-Net v2 (ours)")
    p.add_argument("--ours_no_input_mask", action="store_true",
                   help="Feed ours raw (unmasked) input; default applies the seg-head brain mask "
                        "used in training/val (faithful to the native pipeline).")
    p.add_argument("--extra_runner", action="append", default=[],
                   help="Additional runner-based method, repeatable. Format "
                        "'LABEL::CKPT::RUNNER_ARGS' (use '::' as the separator). Each is built "
                        "and scored in the SAME val pass as --ours so per-sample pairing / "
                        "Wilcoxon stay valid. Used for the attribution baselines "
                        "(ASL-only-MoSSM via --zero_t1, naive-T1-concat via --naive_t1_concat).")
    # PlainUNet baselines (shared arch)
    # Repeatable, so a baseline can enter with as many training seeds as the proposed model
    # does. Each checkpoint becomes its own method, labelled with the seed parsed from its
    # path, and merge_seeds groups them back together -- otherwise a three-seed mean would be
    # compared against a single run of each baseline.
    p.add_argument("--vanilla", action="append", default=[], help="PlainUNet2D (mode=n2n)")
    p.add_argument("--sup", type=str, default=None, help="PlainUNet2D (mode=sup) — full-NEX upper bound")
    p.add_argument("--n2self", type=str, default=None, help="PlainUNet2D (mode=n2self)")
    # SwinIR baseline (recent-architecture reference; arch auto-detected from ckpt)
    p.add_argument("--swinir_sup", type=str, default=None,
                   help="SwinIR2D (mode=sup) — recent-architecture supervised reference (Shou et al., 2024)")
    p.add_argument("--swinir_n2n", action="append", default=[],
                   help="SwinIR2D (mode=n2n) — optional same-regime (self-supervised) SwinIR")
    # Repeatable like --vanilla and --swinir_n2n: one checkpoint per seed, labelled by the
    # seed in its path, which is what merge_seeds groups back together.
    p.add_argument("--concat", action="append", default=[],
                   help="PlainUNet2D with t1_concat: [mean(setA); T1] as a 2-channel input. "
                        "The conventional-fusion control, where T1 reaches the output through "
                        "the same convolutions as the perfusion data.")
    p.add_argument("--pv_affine_control", type=str, default=None,
                   help="PV-affine reviewer control: name of an ASL-only source method already in "
                        "the table (e.g. 'CIG-VSS+EC_T1-off' or 'vanilla_N2N'). Adds a synthetic "
                        "method that shifts the source's per-tissue GM/WM means to the MAIN model's "
                        "while preserving within-tissue residuals. If it reproduces the MAIN CNR at "
                        "the source's uMSE, the CNR lead is a per-tissue rescale, not denoising. "
                        "Requires --ours. OFF by default -> zero effect on existing runs.")
    p.add_argument("--include_naive", action="store_true", help="Add naive mean(setA) (no denoising).")
    p.add_argument("--include_nlm", action="store_true", help="Add classical NL-means baseline.")
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    # NLM params (forwarded to runners.eval_baselines.run_nlm)
    p.add_argument("--nlm_patch_size", type=int, default=5)
    p.add_argument("--nlm_patch_distance", type=int, default=6)
    p.add_argument("--nlm_h_factor", type=float, default=1.0)  # h≈sigma (Buades 2005); 0.6 degenerated to identity
    return p.parse_args()


# ------------------------------------------------------------------
# Per-sample reproducible random n-subset of the FULL frame pool
# (setA ∪ setB), shared across methods.
#
# The n-frame robustness sweep must reach up to the full acquisition (T=12 NEX):
# setA alone holds only 3–6 frames (dataset TA_range=(3,6)), so an 8/10/12-frame
# input can only be assembled from the union of setA and setB. We therefore draw
# the n-frame INPUT from the whole pool and return the held-out remainder as the
# disjoint uMSE target. Determinism in (seed,batch k,nf,sample i) keeps every
# method on the identical frames.
#   nf<=0 => operating point: input = full setA, uMSE target = setB (the main-table
#            anchor — always a valid disjoint hold-out, identical to the pre-sweep code).
#   nf>0  => input = random nf-subset of setA∪setB; target = the remaining T-nf frames.
#            uMSE needs a 3-way split, so it is NaN once <3 frames remain (n>=10);
#            the reference-free QC metrics (CNR/sCoV/lapvar/EFC) stay valid at every n.
# ------------------------------------------------------------------

def _valid_pool(pack, device):
    """setA∪setB valid frames packed into the leading positions (padding stripped),
    plus the per-sample pool length (= lenA+lenB = T)."""
    setA = pack["setA"].to(device); lenA = pack["lenA"].to(device)
    setB = pack["setB"].to(device); lenB = pack["lenB"].to(device)
    B = setA.shape[0]
    poollen = (lenA + lenB).to(device)
    Lp = int(poollen.max().item()) if B > 0 else 0
    pool = torch.zeros((B, Lp) + tuple(setA.shape[2:]), device=device, dtype=setA.dtype)
    for i in range(B):
        la = int(lenA[i].item()); lb = int(lenB[i].item())
        if la > 0:
            pool[i, :la] = setA[i, :la]
        if lb > 0:
            pool[i, la:la + lb] = setB[i, :lb]
    return pool, poollen


def pool_subset(pack, nf, seed, k, device):
    """Return (in_frames, in_len, tgt_frames, tgt_len).

    nf<=0 -> (setA, lenA, setB, lenB): operating point, target disjoint by construction.
    nf>0  -> per-sample random nf-subset of setA∪setB as input; the remaining T-nf
             frames are the held-out (disjoint) uMSE target. Chosen frames sit in the
             leading positions with matching lengths, so direct_mean_from_frames /
             infer_from_subset average exactly those frames."""
    if not nf or nf <= 0:
        setA = pack["setA"].to(device); lenA = pack["lenA"].to(device)
        setB = pack["setB"].to(device); lenB = pack["lenB"].to(device)
        return setA, lenA, setB, lenB
    pool, poollen = _valid_pool(pack, device)
    in_f = torch.zeros_like(pool); in_l = torch.zeros_like(poollen)
    tgt_f = torch.zeros_like(pool); tgt_l = torch.zeros_like(poollen)
    for i in range(pool.shape[0]):
        li = int(poollen[i].item())
        if li <= 0:
            continue
        ki = min(int(nf), li)
        g = torch.Generator().manual_seed(
            (int(seed) * 1_000_003 + int(k) * 1009 + int(nf) * 31 + i * 7 + 3) & 0x7FFFFFFF)
        perm = torch.randperm(li, generator=g)
        in_idx = perm[:ki].sort().values.to(device)
        in_f[i, :ki] = pool[i, in_idx]; in_l[i] = ki
        rest = perm[ki:]
        rj = int(rest.numel())
        if rj > 0:
            tgt_idx = rest.sort().values.to(device)
            tgt_f[i, :rj] = pool[i, tgt_idx]; tgt_l[i] = rj
    return in_f, in_l, tgt_f, tgt_l


def subset_setA(pack, nf, seed, k, device):
    """Input n-subset only (shared by every method's runner). See pool_subset."""
    in_f, in_l, _, _ = pool_subset(pack, nf, seed, k, device)
    return in_f, in_l


def splithalf_frames(pack, seed, k, device):
    """Two DISJOINT equal halves of the setA∪setB pool (seeded permutation), each
    packed into the leading positions with its length. For split-half detail
    reproducibility: reconstruct each half independently and correlate their
    high-frequency content. Because every method reconstructs from setA when
    called with nf=0, feeding a half via a shallow pack copy is method-agnostic."""
    pool, poollen = _valid_pool(pack, device)
    B = pool.shape[0]
    h1 = torch.zeros_like(pool); l1 = torch.zeros_like(poollen)
    h2 = torch.zeros_like(pool); l2 = torch.zeros_like(poollen)
    for i in range(B):
        li = int(poollen[i].item())
        if li < 2:
            continue
        g = torch.Generator().manual_seed(
            (int(seed) * 2_000_003 + int(k) * 1013 + i * 11 + 5) & 0x7FFFFFFF)
        perm = torch.randperm(li, generator=g)
        half = li // 2
        i1 = perm[:half].sort().values.to(device)
        i2 = perm[half:2 * half].sort().values.to(device)
        h1[i, :half] = pool[i, i1]; l1[i] = half
        h2[i, :half] = pool[i, i2]; l2[i] = half
    return h1, l1, h2, l2


def _presubset_pack(pack, nf, seed, k, device):
    """Shallow pack copy whose setA/lenA are the shared subset, so the reused
    eval_baselines runners (called with n_frames=0) consume exactly that subset."""
    setA, lenA = subset_setA(pack, nf, seed, k, device)
    p = dict(pack)
    p["setA"] = setA
    p["lenA"] = lenA
    return p


# ------------------------------------------------------------------
# ours (v2) via runner machinery
# ------------------------------------------------------------------

def build_runner_from(runner_args: str, ckpt: str, label: str = "runner"):
    """Rebuild any runner-based method (v2 or an attribution baseline) from its
    runner_args string + checkpoint. Shared by --ours and --extra_runner so all
    runner-based methods load identically and are scored in one val pass."""
    import gc
    from scripts.eval_select_ckpt import _build_runner, _load_model_weights
    print(f"[{label}] rebuilding arch from runner_args ...")
    runner, _ = _build_runner(runner_args)
    _load_model_weights(runner, ckpt)
    runner.model.eval()
    # Release THIS runner's cached dataset / loaders immediately. Comparison feeds
    # data from its own materialise_split, so each runner only needs model +
    # _predict_with_ema + _mask_asl_inputs. With 3 runner-based methods, keeping 3
    # cached datasets resident OOM-kills the process (137); released, the peak is one.
    for attr in ("loaders", "train_loader", "val_loader"):
        if hasattr(runner, attr):
            try:
                setattr(runner, attr, None)
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return runner


def build_ours_runner(args):
    return build_runner_from(args.ours_runner_args, args.ours, label="ours")


@torch.no_grad()
def run_nlm_local(pack, nf, seed, k, device, args):
    """Classical NL-means baseline with a MAD-based noise estimate, so it does NOT
    depend on skimage.estimate_sigma (which needs PyWavelets, often absent). Uses the
    SAME shared per-sample subset as every other method."""
    from skimage.restoration import denoise_nl_means
    from scipy.ndimage import gaussian_filter
    setA, lenA = subset_setA(pack, nf, seed, k, device)
    meanA = direct_mean_from_frames(setA, lenA)
    if meanA.shape[1] > 1:   # 2.5D: NLM denoises the center slice (pred must be [B,1,H,W])
        meanA = meanA[:, meanA.shape[1] // 2: meanA.shape[1] // 2 + 1]
    out = torch.zeros_like(meanA)
    for b in range(meanA.size(0)):
        arr = meanA[b, 0].cpu().numpy().astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        rng = hi - lo
        if rng < 1e-6:
            out[b, 0] = meanA[b, 0]
            continue
        # Rescale to [0,1] — NL-means' native intensity range. The raw PWI amplitude is small,
        # so sigma/h estimated in raw units were tiny and NLM degenerated to identity (≈ naive).
        # In [0,1] the high-freq-residual std is a meaningful noise level; clamp it to a sane
        # denoising band so h is never so small that NLM becomes a no-op.
        a01 = (arr - lo) / rng
        hp = a01 - gaussian_filter(a01, sigma=1.5)
        sigma = float(np.std(hp))
        sigma = min(max(sigma, 0.02), 0.25)
        d = denoise_nl_means(a01, h=args.nlm_h_factor * sigma, sigma=sigma,
                             patch_size=args.nlm_patch_size, patch_distance=args.nlm_patch_distance,
                             fast_mode=True, channel_axis=None)
        out[b, 0] = torch.from_numpy((d * rng + lo).astype(np.float32)).to(device)
    return out


@torch.no_grad()
def run_ours_full(runner, pack, nf, seed, k, device, apply_input_mask: bool):
    """Full infer_from_subset dict under EMA (asl_recon + EC gate maps when EC)."""
    setA, lenA = subset_setA(pack, nf, seed, k, device)
    t1 = pack["t1"].to(device)
    if apply_input_mask:
        try:
            setA = runner._mask_asl_inputs({"setA": setA, "setB": setA, "t1": t1})["setA"]
        except Exception:  # fall back to a faithful-enough t1 brain mask
            setA = setA * (t1 > 0.05).float().unsqueeze(1)
    # cond_src='fsl': stash the per-batch FSL GM/WM/CSF conditioning on the model
    # before inferring (no-op for pv/rawt1). Must match training exactly.
    runner._apply_cond_pv(runner._unwrap(), pack)
    return runner._predict_with_ema(setA, t1, lenA)


def run_ours_runner(runner, pack, nf, seed, k, device, apply_input_mask: bool):
    return run_ours_full(runner, pack, nf, seed, k, device, apply_input_mask)["asl_recon"]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/4] Materialising '{args.split}' split (seed={args.seed}) ...")
    # batch_size=1 → each unit is one slice carrying its subject_id, so the Wilcoxon
    # can pair at the SUBJECT level (per-batch pooled metrics cannot be split back to
    # subjects). Same total forward compute, just more (smaller) calls.
    val_data = materialise_split(cfg, device, split=args.split, max_samples=args.max_samples,
                                 slice_context=int(getattr(args, "slice_context", 0)),
                                 batch_size=1)
    has_pv = len(val_data) > 0 and ("gm" in val_data[0]) and ("wm" in val_data[0])
    print(f"      {len(val_data)} batches cached. GM/WM PV maps present: {has_pv}")
    if not has_pv:
        print("      [WARN] No GM/WM PV maps -> CNR / sCoV columns will be NaN.")

    # method(pack, nf, k) -> pred [B,1,H,W]; ordered classical -> baselines -> ours last.
    methods: "Dict[str, Callable]" = {}
    if args.include_naive:
        methods["naive_mean"] = lambda pack, nf, k: run_naive(_presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if args.include_nlm:
        methods["NLM"] = lambda pack, nf, k: run_nlm_local(pack, nf, args.seed, k, device, args)
    for _ck in (args.vanilla or []):
        _m = load_unet(_ck, args, device)
        methods["vanilla_N2N%s" % _seed_suffix(_ck)] = (
            lambda pack, nf, k, m=_m: run_unet(m, _presubset_pack(pack, nf, args.seed, k, device), 0, device))
    if args.n2self:
        m_ns = load_unet(args.n2self, args, device)
        methods["Noise2Self"] = lambda pack, nf, k, m=m_ns: run_unet(m, _presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if args.sup:
        m_sup = load_unet(args.sup, args, device)
        methods["UNet_sup(12NEX)"] = lambda pack, nf, k, m=m_sup: run_unet(m, _presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if getattr(args, "swinir_sup", None):
        m_swin = load_unet(args.swinir_sup, args, device)
        methods["SwinIR_sup(12NEX)"] = lambda pack, nf, k, m=m_swin: run_unet(m, _presubset_pack(pack, nf, args.seed, k, device), 0, device)
    for _ck in (getattr(args, "swinir_n2n", None) or []):
        _m = load_unet(_ck, args, device)
        methods["SwinIR_N2N%s" % _seed_suffix(_ck)] = (
            lambda pack, nf, k, m=_m: run_unet(m, _presubset_pack(pack, nf, args.seed, k, device), 0, device))
    for _ck in (getattr(args, "concat", None) or []):
        _m = load_unet(_ck, args, device)
        methods["UNet_T1concat%s" % _seed_suffix(_ck)] = (
            lambda pack, nf, k, m=_m: run_unet(m, _presubset_pack(pack, nf, args.seed, k, device), 0, device))
    ours_runner = None
    if args.ours and args.ours_runner_args:
        ours_runner = build_ours_runner(args)
        methods[args.ours_label] = lambda pack, nf, k: run_ours_runner(
            ours_runner, pack, nf, args.seed, k, device, apply_input_mask=not args.ours_no_input_mask)
    elif args.ours and not args.ours_runner_args:
        raise SystemExit("--ours requires --ours_runner_args (to rebuild the v2 arch).")

    # Extra runner-based methods (attribution baselines), scored in the same pass.
    extra_runners = []  # keep refs alive for the lambdas
    for spec in args.extra_runner:
        parts = spec.split("::")
        if len(parts) != 3:
            raise SystemExit(f"--extra_runner must be 'LABEL::CKPT::RUNNER_ARGS', got: {spec!r}")
        label, ckpt, ra = parts[0].strip(), parts[1].strip(), parts[2].strip()
        r = build_runner_from(ra, ckpt, label=label)
        extra_runners.append(r)
        methods[label] = (lambda pack, nf, k, _r=r: run_ours_runner(
            _r, pack, nf, args.seed, k, device, apply_input_mask=not args.ours_no_input_mask))

    # PV-affine reviewer control (off by default): does a per-tissue rescale of an ASL-only
    # output reproduce the MAIN model's CNR without improving uMSE? If so, the CNR lead is
    # tissue gain, not denoising. Synthetic method = source residuals + MAIN per-tissue means.
    if getattr(args, "pv_affine_control", None):
        _src_name = args.pv_affine_control
        _src_fn = methods.get(_src_name)
        if ours_runner is None:
            print("[pv_affine] --ours required for the PV-affine control — skipping.")
        elif _src_fn is None:
            print(f"[pv_affine] source {_src_name!r} not among {list(methods.keys())} — skipping.")
        else:
            def _center(x):
                return x if x.shape[1] <= 1 else x[:, x.shape[1] // 2: x.shape[1] // 2 + 1]
            def _tmean(x, m):
                return (x * m).sum(dim=(2, 3), keepdim=True) / m.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            def _pv_affine(pack, nf, k, _src=_src_fn):
                y = _src(pack, nf, k)
                ours = run_ours_runner(ours_runner, pack, nf, args.seed, k, device,
                                       apply_input_mask=not args.ours_no_input_mask)
                gm = _center(pack["gm"].to(device)); wm = _center(pack["wm"].to(device))
                gmask = (gm > PV_TISSUE).float(); wmask = (wm > PV_TISSUE).float()
                b_gm = _tmean(ours, gmask) - _tmean(y, gmask)   # per-tissue mean shift, residuals kept
                b_wm = _tmean(ours, wmask) - _tmean(y, wmask)
                return y + b_gm * gmask + b_wm * wmask
            methods[f"PVaffine({_src_name})"] = _pv_affine
            print(f"[pv_affine] added control: source={_src_name}, target tissue-means = MAIN model")

    if not methods:
        raise SystemExit("No methods selected.")
    print(f"[2/4] Methods: {list(methods.keys())}")
    print(f"      n_frames: {args.n_frames}")

    rows: List[Dict] = []
    pooled: Dict = {}          # (method, nf) -> [ssq_sum, svc_sum, n_sum]
    eff_frames: Dict = {}      # (method, nf) -> [sum_eff_len, n_samples]  (transparency)
    failed_methods: set = set()  # methods that raised mid-run are dropped, not fatal
    fig_dir = os.path.join(args.out_dir, "figures")
    if args.save_figures:
        os.makedirs(fig_dir, exist_ok=True)
    panel_cache: List = []     # first N example subjects for the multi-row recon montage

    for nf in args.n_frames:
        for k, pack in enumerate(tqdm(val_data, desc=f"n_frames={nf}")):
            t1 = pack["t1"].to(device)
            brain = (t1 > 0.05).float()
            # Disjoint held-out uMSE target for THIS nf: setB at the operating point
            # (nf<=0), the pool remainder in the sweep. uMSE needs a 3-way split, so
            # it is unavailable (NaN) once <3 frames remain (n_frames >= 10).
            _, eff_len, tgt, tgt_len = pool_subset(pack, nf, args.seed, k, device)
            umse_ok = int(tgt_len.min().item()) >= 3
            union = direct_mean_from_frames(
                torch.cat([pack["setA"].to(device), pack["setB"].to(device)], dim=1),
                (pack["lenA"] + pack["lenB"]).to(device),
            )
            # 2.5D: the model preds are the CENTER slice [B,1,H,W]; reduce the K-slice
            # reference (union) and uMSE target frames (tgt) to the center slice too, so
            # all metrics compare like-for-like (K = 2*ctx+1 > 1 signals a 2.5D window).
            if tgt.shape[2] > 1:
                tgt = tgt[:, :, tgt.shape[2] // 2: tgt.shape[2] // 2 + 1]
            if union.shape[1] > 1:
                union = union[:, union.shape[1] // 2: union.shape[1] // 2 + 1]
            gm = pack["gm"].to(device) if "gm" in pack else None
            m0 = pack["m0"].to(device) if "m0" in pack else None
            wm = pack["wm"].to(device) if "wm" in pack else None
            csf = pack["csf"].to(device) if "csf" in pack else None
            ref_lapvar = _lapvar(union, brain)
            # Batch-level context (same for every method): the 12-NEX reference's own CNR,
            # and shared reference-percentile intensity bounds so image_entropy is comparable
            # across methods (a fixed [0,1] would clip; per-method bounds would not compare).
            cnr_ref = (_cnr(union, gm, wm, threshold=PV_TISSUE)
                       if (gm is not None and wm is not None) else float("nan"))
            # The 12-repetition average is what the full-length acquisition delivers today, so
            # it belongs in the table as a row. Its SNR is measured the same way as every
            # method's: mean in tissue over the CSF standard deviation.
            _s_ref = (_csf_noise(union, csf)
                      if (csf is not None and gm is not None and wm is not None) else None)
            if _s_ref is not None:
                _s_ref = _s_ref.clamp_min(1e-8)
                snr_gm_ref = float(_masked_mean(union, gm) / _s_ref)
                snr_wm_ref = float(_masked_mean(union, wm) / _s_ref)
            else:
                snr_gm_ref = snr_wm_ref = float("nan")
            _ub = union[brain > 0.5]
            if _ub.numel() > 8:
                _ie_lo = float(torch.quantile(_ub.float(), 0.01))
                _ie_hi = float(torch.quantile(_ub.float(), 0.99))
                if not (_ie_hi > _ie_lo):
                    _ie_lo, _ie_hi = 0.0, 1.0
            else:
                _ie_lo, _ie_hi = 0.0, 1.0

            preds = {}
            for name, fn in methods.items():
                if name in failed_methods:
                    continue
                try:
                    pred = fn(pack, nf, k)
                except Exception as e:
                    print(f"[WARN] method '{name}' failed (batch {k}, nf {nf}): {e!r} — dropping it for the rest of the run.")
                    failed_methods.add(name)
                    continue
                preds[name] = pred
                # ---- self-supervised (headline) ----
                if umse_ok:
                    ssq, svc, n_pix, _ = _upsnr_components(pred, tgt, tgt_len, mask=brain)
                else:
                    ssq, svc, n_pix = 0.0, 0.0, 0.0
                acc = pooled.setdefault((name, nf), [0.0, 0.0, 0.0])
                acc[0] += ssq; acc[1] += svc; acc[2] += n_pix
                ef = eff_frames.setdefault((name, nf), [0.0, 0.0])
                ef[0] += float(eff_len.float().mean().item()); ef[1] += 1.0
                umse_batch = ((ssq - svc) / n_pix) if (umse_ok and n_pix > 0) else float("nan")
                cnr_v = _cnr(pred, gm, wm, threshold=PV_TISSUE) if (gm is not None and wm is not None) else float("nan")
                # sCoV is reported on the CBF scale, as Mutsaerts defines it. CBF is
                # proportional to dM/M0 voxel-wise, and the kinetic-model factor is a scalar
                # that cancels in a std/mean ratio, so no quantification constants are needed
                # here -- only the division by the M0 MAP, which is what separates CBF from dM.
                # Measured on 24 subjects it moves sCoV by +65% on average, because GM M0 at 7T
                # has a spatial CoV of 0.34. The dM normalisation drops out as well: its offset
                # is the 1st percentile, which the zero-clipping puts at exactly 0, leaving a
                # pure scale.
                if m0 is None:
                    pred_cbf, gm_c, wm_c = pred, gm, wm
                else:
                    ok = (m0 > _M0_FLOOR).to(pred.dtype)
                    pred_cbf = pred / m0.clamp_min(_M0_FLOOR)
                    gm_c = None if gm is None else gm * ok
                    wm_c = None if wm is None else wm * ok
                scov_gm = _scov(pred_cbf, gm_c, threshold=PV_TISSUE) if gm_c is not None else float("nan")
                scov_wm = _scov(pred_cbf, wm_c, threshold=PV_TISSUE) if wm_c is not None else float("nan")
                efc_v = _efc(pred, brain)
                # tissue SNR = mean_tissue(pred) / std_CSF(pred); CSF ΔM≈0 -> σ_CSF is the
                # noise floor (same definition as eval_select_ckpt's pooled_snr_gm/wm).
                sig_csf = (_csf_noise(pred, csf)
                           if (csf is not None and gm is not None and wm is not None) else None)
                if sig_csf is not None:
                    sig_csf = sig_csf.clamp_min(1e-8)
                    snr_gm = float(_masked_mean(pred, gm) / sig_csf)
                    snr_wm = float(_masked_mean(pred, wm) / sig_csf)
                else:
                    snr_gm = snr_wm = float("nan")
                # The reported gray-white contrast, on the same noise floor as the SNRs above
                # so the two are on one scale. Identically snr_gm - snr_wm, which is linear in
                # the two and therefore survives the per-subject and per-seed averaging that
                # the ratio in `cnr` does not.
                cnr_csf = snr_gm - snr_wm
                # reported gray-white contrast: sigma_CSF cancels, so this is mu_GM/mu_WM
                cnr_ratio = (snr_gm / snr_wm) if (snr_wm == snr_wm and snr_wm > 0) else float('nan')
                # ---- supplementary (biased ref) ----
                valid = brain.sum().clamp_min(1.0)
                l1 = float((((pred - union).abs()) * brain).sum().item() / valid.item())
                psnr_ref, ssim_ref = _compute_psnr_ssim(pred * brain, union * brain)
                hfen_v = _hfen(pred, union, brain)
                lapvar_pred = _lapvar(pred, brain)                       # absolute (no-ref sharpness)
                lapvar_ratio = lapvar_pred / max(ref_lapvar, 1e-8)
                gmsd_v = _gmsd(pred, union, brain)                       # gradient-structure deviation vs ref (↓)
                gmwm_ce = (_gmwm_ce(pred, union, gm, wm)                 # |GM/WM ratio(pred) − ratio(ref)| (↓)
                           if (gm is not None and wm is not None) else float("nan"))
                # ---- no-reference sharpness / texture suite ----
                ten_v = _tenengrad(pred, brain)                         # mean Sobel-grad magnitude (↑ sharper)
                ge_v = _grad_entropy(pred, brain)                       # gradient-magnitude entropy
                ie_v = _image_entropy(pred, brain, vmin=_ie_lo, vmax=_ie_hi)  # intensity entropy (shared bounds)
                bt_v = _bright_tail(pred, brain)                        # p99/|p50| hallucination-spike flag
                # ---- cross-modal T1 dependence (leakage diagnostics) ----
                # MI(pred,T1) read AGAINST MI(pred,union): pred≈union ⇒ natural anatomy corr;
                # MI(T1)≫MI(union) ⇒ excess T1 dependence (a leak signal). Central to the
                # content-guard claim, complementary to the mismatched-T1 gate.
                mi_t1, nmi_t1 = _mi_nmi(pred, t1, brain)
                mi_un, nmi_un = _mi_nmi(pred, union, brain)
                # ---- PRIMARY detail-quality (reference-free, noise-guarded) ----
                # (1) tissue-vs-CSF high-freq ratio: real texture / noise floor.
                if gm is not None and wm is not None and csf is not None:
                    tissue = ((gm + wm) > 0.5).float()
                    hfr_tcsf, tissue_hf = _tcsf_ratio(pred, tissue, csf, erode_iters=2, mode="grad")
                else:
                    hfr_tcsf, tissue_hf = float("nan"), float("nan")
                # (2) split-half HF reproducibility: reconstruct two DISJOINT frame
                # halves, correlate their HF content. Computed once (first n_frames)
                # since it is a per-(method,batch) property, not n-dependent.
                hfc_corr = hfc_energy = float("nan")
                if nf == args.n_frames[0]:
                    try:
                        h1f, h1l, h2f, h2l = splithalf_frames(pack, args.seed, k, device)
                        p1 = fn({**pack, "setA": h1f, "lenA": h1l}, 0, k)
                        p2 = fn({**pack, "setA": h2f, "lenA": h2l}, 0, k)
                        hfc_corr, hfc_energy = _hf_consistency(p1, p2, brain, mode="lap")
                    except Exception as e:
                        print(f"[WARN] split-half '{name}' failed (batch {k}): {e!r}")
                _sid = pack.get("subject_id")
                rows.append({
                    "batch": k, "n_frames": nf, "method": name,
                    # per-slice subject id (bs=1 → the single sample's subject) for
                    # subject-level Wilcoxon aggregation.
                    "subject_id": (_sid[0] if isinstance(_sid, (list, tuple)) and _sid else _sid),
                    "umse": float(umse_batch),
                    "cnr": float(cnr_v), "cnr_csf": float(cnr_csf), "cnr_ratio": float(cnr_ratio), "scov_gm": float(scov_gm), "scov_wm": float(scov_wm),
                    "snr_gm": snr_gm, "snr_wm": snr_wm,
                    "hfr_tcsf": float(hfr_tcsf), "tissue_hf": float(tissue_hf),
                    "hfc_corr": float(hfc_corr), "hfc_energy": float(hfc_energy),
                    "efc": float(efc_v), "cnr_ref": float(cnr_ref),
                    "snr_gm_ref": snr_gm_ref, "snr_wm_ref": snr_wm_ref,
                    "psnr_ref": float(psnr_ref), "ssim_ref": float(ssim_ref),
                    "l1_ref": l1, "hfen": float(hfen_v),
                    "lapvar": float(lapvar_pred), "lapvar_ratio": float(lapvar_ratio),
                    "gmsd": float(gmsd_v), "gmwm_contrast_err": float(gmwm_ce),
                    "tenengrad": float(ten_v), "grad_entropy": float(ge_v),
                    "image_entropy": float(ie_v), "bright_tail": float(bt_v),
                    "mi_t1": float(mi_t1), "nmi_t1": float(nmi_t1),
                    "mi_union": float(mi_un), "nmi_union": float(nmi_un),
                })

            if args.save_figures and nf == args.n_frames[0] and k < 6:
                # cache the first 6 example subjects for the multi-row Figure-4 montage
                panel_cache.append((t1[0, 0].cpu().numpy(), union[0, 0].cpu().numpy(),
                                    {nm: p[0, 0].cpu().numpy() for nm, p in preds.items()}))
                if k < 4:
                    _save_panel(t1, union, preds, os.path.join(fig_dir, f"panel_b{k:02d}.png"))
                    # EC-LRDA gate maps (r_rep / c_sem / γ) for the "ours" EC model. One
                    # extra forward per panel batch (k<4 only); no-op for non-EC ours.
                    if ours_runner is not None:
                        _oout = run_ours_full(ours_runner, pack, nf, args.seed, k, device,
                                              apply_input_mask=not args.ours_no_input_mask)
                        if any(_oout.get(kk) is not None for kk in ("ec_r_rep", "ec_c_sem", "ec_gamma")):
                            _save_ec_gates(t1, _oout, os.path.join(fig_dir, f"ec_gates_b{k:02d}.png"))

    _write_outputs(args, rows, pooled, eff_frames)
    if args.save_figures:
        _save_panel_montage(panel_cache, os.path.join(fig_dir, "recon_montage.png"))
        _plot_degradation(args, rows, pooled, fig_dir)
        _plot_framebudget_box(args, rows, fig_dir)
    print("[4/4] Done.")


# ------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------

def _ms(lst, key):
    a = np.array([r[key] for r in lst], float); a = a[np.isfinite(a)]
    return (float(a.mean()), float(a.std())) if a.size else (float("nan"), float("nan"))


def _write_outputs(args, rows, pooled, eff_frames):
    long_csv = os.path.join(args.out_dir, "comparison_long.csv")
    fields = ["batch", "subject_id", "n_frames", "method", "umse", "cnr_ratio", "cnr_csf", "cnr", "cnr_ref",
              "snr_gm_ref", "snr_wm_ref",
              "scov_gm", "scov_wm", "snr_gm", "snr_wm",
              "hfr_tcsf", "tissue_hf", "hfc_corr", "hfc_energy", "efc",
              "psnr_ref", "ssim_ref", "l1_ref", "hfen", "lapvar", "lapvar_ratio",
              "gmsd", "gmwm_contrast_err", "tenengrad", "grad_entropy", "image_entropy",
              "bright_tail", "mi_t1", "nmi_t1", "mi_union", "nmi_union"]
    with open(long_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[3/4] Wrote {long_csv} ({len(rows)} rows)")

    by = {}
    for r in rows:
        by.setdefault((r["method"], r["n_frames"]), []).append(r)

    sum_csv = os.path.join(args.out_dir, "comparison_summary.csv")
    cols = ["cnr_ratio", "cnr_csf", "cnr", "cnr_ref", "hfr_tcsf", "tissue_hf", "hfc_corr", "hfc_energy",
            "scov_gm", "scov_wm", "snr_gm", "snr_wm", "efc",
            "psnr_ref", "ssim_ref", "hfen", "lapvar", "lapvar_ratio",
            "gmsd", "gmwm_contrast_err", "tenengrad", "grad_entropy", "image_entropy",
            "bright_tail", "mi_t1", "nmi_t1", "mi_union", "nmi_union"]
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["method", "n_frames", "umse_pooled", "upsnr_pooled", "eff_frames", "n_batches"]
        for c in cols:
            header += [f"{c}_mean", f"{c}_std"]
        w.writerow(header)
        for (m, nf), lst in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            ssq, svc, n = pooled[(m, nf)]
            raw_umse = ((ssq - svc) / n) if n > 0 else float("nan")
            # Written unfloored: a negative pooled risk means the noise correction exceeded
            # the squared error, which is how the estimator reports that too few repetitions
            # were held out. uPSNR is guarded separately below and is undefined there.
            umse_pooled = raw_umse
            upsnr_pooled = (10.0 * float(np.log10(1.0 / raw_umse))) if (n > 0 and raw_umse > 1e-8) else float("nan")
            ef = eff_frames[(m, nf)]
            eff = ef[0] / max(ef[1], 1.0)
            row = [m, nf, f"{umse_pooled:.6f}", f"{upsnr_pooled:.3f}", f"{eff:.2f}", len(lst)]
            for c in cols:
                mu, sd = _ms(lst, c); row += [f"{mu:.5f}", f"{sd:.5f}"]
            w.writerow(row)
    print(f"      Wrote {sum_csv}")

    nf0 = min(r["n_frames"] for r in rows)
    md = os.path.join(args.out_dir, "comparison_table.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# CIG-Net v2 comparison @ n_frames={nf0} (0 = full setA)\n\n")
        f.write("Self-supervised headline metrics. **Primary detail-quality (reference-free, "
                "noise-guarded):** HF-ratio = tissue/CSF high-freq (≫1 real texture, ≈1 noise); "
                "HF-consist = split-half HF reproducibility (∈[-1,1], high=real); tissueHF = "
                "absolute tissue high-freq (read WITH the ratio: high ratio + tissueHF≈0 = smooth). "
                "uMSE pooled (Marcos-Morales 2023); ↓ lower better, ↑ higher better.\n\n")
        f.write("| Method | uMSE↓ | uPSNR↑ | CNR↑ | HF-ratio↑ | HF-consist↑ | tissueHF | SNR-GM↑ | SNR-WM↑ | "
                "sCoV-GM↓ | sCoV-WM↓ | EFC↓ | psnr_ref↑ | ssim_ref↑ | lapvR |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for (m, nf), lst in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if nf != nf0:
                continue
            ssq, svc, n = pooled[(m, nf)]
            raw_umse = ((ssq - svc) / n) if n > 0 else float("nan")
            umse_pooled = raw_umse
            upsnr_pooled = (10.0 * float(np.log10(1.0 / raw_umse))) if (n > 0 and raw_umse > 1e-8) else float("nan")
            g = lambda key: _ms(lst, key)[0]
            f.write(f"| {m} | {umse_pooled:.5f} | {upsnr_pooled:.2f} | {g('cnr'):.3f} | {g('hfr_tcsf'):.2f} | "
                    f"{g('hfc_corr'):.3f} | {g('tissue_hf'):.4f} | {g('snr_gm'):.3f} | "
                    f"{g('snr_wm'):.3f} | {g('scov_gm'):.3f} | {g('scov_wm'):.3f} | "
                    f"{g('efc'):.3f} | {g('psnr_ref'):.2f} | {g('ssim_ref'):.3f} | "
                    f"{g('lapvar_ratio'):.3f} |\n")

        # --- supplementary + cross-modal leakage (same operating point n_frames=nf0) ---
        f.write(f"\n## Supplementary + cross-modal leakage (n_frames={nf0})\n\n")
        f.write("Biased-ref: **GMSD↓** (gradient-structure deviation), **GMWMerr↓** (|GM/WM "
                "ratio − ref|). No-ref: **Tenengrad↑** (Sobel sharpness), gradEnt / imgEnt "
                "(texture entropy), **brightTail** (p99/|p50|; hallucination-spike flag, "
                "~stable=ok). **T1-leakage: MI(T1)/NMI(T1)** read AGAINST MI(un)/NMI(un) "
                "(natural anatomy corr with the 12-NEX ref): MI(T1)≫MI(un) = excess T1 "
                "dependence — complements the mismatched-T1 gate. cnr_ref = CNR of the ref.\n\n")
        f.write("| Method | GMSD↓ | GMWMerr↓ | Tenengrad↑ | gradEnt | imgEnt | brightTail | "
                "lapvar | MI(T1) | NMI(T1) | MI(un) | NMI(un) | cnr_ref |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for (m, nf), lst in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if nf != nf0:
                continue
            g = lambda key: _ms(lst, key)[0]
            f.write(f"| {m} | {g('gmsd'):.4f} | {g('gmwm_contrast_err'):.4f} | "
                    f"{g('tenengrad'):.4f} | {g('grad_entropy'):.3f} | {g('image_entropy'):.3f} | "
                    f"{g('bright_tail'):.2f} | {g('lapvar'):.5f} | {g('mi_t1'):.4f} | "
                    f"{g('nmi_t1'):.4f} | {g('mi_union'):.4f} | {g('nmi_union'):.4f} | "
                    f"{g('cnr_ref'):.3f} |\n")
    print(f"      Wrote {md}")

    ours_name = args.ours_label if (args.ours and args.ours_runner_args) else None
    if ours_name is not None:
        try:
            from scipy.stats import wilcoxon
            wpath = os.path.join(args.out_dir, "wilcoxon_ours_vs_baselines.csv")
            # Statistical UNIT = SUBJECT: aggregate each (method, nf) to one value per
            # subject (mean over that subject's slices), then PAIR ours-vs-baseline on
            # the shared subjects. Falls back to per-slice pairing if subject_id absent
            # (the `unit` column records which). Pairing by subject avoids the
            # pseudo-replication (inflated N / anti-conservative p) of per-slice/batch.
            have_sid = any(r.get("subject_id") is not None for r in rows)
            unit = "subject" if have_sid else "slice"

            def _per_subject(method, nf, metric):
                """{subject_id: mean(metric)} over that subject's finite slices."""
                groups = {}
                for r in rows:
                    if r["method"] != method or r["n_frames"] != nf:
                        continue
                    key = r.get("subject_id") if have_sid else r["batch"]
                    val = r[metric]
                    if val == val:                      # drop NaN
                        groups.setdefault(key, []).append(val)
                return {k: float(np.mean(v)) for k, v in groups.items() if v}

            with open(wpath, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["baseline", "n_frames", "metric", "stat", "p_value", "n", "unit"])
                methods_seen = sorted({r["method"] for r in rows})
                for nf in sorted({r["n_frames"] for r in rows}):
                    for m in methods_seen:
                        if m == ours_name:
                            continue
                        for metric in ("umse", "cnr", "hfr_tcsf", "hfc_corr", "snr_gm", "scov_gm",
                                       "gmsd", "nmi_t1"):
                            oa = _per_subject(ours_name, nf, metric)
                            ba = _per_subject(m, nf, metric)
                            shared = sorted(set(oa) & set(ba), key=lambda x: str(x))
                            a = np.array([oa[s] for s in shared], float)
                            bb = np.array([ba[s] for s in shared], float)
                            try:
                                stat, pv = wilcoxon(a, bb, zero_method="wilcox", alternative="two-sided")
                                w.writerow([m, nf, metric, f"{stat:.3f}", f"{pv:.3e}", len(shared), unit])
                            except Exception as e:
                                w.writerow([m, nf, metric, "NA", str(e), len(shared), unit])
            print(f"      Wrote {wpath} (Wilcoxon paired at {unit} level)")
        except ImportError:
            print("      scipy missing — skipped Wilcoxon.")


def _save_panel(t1, union, preds, path):
    try:
        from scipy.ndimage import binary_fill_holes
        t1_np = t1[0, 0].cpu().numpy()
        u_raw = union[0, 0].cpu().numpy()
        # Display mask = T1 brain (t1>0.05) INTERSECTED with ASL validity (|ΔM_mean|>0).
        # Preprocessing zeroed a thin rim (+ a few interior voxels) of every ASL frame that
        # the T1-derived brain mask still covers; without this intersection those exactly-
        # zero, no-signal voxels sit INSIDE the shown brain and render as a black hole/rim in
        # the 12-NEX ref and every recon. Intersecting recedes them into the background (clean
        # edge) and binary_fill_holes closes the tiny enclosed gaps. This is DISPLAY-ONLY —
        # the metrics keep their own t1>0.05 mask. Genuine low-ΔM CSF (nonzero) is untouched:
        # it stays dark because it truly has ~no perfusion, which is correct.
        brain = binary_fill_holes((t1_np > 0.05) & (np.abs(u_raw) > 0)).astype(np.float32)
        u = u_raw * brain
        # SHARED intensity window from the IN-BRAIN reference percentiles (background
        # excluded) so brightness is comparable; T1 keeps its own scale.
        ub = u[brain > 0]
        if ub.size > 0:
            vmin, vmax = float(np.percentile(ub, 1)), float(np.percentile(ub, 99))
        else:
            vmin, vmax = float(u.min()), float(u.max() + 1e-6)
        if not (vmax > vmin):
            vmin, vmax = float(u.min()), float(u.max() + 1e-6)
        n = 2 + len(preds)
        fig, ax = plt.subplots(1, n, figsize=(3 * n, 3))
        ax[0].imshow(t1_np, cmap="gray"); ax[0].set_title("T1w", fontsize=8)
        ax[1].imshow(u, cmap="gray", vmin=vmin, vmax=vmax); ax[1].set_title("12-NEX ref", fontsize=8)
        for i, (name, p) in enumerate(preds.items(), start=2):
            pm = p[0, 0].cpu().numpy() * brain
            ax[i].imshow(pm, cmap="gray", vmin=vmin, vmax=vmax)
            ax[i].set_title(name, fontsize=8)
        for a in ax:
            a.axis("off")
        fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    except Exception:
        plt.close("all")


def _save_panel_montage(cache, path):
    """Multi-example Figure 4: stack several test subjects as ROWS (columns = T1 |
    12-NEX ref | each method), one shared in-brain intensity window per row. Gives the
    qualitative panel several examples instead of one. No-op on empty cache."""
    try:
        from scipy.ndimage import binary_fill_holes
        if not cache:
            return
        names = list(cache[0][2].keys())
        ncol = 2 + len(names)
        nrow = len(cache)
        fig, ax = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow), squeeze=False)
        for ri, (t1_np, u_raw, preds_np) in enumerate(cache):
            brain = binary_fill_holes((t1_np > 0.05) & (np.abs(u_raw) > 0)).astype(np.float32)
            u = u_raw * brain
            ub = u[brain > 0]
            if ub.size > 0:
                vmin, vmax = float(np.percentile(ub, 1)), float(np.percentile(ub, 99))
            else:
                vmin, vmax = float(u.min()), float(u.max() + 1e-6)
            if not (vmax > vmin):
                vmin, vmax = float(u.min()), float(u.max() + 1e-6)
            ax[ri][0].imshow(t1_np, cmap="gray")
            ax[ri][1].imshow(u, cmap="gray", vmin=vmin, vmax=vmax)
            for ci, nm in enumerate(names, start=2):
                ax[ri][ci].imshow(preds_np[nm] * brain, cmap="gray", vmin=vmin, vmax=vmax)
            if ri == 0:
                ax[ri][0].set_title("T1w", fontsize=8)
                ax[ri][1].set_title("12-NEX ref", fontsize=8)
                for ci, nm in enumerate(names, start=2):
                    ax[ri][ci].set_title(nm, fontsize=8)
            for a in ax[ri]:
                a.axis("off")
        fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"      Wrote {path}")
    except Exception:
        plt.close("all")


def _save_ec_gates(t1, out, path):
    """Visualise the EC-LRDA gate maps for one batch: [T1 | recon | r_rep | c_sem | γ].
    r_rep/c_sem/γ ∈ (0,1] (colormap fixed to [0,1]); computed at H/ec_work_div and
    upsampled here for display only. Skips if the model produced no gate maps (non-EC
    or A1/ec_no_calib). These maps never touch the value path — pure diagnostics."""
    try:
        specs = [("ec_r_rep", "r_rep (ASL reproducibility)"),
                 ("ec_c_sem", "c_sem (T1<->ASL agreement)"),
                 ("ec_gamma", "gamma = r_rep * c_sem")]
        avail = [(k, lab) for k, lab in specs if out.get(k) is not None]
        if not avail:
            return
        H, W = t1.shape[-2], t1.shape[-1]
        t1_np = t1[0, 0].detach().cpu().numpy()
        recon = out["asl_recon"][0, 0].detach().cpu().numpy()
        ncol = 2 + len(avail)
        fig, ax = plt.subplots(1, ncol, figsize=(3 * ncol, 3))
        ax[0].imshow(t1_np, cmap="gray"); ax[0].set_title("T1w", fontsize=8)
        ax[1].imshow(recon, cmap="gray"); ax[1].set_title("recon (ours)", fontsize=8)
        for i, (k, lab) in enumerate(avail, start=2):
            m = out[k]
            if m.shape[-2:] != (H, W):
                m = torch.nn.functional.interpolate(m.float(), size=(H, W),
                                                    mode="bilinear", align_corners=False)
            mn = m[0, 0].detach().cpu().numpy()
            im = ax[i].imshow(mn, cmap="viridis", vmin=0.0, vmax=1.0)
            ax[i].set_title(lab, fontsize=8)
            fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        for a in ax:
            a.axis("off")
        fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[WARN] _save_ec_gates failed: {e!r}")
        plt.close("all")


# Metric direction hints for degradation-plot y-labels (ASCII, GBK-safe). Anything
# not listed gets its bare name (context metric — read WITH the guarded ones).
_DEGRAD_UP = {"cnr_ratio", "cnr_csf", "cnr", "cnr_ref", "upsnr", "psnr_ref", "ssim_ref", "hfr_tcsf", "hfc_corr",
              "hfc_energy", "snr_gm", "snr_wm", "tenengrad"}
_DEGRAD_DOWN = {"umse", "scov_gm", "scov_wm", "efc", "l1_ref", "hfen", "lapvar",
                "lapvar_ratio", "gmsd", "gmwm_contrast_err", "mi_t1", "nmi_t1"}


def _degrad_ylabel(name):
    if name in _DEGRAD_UP:
        return f"{name} (higher=better)"
    if name in _DEGRAD_DOWN:
        return f"{name} (lower=better)"
    return name


def _plot_degradation(args, rows, pooled, fig_dir):
    """Quality vs n_frames, one line per method — ONE FIGURE PER METRIC, for ALL
    metrics present in the per-batch rows (user request 2026-07-24). umse uses the
    pooled unbiased-risk estimate; every other numeric column is the per-batch mean.
    Skips nf=0 (full-setA point)."""
    nfs = sorted({r["n_frames"] for r in rows if r["n_frames"] and r["n_frames"] > 0})
    if len(nfs) < 2:
        return
    methods = sorted({r["method"] for r in rows})
    # every numeric metric captured per batch (umse/upsnr handled specially via `pooled`).
    skip = {"batch", "n_frames", "method", "subject_id", "umse"}
    metric_keys = ["umse", "upsnr"] + [k for k in rows[0].keys() if k not in skip]
    for metric in metric_keys:
        is_pooled = metric in ("umse", "upsnr")
        fig, axp = plt.subplots(figsize=(6, 4))
        for m in methods:
            ys = []
            for nf in nfs:
                if is_pooled:
                    ssq, svc, n = pooled.get((m, nf), (0.0, 0.0, 0.0))
                    raw = ((ssq - svc) / n) if n > 0 else float("nan")
                    if metric == "upsnr":
                        ys.append(10.0 * np.log10(1.0 / raw) if (n > 0 and raw > 1e-8) else np.nan)
                    else:
                        ys.append(max(raw, 1e-12) if n > 0 else np.nan)
                else:
                    a = np.array([r[metric] for r in rows if r["method"] == m and r["n_frames"] == nf], float)
                    a = a[np.isfinite(a)]
                    ys.append(float(a.mean()) if a.size else np.nan)
            axp.plot(nfs, ys, marker="o", label=m)
        axp.set_xlabel("n_frames (random subset of the 12-NEX pool setA∪setB)")
        axp.set_ylabel(_degrad_ylabel(metric))
        # Always span the FULL swept range with ticks at the actual n values, so a metric
        # undefined at the top does not silently shrink the x-axis and look like the sweep
        # stopped early. uMSE is the unbiased risk (needs a >=3-frame disjoint hold-out), so
        # it is NaN once <3 frames remain (n_frames >= pool-2, i.e. 10/12 for a 12-NEX pool)
        # and its line ends before the last tick — by design, not a truncated sweep. CNR/sCoV
        # are reference-free and span every n.
        axp.set_xticks(nfs)
        axp.set_xlim(min(nfs) - 0.3, max(nfs) + 0.3)
        if is_pooled:
            axp.set_title("uMSE undefined where <3 frames remain held-out (line ends early)", fontsize=8)
        axp.legend(fontsize=7); axp.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"degradation_{metric}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"      Wrote {p}")


def _plot_framebudget_box(args, rows, fig_dir):
    """Frame-budget BOXPLOTS (replaces the frame-budget summary table): per metric,
    x = n_frames, a group of boxes (one per method, ablations included) showing the
    per-subject distribution. uMSE/uPSNR are per-subject unbiased-risk values (NaN
    where <3 frames remain held-out); CNR / SSIM span every n. One figure per metric."""
    nfs = sorted({r["n_frames"] for r in rows if r["n_frames"] and r["n_frames"] > 0})
    if len(nfs) < 2:
        return
    methods = sorted({r["method"] for r in rows})
    metrics = [("cnr", "GM–WM CNR ↑"), ("upsnr", "uPSNR (dB) ↑"),
               ("umse", "uMSE ↓"), ("ssim_ref", "SSIM vs 12-NEX ↑")]

    def val(r, metric):
        if metric == "upsnr":
            try:
                u = float(r.get("umse"))
                return 10.0 * np.log10(1.0 / u) if u > 1e-8 else np.nan
            except Exception:
                return np.nan
        try:
            return float(r.get(metric))
        except Exception:
            return np.nan

    cmap = plt.get_cmap("tab10")
    nM = len(methods)
    width = 0.8 / max(nM, 1)
    for metric, label in metrics:
        if metric not in rows[0] and metric != "upsnr":
            continue
        fig, ax = plt.subplots(figsize=(1.7 * len(nfs) + 2.5, 4.2))
        for mi, m in enumerate(methods):
            data, pos = [], []
            for j, nf in enumerate(nfs):
                vals = [v for v in (val(r, metric) for r in rows
                                    if r["method"] == m and r["n_frames"] == nf)
                        if v == v]
                data.append(vals)
                pos.append(j + (mi - (nM - 1) / 2.0) * width)
            bp = ax.boxplot(data, positions=pos, widths=width * 0.9,
                            patch_artist=True, showfliers=False, manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=cmap(mi % 10), alpha=0.55, linewidth=0.7)
            for med in bp["medians"]:
                med.set(color="black", linewidth=1.0)
            ax.plot([], [], color=cmap(mi % 10), lw=6, alpha=0.55, label=m)  # legend proxy
        ax.set_xticks(range(len(nfs)))
        ax.set_xticklabels([str(n) for n in nfs])
        ax.set_xlabel("n_frames (random subset of the 12-NEX pool)")
        ax.set_ylabel(label)
        ax.set_title(f"Frame-budget distribution: {label}", fontsize=9)
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        p = os.path.join(fig_dir, f"framebudget_box_{metric}.png")
        fig.tight_layout(); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"      Wrote {p}")


if __name__ == "__main__":
    main()
