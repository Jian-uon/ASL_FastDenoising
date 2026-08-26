# -*- coding: utf-8 -*-
# Runner for ASL T1 DMVAE — symmetric Noise2Noise training with cross-modal attention.
#   - SetTransformer aggregation over T ASL frames
#   - CrossModalFusion: ASL attends to T1 at bottleneck
#   - Single feat_dim feature vector per branch
#   - ASL supervised by cross-split Noise2Noise

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.conf_data import Config
from dataio.dataloaders import get_asl_2d_loaders
from losses.asl_n2n_loss import ASLN2NLoss
from models.asl_t1_model import ASLT1Denoiser


def _parse_stage_mask(s: str, depth: int, flag_name: str = "--stages") -> Optional[list]:
    """Parse comma-separated 0-indexed stage indices into a [depth]-long bool
    list. Empty string ⇒ None (caller picks default).

    Examples (depth=4):
        ""        -> None    (caller selects default)
        "3"       -> [F,F,F,T]
        "2,3"     -> [F,F,T,T]
        "0,2"     -> [T,F,T,F]
    """
    s = (s or "").strip()
    if not s:
        return None
    try:
        idxs = [int(x) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"{flag_name} parse error: {s!r}") from e
    mask = [False] * int(depth)
    for i in idxs:
        if 0 <= i < int(depth):
            mask[i] = True
        else:
            raise ValueError(f"{flag_name}: index {i} out of range [0,{depth})")
    return mask


TensorLike = Union[Tensor, Sequence[Tensor]]

try:
    from utils.training_utils import EMAModel  # type: ignore
except Exception:
    EMAModel = None


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _pool_upsnr(sum_sq: float, sum_vc: float, n: float,
                data_range: float = 1.0, eps: float = 1e-8) -> float:
    """Pooled uPSNR: aggregate raw sums across all val batches, then take one
    log. Avoids the log-of-mean ≠ mean-of-log inflation when uMSE varies
    per batch (Marcos-Morales ICML 2023 §4)."""
    if n <= 0:
        return float("nan")
    umse = max((sum_sq - sum_vc) / n, eps)
    import math
    return 10.0 * math.log10((data_range ** 2) / umse)


def _pool_umse(sum_sq: float, sum_vc: float, n: float, eps: float = 1e-8) -> float:
    """Pooled uMSE — the linear-scale risk estimator. Used as primary objective
    in constrained model selection (uMSE is linear; uPSNR is its log transform,
    which breaks strict orderings under bootstrap CI)."""
    if n <= 0:
        return float("nan")
    return max((sum_sq - sum_vc) / n, eps)


def make_loggers(exp_root: str, run_name: str, verbose: str) -> SummaryWriter:
    log_dir = os.path.join(exp_root, "logs", run_name)
    tb_dir = os.path.join(exp_root, "tensorboard", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    level = getattr(logging, verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported verbose level: {verbose}")

    fmt = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    h1 = logging.StreamHandler()
    h1.setFormatter(fmt)
    h2 = logging.FileHandler(os.path.join(log_dir, "stdout.txt"))
    h2.setFormatter(fmt)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(h1)
    logger.addHandler(h2)
    logger.setLevel(level)

    writer = SummaryWriter(log_dir=tb_dir)
    return writer


def parse_args():
    p = argparse.ArgumentParser("ASL T1 Denoiser runner")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument("--exp", type=str, default="./exp")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", type=str, default="info")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--save_images", action="store_true")
    p.add_argument("--log_images", type=int, default=20)
    p.add_argument("--save_every", type=int, default=100,
                   help="Save a step-tagged checkpoint every N training steps (in addition to latest/best)")
    p.add_argument("--early_stop_patience", type=int, default=20,
                   help="Stop training after N consecutive evals without l1_B improvement (0 = disabled)")
    p.add_argument("--early_stop_min_evals", type=int, default=30,
                   help="Minimum number of evals before early stop is allowed to trigger")

    # Model architecture
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--use_wavelet", action="store_true",
                   help="Replace stride-2 conv down/up with DWT/IDWT (Haar) for "
                        "information-preserving multiscale processing.")
    p.add_argument("--use_t1_cross_fusion", action="store_true",
                   help="Enable multi-scale tissue-gated cross-attention in the ASL "
                        "decoder (v34+): T1 skips + tissue-class similarity bias from "
                        "t1_seg are injected at scales with H*W <= --t1_attn_max_tokens. "
                        "Anti-hallucination preserved (V=ASL inside cross-attention).")
    # v36+: J-invariance input masking (replaces TV) + SWA-from-step
    p.add_argument("--jinv_p", type=float, default=0.0,
                   help="Probability of masking each pixel position in setA frames "
                        "(consistently across T frames) with the local 3x3-mean of the "
                        "frame. Forces approximate J-invariance: model can't directly "
                        "use setA[h,w] in predicting output at (h,w), must use context. "
                        "Standard self-supervised regularizer (Krull N2V 2019, Batson "
                        "N2Self 2019). Recommended 0.1.")
    p.add_argument("--use_swa", action="store_true",
                   help="Maintain a running mean of model weights from --swa_start_step "
                        "onwards via torch.optim.swa_utils.AveragedModel. Saved as swa.pth "
                        "alongside best.pth. Finds flat-minimum solution, less drift to "
                        "noise textures (Izmailov ICLR 2018).")
    p.add_argument("--swa_start_step", type=int, default=200,
                   help="Step from which SWA running mean accumulation begins.")
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="EMA max decay (the constant decay actually used by the "
                        "fallback EMA). Default 0.9999. Lower (e.g. 0.99) makes the "
                        "EMA track the live model more closely → sharper/less-averaged "
                        "evaluated weights, at the cost of less stabilisation.")
    p.add_argument("--ema_start_step", type=int, default=0,
                   help="EMA warmup: for the first N optimizer steps the EMA copies "
                        "the live weights (decay=0), then switches to --ema_decay from "
                        "step N+1. Prevents the EMA from being dragged by the random "
                        "ASL-branch init early on; the average then starts from a "
                        "sensible θ_N. Default 0 (no warmup). Pair with --best_min_step.")
    # Innovation A — Adversarial mismatched-T1 training (identity-invariant
    # cross-modal conditioning). See ASL_dmvae/docs/method_innovations.md §A.
    p.add_argument("--use_adv_t1", action="store_true",
                   help="Enable adversarial mismatched-T1 loss: with prob --p_adv, run a "
                        "second forward pass using a permuted T1 from another subject in "
                        "the batch and penalise output divergence. Forces T1-identity "
                        "invariance, training-time fix for T1 hallucination.")
    p.add_argument("--p_adv", type=float, default=0.3,
                   help="Per-step probability of triggering the mismatched-T1 branch.")
    p.add_argument("--w_adv", type=float, default=0.5,
                   help="Weight on the adversarial L1(f_match, sg(f_mismatch)) term.")
    p.add_argument("--adv_no_stopgrad", action="store_true",
                   help="Disable stop-gradient on the mismatched-T1 branch (default uses "
                        "BYOL-style stop-grad to avoid two-way training instability).")
    # Innovation T — Privileged Information Distillation (LUPI-style).
    # See ASL_dmvae/docs/method_innovations.md §T. T1 is treated as 'privileged information'
    # available only at training time; the model self-distills its T1-aware output
    # into a T1-free forward pass, so deployment can run with T1=0 (zero T1 input
    # path) and hallucination is structurally impossible at inference.
    p.add_argument("--use_pid", action="store_true",
                   help="Enable Privileged Information Distillation: each step run a "
                        "second forward with T1 set to zeros, and add masked L1 loss "
                        "between the zero-T1 output and the (stop-gradient) T1-aware "
                        "output. After training, inference can use T1=0 with no "
                        "T1-induced hallucination path.")
    p.add_argument("--w_pid", type=float, default=0.5,
                   help="Weight on the PID self-distillation loss term.")
    p.add_argument("--pid_zero_t1_at_eval", action="store_true",
                   help="At validation, replace the model's T1 INPUT with zeros "
                        "(brain mask still derived from the real T1). Use this to "
                        "evaluate the deployment-mode model after PID training.")
    # v37+: per-pixel BLUE aggregator + tissue FiLM + heteroscedastic head
    p.add_argument("--use_svfw", action="store_true",
                   help="Use SpatialVaryingFrameWeighting (per-pixel BLUE) instead of "
                        "SetTransformerAggregator (per-frame scalar). Safe-by-design: log-var "
                        "head sees only deviation features (frame - mean), never raw signal "
                        "→ cannot suppress lesions based on signal atypicality.")
    p.add_argument("--use_film", action="store_true",
                   help="Per-channel TissueFiLM modulation in ASL decoder, conditioned on "
                        "globally-pooled T1 PV fractions. Safe-by-design: per-channel only "
                        "(not per-pixel) → cannot locally edit specific regions.")
    p.add_argument("--use_heteroscedastic", action="store_true",
                   help="ASL decoder outputs (μ, log σ²) instead of just μ. Loss switches "
                        "from L1 to Gaussian NLL (Kendall & Gal NeurIPS 2017). σ² serves as "
                        "per-pixel uncertainty / clinical confidence map.")
    p.add_argument("--t1_attn_max_tokens", type=int, default=1024,
                   help="Skip multi-scale T1 fusion at decoder levels exceeding this "
                        "token count (H*W). Default 1024 (=32x32) keeps attention O(N^2) "
                        "memory bounded.")
    # DEPRECATED no-ops (2026-06-21): the entire Var_T(set_a) noise-variance
    # pathway (RGSF / CMAM-BLUE / CADA z_σ) was removed from the model. These
    # flags are still accepted so older run scripts don't crash, but they have
    # no effect. Safe to delete from any new command.
    p.add_argument("--use_rgsf", action="store_true",
                   help="DEPRECATED no-op (noise-var pathway removed 2026-06-21).")
    p.add_argument("--no_noise_var", action="store_true",
                   help="DEPRECATED no-op: noise-var is already gone (removed 2026-06-21).")
    p.add_argument("--use_reshead", action="store_true",
                   help="ResHead-with-cap: asl_recon = agg + δ_max · tanh(decoder_raw). "
                        "Bounds the additive correction magnitude per pixel to ±δ_max. "
                        "Reframes the decoder as a bounded refiner of the SVFW baseline "
                        "rather than a free-form generator.")
    p.add_argument("--reshead_delta_max", type=float, default=0.5,
                   help="δ_max for --use_reshead (default 0.5 in [0,1] ΔM units).")
    # CC-5: Bidirectional N2N cycle with stop-grad (BYOL-style noise-invariance)
    p.add_argument("--use_bdcyc", action="store_true",
                   help="Asymmetric stop-grad cycle (CC-5): require f(set_a) ≈ "
                        "stop_grad(f(set_b)). Both subsets share the same clean "
                        "signal but have independent noise; mimicry of per-target "
                        "noise cannot satisfy this constraint. Anti-mimicry loss.")
    p.add_argument("--w_bdcyc", type=float, default=0.3,
                   help="Weight for CC-5 bidirectional-N2N cycle loss (default 0.3).")
    p.add_argument("--w_ssim_cyc", type=float, default=0.0,
                   help="Weight for ssim_cyc = 1-SSIM(f(A), sg(f(B))). pred-to-pred "
                        "structural consistency; suppresses low-freq mottling without "
                        "touching the noisy target (unlike SSIM-vs-target which would "
                        "reward the reference's noise texture). Reuses the set_b forward "
                        "from --use_bdcyc. 0 disables. Recommended 0.05.")
    p.add_argument("--w_cnr_cyc", type=float, default=0.0,
                   help="Weight for cnr_cyc = |CNR(f(A)) - sg(CNR(f(B)))|, the "
                        "CNR (GM-vs-WM contrast-to-noise) consistency between the two "
                        "disjoint subsets. Stabilises late-training CNR wobble: the "
                        "true contrast is subset-invariant, only noise differs, so "
                        "penalising the A/B CNR gap pushes toward a noise-robust "
                        "contrast. Per-sample CNR, eps-floored sigma_WM, slices with "
                        "<20 GM/WM voxels skipped. Reuses the set_b forward. 0 disables. "
                        "Recommended 0.05.")
    # T1 dropout (2026-05-18, anti-leakage regulariser added after v42f probe
    # showed 16.4% mismatched-T1 sensitivity). With prob p, replace t1 with
    # zeros for that training step. Pushes the network to learn ASL-driven
    # features that work without T1 anatomy, weakening the T1→output circuit.
    p.add_argument("--t1_dropout_p", type=float, default=0.0,
                   help="Per-step prob of zeroing the T1 input during training "
                        "(2026-05-18 anti-T1-leak regulariser). Default 0.0.")
    # v42g encoder anti-leakage flags (2026-05-18 P2+P3; P1 T1-low-pass removed 2026-06-22)
    p.add_argument("--mossm_t1_gated", action="store_true",
                   help="P2: split MoSSM B/Δ projection into ASL base + "
                        "sigmoid-gated T1 residual (gate init σ(-4)≈0.018). "
                        "Trainability invariant: gate closed = ASL-only MoSSM.")
    p.add_argument("--mossm_t1_gate_init", type=float, default=-4.0,
                   help="Initial logit for MoSSM T1 gate. σ(-4)=0.018 (default, conservative). "
                        "σ(-2.5)=0.076 / σ(-2)=0.119 / σ(0)=0.5. Raise to give T1 path "
                        "stronger initial pressure (Phase 2 Run A activation test).")
    p.add_argument("--t1_path_lr_mult", type=float, default=1.0,
                   help="LR multiplier applied to T1-conditioning parameters (mossm "
                        "t1_gate_bd, x_proj_BD T1 path). 1.0 = uniform LR; "
                        "5.0 gives T1 path 5× the main LR so it can catch up if "
                        "starved (Phase 2 Run A).")
    # Phase 2 Run C (2026-05-23): MoSSM gate warmup. Forces t1_gate_bd's sigmoid
    # value to a fixed level (default 0.20 = logit −1.39) and freezes it for the
    # first N steps. This breaks the chicken-and-egg observed in Run A: low gate
    # ⇒ low BD_t1 contribution ⇒ low gate gradient ⇒ gate stays low. By holding
    # the gate open during warmup, BD_t1 projection weights are forced to learn
    # useful T1 features; after the freeze releases, the gate can find its own
    # equilibrium starting from a non-trivial operating point.
    p.add_argument("--mossm_gate_warmup_steps", type=int, default=0,
                   help="Steps during which t1_gate_bd is frozen at "
                        "sigmoid^-1(--mossm_gate_warmup_value). 0 = disabled.")
    p.add_argument("--mossm_gate_warmup_value", type=float, default=0.20,
                   help="Fixed sigmoid value of t1_gate_bd during warmup. "
                        "0.20 -> raw logit -1.386. 0.50 -> 0.0 (gate fully open). "
                        "Higher values risk T1 anatomy leakage - keep <= 0.25.")
    p.add_argument("--cmam_k_rank_div", type=int, default=1,
                   help="P3: low-rank bottleneck on CMAM K, divisor over d_t1. "
                        "k_rank_div=4 means K is compressed to d_t1//4 before "
                        "cross-attention. 1 = off / v42f behaviour.")
    # DEPRECATED no-ops (2026-06-21): RGSF calibration knobs — RGSF was removed
    # with the rest of the noise-var pathway. Accepted for script compat only.
    p.add_argument("--rgsf_a_init", type=float, default=4.0,
                   help="DEPRECATED no-op (RGSF removed 2026-06-21).")
    p.add_argument("--rgsf_b_raw_init", type=float, default=-3.0,
                   help="DEPRECATED no-op (RGSF removed 2026-06-21).")
    p.add_argument("--rgsf_var_zscore", action="store_true",
                   help="DEPRECATED no-op (RGSF removed 2026-06-21).")
    # 2026-05-25 CADA: Channel-Adaptive Dynamics Adapter. Evidence = ASL +
    # dense T1 + z(σ²); gate is standard sigmoid(MLP), no g_max cap, no
    # norm-balance. Per-channel α adapter (zero-init) means step 0 is pure-ASL.
    p.add_argument("--use_cada", action="store_true",
                   help="Enable CADA gate at stages listed in --cada_stages.")
    p.add_argument("--cada_stages", type=str, default="",
                   help="Comma-separated 0-indexed stage list where CADA is "
                        "active, e.g. '2,3'. Empty + --use_cada => deepest 2 "
                        "stages auto-selected.")
    p.add_argument("--cada_n_groups_B", type=int, default=4,
                   help="Number of channel groups for the CADA B-side gate. "
                        "d_state must be divisible by this. Default 4.")
    # 2026-06-15 attribution baselines (see ASL_dmvae/docs/cada_lr_design.md §10).
    p.add_argument("--zero_t1", action="store_true",
                   help="ASL-only-MoSSM baseline: blank the T1 input (t1:=0) before "
                        "the T1 encoder. Same architecture/params as the full model; "
                        "T1 machinery runs on zeros so it carries no subject info. "
                        "Isolates the T1 INFORMATION contribution. Brain mask "
                        "(computed outside the model from real T1) is unaffected.")
    p.add_argument("--pv_mode", choices=["real", "zero", "uniform"], default="real",
                   help="Ablate the anatomical-guidance INFORMATION fed to the soft-PV "
                        "bilinear + gate. 'real'=frozen-PV composition (default); "
                        "'zero'=PV:=0 (bias-free E => modulation is EXACTLY identity => "
                        "plain VSS on ASL = no-anatomy lower bound M0); 'uniform'=PV:=1/C "
                        "(constant tissue prior). STRUCTURAL: train and eval must match. "
                        "Supersedes the confounded --zero_t1.")
    p.add_argument("--naive_t1_concat", action="store_true",
                   help="Naive-T1-concat baseline: T1 enters ONLY as a 2nd input "
                        "channel to the ASL encoder stem (unguarded → flows into the "
                        "value/skip/decoder path); structured guidance skips are "
                        "zeroed. Requires --use_mossm_encoder OR --use_vmamba_encoder; "
                        "for the VSS backbone pair with --vmamba_no_guard and no "
                        "--ec_lrda so the concat is the sole, unguarded T1 route "
                        "(A1 in the manuscript ablation). The un-guarded strawman for "
                        "the content-guarding ablation.")
    # CADA-LR (2026-06-05): Anatomy-Conditioned Low-Rank Dynamics Adapter.
    p.add_argument("--cada_lr", action="store_true",
                   help="Replace the gated T1-residual on B with a T1-conditioned "
                        "low-rank adapter ΔB=U_B(s⊙V_B u) (T1 only reweights an ASL "
                        "projection; no additive T1). Δ stays gated. "
                        "See ASL_dmvae/docs/cada_lr_design.md.")
    p.add_argument("--cada_lr_rank", type=int, default=8,
                   help="Low-rank dimension r for CADA-LR (default 8).")
    p.add_argument("--cada_lr_bound", type=float, default=4.0,
                   help="tanh bound c for CADA-LR coefficients s=c*tanh(phi(t1,z)).")
    # v42j (2026-05-19): NAFNet SimpleGate decoder fusion
    p.add_argument("--use_naf_fusion", action="store_true",
                   help="ASLDetailDecoder uses NAFNet-style fusion (Conv1x1 + "
                        "NAFBlock with SimpleGate + SCA) in place of "
                        "Conv1x1+GN+SiLU. Stronger implicit denoising on the "
                        "fused feature. β/γ=0 init makes it identity at start.")
    # v42k (2026-05-19): Restormer MDTA at NAFFusion (channel-axis attention)
    p.add_argument("--use_mdta_fusion", action="store_true",
                   help="Inside every NAFFusionBlock, replace SCA with Restormer "
                        "MDTA (Multi-Dconv head Transposed Attention; channel-axis "
                        "self-attention). Cost: O(C²·HW). Requires --use_naf_fusion.")
    # v42: T1-Modulated Selective State-Space encoder (MoSSM-ASL)
    p.add_argument("--use_mossm_encoder", action="store_true",
                   help="Replace the ASL ConvEncoder2D with a 4-stage MoSSM encoder. "
                        "Each MoSSM block conditions B/Δ jointly on (x_asl, t1_skip) "
                        "while keeping the readout C ASL-only — an SSM-block-level "
                        "V=ASL invariant that complements CrossModalFusion. v42.")
    p.add_argument("--mossm_blocks_per_scale", type=int, default=2,
                   help="Number of MoSSM blocks at each encoder stage. Default 2.")
    p.add_argument("--mossm_n_directions", type=int, default=1,
                   help="Number of 2D scan directions per MoSSM block (1/2/4). "
                        "Default 1 for speed (4 is VMamba-equivalent cross-scan).")
    # Hybrid CNN-SSM + stage-adaptive scan (2026-06-05). See
    # ASL_dmvae/docs/archive/v2_mossm/hybrid_mossm_stage_adaptive_implementation.md (RETIRED line).
    p.add_argument("--mossm_n_directions_by_stage", type=str, default="",
                   help="Optional comma-separated per-stage MoSSM scan directions, "
                        "e.g. '1,1,2,2'. Overrides --mossm_n_directions when set.")
    p.add_argument("--use_hybrid_mossm_block", action="store_true",
                   help="Enable an ASL-only local CNN branch inside each MoSSMBlock, "
                        "fused with the global MoSSM/CMAM readout (T1-free local path).")
    p.add_argument("--hybrid_local_expand", type=float, default=2.0,
                   help="Channel expand ratio of the hybrid local branch. Default 2.0.")
    p.add_argument("--hybrid_local_kernel", type=int, default=3,
                   help="Depthwise kernel size of the hybrid local branch. Default 3.")
    p.add_argument("--mossm_d_state", type=int, default=16,
                   help="SSM hidden state dimension per channel. Default 16 (MambaIR).")
    # 2026-07 CIG-UNet: content-guard on the plain conv backbone (ASL_dmvae/docs/cig_unet_design.md)
    p.add_argument("--use_conv_encoder", action="store_true",
                   help="CIG-UNet: content-guarded ConvEncoder2D (LRDA-Conv + AKMR) "
                        "+ T1-free plain decoder. Grafts the V=ASL content-guard onto "
                        "the strong conv backbone that beat MoSSM. Mutually exclusive "
                        "with --use_mossm_encoder. T1-off control: add --zero_t1.")
    p.add_argument("--lrda_rank", type=int, default=8,
                   help="LRDA-Conv low-rank bottleneck r. Default 8.")
    p.add_argument("--lrda_bound", type=float, default=4.0,
                   help="LRDA-Conv gate bound c in s=c*tanh(phi(t1)). Default 4.0.")
    p.add_argument("--lrda_variant", type=str, default="c", choices=["a", "b", "c"],
                   help="LRDA-Conv variant: a=per-channel gate, b=low-rank spatial, "
                        "c=low-rank spatial + depthwise mixing (default).")
    p.add_argument("--conv_lrda_stages", type=str, default="",
                   help="Comma-separated LRDA-Conv stages, e.g. '0,1,2,3'. Empty=all.")
    p.add_argument("--no_akmr", action="store_true",
                   help="Force-disable the AKMR cross-attention guard (LRDA-Conv only). "
                        "AKMR is OFF by default in the main model line; use --use_akmr to "
                        "turn it back on for the ablation.")
    p.add_argument("--use_akmr", action="store_true",
                   help="Enable the AKMR anatomy-keyed memory reader (conv/VSS). OFF by "
                        "default in the main line (weak + redundant with LRDA); this flag "
                        "re-enables it for the §4.5 ablation. Overridden by --no_akmr.")
    p.add_argument("--akmr_k_rank_div", type=int, default=4,
                   help="AKMR low-rank K (T1 key) bottleneck divisor. Default 4.")
    # 2026-07 VMamba backbone (baseline + CIG-VMamba)
    p.add_argument("--use_vmamba_encoder", action="store_true",
                   help="VMamba backbone (VSS-block, 4-dir cross-scan) as ASL encoder. "
                        "With content-guard = CIG-VMamba; add --vmamba_no_guard for the "
                        "pure VMamba ASL baseline. Mutually exclusive with "
                        "--use_conv_encoder / --use_mossm_encoder.")
    p.add_argument("--vmamba_no_guard", action="store_true",
                   help="Pure VMamba baseline: no LRDA/AKMR content-guard (ASL-only "
                        "backbone comparison point).")
    p.add_argument("--vmamba_d_state", type=int, default=16,
                   help="VMamba SSM state dim. Default 16.")
    p.add_argument("--vmamba_n_directions", type=int, default=2, choices=[1, 2, 4],
                   help="VMamba 2D scan directions (2 = default, matches the CIG-VSS main line; "
                        "4 = full cross-scan, slower). Default 2.")
    p.add_argument("--vmamba_hv_scan", action="store_true",
                   help="With --vmamba_n_directions 2, scan {->,down} (one horizontal + one "
                        "vertical) instead of the default horizontal-bidirectional {->,<-}: "
                        "isotropic 2D coverage at the same cost. No effect for n=1/4.")
    p.add_argument("--vmamba_blocks_per_scale", type=int, default=2,
                   help="VSS blocks per VMamba encoder stage. Default 2.")
    # 2026-07 single-branch conv inductive bias for VMamba (ASL_dmvae/docs/cig_unet_design.md).
    # Fold the conv bias INTO the token stream instead of a parallel branch, so the
    # SSM's long-range modelling is not averaged away. All default OFF = unchanged.
    p.add_argument("--vmamba_convffn", action="store_true",
                   help="Replace each VSS block's channel-only MLP with a ConvFFN "
                        "(Linear->DWConv3x3->GELU->Linear): injects local spatial "
                        "inductive bias in the channel-mixing stage. Single-branch.")
    p.add_argument("--vmamba_overlap_merge", action="store_true",
                   help="Use overlapping 3x3 stride-2 conv downsampling between VMamba "
                        "stages instead of the non-overlapping 2x2 patch-merge "
                        "(overlapping receptive field + anti-aliasing).")
    p.add_argument("--vmamba_lrda_inscan", action="store_true",
                   help="CIG-VSS (novelty): move the LRDA content-guard INTO the SS2D scan "
                        "as T1-conditioned zero-init low-rank modulation of the dynamics — "
                        "(1) multiplicative B-gate B·(1+U_B(s)) and (2) log-space Δ-mod "
                        "Δ·exp(b·tanh(U_Δ(s))) (b=--lrda_dt_bound). Both rate/routing-only, "
                        "no additive T1 (u/C untouched) ⇒ V=ASL by construction. Replaces "
                        "the post-block LRDAConv2d. Re-verify mismatched-T1 L1=0.")
    # Design A / C — safe realisations of a T1↔ASL agreement gate (need --vmamba_lrda_inscan).
    p.add_argument("--lrda_cond_asl", action="store_true",
                   help="Design A: condition the in-scan LRDA gate φ on [T1 ; ASL] "
                        "jointly (T1 only modulates B where it is compatible with ASL). "
                        "Still multiplicative / no additive T1 ⇒ V=ASL preserved.")
    p.add_argument("--lrda_dt_bound", type=float, default=1.0,
                   help="In-scan LRDA log-space Δ modulation bound b (part of the method, "
                        "always on with --vmamba_lrda_inscan): Δ_mod = Δ·exp(b·tanh(U_Δ(s))), "
                        "U_Δ zero-init, Δ stays in [e^-b,e^b]·Δ. Default 1.0 (~0.37x..2.7x); "
                        "smaller = safer/weaker. Rate-only ⇒ V=ASL preserved.")
    p.add_argument("--lrda_repro_gate", action="store_true",
                   help="Design C: reproducible-ASL reliability gate r∈(0,1) that "
                        "SUPPRESSES T1 guidance (in-scan LRDA B-gate + AKMR residual) "
                        "where the ASL evidence is not reproducible across an even/odd "
                        "frame split. Grounded in raw data (no learnable ASL feature) ⇒ "
                        "no feedback leakage; zero-init guidance ⇒ init L1=0 preserved.")
    p.add_argument("--no_repro_gate_align", action="store_true",
                   help="Design C: drop the T1/ASL gradient-alignment channel from the "
                        "reliability gate (use ASL reproducibility only).")
    p.add_argument("--lrda_coupling_gate", action="store_true",
                   help="T1–ASL agreement gate c∈(0,1) that scales the LRDA modulation "
                        "(B-gate + Δ-gate) by an EXPLICIT, content-safe coupling between the "
                        "frozen T1 and the DETACHED all-frame ASL aggregate (NO even/odd "
                        "split — uses every frame). No learnable ASL feature feeds c ⇒ no "
                        "feedback leakage; c scales zero-init modulation ⇒ init L1=0 holds, "
                        "and a wrong T1 collapses c ⇒ run-time self-suppression. Works on "
                        "--use_conv_encoder and --use_vmamba_encoder (--vmamba_lrda_inscan).")
    p.add_argument("--coupling_embed_dim", type=int, default=16,
                   help="Learned T1/ASL descriptor channels in the coupling gate (default 16).")
    # RKMR — replace AKMR's low-rank T1 key with a reproducible-ASL key (T1-free).
    p.add_argument("--akmr_repro_key", action="store_true",
                   help="Replace AKMR (Q=ASL,K=T1) with RKMR: Q,K from the reproducible "
                        "ASL descriptor φ (T1-free, even/odd split), V=ASL. Non-local "
                        "aggregation ALONG reproducible structure — helps detail; T1-free "
                        "⇒ contributes 0 to mismatched-T1 L1. T1 guidance stays in in-scan LRDA.")
    p.add_argument("--rkmr_t1_veto", action="store_true",
                   help="RKMR ablation: add a bounded low-rank T1 tissue-consistency VETO "
                        "(don't aggregate across tissue boundaries). Reintroduces bounded T1 "
                        "— re-verify the mismatched-T1 gate.")
    p.add_argument("--repro_desc_dim", type=int, default=8,
                   help="RKMR reproducible descriptor φ channel dim. Default 8.")
    # EC-LRDA: soft-PV bilinear conditioning + ASL-evidence calibration (VSS only, t1_task=seg).
    p.add_argument("--ec_lrda", action="store_true",
                   help="EC-LRDA: condition in-scan LRDA on soft-PV composition via a bilinear "
                        "s(PV)⊙a(ASL) adapter, calibrated by γ=r_rep·c_sem. Replaces the raw-T1-"
                        "feature φ conditioning + legacy repro/coupling gates. Needs t1_task=seg. "
                        "This is the MAIN-METHOD conditioning (ASL_dmvae/docs/cig_vss.md §3.4/§8), enabled by "
                        "the main launcher (env/hpc/slurm/submit_min.sh model 0). Kept opt-in (not an "
                        "argparse default) so non-EC baselines/backbones stay unaffected; OFF ⇒ the "
                        "−EC raw-T1-φ baseline. With the γ calibration on, set --w_rep/--w_sem>0.")
    p.add_argument("--ec_no_bilinear", action="store_true",
                   help="EC-LRDA ablation: keep γ calibration but use the plain φ(PV) conditioning "
                        "(no bilinear s⊙a interaction).")
    p.add_argument("--lrda_cond_src", choices=["pv", "fsl", "fsl_t1", "rawt1"], default="pv",
                   help="Source fed to the soft-PV bilinear s. 'fsl'=brain-masked FSL "
                        "GM/WM/CSF (3-ch, NO BG), supplied per batch from the precomputed "
                        "FAST segs (ASL-space, perfusion-independent, no stage-1 dep) — the "
                        "MAIN conditioning (2026-08-19). 'fsl_t1'=HYBRID 4-ch [GM,WM,CSF,T1] "
                        "(FSL PV + brain-masked raw T1 appended) — tissue composition PLUS raw "
                        "anatomy as the condition (still V=ASL: conditions B/Δ only). 'pv'=frozen "
                        "stage-1 soft-PV softmax (4-ch, ABLATION). 'rawt1'=raw T1 area-resized "
                        "into the SAME bilinear, 1-ch + geo, no gate, condition-source control. "
                        "STRUCTURAL (pv_ch differs 3/4/4/1 ⇒ new arch): train and eval must "
                        "match. 'fsl'/'fsl_t1' need gm/wm/csf in the batch; pair with --ec_lrda.")
    p.add_argument("--lrda_cond_deep", action="store_true",
                   help="tier-1: LEARNED CondPyramid (full-res stem + stride-2 downsamples, "
                        "per-stage zero-init head) replaces the shallow area-interp conditioning "
                        "pyramid. Deeper feature extraction + learned pooling for the bilinear s. "
                        "Justified by the geo-liveness probe (geo ACTIVE). V=ASL unchanged (α zero-"
                        "init); pv0 identity kept (bias-free ⇒ CondPyramid(0)=0). STRUCTURAL: "
                        "lrda_pv_ch := cond_deep_ch ⇒ new arch, train and eval must match. rawt1 stays shallow.")
    p.add_argument("--cond_deep_ch", type=int, default=16,
                   help="CondPyramid per-stage output channels feeding E_B (only with --lrda_cond_deep).")
    p.add_argument("--cond_deep_hidden", type=int, default=24,
                   help="CondPyramid stem/downsample hidden width (only with --lrda_cond_deep).")
    p.add_argument("--scan_dropout_p", type=float, default=0.0,
                   help="Batch-3: scan-direction dropout prob (drop each of the 4 SS2D scan directions "
                        "with this prob, keep ≥1). Train-time = mild regulariser; enables inference MC "
                        "uncertainty via model.set_mc_scan_dropout(True) + N passes (per-voxel std). Acts "
                        "on the ASL scan only ⇒ V=ASL. STRUCTURAL-neutral (no shape change) but trained-in.")
    p.add_argument("--lrda_tfdm", action="store_true",
                   help="Tissue-Factored Dynamics Modulation: a standalone conditioning structure that "
                        "replaces the anonymous bilinear/deep with K per-tissue operators W_k mapping ASL "
                        "content to a B/Δ modulation, ROUTED by the raw PV tissue fractions p_k "
                        "(m=Σ_k p_k·W_k·x_in). Interpretable (W_GM/W_WM/W_CSF), rank=K tissues. PRECLUDES "
                        "--lrda_cond_deep (needs raw tissue identity). V=ASL preserved. STRUCTURAL: train==eval.")
    p.add_argument("--dt_rank", type=int, default=1,
                   help="(3) SSM input-dependent Δ projection rank. 1 = legacy scalar (over-simplified, "
                        "starves the Δ selectivity + the a_Δ signal source); 0 = auto = ceil(d_model/16) "
                        "(standard Mamba, per-stage 2/4/8/16). STRUCTURAL: train and eval must match.")
    p.add_argument("--lrda_sig_xin", action="store_true",
                   help="(2) EC-LRDA signal branch reads the d_inner token feature x_in (richer, "
                        "all-ASL ⇒ no extra leakage; removes read-B-to-modulate-B self-reference) "
                        "instead of the thin d_state B_ matrix. STRUCTURAL: train and eval must match.")
    p.add_argument("--no_t1_branch", action="store_true",
                   help="Skip the T1 seg encoder/decoder entirely (fsl/rawt1 only): under those "
                        "sources the T1 seg branch is INERT (external/raw PV conditioning, T1-free "
                        "decoder, loss/brain mask = t1>thr, no gate). Drops the stage-1 dependency "
                        "(no --init_t1_from needed) and saves compute. STRUCTURAL: train and eval must match.")
    p.add_argument("--ec_no_calib", action="store_true",
                   help="EC-LRDA ablation A1: pure soft-PV bilinear conditioning, NO γ gate "
                        "(no r_rep/c_sem calibration).")
    p.add_argument("--ec_no_rep", action="store_true",
                   help="EC-LRDA ablation: drop the r_rep (ASL reproducibility) leg ⇒ γ=c_sem only "
                        "(T1-ASL semantic compatibility only).")
    p.add_argument("--ec_no_sg", action="store_true",
                   help="EC-LRDA ablation: let recon gradient flow into γ (default stop-grad so "
                        "r_rep/c_sem learn only from L_rep/L_sem).")
    p.add_argument("--ec_feat_dim", type=int, default=32,
                   help="EC guidance feature width (frame_enc / sem_feat). Default 32.")
    p.add_argument("--ec_work_div", type=int, default=4,
                   help="EC works at H/ec_work_div resolution (higher SNR + cheaper). Default 4.")
    p.add_argument("--ec_pv_geo", action="store_true",
                   help="EC-LRDA Level-1: PV-only stage-wise composition encoder. Drops the "
                        "redundant neighbourhood channel p̄ (corr(p,p̄)≈0.99+, scripts/"
                        "diag_pv_pyramid.py) and adds a zero-init depthwise-conv residual so "
                        "s(PV) sees local PV geometry a 1×1 map cannot. PV-only (no raw T1) ⇒ "
                        "V=ASL/isolation unchanged — re-run test_mismatched_t1.py to confirm. "
                        "Needs --ec_lrda (bilinear); changes arch ⇒ retrain (old EC ckpts differ).")
    p.add_argument("--ec_sem_bayes", action="store_true",
                   help="EC-LRDA: replace the plug-in cosine c_sem with a Bayesian-model-based PV linear-mixture "
                        "evidence gate — conjugate-posterior prototypes + a PLUG-IN point-estimate H1(PV-tissue)-vs-"
                        "H0(ASL-only null) score (not a calibrated Bayes factor). Shrinkage "
                        "prior stabilises rare classes; absolute (no Norm01) σ-standardised misfit ⇒ "
                        "local-anomaly gate for lesion preservation. Prototypes pure-ASL ⇒ V=ASL. The gate "
                        "is no-LOO. Needs --ec_lrda + --w_sem>0 (gated on E0.3; ASL_dmvae/docs/backlog_future_ideas.md §1).")
    p.add_argument("--ec_sem_bayes_no_loo", action="store_true",
                   help="DEPRECATED no-op (2026-08-05): the Bayesian gate is now always no-LOO, so this "
                        "flag has no effect. Kept only so older launch scripts do not error.")
    p.add_argument("--ec_sem_evidence", action="store_true",
                   help="Bayesian c_sem 'no-LOO gate + global evidence' mode: (i) forces the per-pixel "
                        "gate to the plain-residual (no-LOO) path; (ii) ALSO computes a per-image "
                        "closed-form log Bayes factor E=log p(F|H1)-log p(F|H0) with the Occam complexity "
                        "term — the whole-image T1/PV-trust signal the per-pixel gate self-normalises away "
                        "(wrong-subject/misregistration axis). E is DIAGNOSTIC (detached, exposed via the "
                        "ec out dict); it does NOT modify the gate. No new params ⇒ strict-load compatible. "
                        "Needs --ec_sem_bayes; forward-only toggle, must be replayed at eval.")
    p.add_argument("--pvc_residual", action="store_true",
                   help="PVC residual decomposition (design (d), arch level): assemble the output as "
                        "ŷ = E + w⊙r̂ AFTER the T1-free decoder. E=P·m̄ is a closed-form image-space "
                        "partial-volume base (per-image ridge fit of a per-tissue ΔM level to the input "
                        "aggregate, mixed back by the soft-PV composition); the decoder output is "
                        "reinterpreted as the residual r̂ (N2N on ŷ trains it so); w is a cross-frame "
                        "reproducibility gate that keeps reproducible deviations and suppresses single-"
                        "frame noise (counters near-MMSE anomaly over-smoothing). V=ASL preserved "
                        "(m̄ from ASL, E∈span(ASL); w only attenuates). Needs t1_task='seg'; opt-in ⇒ "
                        "baselines byte-unaffected. §3 note: E carries PV tissue boundaries into the "
                        "output (piecewise-const, no high-freq) — mismatch-T1-monitored relaxation.")
    p.add_argument("--pvc_no_wgate", action="store_true",
                   help="Ablation for --pvc_residual: drop the reproducibility gate (ŷ = E + r̂). "
                        "Isolates w's contribution to anomaly preservation (E0.3 retention).")
    p.add_argument("--dw_n2n", action="store_true",
                   help="Loss-level residual preservation (deviation-weighted N2N). Computes the "
                        "tissue expectation E=P·m̄ (fixed-ridge PVC) from the set_a aggregate + frozen "
                        "PV, forms the normalized deviation d_norm=|agg−E|∈[0,1], and (i) up-weights "
                        "the N2N L1 by (1+w_dev_up·d_norm) and (ii) relaxes the edge/grad smoothing by "
                        "(1−w_dev_relax·d_norm) at candidate anomalies. E NEVER enters the output — the "
                        "reconstruction stays pure ASL (strict V=ASL, no T1 texture), so this is the "
                        "safe alternative to the architecture-level --pvc_residual. d_norm is "
                        "set_a/PV-derived ⇒ ⊥ set_b ⇒ N2N-unbiased. PV source via --dw_pv_source "
                        "(default 'fsl' = precomputed gm/wm/csf; 'seg' needs t1_task='seg').")
    p.add_argument("--w_dev_up", type=float, default=1.0,
                   help="DW-N2N: N2N up-weight strength at deviations (weight = 1 + w_dev_up·d). "
                        "up=1.0 ⇒ suspected anomalies get at most 2x weight (design default). "
                        "Ablation w_dev_up=0 + w_dev_relax=0 recovers plain N2N.")
    p.add_argument("--w_dev_relax", type=float, default=0.0,
                   help="DW-N2N: edge/grad smoothing relaxation at deviations "
                        "(smoothing weight = 1 − w_dev_relax·d, clamped ≥0). Design: OFF by default; "
                        "used only as the 4th ablation arm (benefit direction uncertain).")
    p.add_argument("--dw_prior_json", type=str, default="",
                   help="DW-N2N: path to the dataset-level per-tissue prior means JSON "
                        "({'m0':[...],'tau':64}) from scripts/estimate_dw_prior.py. "
                        "Missing ⇒ legacy shrinkage (τ=0.5, m0=0) + a WARN.")
    p.add_argument("--dw_pv_source", type=str, default="fsl", choices=["fsl", "seg"],
                   help="DW-N2N: source of the partial-volume maps that define the tissue "
                        "expectation E=P·m̄. 'fsl' (default, RECOMMENDED) = the precomputed FSL PV "
                        "already loaded into the batch (gm_asl/wm_asl/csf_asl; ASL-space, "
                        "perfusion-independent, NO stage-1 dependency); BG:=clamp(1−GM−WM−CSF). "
                        "'seg' = softmax of the frozen stage-1 T1 seg head (needs t1_task='seg'). "
                        "PV here is TRAINING-ONLY (never at inference) ⇒ the FSL PV is strictly the "
                        "better estimate and lets DW run on arms without a T1 seg head. "
                        "See ASL_dmvae/docs/dwn2n_design.md §3.1.")
    p.add_argument("--w_rep", type=float, default=0.0,
                   help="Weight for L_rep = BCE(r_rep, 1-mean_k M_k) on injected artifact masks "
                        "(needs --patch_artifact_p>0). Supervises ASL reproducibility. Default 0.")
    p.add_argument("--w_sem", type=float, default=0.0,
                   help="Weight for L_sem = -log σ(c_match - c_mismatch), a match>mismatch ranking "
                        "on cross-subject mismatched PV. Supervises T1-ASL semantic compatibility. "
                        "Default 0.")
    p.add_argument("--slice_context", type=int, default=0,
                   help="2.5D: ASL input = 2*ctx+1 adjacent z-slices as channels (0 = 2D). "
                        "T1 stays 2D (center slice); output/target are the center slice.")
    p.add_argument("--cache_rate", type=float, default=1.0,
                   help="Fraction of subjects MONAI preloads into RAM at loader build "
                        "(default 1.0 = cache all, best for multi-epoch training). Set low "
                        "(e.g. 0.0) for a one-pass EVAL so the build does not preload the "
                        "whole cohort — eval touches only val once, so on-demand is faster.")
    # Reproducible-HF texture loss: reuse f(set_b) instead of a fresh pool split.
    p.add_argument("--tex_use_setb", action="store_true",
                   help="Texture loss: use f(set_a) vs stop-grad f(set_b) (reuses the "
                        "N2N a/b partition + the bdcyc forward) instead of a fresh pool "
                        "split. Cheaper + target-style (avoids the trivial-constant "
                        "attractor). Pair with --w_bdcyc 0 to drop the smoothing cycle.")
    p.add_argument("--t1_task", type=str, default="recon", choices=["seg", "recon"],
                   help="Frozen T1 branch pretext — MUST match the stage-1 --t1_task "
                        "used for --init_t1_from. 'recon' = T1 autoencoder (1-ch head, "
                        "appearance features, DEFAULT); 'seg' = 4-class PV segmentation. "
                        "Sizes the frozen t1_decoder head so it loads strict.")
    # DEFAULT is 'var' for new runs, but ASLT1Denoiser's kwarg default stays 'fra' so a
    # pre-2026-08-26 checkpoint (whose stored arch has no `aggregator` key) still rebuilds
    # with the transformer aggregator its weights belong to and loads strict.
    p.add_argument("--aggregator", type=str, default="var", choices=["fra", "var"],
                   help="Frame aggregator. 'var' (DEFAULT) = VarianceFrameAggregator: "
                        "BLUE weighting computed in closed form from the per-frame variance, "
                        "one learnable tau. 'fra' = the old FrameReliabilityAggregator "
                        "(per-frame CNN + 2-layer Transformer + log-var head, 86K params), "
                        "kept for the ablation. Probes show FRA learns exactly 'veto the bad "
                        "frame, 1/N on the rest'; measured on real frames against a trained "
                        "FRA, the closed form reproduces it (w_bad 0.004 vs 0.000 at sigma=1.5, "
                        "0.001 vs 0.000 at sigma=3.0; uniform would be 0.125).")
    p.add_argument("--agg_tau_init", type=float, default=1.0,
                   help="Initial tau for --aggregator var. tau=1 is exact BLUE (weight "
                        "proportional to 1/variance); tau near 0 degenerates to the plain "
                        "uniform mean. It is learnable, so its trajectory reads out how much "
                        "variance weighting the data actually wanted.")
    p.add_argument("--window_fusion_levels", type=int, default=0, choices=[0, 1, 2],
                   help="Multi-scale window cross-fusion on the decoder's fine scales, "
                        "counted from the finest down. 0 = off (modules are not built; "
                        "bit-exact to the old arch). 1 = 128x128 only. 2 = 64x64 + 128x128. "
                        "Q=ASL K=T1 V=ASL-unprojected, so the fused output is a convex "
                        "combination of ASL values. See docs/multiscale_window_design.md.")
    p.add_argument("--window_size", type=int, default=8,
                   help="Attention window ws for --window_fusion_levels. Cost per level is "
                        "(H*W/ws^2) windows x ws^4 pairs; ws=8 at 128x128 costs the same as "
                        "the existing 32x32 global CMF1.")
    p.add_argument("--window_heads", type=int, default=4,
                   help="Attention heads in the window fusion (halved until it divides the "
                        "channel count).")
    p.add_argument("--window_gate_init", type=float, default=-3.0,
                   help="Gate logit init; g = sigmoid(a). -3.0 => g ~ 0.047 (~2 pct feature "
                        "perturbation at init). Do NOT set a clamped 0 gate: the attention "
                        "params would receive exactly zero gradient and never train.")
    p.add_argument("--window_k_source", type=str, default="t1", choices=["t1", "asl"],
                   help="Attention KEY source. 't1' = anatomy-grouped cross-attention. "
                        "'asl' = self-attention control: the decoder stays T1-free and the "
                        "run answers 'is the gain from anatomy, or just from multi-scale "
                        "non-local averaging?'. Same module, same params either way.")
    p.add_argument("--keep_t1_decoder", action="store_true",
                   help="Keep the T1 decoder head under --t1_task recon. By DEFAULT it is "
                        "dropped (0.67M params, ~16 pct of the model): with w_anat_roi=0 it "
                        "gets zero gradient and its output is unused, while the T1 encoder "
                        "still feeds CMF0/CMF1. Auto-kept when --t1_task seg or "
                        "w_anat_roi>0. Set this only to restore the T1-recon val panel.")
    # v42 sub-components (default ON when --use_mossm_encoder; can be disabled
    # individually for ablation studies — see ASL_dmvae/docs/v42_design_rationale.md)
    p.add_argument("--no_tabs", action="store_true",
                   help="Ablation: disable Tissue-Aware Bidirectional Scan inside "
                        "MoSSM. Falls back to plain row-major scan order. v42.")
    p.add_argument("--no_blue_attn", action="store_true",
                   help="DEPRECATED no-op (BLUE-Attn / noise-var removed 2026-06-21).")
    p.add_argument("--no_cmam", action="store_true",
                   help="Ablation: disable Cross-Modal Attentive Memory inside "
                        "MoSSM blocks. Reverts to v42's A.1 fallback: plain MoSSM "
                        "encoder + bottleneck CrossModalFusion (ASL self-attn). v42.")
    p.add_argument("--cmam_max_tokens", type=int, default=1024,
                   help="Skip CMAM at MoSSM stages where H*W exceeds this token "
                        "count, to bound attention's O(N^2) memory. Default 1024 "
                        "= 32x32, enables CMAM at the deepest 1-2 stages. v42.")
    p.add_argument("--cmam_n_heads", type=int, default=4,
                   help="Number of attention heads inside CMAM. Default 4. v42.")

    # 2026-07 Reproducible-HF texture loss (docs/validation_metrics.md). Real
    # texture = high-frequency structure REPRODUCIBLE across two disjoint frame
    # subsets. Reward reproducible HF power in tissue, penalise it in CSF (ΔM≈0 ⇒
    # any reproducible HF there is common-mode artifact). Training-time analogue of
    # the tissue/CSF HF-ratio + split-half consistency eval metrics. Default OFF.
    p.add_argument("--w_tex", type=float, default=0.0,
                   help="Weight of the split-half reproducible-HF-power REWARD in "
                        "tissue (GM∪WM). 0 = off. Keep SMALL (nudge, not driver): the "
                        "power form is unbiased to frame-independent noise but rewards "
                        "frame-CORRELATED artifacts as texture — anchored by N2N + CSF "
                        "penalty + mismatched-T1 gate.")
    p.add_argument("--w_tex_csf", type=float, default=0.0,
                   help="Weight of the reproducible-HF PENALTY in CSF (common-mode "
                        "artifact suppression). 0 = off. Recommend setting alongside "
                        "--w_tex, similar magnitude.")
    p.add_argument("--tex_min_pool", type=int, default=4,
                   help="Skip the texture loss on a batch whose smallest frame pool "
                        "(len_a+len_b) is below this, so each disjoint half has >=2 "
                        "frames. Default 4.")

    # Bad-frame injection (forces aggregator to learn non-uniform weights)
    p.add_argument("--bad_frame_p", type=float, default=0.3,
                   help="Per-step probability of corrupting one frame in setA")
    p.add_argument("--bad_frame_noise_min", type=float, default=1.5)
    p.add_argument("--bad_frame_noise_max", type=float, default=3.0)

    # Phase 2 Run B' (2026-05-23): Local patch artifact injection. Unlike
    # bad_frame (whole-frame Gaussian noise), this injects high-sigma noise into
    # a single rectangular patch of one set_a frame. Required for diagnosing
    # RGSF: whole-frame corruption leaves Var_T(set_a) spatially uniform, so
    # RGSF gate has no spatial signal to learn against. Local patches create the
    # heterogeneous noise_var map RGSF was designed for.
    p.add_argument("--patch_artifact_p", type=float, default=0.0,
                   help="Per-step probability of injecting a local patch artifact "
                        "(in set_a only). 0 disables. Recommended 0.5 for RGSF Run B'.")
    p.add_argument("--patch_artifact_size", type=int, default=16,
                   help="Patch side length in pixels. Default 16 (i.e. 16x16 region).")
    p.add_argument("--patch_artifact_noise_min", type=float, default=1.0,
                   help="Min noise std for the patch artifact.")
    p.add_argument("--patch_artifact_noise_max", type=float, default=2.5,
                   help="Max noise std for the patch artifact.")
    # --- EC r_rep supervision strength (2026-07-26) -----------------------------
    # The r_rep leg under-trained because a single-frame artifact gives a per-pixel
    # corrupt fraction ~1/T, so L_rep's target r*=1-frac ~0.92 — near-1, learnable
    # with a ~flat slope (softplus(beta_rep) stayed ~0.028). These three raise the
    # corrupt fraction AND sharpen the target so r_rep must grow a real slope. All
    # default to the LEGACY behaviour (1 frame, honest 1-frac target).
    p.add_argument("--patch_artifact_frames_max", type=int, default=1,
                   help="Max #frames corrupted at the injected patch; k~U[1,this] VALID "
                        "frames get the same patch. >1 raises the per-pixel corrupt fraction "
                        "so L_rep's target actually drops. Default 1 = legacy single-frame.")
    p.add_argument("--rep_target_gain", type=float, default=1.0,
                   help="L_rep target sharpening: r*=(1-gain*corrupt_frac).clamp(rep_target_floor,1). "
                        "gain>1 drives r_rep low even for a small corrupt fraction (real dynamic "
                        "range, forces the slope to grow). Default 1.0 = honest clean-fraction.")
    p.add_argument("--rep_target_floor", type=float, default=0.0,
                   help="Lower clamp for the L_rep target r* (default 0).")

    # L_sharp hinge (Plan B, 2026-05-22). Counteracts over-smooth collapse seen
    # in run_full_cig_net where lapvR plateaued at 0.33 < τS=0.35 with steadily
    # decreasing psnr_ref (-2.5 dB over 250 steps). w_sharp=0 disables.
    p.add_argument("--w_sharp", type=float, default=0.0,
                   help="Weight on lapvar-ratio hinge loss. 0 disables; "
                        "0.05 recommended with tau_sharp_train ≈ τS used for selection.")
    p.add_argument("--tau_sharp_train", type=float, default=0.35,
                   help="Training-time lapvar-ratio hinge threshold. Below this, "
                        "L_sharp = (τ_sharp − lapvR)² applies. Match to "
                        "--constrained_rho_lapvar for consistency.")

    # LR scheduler
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "none"],
                   help="LR scheduler type (cosine=CosineAnnealingLR, none=constant)")
    p.add_argument("--lr_min", type=float, default=1e-5,
                   help="Minimum LR for cosine scheduler (eta_min)")
    # v42h (2026-05-19): linear LR warmup + grad-clip CLI
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Linear LR warmup over first N outer steps (0 = no warmup).")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Max grad-norm for clip_grad_norm_ (default 1.0).")
    # --- 2026-08-07 additions: spatial-flip aug + AMP + in-loop per-metric-best ---
    p.add_argument("--flip_p", type=float, default=0.0,
                   help="Per-axis prob of a random spatial flip (train-only aug), applied "
                        "CONSISTENTLY to setA/setB/T1/PV so the N2N target stays aligned and "
                        "the T1<->ASL correspondence is preserved (N2N-unbiased: linear). 0=off.")
    p.add_argument("--amp", action="store_true",
                   help="Enable bf16 autocast around the train forward+loss (~1.5-2x faster). "
                        "bf16 (not fp16) keeps fp32's exponent range so no GradScaler is needed.")
    p.add_argument("--save_per_metric_best", action="store_true",
                   help="At each eval (after best_min_step), keep the best EMA ckpt for EVERY "
                        "self-supervised val metric: writes best_<metric>.pth + best_metrics.json. "
                        "Independent of --best_criterion (recoverability only, never selection).")
    p.add_argument("--adam_beta1", type=float, default=0.9,
                   help="AdamW beta1 (default 0.9).")
    p.add_argument("--adam_beta2", type=float, default=0.999,
                   help="AdamW beta2 (default 0.999). Lower (e.g. 0.99) decays the "
                        "second-moment estimate over ~100 steps instead of ~1000 — "
                        "more appropriate for short (<=500-step) runs.")

    # Stage-2: load pretrained T1 branch + optionally freeze it
    p.add_argument("--init_t1_from", type=str, default=None,
                   help="Path to T1Branch ckpt (from runners/train_t1.py); load t1_encoder + t1_decoder weights into ASLT1Denoiser")
    p.add_argument("--freeze_t1", action="store_true",
                   help="Freeze t1_encoder + t1_decoder during training (use with --init_t1_from)")
    p.add_argument("--premask_asl_inputs", action="store_true",
                   help="Legacy: pre-multiply setA/setB by the (seg-derived) brain mask "
                        "before the model. OFF by default — brain masking now lives only in "
                        "the loss (t1>thr) and at output/eval, which removes a train/infer "
                        "consistency footgun and an extra frozen-T1 forward. Re-enable only "
                        "to reproduce the old input-masking behaviour.")
    p.add_argument("--max_steps", type=int, default=0,
                   help="Override config max_steps. 0 = use config value.")
    p.add_argument("--eval_every", type=int, default=0,
                   help="Override config eval_every (EPOCHS between validations; 1 step = 1 epoch). "
                        "0 = use config value. The config value is calibrated against a given "
                        "max_steps: if the cohort size changes, one epoch changes size, max_steps "
                        "must be rescaled, and eval_every must be rescaled with it or the run "
                        "validates too few times for best-ckpt selection and the score EMA.")
    p.add_argument("--best_criterion", type=str, default="scov_gm",
                   choices=["upsnr_cyc", "umse", "constrained_umse", "scov_gm", "none"],
                   help="Best ckpt selection (all self-supervised, no biased reference). "
                        "'none' disables best selection AND early stopping; trains to full "
                        "max_steps and relies on latest.pth + swa.pth at end. "
                        "umse: pure argmin uMSE (Marcos-Morales ICML 2023). No anti-collapse "
                        "floor needed — uMSE penalises BOTH noise (variance) and over-smoothing "
                        "(bias), so it self-corrects. sCoV/lapvR reported alongside; final-model "
                        "sCoV folding happens post-hoc via the uMSE 1-SE feasibility set. "
                        "scov_gm: GM sCoV (Wang 2003 within-tissue homogeneity; lower = "
                        "cleaner), pure no-reference. Optionally constrained by "
                        "--select_lapvar_floor to avoid selecting over-smooth ckpts. "
                        "constrained_umse: argmin uMSE s.t. cyc/lapvar constraints. "
                        "upsnr_cyc: uPSNR (Marcos-Morales ICML 2023) − α · subset_consistency; "
                        "α via --upsnr_cyc_alpha (default 30 puts cyc and uPSNR on equal footing).")
    p.add_argument("--select_lapvar_floor", type=float, default=0.0,
                   help="lapvar-ratio floor for scov_gm selection. Ckpts with lapvR < floor "
                        "are infeasible (excluded), preventing sCoV-min from collapsing to the "
                        "blurriest/over-smooth ckpt. 0 = no floor. Recommended ~0.6.")
    p.add_argument("--upsnr_cyc_alpha", type=float, default=30.0,
                   help="Weight on subset_consistency penalty for upsnr_cyc criterion. "
                        "score = uPSNR − α · cyc. α=30 ≈ equal-std-error scaling on this dataset.")
    # v42k (2026-05-20): constrained-uMSE selection — replaces the ad-hoc linear
    # combination `uPSNR − α · cyc` with a principled ε-constraint formulation:
    #   θ* = argmin uMSE(θ)
    #         s.t. cyc(θ)         ≤ ρ_cyc · noise_floor_L1     (anti-mimicry)
    #              lapvar_ratio(θ) ≥ ρ_lapvar                  (anti-collapse)
    # noise_floor_L1 = E[|b − c|] from the 3-way disjoint split of set_b (the
    # held-out noisy reference repeatability — natural data-driven τ_C). The
    # anti-collapse lower bound prevents the trivial 'output brain-mask mean'
    # local minimum we observed in v42k-rev1.
    p.add_argument("--constrained_rho_cyc", type=float, default=1.0,
                   help="ρ_cyc: τ_C = ρ_cyc · noise_floor_L1. 1.0 = model may be "
                        "at most as inconsistent as the held-out noisy reference itself.")
    p.add_argument("--constrained_rho_lapvar", type=float, default=0.5,
                   help="ρ_lapvar: τ_S lower bound on lapvar(recon) / lapvar(ref). "
                        "0.5 = recon must retain at least 50%% of the high-freq "
                        "content of the noisy reference (anti-over-smooth).")
    p.add_argument("--best_min_step", type=int, default=-1,
                   help="Minimum global_step before best is allowed to update. -1 (default) = "
                        "use sure_anneal_start from config (so best is gated until SURE has "
                        "fully ramped). Set to 0 to disable the gate.")
    p.add_argument("--score_ema_alpha", type=float, default=0.0,
                   help="Exponential moving average smoothing on best-selection score. "
                        "0.0 (default) = no smoothing (raw per-validation score used). "
                        "α ∈ (0, 1] gives ema_score = (1-α)·prev + α·current. "
                        "Recommended 0.3 (≈5-validation window) for noisy-target self-supervised setups.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Fallback EMA (used if utils.training_utils.EMAModel is unavailable)
# ---------------------------------------------------------------------------

class _FallbackEMAModel:
    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        max_value: float = 0.9999,
        device: Optional[torch.device] = None,
    ) -> None:
        del inv_gamma, power, min_value
        self.device = device
        self.update_after_step = int(update_after_step)
        self.max_value = float(max_value)
        self.optimization_step = 0
        self.ema_state = {
            k: v.detach().clone().to(device=device) if device is not None else v.detach().clone()
            for k, v in self._state_dict(model).items()
        }
        self._backup_state: Optional[Dict[str, Tensor]] = None

    def _state_dict(self, model: nn.Module) -> Dict[str, Tensor]:
        base = model.module if isinstance(model, nn.DataParallel) else model
        return base.state_dict()

    def _decay(self) -> float:
        if self.optimization_step <= self.update_after_step:
            return 0.0
        return self.max_value

    @torch.no_grad()
    def step(self, model: nn.Module):
        self.optimization_step += 1
        decay = self._decay()
        src = self._state_dict(model)
        for k, v in src.items():
            target = self.ema_state[k]
            # Integer/bool buffers (e.g. SwinIR window-attention `relative_position_index`,
            # attention masks, BatchNorm `num_batches_tracked`) are not EMA-able — a float
            # `mul_(decay)` on a Long tensor raises "result type Float can't be cast to Long".
            # Copy the current value verbatim (these are constant lookups / counters, so the
            # EMA state just mirrors them). The main VSS model has no such buffers → unaffected.
            if not torch.is_floating_point(target):
                target.copy_(v.detach().to(target.device))
                continue
            target.mul_(decay).add_(v.detach().to(target.device), alpha=1.0 - decay)

    @torch.no_grad()
    def store(self, model: nn.Module):
        self._backup_state = {k: v.detach().clone() for k, v in self._state_dict(model).items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        base = model.module if isinstance(model, nn.DataParallel) else model
        state = {k: v.detach().clone().to(base.state_dict()[k].device) for k, v in self.ema_state.items()}
        base.load_state_dict(state, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup_state is None:
            return
        base = model.module if isinstance(model, nn.DataParallel) else model
        base.load_state_dict(self._backup_state, strict=True)
        self._backup_state = None


# ---------------------------------------------------------------------------
# Differentiable per-sample CNR (for the cnr_cyc consistency loss)
# ---------------------------------------------------------------------------

def _cnr_per_sample(
    image: Tensor,
    gm_mask: Tensor,
    wm_mask: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    min_vox: float = 20.0,
) -> Tuple[Tensor, Tensor]:
    """Differentiable CNR = |mu_GM - mu_WM| / sigma_WM, computed per image.

    Matches utils.metrics.cnr but (a) keeps the graph (no .item()), (b) returns
    one CNR per batch element so the consistency loss gets a dense gradient, and
    (c) floors sigma_WM and skips slices with too few GM/WM voxels to avoid the
    small-denominator gradient blow-up flagged in design review.

    Returns ([B] cnr, [B] valid-mask in {0,1}). The mask threshold acts on the
    (constant) PV maps, not the image, so the image gradient path stays smooth.
    """
    gm = (gm_mask > threshold).float().flatten(1)   # [B, HW]
    wm = (wm_mask > threshold).float().flatten(1)
    img = image.flatten(1)                          # [B, HW]
    n_gm = gm.sum(1)                                 # [B]
    n_wm = wm.sum(1)
    mu_gm = (img * gm).sum(1) / n_gm.clamp_min(1.0)
    mu_wm = (img * wm).sum(1) / n_wm.clamp_min(1.0)
    var_wm = (((img - mu_wm.unsqueeze(1)) ** 2) * wm).sum(1) / n_wm.clamp_min(1.0)
    sigma_wm = var_wm.clamp_min(0.0).sqrt().clamp_min(eps)
    cnr = (mu_gm - mu_wm).abs() / sigma_wm
    valid = ((n_gm > min_vox) & (n_wm > min_vox)).float()
    return cnr, valid


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _pad_sequence_tensors(items: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    if len(items) == 0:
        raise ValueError("Empty tensor list cannot be padded.")
    lengths = torch.tensor([int(x.shape[0]) for x in items], dtype=torch.long)
    t_max = int(lengths.max().item())
    sample = items[0]
    if sample.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got {tuple(sample.shape)}")
    b = len(items)
    _, c, h, w = sample.shape
    out = sample.new_zeros((b, t_max, c, h, w))
    for i, x in enumerate(items):
        out[i, : x.shape[0]] = x
    return out, lengths


def _ensure_batched_frames(x: TensorLike, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    if isinstance(x, (list, tuple)):
        return _pad_sequence_tensors(x)
    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported frame container type: {type(x)!r}")
    if x.ndim == 5:
        b, t = x.shape[:2]
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
        return x, lengths
    if x.ndim == 4:
        frames = x.unsqueeze(0)
        lengths = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        return frames, lengths
    raise ValueError(f"Expected 4D/5D tensor or list of 4D tensors, got {tuple(x.shape)}")


def _ensure_image_batch(x: TensorLike) -> Tensor:
    if isinstance(x, (list, tuple)):
        x = torch.stack(list(x), dim=0)
    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported image container type: {type(x)!r}")
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
    return x


def _extract_lengths(batch: Dict[str, TensorLike], aliases: Sequence[str]) -> Optional[Tensor]:
    for key in aliases:
        if key in batch and torch.is_tensor(batch[key]):
            return batch[key]  # type: ignore[return-value]
    return None


def _lengths_from_mask(mask: Optional[TensorLike]) -> Optional[Tensor]:
    if mask is None:
        return None
    if isinstance(mask, (list, tuple)):
        return torch.tensor([int(m.shape[0]) for m in mask], dtype=torch.long)
    if not torch.is_tensor(mask):
        return None
    if mask.ndim == 1:
        return mask.long()
    dims = tuple(range(2, mask.ndim)) if mask.ndim >= 3 else ()
    valid = ~mask.bool() if mask.dtype == torch.bool else (mask > 0)
    if dims:
        valid = valid.any(dim=dims)
    return valid.long().sum(dim=1)


def prepare_asl_pair_batch(batch: Dict[str, TensorLike], device: torch.device) -> Dict[str, Tensor]:
    len_a = _extract_lengths(batch, ["lenA", "setA_len", "setA_lengths", "lengthsA", "lengths_setA"])
    len_b = _extract_lengths(batch, ["lenB", "setB_len", "setB_lengths", "lengthsB", "lengths_setB"])

    if len_a is None and "maskA" in batch:
        len_a = _lengths_from_mask(batch.get("maskA"))
    if len_b is None and "maskB" in batch:
        len_b = _lengths_from_mask(batch.get("maskB"))

    set_a, len_a = _ensure_batched_frames(batch["setA"], len_a)
    set_b, len_b = _ensure_batched_frames(batch["setB"], len_b)
    t1 = _ensure_image_batch(batch["t1"])

    pack: Dict[str, Tensor] = {
        "setA": set_a.to(device=device, dtype=torch.float32, non_blocking=True),
        "setB": set_b.to(device=device, dtype=torch.float32, non_blocking=True),
        "lenA": len_a.to(device=device, dtype=torch.long, non_blocking=True),
        "lenB": len_b.to(device=device, dtype=torch.long, non_blocking=True),
        "t1": t1.to(device=device, dtype=torch.float32, non_blocking=True),
    }
    if "maskA" in batch and torch.is_tensor(batch["maskA"]):
        pack["maskA"] = batch["maskA"].to(device=device, non_blocking=True)
    if "maskB" in batch and torch.is_tensor(batch["maskB"]):
        pack["maskB"] = batch["maskB"].to(device=device, non_blocking=True)
    # Optional partial-volume maps for multi-task T1 segmentation + ASL contrast loss
    for k in ("gm", "wm", "csf"):
        if k in batch:
            pack[k] = _ensure_image_batch(batch[k]).to(device=device, dtype=torch.float32, non_blocking=True)
    return pack


@torch.no_grad()
def direct_mean_from_frames(frames: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
    if lengths is None:
        return frames.mean(dim=1)
    b, t = frames.shape[:2]
    mask = torch.arange(t, device=frames.device).unsqueeze(0) < lengths.view(b, 1)
    mask = mask[:, :, None, None, None].float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (frames * mask).sum(dim=1) / denom


# ---------------------------------------------------------------------------
# PSNR / SSIM helpers (torchmetrics preferred, manual fallback)
# ---------------------------------------------------------------------------

def _compute_psnr_ssim(pred: Tensor, target: Tensor):
    """Return (psnr_float, ssim_float) for a [B,C,H,W] pair in [0,1]."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    try:
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio as _psnr,
            structural_similarity_index_measure as _ssim,
        )
        psnr = float(_psnr(pred, target, data_range=1.0).item())
        ssim = float(_ssim(pred, target, data_range=1.0).item())
        return psnr, ssim
    except Exception:
        pass

    # Manual fallback
    mse = F.mse_loss(pred, target).item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    # Simple SSIM approximation via luminance + contrast terms
    mu_p = pred.mean().item()
    mu_t = target.mean().item()
    sig_p = pred.var().item() ** 0.5
    sig_t = target.var().item() ** 0.5
    sig_pt = ((pred - mu_p) * (target - mu_t)).mean().item()
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)) / \
           ((mu_p ** 2 + mu_t ** 2 + c1) * (sig_p ** 2 + sig_t ** 2 + c2))
    return psnr, ssim


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, args):
        self.args = args
        self.cfg = Config(args.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.log_dir = os.path.join(args.exp, "logs", args.name)
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.val_img_dir = os.path.join(self.log_dir, "val_images")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if args.save_images:
            os.makedirs(self.val_img_dir, exist_ok=True)

        self.loaders = get_asl_2d_loaders(
            self.cfg,
            modes=["train", "val"],
            asl_hw=self.cfg.asl_denoiser_train_params.asl_hw,
            asl_z=self.cfg.asl_denoiser_train_params.asl_z,
            t1_hw=self.cfg.asl_denoiser_train_params.t1_hw,
            t1_z=self.cfg.asl_denoiser_train_params.t1_z,
            num_workers=self.cfg.data_loading.num_workers,   # now actually honoured (Linux); Windows forced to 0
            slice_context=int(getattr(args, "slice_context", 0)),   # 2.5D
            cache_rate=float(getattr(args, "cache_rate", 1.0)),     # eval can set 0 to skip preloading the cohort
        )
        self.train_loader = self.loaders["train"]
        self.val_loader = self.loaders["val"]

        # --- RETIRED 2026-07-16: MoSSM / CIG-Net-v2 backbone + NAFDecoder (CORD) ---
        # The main method is CIG-VSS + EC-LRDA (--use_vmamba_encoder ... --ec_lrda) with a
        # plain T1-free ConvDecoderWithSkips2D decoder (ASL_dmvae/docs/cig_vss.md). The MoSSM encoder,
        # the v2 CIGNet, and the NAFDecoder/CORD decoder are no longer part of any runnable
        # path; their code is retained (marked RETIRED) only so old v2 ckpts remain readable
        # from git history. Fail loudly rather than silently building a retired arch.
        if bool(getattr(args, "use_mossm_encoder", False)) or bool(getattr(args, "use_naf_fusion", False)):
            raise SystemExit(
                "--use_mossm_encoder / --use_naf_fusion (MoSSM/CIG-Net-v2 + NAFDecoder/CORD) "
                "were RETIRED 2026-07-16. Use the main line: CIG-VSS + EC-LRDA "
                "(--use_vmamba_encoder --vmamba_convffn --vmamba_overlap_merge "
                "--vmamba_lrda_inscan --ec_lrda ...) with the plain ConvDecoderWithSkips2D. "
                "To re-run the v2 line, restore it from git history before this commit.")

        tp = self.cfg.asl_denoiser_train_params
        # No-seg arm (2026-08-25): build the T1 decoder head only if something actually
        # consumes it — seg logits (CMF tissue bias / loss_seg / soft pre-mask), a
        # non-zero w_anat_roi T1-recon term, or an explicit --keep_t1_decoder.
        _t1_task = str(getattr(args, "t1_task", "seg"))
        _keep_t1_dec = (_t1_task == "seg"
                        or float(getattr(tp, "w_anat_roi", 0.0)) > 0.0
                        or bool(getattr(args, "keep_t1_decoder", False)))
        if not _keep_t1_dec:
            logging.info("[arch] t1_task=recon + w_anat_roi=0 -> T1 decoder head DROPPED "
                         "(dead weight: zero grad, output unused). T1 encoder kept as the "
                         "CMF K-path. --keep_t1_decoder restores it.")
        denoiser_kwargs = dict(
            asl_hw=int(tp.asl_hw),
            t1_hw=int(tp.t1_hw),
            in_ch=1,
            base_ch=int(args.base_ch),
            depth=int(args.depth),
            use_wavelet=bool(getattr(args, "use_wavelet", False)),
            use_t1_cross_fusion=bool(getattr(args, "use_t1_cross_fusion", False)),
            t1_attn_max_tokens=int(getattr(args, "t1_attn_max_tokens", 1024)),
            use_svfw=bool(getattr(args, "use_svfw", False)),
            use_film=bool(getattr(args, "use_film", False)),
            use_heteroscedastic=bool(getattr(args, "use_heteroscedastic", False)),
            use_mossm_encoder=bool(getattr(args, "use_mossm_encoder", False)),
            mossm_blocks_per_scale=int(getattr(args, "mossm_blocks_per_scale", 2)),
            mossm_n_directions=(
                [int(v) for v in str(getattr(args, "mossm_n_directions_by_stage", "") or "").split(",") if v.strip()]
                if str(getattr(args, "mossm_n_directions_by_stage", "") or "").strip()
                else int(getattr(args, "mossm_n_directions", 1))
            ),
            use_hybrid_mossm_block=bool(getattr(args, "use_hybrid_mossm_block", False)),
            hybrid_local_expand=float(getattr(args, "hybrid_local_expand", 2.0)),
            hybrid_local_kernel=int(getattr(args, "hybrid_local_kernel", 3)),
            mossm_d_state=int(getattr(args, "mossm_d_state", 16)),
            use_tabs=not bool(getattr(args, "no_tabs", False)),
            use_cmam=not bool(getattr(args, "no_cmam", False)),
            cmam_max_tokens=int(getattr(args, "cmam_max_tokens", 1024)),
            cmam_n_heads=int(getattr(args, "cmam_n_heads", 4)),
            use_reshead=bool(getattr(args, "use_reshead", False)),
            reshead_delta_max=float(getattr(args, "reshead_delta_max", 0.5)),
            mossm_t1_gated=bool(getattr(args, "mossm_t1_gated", False)),
            mossm_t1_gate_init=float(getattr(args, "mossm_t1_gate_init", -4.0)),
            cmam_k_rank_div=int(getattr(args, "cmam_k_rank_div", 1)),
            use_naf_fusion=bool(getattr(args, "use_naf_fusion", False)),
            use_mdta_fusion=bool(getattr(args, "use_mdta_fusion", False)),
            use_cada=bool(getattr(args, "use_cada", False)),
            cada_active_stages=_parse_stage_mask(
                getattr(args, "cada_stages", ""), int(args.depth),
                flag_name="--cada_stages",
            ),
            cada_n_groups_B=int(getattr(args, "cada_n_groups_B", 4)),
            cada_lr=bool(getattr(args, "cada_lr", False)),
            cada_lr_rank=int(getattr(args, "cada_lr_rank", 8)),
            cada_lr_bound=float(getattr(args, "cada_lr_bound", 4.0)),
            use_conv_encoder=bool(getattr(args, "use_conv_encoder", False)),
            lrda_rank=int(getattr(args, "lrda_rank", 8)),
            lrda_bound=float(getattr(args, "lrda_bound", 4.0)),
            lrda_variant=str(getattr(args, "lrda_variant", "c")),
            conv_lrda_stages=(
                [int(v) for v in str(getattr(args, "conv_lrda_stages", "") or "").split(",") if v.strip()]
                or None
            ),
            # The reader is enabled by --use_akmr (T1-keyed AKMR) OR --akmr_repro_key
            # (T1-free RKMR — it brings its own reader); --no_akmr force-disables either.
            conv_use_akmr=(bool(getattr(args, "use_akmr", False)) or bool(getattr(args, "akmr_repro_key", False)))
                          and not bool(getattr(args, "no_akmr", False)),
            akmr_k_rank_div=int(getattr(args, "akmr_k_rank_div", 4)),
            use_vmamba_encoder=bool(getattr(args, "use_vmamba_encoder", False)),
            vmamba_no_guard=bool(getattr(args, "vmamba_no_guard", False)),
            vmamba_d_state=int(getattr(args, "vmamba_d_state", 16)),
            vmamba_n_directions=int(getattr(args, "vmamba_n_directions", 2)),
            vmamba_hv_scan=bool(getattr(args, "vmamba_hv_scan", False)),
            vmamba_blocks_per_scale=int(getattr(args, "vmamba_blocks_per_scale", 2)),
            vmamba_convffn=bool(getattr(args, "vmamba_convffn", False)),
            vmamba_overlap_merge=bool(getattr(args, "vmamba_overlap_merge", False)),
            vmamba_lrda_inscan=bool(getattr(args, "vmamba_lrda_inscan", False)),
            lrda_cond_asl=bool(getattr(args, "lrda_cond_asl", False)),
            lrda_dt_bound=float(getattr(args, "lrda_dt_bound", 1.0)),
            lrda_repro_gate=bool(getattr(args, "lrda_repro_gate", False)),
            repro_gate_align=not bool(getattr(args, "no_repro_gate_align", False)),
            lrda_coupling_gate=bool(getattr(args, "lrda_coupling_gate", False)),
            coupling_embed_dim=int(getattr(args, "coupling_embed_dim", 16)),
            akmr_repro_key=bool(getattr(args, "akmr_repro_key", False)),
            rkmr_t1_veto=bool(getattr(args, "rkmr_t1_veto", False)),
            repro_desc_dim=int(getattr(args, "repro_desc_dim", 8)),
            slice_context=int(getattr(args, "slice_context", 0)),
            t1_task=str(getattr(args, "t1_task", "seg")),
            use_t1_decoder=bool(_keep_t1_dec),
            window_fusion_levels=int(getattr(args, "window_fusion_levels", 0)),
            window_size=int(getattr(args, "window_size", 8)),
            window_heads=int(getattr(args, "window_heads", 4)),
            window_gate_init=float(getattr(args, "window_gate_init", -3.0)),
            window_k_source=str(getattr(args, "window_k_source", "t1")),
            aggregator=str(getattr(args, "aggregator", "fra")),
            agg_tau_init=float(getattr(args, "agg_tau_init", 1.0)),
            zero_t1=bool(getattr(args, "zero_t1", False)),
            pv_mode=str(getattr(args, "pv_mode", "real")),
            naive_t1_concat=bool(getattr(args, "naive_t1_concat", False)),
            ec_lrda=bool(getattr(args, "ec_lrda", False)),
            ec_bilinear=not bool(getattr(args, "ec_no_bilinear", False)),
            lrda_cond_src=str(getattr(args, "lrda_cond_src", "pv")),
            lrda_cond_deep=bool(getattr(args, "lrda_cond_deep", False)),
            cond_deep_ch=int(getattr(args, "cond_deep_ch", 16)),
            cond_deep_hidden=int(getattr(args, "cond_deep_hidden", 24)),
            no_t1_branch=bool(getattr(args, "no_t1_branch", False)),
            lrda_sig_xin=bool(getattr(args, "lrda_sig_xin", False)),
            dt_rank=int(getattr(args, "dt_rank", 1)),
            lrda_tfdm=bool(getattr(args, "lrda_tfdm", False)),
            scan_dropout_p=float(getattr(args, "scan_dropout_p", 0.0)),
            ec_calibrate=not bool(getattr(args, "ec_no_calib", False)),
            ec_use_rep=not bool(getattr(args, "ec_no_rep", False)),
            ec_sg_gamma=not bool(getattr(args, "ec_no_sg", False)),
            ec_feat_dim=int(getattr(args, "ec_feat_dim", 32)),
            ec_work_div=int(getattr(args, "ec_work_div", 4)),
            ec_pv_geo=bool(getattr(args, "ec_pv_geo", False)),
            ec_sem_bayes=bool(getattr(args, "ec_sem_bayes", False)),
            ec_sem_bayes_loo=False,  # LOO dropped 2026-08-05 (empirically inert). The Bayesian gate is ALWAYS no-LOO now; --ec_sem_bayes_no_loo is a deprecated no-op. Old LOO ckpts still load: their arch dict serializes ec_sem_bayes_loo=True and ASLT1Denoiser(**arch) replays it explicitly.
            ec_sem_evidence=bool(getattr(args, "ec_sem_evidence", False)),
            pvc_residual=bool(getattr(args, "pvc_residual", False)),
            pvc_no_wgate=bool(getattr(args, "pvc_no_wgate", False)),
        )
        # EC-LRDA guard: with calibration ON (γ = r_rep·c_sem) AND stop-grad γ
        # (default), the ONLY gradient into the r_rep / c_sem branches is L_rep /
        # L_sem. If NEITHER is weighted, those branches never move off their random
        # init and γ becomes a FROZEN random-init attenuation silently multiplying
        # the anatomical guidance (no error otherwise). Require supervision, or turn
        # the gate off with --ec_no_calib.
        if (bool(getattr(args, "ec_lrda", False))
                and not bool(getattr(args, "ec_no_calib", False))
                and not bool(getattr(args, "ec_no_sg", False))
                and float(getattr(args, "w_rep", 0.0)) <= 0.0
                and float(getattr(args, "w_sem", 0.0)) <= 0.0):
            raise ValueError(
                "--ec_lrda with the calibration gate (γ=r_rep·c_sem) and stop-grad γ "
                "needs supervision: set --w_sem>0 and/or --w_rep>0 (the latter with "
                "--patch_artifact_p>0), or disable the gate with --ec_no_calib. "
                "Otherwise the gate never trains and γ is a frozen random init that "
                "silently attenuates the T1/PV guidance.")
        # Persist the EXACT construction kwargs so inference (infer_pwi.py) can
        # rebuild the identical architecture straight from the checkpoint — no need
        # to re-pass ~60 CLI flags and no silent arch/shape mismatch on load. These
        # are already-resolved ASLT1Denoiser kwargs (all JSON/torch-serialisable:
        # int/float/str/bool/list/None), so infer just does ASLT1Denoiser(**arch).
        self._arch_kwargs = dict(denoiser_kwargs)
        self.model = ASLT1Denoiser(**denoiser_kwargs).to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # Differential LR for T1-conditioning parameters (Phase 2 Run A). If
        # --t1_path_lr_mult > 1.0, split parameters into two groups so the T1
        # gate / T1-side x_proj_BD projection get a higher LR.
        # Goal: probe whether the T1 path is dormant by design (init too small,
        # ASL path dominant) vs. dormant in principle. The discriminator names
        # below must match named_parameters() exactly.
        t1_lr_mult = float(getattr(args, "t1_path_lr_mult", 1.0))
        T1_PATH_KEYS = ("t1_gate_bd", "x_proj_BD")
        ASL_PATH_KEYS = ("x_proj_BD_asl",)  # must take precedence over x_proj_BD
        if t1_lr_mult != 1.0:
            t1_params, base_params = [], []
            t1_names, base_names = [], []
            for name, prm in self.model.named_parameters():
                if not prm.requires_grad:
                    continue
                is_asl_path = any(k in name for k in ASL_PATH_KEYS)
                is_t1_path = (not is_asl_path) and any(k in name for k in T1_PATH_KEYS)
                if is_t1_path:
                    t1_params.append(prm)
                    t1_names.append(name)
                else:
                    base_params.append(prm)
                    base_names.append(name)
            logging.info(
                f"[t1_path_lr_mult={t1_lr_mult}] T1-path params: {len(t1_params)} tensors, "
                f"base params: {len(base_params)} tensors"
            )
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": tp.lr},
                    {"params": t1_params, "lr": tp.lr * t1_lr_mult},
                ],
                weight_decay=tp.weight_decay,
                betas=(float(getattr(args, "adam_beta1", 0.9)),
                       float(getattr(args, "adam_beta2", 0.999))),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=tp.lr,
                weight_decay=tp.weight_decay,
                betas=(float(getattr(args, "adam_beta1", 0.9)),
                       float(getattr(args, "adam_beta2", 0.999))),
            )
        # Stamp initial_lr on every group so the warmup loop can restore per-group
        # LRs without depending on the scheduler having attached first.
        for g in self.optimizer.param_groups:
            g.setdefault("initial_lr", float(g["lr"]))

        # Phase 2 Run C: MoSSM gate warmup. Collect t1_gate_bd Parameters, set
        # their data to logit(warmup_value), and freeze. They are unfrozen by
        # `_release_mossm_gate_warmup` once global_step >= warmup_steps.
        self._mossm_gate_warmup_steps = int(getattr(args, "mossm_gate_warmup_steps", 0))
        self._mossm_gate_warmup_released = False
        self._mossm_gate_params: List[torch.nn.Parameter] = []
        if self._mossm_gate_warmup_steps > 0:
            v = float(getattr(args, "mossm_gate_warmup_value", 0.20))
            v = max(min(v, 0.99), 0.01)
            target_logit = float(math.log(v / (1.0 - v)))
            for name, prm in self._unwrap().named_parameters():
                if "t1_gate_bd" in name:
                    with torch.no_grad():
                        prm.data.fill_(target_logit)
                    prm.requires_grad = False
                    self._mossm_gate_params.append(prm)
            logging.info(
                f"[gate_warmup] froze {len(self._mossm_gate_params)} t1_gate_bd "
                f"params at sigmoid={v:.3f} (logit={target_logit:+.4f}) for "
                f"{self._mossm_gate_warmup_steps} steps"
            )

        max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(tp.max_steps)
        if args.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps, eta_min=args.lr_min
            )
        else:
            self.scheduler = None

        self.criterion = ASLN2NLoss.from_config(self.cfg).to(self.device)
        # CLI overrides for the L_sharp hinge so we can A/B without touching
        # the YAML config.
        if float(getattr(args, "w_sharp", 0.0)) > 0.0:
            self.criterion.weights.w_sharp = float(args.w_sharp)
            self.criterion.weights.tau_sharp_ratio = float(args.tau_sharp_train)
        # DW-N2N (loss-level residual preservation): activate the deviation-weighted
        # N2N up-weight + smoothing relaxation. E=P·m̄ is computed per step in the
        # training loop and passed as dev_weight; it never enters the output.
        self.dw_n2n = bool(getattr(args, "dw_n2n", False))
        self._dw_m0 = None
        self.dw_pv_source = str(getattr(args, "dw_pv_source", "fsl"))
        if self.dw_n2n:
            self.criterion.weights.up_dev = float(getattr(args, "w_dev_up", 1.0))
            self.criterion.weights.relax_smooth = float(getattr(args, "w_dev_relax", 0.0))
            _pj = str(getattr(args, "dw_prior_json", "") or "")
            if _pj and os.path.isfile(_pj):
                import json as _json
                _m0 = _json.load(open(_pj)).get("m0")
                if _m0:
                    self._dw_m0 = torch.tensor(_m0, dtype=torch.float32, device=self.device)
                    print(f"[dw-n2n] pv_source={self.dw_pv_source} up_dev={self.criterion.weights.up_dev} "
                          f"relax_smooth={self.criterion.weights.relax_smooth} "
                          f"m0={[round(float(v), 4) for v in _m0]} (tau=64) from {_pj}")
            if self._dw_m0 is None:
                print(f"[dw-n2n] pv_source={self.dw_pv_source} up_dev={self.criterion.weights.up_dev} "
                      f"relax_smooth={self.criterion.weights.relax_smooth}; WARN no valid "
                      f"--dw_prior_json ({_pj!r}) -> legacy shrinkage tau=0.5/m0=0 "
                      f"(run scripts/estimate_dw_prior.py).")

        ema_cls = EMAModel if EMAModel is not None else _FallbackEMAModel
        self.ema = ema_cls(
            self.model,
            update_after_step=int(getattr(args, "ema_start_step", 0)),
            inv_gamma=1.0,
            power=2 / 3,
            min_value=0.0,
            max_value=float(getattr(args, "ema_decay", 0.9999)),
            device=self.device,
        )

        # SWA running mean (Izmailov ICLR 2018). Accumulates from --swa_start_step.
        # Saved as `swa.pth` alongside `best.pth` at the end of training.
        if bool(getattr(args, "use_swa", False)):
            from torch.optim.swa_utils import AveragedModel
            self.swa_model = AveragedModel(self._unwrap()).to(self.device)
            self.swa_n_avg = 0
        else:
            self.swa_model = None
            self.swa_n_avg = 0

        self.global_step = 0
        # Model selection criterion: PSNR vs 12-NEX reference (higher is better).
        # See learning curve diagnostics — L1 vs setB diverges from gold reference
        # because mean(setB) is itself noisy; PSNR/SSIM vs union(A,B) is the
        # unbiased target metric to track.
        self.best_val = -float("inf")
        self.best_metrics: Dict[str, dict] = {}   # in-loop per-metric-best (name -> {value,step,dir})
        # Score EMA for best-selection (suppresses single-validation noise).
        # ema_score = (1-α)·prev + α·current. α=0 ⇒ no smoothing (raw score).
        self.score_ema_alpha = float(getattr(args, "score_ema_alpha", 0.0))
        self.score_ema: Optional[float] = None
        self.no_improve_count = 0
        self.eval_count = 0

        _has_t1_branch = getattr(self._unwrap(), "t1_encoder", None) is not None
        if args.resume:
            self._try_resume()
        elif args.init_t1_from and _has_t1_branch:
            self._init_t1_from_ckpt(args.init_t1_from)
        elif args.init_t1_from and not _has_t1_branch:
            logging.info("[no_t1_branch] --init_t1_from ignored (T1 seg branch skipped; fsl/rawt1 needs no stage-1 prior).")
        if args.freeze_t1 and _has_t1_branch:
            self._freeze_t1()

    def _unwrap(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _ckpt(self, tag: str) -> str:
        return os.path.join(self.ckpt_dir, f"{tag}.pth")

    def _init_t1_from_ckpt(self, ckpt_path: str):
        """Load t1_encoder + t1_decoder weights from a stage-1 T1Branch checkpoint.

        Critical: also overwrites the EMA snapshot for these keys, otherwise
        validation (which uses EMA weights) would see random T1 features for
        the first few hundred steps until the EMA decays into the loaded values.
        """
        st = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        # Guard the silent encoder-only load: if stage-1 recorded its pretext and it
        # disagrees with stage-2 --t1_task, the frozen t1_decoder head is the wrong
        # width (recon=1ch vs seg=4ch) and would be shape-skipped → frozen random
        # head → garbage brain mask / seg. Fail loudly instead. (Older ckpts without
        # this field are unaffected.)
        ckpt_task = st.get("t1_task")
        want_task = str(getattr(self.args, "t1_task", "seg"))
        if ckpt_task is not None and str(ckpt_task) != want_task:
            raise ValueError(
                f"Stage-1 prior was pretrained with --t1_task='{ckpt_task}', but stage-2 "
                f"got --t1_task='{want_task}'. The frozen t1_decoder head widths differ "
                f"(seg=4ch vs recon=1ch) — fix --t1_task or re-pretrain the prior. ({ckpt_path})")
        sd = st.get("ema") or st["model"]
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        # T1Branch keys are already named t1_encoder.* / t1_decoder.*, matching ASLT1Denoiser.
        target = self._unwrap().state_dict()
        loaded = []
        for k, v in sd.items():
            if k in target and target[k].shape == v.shape and (
                k.startswith("t1_encoder.") or k.startswith("t1_decoder.")
            ):
                target[k] = v
                loaded.append(k)
        self._unwrap().load_state_dict(target, strict=False)
        # Also refresh the EMA snapshot for these keys so validation uses the
        # loaded T1 weights from step 0 (not the random init that was snapshotted
        # when self.ema was constructed before this method was called).
        model_sd = self._unwrap().state_dict()
        ema_state = getattr(self.ema, "ema_state", None)
        if ema_state is not None:
            refreshed = 0
            for k in loaded:
                if k in ema_state:
                    ema_state[k] = model_sd[k].detach().to(ema_state[k].device).clone()
                    refreshed += 1
            logging.info(f"Loaded {len(loaded)} T1-branch weights from {ckpt_path} "
                         f"(refreshed {refreshed} EMA entries)")
        else:
            logging.info(f"Loaded {len(loaded)} T1-branch weights from {ckpt_path}")

    def _freeze_t1(self):
        m = self._unwrap()
        n = 0
        for p in m.t1_encoder.parameters():
            p.requires_grad = False; n += 1
        if m.t1_decoder is not None:
            for p in m.t1_decoder.parameters():
                p.requires_grad = False; n += 1
        logging.info(f"Froze {n} T1-branch parameters (t1_encoder + t1_decoder).")

    def _filter_ckpt_state(self, sd: Dict[str, Tensor], own, what: str) -> Dict[str, Tensor]:
        """Drop ckpt keys the current arch no longer defines — currently only the
        t1_decoder head (removed in the no-seg arm, where it never received gradient,
        so dropping it is exact). Any OTHER unexpected key is an arch mismatch and
        still fails loudly instead of being silently discarded."""
        own = set(own)
        extra = [k for k in sd if k not in own]
        if not extra:
            return sd
        bad = [k for k in extra if not k.startswith("t1_decoder.")]
        if bad:
            raise ValueError(
                f"[resume] {what}: checkpoint carries {len(extra)} keys absent from the "
                f"model, {len(bad)} of them outside t1_decoder.* (e.g. {bad[:3]}). "
                "Architecture mismatch — refusing to drop them silently.")
        logging.info(f"[resume] {what}: dropped {len(extra)} t1_decoder.* keys "
                     "(head removed from the arch; those weights were never trained).")
        return {k: v for k, v in sd.items() if k in own}

    def _try_resume(self):
        latest = self._ckpt("latest")
        if not os.path.exists(latest):
            return
        state = torch.load(latest, map_location=self.device, weights_only=False)
        state["model"] = self._filter_ckpt_state(
            state["model"], self.model.state_dict().keys(), "model")
        self.model.load_state_dict(state["model"])
        self.global_step = state.get("step", 0)
        # If the selection criterion has changed since the ckpt was saved, the
        # stored best_val is in a different unit system (e.g., upsnr_cyc ≈ 20 dB
        # vs constrained_umse ≈ −1e−4) and would always block new bests. Reset
        # in that case. Legacy ckpts (saved before best_criterion was stored)
        # also trigger reset, since we cannot verify compatibility.
        prior_crit = state.get("best_criterion", "__legacy__")
        criterion_changed = prior_crit != self.args.best_criterion
        if criterion_changed:
            logging.info(
                f"[resume] best_criterion: stored={prior_crit!r} → "
                f"current={self.args.best_criterion!r}. Resetting "
                f"best_val=-inf and no_improve_count=0 (new criterion is "
                f"in a different unit system and cannot be compared)."
            )
            self.best_val = float("-inf")
            self.no_improve_count = 0
        else:
            self.best_val = state.get("best_val", float("inf"))
        # Restore early-stop counters; older ckpts (pre-fix) lack these keys
        # so we fall back to 0 — equivalent to giving the resumed run a fresh
        # patience window, slightly more conservative than continuing the count.
        # If the criterion changed (handled above), keep no_improve_count = 0.
        self.eval_count = state.get("eval_count", 0)
        self.best_metrics = state.get("best_metrics", {}) or {}   # resume in-loop per-metric-best
        if not criterion_changed:
            self.no_improve_count = state.get("no_improve_count", 0)
            # Restore score EMA if criterion didn't change. Reset on criterion
            # change (different unit system); fresh EMA will rebuild.
            stored_ema = state.get("score_ema", None)
            if stored_ema is not None:
                self.score_ema = float(stored_ema)
        else:
            self.score_ema = None
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            _ema_sd = self._filter_ckpt_state(
                state["ema"], self.ema.ema_state.keys(), "ema")
            self.ema.ema_state = {k: v.to(self.device) for k, v in _ema_sd.items()}
            self.ema.optimization_step = state.get("ema_optimization_step", 0)
        if "scheduler" in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        logging.info(f"Resumed from {latest} (step={self.global_step}, best_val={self.best_val:.6f})")

    def _save(self, tag: str):
        torch.save({
            "model": self.model.state_dict(),
            # Architecture kwargs (exact ASLT1Denoiser(**arch) construction args) so
            # inference rebuilds the identical model from the ckpt alone. Also carries
            # t1_task — infer/eval can assert the frozen T1 head matches (guards the
            # silent encoder-only load when stage-1/stage-2 --t1_task disagree).
            "arch": getattr(self, "_arch_kwargs", None),
            "step": self.global_step,
            "best_val": self.best_val,
            "best_metrics": self.best_metrics,            # in-loop per-metric-best map (resume-safe)
            "best_criterion": self.args.best_criterion,   # for resume change-detection
            "eval_count": self.eval_count,
            "no_improve_count": self.no_improve_count,
            "score_ema": self.score_ema,
            "score_ema_alpha": self.score_ema_alpha,
            "ema": {k: v.detach().cpu() for k, v in self.ema.ema_state.items()},
            "ema_optimization_step": getattr(self.ema, "optimization_step", 0),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }, self._ckpt(tag))

    def _maybe_flip(self, pack: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Random spatial flip (train-only aug). Applied CONSISTENTLY to setA/setB/T1/PV so
        the N2N target stays aligned and the T1<->ASL correspondence is preserved (flip is
        linear => N2N-unbiased). Independent coin per axis (W = L-R, H = A-P)."""
        p = float(getattr(self.args, "flip_p", 0.0))
        if p <= 0.0:
            return pack
        out = dict(pack)
        for dim in (-1, -2):
            if float(torch.rand((), device=out["setA"].device).item()) < p:
                for k in ("setA", "setB", "t1", "gm", "wm", "csf"):
                    if k in out and torch.is_tensor(out[k]):
                        out[k] = torch.flip(out[k], dims=[dim])
        return out

    def _update_per_metric_best(self):
        """In-loop per-metric-best: keep the best EMA ckpt for EVERY self-supervised val metric
        (best_<metric>.pth) + a best_metrics.json map. Independent of --best_criterion
        (recoverability only, NEVER selection). Reads self._last_val_extras (EMA-computed)."""
        if not bool(getattr(self.args, "save_per_metric_best", False)):
            return
        ex = getattr(self, "_last_val_extras", None)
        if not ex:
            return
        dirs = {
            "umse": "min", "upsnr": "max", "cnr_pred": "max",
            "scov_gm_pred": "min", "scov_wm_pred": "min",
            "hfen": "min", "gmsd": "min", "efc_pred": "min",
            "nrmse": "min", "tg_pred": "max", "subset_consistency": "min",
        }
        for name, direction in dirs.items():
            try:
                v = float(ex.get(name))
            except (TypeError, ValueError):
                continue
            if v != v or v in (float("inf"), float("-inf")):
                continue
            prev = self.best_metrics.get(name)
            better = (prev is None
                      or (direction == "min" and v < prev["value"])
                      or (direction == "max" and v > prev["value"]))
            if better:
                self.best_metrics[name] = {"value": v, "step": int(self.global_step), "dir": direction}
                self._save(f"best_{name}")
        try:
            import json
            with open(os.path.join(os.path.dirname(self._ckpt("best_metrics")), "best_metrics.json"), "w") as f:
                json.dump(self.best_metrics, f, indent=2)
        except Exception as e:
            logging.warning(f"[per-metric-best] json write failed: {e}")

    def _maybe_inject_bad_frame(self, pack: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """With probability bad_frame_p, corrupt one valid frame of setA with high noise.
        set_b is NEVER corrupted (it's the N2N target)."""
        p = float(self.args.bad_frame_p)
        if p <= 0.0 or torch.rand(1).item() >= p:
            return pack
        set_a = pack["setA"]
        B, T = set_a.shape[0], set_a.shape[1]
        len_a = pack.get("lenA")
        if len_a is not None:
            valid = len_a.to(set_a.device).float().clamp_min(1.0)
            bad_idx = (torch.rand(B, device=set_a.device) * valid).long().clamp_max(T - 1)
        else:
            bad_idx = torch.randint(0, T, (B,), device=set_a.device)
        nmin, nmax = self.args.bad_frame_noise_min, self.args.bad_frame_noise_max
        scale = torch.rand(B, device=set_a.device) * (nmax - nmin) + nmin   # [B]
        frame_mask = torch.zeros(B, T, device=set_a.device)
        frame_mask.scatter_(1, bad_idx.unsqueeze(1), 1.0)
        noise = torch.randn_like(set_a) * scale.view(B, 1, 1, 1, 1)
        new_set_a = set_a + frame_mask.view(B, T, 1, 1, 1) * noise
        # Surface the per-sample corrupted-frame index so the training loop can
        # probe whether the aggregator down-weights it (frame_mask shape [B,T]).
        return {**pack, "setA": new_set_a, "_bad_frame_mask": frame_mask}

    def _maybe_inject_patch_artifact(self, pack: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Phase 2 Run B': inject high-sigma noise into a local patch of one
        set_a frame. Generates spatially-heterogeneous Var_T(set_a) so RGSF can
        learn to discriminate clean vs corrupt regions per pixel. set_b never
        touched. Combines with bad_frame_p (which provides whole-frame noise);
        having both gives the aggregator (frame-level) and RGSF (pixel-level)
        different but consistent training signals."""
        p = float(getattr(self.args, "patch_artifact_p", 0.0))
        if p <= 0.0 or torch.rand(1).item() >= p:
            return pack
        set_a = pack["setA"]
        B, T, _, H, W = set_a.shape
        device = set_a.device
        ps = int(getattr(self.args, "patch_artifact_size", 16))
        ps = max(1, min(ps, min(H, W)))
        nmin = float(self.args.patch_artifact_noise_min)
        nmax = float(self.args.patch_artifact_noise_max)
        # Pick k~U[1,frames_max] VALID frames + one patch per sample.
        len_a = pack.get("lenA")
        valid = (len_a.to(device).float().clamp_min(1.0) if len_a is not None
                 else torch.full((B,), float(T), device=device))
        fmax = max(1, int(getattr(self.args, "patch_artifact_frames_max", 1)))
        ys = torch.randint(0, max(1, H - ps + 1), (B,), device=device)
        xs = torch.randint(0, max(1, W - ps + 1), (B,), device=device)
        scale = torch.rand(B, device=device) * (nmax - nmin) + nmin
        # Build per-sample mask [B,T,1,H,W]: 1.0 inside the patch on k_b chosen VALID
        # frames and 0 elsewhere. Corrupting >1 frame raises the per-pixel corrupt
        # fraction so L_rep's target 1-frac drops enough for r_rep to learn a real
        # attenuation (single-frame => frac~1/T => target~0.92 => ~flat slope). Loop
        # over B (≤32) is cheap; the noise add is fully vectorised.
        mask = torch.zeros(B, T, 1, H, W, device=device)
        for b in range(B):
            nv = max(1, min(int(valid[b].item()), T))
            kb = int(torch.randint(1, min(fmax, nv) + 1, (1,), device=device).item())
            fsel = torch.randperm(nv, device=device)[:kb]
            y0 = int(ys[b].item()); x0 = int(xs[b].item())
            mask[b, fsel, 0, y0:y0 + ps, x0:x0 + ps] = 1.0
        noise = torch.randn_like(set_a) * scale.view(B, 1, 1, 1, 1)
        new_set_a = set_a + mask * noise
        # Surface the per-frame per-pixel artifact mask [B,T,1,H,W] so L_rep can
        # supervise the EC reproducibility map r_rep = 1 - mean_k M_k.
        return {**pack, "setA": new_set_a, "_patch_artifact_mask": mask}

    def _maybe_apply_jinv_mask(self, pack: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """J-invariance input masking (Krull N2V CVPR 2019 / Batson N2Self ICML 2019).

        With probability `jinv_p` per pixel position (consistent across all T frames),
        replace the value in every setA frame at that position with the local 3x3 mean
        (excluding the centre). Forces approximate J-invariance: the model cannot
        directly use setA[h,w] in predicting output at (h,w), it must use context.

        Standard self-supervised regulariser; replaces TV term in v36+ since TV
        is structurally agnostic and over-smooths cortex (v35e diagnosis).
        """
        p = float(self.args.jinv_p)
        if p <= 0.0:
            return pack
        set_a = pack["setA"]
        B, T, C, H, W = set_a.shape
        # Per-pixel mask, shared across T frames (so all T values at (h,w) get masked).
        mask_2d = (torch.rand(B, 1, H, W, device=set_a.device) < p).float()  # [B,1,H,W]
        mask_5d = mask_2d.unsqueeze(1).expand(B, T, C, H, W)
        # 3x3 neighbour-mean kernel (centre excluded), per channel.
        kernel = torch.full((1, 1, 3, 3), 1.0 / 8.0, device=set_a.device)
        kernel[0, 0, 1, 1] = 0.0
        kernel = kernel.expand(C, 1, 3, 3)
        flat = set_a.reshape(B * T, C, H, W)
        local_mean = F.conv2d(flat, kernel, padding=1, groups=C).reshape(B, T, C, H, W)
        new_set_a = set_a * (1.0 - mask_5d) + local_mean * mask_5d
        return {**pack, "setA": new_set_a}

    def _mask_asl_inputs(self, pack: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Zero out setA / setB outside the brain mask.

        Uses the **frozen T1 stage-1 segmentation head** to derive a soft
        brain mask `brain = 1 - softmax(logits)[BG]`, which has clean smooth
        boundaries and **includes CSF / ventricles** at their proper soft
        edge. The legacy `t1 > 0.05` threshold incorrectly excluded CSF
        (T1-dark voxels), producing visible holes at the ventricles in the
        recon. Falls back to `t1 > thr` if the seg head is missing or has
        fewer than 4 channels.

        Replaces the explicit `loss_bg` term: with masked inputs the network
        never has a reason to produce non-zero output outside the brain
        (the N2N target is also zero there). T1 is left unmasked
        (encoder/cross-attention still benefits from anatomical context).

        DISABLED BY DEFAULT (2026-07; --premask_asl_inputs to re-enable). Pre-masking
        the ASL INPUT is dropped in favour of masking only the LOSS (t1>thr, already
        applied) and the OUTPUT/eval. This (a) removes the seg-soft-mask + 2.5D
        broadcast + train/infer-consistency footgun entirely, and (b) lets the frozen
        T1 branch run ONCE inside model.forward instead of an extra forward here. The
        ASL diff is ~0 outside the brain, so the N2N target already drives output→0
        there without input pre-masking.
        """
        if not bool(getattr(self.args, "premask_asl_inputs", False)):
            return pack
        model = self._unwrap()
        brain: Optional[Tensor] = None
        # Only the 4-class seg head yields a usable soft brain mask. Under
        # t1_task='recon' the 1-ch output is discarded below, so running the T1 branch
        # here is pure waste (an extra no-grad encoder+decoder forward every step) —
        # go straight to the t1>thr fallback.
        if (str(getattr(model, "t1_task", "seg")) == "seg"
                and getattr(model, "t1_encoder", None) is not None
                and getattr(model, "t1_decoder", None) is not None):
            with torch.no_grad():
                t1_feat_map, _, t1_skips = model.t1_encoder(pack["t1"])
                t1_seg_logits = model.t1_decoder(t1_feat_map, t1_skips)
                if t1_seg_logits.size(1) >= 4:
                    # BG is channel 3. P(BG) via SOFTMAX over the 4 PV classes (a
                    # simplex; phase1 trains the seg head with --seg_softmax). Using
                    # sigmoid on the BG logit alone mis-reads P(BG) under a softmax-
                    # trained head — the BG logit's absolute scale is unconstrained, so
                    # sigmoid inflates P(BG) in-brain and shrinks the mask, dimming
                    # setA/setB. brain = P(GM)+P(WM)+P(CSF) = 1 - softmax(logits)[BG].
                    brain = (1.0 - torch.softmax(t1_seg_logits, dim=1)[:, 3:4]).clamp(0.0, 1.0)
        if brain is None:
            thr = float(self.criterion.weights.mask_threshold)
            brain = (pack["t1"] > thr).float()
        brain_T = brain.unsqueeze(1)              # [B,1,1,H,W] — broadcast over T
        out_pack = dict(pack)
        out_pack["setA"] = pack["setA"] * brain_T
        out_pack["setB"] = pack["setB"] * brain_T
        return out_pack

    def _apply_cond_pv(self, model, pack: Dict[str, Tensor]) -> None:
        """For cond_src in {'fsl','fsl_t1'}, build the brain-masked FSL GM/WM/CSF
        conditioning PV from the batch and stash it on the model (set_cond_pv). Every
        subsequent forward/infer on that model instance (main, aux losses, val, SURE)
        reads it. fsl_t1 appends the raw T1 as a 4th channel (HYBRID). No-op for
        pv/rawt1 (they build PV internally from softmax(t1_seg) / raw T1)."""
        _src = getattr(model, "lrda_cond_src", "pv")
        if _src not in ("fsl", "fsl_t1"):
            return
        gm, wm, csf = pack.get("gm"), pack.get("wm"), pack.get("csf")
        if gm is None or wm is None or csf is None:
            raise ValueError(
                f"lrda_cond_src='{_src}' needs gm/wm/csf in the batch (precomputed FSL PV segs).")
        from models.asl_t1_model import build_fsl_cond_pv
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = gm.device
        t1 = pack.get("t1")
        model.set_cond_pv(build_fsl_cond_pv(
            gm.to(dev), wm.to(dev), csf.to(dev),
            t1=(t1.to(dev) if t1 is not None else None),
            with_t1=(_src == "fsl_t1")))

    def _forward_train(self, pack: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        # 1) Inject bad-frame noise FIRST (it adds randn over the full image).
        # 2) Apply J-invariance pixel masking (v36+) — random ~jinv_p positions
        #    in setA replaced with local mean across all T frames. Replaces TV
        #    as the texture-regularisation mechanism.
        # 3) Then zero out everything outside the brain mask. This way the
        #    aggregator sees high-σ corruption only inside brain on the bad
        #    frame, not random noise outside which the model would just learn
        #    to discard. Background outside is always 0 — no loss_bg needed.
        pack = self._maybe_inject_bad_frame(pack)
        # Phase 2 Run B': local patch artifact gives RGSF a per-pixel signal.
        pack = self._maybe_inject_patch_artifact(pack)
        pack = self._maybe_apply_jinv_mask(pack)
        pack = self._mask_asl_inputs(pack)
        model = self._unwrap()
        self._apply_cond_pv(model, pack)
        outputs = model(
            set_a=pack["setA"],
            set_b=pack["setB"],
            t1=pack["t1"],
            len_a=pack.get("lenA"),
            len_b=pack.get("lenB"),
            mask_a=pack.get("maskA"),
            mask_b=pack.get("maskB"),
        )
        # DW-N2N: per-pixel deviation weight d in [0,1] = candidate reproducible anomaly
        # (ASL_dmvae/docs/dwn2n_design.md). E=P·m̄ (shrinkage ridge toward m0) -> noise-calibrated z
        # -> even/odd reproducibility gate -> fixed absolute squash. setA/PV-derived => ⊥ setB
        # => N2N-unbiased; @no_grad, never enters the output (strict V=ASL).
        dev_weight = None
        dw_stats = {}
        if getattr(self, "dw_n2n", False):
            # PV for E=P·m̄. 'fsl' (default): precomputed gm/wm/csf already in the batch
            # (ASL-space, perfusion-independent, no stage-1 dep); 'seg': frozen T1 head.
            _src = getattr(self, "dw_pv_source", "fsl")
            pv = None
            if _src == "fsl":
                if all(pack.get(k) is not None for k in ("gm", "wm", "csf")):
                    _gm, _wm, _csf = pack["gm"], pack["wm"], pack["csf"]
                    _bg = (1.0 - (_gm + _wm + _csf)).clamp(0.0, 1.0)
                    pv = torch.cat([_gm, _wm, _csf, _bg], dim=1)       # [B,4,H,W] GM,WM,CSF,BG
            elif outputs.get("t1_seg") is not None:                    # 'seg'
                pv = torch.softmax(outputs["t1_seg"], dim=1)           # [B,C,H,W]
            if pv is not None:
                from losses.dw_weight import dw_weight_map
                brain = (pack["t1"] > self.criterion.weights.mask_threshold).float()
                _ctx = int(getattr(self.args, "slice_context", 0))
                setA = pack["setA"]
                if _ctx > 0:                                           # 2.5-D: center slice (dim=2)
                    _ic = setA.shape[2] // (2 * _ctx + 1)
                    _c0 = _ic * _ctx
                    setA_c = setA[:, :, _c0:_c0 + _ic]
                else:
                    setA_c = setA
                dev_weight, dw_stats = dw_weight_map(
                    setA_c, outputs["agg"], pv, brain,
                    lengths=pack.get("lenA"), mask=pack.get("maskA"),
                    m0=getattr(self, "_dw_m0", None))
        loss, stats = self.criterion(
            outputs, pack["t1"],
            len_a=pack.get("lenA"),
            len_b=pack.get("lenB"),
            gm=pack.get("gm"),
            wm=pack.get("wm"),
            csf=pack.get("csf"),
            dev_weight=dev_weight,
        )
        if dw_stats:
            stats.update(dw_stats)
        # Aggregator activity probe: surface (a) softmax entropy across frames
        # and (b) bad-vs-normal weight ratio when a corrupted frame was injected.
        # Probe-only; no grad path. Detached → cheap.
        w_a = outputs.get("weights")
        if w_a is not None and w_a.ndim == 2:
            with torch.no_grad():
                w_d = w_a.detach()
                ent = -(w_d.clamp_min(1e-12) * w_d.clamp_min(1e-12).log()).sum(dim=1)
                stats["probe_agg_entropy"] = float(ent.mean().item())
                bad_mask = pack.get("_bad_frame_mask")
                if bad_mask is not None and bad_mask.shape == w_d.shape:
                    bm = bad_mask.to(w_d.dtype)
                    bad_w = (w_d * bm).sum(dim=1).clamp_min(1e-12)
                    norm_w = (w_d * (1.0 - bm)).sum(dim=1).clamp_min(1e-12)
                    n_norm = (1.0 - bm).sum(dim=1).clamp_min(1.0)
                    # Mean per-frame weight: bad frame (single) vs normal frames.
                    stats["probe_agg_w_bad"] = float(bad_w.mean().item())
                    stats["probe_agg_w_normal_per_frame"] = float(
                        (norm_w / n_norm).mean().item()
                    )
        # EC-LRDA auxiliary supervision (decoupled from N2N; γ is stop-grad'd out of
        # the recon path in the model so neither branch can be co-opted / collapse).
        w_rep = float(getattr(self.args, "w_rep", 0.0))
        w_sem = float(getattr(self.args, "w_sem", 0.0))
        if getattr(model, "ec_lrda", False) and (w_rep > 0.0 or w_sem > 0.0):
            thr = self.criterion.weights.mask_threshold
            brain = (pack["t1"] > thr).float()                          # [B,1,H,W]
            # r_rep / L_rep retired 2026-08-11 (empirically inert, 0.18% same-weights ablation):
            # the reproducibility leg + its patch-artifact supervision are removed. Only the
            # c_sem / L_sem leg below trains the gate. r_rep params/modules (ECGuidance.frame_enc,
            # a0_rep, beta_rep) and the --w_rep/--patch_artifact_*/--rep_target_* flags are kept as
            # no-ops (default 0) so pre-existing checkpoints still strict-load.
            # L_sem: matched PV must be MORE ASL-compatible than cross-subject mismatched PV.
            c_m = outputs.get("ec_c_sem"); c_x = outputs.get("ec_c_sem_mis")
            if w_sem > 0.0 and c_m is not None and c_x is not None:
                hw = c_m.shape[-2:]
                brain_d = F.interpolate(brain, size=hw, mode="area")
                den = brain_d.sum(dim=(1, 2, 3)).clamp_min(1.0)
                cm = (c_m * brain_d).sum(dim=(1, 2, 3)) / den            # [B]
                cx = (c_x * brain_d).sum(dim=(1, 2, 3)) / den            # [B]
                loss_sem = -F.logsigmoid(cm - cx).mean()
                loss = loss + w_sem * loss_sem
                stats["loss_sem"] = float(loss_sem.detach().item())
                stats["probe_c_match"] = float(cm.mean().item())
                stats["probe_c_mismatch"] = float(cx.mean().item())
        # Innovation A — Adversarial mismatched-T1 training (identity-invariant
        # cross-modal conditioning). With probability --p_adv, run a second
        # forward pass using a within-batch-permuted T1 (no fixed point) and
        # penalise the output divergence. Stop-grad on the mismatch branch
        # (BYOL-style, default) makes the matched branch chase a frozen target
        # and avoids two-way training instability.
        w_adv = float(getattr(self.args, "w_adv", 0.0))
        B = pack["setA"].shape[0]
        if (bool(getattr(self.args, "use_adv_t1", False))
                and w_adv > 0.0 and B >= 2
                and torch.rand((), device=pack["t1"].device).item() < float(self.args.p_adv)):
            shift = int(torch.randint(1, B, (1,)).item())
            t1_mm = pack["t1"].roll(shifts=shift, dims=0)
            stop_grad = not bool(getattr(self.args, "adv_no_stopgrad", False))
            ctx = torch.no_grad() if stop_grad else contextlib.nullcontext()
            with ctx:
                out_mm = model(
                    set_a=pack["setA"],
                    set_b=pack["setB"],
                    t1=t1_mm,
                    len_a=pack.get("lenA"),
                    len_b=pack.get("lenB"),
                    mask_a=pack.get("maskA"),
                    mask_b=pack.get("maskB"),
                )
            pred_match = outputs["asl_recon"]
            pred_mm = out_mm["asl_recon"]
            if stop_grad:
                pred_mm = pred_mm.detach()
            brain_mask = (pack["t1"] > self.criterion.weights.mask_threshold).float()
            denom = brain_mask.sum().clamp_min(1.0)
            loss_adv = ((pred_match - pred_mm).abs() * brain_mask).sum() / denom
            loss = loss + w_adv * loss_adv
            stats["loss_adv"] = float(loss_adv.detach().item())
            stats["adv_triggered"] = 1.0
        else:
            stats["adv_triggered"] = 0.0

        # CC-5 — Bidirectional N2N cycle (BYOL-style stop-grad). set_a and
        # set_b are disjoint noisy realisations of the SAME clean signal →
        # f(set_a) and f(set_b) should converge to the same denoised output.
        # The target branch is stop-gradient (no two-way training instability).
        #
        # ssim_cyc (2026-05-29) shares the same set_b forward: a pred-to-pred
        # structural-consistency term 1-SSIM(f(A), sg(f(B))). Because f(A)/f(B)
        # have independent noise, low-freq mottling differs between them and is
        # penalised → pushes toward the shared clean structure. Unlike
        # SSIM-vs-noisy-target, it never rewards the reference's noise texture.
        w_bdcyc = float(getattr(self.args, "w_bdcyc", 0.0))
        w_ssim_cyc = float(getattr(self.args, "w_ssim_cyc", 0.0))
        w_cnr_cyc = float(getattr(self.args, "w_cnr_cyc", 0.0))
        use_bdcyc = bool(getattr(self.args, "use_bdcyc", False)) and w_bdcyc > 0.0
        _setb_recon = None            # stash f(set_b) (detached) for reuse by the texture loss
        if use_bdcyc or w_ssim_cyc > 0.0 or w_cnr_cyc > 0.0:
            out_b = model.infer_from_subset(
                pack["setB"], pack["t1"],
                lengths=pack.get("lenB"), mask=pack.get("maskB"),
            )
            pred_a = outputs["asl_recon"]
            pred_b = out_b["asl_recon"].detach()
            _setb_recon = pred_b
            brain_mask_c = (pack["t1"] > self.criterion.weights.mask_threshold).float()
            denom_c = brain_mask_c.sum().clamp_min(1.0)
            if use_bdcyc:
                loss_bdcyc = ((pred_a - pred_b).abs() * brain_mask_c).sum() / denom_c
                loss = loss + w_bdcyc * loss_bdcyc
                stats["loss_bdcyc"] = float(loss_bdcyc.detach().item())
            if w_ssim_cyc > 0.0:
                from losses.asl_n2n_loss import ssim_loss
                loss_ssim_cyc = ssim_loss(pred_a * brain_mask_c, pred_b * brain_mask_c)
                loss = loss + w_ssim_cyc * loss_ssim_cyc
                stats["loss_ssim_cyc"] = float(loss_ssim_cyc.detach().item())
            if w_cnr_cyc > 0.0 and ("gm" in pack) and ("wm" in pack):
                # CNR (GM-vs-WM contrast-to-noise) consistency between the two
                # disjoint subsets. pred_b is already stop-grad (detached above),
                # so the gradient only pulls CNR(f(A)) toward the current
                # CNR(f(B)) estimate — enforces a subset-invariant contrast.
                cnr_a, valid_a = _cnr_per_sample(pred_a, pack["gm"], pack["wm"])
                cnr_b, valid_b = _cnr_per_sample(pred_b, pack["gm"], pack["wm"])
                valid = valid_a * valid_b
                vd = valid.sum().clamp_min(1.0)
                loss_cnr_cyc = ((cnr_a - cnr_b.detach()).abs() * valid).sum() / vd
                loss = loss + w_cnr_cyc * loss_cnr_cyc
                stats["loss_cnr_cyc"] = float(loss_cnr_cyc.detach().item())
                # One-time activation log so a multi-hour run proves cnr_cyc is
                # actually contributing (per-step stats are not written to file).
                if not getattr(self, "_cnr_cyc_logged", False):
                    logging.info(
                        f"[cnr_cyc] ACTIVE w={w_cnr_cyc:.3f} "
                        f"loss_cnr_cyc={float(loss_cnr_cyc.detach().item()):.5f} "
                        f"valid_slices={float(valid.sum().item()):.0f}/{valid.numel()}"
                    )
                    self._cnr_cyc_logged = True
            elif w_cnr_cyc > 0.0 and not getattr(self, "_cnr_cyc_warned", False):
                # w_cnr_cyc requested but gm/wm absent from the train batch →
                # term silently skipped. Warn loudly once so it is never a silent
                # no-op across a long run.
                logging.warning(
                    "[cnr_cyc] w_cnr_cyc=%.3f but 'gm'/'wm' missing from train "
                    "batch — cnr_cyc loss is being SKIPPED. Check the dataloader "
                    "provides PV maps to the training split.", w_cnr_cyc
                )
                self._cnr_cyc_warned = True

        # Innovation T — Privileged Information Distillation (LUPI). Force the
        # model's T1-zero forward to match its T1-aware forward. The T1-aware
        # branch is the teacher (stop-grad); the T1-zero branch is the student
        # being trained. After convergence, deployment can use T1=0 input,
        # making T1 hallucination structurally impossible at inference.
        w_pid = float(getattr(self.args, "w_pid", 0.0))
        if bool(getattr(self.args, "use_pid", False)) and w_pid > 0.0:
            t1_zero = torch.zeros_like(pack["t1"])
            out_pid = model(
                set_a=pack["setA"],
                set_b=pack["setB"],
                t1=t1_zero,
                len_a=pack.get("lenA"),
                len_b=pack.get("lenB"),
                mask_a=pack.get("maskA"),
                mask_b=pack.get("maskB"),
            )
            pred_zero = out_pid["asl_recon"]
            pred_full = outputs["asl_recon"].detach()
            brain_mask_pid = (pack["t1"] > self.criterion.weights.mask_threshold).float()
            denom_pid = brain_mask_pid.sum().clamp_min(1.0)
            loss_pid = ((pred_zero - pred_full).abs() * brain_mask_pid).sum() / denom_pid
            loss = loss + w_pid * loss_pid
            stats["loss_pid"] = float(loss_pid.detach().item())
        # Optional SURE term — needs an extra perturbed forward through the model.
        # Schedule:
        #   step 0..warmup:        ramp 0 → target (let N2N converge first)
        #   step warmup..anneal_start: hold at target (full SURE pressure)
        #   step anneal_start..max: linear decay target → anneal_to (avoid late-stage over-smoothing)
        # Clamp on SURE value protects against MC outliers.
        w_sure_target = float(self.criterion.weights.w_sure)
        if w_sure_target > 0.0:
            from losses.asl_n2n_loss import mc_sure_term
            warmup_steps   = int(getattr(self.criterion.weights, "sure_warmup_steps", 30))
            anneal_start   = int(getattr(self.criterion.weights, "sure_anneal_start", 0))
            anneal_to      = float(getattr(self.criterion.weights, "sure_anneal_to", 0.0))
            max_steps_cfg  = int(self.cfg.asl_denoiser_train_params.max_steps)
            step = self.global_step
            if step < warmup_steps:
                w_sure_eff = w_sure_target * (step / max(1.0, float(warmup_steps)))
            elif anneal_start > 0 and step >= anneal_start and max_steps_cfg > anneal_start:
                t = min(1.0, (step - anneal_start) / float(max_steps_cfg - anneal_start))
                w_sure_eff = w_sure_target + (anneal_to - w_sure_target) * t
            else:
                w_sure_eff = w_sure_target
            stats["w_sure_eff"] = w_sure_eff
            if w_sure_eff > 0.0:
                brain_mask = (pack["t1"] > self.criterion.weights.mask_threshold).float()
                sure = mc_sure_term(
                    model, pack["setA"], pack["setB"], pack["t1"], brain_mask,
                    len_a=pack.get("lenA"), len_b=pack.get("lenB"),
                    mask_a=pack.get("maskA"), mask_b=pack.get("maskB"),
                    eps=float(self.criterion.weights.sure_eps),
                )
                # Clamp the raw SURE value to prevent rare blowups from killing
                # the run. We use |SURE| as the actual loss term so that BOTH
                # under-denoising (SURE >> 0) AND over-smoothing (SURE << 0,
                # arising from MC variance + small fid) are penalised — pushing
                # the estimator toward its theoretical optimum at 0.
                # Using raw `sure` (signed) caused over-smooth collapse:
                # negative SURE batches received gradient maximising |SURE| in
                # the wrong direction (rewarding more smoothing).
                sure = sure.clamp(min=-1.0, max=1.0)
                loss = loss + w_sure_eff * sure.abs()
                stats["loss_sure"] = float(sure.detach().item())              # raw, for diagnostics
                stats["loss_sure_abs"] = float(sure.detach().abs().item())   # what's added to loss
                stats["loss"] = float(loss.detach().item())

        # Innovation X — split-half reproducible-HF texture loss. Real texture =
        # high-frequency structure REPRODUCIBLE across two DISJOINT frame subsets
        # (independent noise realisations). Reward reproducible HF power in tissue,
        # PENALISE it in CSF (ΔM≈0 ⇒ any reproducible HF there is common-mode
        # artifact, not perfusion). Training-time analogue of the tissue/CSF
        # HF-ratio + split-half consistency eval metrics; self-supervised, T1-free.
        #
        # POWER form E[Lap(y1)·Lap(y2)]: independent noise decorrelates ⇒ its
        # cross-term is 0 in expectation at ANY frame count (small pool only raises
        # variance, not bias). Blind spot: frame-CORRELATED artifacts appear in
        # both halves and get rewarded — held in check by keeping w_tex small (N2N
        # anchors input-dependence), the CSF penalty (diffuse common-mode HF shows
        # in CSF), and the mismatched-T1 gate (T1-driven HF fails it). Uses the FULL
        # pool (setA∪setB) split in two, decoupled from the N2N a/b partition.
        w_tex = float(getattr(self.args, "w_tex", 0.0))
        w_tex_csf = float(getattr(self.args, "w_tex_csf", 0.0))
        if w_tex > 0.0 or w_tex_csf > 0.0:
            from utils.metrics import _laplacian_response
            repro = None; tex_pool = 0.0
            if bool(getattr(self.args, "tex_use_setb", False)):
                # Reuse f(set_a) [main, grad] vs stop-grad f(set_b). f(set_b) is the
                # bdcyc forward if it ran, else one extra forward here. Target-style
                # (only pred_a has grad) — cheaper + avoids the trivial-constant attractor.
                pred_a = outputs["asl_recon"]
                if _setb_recon is not None:
                    pred_b = _setb_recon
                else:
                    with torch.no_grad():
                        pred_b = model.infer_from_subset(
                            pack["setB"], pack["t1"],
                            lengths=pack.get("lenB"), mask=pack.get("maskB"))["asl_recon"]
                repro = _laplacian_response(pred_a) * _laplacian_response(pred_b)
                lb = pack.get("lenB")
                tex_pool = float(lb.float().mean().item()) if lb is not None else float(pack["setB"].shape[1])
            else:
                h1, l1, h2, l2 = self._pool_split_halves(
                    pack, min_pool=int(getattr(self.args, "tex_min_pool", 4)))
                if h1 is not None:
                    out1 = model.infer_from_subset(h1, pack["t1"], lengths=l1)
                    out2 = model.infer_from_subset(h2, pack["t1"], lengths=l2)
                    repro = _laplacian_response(out1["asl_recon"]) * \
                            _laplacian_response(out2["asl_recon"])
                    tex_pool = float((l1 + l2).float().mean().item())
            if repro is not None:                              # signed reproducible HF power
                gm = pack.get("gm"); wm = pack.get("wm")
                if gm is not None and wm is not None:
                    tissue = ((gm + wm) > 0.5).float()
                else:
                    tissue = (pack["t1"] > self.criterion.weights.mask_threshold).float()
                power_t = (repro * tissue).sum() / tissue.sum().clamp_min(1.0)
                loss_tex = -w_tex * power_t
                stats["tex_power_tissue"] = float(power_t.detach().item())
                csf = pack.get("csf")
                if w_tex_csf > 0.0 and csf is not None:
                    csf_m = (csf > 0.5).float()
                    power_c = (repro * csf_m).sum() / csf_m.sum().clamp_min(1.0)
                    loss_tex = loss_tex + w_tex_csf * power_c
                    stats["tex_power_csf"] = float(power_c.detach().item())
                loss = loss + loss_tex
                stats["loss_tex"] = float(loss_tex.detach().item())
                stats["tex_pool"] = tex_pool
                stats["loss"] = float(loss.detach().item())
                if not getattr(self, "_tex_logged", False):
                    logging.info(
                        "[tex] ACTIVE w_tex=%.4f w_tex_csf=%.4f setb=%s power_tissue=%.5f",
                        w_tex, w_tex_csf, bool(getattr(self.args, "tex_use_setb", False)),
                        float(power_t.detach().item()))
                    self._tex_logged = True
        return loss, stats

    def _pool_split_halves(
        self, pack: Dict[str, Tensor], min_pool: int = 4
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Split the full frame pool (setA ∪ setB) into two DISJOINT random halves
        per sample, for the reproducible-HF texture loss. Fresh split every call
        (decoupled from the N2N a/b partition). Returns padded (h1, l1, h2, l2) each
        [B, kmax, 1, H, W] / [B], or four Nones if any sample's pool < min_pool
        (so both halves always have >= min_pool//2 frames)."""
        setA = pack["setA"]; setB = pack["setB"]
        B, TA = setA.shape[0], setA.shape[1]
        dev = setA.device
        lenA = pack.get("lenA");  lenB = pack.get("lenB")
        if lenA is None:
            lenA = torch.full((B,), TA, device=dev, dtype=torch.long)
        if lenB is None:
            lenB = torch.full((B,), setB.shape[1], device=dev, dtype=torch.long)
        total = (lenA + lenB).to(torch.long)
        if int(total.min().item()) < int(min_pool):
            return None, None, None, None
        half = (total // 2).to(torch.long)
        kmax = int(half.max().item())
        h1 = setA.new_zeros((B, kmax) + tuple(setA.shape[2:]))
        h2 = setA.new_zeros((B, kmax) + tuple(setA.shape[2:]))
        l1 = torch.zeros(B, device=dev, dtype=torch.long)
        l2 = torch.zeros(B, device=dev, dtype=torch.long)
        for i in range(B):
            la = int(lenA[i].item()); lb = int(lenB[i].item())
            pool_i = torch.cat([setA[i, :la], setB[i, :lb]], dim=0)   # [total_i, 1, H, W]
            ti = pool_i.shape[0]; hi = ti // 2
            perm = torch.randperm(ti, device=dev)
            h1[i, :hi] = pool_i[perm[:hi]]
            h2[i, :hi] = pool_i[perm[hi:2 * hi]]
            l1[i] = hi; l2[i] = hi
        return h1, l1, h2, l2

    @torch.no_grad()
    def _predict_with_ema(self, set_x: Tensor, t1: Tensor, lengths: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        base_model = self._unwrap()
        base_model.eval()
        self.ema.store(base_model)
        self.ema.copy_to(base_model)
        pred = base_model.infer_from_subset(set_x, t1, lengths=lengths, mask=mask)
        self.ema.restore(base_model)
        return pred

    @torch.no_grad()
    def _compute_sure_val(self, pack: Dict[str, Tensor], brain_mask: Tensor, eps: float = 1e-3) -> float:
        """Monte-Carlo SURE estimate of MSE-to-clean for the val batch using EMA weights.

        SURE is a noisy-ref-free metric of denoising quality (low → close to clean GT).
        Reuses losses.asl_n2n_loss.mc_sure_term but applies EMA weights first.
        """
        if int(getattr(self.args, "slice_context", 0)) > 0:
            return float("nan")   # mc_sure_term compares to K-slice-window frames; skip for 2.5D (supplementary)
        try:
            from losses.asl_n2n_loss import mc_sure_term
        except Exception:
            return float("nan")
        base_model = self._unwrap()
        base_model.eval()
        self.ema.store(base_model)
        self.ema.copy_to(base_model)
        try:
            sure = mc_sure_term(
                base_model, pack["setA"], pack["setB"], pack["t1"], brain_mask,
                len_a=pack.get("lenA"), len_b=pack.get("lenB"),
                mask_a=pack.get("maskA"), mask_b=pack.get("maskB"),
                eps=float(eps),
            )
            sure_v = float(sure.clamp(min=-1.0, max=1.0).item())
        except Exception:
            sure_v = float("nan")
        finally:
            self.ema.restore(base_model)
        return sure_v

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, float, float, float]:
        """
        Returns (l1_vs_B, psnr_vs_B, ssim_vs_B, l1_vs_union, psnr_vs_union, ssim_vs_union).

        *_vs_B    — N2N held-out estimate (used for model selection, unbiased).
        *_vs_union — vs. full 12-NEX mean (A∪B); gold-standard reference metric.

        Additionally accumulates GMSD / GM-WM contrast error / Laplacian-variance ratio
        and writes them as TensorBoard scalars at the end of the call (no impact on
        model-selection criterion, just for monitoring).
        """
        from utils.metrics import (
            gmsd, gm_wm_contrast_error, laplacian_variance, hfen,
            tenengrad, image_entropy, gradient_entropy,
            entropy_focus_criterion, scov, unsupervised_psnr,
            upsnr_components, cnr,
        )
        self.model.eval()
        acc = {k: 0.0 for k in (
            "l1_b", "psnr_b", "ssim_b", "l1_ref", "psnr_ref", "ssim_ref",
            "gmsd", "gmwm_err", "lapvar_ratio",
            "hfen", "nrmse", "sure",
            "tg_pred", "tg_ref", "ie_pred", "ie_ref", "ge_pred", "ge_ref",
            "efc_pred", "efc_ref", "scov_gm_pred", "scov_gm_ref",
            "scov_wm_pred", "scov_wm_ref",
            "cnr_pred", "cnr_ref",
            "subset_consistency",
        )}
        # Pooled-aggregation accumulators for uPSNR (avoid log-of-mean artefact)
        upsnr_acc = {"sum_sq": 0.0, "sum_vc": 0.0, "n": 0.0,
                     "sum_sq_ref": 0.0, "sum_vc_ref": 0.0, "n_ref": 0.0,
                     "noise_floor_l1_sum": 0.0}
        scov_count = 0
        scov_wm_count = 0
        cnr_count = 0
        cyc_count = 0
        gmwm_count = 0
        n = 0
        saved = 0

        it = tqdm(self.val_loader, desc=f"Validate (step={self.global_step})", dynamic_ncols=True)
        for vb in it:
            pack = prepare_asl_pair_batch(vb, self.device)
            # Same input masking as training (replaces loss_bg).
            pack = self._mask_asl_inputs(pack)
            # cond_src='fsl': set the per-batch FSL-PV conditioning on the (shared,
            # unwrapped) model instance so every _predict_with_ema / _compute_sure_val
            # forward below picks it up. No-op for pv/rawt1.
            self._apply_cond_pv(self._unwrap(), pack)

            # 2.5D: reference images/metrics use the CENTER slice of the K-slice window
            # (dim=2), matching the model's center-slice recon (out is [B,1,H,W]). The
            # model itself still receives the FULL window via pack["setA"] below.
            _ctx = int(getattr(self.args, "slice_context", 0))
            setA_c, setB_c = pack["setA"], pack["setB"]
            if _ctx > 0:
                _n = 2 * _ctx + 1
                _ic = pack["setA"].shape[2] // _n           # ASL in_ch (=1 for ΔM)
                _c0 = _ic * _ctx
                setA_c = pack["setA"][:, :, _c0:_c0 + _ic]
                setB_c = pack["setB"][:, :, _c0:_c0 + _ic]
            meanA = direct_mean_from_frames(setA_c, pack.get("lenA"))
            meanB = direct_mean_from_frames(setB_c, pack.get("lenB"))
            # union = mean of ALL frames (12-NEX gold-standard reference PWI)
            union = direct_mean_from_frames(
                torch.cat([setA_c, setB_c], dim=1),
                pack["lenA"] + pack["lenB"],
            )

            # Innovation T (PID) deployment mode: replace the model's T1 input with
            # zeros if --pid_zero_t1_at_eval is set. The brain mask + reference metrics
            # still use the real T1 (mask quality is unaffected).
            t1_for_model = (torch.zeros_like(pack["t1"])
                            if bool(getattr(self.args, "pid_zero_t1_at_eval", False))
                            else pack["t1"])
            pred = self._predict_with_ema(pack["setA"], t1_for_model, pack["lenA"], pack.get("maskA"))
            out = pred["asl_recon"]
            baseline = pred.get("agg", None)
            weights  = pred.get("weights", None)
            t1_rec   = pred.get("t1_recon", None)
            t1_seg_logits = pred.get("t1_seg", None)

            # vs held-out set B (N2N estimate)
            l1_b, (psnr_b, ssim_b) = F.l1_loss(out, meanB).item(), _compute_psnr_ssim(out, meanB)
            # vs 12-NEX union (gold-standard)
            l1_ref, (psnr_ref, ssim_ref) = F.l1_loss(out, union).item(), _compute_psnr_ssim(out, union)

            acc["l1_b"]    += l1_b;    acc["psnr_b"]   += psnr_b;   acc["ssim_b"]   += ssim_b
            acc["l1_ref"]  += l1_ref;  acc["psnr_ref"] += psnr_ref; acc["ssim_ref"] += ssim_ref

            # Sharpness / contrast-aware metrics (TB-only, not used for model selection)
            mask_brain = (pack["t1"] > 0.05).float()
            ref_lapvar = laplacian_variance(union, mask_brain)
            pred_lapvar = laplacian_variance(out, mask_brain)
            acc["gmsd"] += gmsd(out, union, mask_brain)
            acc["lapvar_ratio"] += pred_lapvar / max(ref_lapvar, 1e-8)
            if "gm" in pack and "wm" in pack:
                acc["gmwm_err"] += gm_wm_contrast_error(out, union, pack["gm"], pack["wm"])
                gmwm_count += 1
            # HFEN: noise-tolerant high-frequency error vs reference
            acc["hfen"] += hfen(out, union, mask_brain)
            # No-reference IQA: Tenengrad (sharpness), Image Entropy (intensity
            # diversity), Gradient Entropy (textural richness). Reported for
            # both pred and ref (12-NEX union) — the *closeness* between
            # pred/ref values is the comparison the paper will make.
            acc["tg_pred"] += tenengrad(out, mask_brain)
            acc["tg_ref"]  += tenengrad(union, mask_brain)
            acc["ie_pred"] += image_entropy(out, mask_brain)
            acc["ie_ref"]  += image_entropy(union, mask_brain)
            acc["ge_pred"] += gradient_entropy(out, mask_brain)
            acc["ge_ref"]  += gradient_entropy(union, mask_brain)
            # EFC: Atkinson 1997 — MRI motion/blur metric (lower = sharper)
            acc["efc_pred"] += entropy_focus_criterion(out, mask_brain)
            acc["efc_ref"]  += entropy_focus_criterion(union, mask_brain)
            # sCoV in GM / WM: scale-invariant homogeneity (Wang 2003 ASL standard)
            # WM has lower mean signal → sCoV more sensitive to noise (potentially
            # better dynamic range than GM).
            if "gm" in pack:
                acc["scov_gm_pred"] += scov(out, pack["gm"], threshold=0.5)
                acc["scov_gm_ref"]  += scov(union, pack["gm"], threshold=0.5)
                scov_count += 1
            if "wm" in pack:
                acc["scov_wm_pred"] += scov(out, pack["wm"], threshold=0.5)
                acc["scov_wm_ref"]  += scov(union, pack["wm"], threshold=0.5)
                scov_wm_count += 1
            # uPSNR: Marcos-Morales et al. ICML 2023 — unsupervised PSNR using
            # 3-way disjoint split of held-out set_b. Theoretically unbiased
            # estimator of PSNR-to-clean. Accumulate components for **pooled**
            # aggregation (single uMSE over the whole val set, then one log).
            ssq, svc, npx, nfl1 = upsnr_components(out, setB_c, lengths_b=pack.get("lenB"),
                                                    mask=mask_brain)
            upsnr_acc["sum_sq"] += ssq
            upsnr_acc["sum_vc"] += svc
            upsnr_acc["n"]      += npx
            upsnr_acc["noise_floor_l1_sum"] += nfl1
            # Same uPSNR computed on union-vs-pred would be the biased version;
            # for sanity, also compute on the union as "uPSNR_ref".
            ssq_r, svc_r, npx_r, _ = upsnr_components(union, setB_c, lengths_b=pack.get("lenB"),
                                                       mask=mask_brain)
            upsnr_acc["sum_sq_ref"] += ssq_r
            upsnr_acc["sum_vc_ref"] += svc_r
            upsnr_acc["n_ref"]      += npx_r
            # CNR (GM-WM) — Wang ASL convention; σ_WM as noise reference
            if "gm" in pack and "wm" in pack:
                acc["cnr_pred"] += cnr(out, pack["gm"], pack["wm"], threshold=0.5)
                acc["cnr_ref"]  += cnr(union, pack["gm"], pack["wm"], threshold=0.5)
                cnr_count += 1
            # Sub-set self-consistency: |f(set_a[:k]) - f(set_a[k:])|_1 — model
            # stability under input sub-sampling. Low = robust denoising,
            # not chasing input-noise idiosyncrasies.
            T_a_min = int(pack["lenA"].min().item())
            if T_a_min >= 4:
                k = T_a_min // 2
                set_a1 = pack["setA"][:, :k]
                set_a2 = pack["setA"][:, k:2 * k]
                len_a1 = torch.full_like(pack["lenA"], k)
                pred1 = self._predict_with_ema(set_a1, t1_for_model, len_a1)["asl_recon"]
                pred2 = self._predict_with_ema(set_a2, t1_for_model, len_a1)["asl_recon"]
                cyc_l1 = ((pred1 - pred2).abs() * mask_brain).sum() / mask_brain.sum().clamp_min(1.0)
                acc["subset_consistency"] += float(cyc_l1.item())
                cyc_count += 1
            # NRMSE: scale-invariant L1 (l1_ref / mean(union inside brain))
            ref_mean = (union * mask_brain).sum() / mask_brain.sum().clamp_min(1.0)
            acc["nrmse"] += float(((out - union).abs() * mask_brain).sum().item()
                                  / (mask_brain.sum().clamp_min(1.0).item()
                                     * max(ref_mean.item(), 1e-8)))

            # Monte-Carlo SURE estimate of MSE-to-clean (noisy-ref-free,
            # supplementary reported metric). One extra EMA forward per batch.
            sure_v = self._compute_sure_val(pack, mask_brain)
            if not (sure_v != sure_v):  # not NaN
                acc["sure"] += sure_v

            n += 1
            it.set_postfix(l1_B=f"{l1_b:.4f}", psnr_ref=f"{psnr_ref:.2f}", sure=f"{sure_v:.4f}")

            if self.args.save_images and saved < self.args.log_images:
                try:
                    # Brain mask for VISUALISATION ONLY: prefer the learned BG channel
                    # from the (frozen) T1 stage-1 segmentation — `mask = 1 - sigmoid(BG)`
                    # — because it has clean smooth boundaries and includes CSF/ventricles
                    # at their proper soft-edge. Falls back to (t1 > 0.05) if seg head
                    # is missing or has < 4 channels. Metrics above still use t1>0.05
                    # for numerical comparability with earlier experiments.
                    #
                    # --zero_t1: the seg logits come from a BLANK T1, so the BG channel
                    # is a degenerate near-uniform field that does not crop to the brain.
                    # The ASL-difference panels are ~0 outside the brain so they still look
                    # clean, but the decoder's ASL recon fills the whole FOV with a smooth
                    # field, and a near-uniform mask leaves that background box visible —
                    # which reads as a "cropped/boxy" recon. Use the real-T1 brain mask for
                    # the visualisation in that case so the panels stay comparable.
                    if (t1_seg_logits is not None and t1_seg_logits.size(1) >= 4
                            and not bool(getattr(self.args, "zero_t1", False))):
                        viz_mask = (1.0 - torch.sigmoid(t1_seg_logits[:, 3:4])).clamp(0.0, 1.0)
                    else:
                        viz_mask = mask_brain
                    bm_np = viz_mask[0, 0].detach().cpu().numpy()
                    def _m(im):
                        return im * bm_np
                    out_np      = _m(out[0, 0].detach().cpu().numpy())
                    union_np    = _m(union[0, 0].detach().cpu().numpy())
                    meanA_np    = _m(meanA[0, 0].detach().cpu().numpy())
                    meanB_np    = _m(meanB[0, 0].detach().cpu().numpy())
                    # Shared vmin/vmax across PWI panels using 0.5/99.5 percentile
                    # of the union reference — wider than 5/95 to preserve true
                    # high/low extremes while still trimming hot pixel outliers.
                    vmin, vmax = float(np.percentile(union_np, 0.5)), float(np.percentile(union_np, 99.5))

                    imgs: List[Tuple[np.ndarray, str, float, float]] = [
                        (pack["t1"][0, 0].detach().cpu().numpy(),  "T1w",                      None, None),
                        (union_np,                                 "12-NEX union (ref)",       vmin, vmax),
                        (meanA_np,                                 "A direct mean",            vmin, vmax),
                        (meanB_np,                                 "B direct mean (N2N tgt)",  vmin, vmax),
                    ]
                    if baseline is not None:
                        imgs.append((_m(baseline[0, 0].detach().cpu().numpy()), "A learned agg", vmin, vmax))
                    imgs.append((out_np, "ASL recon", vmin, vmax))
                    if t1_rec is not None:
                        imgs.append((_m(t1_rec[0, 0].detach().cpu().numpy()), "T1 recon", None, None))
                    imgs.append(((out_np - union_np).__abs__(), "|recon − union|", None, None))

                    n_img = len(imgs)
                    rows = 1 if n_img <= 4 else 2
                    cols = int(np.ceil(n_img / rows))
                    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
                    axes = np.array(axes).reshape(-1)
                    for i, (im, title, vlo, vhi) in enumerate(imgs):
                        if vlo is not None and vhi is not None:
                            axes[i].imshow(im, cmap="gray", vmin=vlo, vmax=vhi)
                        else:
                            axes[i].imshow(im, cmap="gray")
                        axes[i].set_title(title, fontsize=7)
                        axes[i].axis("off")
                    for j in range(i + 1, len(axes)):
                        axes[j].axis("off")
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(self.val_img_dir, f"step{self.global_step}_val{saved}_summary.png"),
                        dpi=80, bbox_inches="tight",
                    )
                    plt.close(fig)

                    # T1 segmentation panel: 4-class GM/WM/CSF/BG (sigmoid prediction)
                    # Top row: pred. Bottom row: GT (BG derived as 1-(GM+WM+CSF)).
                    if t1_seg_logits is not None and t1_seg_logits.size(1) >= 4 and "gm" in pack:
                        # Render with SOFTMAX (the 4 PV classes are a simplex) to match how the
                        # frozen prior was trained (phase1 seg default --seg_softmax). Using
                        # sigmoid here mis-renders softmax-trained logits: GM's absolute logit is
                        # unconstrained under softmax, so per-channel sigmoid blows GM up into a
                        # diffuse blob across the whole brain (WM/CSF look fine) — a viz artifact,
                        # not a bad prior. Diagnostic-only; no effect on model / loss / selection.
                        prob = torch.softmax(t1_seg_logits[0], dim=0).detach().cpu().numpy()  # [4,H,W]
                        gt_gm  = pack["gm"][0, 0].detach().cpu().numpy()
                        gt_wm  = pack["wm"][0, 0].detach().cpu().numpy()
                        gt_csf = pack["csf"][0, 0].detach().cpu().numpy()
                        gt_bg  = (1.0 - (gt_gm + gt_wm + gt_csf)).clip(0, 1)
                        gt = [gt_gm, gt_wm, gt_csf, gt_bg]
                        titles = ["GM", "WM", "CSF", "BG"]
                        fig3, ax3 = plt.subplots(2, 4, figsize=(12, 6))
                        for ci in range(4):
                            ax3[0, ci].imshow(prob[ci], cmap="gray", vmin=0, vmax=1)
                            ax3[0, ci].set_title(f"pred {titles[ci]}", fontsize=8); ax3[0, ci].axis("off")
                            ax3[1, ci].imshow(gt[ci], cmap="gray", vmin=0, vmax=1)
                            ax3[1, ci].set_title(f"GT {titles[ci]}", fontsize=8); ax3[1, ci].axis("off")
                        fig3.tight_layout()
                        fig3.savefig(
                            os.path.join(self.val_img_dir, f"step{self.global_step}_val{saved}_seg.png"),
                            dpi=80, bbox_inches="tight",
                        )
                        plt.close(fig3)

                    if weights is not None:
                        valid_t = int(pack["lenA"][0].item())
                        fig2 = plt.figure(figsize=(3 * valid_t, 3))
                        ax2 = fig2.subplots(1, valid_t)
                        if valid_t == 1:
                            ax2 = [ax2]
                        for frame in range(valid_t):
                            im = pack["setA"][0, frame, 0].detach().cpu().numpy()
                            w_t = weights[0, frame]
                            # SetTransformer: scalar [B,T]; SVFW: per-pixel [B,T,1,H,W]
                            w = float(w_t.mean()) if w_t.numel() > 1 else float(w_t)
                            ax2[frame].imshow(im, cmap="gray")
                            ax2[frame].set_title(f"frame {frame}\nw={w:.3f}", fontsize=7)
                            ax2[frame].axis("off")
                        fig2.tight_layout()
                        fig2.savefig(
                            os.path.join(self.val_img_dir, f"step{self.global_step}_val{saved}_weights.png"),
                            dpi=80, bbox_inches="tight",
                        )
                        plt.close(fig2)

                    saved += 1
                except Exception as e:
                    plt.close("all")
                    logging.warning(f"Failed to save val image (step={self.global_step}): {e}")

        m = max(1, n)
        # Stash sharpness metrics on self for the train loop to log to TB.
        gw = (acc["gmwm_err"] / gmwm_count) if gmwm_count > 0 else float("nan")
        self._last_val_extras = {
            "gmsd":         acc["gmsd"] / m,
            "gmwm_err":     gw,
            "lapvar_ratio": acc["lapvar_ratio"] / m,
            "hfen":         acc["hfen"] / m,
            "nrmse":        acc["nrmse"] / m,
            "sure":         acc["sure"] / m,
            # No-reference IQA: pred / ref values; the closeness between them
            # quantifies how well the recon preserves sharpness, intensity
            # diversity, and textural richness without needing a clean GT.
            "tg_pred":      acc["tg_pred"] / m,
            "tg_ref":       acc["tg_ref"]  / m,
            "ie_pred":      acc["ie_pred"] / m,
            "ie_ref":       acc["ie_ref"]  / m,
            "ge_pred":      acc["ge_pred"] / m,
            "ge_ref":       acc["ge_ref"]  / m,
            "efc_pred":     acc["efc_pred"] / m,
            "efc_ref":      acc["efc_ref"]  / m,
            "scov_gm_pred": (acc["scov_gm_pred"] / scov_count) if scov_count > 0 else float("nan"),
            "scov_gm_ref":  (acc["scov_gm_ref"]  / scov_count) if scov_count > 0 else float("nan"),
            "scov_wm_pred": (acc["scov_wm_pred"] / scov_wm_count) if scov_wm_count > 0 else float("nan"),
            "scov_wm_ref":  (acc["scov_wm_ref"]  / scov_wm_count) if scov_wm_count > 0 else float("nan"),
            "cnr_pred": (acc["cnr_pred"] / cnr_count) if cnr_count > 0 else float("nan"),
            "cnr_ref":  (acc["cnr_ref"]  / cnr_count) if cnr_count > 0 else float("nan"),
            "subset_consistency": (acc["subset_consistency"] / cyc_count) if cyc_count > 0 else float("nan"),
            # Pooled uPSNR: single uMSE over whole val set, then 10·log10(1/uMSE)
            "upsnr":     _pool_upsnr(upsnr_acc["sum_sq"], upsnr_acc["sum_vc"], upsnr_acc["n"]),
            "upsnr_ref": _pool_upsnr(upsnr_acc["sum_sq_ref"], upsnr_acc["sum_vc_ref"], upsnr_acc["n_ref"]),
            # v42k 2026-05-20: uMSE (linear-scale risk) + per-pixel L1 noise floor
            # (mean |b−c| across val pixels). These are the inputs to the
            # constrained-uMSE model-selection criterion (see best_criterion=
            # 'constrained_umse'): primary risk = uMSE; consistency threshold
            # τ_C = ρ · noise_floor_l1.
            "umse":           _pool_umse(upsnr_acc["sum_sq"], upsnr_acc["sum_vc"], upsnr_acc["n"]),
            "noise_floor_l1": (upsnr_acc["noise_floor_l1_sum"] / upsnr_acc["n"]) if upsnr_acc["n"] > 0 else float("nan"),
        }
        return (acc["l1_b"]/m, acc["psnr_b"]/m, acc["ssim_b"]/m,
                acc["l1_ref"]/m, acc["psnr_ref"]/m, acc["ssim_ref"]/m)

    def train(self, writer: SummaryWriter):
        max_steps = int(self.args.max_steps) if int(self.args.max_steps) > 0 \
                     else int(self.cfg.asl_denoiser_train_params.max_steps)
        eval_every = (int(self.args.eval_every) if int(getattr(self.args, "eval_every", 0)) > 0
                      else int(self.cfg.asl_denoiser_train_params.eval_every))

        # v42h (2026-05-19): linear LR warmup over first --warmup_steps outer steps.
        warmup_steps = int(getattr(self.args, "warmup_steps", 0))
        base_lr = float(self.cfg.asl_denoiser_train_params.lr)

        # One "step" is one EPOCH, so every epoch-indexed hyper-parameter below is
        # implicitly denominated in batches-per-epoch — i.e. in COHORT SIZE. Growing
        # the dataset silently rescales all of them at once. Print the budget so a
        # mismatch is visible at launch instead of at the wall clock, and warn on the
        # combinations that are silently degenerate.
        _bpe = len(self.train_loader)
        logging.info(
            f"[budget] {_bpe} batches/epoch x {max_steps} epochs = {_bpe * max_steps} optimizer "
            f"steps | warmup={warmup_steps}ep best_min={self.args.best_min_step}ep "
            f"save_every={self.args.save_every}ep eval_every={eval_every}ep "
            f"=> ~{max_steps // max(eval_every, 1)} evals, "
            f"~{max_steps // max(int(self.args.save_every), 1)} step-ckpts"
        )
        if warmup_steps >= max_steps:
            logging.warning(f"[budget] warmup_steps={warmup_steps} >= max_steps={max_steps}: "
                            "the LR never finishes ramping and cosine never anneals.")
        if int(self.args.best_min_step) >= max_steps:
            logging.warning(f"[budget] best_min_step={self.args.best_min_step} >= max_steps="
                            f"{max_steps}: best.pth can NEVER update.")
        if max_steps // max(eval_every, 1) < 5:
            logging.warning(f"[budget] only ~{max_steps // max(eval_every, 1)} validations will run "
                            f"(max_steps={max_steps}, eval_every={eval_every}): too few for the "
                            "score EMA and best-ckpt selection to mean anything.")

        while self.global_step < max_steps:
            self.global_step += 1
            # Phase 2 Run C: release gate warmup as soon as we cross threshold.
            # `_mossm_gate_warmup_released` makes this idempotent if resume picks
            # up mid-warmup.
            if (self._mossm_gate_warmup_steps > 0
                    and not self._mossm_gate_warmup_released
                    and self.global_step > self._mossm_gate_warmup_steps):
                for prm in self._mossm_gate_params:
                    prm.requires_grad = True
                self._mossm_gate_warmup_released = True
                logging.info(
                    f"[gate_warmup] released {len(self._mossm_gate_params)} "
                    f"t1_gate_bd params at step {self.global_step}; gradient "
                    "now flows through gate."
                )
            if warmup_steps > 0 and self.global_step <= warmup_steps:
                ramp = float(self.global_step) / float(warmup_steps)
                # Preserve per-group LR ratios (e.g. T1-path 5×) by ramping
                # each group's `initial_lr` rather than overwriting all groups
                # to `base_lr * ramp`. `initial_lr` is set by AdamW when a
                # scheduler attaches; fall back to per-group base if absent.
                for g in self.optimizer.param_groups:
                    g_init = float(g.get("initial_lr", g.get("lr", base_lr)))
                    g["lr"] = g_init * ramp
            pbar = tqdm(self.train_loader, desc=f"Train ({self.global_step}/{max_steps})", dynamic_ncols=True)
            self.model.train()
            # Frozen T1 branch must stay in EVAL even during training: --freeze_t1
            # only stops gradients (requires_grad=False), it does NOT disable the
            # decoder's Dropout2d(skip_dropout=0.3). Left in train() mode it would
            # randomly zero 30% of the T1 seg-head skips every step, so the soft
            # brain mask (_mask_asl_inputs) and TABS prior would jitter per step —
            # a train/eval mismatch (eval already runs model.eval()). Pin to eval.
            if bool(getattr(self.args, "freeze_t1", False)):
                m = self._unwrap()
                if getattr(m, "t1_encoder", None) is not None: m.t1_encoder.eval()
                if getattr(m, "t1_decoder", None) is not None: m.t1_decoder.eval()

            last_stats: Dict = {}
            last_loss: float = 0.0

            for batch in pbar:
                pack = prepare_asl_pair_batch(batch, self.device)
                pack = self._maybe_flip(pack)   # train-only spatial flip aug (consistent setA/setB/T1/PV, N2N-safe)
                # T1 dropout (2026-05-18): zero out T1 input for this step with
                # prob --t1_dropout_p. Forces ASL-driven feature extraction and
                # weakens the T1→output anatomy-routing circuit identified by
                # the v42f mismatched-T1 probe (16.4% leak).
                t1_drop_p = float(getattr(self.args, "t1_dropout_p", 0.0))
                if t1_drop_p > 0.0 and torch.rand((), device=pack["t1"].device).item() < t1_drop_p:
                    pack["t1"] = torch.zeros_like(pack["t1"])
                if bool(getattr(self.args, "amp", False)):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss, stats = self._forward_train(pack)
                else:
                    loss, stats = self._forward_train(pack)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # T1-path activity probe: at sparse intervals, capture grad
                # magnitudes of the T1-residual components so we can tell
                # whether MoSSM gate / x_proj_BD path is being trained. Cheap:
                # iterate named_parameters() once, sum abs grads, no extra fwd.
                if (self.global_step % 20 == 0):
                    g_gate, g_bd_t1, g_bd_asl = 0.0, 0.0, 0.0
                    n_gate = n_bd_t1 = n_bd_asl = 0
                    for name, prm in self._unwrap().named_parameters():
                        if prm.grad is None:
                            continue
                        if "t1_gate_bd" in name:
                            g_gate += float(prm.grad.detach().abs().sum().item())
                            n_gate += int(prm.numel())
                        elif "x_proj_BD_asl" in name:
                            g_bd_asl += float(prm.grad.detach().abs().sum().item())
                            n_bd_asl += int(prm.numel())
                        elif "x_proj_BD" in name and "_asl" not in name:
                            g_bd_t1 += float(prm.grad.detach().abs().sum().item())
                            n_bd_t1 += int(prm.numel())
                    if n_gate > 0 or n_bd_t1 > 0 or n_bd_asl > 0:
                        writer.add_scalar("probe/grad_t1_gate_bd",
                                          g_gate / max(n_gate, 1), self.global_step)
                        writer.add_scalar("probe/grad_x_proj_BD_t1",
                                          g_bd_t1 / max(n_bd_t1, 1), self.global_step)
                        writer.add_scalar("probe/grad_x_proj_BD_asl",
                                          g_bd_asl / max(n_bd_asl, 1), self.global_step)
                        # Ratio surfaces "is T1 path getting comparable training
                        # pressure to the ASL path?". <0.1 ⇒ T1 path starved.
                        if g_bd_asl > 0:
                            writer.add_scalar(
                                "probe/grad_ratio_t1_over_asl",
                                (g_bd_t1 / max(n_bd_t1, 1)) /
                                ((g_bd_asl / max(n_bd_asl, 1)) + 1e-12),
                                self.global_step,
                            )
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                          float(getattr(self.args, "grad_clip", 1.0)))
                self.optimizer.step()
                self.ema.step(self._unwrap())

                last_loss = stats["loss"]
                last_stats = stats
                pbar.set_postfix(loss=f"{last_loss:.4f}")

            # Skip cosine step during linear warmup (LR is set manually above).
            if self.scheduler is not None and self.global_step > warmup_steps:
                self.scheduler.step()

            # SWA running mean update (Izmailov ICLR 2018). Activates from
            # --swa_start_step onwards; updates once per epoch (post-step).
            if (self.swa_model is not None
                and self.global_step >= int(self.args.swa_start_step)):
                self.swa_model.update_parameters(self._unwrap())
                self.swa_n_avg += 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, self.global_step)
            writer.add_scalar("train/total", last_loss, self.global_step)
            # Window-fusion probes (once per epoch, floats stashed by the last
            # forward): gate = how much guidance the model asked for; entropy ==
            # ln(window tokens) means the grouping has NOT been learnt and the
            # module is still a box blur; delta = per-level feature perturbation.
            for _wk, _wv in self._unwrap().window_stats().items():
                writer.add_scalar(f"window/{_wk}", _wv, self.global_step)
            for key in ("loss_n2n", "loss_t1", "loss_cos", "loss_tv", "loss_bg",
                        "loss_grad", "loss_ssim", "loss_seg", "loss_contrast",
                        "loss_adv", "adv_triggered", "loss_pid", "loss_sharp",
                        "probe_agg_entropy", "probe_agg_w_bad",
                        "probe_agg_w_normal_per_frame",
                        # EC-LRDA: aux losses + leg-liveness probes (ASL_dmvae/docs/experiment_plan_ec_min.md ④).
                        # r_rep learns iff art<cln; c_sem learns iff match>mismatch.
                        "loss_rep", "loss_sem",
                        "probe_r_rep_art", "probe_r_rep_cln",
                        "probe_c_match", "probe_c_mismatch",
                        # DW-N2N anomaly-aware deviation weighting monitors.
                        "dw_mean", "dw_active_frac", "dw_boundary_frac"):
                if key in last_stats:
                    writer.add_scalar(f"train/{key}", last_stats[key], self.global_step)

            # Step-tagged checkpoint for post-hoc best-CNR selection. Ungated by
            # eval_every so --save_every N literally means "every N epochs" — older
            # code nested this inside the eval block below, so a checkpoint only
            # landed on epochs divisible by BOTH save_every AND eval_every (e.g.
            # save_every=25 + eval_every=10 => saves at {50,100,150,200}, not /25).
            if self.args.save_every > 0 and self.global_step % self.args.save_every == 0:
                self._save(f"step{self.global_step:06d}")

            if self.global_step % eval_every == 0 or self.global_step >= max_steps:
                l1_b, psnr_b, ssim_b, l1_ref, psnr_ref, ssim_ref = self.validate()
                # N2N held-out (model selection criterion)
                writer.add_scalar("val/l1_vs_B",    l1_b,    self.global_step)
                writer.add_scalar("val/psnr_vs_B",  psnr_b,  self.global_step)
                writer.add_scalar("val/ssim_vs_B",  ssim_b,  self.global_step)
                # vs 12-NEX union (gold-standard reporting metric)
                writer.add_scalar("val/l1_vs_ref",   l1_ref,   self.global_step)
                writer.add_scalar("val/psnr_vs_ref", psnr_ref, self.global_step)
                writer.add_scalar("val/ssim_vs_ref", ssim_ref, self.global_step)
                # Sharpness / contrast metrics (no impact on best selection)
                extras = getattr(self, "_last_val_extras", {})
                for k, v in extras.items():
                    writer.add_scalar(f"val/{k}", v, self.global_step)

                self._save("latest")   # resume ckpt (step-tagged save is done above, ungated)

                self.eval_count += 1
                patience = int(self.args.early_stop_patience)
                extras = getattr(self, "_last_val_extras", {})

                # 'none' mode (Self2Self / N2N2 convention): no best ckpt, no early
                # stop, train full max_steps and rely on latest.pth + swa.pth at the end.
                # Avoids selection bias when no clean reference is available.
                if self.args.best_criterion == "none":
                    upsnr_v = float(extras.get("upsnr", float("nan")))
                    cyc_v   = float(extras.get("subset_consistency", float("nan")))
                    sure_v  = float(extras.get("sure", float("nan")))
                    lapv    = float(extras.get("lapvar_ratio", float("nan")))
                    logging.info(
                        f"[VAL ] step={self.global_step} (no-selection mode) | "
                        f"upsnr={upsnr_v:.4f} cyc={cyc_v:.4f} | "
                        f"psnr_b={psnr_b:.2f} psnr_ref={psnr_ref:.2f} l1_B={l1_b:.4f} | "
                        f"sure={sure_v:.4f} lapvar={lapv:.3f}"
                    )
                    if self.global_step >= max_steps:
                        break
                    continue

                # Model selection criterion (higher crit_val = better). All
                # options are self-supervised (no biased reference): see the
                # --best_criterion help for the menu. psnr_ref / psnr_b are
                # still computed and reported as supplementary metrics, but are
                # not selectable criteria (they reward reference-noise mimicry).
                if self.args.best_criterion == "constrained_umse":
                    # Constrained-uMSE selection (v42k 2026-05-20, replaces upsnr_cyc):
                    #     θ* = argmin uMSE(θ)
                    #           s.t. cyc(θ)         ≤ ρ_cyc · noise_floor_L1
                    #                lapvar_ratio(θ) ≥ ρ_lapvar
                    # Principles:
                    # • uMSE = linear-scale unbiased risk estimator (Marcos-Morales
                    #   ICML 2023). Used directly (not its log uPSNR) so that
                    #   bootstrap CIs / 1-SE rule comparisons are valid.
                    # • cyc threshold τ_C is data-driven from the held-out noisy
                    #   reference repeatability E[|b−c|]. Model must be at most
                    #   ρ_cyc× as inconsistent as the noisy ref itself.
                    # • lapvar_ratio = lapvar(recon) / lapvar(ref). Floor prevents
                    #   the 'output brain-mask mean' trivial collapse (cyc≈0,
                    #   lapvar≪ref) observed in v42k-rev1.
                    # • Score is -uMSE inside the feasible set, -inf outside, so
                    #   that the existing argmax convention selects the minimum
                    #   uMSE subject to constraints.
                    umse_v = float(extras.get("umse", float("inf")))
                    cyc_v  = float(extras.get("subset_consistency", float("inf")))
                    nfl1   = float(extras.get("noise_floor_l1", float("nan")))
                    lapv_r = float(extras.get("lapvar_ratio", 0.0))
                    rho_cyc    = float(self.args.constrained_rho_cyc)
                    rho_lapvar = float(self.args.constrained_rho_lapvar)
                    if cyc_v != cyc_v:   # NaN guard (when T_a < 4 → no cyc)
                        cyc_v = 0.0
                    if nfl1 != nfl1 or nfl1 <= 0:
                        # Cannot compute constraint (insufficient T_b for 3-way
                        # split) → degenerate to uMSE-only selection.
                        feasible = (lapv_r >= rho_lapvar)
                        tau_C_str = "n/a"
                    else:
                        tau_C = rho_cyc * nfl1
                        feasible = (cyc_v <= tau_C) and (lapv_r >= rho_lapvar)
                        tau_C_str = f"{tau_C:.4f}"
                    crit_val = (-umse_v) if feasible else -1e9
                    crit_name = "constrained_umse"
                    # Annotate the log line below: store diagnostic on self for printing.
                    self._last_constraint_diag = (
                        f"umse={umse_v:.5f} cyc={cyc_v:.4f}/τC={tau_C_str} "
                        f"lapvR={lapv_r:.3f}/τS={rho_lapvar:.2f} feasible={feasible}"
                    )
                elif self.args.best_criterion == "upsnr_cyc":
                    # uPSNR (Marcos-Morales ICML 2023) − α · subset_consistency.
                    # uPSNR = pooled unbiased PSNR-to-clean estimator using a 3-way
                    # disjoint split of held-out set_b — captures **fidelity**.
                    # subset_consistency (cyc) = L1 between f(set_a[:k]) and
                    # f(set_a[k:2k]) — captures **input-stability**; lower is
                    # better. Together: maximise fidelity, minimise instability,
                    # entirely without relying on a (biased) reference image.
                    upsnr_v = float(extras.get("upsnr", 0.0))
                    cyc_v   = float(extras.get("subset_consistency", float("inf")))
                    if cyc_v != cyc_v:   # NaN guard (when T_a < 4)
                        cyc_v = 0.0
                    alpha   = float(self.args.upsnr_cyc_alpha)
                    crit_val  = upsnr_v - alpha * cyc_v
                    crit_name = "upsnr_cyc"
                elif self.args.best_criterion == "umse":
                    # Pure unbiased-MSE selection (Marcos-Morales ICML 2023).
                    # uMSE = linear-scale unbiased risk-to-clean estimator from the
                    # 3-way disjoint split of held-out set_b. Used directly (not its
                    # log uPSNR) so bootstrap-CI / 1-SE comparisons stay valid.
                    # No anti-collapse floor: uMSE already rises under BOTH noise
                    # (variance, e.g. from the w_grad probe) and over-smoothing
                    # (bias, e.g. from ssim_cyc), so it self-corrects without a
                    # hand-tuned lapvar threshold. sCoV + lapvR are reported only.
                    # Final-model selection folds sCoV in post-hoc via the uMSE
                    # 1-SE feasibility set (scripts/eval_select_ckpt.py).
                    umse_v = float(extras.get("umse", float("inf")))
                    crit_val  = -umse_v
                    crit_name = "umse"
                    # Texture-vs-noise guardrail reported live: sCoV (Wang 2003)
                    # and CNR (|μGM-μWM|/σWM) together disambiguate the grad-0.5
                    # texture — CNR↑ with stable sCoV = real GM/WM contrast;
                    # sCoV↑ with flat/falling CNR = injected noise. cnr_ref is the
                    # 12-NEX union's CNR (the bar the recon should beat).
                    self._last_constraint_diag = (
                        f"umse={umse_v:.5f} | "
                        f"scov_gm={float(extras.get('scov_gm_pred', float('nan'))):.4f} "
                        f"scov_wm={float(extras.get('scov_wm_pred', float('nan'))):.4f} "
                        f"cnr={float(extras.get('cnr_pred', float('nan'))):.3f}"
                        f"/ref={float(extras.get('cnr_ref', float('nan'))):.3f} "
                        f"lapvR={float(extras.get('lapvar_ratio', float('nan'))):.3f} (reported)"
                    )
                elif self.args.best_criterion == "scov_gm":
                    # GM sCoV primary selection (Wang 2003). Lower = more uniform
                    # within grey matter = less noise/mottling. Pure no-reference
                    # (GM is where perfusion signal is strongest; WM's low signal
                    # makes its sCoV estimate noisy, so GM-only is the cleaner
                    # criterion). Optional lapvar floor (--select_lapvar_floor):
                    # sCoV alone has a trivial minimiser (over-smooth blob), so a
                    # floor on lapvar_ratio excludes over-smooth ckpts. Same
                    # infeasible-sentinel mechanism as constrained_umse.
                    scov_gm = float(extras.get("scov_gm_pred", float("nan")))
                    lapv_r  = float(extras.get("lapvar_ratio", 0.0))
                    floor   = float(getattr(self.args, "select_lapvar_floor", 0.0))
                    feasible = (lapv_r >= floor) if floor > 0.0 else True
                    crit_val  = (-scov_gm) if feasible else -1e9
                    crit_name = "scov_gm"
                    self._last_constraint_diag = (
                        f"scov_gm={scov_gm:.4f} | "
                        f"lapvR={lapv_r:.3f}/floor={floor:.2f} feasible={feasible} | "
                        f"umse={float(extras.get('umse', float('nan'))):.5f}"
                    )
                else:
                    raise ValueError(
                        f"unhandled best_criterion {self.args.best_criterion!r}"
                    )
                writer.add_scalar(f"val/{crit_name}_score", crit_val, self.global_step)

                # Best gating: don't accept best until SURE has fully ramped (early
                # steps are dominated by noise-mimicry artefacts). Default is
                # config.sure_anneal_start; CLI --best_min_step overrides.
                if int(self.args.best_min_step) >= 0:
                    min_step = int(self.args.best_min_step)
                else:
                    min_step = int(getattr(self.criterion.weights, "sure_anneal_start", 0))
                allow_best = (self.global_step >= min_step)

                constraint_str = (
                    f" | {self._last_constraint_diag}"
                    if getattr(self, "_last_constraint_diag", None)
                    else ""
                )
                # An infeasible constrained-uMSE evaluation produces crit_val
                # = −1e9 (sentinel). best_val is reset to −∞ on criterion
                # change, so without this guard the very first infeasible
                # eval would be saved as 'best' (since −1e9 > −∞). Treat any
                # sentinel-level score as never-best.
                _INFEASIBLE_SENTINEL = -1e8
                # Score smoothing (exp-MA): if score_ema_alpha > 0, compare
                # against a moving average instead of raw per-validation score.
                # Skips infeasible sentinel values to avoid polluting the EMA.
                if crit_val > _INFEASIBLE_SENTINEL and self.score_ema_alpha > 0.0:
                    if self.score_ema is None:
                        self.score_ema = float(crit_val)
                    else:
                        self.score_ema = (
                            (1.0 - self.score_ema_alpha) * self.score_ema
                            + self.score_ema_alpha * float(crit_val)
                        )
                    score_for_best = self.score_ema
                else:
                    score_for_best = crit_val
                if (allow_best
                    and score_for_best > self.best_val
                    and crit_val > _INFEASIBLE_SENTINEL):
                    self.best_val = score_for_best
                    self.no_improve_count = 0
                    self._save("best")
                    ema_tag = f" (ema={self.score_ema:.4f})" if self.score_ema_alpha > 0 and self.score_ema is not None else ""
                    logging.info(
                        f"[BEST] step={self.global_step} "
                        f"{crit_name}={crit_val:.4f}{ema_tag} | "
                        f"sure={extras.get('sure', float('nan')):.4f} "
                        f"lapvar={extras.get('lapvar_ratio', float('nan')):.3f} "
                        f"psnr_ref={psnr_ref:.2f} hfen={extras.get('hfen', float('nan')):.3f} "
                        f"l1_B={l1_b:.4f}"
                        f"{constraint_str}"
                    )
                else:
                    self.no_improve_count += 1
                    tail = f" (best {self.best_val:.4f}"
                    if patience > 0:
                        tail += f", no-improve {self.no_improve_count}/{patience}"
                    if not allow_best:
                        tail += f", best gated until step {min_step}"
                    tail += ")"
                    logging.info(
                        f"[VAL ] step={self.global_step} "
                        f"{crit_name}={crit_val:.4f} | "
                        f"sure={extras.get('sure', float('nan')):.4f} "
                        f"lapvar={extras.get('lapvar_ratio', float('nan')):.3f} "
                        f"psnr_ref={psnr_ref:.2f}"
                        f"{tail}"
                        f"{constraint_str}"
                    )
                # Clear constraint diag so old line never leaks into next eval if
                # the criterion is switched mid-run.
                self._last_constraint_diag = None

                # In-loop per-metric-best (recoverability; --save_per_metric_best). Gated
                # like best.pth (allow_best) to avoid early noise-mimicry peaks. Independent
                # of the selection criterion above — saves best_<metric>.pth + best_metrics.json.
                if allow_best:
                    self._update_per_metric_best()

                # Early stop check
                if (patience > 0
                    and self.eval_count >= int(self.args.early_stop_min_evals)
                    and self.no_improve_count >= patience):
                    logging.info(
                        f"[EARLY STOP] step={self.global_step}: {self.args.best_criterion} "
                        f"has not improved for {self.no_improve_count} consecutive evals; "
                        f"best_val={self.best_val:.4f}"
                    )
                    break

            if self.global_step >= max_steps:
                break

        # Save SWA snapshot if active. The internal AveragedModel wraps the base
        # network; we save the underlying state_dict to make swa.pth a drop-in
        # replacement for best.pth (same key namespace, same loader code).
        if self.swa_model is not None and self.swa_n_avg > 0:
            try:
                inner = self.swa_model.module
            except AttributeError:
                inner = self.swa_model
            swa_sd = inner.state_dict()
            torch.save({
                "model": swa_sd,
                "ema": {k: v.detach().cpu() for k, v in swa_sd.items()},  # use SWA itself as 'ema' for eval scripts
                "step": self.global_step,
                "best_val": self.best_val,
                "swa_n_avg": int(self.swa_n_avg),
                "swa_start_step": int(self.args.swa_start_step),
            }, self._ckpt("swa"))
            logging.info(
                f"SWA saved: {self._ckpt('swa')} (avg over {self.swa_n_avg} updates "
                f"from step {self.args.swa_start_step})"
            )

        logging.info(f"Training done. Best {self.args.best_criterion}={self.best_val:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)

    run_log_dir = os.path.join(args.exp, "logs", args.name)
    if os.path.exists(run_log_dir) and not args.resume:
        shutil.rmtree(run_log_dir)
    # WSL DrvFs (/mnt/c) sometimes returns FileExistsError from mkdir even with
    # exist_ok=True, and the path is also not a real directory at that instant
    # (Windows-host metadata transient state after rmtree). Retry with delay.
    import time as _time
    _mk_ok = False
    for _attempt in range(8):
        try:
            os.makedirs(run_log_dir, exist_ok=True)
            _mk_ok = True
            break
        except FileExistsError:
            if os.path.isdir(run_log_dir):
                _mk_ok = True
                break
            _time.sleep(0.5)
    if not _mk_ok:
        raise RuntimeError(
            f"Could not create run_log_dir {run_log_dir!r} after 8 retries. "
            "WSL DrvFs reports the path exists but it is not a directory."
        )

    writer = make_loggers(args.exp, args.name, args.verbose)
    try:
        runner = Runner(args)
        runner.train(writer)
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
