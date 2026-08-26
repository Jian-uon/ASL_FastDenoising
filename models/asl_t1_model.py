from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from .blocks import (
        ASLDetailDecoder,
        ConvDecoder2D,
        ConvDecoderWithSkips2D,
        ConvEncoder2D,
        CrossModalFusion,
        NAFDecoder,
        SetTransformerAggregator,
        SpatialVaryingFrameWeighting,
        T1GuidedCoarseHead,
    )
except Exception:
    from blocks import (  # type: ignore
        ASLDetailDecoder,
        ConvDecoder2D,
        ConvDecoderWithSkips2D,
        ConvEncoder2D,
        CrossModalFusion,
        NAFDecoder,
        SetTransformerAggregator,
        SpatialVaryingFrameWeighting,
        T1GuidedCoarseHead,
    )

try:
    from .mossm_encoder import MoSSMEncoder2D
except Exception:
    from mossm_encoder import MoSSMEncoder2D  # type: ignore

try:
    from .content_guard_conv import ContentGuidedConvEncoder2D, ReproReliabilityGate, ReproDescriptor, CouplingGate
except Exception:
    from content_guard_conv import ContentGuidedConvEncoder2D, ReproReliabilityGate, ReproDescriptor, CouplingGate  # type: ignore

try:
    from .content_guard_vmamba import VMambaEncoder2D
except Exception:
    from content_guard_vmamba import VMambaEncoder2D  # type: ignore

try:
    from .ec_guidance import ECGuidance
except Exception:
    from ec_guidance import ECGuidance  # type: ignore

try:
    from .pvc_residual import PVCResidualHead
except Exception:
    from pvc_residual import PVCResidualHead  # type: ignore


def _direct_mean(
    set_x: "Tensor",
    lengths: "Optional[Tensor]" = None,
    mask: "Optional[Tensor]" = None,
) -> "Tensor":
    """Unweighted mean over valid frames. set_x: [B,T,1,H,W]. Returns [B,1,H,W]."""
    if mask is None and lengths is not None:
        T = set_x.shape[1]
        mask = torch.arange(T, device=set_x.device).unsqueeze(0) >= lengths.unsqueeze(1)
    if mask is None:
        return set_x.mean(dim=1)
    valid = (~mask).float()                                    # [B,T]
    w = valid.view(set_x.size(0), set_x.size(1), 1, 1, 1)
    num = (set_x * w).sum(dim=1)                               # [B,1,H,W]
    denom = valid.sum(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
    return num / denom


def build_fsl_cond_pv(gm, wm, csf, t1=None, brain_thr: float = 0.05, with_t1: bool = False):
    """FSL-PV conditioning tensor for ``lrda_cond_src in {'fsl','fsl_t1'}``.

    Default: brain-masked ``[B,3,H,W]`` soft GM/WM/CSF composition (NO BG channel)
    — the MAIN anatomical prior (2026-08-19), from the precomputed FSL FAST segs
    (ASL-space, perfusion-independent, no stage-1 dep). Outside-brain voxels zeroed;
    ``brain = t1 > brain_thr`` when a T1 is given (matches the ``t1 > 0.05`` mask
    used everywhere else), else the FSL segs are used as-is (FAST is brain-limited).

    ``with_t1=True`` (lrda_cond_src='fsl_t1', HYBRID) appends the brain-masked raw
    T1 as a 4th channel ⇒ ``[B,4,H,W] = [gm,wm,csf,t1]``: tissue composition PLUS
    raw anatomical detail as the conditioning source.

    V=ASL is preserved exactly in BOTH modes: this tensor only conditions the SS2D
    bilinear s(·), which scales the zero-init B/Δ modulation — it NEVER enters the
    Value/output. (Raw T1 here modulates dynamics only; it cannot inject content.)
    """
    pv = torch.cat([gm, wm, csf], dim=1).clamp(0.0, 1.0)      # [B,3,H,W]
    brain = None
    if t1 is not None:
        brain = (t1[:, :1] > float(brain_thr)).to(pv.dtype)
        pv = pv * brain
    if with_t1:
        if t1 is None:
            raise ValueError("build_fsl_cond_pv(with_t1=True) needs a t1 to append as the 4th channel.")
        t1c = t1[:, :1] * (brain if brain is not None else 1.0)   # brain-masked raw T1
        pv = torch.cat([pv, t1c], dim=1)                         # [B,4,H,W] = [gm,wm,csf,t1]
    return pv


class CondPyramid(nn.Module):
    """tier-1 LEARNED conditioning pyramid (opt-in via --lrda_cond_deep).

    Replaces the shallow ``_build_pv_pyramid`` (area-interp of raw PV) with a small
    strided-conv encoder: a full-res stem, then stride-2 downsamples, emitting a
    fixed ``out_ch`` feature map at EACH of ``n_stages`` resolutions (matching the ASL
    encoder stages, hw0 // 2**s). This gives the conditioning branch real depth AND a
    LEARNED pooling — fixing the fixed 8×8 area-average that blurred PV at coarse
    stages. Motivated by the tier-0 geo-liveness probe (geo ACTIVE ⇒ the ±1px branch
    already saturates its capacity), 2026-08-19.

    All convs are bias-free (so ``CondPyramid(0)=0`` EXACTLY ⇒ ``--pv_mode zero`` (pv0)
    stays exact identity) but the heads are NORMALLY initialized (2026-08-20): WARM-START
    identity + V=ASL is guaranteed SOLELY by the encoder's zero-init α (B'=B·exp(γ·α⊙tanh(m));
    α=0 at init), NOT by this module. Zero-init heads would starve α's early gradient
    (m≈0 ⇒ tanh(m)≈0 ⇒ α stuck) and the conditioning would learn slowly; open heads give
    m≠0 from step 1 ⇒ α gets an O(1) gradient. Depth cannot break V=ASL — it can only raise
    empirical leakage (gate on the D2a lesion test). Feeds E_B with ``lrda_pv_ch := out_ch``.
    """

    def __init__(self, in_ch: int, out_ch: int = 16, n_stages: int = 4, hidden: int = 24):
        super().__init__()
        self.out_ch = int(out_ch)
        self.n_stages = int(n_stages)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False), nn.SiLU(inplace=True))
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, bias=False),
                          nn.SiLU(inplace=True))
            for _ in range(max(0, n_stages - 1))])
        # bias-free heads, NORMALLY initialized (warm-start via α, not head zero-init)
        self.heads = nn.ModuleList([nn.Conv2d(hidden, out_ch, 1, bias=False)
                                    for _ in range(n_stages)])

    def forward(self, pv: Tensor, target_hw: Optional[list] = None) -> list:
        f = self.stem(pv)
        outs = [self.heads[0](f)]
        for i, down in enumerate(self.downs):
            f = down(f)
            outs.append(self.heads[i + 1](f))
        if target_hw is not None:                              # safety: exact stage sizes (odd dims)
            outs = [o if o.shape[-1] == hw and o.shape[-2] == hw
                    else F.interpolate(o, size=(hw, hw), mode="area")
                    for o, hw in zip(outs, target_hw)]
        return outs


class ASLT1Denoiser(nn.Module):
    """CIG-Net top-level model: Content-Isolated Anatomical Guidance for
    self-supervised 7T ASL denoising.

    See ``docs/cig_vss.md`` for the complete architecture spec, training
    protocol, and selection criterion. This docstring is a high-level pointer.
    The class is kept under its original name ``ASLT1Denoiser`` (rather than
    ``CIGNet``) so that all existing checkpoint state-dict keys remain valid;
    new code should import via ``from models.cig_net import CIGNet``.

    Four structural design contracts (Section 2 of the spec):

      C1 — Lesion-safe aggregation. ``SetTransformerAggregator`` produces
           per-frame scalar weights from permutation-equivariant frame
           embeddings; it cannot suppress consistent low signal.
      C2 — T1 content isolation (V = ASL). T1 modulates state dynamics in
           ``MoSSMEncoder2D`` (sigmoid-gated B/Δ residual, P2) and attention
           addressing in ``CMAM`` (low-rank Key projection, P3), but the
           Value path is exclusively ASL-derived at both stages.
      C3 — T1-free decoder. ``NAFDecoder`` takes only the ASL bottleneck and
           ASL skip features; T1 has no entry point into the reconstruction
           stage at the type level.
      C4 — Unbiased self-supervised selection (handled by the runner): uMSE
           under data-driven equivariance + sharpness constraints.

    Submodule map:
      - ``self.aggregator``         : ``SetTransformerAggregator``  (C1)
      - ``self.t1_encoder``         : ``ConvEncoder2D``             (frozen in stage 2)
      - ``self.t1_decoder``         : ``ConvDecoderWithSkips2D``    (frozen, 4-class PV seg)
      - ``self.asl_encoder``        : ``MoSSMEncoder2D``            (C2: P2 + P3)
      - ``self.naf_decoder``        : ``NAFDecoder``                (C3, T1-free; NAFNet
                                                                     + Restormer MDTA fusion)

    Training-time only:
      - Loss is computed by ``losses.asl_n2n_loss.ASLN2NLoss``:
            L = 0.35·L_n2n + 0.20·L_grad + 0.25·L_ssim
              + 0.05·L_contrast + 0.30·L_bdcyc
      - Best-checkpoint selection: ``--best_criterion constrained_umse`` in the
        runner (Section 3.10 of the spec).

    Deprecated kwargs (``use_prior_expert``, ``asl_only``, ``no_poe``,
    ``shared_dim``, ``t1_private_dim``, ``asl_private_dim``, ``latent_dim``,
    ``feat_dim``) are absorbed and ignored — kept in the signature so that old
    config files load without error.
    """

    def __init__(
        self,
        asl_hw: int = 128,
        t1_hw: int = 128,
        in_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        use_wavelet: bool = False,
        use_t1_cross_fusion: bool = False,
        t1_attn_max_tokens: int = 1024,
        use_svfw: bool = False,
        use_film: bool = False,
        use_heteroscedastic: bool = False,
        use_mossm_encoder: bool = False,
        mossm_blocks_per_scale: int = 2,
        mossm_n_directions: Union[int, List[int]] = 1,
        mossm_d_state: int = 16,
        # Hybrid CNN-SSM (2026-06-05): ASL-only local branch per MoSSM block.
        use_hybrid_mossm_block: bool = False,
        hybrid_local_expand: float = 2.0,
        hybrid_local_kernel: int = 3,
        use_tabs: bool = True,
        use_cmam: bool = True,
        cmam_max_tokens: int = 1024,
        cmam_n_heads: int = 4,
        use_reshead: bool = False,
        reshead_delta_max: float = 0.5,
        # v42g (2026-05-18): anti-T1-leakage encoder mods
        mossm_t1_gated: bool = False,       # P2: gated B/Δ residual in MoSSM
        mossm_t1_gate_init: float = -4.0,   # logit init for MoSSM T1 gate (σ(-4)=0.018)
        cmam_k_rank_div: int = 1,           # P3: low-rank K bottleneck in CMAM
        # v42j (2026-05-19): NAFNet-style decoder fusion (anti-skip-noise)
        use_naf_fusion: bool = False,       # ASLDetailDecoder uses NAFFusionBlock
        # v42k (2026-05-19): Restormer MDTA at NAFFusion (channel-axis attention)
        use_mdta_fusion: bool = False,      # Replace SCA with MDTA in all fusion NAFBlocks
        # 2026-05-25 CADA: Channel-Adaptive Dynamics Adapter. Per-pixel
        # evidence gate + per-channel zero-init α adapter. See
        # models/mossm_encoder.py:CADA.
        use_cada: bool = False,
        cada_active_stages: Optional[list] = None,
        cada_n_groups_B: int = 4,
        cada_lr: bool = False,
        cada_lr_rank: int = 8,
        cada_lr_bound: float = 4.0,
        # 2026-07 CIG-UNet: graft the content-guard (LRDA + AKMR) onto the plain
        # ConvEncoder2D backbone (which beat MoSSM). Conv-native content-safe
        # guidance. See docs/cig_unet_design.md. Mutually exclusive with
        # use_mossm_encoder. With all guards zero-init the model == vanilla
        # PlainUNet2D at step 0; the T1-off control is --zero_t1.
        use_conv_encoder: bool = False,
        lrda_rank: int = 8,
        lrda_bound: float = 4.0,
        lrda_variant: str = "c",           # a=per-channel, b=low-rank spatial, c=+depthwise
        conv_lrda_stages: Optional[list] = None,   # None = all stages
        conv_use_akmr: bool = True,        # T1-key cross-attention at low-res stages
        akmr_k_rank_div: int = 4,
        # 2026-07 VMamba backbone (baseline + CIG-VMamba). Same content-guard as
        # CIG-UNet, hosted on a VSS-block (4-dir cross-scan) encoder. Mutually
        # exclusive with use_conv_encoder / use_mossm_encoder.
        use_vmamba_encoder: bool = False,
        vmamba_no_guard: bool = False,     # True = pure VMamba ASL baseline (no LRDA/AKMR)
        vmamba_d_state: int = 16,
        vmamba_n_directions: int = 2,      # 2 = default (CIG-VSS main); 4 = full VMamba cross-scan
        vmamba_hv_scan: bool = False,      # reinterpret n=2 scan as {→,↓} (H+V) instead of {→,←}
        vmamba_blocks_per_scale: int = 2,
        # 2026-07 single-branch conv inductive bias + in-scan LRDA (CIG-VSS).
        vmamba_convffn: bool = False,          # ConvFFN in place of the plain MLP
        vmamba_overlap_merge: bool = False,    # 3x3 stride-2 conv downsample
        vmamba_lrda_inscan: bool = False,      # in-scan LRDA: T1-modulated B-gate + log-space Δ-mod inside SS2D
        lrda_cond_asl: bool = False,           # Design A: in-scan LRDA φ conditioned on [T1 ; ASL]
        lrda_dt_bound: float = 1.0,            # in-scan Δ log-mod bound: Δ=Δ·exp(b·tanh(U_Δ(s))) ∈ [e^-b,e^b]·Δ
        lrda_repro_gate: bool = False,         # Design C: reproducible-ASL reliability gate on T1 guidance
        repro_gate_align: bool = True,         #   use the T1/ASL alignment channel in the gate
        lrda_coupling_gate: bool = False,      # T1–ASL agreement gate on the LRDA modulation (no frame split)
        coupling_embed_dim: int = 16,          #   learned T1/ASL descriptor channels for the coupling
        akmr_repro_key: bool = False,          # RKMR: replace AKMR's T1 key with a reproducible-ASL key
        rkmr_t1_veto: bool = False,            #   optional low-rank T1 tissue-consistency veto (ablation)
        repro_desc_dim: int = 8,               #   reproducible descriptor φ channels
        # EC-LRDA (2026-07): soft-PV bilinear conditioning + ASL-evidence calibration.
        ec_lrda: bool = False,                 #   master switch (VSS in-scan LRDA only; needs t1_task='seg')
        ec_bilinear: bool = True,              #   bilinear s(PV)⊙a(ASL) modulation (vs raw-T1-feature φ)
        ec_calibrate: bool = True,             #   γ=r_rep·c_sem gate on the modulation (off ⇒ pure soft-PV bilinear = A1)
        ec_use_rep: bool = True,               #   include r_rep (ASL reproducibility) leg in γ (else γ=c_sem)
        ec_sg_gamma: bool = True,              #   stop-grad γ into the modulation (R learns only from L_rep/L_sem)
        ec_feat_dim: int = 32,                 #   EC guidance feature width
        ec_work_div: int = 4,                  #   EC works at H/ec_work_div (higher SNR, cheaper)
        ec_pv_geo: bool = False,               #   Level-1: PV-only geometry encoder (drop redundant p̄)
        lrda_cond_src: str = "pv",             #   {'pv','fsl','rawt1'}: bilinear s source. fsl=brain-masked FSL GM/WM/CSF (3-ch, MAIN, via set_cond_pv); pv=stage-1 soft-PV softmax (4-ch, ablation); rawt1=raw-T1↓ (1-ch, ablation)
        lrda_cond_deep: bool = False,          #   tier-1: LEARNED CondPyramid (strided-conv) replaces the shallow area-interp conditioning (justified by geo-ACTIVE probe). Structural: train==eval.
        cond_deep_ch: int = 16,                #   CondPyramid per-stage output channels feeding E_B (encoder lrda_pv_ch := this when deep)
        cond_deep_hidden: int = 24,            #   CondPyramid stem/downsample hidden width
        no_t1_branch: bool = False,            #   skip the (inert) T1 seg encoder/decoder entirely — fsl/rawt1 only (external/raw PV, T1-free decoder, brain=t1>thr, no gate). Drops the stage-1 dependency + saves compute.
        lrda_sig_xin: bool = False,            #   (2) a_B reads the d_inner token feature x_in (richer, all-ASL) instead of the thin d_state B_ matrix
        dt_rank: int = 1,                      #   (3) input-dependent Δ projection rank (0=auto=ceil(d_model/16); 1=legacy scalar)
        lrda_tfdm: bool = False,               #   TF-DM: standalone tissue-factored conditioning (K per-tissue W_k routed by raw PV); precludes deep
        scan_dropout_p: float = 0.0,           #   batch-3: scan-direction dropout (train regulariser + inference MC-uncertainty via set_mc_scan_dropout)
        ec_sem_bayes: bool = False,            #   opt-in: Bayesian PV-mixture + LOO evidence c_sem (else cosine)
        ec_sem_bayes_loo: bool = True,         #   Bayesian c_sem: use leave-one-out (False = plug-in ablation)
        ec_sem_evidence: bool = False,         #   opt-in: no-LOO gate + per-image global evidence E (diagnostic)
        # PVC residual decomposition (2026-08, design (d) architecture level; opt-in).
        pvc_residual: bool = False,            #   ŷ = E + w⊙r̂  (E=P·m̄ closed-form PVC base; recon←residual)
        pvc_no_wgate: bool = False,            #   ablation: drop the reproducibility gate (ŷ = E + r̂)
        slice_context: int = 0,                # 2.5D: ASL input = 2*ctx+1 adjacent z-slices as
                                               #   channels (0 = 2D). T1 stays 2D (center slice);
                                               #   output/target/repro are the CENTER slice.
        # 2026-06-15 attribution baselines (see docs/cada_lr_design.md §10):
        #   zero_t1        — blank the T1 INPUT (t1 := 0) before the T1 encoder.
        #                    The full T1 machinery (CADA/CMAM/seg) still runs but
        #                    on zeros, so it carries no subject information.
        #                    Architecture + param count identical to the full
        #                    model. This is the ASL-only-MoSSM information control
        #                    (isolates "does T1 information help?").
        #   naive_t1_concat — route T1 ONLY as a 2nd input channel into the ASL
        #                    encoder stem (T1 flows UNGUARDED into the value/skip/
        #                    decoder path), and zero the structured guidance skips
        #                    feeding CADA/CMAM/B-Δ. The un-guarded strawman the
        #                    content-guarded design is compared against (isolates
        #                    "does the V=ASL guarding matter?"). Requires MoSSM.
        zero_t1: bool = False,
        pv_mode: str = "real",       # {'real','zero','uniform'}: ablate PV INFO into the soft-PV bilinear + gate (M0='zero' => modulation exactly identity => plain VSS on ASL)
        naive_t1_concat: bool = False,
        t1_final_activation: Optional[nn.Module] = None,
        # 2026-07: stage-1 pretext of the frozen T1 branch. 'recon' = T1
        # autoencoder (1-ch head, DEFAULT); 'seg' = 4-class PV segmentation.
        # Must match the --t1_task used to pretrain --init_t1_from, so the
        # frozen t1_decoder head loads strict.
        t1_task: str = "recon",
        # 2026-08-25 (no-seg default arm): under t1_task='recon' with w_anat_roi=0 the
        # T1 decoder head is DEAD WEIGHT — 0.67M params (16% of the model) that receive
        # zero loss gradient and whose output is discarded. use_t1_decoder=False drops
        # it entirely (construction + forward). The T1 ENCODER is untouched: it still
        # feeds CMF0/CMF1 as the attention K-path, which is T1's only route into the
        # ASL branch in this arm. Must stay True for t1_task='seg' (the CMF tissue
        # bias, loss_seg and the pre-mask brain mask all read t1_seg_logits).
        use_t1_decoder: bool = True,
        # 2026-08-26 multi-scale window cross-fusion at the decoder's fine scales
        # (docs/multiscale_window_design.md). The coarse scales keep CMF0/CMF1;
        # these add the SAME guidance at the feature's own resolution, bounded by
        # ws x ws windows instead of by pooling. Q=ASL, K=T1, V=ASL unprojected =>
        # the fused output is a convex combination of ASL values (maximum
        # principle), so T1 steers the averaging but injects no content.
        # window_fusion_levels=0 does not build the modules at all => bit-exact
        # equality with the pre-2026-08-26 arch, and old checkpoints still load.
        window_fusion_levels: int = 0,
        window_size: int = 8,
        window_heads: int = 4,
        window_gate_init: float = -3.0,
        window_k_source: str = "t1",   # 't1' = anatomy-grouped; 'asl' = T1-free control
        **_kwargs,  # absorbs deprecated args
    ) -> None:
        super().__init__()
        self.zero_t1 = bool(zero_t1)
        self.pv_mode = str(pv_mode)
        self.naive_t1_concat = bool(naive_t1_concat)
        if self.naive_t1_concat and not (bool(use_mossm_encoder) or bool(use_vmamba_encoder)):
            raise ValueError("naive_t1_concat requires --use_mossm_encoder or "
                             "--use_vmamba_encoder (the concat targets the ASL encoder "
                             "stem). Pair with --vmamba_no_guard and no --ec_lrda so the "
                             "concat is the sole, unguarded T1 route.")
        if self.zero_t1 and self.naive_t1_concat:
            raise ValueError("zero_t1 and naive_t1_concat are mutually exclusive.")
        if self.zero_t1:
            print("[ASLT1Denoiser] ZERO-T1 baseline: T1 input blanked → ASL-only "
                  "(full T1 machinery present but information-free).")
        if self.naive_t1_concat:
            print("[ASLT1Denoiser] NAIVE-T1-CONCAT baseline: T1 enters ONLY as a "
                  "2nd encoder-input channel (unguarded V); guidance skips zeroed.")
        self.t1_hw = t1_hw
        self.use_t1_cross_fusion = bool(use_t1_cross_fusion)
        self.use_heteroscedastic = bool(use_heteroscedastic)
        self.use_mossm_encoder = bool(use_mossm_encoder)
        self.use_conv_encoder = bool(use_conv_encoder)
        self.use_vmamba_encoder = bool(use_vmamba_encoder)
        if sum([self.use_mossm_encoder, self.use_conv_encoder, self.use_vmamba_encoder]) > 1:
            raise ValueError("use_mossm_encoder / use_conv_encoder / use_vmamba_encoder "
                             "are mutually exclusive (pick one ASL backbone).")
        if self.use_conv_encoder:
            print("[ASLT1Denoiser] CIG-UNet: content-guarded ConvEncoder2D "
                  f"(LRDA variant={lrda_variant} rank={lrda_rank} bound={lrda_bound}, "
                  f"AKMR={'on' if conv_use_akmr else 'off'}); T1-free plain decoder.")
        if self.use_vmamba_encoder:
            print("[ASLT1Denoiser] VMamba backbone "
                  f"({'PURE baseline (no guard)' if vmamba_no_guard else 'CIG-VMamba: guarded'}, "
                  f"n_dir={vmamba_n_directions}, d_state={vmamba_d_state}); T1-free plain decoder.")
        # The Var_T(set_a) noise-variance side-channel (CADA z_σ / CMAM-BLUE /
        # RGSF) was removed 2026-06-21 — it was inert-to-harmful in v2 and is no
        # longer computed or wired anywhere. T1 still modulates B/Δ via CADA-LR.
        # ResHead-with-cap: output = agg + δ_max * tanh(decoder_raw); bounds
        # the additive correction to ±δ_max in ΔM ∈ [0,1] units.
        self.use_reshead = bool(use_reshead)
        self.reshead_delta_max = float(reshead_delta_max)
        # v42 sub-components: default ON when MoSSM is on; can be disabled
        # individually for ablation. No effect when use_mossm_encoder=False.
        self.t1_task = str(t1_task)
        if self.t1_task not in ("seg", "recon"):
            raise ValueError(f"t1_task must be 'seg' or 'recon', got {t1_task!r}")
        self.window_fusion_levels = int(window_fusion_levels)
        self.window_k_source = str(window_k_source)
        if self.window_fusion_levels > 0 and self.window_k_source == "t1" and bool(no_t1_branch):
            raise ValueError("window_fusion_levels>0 with window_k_source='t1' needs the T1 "
                             "encoder (it supplies the attention keys); use "
                             "window_k_source='asl' with no_t1_branch.")
        self.use_t1_decoder = bool(use_t1_decoder)
        if self.t1_task == "seg" and not self.use_t1_decoder:
            raise ValueError("t1_task='seg' needs the T1 decoder (it produces the 4-class "
                             "PV logits consumed by the CMF tissue bias / loss_seg / "
                             "input pre-mask); use_t1_decoder=False is recon-only.")
        self.use_tabs = bool(use_tabs) and self.use_mossm_encoder
        if self.t1_task == "recon":
            # No PV-seg logits to drive TABS scan; recon head is 1-ch.
            self.use_tabs = False
        # CMAM: cross-modal attentive memory inside MoSSM blocks (v42 main contrib).
        # When CMAM is on, the separate bottleneck CrossModalFusion is removed —
        # all cross-modal attention happens INSIDE the encoder, unified arch.
        self.use_cmam = bool(use_cmam) and self.use_mossm_encoder

        # 2.5D: the ASL path (aggregator + ASL encoder) sees 2*ctx+1 adjacent
        # z-slices stacked as channels; the T1 path stays single-slice (center).
        # Output/target/repro are the CENTER slice (channels [center_ch:center_ch+in_ch]).
        self.in_ch = int(in_ch)
        self.slice_context = int(slice_context)
        self.n_slices = 2 * self.slice_context + 1
        self.center_ch = self.in_ch * self.slice_context     # first channel of the center slice
        asl_in_ch = self.in_ch * self.n_slices

        # Frame aggregation. v37+ default = SpatialVaryingFrameWeighting (per-pixel
        # BLUE). Earlier versions = SetTransformerAggregator (per-frame scalar).
        if bool(use_svfw):
            self.aggregator = SpatialVaryingFrameWeighting(in_ch=asl_in_ch, hidden=32)
        else:
            hidden_ch = max(64, base_ch * 2)
            hidden_ch = (hidden_ch // 4) * 4  # ensure divisibility by 4 heads
            self.aggregator = SetTransformerAggregator(in_ch=asl_in_ch, hidden_ch=hidden_ch, n_heads=4)

        # Encoders. T1 encoder is ConvEncoder2D (multi-scale features MoSSM uses as
        # modulation) — but SKIPPED entirely when no_t1_branch (inert under fsl/rawt1).
        # ASL encoder: ConvEncoder2D by default; MoSSMEncoder2D when v42 is on.
        self.no_t1_branch = bool(no_t1_branch)
        self._n_stages = int(depth)            # pyramid stage count (== len(t1_skips)) when the T1 branch is skipped
        self.t1_encoder = (None if self.no_t1_branch
                           else ConvEncoder2D(in_ch=in_ch, base_ch=base_ch, depth=depth, use_wavelet=use_wavelet))
        if self.use_mossm_encoder:
            # Declare the per-stage spatial sizes so MoSSMEncoder can gate CMAM
            # by H*W <= cmam_max_tokens (avoids O(N^2) blow-up at large scales).
            hw_per_scale = [asl_hw // (2 ** s) for s in range(depth)]
            # Default CADA stage selectivity: when enabled but no list given,
            # activate at ALL stages. Rationale: with zero-init α adapters,
            # each stage's T1 contribution starts at exactly zero and only
            # grows where the N2N loss rewards it. Manual stage selection is
            # therefore unnecessary — the model self-prunes via α. Compute
            # cost at high-res stages (0,1) is accepted as a principled cost
            # of full-freedom design.
            if bool(use_cada) and cada_active_stages is None:
                _cada_stages = [True] * depth
            else:
                _cada_stages = (
                    list(cada_active_stages)
                    if cada_active_stages is not None
                    else [False] * depth
                )
                if not bool(use_cada):
                    _cada_stages = [False] * depth
            self.asl_encoder = MoSSMEncoder2D(
                in_ch=asl_in_ch + (1 if self.naive_t1_concat else 0),
                base_ch=base_ch, depth=depth,
                t1_channels_per_scale=[base_ch * (2 ** i) for i in range(depth)],
                d_state=int(mossm_d_state),
                n_blocks_per_scale=int(mossm_blocks_per_scale),
                n_directions=mossm_n_directions,
                use_cmam=self.use_cmam,
                cmam_max_tokens=int(cmam_max_tokens),
                cmam_n_heads=int(cmam_n_heads),
                hw_per_scale=hw_per_scale,
                t1_gated_bd=bool(mossm_t1_gated),
                t1_gate_init=float(mossm_t1_gate_init),
                cmam_k_rank_div=int(cmam_k_rank_div),
                cada_active_stages=_cada_stages,
                cada_n_groups_B=int(cada_n_groups_B),
                cada_lr=bool(cada_lr),
                cada_lr_rank=int(cada_lr_rank),
                cada_lr_bound=float(cada_lr_bound),
                use_hybrid_mossm_block=bool(use_hybrid_mossm_block),
                hybrid_local_expand=float(hybrid_local_expand),
                hybrid_local_kernel=int(hybrid_local_kernel),
            )
        elif self.use_conv_encoder:
            # CIG-UNet: PlainUNet2D's ConvEncoder2D + per-stage content-guard.
            self.asl_encoder = ContentGuidedConvEncoder2D(
                in_ch=asl_in_ch, base_ch=base_ch, depth=depth, input_hw=asl_hw,
                t1_channels_per_scale=[base_ch * (2 ** i) for i in range(depth)],
                lrda_rank=int(lrda_rank),
                lrda_bound=float(lrda_bound),
                lrda_variant=str(lrda_variant),
                lrda_stages=(list(conv_lrda_stages) if conv_lrda_stages is not None else None),
                akmr_enabled=bool(conv_use_akmr),
                akmr_max_tokens=int(cmam_max_tokens),
                akmr_heads=int(cmam_n_heads),
                akmr_k_rank_div=int(akmr_k_rank_div),
            )
        elif self.use_vmamba_encoder:
            # VMamba backbone (VSS-block, 4-dir cross-scan). Pure baseline when
            # vmamba_no_guard; else CIG-VMamba with the same content-guard.
            # EC-LRDA: the in-scan LRDA is conditioned on soft PV composition (t1_skip
            # carries [p;p̄] = 2*pv_ch channels) via a bilinear s(PV)⊙a(ASL) adapter.
            _cond_rawt1 = (str(lrda_cond_src) == "rawt1")
            _cond_fsl = (str(lrda_cond_src) == "fsl")
            _ec_bilinear = bool(ec_lrda) and bool(ec_bilinear)
            # raw conditioning channels feeding the conditioning branch:
            # rawt1: 1 raw-T1; fsl: 3 FSL GM/WM/CSF (no BG); fsl_t1(HYBRID): 4 = GM/WM/CSF/T1;
            # pv (stage-1): 4 GM/WM/CSF/BG. fsl_t1 and pv both fall into the 'else 4' branch.
            _raw_pv_ch = 1 if _cond_rawt1 else (3 if _cond_fsl else 4)
            # TF-DM: standalone tissue-factored conditioning (K per-tissue operators routed by
            # raw PV). Uses the RAW K-ch PV (tissue identity), so it PRECLUDES deep/CondPyramid
            # (which produces abstract features). rawt1 excluded (no tissue fractions).
            _tfdm = _ec_bilinear and bool(lrda_tfdm) and not _cond_rawt1
            # tier-1 deep conditioning: a learned CondPyramid replaces the shallow area-
            # interp; the encoder's E_B then sees cond_deep_ch learned channels (single-
            # source, no p̄). rawt1 stays shallow (raw-T1 geo). Structural: train==eval.
            _cond_deep = _ec_bilinear and bool(lrda_cond_deep) and not _cond_rawt1 and not _tfdm
            # channels that FEED the conditioning: cond_deep_ch when deep, else raw (TF-DM: raw K).
            _pv_ch = int(cond_deep_ch) if _cond_deep else _raw_pv_ch
            # pv_geo/deep/rawt1/tfdm all use the single-source structure (no p̄ channel).
            _ec_pv_geo = _ec_bilinear and (bool(ec_pv_geo) or _cond_rawt1 or _cond_deep or _tfdm)
            _vmamba_t1ch = ([(_pv_ch if _ec_pv_geo else 2 * _pv_ch)] * depth if _ec_bilinear
                            else [base_ch * (2 ** i) for i in range(depth)])
            # CondPyramid (tier-1): input = raw PV channels, output = cond_deep_ch per stage.
            self.lrda_cond_deep = bool(_cond_deep)
            self.cond_pyramid = (CondPyramid(in_ch=_raw_pv_ch, out_ch=int(cond_deep_ch),
                                             n_stages=int(depth), hidden=int(cond_deep_hidden))
                                 if _cond_deep else None)
            self.asl_encoder = VMambaEncoder2D(
                in_ch=asl_in_ch + (1 if self.naive_t1_concat else 0),
                base_ch=base_ch, depth=depth, input_hw=asl_hw,
                d_state=int(vmamba_d_state),
                n_directions=int(vmamba_n_directions),
                hv_scan=bool(vmamba_hv_scan),
                blocks_per_scale=int(vmamba_blocks_per_scale),
                convffn=bool(vmamba_convffn),
                overlap_merge=bool(vmamba_overlap_merge),
                lrda_inscan=bool(vmamba_lrda_inscan),
                lrda_cond_asl=bool(lrda_cond_asl),
                lrda_dt_bound=float(lrda_dt_bound),
                lrda_bilinear=_ec_bilinear,
                lrda_pv_geo=_ec_pv_geo,
                lrda_pv_ch=_pv_ch,
                akmr_repro_key=bool(akmr_repro_key),
                rkmr_t1_veto=bool(rkmr_t1_veto),
                repro_desc_dim=int(repro_desc_dim),
                use_guard=not bool(vmamba_no_guard),
                t1_channels_per_scale=_vmamba_t1ch,
                lrda_rank=int(lrda_rank),
                lrda_bound=float(lrda_bound),
                lrda_variant=str(lrda_variant),
                lrda_stages=(list(conv_lrda_stages) if conv_lrda_stages is not None else None),
                akmr_enabled=bool(conv_use_akmr),
                akmr_max_tokens=int(cmam_max_tokens),
                akmr_heads=int(cmam_n_heads),
                akmr_k_rank_div=int(akmr_k_rank_div),
                sig_xin=bool(lrda_sig_xin),
                dt_rank=int(dt_rank),
                tfdm=bool(_tfdm),
                scan_dropout_p=float(scan_dropout_p),
            )
        else:
            self.asl_encoder = ConvEncoder2D(in_ch=asl_in_ch, base_ch=base_ch, depth=depth, use_wavelet=use_wavelet)
        bottleneck_ch = self.asl_encoder.out_ch  # base_ch * 2^(depth-1)

        # Design C: reproducible-ASL reliability gate (VMamba/in-scan LRDA only).
        # Suppresses T1 guidance where the ASL evidence is not reproducible.
        self.lrda_repro_gate = bool(lrda_repro_gate) and self.use_vmamba_encoder
        self.repro_gate = (
            ReproReliabilityGate(use_align=bool(repro_gate_align))
            if self.lrda_repro_gate else None
        )
        # RKMR: reproducible-structure descriptor φ that keys the non-local reader.
        self.akmr_repro_key = bool(akmr_repro_key) and self.use_vmamba_encoder
        self.repro_desc = (
            ReproDescriptor(d=int(repro_desc_dim)) if self.akmr_repro_key else None
        )
        # Coupling gate: content-safe T1–ASL agreement that scales the LRDA
        # modulation. Uses the DETACHED all-frame aggregate (no even/odd split) +
        # frozen T1 ⇒ no learnable ASL feature feeds it. Backbone-agnostic
        # (conv CIG-UNet + VSS in-scan LRDA); it only scales the zero-init,
        # multiplicative, V=ASL modulation so the mismatched-T1 invariant holds.
        self.lrda_coupling_gate = bool(lrda_coupling_gate)
        self.coupling_gate = (
            CouplingGate(asl_ch=int(self.in_ch), t1_ch=1, embed=int(coupling_embed_dim))
            if self.lrda_coupling_gate else None
        )

        # EC-LRDA guidance: soft-PV bilinear conditioning + ASL-evidence calibration
        # γ = r_rep · c_sem (VSS in-scan LRDA only; PV needs t1_task='seg').
        self.ec_lrda = bool(ec_lrda) and self.use_vmamba_encoder
        self.ec_bilinear = bool(ec_bilinear)
        self.ec_calibrate = bool(ec_calibrate)
        self.ec_use_rep = bool(ec_use_rep)
        self.ec_sg_gamma = bool(ec_sg_gamma)
        self.ec_pv_geo = bool(ec_pv_geo)
        self.lrda_cond_src = str(lrda_cond_src)          # {'pv','fsl','rawt1'}
        # 'fsl' = brain-masked FSL GM/WM/CSF (3-ch, no BG), supplied externally per
        # batch via set_cond_pv() — the MAIN conditioning source (2026-08-19). 'pv' =
        # frozen stage-1 soft-PV softmax (4-ch, ABLATION). 'rawt1' = raw-T1↓ (1-ch, ABLATION).
        self.ec_pv_ch = (1 if self.lrda_cond_src == "rawt1"
                         else 3 if self.lrda_cond_src == "fsl" else 4)
        # Per-batch FSL-PV conditioning tensor [B,3,H,W] (brain-masked GM/WM/CSF),
        # set via set_cond_pv() before each forward when cond_src='fsl'. NOT a
        # Parameter/buffer (EMA-agnostic); reset to None on model reconstruction so a
        # forgotten set_cond_pv() fails loudly rather than reusing a stale batch's PV.
        self._cond_pv_ext = None
        # Only the 'pv' conditioning source reads softmax(t1_seg) ⇒ needs the seg branch.
        # fsl (external PV) and rawt1 (raw T1) do NOT, so they may set no_t1_branch to drop
        # the (inert) stage-1 dependency entirely.
        if self.ec_lrda and self.lrda_cond_src == "pv" and self.t1_task != "seg":
            raise ValueError("ec_lrda with lrda_cond_src='pv' requires t1_task='seg' (soft-PV composition prior).")
        if self.no_t1_branch and self.lrda_cond_src == "pv":
            raise ValueError("no_t1_branch needs an external/raw PV source (lrda_cond_src in {fsl, rawt1}); "
                             "'pv' reads softmax(t1_seg) and so requires the T1 seg branch.")
        self.ec_sem_bayes = bool(ec_sem_bayes)
        self.ec_sem_bayes_loo = bool(ec_sem_bayes_loo)
        self.ec_sem_evidence = bool(ec_sem_evidence)
        if self.ec_lrda and self.ec_calibrate:
            self.ec_guidance = ECGuidance(
                asl_in_ch=asl_in_ch, pv_ch=self.ec_pv_ch,
                feat_dim=int(ec_feat_dim), work_div=int(ec_work_div),
                sem_bayes=self.ec_sem_bayes, sem_bayes_loo=self.ec_sem_bayes_loo,
                sem_evidence=self.ec_sem_evidence)
        else:
            self.ec_guidance = None

        # PVC residual decomposition head (opt-in). Assembles ŷ = E + w⊙r̂ AFTER the
        # T1-free decoder, so the decoder stays byte-identical/T1-free; only the output
        # assembly changes. Needs the soft-PV composition (t1_task='seg'). Incompatible
        # with ResHead/heteroscedastic (both redefine the decoder output semantics).
        self.pvc_residual = bool(pvc_residual)
        self.pvc_no_wgate = bool(pvc_no_wgate)
        if self.pvc_residual:
            if self.t1_task != "seg":
                raise ValueError("pvc_residual requires t1_task='seg' (soft-PV composition).")
            if self.use_reshead or self.use_heteroscedastic:
                raise ValueError("pvc_residual is incompatible with --use_reshead / heteroscedastic.")
            self.pvc_head = PVCResidualHead(pv_ch=self.ec_pv_ch)
        else:
            self.pvc_head = None

        # --- Refactor 2026-05-15: split ASL decoder path into
        #     T1GuidedCoarseHead (L0+L1, T1-aware) + ASLDetailDecoder (L2+L3, pure ASL).
        #
        # T1 path is structurally confined to coarse semantic scales (16x16 + 32x32).
        # ASLDetailDecoder has no T1 input -> fine spatial detail cannot carry T1
        # grayscale by construction (V=ASL extended to module boundary).
        #
        # CMF0 (bottleneck) is enabled in all configs EXCEPT v42-default (CMAM
        # inside MoSSM encoder replaces it). When enabled:
        #   - v37 path (no MoSSM): cross-attn with K=T1, tissue-class bias.
        #   - v42 ablation (--no_cmam): ASL self-attn + BLUE-Attn, no tissue bias.
        # CMF1 (L1) is enabled when use_t1_cross_fusion=True (all v37/v42 paths).
        cmf0_enabled = not (self.use_mossm_encoder and self.use_cmam)
        cmf0_t1_as_key = not self.use_mossm_encoder
        cmf0_use_blue = False  # noise-var BLUE removed 2026-06-21
        cmf1_enabled = bool(self.use_t1_cross_fusion)
        # v42-clean (2026-05-15): when MoSSM encoder + CMAM are both on, ALL
        # cross-modal T1 fusion lives inside the encoder. Decoder is T1-free
        # by design — having a duplicate CMF1 at L1 (same scale as CMAM stage 2)
        # creates contribution ambiguity. Force CMF1 off to enforce this.
        if self.use_mossm_encoder and self.use_cmam and cmf1_enabled:
            cmf1_enabled = False
        # Token-budget check: CMF1 attention at 32x32 has N=1024 tokens. Disable
        # if the user explicitly lowered t1_attn_max_tokens below that.
        if cmf1_enabled and (asl_hw // (2 ** (depth - 2))) ** 2 > int(t1_attn_max_tokens):
            cmf1_enabled = False

        channels = [base_ch * (2 ** i) for i in range(depth)]
        l1_ch = channels[depth - 2]    # L1 feature channels (32x32 scale)

        # v42j (2026-05-19): when use_naf_fusion=True, replace the entire
        # decoder stack (T1GuidedCoarseHead + ASLDetailDecoder) with a single
        # unified NAFDecoder that uses NAFNet SimpleGate fusion at every level.
        # This removes the T1GuidedCoarseHead module (and its CMF₀/CMF₁ paths)
        # — encoder→decoder transition is fully T1-free and uses the same
        # NAFFusionBlock at all 3 up-sample stages.
        self.use_naf_fusion = bool(use_naf_fusion)
        self.use_mdta_fusion = bool(use_mdta_fusion)
        asl_out_ch = 2 if self.use_heteroscedastic else 1
        # CIG-UNet / VMamba plain decoder (set below). None for all other configs.
        self.conv_asl_decoder = None
        if (self.use_conv_encoder or self.use_vmamba_encoder) and not self.use_naf_fusion:
            # CIG-UNet / CIG-VMamba default: plain T1-free decoder — byte-identical
            # to the winning PlainUNet2D decoder (ConvDecoderWithSkips2D with T1
            # cross-fusion + FiLM OFF). Keeps the ONLY change vs vanilla in the
            # encoder, so the T1-guard contribution is cleanly attributable.
            self.t1_head = None
            self.asl_decoder = None
            self.naf_decoder = None
            self.conv_asl_decoder = ConvDecoderWithSkips2D(
                in_ch=bottleneck_ch, out_ch=asl_out_ch, out_hw=asl_hw,
                base_ch=base_ch, depth=depth, skip_dropout=0.3,
                use_wavelet=use_wavelet,
                use_t1_cross_fusion=False, use_film=False,
            )
        elif self.use_naf_fusion:
            self.t1_head = None
            self.asl_decoder = None
            self.naf_decoder = NAFDecoder(
                in_ch=bottleneck_ch, out_ch=asl_out_ch, out_hw=asl_hw,
                base_ch=base_ch, depth=depth,
                skip_dropout=0.3,
                use_wavelet=use_wavelet,
                use_rgsf=False,  # noise-var RGSF removed 2026-06-21
                use_mdta=self.use_mdta_fusion,
            )
        else:
            self.t1_head = T1GuidedCoarseHead(
                in_ch=bottleneck_ch,
                bottleneck_ch=bottleneck_ch,
                l1_ch=l1_ch,
                use_cmf0=cmf0_enabled,
                use_cmf1=cmf1_enabled,
                cmf0_t1_as_key=cmf0_t1_as_key,
                cmf0_use_blue=cmf0_use_blue,
                cmf_n_heads=4,
                cmf_dropout=0.2,
                use_wavelet=use_wavelet,
                skip_dropout=0.3,
            )
            self.asl_decoder = ASLDetailDecoder(
                in_ch=l1_ch, out_ch=asl_out_ch, out_hw=asl_hw,
                base_ch=base_ch, depth=depth,
                skip_dropout=0.3,
                use_wavelet=use_wavelet,
                use_rgsf=False,  # noise-var RGSF removed 2026-06-21
                window_fusion_levels=self.window_fusion_levels,
                window_size=int(window_size),
                window_heads=int(window_heads),
                window_gate_init=float(window_gate_init),
                window_k_source=self.window_k_source,
            )
            self.naf_decoder = None
        # Legacy attribute alias: scripts and ckpts may probe self.cross_fusion.
        # After the refactor the bottleneck CMF lives inside self.t1_head.cmf0.
        # Expose None so `cross_fusion is None` checks still behave correctly
        # (the v42-default branch in eval scripts depends on this).
        self.cross_fusion = None
        # T1 decoder: 4-channel head = GM/WM/CSF/BG logits (PV segmentation).
        # The cross-attention transfers tissue semantics to the ASL branch.
        # Dropped entirely when use_t1_decoder=False (the no-seg default arm, where it
        # is dead weight) or no_t1_branch — see the use_t1_decoder kwarg comment.
        t1_dec_out_ch = 1 if self.t1_task == "recon" else 4
        self.t1_decoder = (None if (self.no_t1_branch or not self.use_t1_decoder)
                           else ConvDecoderWithSkips2D(
            in_ch=bottleneck_ch, out_ch=t1_dec_out_ch, out_hw=t1_hw,
            base_ch=base_ch, depth=depth,
            final_activation=None,
            skip_dropout=0.3,
            use_wavelet=use_wavelet,
        ))
        self.t1_recon_activation = t1_final_activation  # kept for API compat, unused

    def aggregate_asl(
        self,
        frames: Tensor,
        lengths: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.aggregator(frames, lengths=lengths, mask=mask)

    def _even_odd_halfmeans(
        self, frames: Tensor, lengths: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Even/odd frame split → two independent noisy means (m1, m2) [B,1,H,W]
        + valid_pair [B,1,1,1] (1 where both halves have ≥1 frame). Deterministic
        (no RNG) so the reliability gate is reproducible train↔infer."""
        B, T = frames.shape[0], frames.shape[1]
        dev = frames.device
        if lengths is None:
            lengths = torch.full((B,), T, device=dev, dtype=torch.long)
        idx = torch.arange(T, device=dev)
        valid = (idx.view(1, T) < lengths.view(B, 1)).float()            # [B,T]
        even = valid * (idx % 2 == 0).float().view(1, T)
        odd = valid * (idx % 2 == 1).float().view(1, T)

        def wmean(w: Tensor) -> Tuple[Tensor, Tensor]:
            n = w.sum(1)                                                  # [B]
            m = (frames * w.view(B, T, 1, 1, 1)).sum(1) / n.clamp_min(1e-6).view(B, 1, 1, 1)
            return m, n

        m1, n1 = wmean(even)
        m2, n2 = wmean(odd)
        valid_pair = ((n1 >= 0.5) & (n2 >= 0.5)).float().view(B, 1, 1, 1)
        return m1, m2, valid_pair

    def _repro_signals(
        self, frames: Tensor, lengths: Optional[Tensor],
        mask: Optional[Tensor], t1: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Compute the reproducibility signals from ONE even/odd frame split:
          r   — Design C reliability gate ∈(0,1) (None unless --lrda_repro_gate)
          phi — RKMR reproducible descriptor [B,d,H,W] (None unless --akmr_repro_key)
        Both share the same half-means; degenerate (<2 frames) → r=1 (ungated)."""
        if self.repro_gate is None and self.repro_desc is None:
            return None, None
        if lengths is None and mask is not None:
            lengths = mask.float().sum(dim=1).to(torch.long)
        m1, m2, vp = self._even_odd_halfmeans(frames, lengths)
        # 2.5D: half-means are [B, in_ch*n_slices, H, W]; the gate/descriptor are
        # single-slice → key on the CENTER slice (no-op when slice_context=0).
        c0, c1 = self.center_ch, self.center_ch + self.in_ch
        m1, m2 = m1[:, c0:c1], m2[:, c0:c1]
        r = None
        if self.repro_gate is not None:
            r = self.repro_gate(m1, m2, t1)                             # [B,1,H,W] ∈ (0,1)
            r = vp * r + (1.0 - vp)                                     # degenerate → 1 (ungated)
        phi = self.repro_desc(m1, m2) if self.repro_desc is not None else None
        return r, phi

    def _compute_coupling_gate(self, agg_center: Tensor, t1: Tensor) -> Optional[Tensor]:
        """T1–ASL agreement gate c∈(0,1) from the DETACHED all-frame aggregate +
        frozen T1 (no even/odd split). Returns None when disabled."""
        if self.coupling_gate is None:
            return None
        return self.coupling_gate(t1, agg_center)                      # [B,1,H,W]; detach inside

    @staticmethod
    def _merge_gates(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
        """Combine two ∈(0,1) modulation gates (Design-C reliability × coupling).
        Either may be None; both suppress ⇒ product."""
        if a is None:
            return b
        if b is None:
            return a
        if a.shape[-2:] != b.shape[-2:]:
            b = F.adaptive_avg_pool2d(b, a.shape[-2:])
        return a * b

    def _encode_asl_branch(
        self,
        agg: Tensor,
        t1_skips: Optional[list] = None,
        tabs_seg: Optional[Tensor] = None,
        t1_seg_probs: Optional[Tensor] = None,
        repro_gate: Optional[Tensor] = None,
        repro_desc: Optional[Tensor] = None,
        coupling_gate: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list]:
        """Encode ASL aggregate. Returns (bottleneck feat_map, asl_skips).

        Bottleneck cross-fusion was moved into self.t1_head (refactor 2026-05-15)
        and is no longer applied here.

        Args:
            agg:         ASL aggregate [B, 1, H, W]
            t1_skips:    multi-scale T1 features (required by MoSSM encoder; CMAM
                         inside MoSSM consumes them at every stage)
            tabs_seg:    tissue PV map for MoSSM's TABS scan reordering (v42)
            t1_seg_probs: unused (legacy parameter, kept for signature compat)
        """
        if self.use_vmamba_encoder:
            assert t1_skips is not None, "guided ASL encoder requires t1_skips"
            feat_map, _, asl_skips = self.asl_encoder(
                agg, t1_skips,
                tissue_seg=tabs_seg,
                t1_seg_probs=t1_seg_probs,
                repro_gate=self._merge_gates(repro_gate, coupling_gate),  # Design C × coupling
                repro_desc=repro_desc,           # RKMR φ (None unless enabled)
            )
        elif self.use_conv_encoder:
            assert t1_skips is not None, "guided ASL encoder requires t1_skips"
            feat_map, _, asl_skips = self.asl_encoder(
                agg, t1_skips,
                tissue_seg=tabs_seg,
                t1_seg_probs=t1_seg_probs,
                coupling_gate=coupling_gate,     # T1–ASL agreement gate on LRDA/AKMR
            )
        elif self.use_mossm_encoder:
            assert t1_skips is not None, "guided ASL encoder requires t1_skips"
            feat_map, _, asl_skips = self.asl_encoder(
                agg, t1_skips,
                tissue_seg=tabs_seg,
                t1_seg_probs=t1_seg_probs,
            )
        else:
            feat_map, _, asl_skips = self.asl_encoder(agg)
        return feat_map, asl_skips

    def _decode_asl(
        self,
        feat_map: Tensor,
        asl_skips: list,
        t1_feat_map: Tensor,
        t1_skips: list,
        head_tissue_seg: Optional[Tensor],
    ) -> Tensor:
        """Run T1GuidedCoarseHead (L0+L1) then ASLDetailDecoder (L2+L3).

        head_tissue_seg is forwarded to the coarse head; the detail decoder
        never sees T1 by construction.
        """
        # CIG-UNet: plain T1-free decoder (identical interface to PlainUNet2D).
        if self.conv_asl_decoder is not None:
            return self.conv_asl_decoder(feat_map, asl_skips)
        # depth = len(asl_skips). For depth=4 skips are ordered [stem(128x128),
        # down1(64x64), down2(32x32), down3(16x16)]; L1 skip is index depth-2.
        depth = len(asl_skips)
        # v42j (2026-05-19): unified NAFDecoder path. Bypass T1GuidedCoarseHead
        # entirely and feed bottleneck + all 3 skips directly to NAFDecoder.
        if self.use_naf_fusion:
            # Skips ordered coarse-first → fine-last:
            # [skip_32 (idx depth-2), skip_64 (idx depth-3), skip_128 (idx 0)]
            naf_skips = [asl_skips[depth - 2 - i] for i in range(depth - 1)]
            return self.naf_decoder(feat_map, naf_skips)
        asl_skip_l1 = asl_skips[depth - 2]
        t1_skip_l1 = t1_skips[depth - 2] if (self.use_t1_cross_fusion and t1_skips is not None) else None
        feat_l1 = self.t1_head(
            asl_feat_bot=feat_map,
            t1_feat_bot=t1_feat_map,
            asl_skip_l1=asl_skip_l1,
            t1_skip_l1=t1_skip_l1,
            tissue_seg=head_tissue_seg,
        )
        # Detail skips, ordered high-res-last: [skip_64, skip_128] for depth=4.
        asl_skips_detail = list(reversed(asl_skips[: depth - 2]))
        # Same ordering for the T1 side; consumed only by the window fusions.
        t1_skips_detail = (list(reversed(t1_skips[: depth - 2]))
                           if t1_skips is not None else None)
        return self.asl_decoder(feat_l1, asl_skips_detail,
                                t1_skips_detail=t1_skips_detail,
                                tissue_seg=head_tissue_seg)

    def window_stats(self) -> Dict[str, float]:
        """Per-level window-fusion probes from the most recent forward:
        gate = sigmoid(a) (how much guidance the model asked for), entropy (ln N
        == uniform averaging, i.e. grouping has NOT been learnt), and delta =
        ||x'-x||/||x||. Empty when window_fusion_levels=0."""
        out: Dict[str, float] = {}
        dec = getattr(self, "asl_decoder", None)
        fusions = getattr(dec, "window_fusions", None) if dec is not None else None
        if fusions is None:
            return out
        for i, m in enumerate(fusions):
            for k, v in getattr(m, "last_stats", {}).items():
                out[f"wf{i}_{k}"] = v
        return out

    def _route_encoder_input(
        self, agg: Tensor, t1: Tensor, t1_skips: list
    ) -> Tuple[Tensor, list]:
        """Baseline-aware routing of the ASL encoder input + guidance skips.

        Full model / zero_t1: returns (agg, t1_skips) unchanged (zero_t1 acts
        upstream by blanking the t1 tensor before the T1 encoder).
        naive_t1_concat: concatenates T1 as a 2nd input channel (T1 → value path)
        and zeros the structured guidance skips so T1's ONLY route is the concat.
        """
        if not self.naive_t1_concat:
            return agg, t1_skips
        t1_r = (
            t1 if t1.shape[-2:] == agg.shape[-2:]
            else F.interpolate(t1, size=agg.shape[-2:], mode="bilinear", align_corners=False)
        )
        enc_in = torch.cat([agg, t1_r], dim=1)                      # [B,2,H,W] — T1 into V
        enc_t1_skips = [torch.zeros_like(s) for s in t1_skips]      # neutralize guidance
        return enc_in, enc_t1_skips

    def _build_pv_pyramid(self, pv: Tensor, depth: int, with_pbar: bool = True) -> list:
        """Soft-PV composition pyramid for EC-LRDA: per stage p (and, unless pv_geo,
        the neighbourhood-mean p̄) area-resized to the stage resolution. Area interp
        preserves partial-volume fractions on downsample. Returns depth tensors
        [B, 2*pv_ch, hw_s, hw_s] (with p̄) or [B, pv_ch, hw_s, hw_s] (pv_geo, p only —
        the p̄ channel is 0.99+ redundant, see scripts/diag_pv_pyramid.py)."""
        p_bar = F.avg_pool2d(pv, kernel_size=3, stride=1, padding=1) if with_pbar else None
        hw0 = pv.shape[-1]
        pyr = []
        for s in range(depth):
            hw = max(1, hw0 // (2 ** s))
            ps = pv if hw == hw0 else F.interpolate(pv, size=(hw, hw), mode="area")
            if not with_pbar:
                pyr.append(ps)
                continue
            pbs = p_bar if hw == hw0 else F.interpolate(p_bar, size=(hw, hw), mode="area")
            pyr.append(torch.cat([ps, pbs], dim=1))
        return pyr

    def set_mc_scan_dropout(self, active: bool) -> None:
        """Toggle scan-direction dropout at INFERENCE on every SS2D block (batch-3 MC
        uncertainty). With it on, run N stochastic infer passes → per-voxel std = uncertainty
        (mean = ensembled recon). No-op if the model was trained with scan_dropout_p=0."""
        for m in self.modules():
            if hasattr(m, "_mc_scan_dropout"):
                m._mc_scan_dropout = bool(active)

    def set_cond_pv(self, pv_ext: Optional[Tensor]) -> None:
        """Provide the per-batch FSL-PV conditioning tensor ``[B,3,H,W]``
        (brain-masked GM/WM/CSF, e.g. from :func:`build_fsl_cond_pv`) used when
        ``lrda_cond_src='fsl'``. MUST be called before every forward/infer on an
        fsl model — a missing call makes the forward raise (never silently reuse a
        stale batch). No-op semantics for other cond_src (the attr is ignored)."""
        self._cond_pv_ext = pv_ext

    def forward(
        self,
        set_a: Tensor,
        set_b: Tensor,
        t1: Tensor,
        len_a: Optional[Tensor] = None,
        len_b: Optional[Tensor] = None,
        mask_a: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # Input path: only set_a goes through the aggregator + encoder + decoder.
        # set_b is only used to provide the N2N training target (direct mean).
        agg_a, w_a = (
            self.aggregate_asl(set_a, mask=mask_a)
            if mask_a is not None
            else self.aggregate_asl(set_a, lengths=len_a)
        )
        # 2.5D: agg_a is [B, in_ch*n_slices, H, W]; the encoder consumes ALL slices,
        # but the returned output/target are the CENTER slice only. (c0:c1 = center;
        # for slice_context=0 this is [0:in_ch] i.e. a no-op → 2D behaviour.)
        c0, c1 = self.center_ch, self.center_ch + self.in_ch

        # Encode T1 (feature map + skip connections) and produce segmentation
        # logits up-front: t1_seg drives tissue-class similarity bias in
        # CrossModalFusion at the bottleneck and at every multi-scale fusion in
        # the ASL decoder. When --freeze_t1 is set, t1_seg has no grad through
        # T1 params; the cross-attention modules still get a fresh per-sample
        # tissue prior every forward pass.
        if self.zero_t1:                       # ASL-only-MoSSM baseline
            t1 = torch.zeros_like(t1)
        if self.no_t1_branch:
            # fsl/rawt1 conditioning: the T1 seg branch is INERT (external/raw-T1 PV,
            # T1-free decoder, loss/brain mask = t1>thr, no gate) → skip it entirely.
            t1_feat_map = t1_skips = t1_seg_logits = t1_recon_out = None
        elif self.t1_decoder is None:
            # No-seg default arm: T1 ENCODER only. Its features are the CMF0/CMF1
            # K-path (T1's sole route into the ASL branch); the recon head was
            # dropped as dead weight (use_t1_decoder=False).
            t1_feat_map, _, t1_skips = self.t1_encoder(t1)
            t1_seg_logits = t1_recon_out = None
        else:
            t1_feat_map, _, t1_skips = self.t1_encoder(t1)
            t1_dec_out = self.t1_decoder(t1_feat_map, t1_skips)  # [B,4,H,W] seg or [B,1,H,W] recon
            # In 'recon' mode there are no PV-seg logits; downstream tissue-bias
            # consumers (TABS / CMF) are disabled, so t1_seg_logits is None.
            t1_seg_logits = t1_dec_out if self.t1_task == "seg" else None
            t1_recon_out = t1_dec_out if self.t1_task == "recon" else None
        _ndepth = self._n_stages if t1_skips is None else len(t1_skips)   # pyramid stages

        # v42 (use_mossm_encoder): two T1 channels are wired into the encoder:
        #   1. t1_skips → MoSSM B/Δ at every stage (state-dynamics modulation)
        #   2. tabs_seg = t1_seg_logits → MoSSM scan reordering (TABS / N1)
        # (The former Var_T(set_a) noise channel was removed 2026-06-21.)
        tabs_seg = t1_seg_logits if self.use_tabs else None
        enc_in, enc_t1_skips = self._route_encoder_input(agg_a, t1, t1_skips)
        repro_gate, repro_desc = self._repro_signals(set_a, len_a, mask_a, t1)
        coupling_gate = self._compute_coupling_gate(agg_a[:, c0:c1], t1)
        # EC-LRDA: replace the raw-T1-feature skips with a soft-PV pyramid and the
        # legacy guidance gates with γ = r_rep · c_sem. PV pyramid drives the SS2D
        # bilinear conditioning; γ scales the (zero-init, B/Δ-only) modulation ⇒ V=ASL.
        ec_out = None
        if self.ec_lrda and (t1_seg_logits is not None or self.lrda_cond_src in ("fsl", "fsl_t1", "rawt1")):
            if self.lrda_cond_src in ("fsl", "fsl_t1"):
                pv = self._cond_pv_ext                              # [B,3,H,W] brain-masked FSL GM/WM/CSF (no BG)
                if pv is None:
                    raise ValueError(
                        "lrda_cond_src='fsl' requires model.set_cond_pv([B,3,H,W]) before forward().")
            elif self.lrda_cond_src == "rawt1":
                pv = t1[:, :1]                                      # raw-T1 (pyramid built from t1 below); pv only used by pv_mode/gate
            else:
                pv = torch.softmax(t1_seg_logits, dim=1)           # [B,4,H,W] = GM,WM,CSF,BG (ablation)
            if self.pv_mode == "zero":
                pv = torch.zeros_like(pv)                           # M0: bias-free E => modulation EXACTLY identity (plain VSS on ASL)
            elif self.pv_mode == "uniform":
                pv = torch.full_like(pv, 1.0 / pv.shape[1])         # constant tissue prior
            if self.lrda_cond_src == "rawt1":
                enc_t1_skips = self._build_pv_pyramid(t1[:, :1], _ndepth, with_pbar=False)   # Raw-T1-bilinear: raw T1↓ into s (no PV, no gate)
            elif getattr(self, "lrda_cond_deep", False) and getattr(self, "cond_pyramid", None) is not None:
                _hw0 = pv.shape[-1]                                  # tier-1: learned conditioning pyramid
                _tgt = [max(1, _hw0 // (2 ** s)) for s in range(_ndepth)]
                enc_t1_skips = self.cond_pyramid(pv, target_hw=_tgt)
            else:
                enc_t1_skips = self._build_pv_pyramid(pv, _ndepth, with_pbar=not self.ec_pv_geo)
            coupling_gate = None
            if self.ec_calibrate:
                # cross-subject PV for the L_sem match>mismatch ranking. roll-by-1 on
                # the batch is a valid derangement only for B>=2; at B==1 it returns
                # the SAME pv → c_sem_mis==c_sem → L_sem is a no-op (skip it).
                pv_mis = (torch.roll(pv, shifts=1, dims=0)
                          if (self.training and pv.shape[0] >= 2) else None)
                ec_out = self.ec_guidance(
                    set_a, agg_a, pv, lengths=len_a, mask=mask_a,
                    pv_mismatch=pv_mis, use_rep=self.ec_use_rep)
                g = ec_out["gamma"]
                repro_gate = g.detach() if self.ec_sg_gamma else g   # sg ⇒ R learns only from L_rep/L_sem
                repro_desc = None
            else:
                repro_gate, repro_desc = None, None                 # A1: pure soft-PV bilinear
        feat_map, asl_skips = self._encode_asl_branch(
            enc_in, t1_skips=enc_t1_skips, tabs_seg=tabs_seg,
            t1_seg_probs=None, repro_gate=repro_gate, repro_desc=repro_desc,
            coupling_gate=coupling_gate,
        )

        # T1GuidedCoarseHead (L0+L1) + ASLDetailDecoder (L2+L3).
        head_tissue_seg = t1_seg_logits if self.use_t1_cross_fusion else None
        asl_out = self._decode_asl(
            feat_map=feat_map, asl_skips=asl_skips,
            t1_feat_map=t1_feat_map, t1_skips=t1_skips,
            head_tissue_seg=head_tissue_seg,
        )
        if self.use_heteroscedastic:
            asl_recon = asl_out[:, :1]
            asl_logvar = asl_out[:, 1:2]
        else:
            asl_recon = asl_out
            asl_logvar = None

        # ResHead-with-cap: model predicts a bounded additive correction to agg.
        # asl_recon = agg + δ_max · tanh(raw); |asl_recon − agg| ≤ δ_max.
        if self.use_reshead:
            asl_recon = agg_a[:, c0:c1] + self.reshead_delta_max * torch.tanh(asl_recon)

        # PVC residual decomposition (opt-in): ŷ = E + w⊙r̂. E=P·m̄ is the closed-form
        # tissue-explained base; asl_recon is reinterpreted as the residual r̂ (the N2N
        # loss on ŷ trains it so). V=ASL preserved (m̄ from ASL; E∈span(ASL)).
        pvc_base = pvc_w = None
        if self.pvc_head is not None and t1_seg_logits is not None:
            pv_pvc = torch.softmax(t1_seg_logits, dim=1)            # [B,4,H,W]
            pvc_out = self.pvc_head(
                agg_a[:, c0:c1], pv_pvc, asl_recon, set_a,
                lengths=len_a, mask=mask_a, no_wgate=self.pvc_no_wgate)
            asl_recon, pvc_base, pvc_w = pvc_out["recon"], pvc_out["base"], pvc_out["w"]

        # N2N target = direct unweighted mean of set_b (CENTER slice for 2.5D)
        target_b = _direct_mean(set_b, lengths=len_b, mask=mask_b)[:, c0:c1]

        return {
            "agg":         agg_a[:, c0:c1],
            "weights":     w_a,
            "asl_recon":   asl_recon,
            "asl_logvar":  asl_logvar,   # None if not heteroscedastic
            "t1_recon":    t1_recon_out,
            "t1_seg":      t1_seg_logits,
            "target":      target_b,
            "pvc_base":    pvc_base,      # E=P·m̄ (None unless --pvc_residual)
            "pvc_w":       pvc_w,         # reproducibility gate w
            # EC-LRDA reliability/consistency maps (for L_rep / L_sem + diagnostics)
            "ec_r_rep":    (ec_out["r_rep"] if ec_out is not None else None),
            "ec_c_sem":    (ec_out["c_sem"] if ec_out is not None else None),
            "ec_c_sem_mis": (ec_out.get("c_sem_mis") if ec_out is not None else None),
            "ec_gamma":    (ec_out["gamma"] if ec_out is not None else None),
        }

    @torch.no_grad()
    def infer_from_subset(
        self,
        set_x: Tensor,
        t1: Tensor,
        lengths: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Infer PWI from a subset of ASL frames.

        Args:
            set_x:   (B, T, 1, H, W) ASL frame subset
            t1:      (B, 1, H, W) T1w image
            lengths: (B,) number of valid frames (or None)
            mask:    (B, T) padding mask (or None)

        Returns:
            dict with keys: asl_recon, t1_recon, agg, weights, feat_asl
        """
        agg, weights = (
            self.aggregate_asl(set_x, mask=mask)
            if mask is not None
            else self.aggregate_asl(set_x, lengths=lengths)
        )

        if self.zero_t1:                       # ASL-only-MoSSM baseline
            t1 = torch.zeros_like(t1)
        if self.no_t1_branch:                  # inert T1 seg branch skipped (see forward())
            t1_feat_map = t1_skips = t1_seg_logits = t1_recon_out = None
        elif self.t1_decoder is None:          # no-seg arm: encoder-only (see forward())
            t1_feat_map, _, t1_skips = self.t1_encoder(t1)
            t1_seg_logits = t1_recon_out = None
        else:
            t1_feat_map, _, t1_skips = self.t1_encoder(t1)
            t1_dec_out = self.t1_decoder(t1_feat_map, t1_skips)     # [B,4,H,W] seg or [B,1,H,W] recon
            # Gate the decoder output by task, exactly like forward(): in 'recon'
            # mode there are no 4-class PV-seg logits, so t1_seg_logits is None (the
            # 1-channel recon must NOT masquerade as a seg — it breaks the val seg
            # panel and any tissue-bias consumer that assumes 4 channels).
            t1_seg_logits = t1_dec_out if self.t1_task == "seg" else None
            t1_recon_out  = t1_dec_out if self.t1_task == "recon" else None
        _ndepth = self._n_stages if t1_skips is None else len(t1_skips)
        # v42 inference path: feed TABS seg (from t1_decoder) into the encoder.
        # (The former Var_T(set_x) noise channel was removed 2026-06-21.)
        tabs_seg = t1_seg_logits if self.use_tabs else None
        enc_in, enc_t1_skips = self._route_encoder_input(agg, t1, t1_skips)
        repro_gate, repro_desc = self._repro_signals(set_x, lengths, mask, t1)
        _cc0, _cc1 = self.center_ch, self.center_ch + self.in_ch
        coupling_gate = self._compute_coupling_gate(agg[:, _cc0:_cc1], t1)
        # EC-LRDA inference: same soft-PV pyramid + γ = r_rep·c_sem as forward()
        # (no mismatch branch at inference). ec_out is surfaced in the return dict so
        # eval can visualise the r_rep / c_sem / γ gate maps (diagnostic only — the
        # maps never enter the value path; V=ASL is unaffected).
        ec_out = None
        if self.ec_lrda and (t1_seg_logits is not None or self.lrda_cond_src in ("fsl", "fsl_t1", "rawt1")):
            if self.lrda_cond_src in ("fsl", "fsl_t1"):
                pv = self._cond_pv_ext                              # [B,3,H,W] brain-masked FSL GM/WM/CSF (no BG)
                if pv is None:
                    raise ValueError(
                        "lrda_cond_src='fsl' requires model.set_cond_pv([B,3,H,W]) before infer_from_subset().")
            elif self.lrda_cond_src == "rawt1":
                pv = t1[:, :1]                                      # raw-T1 (pyramid from t1 below); pv only touched by pv_mode/gate
            else:
                pv = torch.softmax(t1_seg_logits, dim=1)
            if self.pv_mode == "zero":
                pv = torch.zeros_like(pv)                           # M0 (inference): must match training pv_mode
            elif self.pv_mode == "uniform":
                pv = torch.full_like(pv, 1.0 / pv.shape[1])
            if self.lrda_cond_src == "rawt1":
                enc_t1_skips = self._build_pv_pyramid(t1[:, :1], _ndepth, with_pbar=False)   # Raw-T1-bilinear (inference)
            elif getattr(self, "lrda_cond_deep", False) and getattr(self, "cond_pyramid", None) is not None:
                _hw0 = pv.shape[-1]                                  # tier-1: learned conditioning pyramid (inference)
                _tgt = [max(1, _hw0 // (2 ** s)) for s in range(_ndepth)]
                enc_t1_skips = self.cond_pyramid(pv, target_hw=_tgt)
            else:
                enc_t1_skips = self._build_pv_pyramid(pv, _ndepth, with_pbar=not self.ec_pv_geo)
            coupling_gate = None
            if self.ec_calibrate:
                ec_out = self.ec_guidance(
                    set_x, agg, pv, lengths=lengths, mask=mask,
                    pv_mismatch=None, use_rep=self.ec_use_rep)
                repro_gate, repro_desc = ec_out["gamma"], None
            else:
                repro_gate, repro_desc = None, None
        feat_map, asl_skips = self._encode_asl_branch(
            enc_in, t1_skips=enc_t1_skips, tabs_seg=tabs_seg,
            t1_seg_probs=None, repro_gate=repro_gate, repro_desc=repro_desc,
            coupling_gate=coupling_gate,
        )

        head_tissue_seg = t1_seg_logits if self.use_t1_cross_fusion else None
        asl_out = self._decode_asl(
            feat_map=feat_map, asl_skips=asl_skips,
            t1_feat_map=t1_feat_map, t1_skips=t1_skips,
            head_tissue_seg=head_tissue_seg,
        )
        if self.use_heteroscedastic:
            asl_recon = asl_out[:, :1]
            asl_logvar = asl_out[:, 1:2]
        else:
            asl_recon = asl_out
            asl_logvar = None

        c0, c1 = self.center_ch, self.center_ch + self.in_ch    # center slice (2.5D; no-op if 2D)
        if self.use_reshead:
            asl_recon = agg[:, c0:c1] + self.reshead_delta_max * torch.tanh(asl_recon)

        pvc_base = pvc_w = None
        if self.pvc_head is not None and t1_seg_logits is not None:
            pv_pvc = torch.softmax(t1_seg_logits, dim=1)
            pvc_out = self.pvc_head(
                agg[:, c0:c1], pv_pvc, asl_recon, set_x,
                lengths=lengths, mask=mask, no_wgate=self.pvc_no_wgate)
            asl_recon, pvc_base, pvc_w = pvc_out["recon"], pvc_out["base"], pvc_out["w"]

        return {
            "asl_recon": asl_recon,
            "asl_logvar": asl_logvar,
            "t1_recon":  t1_recon_out,
            "t1_seg":    t1_seg_logits,
            "agg":       agg[:, c0:c1],
            "weights":   weights,
            "pvc_base":  pvc_base,       # E=P·m̄ (None unless --pvc_residual)
            "pvc_w":     pvc_w,          # reproducibility gate w
            # EC-LRDA gate maps (None unless ec_lrda + ec_calibrate). Diagnostic only.
            "ec_r_rep":  (ec_out["r_rep"] if ec_out is not None else None),
            "ec_c_sem":  (ec_out["c_sem"] if ec_out is not None else None),
            "ec_gamma":  (ec_out["gamma"] if ec_out is not None else None),
        }
