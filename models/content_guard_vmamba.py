"""VMamba backbone for ASL denoising — baseline + optional content-guard (2026-07).

A clean VMamba (VSS-block, 2D 4-direction cross-scan; Liu et al. NeurIPS 2024)
encoder that plugs into `ASLT1Denoiser` with the SAME forward contract as
`ContentGuidedConvEncoder2D` / `MoSSMEncoder2D`:

    forward(x, t1_skips, tissue_seg=None, t1_seg_probs=None) -> (feat, None, skips)

Two roles from one class (`use_guard`):
  * `use_guard=False` — pure VMamba, ASL-only. The **backbone baseline** for the
    "plain U-Net vs MoSSM vs VMamba" comparison (fair: same [128,64,32,16]
    resolution ladder and skip shapes as the conv/MoSSM encoders, unlike a
    patch-embedded VMamba).
  * `use_guard=True` — **CIG-VMamba**: the SAME content-guard as CIG-UNet
    (LRDAConv2d all stages + AKMRConv2d low-res), demonstrating the V=ASL
    content-isolation invariant is backbone-agnostic.

Reuses the repo's `VSSBlock` / `_PatchMerge` (pure-PyTorch selective scan on
CPU/Windows, mamba_ssm CUDA kernel on GPU) and the conv content-guard modules.
VSS blocks run in NHWC; the guard + skips are NCHW (permuted at the boundary).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from .asl_mamba_student import VSSBlock, _PatchMerge, _PatchMergeConv
except Exception:  # pragma: no cover
    from asl_mamba_student import VSSBlock, _PatchMerge, _PatchMergeConv  # type: ignore

try:
    from .content_guard_conv import AKMRConv2d, LRDAConv2d, RKMR
except Exception:  # pragma: no cover
    from content_guard_conv import AKMRConv2d, LRDAConv2d, RKMR  # type: ignore


class VMambaEncoder2D(nn.Module):
    """4-stage VSS-block encoder over [128,64,32,16] with optional content-guard."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        input_hw: int = 128,
        d_state: int = 16,
        n_directions: int = 2,
        hv_scan: bool = False,             # n=2 scan as {→,↓} instead of {→,←}
        blocks_per_scale: int = 2,
        mlp_ratio: float = 2.0,
        # single-branch conv inductive bias (2026-07); all default OFF = unchanged.
        convffn: bool = False,
        overlap_merge: bool = False,
        lrda_inscan: bool = False,
        lrda_cond_asl: bool = False,       # Design A: φ conditioned on [T1 ; ASL]
        lrda_dt_bound: float = 1.0,        # in-scan Δ log-modulation bound: Δ ∈ [e^-b, e^b]·Δ
        lrda_bilinear: bool = False,       # EC-LRDA: soft-PV bilinear (s(PV)⊙a(ASL)) conditioning
        lrda_pv_geo: bool = False,         # Level-1: PV-only geometry encoder (drop p̄, t1_skip = pv_ch)
        lrda_pv_ch: int = 4,               # PV tissue classes (t1_skip carries [p;p̄] = 2*pv_ch, or p = pv_ch if pv_geo)
        # content-guard (CIG-VMamba); use_guard=False -> pure VMamba baseline
        use_guard: bool = True,
        t1_channels_per_scale: Optional[List[int]] = None,
        lrda_rank: int = 8,
        lrda_bound: float = 4.0,
        lrda_variant: str = "c",
        lrda_stages: Optional[List[int]] = None,
        akmr_enabled: bool = True,
        akmr_max_tokens: int = 1024,
        akmr_heads: int = 4,
        akmr_k_rank_div: int = 4,
        # RKMR: replace AKMR's T1 key with a reproducible-ASL key (T1-free).
        akmr_repro_key: bool = False,
        rkmr_t1_veto: bool = False,
        repro_desc_dim: int = 8,
        sig_xin: bool = False,             # (2) a_B reads x_in (d_inner) instead of B_ (d_state)
        dt_rank: int = 1,                  # (3) input-dependent Δ projection rank (0=auto=ceil(d_model/16))
        tfdm: bool = False,                # Tissue-Factored Dynamics Modulation (per-tissue W_k routed by PV)
        scan_dropout_p: float = 0.0,       # batch-3: scan-direction dropout (train regulariser + MC-uncertainty)
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.use_guard = bool(use_guard)
        channels = [base_ch * (2 ** i) for i in range(depth)]
        t1ch = list(t1_channels_per_scale) if t1_channels_per_scale is not None else list(channels)
        self.out_ch = channels[-1]

        # Which stages carry LRDA (post-block conv OR in-scan B-gate).
        lrda_stages = list(range(depth)) if lrda_stages is None else list(lrda_stages)
        # In-scan LRDA lives INSIDE the SS2D of guarded stages and REPLACES the
        # post-block LRDAConv2d there (so it is not applied twice).
        self.lrda_inscan = bool(use_guard and lrda_inscan)
        self.lrda_inscan_on = [self.lrda_inscan and (s in lrda_stages) for s in range(depth)]

        # Full-resolution stem (no patch-embed downsample) so the stage ladder
        # is [128,64,32,16] — identical to the conv/MoSSM encoders.
        self.stem = nn.Conv2d(in_ch, channels[0], kernel_size=3, padding=1)
        self.stages = nn.ModuleList([
            nn.ModuleList([
                VSSBlock(
                    channels[s], d_state=d_state, mlp_ratio=mlp_ratio,
                    n_directions=n_directions, use_conv_ffn=bool(convffn),
                    lrda_inscan=self.lrda_inscan_on[s],
                    lrda_t1_ch=(t1ch[s] if self.lrda_inscan_on[s] else None),
                    lrda_rank=lrda_rank, lrda_bound=lrda_bound,
                    lrda_cond_asl=bool(lrda_cond_asl), lrda_dt_bound=float(lrda_dt_bound),
                    lrda_bilinear=bool(lrda_bilinear), lrda_pv_ch=int(lrda_pv_ch),
                    lrda_pv_geo=bool(lrda_pv_geo), hv_scan=bool(hv_scan),
                    sig_xin=bool(sig_xin), dt_rank=int(dt_rank), tfdm=bool(tfdm),
                    scan_dropout_p=float(scan_dropout_p),
                )
                for _ in range(int(blocks_per_scale))
            ]) for s in range(depth)
        ])
        merge_cls = _PatchMergeConv if overlap_merge else _PatchMerge
        self.merges = nn.ModuleList([
            merge_cls(channels[s], channels[s + 1]) for s in range(depth - 1)
        ])

        # Content-guard (identical to CIG-UNet). Built only when use_guard.
        if self.use_guard:
            # Post-block LRDAConv2d is disabled on stages handled by in-scan LRDA.
            self.lrda_on = [(s in lrda_stages) and not self.lrda_inscan_on[s] for s in range(depth)]
            self.lrda = nn.ModuleList([
                LRDAConv2d(channels[s], t1ch[s], rank=lrda_rank, bound=lrda_bound, variant=lrda_variant)
                if self.lrda_on[s] else nn.Identity()
                for s in range(depth)
            ])
            hw_per_scale = [max(1, int(input_hw) // (2 ** s)) for s in range(depth)]
            self.akmr_on = [
                bool(akmr_enabled) and (hw_per_scale[s] * hw_per_scale[s] <= int(akmr_max_tokens))
                for s in range(depth)
            ]
            self.akmr_repro_key = bool(akmr_repro_key)
            if self.akmr_repro_key:
                # RKMR: reproducibility-keyed (T1-free) memory reader replaces AKMR.
                self.akmr = nn.ModuleList([
                    RKMR(channels[s], int(repro_desc_dim), n_heads=akmr_heads,
                         t1_veto=bool(rkmr_t1_veto), t1_ch=t1ch[s], veto_rank_div=akmr_k_rank_div)
                    if self.akmr_on[s] else nn.Identity()
                    for s in range(depth)
                ])
            else:
                self.akmr = nn.ModuleList([
                    AKMRConv2d(channels[s], t1ch[s], n_heads=akmr_heads, k_rank_div=akmr_k_rank_div)
                    if self.akmr_on[s] else nn.Identity()
                    for s in range(depth)
                ])
        else:
            self.lrda_on = [False] * depth
            self.akmr_on = [False] * depth
            self.akmr_repro_key = False

    def _guard(self, s: int, feat_nchw: Tensor, t1_nchw: Tensor,
               gate: Optional[Tensor] = None, phi: Optional[Tensor] = None) -> Tensor:
        if self.lrda_on[s]:
            feat_nchw = self.lrda[s](feat_nchw, t1_nchw)
        if self.akmr_on[s]:
            if self.akmr_repro_key:
                feat_nchw = self.akmr[s](feat_nchw, phi, t1_nchw)      # RKMR(x, φ, t1[veto])
            else:
                feat_nchw = self.akmr[s](feat_nchw, t1_nchw, gate)     # AKMR(x, t1, gate)
        return feat_nchw

    def forward(
        self,
        x: Tensor,
        t1_skips: Optional[List[Tensor]] = None,
        tissue_seg: Optional[Tensor] = None,
        t1_seg_probs: Optional[Tensor] = None,
        repro_gate: Optional[Tensor] = None,     # Design C: [B,1,H,W] in (0,1)
        repro_desc: Optional[Tensor] = None,     # RKMR: [B,d,H,W] reproducible descriptor φ
    ) -> Tuple[Tensor, None, List[Tensor]]:
        if self.use_guard:
            assert t1_skips is not None and len(t1_skips) >= self.depth, \
                "CIG-VMamba (use_guard) requires depth T1 skips"
        h = self.stem(x).permute(0, 2, 3, 1).contiguous()      # NCHW -> NHWC
        skips: List[Tensor] = []
        for s in range(self.depth):
            # In-scan LRDA: feed the stage-s T1 skip (NCHW) into each block's SS2D
            # B-gate. Resolution matches (both at hw[s]); the phi Conv2d + per-
            # direction serialize align it to the ASL token order.
            t1_s = t1_skips[s] if self.lrda_inscan_on[s] else None
            hw = (h.shape[1], h.shape[2])
            # Design C reliability gate, avg-pooled to this stage's resolution.
            r_s = None
            if repro_gate is not None:
                r_s = repro_gate if repro_gate.shape[-2:] == hw else \
                    F.adaptive_avg_pool2d(repro_gate, hw)
            # RKMR descriptor φ, avg-pooled to this stage's resolution.
            phi_s = None
            if self.akmr_repro_key and repro_desc is not None:
                phi_s = repro_desc if repro_desc.shape[-2:] == hw else \
                    F.adaptive_avg_pool2d(repro_desc, hw)
            for blk in self.stages[s]:
                h = blk(h, t1_s, r_s)                          # NHWC
            feat_nchw = h.permute(0, 3, 1, 2).contiguous()     # NHWC -> NCHW
            if self.use_guard:
                assert feat_nchw.shape[-2:] == t1_skips[s].shape[-2:], \
                    f"stage{s} ASL {tuple(feat_nchw.shape[-2:])} vs T1 {tuple(t1_skips[s].shape[-2:])}"
                feat_nchw = self._guard(s, feat_nchw, t1_skips[s], r_s, phi_s)
            skips.append(feat_nchw)                            # NCHW skip
            if s < self.depth - 1:
                h = feat_nchw.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC (post-guard)
                h = self.merges[s](h)                          # 2x down, channels[s]->channels[s+1]
        return skips[-1], None, skips
