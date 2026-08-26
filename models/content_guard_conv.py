"""Content-guarded convolutional ASL encoder (CIG-UNet, 2026-07).

The v2 content-guard (CADA-LR / AKMR) lived inside a Mamba/SSM encoder. The
plain conv U-Net (PlainUNet2D) empirically beat that SSM backbone on both uMSE
and CNR (seed-42 sweep), while the T1 guidance itself was shown to help *within*
the SSM family. This module grafts the SAME content-guard invariants onto the
conv backbone so the guidance rides on the stronger encoder.

Design (see ASL_dmvae/docs/cig_unet_design.md):
  * Backbone == PlainUNet2D's ``ConvEncoder2D`` (identical ResidualBlock stages),
    so with all guards zero-initialised the encoder is byte-for-byte the vanilla
    winner at step 0.
  * ``LRDAConv2d`` — the conv analogue of CADA-LR/LRDA. T1 emits ONLY a bounded,
    zero-init, low-rank MULTIPLICATIVE gate on ASL's own features; no additive T1
    term (FiLM's β is deliberately dropped — β is the content-injection channel).
    A strictly stronger V=ASL statement than the SSM version (no C/u readout to
    reason about): ``y = x + U( s(t1) ⊙ mix(V(x)) )``, ``s = bound·tanh(φ(t1))``,
    ``U`` zero-init ⇒ ∂y/∂t1 routes only through the multiplicative gate.
  * ``AKMRConv2d`` — cross-attention, Q=ASL, K=T1 (low-rank), V=ASL. Ported
    verbatim from the SSM AKMR; zero-init out-proj. Low-res stages only.

Hard constraints (CLAUDE.md §3) preserved:
  V = ASL everywhere (T1 never a value/content source); decoder stays T1-free by
  living outside this module; guards zero-init so the model starts == vanilla.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from .blocks import DownBlock, ResidualBlock
except Exception:  # pragma: no cover - script/relative import fallback
    from blocks import DownBlock, ResidualBlock  # type: ignore


# ---------------------------------------------------------------------------
# LRDA-Conv — T1-gated low-rank multiplicative residual on ASL features.
# ---------------------------------------------------------------------------
class LRDAConv2d(nn.Module):
    """Low-Rank Dynamics Adapter, conv variant (content-safe).

    ``y = x + U( s ⊙ mix(V(x)) )`` with ``s = bound · tanh(φ(t1))``.

    * ``V`` : 1×1 conv, ASL → rank ``r``  (the CONTENT factor — ASL only).
    * ``mix``: depthwise 3×3 on the ``r`` channels (variant 'c') — local spatial
      mixing, the conv analogue of the SSM's B-matrix "dynamics".
    * ``φ`` : 1×1 conv, T1 → ``r``, bounded by ``bound·tanh`` (the OPERATOR — T1
      only). No additive T1 term anywhere.
    * ``U`` : 1×1 conv, ``r`` → ASL, ZERO-INIT ⇒ residual is 0 at step 0.

    Variants (the sharpness/strength knob):
      'a' — per-channel global gate (pooled T1): weakest, == β-free TissueFiLM.
      'b' — per-pixel low-rank spatial gate, no depthwise mixing.
      'c' — per-pixel low-rank spatial gate + depthwise mixing (default).

    T1 (``t1``) and ASL (``x``) must share H×W (guaranteed: t1_skips[s] resolution
    equals ASL stage-s resolution — both come from a depth-matched ConvEncoder2D).
    """

    def __init__(
        self,
        ch: int,
        t1_ch: int,
        rank: int = 8,
        bound: float = 4.0,
        variant: str = "c",
    ) -> None:
        super().__init__()
        r = max(1, int(rank))
        self.rank = r
        self.bound = float(bound)
        self.variant = str(variant)
        if self.variant not in ("a", "b", "c"):
            raise ValueError(f"LRDAConv2d variant must be a/b/c, got {variant!r}")

        # Content factor: ASL-only low-rank projection.
        self.V = nn.Conv2d(ch, r, kernel_size=1, bias=False)
        # Local spatial mixing (dynamics) — variant 'c' only.
        self.dw = (
            nn.Conv2d(r, r, kernel_size=3, padding=1, groups=r, bias=False)
            if self.variant == "c" else None
        )
        # Operator: T1 → r bounded gate coefficients (T1 ONLY).
        self.phi = nn.Conv2d(t1_ch, r, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1) if self.variant == "a" else None
        # Expand back to ASL channels — ZERO-INIT ⇒ identity at start.
        self.U = nn.Conv2d(r, ch, kernel_size=1, bias=False)
        nn.init.zeros_(self.U.weight)

    def forward(self, x: Tensor, t1: Tensor, gate: Optional[Tensor] = None) -> Tensor:
        v = self.V(x)                              # [B, r, H, W]  ASL content
        if self.dw is not None:
            v = self.dw(v)                         # local spatial mixing
        g = self.pool(t1) if self.pool is not None else t1
        s = self.bound * torch.tanh(self.phi(g))   # [B, r, (1|H), (1|W)]  T1 gate
        sv = s * v                                 # [B, r, H, W]  T1 modulation of ASL content
        if gate is not None:                       # coupling gate c∈(0,1): scale T1's
            if gate.shape[-2:] != sv.shape[-2:]:   # influence by T1–ASL agreement
                gate = F.adaptive_avg_pool2d(gate, sv.shape[-2:])
            sv = sv * gate                         # [B,1,H,W] broadcast over r channels
        dx = self.U(sv)                            # [B, ch, H, W], 0 at init (U zero-init)
        return x + dx


# ---------------------------------------------------------------------------
# AKMR-Conv — anatomy-keyed cross-attention. Q=ASL, K=T1 (low-rank), V=ASL.
# ---------------------------------------------------------------------------
class AKMRConv2d(nn.Module):
    """Anatomy-Keyed Memory Reader (conv wrapper).

    T1 is the addressing KEY only; the VALUE is always the ASL feature map, so
    T1 supplies no pixel content to the attention output (V=ASL invariant). The
    out-projection is zero-init so the module contributes exactly 0 at step 0.
    Used at low-resolution stages only (attention is O(N²)).
    """

    def __init__(
        self,
        ch: int,
        t1_ch: int,
        n_heads: int = 4,
        k_rank_div: int = 4,
    ) -> None:
        super().__init__()
        heads = max(1, int(n_heads))
        while ch % heads != 0 and heads > 1:
            heads //= 2
        self.norm_q = nn.GroupNorm(min(8, ch), ch)
        self.norm_k = nn.GroupNorm(min(8, t1_ch), t1_ch)
        self.norm_v = nn.GroupNorm(min(8, ch), ch)
        kdim = max(1, t1_ch // max(1, int(k_rank_div)))
        self.k_compress = (
            nn.Conv2d(t1_ch, kdim, kernel_size=1, bias=False)
            if kdim != t1_ch else None
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=ch, num_heads=heads, kdim=kdim, vdim=ch, batch_first=True,
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: Tensor, t1: Tensor, gate: Optional[Tensor] = None) -> Tensor:
        b, c, h, w = x.shape
        q = self.norm_q(x).flatten(2).transpose(1, 2)          # [B, N, C]  ASL query
        k_src = self.norm_k(t1)
        if self.k_compress is not None:
            k_src = self.k_compress(k_src)
        k = k_src.flatten(2).transpose(1, 2)                   # [B, N, kdim]  T1 key
        v = self.norm_v(x).flatten(2).transpose(1, 2)          # [B, N, C]  ASL value
        out, _ = self.attn(q, k, v, need_weights=False)        # [B, N, C]
        out = out.transpose(1, 2).reshape(b, c, h, w)
        if gate is not None:                                   # Design C: suppress where ASL not reproducible
            out = out * gate                                   # gate [B,1,h,w] in (0,1)
        return x + out


# ---------------------------------------------------------------------------
# Repro-reliability gate (Design C) — "trust T1 here" map from REPRODUCIBLE ASL.
# ---------------------------------------------------------------------------
class ReproReliabilityGate(nn.Module):
    """Per-location gate r∈(0,1) = "how much to trust T1 guidance here", grounded
    in REPRODUCIBLE ASL structure (Design C).

    Inputs are RAW frame half-means (even/odd split) + the (frozen) T1 image — NO
    learnable ASL feature feeds r, so the "make ASL look like T1 to open the gate"
    feedback loop is structurally absent (not merely detached). r only SUPPRESSES
    (∈(0,1)) the already content-safe T1 guidance (LRDA B-gate / AKMR residual),
    so V=ASL and the mismatched-T1 init-invariant are preserved (the guidance it
    scales is zero-init, so r·guidance = 0 at step 0 regardless of r).

    r high  ⇔ ASL has reproducible structure here (both halves agree) → trust T1.
    r low   ⇔ T1 has structure the ASL evidence does not reproducibly support →
              suppress (the hallucination case).
    """

    def __init__(self, hidden: int = 8, use_align: bool = True) -> None:
        super().__init__()
        self.use_align = bool(use_align)
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", kx.transpose(-1, -2).contiguous())
        cin = 3 if self.use_align else 1                       # [E_repro, E_t1, align] or [E_repro]
        self.gate = nn.Sequential(
            nn.Conv2d(cin, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        # zero-init last conv + positive bias ⇒ r ≈ σ(4) ≈ 0.98 everywhere at init
        # (starts ≈ ungated; learns WHERE to distrust T1). Note: init L1=0 does NOT
        # depend on this — the guidance r multiplies is itself zero-init.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 4.0)

    def _grad(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return F.conv2d(x, self.kx, padding=1), F.conv2d(x, self.ky, padding=1)

    @staticmethod
    def _inorm(z: Tensor) -> Tensor:                           # per-image standardise (scale-stable)
        mu = z.mean(dim=(2, 3), keepdim=True)
        sd = z.std(dim=(2, 3), keepdim=True) + 1e-6
        return (z - mu) / sd

    def forward(self, m1: Tensor, m2: Tensor, t1: Tensor) -> Tensor:
        # m1,m2: even/odd frame half-means [B,1,H,W]; t1: (frozen) [B,1,H,W].
        m1, m2, t1 = m1.detach(), m2.detach(), t1.detach()     # data-driven prior, no backbone grad
        if t1.shape[-2:] != m1.shape[-2:]:
            t1 = F.interpolate(t1, size=m1.shape[-2:], mode="bilinear", align_corners=False)
        g1x, g1y = self._grad(m1)
        g2x, g2y = self._grad(m2)
        e_repro = F.relu(g1x * g2x + g1y * g2y)                # ≥0; ~0 on independent noise
        feats = [self._inorm(e_repro)]
        if self.use_align:
            tx, ty = self._grad(t1)
            e_t1 = (tx * tx + ty * ty).sqrt()
            rx, ry = 0.5 * (g1x + g2x), 0.5 * (g1y + g2y)
            align = F.relu((tx * rx + ty * ry) / (e_t1 * (rx * rx + ry * ry).sqrt() + 1e-6))
            feats += [self._inorm(e_t1), align]
        return torch.sigmoid(self.gate(torch.cat(feats, dim=1)))   # [B,1,H,W] ∈ (0,1)


# ---------------------------------------------------------------------------
# RKMR — Reproducibility-Keyed Memory Reader (replaces AKMR's T1 key).
# ---------------------------------------------------------------------------
class ReproDescriptor(nn.Module):
    """Per-location reproducible-structure embedding φ [B,d,H,W] from even/odd
    frame half-means (T1-FREE, detached). Keys the RKMR non-local aggregation:
    two locations with similar reproducible structure attend to each other. Being
    T1-free, it can never leak T1 and is invariant to the mismatched-T1 test."""

    def __init__(self, d: int = 8, hidden: int = 16) -> None:
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", kx.transpose(-1, -2).contiguous())
        self.d = int(d)
        self.embed = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, self.d, kernel_size=1),
        )

    def _grad(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return F.conv2d(x, self.kx, padding=1), F.conv2d(x, self.ky, padding=1)

    @staticmethod
    def _inorm(z: Tensor) -> Tensor:
        mu = z.mean(dim=(2, 3), keepdim=True)
        sd = z.std(dim=(2, 3), keepdim=True) + 1e-6
        return (z - mu) / sd

    def forward(self, m1: Tensor, m2: Tensor) -> Tensor:            # [B,1,H,W] each
        m1, m2 = m1.detach(), m2.detach()                          # data prior, no feedback
        g1x, g1y = self._grad(m1)
        g2x, g2y = self._grad(m2)
        e = F.relu(g1x * g2x + g1y * g2y)                          # reproducible edge energy ≥0
        return self.embed(torch.cat([0.5 * (m1 + m2), self._inorm(e)], dim=1))   # [B,d,H,W]


class RKMR(nn.Module):
    """Reproducibility-Keyed Memory Reader. Q,K from the reproducible descriptor φ
    (T1-free), V from ASL. ``out = x + Wo(softmax(QKᵀ/√dh)·V)`` with ``Wo`` zero-init
    ⇒ 0 contribution at step 0. Single similarity (reproducible structure), no T1 in
    the key ⇒ V=ASL trivially, and (without the veto) contributes exactly 0 to the
    mismatched-T1 L1 always. Optional low-rank T1 tissue-consistency VETO (ablation):
    a bounded multiplicative gate that discourages aggregating across tissue
    boundaries — reintroduces (bounded) T1, so re-verify the mismatch gate."""

    def __init__(self, ch: int, desc_dim: int, n_heads: int = 4,
                 t1_veto: bool = False, t1_ch: Optional[int] = None,
                 veto_rank_div: int = 4) -> None:
        super().__init__()
        heads = max(1, int(n_heads))
        while ch % heads != 0 and heads > 1:
            heads //= 2
        self.heads = heads
        self.ch = int(ch)
        self.dh = ch // heads
        self.q = nn.Conv2d(desc_dim, ch, kernel_size=1)
        self.k = nn.Conv2d(desc_dim, ch, kernel_size=1)
        self.norm_v = nn.GroupNorm(min(8, ch), ch)
        self.v = nn.Conv2d(ch, ch, kernel_size=1)
        self.o = nn.Conv2d(ch, ch, kernel_size=1)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)                                # out zero-init → init contribution 0
        self.t1_veto = bool(t1_veto)
        if self.t1_veto:
            assert t1_ch is not None, "RKMR t1_veto needs t1_ch"
            kdim = max(1, int(t1_ch) // max(1, int(veto_rank_div)))
            self.t1_proj = nn.Conv2d(int(t1_ch), kdim, kernel_size=1, bias=False)   # low-rank T1
            self.veto_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, phi: Optional[Tensor], t1: Optional[Tensor] = None) -> Tensor:
        if phi is None:                                            # no descriptor → passthrough
            return x
        B, C, H, W = x.shape
        N = H * W
        if phi.shape[-2:] != (H, W):
            phi = F.adaptive_avg_pool2d(phi, (H, W))

        def split(t: Tensor) -> Tensor:                           # [B,N,C] → [B,heads,N,dh]
            return t.view(B, N, self.heads, self.dh).permute(0, 2, 1, 3)

        q = split(self.q(phi).flatten(2).transpose(1, 2))
        k = split(self.k(phi).flatten(2).transpose(1, 2))
        v = split(self.v(self.norm_v(x)).flatten(2).transpose(1, 2))
        logits = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)      # [B,heads,N,N]
        A = torch.softmax(logits, dim=-1)
        if self.t1_veto and t1 is not None:
            if t1.shape[-2:] != (H, W):
                t1 = F.adaptive_avg_pool2d(t1, (H, W))
            tp = self.t1_proj(t1).flatten(2).transpose(1, 2)      # [B,N,kdim]  low-rank T1
            sim = (tp @ tp.transpose(-2, -1)) / (tp.shape[-1] ** 0.5)   # [B,N,N] tissue similarity
            s = torch.sigmoid(self.veto_scale * sim).unsqueeze(1)      # [B,1,N,N] ∈(0,1) veto
            A = A * s
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)   # renorm
        out = (A @ v).permute(0, 2, 1, 3).reshape(B, N, C).transpose(1, 2).reshape(B, C, H, W)
        return x + self.o(out)


# ---------------------------------------------------------------------------
# CouplingGate — content-safe T1–ASL agreement gate (no frame splitting).
# ---------------------------------------------------------------------------
class CouplingGate(nn.Module):
    """Per-location gate ``c ∈ (0,1)`` = "how much to trust the T1 modulation
    here", from an EXPLICIT T1–ASL agreement measure.

    Inputs are the (frozen) T1 image and a DETACHED ASL aggregate (the
    reliability-weighted mean of ALL frames — no even/odd split, so no SNR is
    wasted). Both are fixed per sample: T1 is frozen, the aggregate is detached
    ⇒ NO learnable ASL feature feeds ``c``, so the "push ASL toward T1 to open
    the gate" feedback loop is structurally absent (not merely a detach on a
    value path). ``c`` only SCALES the already zero-init, multiplicative,
    V=ASL T1 modulation (LRDA B-gate / Δ-gate), so:
      * the mismatched-T1 init invariant (L1=0) holds regardless of ``c`` (the
        modulation it scales is itself zero at step 0);
      * a WRONG T1 collapses the agreement ⇒ ``c → small`` ⇒ the modulation
        self-suppresses at run time (behavioural anti-hallucination — stronger
        than the init-only invariant).

    "Heavy" agreement (default): a learned local bilinear (cosine of learned
    per-pixel embeddings) between a T1 embedding and an ASL-aggregate embedding,
    plus an interpretable gradient-cosine channel. The gate head is zero-init +
    positive bias ⇒ ``c ≈ σ(4) ≈ 0.98`` everywhere at init (starts ≈ ungated;
    learns WHERE to distrust T1).
    """

    def __init__(self, asl_ch: int = 1, t1_ch: int = 1, embed: int = 16,
                 hidden: int = 16) -> None:
        super().__init__()
        self.t1_embed = nn.Sequential(
            nn.Conv2d(t1_ch, embed, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(embed, embed, kernel_size=3, padding=1),
        )
        self.asl_embed = nn.Sequential(
            nn.Conv2d(asl_ch, embed, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(embed, embed, kernel_size=3, padding=1),
        )
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", kx.transpose(-1, -2).contiguous())
        # gate head over [bilinear-cosine, gradient-cosine] → c∈(0,1)
        self.gate = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 4.0)     # c ≈ σ(4) ≈ 0.98 at init

    def _grad(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.shape[1] > 1:                             # reduce multi-channel to mean map
            x = x.mean(dim=1, keepdim=True)
        return F.conv2d(x, self.kx, padding=1), F.conv2d(x, self.ky, padding=1)

    @staticmethod
    def _inorm(z: Tensor) -> Tensor:                   # per-image standardise (scale-stable)
        mu = z.mean(dim=(2, 3), keepdim=True)
        sd = z.std(dim=(2, 3), keepdim=True) + 1e-6
        return (z - mu) / sd

    def forward(self, t1: Tensor, a: Tensor) -> Tensor:
        # t1: (frozen) [B,t1ch,H,W];  a: ASL aggregate [B,aslch,H,W]
        a = a.detach()                                 # content-safe: no grad to the aggregator
        if t1.shape[-2:] != a.shape[-2:]:
            t1 = F.interpolate(t1, size=a.shape[-2:], mode="bilinear", align_corners=False)
        et = self.t1_embed(t1)                         # [B,embed,H,W]  learned T1 descriptor
        ea = self.asl_embed(a)                         # [B,embed,H,W]  learned ASL descriptor
        sim = (F.normalize(et, dim=1, eps=1e-6)
               * F.normalize(ea, dim=1, eps=1e-6)).sum(1, keepdim=True)   # cosine ∈[-1,1]
        tx, ty = self._grad(t1)
        ax, ay = self._grad(a)
        gc = (tx * ax + ty * ay) / (
            (tx * tx + ty * ty).sqrt() * (ax * ax + ay * ay).sqrt() + 1e-6
        )                                              # gradient cosine ∈[-1,1]
        c = torch.sigmoid(self.gate(torch.cat([self._inorm(sim), self._inorm(gc)], dim=1)))
        return c                                       # [B,1,H,W] ∈ (0,1), ≈0.98 at init


# ---------------------------------------------------------------------------
# Content-guided conv encoder — same forward contract as MoSSMEncoder2D.
# ---------------------------------------------------------------------------
class ContentGuidedConvEncoder2D(nn.Module):
    """PlainUNet2D's ConvEncoder2D + per-stage content-guard (LRDA-Conv + AKMR).

    forward(x, t1_skips, tissue_seg=None, t1_seg_probs=None) -> (feat, None, skips)
    matches MoSSMEncoder2D exactly, so ASLT1Denoiser can swap encoders by flag.
    ``tissue_seg`` / ``t1_seg_probs`` are accepted for signature compatibility and
    ignored (no TABS scan in the conv backbone).
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        input_hw: int = 128,
        t1_channels_per_scale: Optional[List[int]] = None,
        lrda_rank: int = 8,
        lrda_bound: float = 4.0,
        lrda_variant: str = "c",
        lrda_stages: Optional[List[int]] = None,
        akmr_enabled: bool = True,
        akmr_max_tokens: int = 1024,
        akmr_heads: int = 4,
        akmr_k_rank_div: int = 4,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        channels = [base_ch * (2 ** i) for i in range(depth)]
        t1ch = list(t1_channels_per_scale) if t1_channels_per_scale is not None else list(channels)
        self.out_ch = channels[-1]

        # Backbone (identical to ConvEncoder2D): stem stride-1 + (depth-1) DownBlocks.
        self.stem = ResidualBlock(in_ch, channels[0], stride=1, groups=groups)
        self.downs = nn.ModuleList(
            [DownBlock(channels[i - 1], channels[i], groups=groups) for i in range(1, depth)]
        )

        # LRDA-Conv guard — all stages by default (like --cada_stages 0,1,2,3).
        lrda_stages = list(range(depth)) if lrda_stages is None else list(lrda_stages)
        self.lrda_on = [s in lrda_stages for s in range(depth)]
        self.lrda = nn.ModuleList([
            LRDAConv2d(channels[s], t1ch[s], rank=lrda_rank, bound=lrda_bound, variant=lrda_variant)
            if self.lrda_on[s] else nn.Identity()
            for s in range(depth)
        ])

        # AKMR guard — low-resolution stages only (token budget).
        hw_per_scale = [max(1, int(input_hw) // (2 ** s)) for s in range(depth)]
        self.akmr_on = [
            bool(akmr_enabled) and (hw_per_scale[s] * hw_per_scale[s] <= int(akmr_max_tokens))
            for s in range(depth)
        ]
        self.akmr = nn.ModuleList([
            AKMRConv2d(channels[s], t1ch[s], n_heads=akmr_heads, k_rank_div=akmr_k_rank_div)
            if self.akmr_on[s] else nn.Identity()
            for s in range(depth)
        ])

    def _guard(self, s: int, feat: Tensor, t1: Tensor,
               gate: Optional[Tensor] = None) -> Tensor:
        # coupling gate (T1–ASL agreement) pooled to this stage's resolution once,
        # then shared by LRDA (scales the B/Δ modulation) and AKMR (scales the read).
        if gate is not None and gate.shape[-2:] != feat.shape[-2:]:
            gate = F.adaptive_avg_pool2d(gate, feat.shape[-2:])
        if self.lrda_on[s]:
            feat = self.lrda[s](feat, t1, gate=gate)
        if self.akmr_on[s]:
            feat = self.akmr[s](feat, t1, gate=gate)
        return feat

    def forward(
        self,
        x: Tensor,
        t1_skips: List[Tensor],
        tissue_seg: Optional[Tensor] = None,
        t1_seg_probs: Optional[Tensor] = None,
        coupling_gate: Optional[Tensor] = None,
    ) -> Tuple[Tensor, None, List[Tensor]]:
        assert t1_skips is not None and len(t1_skips) >= self.depth, \
            "ContentGuidedConvEncoder2D requires depth T1 skips"
        skips: List[Tensor] = []
        feat = self.stem(x)                                    # stage 0
        assert feat.shape[-2:] == t1_skips[0].shape[-2:], \
            f"stage0 ASL {tuple(feat.shape[-2:])} vs T1 {tuple(t1_skips[0].shape[-2:])}"
        feat = self._guard(0, feat, t1_skips[0], gate=coupling_gate)
        skips.append(feat)
        for i, blk in enumerate(self.downs, start=1):
            feat = blk(feat)                                   # stage i
            assert feat.shape[-2:] == t1_skips[i].shape[-2:], \
                f"stage{i} ASL {tuple(feat.shape[-2:])} vs T1 {tuple(t1_skips[i].shape[-2:])}"
            feat = self._guard(i, feat, t1_skips[i], gate=coupling_gate)
            skips.append(feat)
        return feat, None, skips
