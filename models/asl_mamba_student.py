# -*- coding: utf-8 -*-
"""ASL-Mamba: T1-free distilled student for v40.

Background
----------
Selective State Space Model (SSM) student for multi-frame ASL denoising,
inspired by:
  - Gu & Dao 2023, Mamba: Linear-Time Sequence Modeling with Selective State Spaces
  - Liu et al. NeurIPS 2024, VMamba: Visual State Space Model
  - Guo et al. ECCV 2024, MambaIR: Image Restoration with State-Space Model

Distilled from a frozen v39 ASLT1Denoiser teacher (which uses set_a + T1 +
cross-modal fusion). The student takes ONLY ASL frames, aggregated via SVFW
(safe-by-design, T1-free), and processes the resulting feature map through a
hierarchical SS2D backbone with no T1 input modality. Architecturally
hallucination-impossible.

Implementation notes
--------------------
This module implements a SIMPLIFIED selective-scan SSM in pure PyTorch
(no mamba-ssm/CUDA dependency). The reference Mamba CUDA kernel achieves
linear complexity in HW; our pure-PyTorch fallback uses a sequential scan
in Python (slower at training time, asymptotically O(L*D*N) per direction).
Inference can be optimized later by swapping SS2D with mamba_ssm.SelectiveScan.

For research-prototype purposes the simplified version is sufficient: it
captures the architectural idea (4-direction 2D scan, gated state-space
blocks, U-Net hierarchy) and reproducibly trains on Windows without
CUDA-kernel build dependencies.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.blocks import SpatialVaryingFrameWeighting


def _icnr_init_(tensor: Tensor, upscale_factor: int,
                init=nn.init.kaiming_normal_) -> None:
    """ICNR (Aitken et al. 2017) init for the conv that feeds nn.PixelShuffle.

    Initialises a [out_ch * r², in_ch, k, k] kernel by drawing a single
    [out_ch, in_ch, k, k] sub-kernel and replicating it r² times along
    dim 0. PixelShuffle then produces an upsampled image equivalent to
    nearest-neighbour upsample at init — eliminating the checkerboard
    sub-pixel bias that vanilla random init exhibits.
    """
    out_ch_total, in_ch, kh, kw = tensor.shape
    r2 = upscale_factor * upscale_factor
    assert out_ch_total % r2 == 0, (out_ch_total, r2)
    sub = torch.zeros(out_ch_total // r2, in_ch, kh, kw)
    init(sub)
    sub = sub.repeat_interleave(r2, dim=0)
    with torch.no_grad():
        tensor.copy_(sub)


# -----------------------------------------------------------------------------
# Selective Scan dispatch: official mamba_ssm CUDA kernel when available,
# pure-PyTorch fallback otherwise.
# -----------------------------------------------------------------------------

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_ssm_cuda_fn
    _HAS_MAMBA_SSM = True
except Exception:
    _mamba_ssm_cuda_fn = None
    _HAS_MAMBA_SSM = False


def selective_scan_pytorch(
    u: Tensor,          # [B, L, D]
    delta: Tensor,      # [B, L, D]  (already softplus-activated)
    A: Tensor,          # [D, N]   (negative-real-part log-parametrised)
    B_: Tensor,         # [B, L, N]
    C_: Tensor,         # [B, L, N]
    D_skip: Tensor,     # [D]
) -> Tensor:
    """Selective scan. Dispatches to official mamba_ssm CUDA kernel when
    available (50-100× faster on long sequences), otherwise falls back to a
    sequential Python loop reference. Returns [B, L, D].
    """
    if _HAS_MAMBA_SSM and u.is_cuda:
        # mamba_ssm.selective_scan_fn expects [B, D, L] for u/delta and
        # [B, N, L] for B/C; delta_softplus=False since our delta is already
        # softplus-activated upstream.
        #
        # AMP safety: the CUDA kernel requires u/delta/B/C to share one dtype
        # (it asserts delta.scalar_type() == u.scalar_type()). Under autocast,
        # u comes from the (bf16) conv path while delta is EC-LRDA-modulated via
        # exp[gamma*alpha*tanh(m)] over the frozen T1/PV path and can stay fp32,
        # tripping "Expected delta.scalar_type() == input_type". Run the scan in
        # fp32 with autocast disabled (the SSM recurrence is the numerically
        # sensitive part; mamba_ssm accumulates state in fp32 anyway) and cast
        # the output back so the surrounding (possibly bf16) residual add matches.
        # No-op when AMP is off (inputs already fp32).
        orig_dtype = u.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            u_t     = u.transpose(1, 2).contiguous().float()     # [B, D, L]
            delta_t = delta.transpose(1, 2).contiguous().float() # [B, D, L]
            B_t     = B_.transpose(1, 2).contiguous().float()    # [B, N, L]
            C_t     = C_.transpose(1, 2).contiguous().float()    # [B, N, L]
            y = _mamba_ssm_cuda_fn(
                u_t, delta_t, A.float(), B_t, C_t,
                D=D_skip.float(), z=None, delta_bias=None, delta_softplus=False,
            )
            y = y.transpose(1, 2).contiguous()       # [B, L, D]
        return y.to(orig_dtype)

    # Pure-PyTorch fallback (slow; CPU or no mamba_ssm)
    Bsz, L, D = u.shape
    N = A.shape[1]
    # discretise:  h_{t+1} = exp(delta * A) h_t + (delta * B) u_t
    deltaA = torch.exp(delta.unsqueeze(-1) * A)              # [B, L, D, N]
    deltaB_u = delta.unsqueeze(-1) * B_.unsqueeze(2) * u.unsqueeze(-1)
    # sequential scan in time (Python loop)
    h = u.new_zeros(Bsz, D, N)
    ys = []
    for t in range(L):
        h = deltaA[:, t] * h + deltaB_u[:, t]                # [B, D, N]
        y = (h * C_[:, t].unsqueeze(1)).sum(-1)              # [B, D]
        ys.append(y)
    y = torch.stack(ys, dim=1)
    return y + D_skip * u


# -----------------------------------------------------------------------------
# 2D Selective State Space block (4-direction)
# -----------------------------------------------------------------------------

class SS2D(nn.Module):
    """2D selective scan: 4 directions (→ ← ↓ ↑) merged by mean.

    Modeled after VMamba's VSSBlock minus the CUDA kernel. Each direction
    serializes the [H,W] grid in a different order then runs the selective
    scan along that ordering.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3,
                 expand: int = 2, n_directions: int = 1,
                 lrda_inscan: bool = False, lrda_t1_ch: Optional[int] = None,
                 lrda_rank: int = 8, lrda_bound: float = 4.0,
                 lrda_cond_asl: bool = False, lrda_dt_bound: float = 1.0,
                 lrda_bilinear: bool = False, lrda_pv_ch: int = 4,
                 lrda_pv_geo: bool = False, hv_scan: bool = False,
                 sig_xin: bool = False, dt_rank: int = 1, tfdm: bool = False,
                 scan_dropout_p: float = 0.0):
        super().__init__()
        assert n_directions in (1, 2, 4), f"n_directions must be 1/2/4, got {n_directions}"
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.sig_xin = bool(sig_xin)   # (2) a_B reads x_in (d_inner) instead of B_ (d_state)
        self.tfdm = bool(tfdm)         # Tissue-Factored Dynamics Modulation (standalone cond. structure)
        self.scan_dropout_p = float(scan_dropout_p)   # batch-3: scan-direction dropout (train + MC-uncertainty)
        self._mc_scan_dropout = False                 # eval toggle: force dropout at inference for MC passes
        # (3) dt_rank: input-dependent Δ projection rank. 1 = legacy scalar (over-simplified,
        # starves Δ selectivity); 0 = auto = ceil(d_model/16) (standard Mamba). Feeds the Δ lever.
        self._dt_rank = (max(1, (d_model + 15) // 16) if int(dt_rank) == 0 else int(dt_rank))
        self.d_state = d_state
        self.n_directions = n_directions
        # Which of the 4 serialization orders (0:→ 1:← 2:↓ 3:↑) the n scans use.
        # Default n=2 is horizontal-bidirectional {→,←}; hv_scan reinterprets n=2
        # as {→,↓} (one horizontal + one vertical, each forward) for isotropic 2D
        # coverage. n=1/4 unaffected. Parameters stay position-indexed (0..n-1);
        # the serialization order is the semantic direction here.
        if bool(hv_scan) and n_directions == 2:
            self._scan_dirs = (0, 2)              # → , ↓
        else:
            self._scan_dirs = tuple(range(n_directions))
        # input projection: x and gate
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # depthwise conv on the SSM input
        self.dwconv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv // 2, groups=self.d_inner)
        # selective parameters (input-dependent B, C, dt) for each direction.
        # dt slice width = self._dt_rank (3); B_/C_ are d_state each.
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, d_state * 2 + self._dt_rank, bias=False) for _ in range(n_directions)
        ])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self._dt_rank, self.d_inner, bias=True) for _ in range(n_directions)
        ])
        # static A (diagonal, parameterised by log -A so A < 0 is enforced by exp)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.ParameterList([
            nn.Parameter(torch.log(A), requires_grad=True) for _ in range(n_directions)
        ])
        # static D skip
        self.D_skip = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner)) for _ in range(n_directions)
        ])
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

        # In-scan LRDA (CIG-VSS): T1-conditioned zero-init low-rank modulation of the
        # SSM dynamics — unified identity-centred log-scale adapter. z = RMSNorm_r(φ),
        # factor = exp(α ⊙ tanh(U z)) with α per-channel. Both levers content-free (T1
        # never enters the value stream u=seq or the read matrix C ⇒ V=ASL by
        # construction; verify with the mismatched-T1 L1=0 gate):
        #   (1) B-gate: B' = B · exp(α_B ⊙ tanh(U_B z)) — how much ASL content enters
        #       the state, per state-channel, at fixed timescale.
        #   (2) Δ log-space: Δ' = Δ · exp(α_Δ ⊙ tanh(U_Δ z)) — the scan TIMESCALE/decay
        #       (Ā=exp(ΔA)); a lever distinct from the B-gate.
        # exp(·)>0 keeps EVERY factor positive — no sign flip (the old B·(1+U_B(s)) had
        # an UNBOUNDED U → could go negative / blow up; only Δ had a tanh). tanh bounds
        # the exponent and learnable per-channel α (0-init) sets its scale, replacing
        # the hand-set lrda_bound / lrda_dt_bound. α=0 ⇒ factor=1 ⇒ plain VSS at init.
        self.lrda_inscan = bool(lrda_inscan)
        self.lrda_cond_asl = bool(lrda_cond_asl)
        self.lrda_pv_geo = bool(lrda_pv_geo)          # Level-1: PV-only geometry encoder
        self.lrda_bilinear = bool(lrda_bilinear) or self.lrda_pv_geo   # pv_geo is a bilinear sub-mode
        self.lrda_pv_ch = int(lrda_pv_ch)
        if self.lrda_inscan:
            assert lrda_t1_ch is not None, "in-scan LRDA needs lrda_t1_ch"
            self.lrda_rank = max(1, int(lrda_rank))
            self.lrda_rms_eps = 1e-5
            self.lrda_U = nn.Linear(self.lrda_rank, d_state, bias=False)          # B direction
            self.lrda_alpha_B = nn.Parameter(torch.zeros(d_state))                # per-channel gain, 0-init
            self.lrda_U_dt = nn.Linear(self.lrda_rank, self.d_inner, bias=False)  # Δ direction
            self.lrda_alpha_dt = nn.Parameter(torch.zeros(self.d_inner))          # per-channel gain, 0-init
            if self.tfdm:
                # Tissue-Factored Dynamics Modulation (TF-DM): K per-tissue operators W_k
                # map the ASL token feature x_in -> a B/Δ modulation; the raw PV fractions
                # p_k (tissue composition) ROUTE/mix them. m = Σ_k p_k · (W_k · x_in). The
                # "rank" is K tissues (interpretable, not an arbitrary bottleneck); W_GM/W_WM/
                # W_CSF are visualisable per-tissue ASL→dynamics maps. V=ASL: W_k read ASL only,
                # p_k just route, result multiplies B/Δ; α=0 ⇒ warm-start identity; p=0 (pv0) ⇒
                # m=0 ⇒ EXACT identity. Replaces the E_B/geo/V/U bilinear (no lrda_U used).
                K = self.lrda_pv_ch
                self.tfdm_K = K
                self.tfdm_W_B  = nn.Linear(self.d_inner, K * d_state, bias=False)      # x_in → per-tissue B-mod
                self.tfdm_W_dt = nn.Linear(self.d_inner, K * self.d_inner, bias=False)  # x_in → per-tissue Δ-mod
            elif self.lrda_bilinear:
                # EC-LRDA (soft-PV bilinear): the low-rank modulation direction is a
                # MULTIPLICATIVE interaction between a PV-derived tissue coefficient
                # s (WHICH tissue-dynamics basis T1 proposes) and an ASL-content
                # coefficient a (HOW MUCH content the ASL base dynamics carry on that
                # basis). Only components BOTH activate can modify the state dynamics.
                #   s_B = E_B·p + Ē_B·p̄  (PV composition + neighbourhood context)
                #   a_B = V_B·LN(B_ASL)   (ASL content, per token)
                #   m_B = U_B(s_B ⊙ a_B); B' = B·exp(γ·α_B⊙tanh(m_B))
                # α zero-init ⇒ identity at init; s from PV, a from B_ASL ⇒ T1/PV only
                # ever multiplies B/Δ (u, C, D untouched) ⇒ V=ASL preserved.
                pv, r = self.lrda_pv_ch, self.lrda_rank
                self.lrda_E_B     = nn.Conv2d(pv, r, 1, bias=False)  # PV → s_B (composition)
                self.lrda_E_dt    = nn.Conv2d(pv, r, 1, bias=False)
                if self.lrda_pv_geo:
                    # Level-1 (2026-07-31): drop the neighbourhood channel p̄ (diag_pv_pyramid:
                    # corr(p,p̄)≈0.99–0.9996 ⇒ redundant) and instead let s see LOCAL PV
                    # geometry via a ZERO-INIT depthwise-conv residual (s = E(p) + geo(E(p))).
                    # PV-only (no raw T1) ⇒ V=ASL / isolation unchanged — re-check mismatch-L1.
                    self.lrda_geo_B  = self._pv_geo_block(r)
                    self.lrda_geo_dt = self._pv_geo_block(r)
                else:
                    self.lrda_Ebar_B  = nn.Conv2d(pv, r, 1, bias=False)  # PV neighbourhood → s_B
                    self.lrda_Ebar_dt = nn.Conv2d(pv, r, 1, bias=False)
                # a_B signal source: (2) --lrda_sig_xin reads the d_inner token feature x_in
                # (richer, all-ASL ⇒ no extra leakage; removes the read-B-to-modulate-B self-ref);
                # default reads the thin d_state B_ matrix.
                _vb_in = self.d_inner if bool(sig_xin) else d_state
                self.lrda_V_B     = nn.Linear(_vb_in, r, bias=False)  # ASL content → a_B (x_in if sig_xin else B_)
                self.lrda_V_dt    = nn.Linear(self.d_inner, r, bias=False)  # logΔ_ASL → a_Δ
                self.lrda_ln_B    = nn.LayerNorm(_vb_in)
                self.lrda_ln_dt   = nn.LayerNorm(self.d_inner)
            else:
                # Design A (--lrda_cond_asl): φ also sees the ASL scan input so T1
                # modulates B only where COMPATIBLE with ASL. RMSNorm + learnable α
                # replace lrda_bound/lrda_dt_bound. α ZERO-init ⇒ V=ASL at init.
                phi_in_ch = int(lrda_t1_ch) + (self.d_inner if self.lrda_cond_asl else 0)
                self.lrda_phi = nn.Conv2d(phi_in_ch, self.lrda_rank, kernel_size=1)

    @staticmethod
    def _pv_geo_block(r: int) -> nn.Module:
        """PV-only local-geometry residual: 1×1 → SiLU → depthwise 3×3 → 1×1.
        NORMALLY initialized (2026-08-20): the WARM-START identity + V=ASL is guaranteed
        SOLELY by the zero-init modulation gain α (lrda_alpha_B/dt); at init gb=tanh(m)·α=0
        ⇒ B'=B regardless of geo. The old zero-init tail here was a superseded pv_geo-
        comparability trick that starved α's gradient early (m≈0 ⇒ tanh(m)≈0 ⇒ α stuck):
        with geo open from step 1, m≠0 ⇒ α gets an O(1) gradient and the modulation
        actually learns. All convs bias-free ⇒ geo(0)=0 ⇒ pv0 (pv=0) stays exact identity.
        Depthwise 3×3 gives the local PV geometry a 1×1 cannot."""
        return nn.Sequential(
            nn.Conv2d(r, r, 1, bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(r, r, 3, padding=1, groups=r, bias=False),
            nn.Conv2d(r, r, 1, bias=False),
        )

    def _serialize(self, x: Tensor, direction: int) -> Tensor:
        """Reorder [B, C, H, W] into [B, L=H*W, C] under one of 4 scan orders."""
        B, C, H, W = x.shape
        if direction == 0:    # row-major →
            seq = x.flatten(2).transpose(1, 2)
        elif direction == 1:  # row-major reversed ←
            seq = x.flatten(2).transpose(1, 2).flip(1)
        elif direction == 2:  # column-major ↓
            seq = x.transpose(2, 3).flatten(2).transpose(1, 2)
        else:                 # column-major reversed ↑
            seq = x.transpose(2, 3).flatten(2).transpose(1, 2).flip(1)
        return seq

    def _deserialize(self, seq: Tensor, B: int, C: int, H: int, W: int,
                     direction: int) -> Tensor:
        if direction == 1 or direction == 3:
            seq = seq.flip(1)
        if direction <= 1:    # row-major
            return seq.transpose(1, 2).reshape(B, C, H, W)
        # column-major
        return seq.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)

    def forward(self, x: Tensor, t1: Optional[Tensor] = None,
                repro_gate: Optional[Tensor] = None) -> Tensor:  # x:[B,H,W,C], t1:[B,t1ch,H,W], repro_gate:[B,1,H,W]
        B, H, W, C = x.shape
        xz = self.in_proj(x)                                # [B,H,W, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)
        # depthwise conv expects [B, C, H, W]
        x_in = x_in.permute(0, 3, 1, 2).contiguous()
        x_in = self.act(self.dwconv(x_in))
        # In-scan LRDA gate coefficients from T1/PV (computed once, serialized per
        # direction to align with each direction's token order). α is zero-init so
        # the gate is ×1 at step 0 (== plain VSS).
        lrda_active = bool(self.lrda_inscan and t1 is not None)
        z_map = None          # phi path (raw-T1-feature conditioning)
        sB_map = sdt_map = None   # bilinear path (soft-PV composition conditioning)
        pvr_map = None        # TF-DM: raw PV tissue fractions [B, K, H, W] used as routing p_k
        if lrda_active:
            if self.tfdm:
                pvr_map = t1[:, :self.lrda_pv_ch]                            # [B, K, H, W] raw PV routing
            elif self.lrda_bilinear:
                if self.lrda_pv_geo:
                    # t1 carries only p (pv_ch); s = E(p) + geo residual (geo normally-init
                    # 2026-08-20; warm-start identity is guaranteed by α=0, not geo).
                    p = t1[:, :self.lrda_pv_ch]
                    eB, edt = self.lrda_E_B(p), self.lrda_E_dt(p)
                    sB_map = eB + self.lrda_geo_B(eB)                        # [B, r, H, W]
                    sdt_map = edt + self.lrda_geo_dt(edt)                    # [B, r, H, W]
                else:
                    # t1 carries [p ; p̄] (soft-PV composition + neighbourhood), 2*pv_ch.
                    p, p_bar = t1[:, :self.lrda_pv_ch], t1[:, self.lrda_pv_ch:2 * self.lrda_pv_ch]
                    sB_map = self.lrda_E_B(p) + self.lrda_Ebar_B(p_bar)      # [B, r, H, W]
                    sdt_map = self.lrda_E_dt(p) + self.lrda_Ebar_dt(p_bar)   # [B, r, H, W]
            else:
                # Design A: φ sees [T1 ; ASL] so the gate depends on their compatibility.
                phi_in = torch.cat([t1, x_in], dim=1) if self.lrda_cond_asl else t1
                phi = self.lrda_phi(phi_in)                                   # [B, r, H, W]
                z_map = phi * torch.rsqrt(phi.pow(2).mean(dim=1, keepdim=True) + self.lrda_rms_eps)
        # Per-location reliability/compatibility gate γ∈(0,1) (EC: r_rep·c_sem; or the
        # legacy repro gate). Suppresses the T1/PV modulation where ASL evidence is
        # unreliable or PV is not ASL-supported.
        rg = None
        if lrda_active and repro_gate is not None:
            rg = repro_gate if repro_gate.shape[-2:] == (H, W) else \
                F.adaptive_avg_pool2d(repro_gate, (H, W))     # [B,1,H,W]
        # 1/2/4-direction scans (speed/quality trade-off)
        # Scan-direction dropout (batch-3): drop each of the n directions with prob
        # scan_dropout_p (keep ≥1), averaging only the KEPT ones. Active in TRAIN (a mild
        # structured regulariser) and, when toggled, at inference for MC-uncertainty
        # (N stochastic passes → per-voxel std). Acts on the ASL scan only ⇒ V=ASL.
        _dirs = list(enumerate(self._scan_dirs))
        if self.scan_dropout_p > 0.0 and len(_dirs) > 1 and (self.training or self._mc_scan_dropout):
            _keep = [pd for pd in _dirs if float(torch.rand(())) >= self.scan_dropout_p]
            _dirs = _keep if _keep else [_dirs[int(torch.randint(len(_dirs), ()))]]
        out_acc = 0.0
        for p_idx, d_idx in _dirs:
            # p_idx indexes the per-scan parameters (0..n-1); d_idx is the
            # semantic serialization order (0:→ 1:← 2:↓ 3:↑).
            seq = self._serialize(x_in, d_idx)              # [B, L, d_inner]
            # selective parameters per token
            xb = self.x_proj[p_idx](seq)                    # [B, L, 2*N+1]
            dt_raw, B_, C_ = xb.split([self._dt_rank, self.d_state, self.d_state], dim=-1)
            if lrda_active:
                # B' = B · exp(α_B ⊙ tanh(m_B)); α 0-init ⇒ ×1 at init, exp>0 (no sign
                # flip). content (seq) and C_ untouched → V=ASL.
                if self.tfdm:
                    # TF-DM: m_B = Σ_k p_k · (W_B^k · x_in). p from raw PV, W from ASL (seq).
                    p_seq = self._serialize(pvr_map, d_idx)                  # [B, L, K]
                    wxB = self.tfdm_W_B(seq).view(seq.shape[0], seq.shape[1], self.tfdm_K, self.d_state)
                    m_B = torch.einsum('blk,blkd->bld', p_seq, wxB)          # [B, L, d_state]
                elif self.lrda_bilinear:
                    _asrc = seq if self.sig_xin else B_                      # (2) x_in (d_inner) vs B_ (d_state)
                    a_B = self.lrda_V_B(self.lrda_ln_B(_asrc))              # [B, L, r] ASL content coeff
                    m_B = self.lrda_U(self._serialize(sB_map, d_idx) * a_B)  # s⊙a → d_state
                else:
                    m_B = self.lrda_U(self._serialize(z_map, d_idx))
                gb = torch.tanh(m_B) * self.lrda_alpha_B
                if rg is not None:
                    gb = gb * self._serialize(rg, d_idx)             # [B, L, 1] broadcast → suppress
                B_ = B_ * torch.exp(gb)
            delta = F.softplus(self.dt_proj[p_idx](dt_raw)) # [B, L, d_inner]
            if lrda_active:
                # Δ' = Δ · exp(α_Δ ⊙ tanh(m_Δ)); rate-only (u/C untouched) → V=ASL.
                if self.tfdm:
                    # p_seq reused from the B-leg (same tokens); W_dt maps x_in → per-tissue Δ-mod.
                    wxdt = self.tfdm_W_dt(seq).view(seq.shape[0], seq.shape[1], self.tfdm_K, self.d_inner)
                    m_dt = torch.einsum('blk,blkd->bld', p_seq, wxdt)        # [B, L, d_inner]
                elif self.lrda_bilinear:
                    a_dt = self.lrda_V_dt(self.lrda_ln_dt(torch.log(delta + 1e-6)))
                    m_dt = self.lrda_U_dt(self._serialize(sdt_map, d_idx) * a_dt)
                else:
                    m_dt = self.lrda_U_dt(self._serialize(z_map, d_idx))
                gd = torch.tanh(m_dt) * self.lrda_alpha_dt
                if rg is not None:
                    gd = gd * self._serialize(rg, d_idx)             # suppress where γ low
                delta = delta * torch.exp(gd)
            A = -torch.exp(self.A_log[p_idx])               # [d_inner, N]
            y = selective_scan_pytorch(
                u=seq, delta=delta, A=A, B_=B_, C_=C_, D_skip=self.D_skip[p_idx]
            )
            out = self._deserialize(y, B, self.d_inner, H, W, d_idx)
            out_acc = out_acc + out
        out_acc = out_acc / float(len(_dirs))          # average over the KEPT directions (scan-dropout)
        # gate (silu(z) is the gate value)
        out_acc = out_acc.permute(0, 2, 3, 1).contiguous()  # [B, H, W, d_inner]
        out_acc = out_acc * self.act(z)
        return self.out_proj(out_acc)                       # [B, H, W, d_model]


class ConvFFN(nn.Module):
    """FFN with a depthwise 3×3 conv between the two linears (PVTv2/LocalViT):
    injects LOCAL spatial inductive bias into the otherwise 1×1 channel-mixing
    stage — a single-branch alternative to a parallel conv branch. NHWC in/out."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:                 # [B, H, W, C]
        x = self.fc1(x)                                     # [B,H,W,hidden]
        x = x.permute(0, 3, 1, 2).contiguous()              # NCHW
        x = self.act(self.dw(x))
        x = x.permute(0, 2, 3, 1).contiguous()              # NHWC
        return self.fc2(x)


class VSSBlock(nn.Module):
    """Pre-norm SS2D block + FFN. Optional single-branch conv bias (ConvFFN) and
    in-scan LRDA (T1-conditioned B-gate inside SS2D)."""

    def __init__(self, d_model: int, d_state: int = 16, mlp_ratio: float = 2.0,
                 drop_path: float = 0.0, n_directions: int = 1,
                 use_conv_ffn: bool = False,
                 lrda_inscan: bool = False, lrda_t1_ch: Optional[int] = None,
                 lrda_rank: int = 8, lrda_bound: float = 4.0,
                 lrda_cond_asl: bool = False, lrda_dt_bound: float = 1.0,
                 lrda_bilinear: bool = False, lrda_pv_ch: int = 4,
                 lrda_pv_geo: bool = False, hv_scan: bool = False,
                 sig_xin: bool = False, dt_rank: int = 1, tfdm: bool = False,
                 scan_dropout_p: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ss2d = SS2D(d_model, d_state=d_state, n_directions=n_directions,
                         lrda_inscan=lrda_inscan, lrda_t1_ch=lrda_t1_ch,
                         lrda_rank=lrda_rank, lrda_bound=lrda_bound,
                         lrda_cond_asl=lrda_cond_asl, lrda_dt_bound=lrda_dt_bound,
                         lrda_bilinear=lrda_bilinear, lrda_pv_ch=lrda_pv_ch,
                         lrda_pv_geo=lrda_pv_geo, hv_scan=hv_scan, sig_xin=sig_xin,
                         dt_rank=dt_rank, tfdm=tfdm, scan_dropout_p=scan_dropout_p)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = (
            ConvFFN(d_model, hidden) if use_conv_ffn
            else nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                               nn.Linear(hidden, d_model))
        )
        # DropPath omitted for simplicity (small student)

    def forward(self, x: Tensor, t1: Optional[Tensor] = None,
                repro_gate: Optional[Tensor] = None) -> Tensor:  # [B, H, W, C]
        x = x + self.ss2d(self.norm1(x), t1, repro_gate)
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Hierarchical encoder/decoder over VSSBlock stacks
# -----------------------------------------------------------------------------

class _PatchMerge(nn.Module):
    """2× spatial down by concatenating 2x2 neighbours along channel."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * c_in)
        self.proj = nn.Linear(4 * c_in, c_out, bias=False)

    def forward(self, x: Tensor) -> Tensor:                 # [B, H, W, C]
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, (H, W)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)             # [B, H/2, W/2, 4C]
        x = self.norm(x)
        return self.proj(x)


class _PatchMergeConv(nn.Module):
    """2× spatial down via an OVERLAPPING 3×3 stride-2 conv (vs _PatchMerge's
    non-overlapping 2×2 concat). Overlapping receptive field + anti-aliasing give
    a stronger local inductive bias at every stage transition. NHWC in/out."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(c_in)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:                 # [B, H, W, C]
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, (H, W)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()              # NCHW
        x = self.conv(x)                                    # [B, c_out, H/2, W/2]
        return x.permute(0, 2, 3, 1).contiguous()           # NHWC


class _PatchExpand(nn.Module):
    """2× spatial up via linear expansion + reshape."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.Linear(c_in, 4 * c_out, bias=False)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: Tensor) -> Tensor:                 # [B, H, W, C]
        B, H, W, C = x.shape
        x = self.proj(x)                                    # [B, H, W, 4*c_out]
        x = x.view(B, H, W, 2, 2, -1).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, -1)
        return self.norm(x)


class ASLMambaStudent(nn.Module):
    """T1-free Mamba-based student for distillation from v39.

    Forward:
        set_a [B, T, 1, H, W] -> SVFW aggregator -> [B, 1, H, W]
                              -> patch embed (4x4) -> [B, base_ch, H/4, W/4]
                              -> 3-stage VSSBlock encoder
                              -> bottleneck VSSBlock × N_b
                              -> 3-stage VSSBlock decoder with skip-add
                              -> patch unembed -> [B, 1, H, W]

    The bottleneck feature map ([B, base_ch * 4, H/16, W/16]) is exposed as
    `student.feat_bottleneck` for adapter-bridged feature distillation
    against the teacher's `fused_map [B, 256, H/16, W/16]`.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 24,
        depth_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
        d_state: int = 16,
        patch_size: int = 4,
        teacher_bottleneck_ch: int = 256,
        n_directions: int = 1,
    ) -> None:
        super().__init__()
        self.svfw = SpatialVaryingFrameWeighting(in_ch=in_ch)
        self.patch_size = patch_size

        # patch embed (4x4 conv)
        self.embed = nn.Conv2d(in_ch, base_ch, kernel_size=patch_size,
                               stride=patch_size)
        # encoder: 3 levels of (VSSBlock x N) + downsample
        c0, c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4, base_ch * 4
        self.enc0 = nn.ModuleList([VSSBlock(c0, d_state, n_directions=n_directions) for _ in range(depth_blocks[0])])
        self.down0 = _PatchMerge(c0, c1)
        self.enc1 = nn.ModuleList([VSSBlock(c1, d_state, n_directions=n_directions) for _ in range(depth_blocks[1])])
        self.down1 = _PatchMerge(c1, c2)
        # bottleneck
        self.bottleneck = nn.ModuleList([
            VSSBlock(c2, d_state, n_directions=n_directions) for _ in range(depth_blocks[2])
        ])
        # decoder symmetric
        self.up1 = _PatchExpand(c2, c1)
        self.dec1 = nn.ModuleList([VSSBlock(c1, d_state, n_directions=n_directions) for _ in range(depth_blocks[3])])
        self.up0 = _PatchExpand(c1, c0)
        self.dec0 = nn.ModuleList([VSSBlock(c0, d_state, n_directions=n_directions) for _ in range(depth_blocks[3])])

        # patch unembed: pixel shuffle from c0 → 1
        # NOTE: bare PixelShuffle produces visible checkerboard / patch-grid
        # artifacts because the 4 sub-pixel channels are learned independently
        # and have no inductive bias to be smooth across the upsample boundary.
        # Fixes:
        #   (a) ICNR init (Aitken et al. 2017): initialise the pre-shuffle conv
        #       so that all 4 sub-channels start from the same kernel — the
        #       network begins as nearest-neighbour upsample, no grid bias.
        #   (b) 3×3 refinement conv after PixelShuffle to smooth residual
        #       sub-pixel discontinuities (standard practice in SR networks).
        pre_shuffle = nn.Conv2d(c0, in_ch * patch_size * patch_size, kernel_size=1)
        _icnr_init_(pre_shuffle.weight, upscale_factor=patch_size)
        self.unembed = nn.Sequential(
            pre_shuffle,
            nn.PixelShuffle(patch_size),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
        )

        # adapter for feature distillation: c2 → teacher_bottleneck_ch
        # student bottleneck spatial = H/16, teacher fused_map spatial = H/16 → match
        self.bottleneck_adapter = nn.Conv2d(c2, teacher_bottleneck_ch, kernel_size=1)

        self.bottleneck_ch = c2

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, x: Tensor) -> Tensor:
        for blk in blocks:
            x = blk(x)
        return x

    def forward(
        self,
        set_a: Tensor,                                        # [B, T, 1, H, W]
        lengths: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, Tensor]:
        # 1) SVFW aggregation (T1-free). Pass mask OR lengths, never both
        # (SVFW raises if both given; matches ASLT1Denoiser convention).
        if mask is not None:
            agg, _ = self.svfw(set_a, mask=mask)
        else:
            agg, _ = self.svfw(set_a, lengths=lengths)

        # 2) patch embed
        x = self.embed(agg)                                    # [B, c0, H/4, W/4]
        x = x.permute(0, 2, 3, 1).contiguous()                # [B, H/4, W/4, c0]

        # 3) encoder
        e0 = self._run_blocks(self.enc0, x)                    # [B, H/4, W/4, c0]
        e1_in = self.down0(e0)                                 # [B, H/8, W/8, c1]
        e1 = self._run_blocks(self.enc1, e1_in)
        e2_in = self.down1(e1)                                 # [B, H/16, W/16, c2]

        # 4) bottleneck
        b = self._run_blocks(self.bottleneck, e2_in)
        bottleneck_feat = b                                    # [B, H/16, W/16, c2]

        # 5) decoder
        d1_in = self.up1(b) + e1                               # [B, H/8, W/8, c1]
        d1 = self._run_blocks(self.dec1, d1_in)
        d0_in = self.up0(d1) + e0                              # [B, H/4, W/4, c0]
        d0 = self._run_blocks(self.dec0, d0_in)

        # 6) patch unembed
        d0 = d0.permute(0, 3, 1, 2).contiguous()              # [B, c0, H/4, W/4]
        recon = self.unembed(d0)                              # [B, 1, H, W]

        out: Dict[str, Tensor] = {"asl_recon": recon}
        if return_features:
            # adapter to teacher dim for distillation
            bf = bottleneck_feat.permute(0, 3, 1, 2).contiguous()   # [B, c2, H/16, W/16]
            out["feat_bottleneck"] = bf
            out["feat_bottleneck_adapted"] = self.bottleneck_adapter(bf)
        return out
