# -*- coding: utf-8 -*-
# ============================================================================
# RETIRED 2026-07-16 — MoSSM / CIG-Net-v2 backbone.
# Not part of any runnable path: the runner rejects --use_mossm_encoder
# (asl_t1_guided_runner_dmvae_n2n.py). The main method is CIG-VSS + EC-LRDA
# (models/content_guard_vmamba.py::VMambaEncoder2D) with a plain
# ConvDecoderWithSkips2D decoder — see ASL_dmvae/docs/cig_vss.md. This file is kept
# (not physically archived) only because models/blocks.py + the VSS/conv
# guards share type/forward-contract references and old v2 ckpts must stay
# readable from git history. Do not extend.
# ============================================================================
"""MoSSM-ASL: Modulated Selective State-Space Module + Hierarchical Encoder.

The novelty for v42 (KBS submission): an SSM-block-level analogue of V=ASL
cross-attention. The selective SSM has three input-dependent matrices:
    B (write-into-state),  C (read-from-state),  Δ (per-token timescale).

In a vanilla VMamba / MambaIR block all three are projected from the same
input. Here we expose two inputs — x_asl and t1_cond — and route them
asymmetrically:

    B_a(x_asl, t1_cond)   # T1 may participate in WHAT gets written
    Δ_a(x_asl, t1_cond)   # T1 may participate in HOW FAST state evolves
    C_a(x_asl)            # ⚠️ ASL ONLY — V=ASL invariant on the READOUT
    u_a = x_asl           # input value stream is ASL-only

Anti-hallucination proof (one-liner for the paper):
    y = C(x_asl) · h,    h is a state built up via u=x_asl, modulated by
    (B,Δ) which depend on (x_asl, t1). Therefore ∂y/∂t1 routes only through
    h's *dynamics* (the recurrent path), never directly through the readout
    projection. T1 cannot inject a pixel value into y; it can only modulate
    which ASL information is preserved / suppressed at each spatial position.

Inspired by:
  - Gu & Dao 2023, Mamba: Selective State Spaces
  - Liu et al. NeurIPS 2024, VMamba: 2D Cross-Scan
  - Guo et al. CVPR 2025, MambaIRv2: Attentive SSM for Image Restoration
  - This paper (v42): T1-modulated readout-preserving SSM for cross-modal
    medical denoising with structural anti-hallucination guarantee.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.asl_mamba_student import selective_scan_pytorch


class DynamicsAdapter(nn.Module):
    """Dynamics Adapter — paper: low-rank mode (cada_lr=True, CIG-Net v2) is the
    **LRDA (Low-Rank Dynamics Adapter)**; the gated-additive mode is the v1
    ablation baseline. (was CADA.)

    Channel-Adaptive Dynamics Adapter.

    Two decoupled components, zero architectural caps:

    Two modes:
      - **CADA-LR** (`use_lowrank=True`, the v2 model): returns only the low-rank
        B residual ΔB; the Δ-timestep residual is a separate zero-init projection
        in MoSSM2D (no gate, no α scalar).
      - **Gated CADA** (`use_lowrank=False`, v1 ablation baseline): per-pixel
        evidence gates `g = sigmoid(Conv1d([x_asl, t1_dense]))` ∈ [0,1] plus
        zero-init per-channel adapters `α` (vector for B, scalar for Δ). At step 0
        `α=0` ⇒ the T1 residual is exactly zero ⇒ ASL-only start.

    Anti-hallucination relies on:
      - Structural V=ASL contract (C·h readout untouched by T1)
      - CMAM V=mossm_mem (ASL only)
      - T1-free decoder (NAFDecoder signature)
      - Empirical mismatched-T1 ε measurement

    No architectural ceiling. No norm-balance. No specially-crafted init.

    Inputs (per direction, in 1-D scan order):
        x_asl_seq:   [B, L, d_inner]   ASL features
        t1_seq:      [B, L, d_inner]   T1 dense features (already aligned via t1_proj)

    Outputs:
        CADA-LR:    ΔB [B, L, d_state]                 (low-rank B residual only)
        Gated CADA: (g_B [B, L, n_groups_B], g_D [B, L, 1])   per-pixel gates ∈ [0,1]

    Caller (gated CADA) applies the gate as:
        B_combined  = B_asl  + (g_B  * α_B) ⊙ B_t1_part   (with α_B broadcast)
        dt_combined = dt_asl + (g_D  * α_D) ⊙ dt_t1_part
    """

    def __init__(self, d_asl_inner: int, n_groups_B: int = 4,
                 use_lowrank: bool = False, d_state: int = 16,
                 lr_rank: int = 8, lr_bound: float = 4.0) -> None:
        super().__init__()
        # Evidence channels: ASL features + dense T1 features. T1 here is the
        # dense feature (already projected to d_inner via t1_proj in MoSSM2D),
        # NOT categorical PV — matches the "structural feature uses T1" choice.
        # (The legacy z(log noise_var) side-channel was removed 2026-06-21.)
        in_ch = d_asl_inner + d_asl_inner          # x_asl + t1
        self.use_lowrank = bool(use_lowrank)
        self.n_groups_B = int(n_groups_B)
        if self.use_lowrank:
            # CADA-LR (Anatomy-Conditioned Low-Rank Dynamics Adapter; see
            # ASL_dmvae/docs/cada_lr_design.md). B residual becomes a T1-conditioned
            # low-rank adapter ACTING ON an ASL projection:
            #     ΔB = U_B( s ⊙ V_B u ),   s = lr_bound · tanh(φ_B(t1)).
            # T1 enters ONLY through s (the rank coefficients); the content
            # V_B u is pure ASL. U_B is zero-init so ΔB=0 at step 0 (backbone
            # = ASL-only SSM). φ_B reads ONLY t1 — "T1 controls the operator,
            # ASL supplies the content". No additive T1 term exists.
            # The Δ residual is NOT handled here in CADA-LR: it is a plain
            # zero-init projection of (u,t1) in MoSSM2D, with softplus on the
            # discretized step keeping Δ>0 (2026-06-23: dropped the per-pixel g_D
            # gate + scalar α_D — they added no expressiveness over the zero-init
            # Δ projection, and the bounded gate was redundant with softplus).
            self.lr_rank = int(lr_rank)
            self.lr_bound = float(lr_bound)
            self.V_B = nn.Linear(d_asl_inner, int(lr_rank), bias=False)
            self.phi_B = nn.Conv1d(d_asl_inner, int(lr_rank), kernel_size=1)  # cond = T1 only
            self.U_B = nn.Linear(int(lr_rank), int(d_state), bias=False)
            nn.init.zeros_(self.U_B.weight)
        else:
            # Gated-CADA (v1 ablation baseline) keeps the per-pixel Δ gate + the
            # zero-init α scalars for both B and Δ.
            self.conv_D = nn.Conv1d(in_ch, 1, kernel_size=1)
            self.alpha_D = nn.Parameter(torch.zeros(1))
            self.conv_B = nn.Conv1d(in_ch, n_groups_B, kernel_size=1)
            # Standard Kaiming init (PyTorch default for Conv1d) — no special init.
            # Biases default to small uniform values from Kaiming uniform; that's
            # fine because sigmoid(small) ≈ 0.5, and α=0 makes T1 contribution zero
            # regardless of gate value at step 0.
            self.alpha_B = nn.Parameter(torch.zeros(n_groups_B))

    def forward(
        self,
        x_asl_seq: Tensor,
        t1_seq: Tensor,
    ):
        # All inputs [B, L, *]. Conv1d expects [B, C, L].
        if self.use_lowrank:
            # CADA-LR: returns ΔB only ([B,L,d_state]); ΔB feeds B directly. The Δ
            # residual is handled by a zero-init projection in MoSSM2D.
            v = self.V_B(x_asl_seq)                                       # [B, L, r]  (ASL content)
            cond = t1_seq.transpose(1, 2)                                 # [B, d_inner, L]
            s = self.lr_bound * torch.tanh(self.phi_B(cond)).transpose(1, 2)  # [B, L, r]  (T1 op)
            dB = self.U_B(s * v)                                          # [B, L, d_state]; 0 at init
            return dB
        inp_t = torch.cat([x_asl_seq, t1_seq], dim=-1).transpose(1, 2)    # [B, in_ch, L]
        g_D = torch.sigmoid(self.conv_D(inp_t)).transpose(1, 2)           # [B, L, 1]   ∈ [0,1]
        g_B = torch.sigmoid(self.conv_B(inp_t)).transpose(1, 2)           # [B, L, G_B] ∈ [0,1]
        return g_B, g_D


# -----------------------------------------------------------------------------
# T1-Modulated Selective State-Space Module (2D, 4-direction-capable)
# -----------------------------------------------------------------------------

class AnatomyConditionedSSM2D(nn.Module):
    """Anatomy-Conditioned SSM Unit (AC-SSM Unit) — the global SSM branch; T1
    only modulates the state dynamics, never injects content (was MoSSM2D).

    T1-modulated 2D selective scan.

    Inputs (NHWC):
        x_asl  : [B, H, W, C_asl]
        t1_cond: [B, H, W, C_t1]  — already at this stage's spatial scale
    Output:
        [B, H, W, C_asl]
    """

    def __init__(
        self,
        d_asl: int,
        d_t1: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        n_directions: int = 1,
        t1_gated_bd: bool = False,
        t1_gate_init: float = -4.0,
        # CADA (2026-05-25): Channel-Adaptive Dynamics Adapter. Per-pixel
        # evidence gate (no caps) using dense T1 evidence + zero-initialised
        # per-channel α adapters. Mutually exclusive with t1_gated_bd at the
        # same stage.
        use_cada: bool = False,
        cada_n_groups_B: int = 4,
        # CADA-LR (2026-06-05): T1-conditioned low-rank dynamics adapter on B.
        # When True, the gated T1-residual on B is replaced by U_B(s⊙V_B u);
        # see ASL_dmvae/docs/cada_lr_design.md. Δ keeps the gated form.
        cada_lr: bool = False,
        cada_lr_rank: int = 8,
        cada_lr_bound: float = 4.0,
    ) -> None:
        super().__init__()
        assert n_directions in (1, 2, 4), n_directions
        self.d_asl = d_asl
        self.d_t1 = d_t1
        self.d_state = d_state
        self.d_inner = expand * d_asl
        self.n_directions = n_directions
        # v42g (P2, 2026-05-18): split B/Δ projection into ASL base + gated T1
        # residual. When False (v42f behaviour), B/Δ take concat(u, t1_aligned)
        # at full strength. When True, B/Δ = BD_asl(u) + σ(gate)·BD_t1(joint),
        # with gate init sigmoid(-4)≈0.018 so the network starts as ASL-only
        # MoSSM and only opens the T1 path if training rewards it.
        self.t1_gated_bd = bool(t1_gated_bd)

        # CADA: per-pixel evidence gate (no caps) + per-channel α adapter.
        self.use_cada = bool(use_cada)
        self.cada_n_groups_B = int(cada_n_groups_B)
        self.cada_lr = bool(cada_lr)
        if self.use_cada:
            self.cada_gates = nn.ModuleList([
                CADA(
                    d_asl_inner=self.d_inner,
                    n_groups_B=int(cada_n_groups_B),
                    use_lowrank=self.cada_lr,
                    d_state=int(d_state),
                    lr_rank=int(cada_lr_rank),
                    lr_bound=float(cada_lr_bound),
                )
                for _ in range(n_directions)
            ])
            if not self.cada_lr:
                # Group-broadcast path needs d_state divisible by n_groups_B.
                # CADA-LR has no group structure, so the constraint is lifted.
                assert d_state % int(cada_n_groups_B) == 0, (
                    f"d_state={d_state} must be divisible by cada_n_groups_B="
                    f"{cada_n_groups_B}"
                )

        # ASL input projection: produces u (the value stream) and z (the gate)
        self.in_proj_asl = nn.Linear(d_asl, 2 * self.d_inner, bias=False)
        # T1 alignment: align T1 channel dim to d_inner for joint conditioning
        self.t1_proj = nn.Linear(d_t1, self.d_inner, bias=False)
        # Depthwise conv on the SSM input (ASL-derived only)
        self.dwconv = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv // 2, groups=self.d_inner,
        )

        # B and Δ projections — joint conditioning on (u, t1_aligned).
        if self.cada_lr:
            # CADA-LR supplies B via the low-rank adapter; from the joint (u, t1)
            # stream it needs ONLY the 1-channel Δ_t1 residual. A full (d_state+1)
            # x_proj_BD would compute then discard the entire B-part, so use a
            # thin Δ-only projection instead (2026-06-21 cleanup, #1).
            self.x_proj_dt_t1 = nn.ModuleList([
                nn.Linear(2 * self.d_inner, 1, bias=False)
                for _ in range(n_directions)
            ])
            # Zero-init so the Δ-T1 residual is exactly 0 at step 0 (ASL-only
            # start), replacing the dropped scalar α_D (2026-06-23).
            for _lin in self.x_proj_dt_t1:
                nn.init.zeros_(_lin.weight)
        else:
            self.x_proj_BD = nn.ModuleList([
                nn.Linear(2 * self.d_inner, d_state + 1, bias=False)
                for _ in range(n_directions)
            ])
        if self.t1_gated_bd or self.use_cada:
            # ASL-only B/Δ base path: takes u only (no T1). Identity-safe when
            # the T1 residual gate is closed. Shared between the scalar
            # (t1_gated_bd) and CADA gating modes.
            self.x_proj_BD_asl = nn.ModuleList([
                nn.Linear(self.d_inner, d_state + 1, bias=False)
                for _ in range(n_directions)
            ])
        if self.t1_gated_bd:
            # Per-direction scalar T1 gate (logit). σ(-4)≈0.018 → near-zero T1
            # contribution at init; training learns when to open.
            self.t1_gate_bd = nn.ParameterList([
                nn.Parameter(torch.tensor(float(t1_gate_init)))
                for _ in range(n_directions)
            ])
        # Δ MLP head (same as vanilla Mamba): 1 -> d_inner with softplus
        self.dt_proj = nn.ModuleList([
            nn.Linear(1, self.d_inner, bias=True) for _ in range(n_directions)
        ])
        # C projection ONLY from u (ASL stream). This is the V=ASL invariant.
        # No T1 channel ever touches C.
        self.x_proj_C = nn.ModuleList([
            nn.Linear(self.d_inner, d_state, bias=False) for _ in range(n_directions)
        ])

        # Static A (diag, enforced negative via -exp(log A)), and D skip
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.ParameterList([
            nn.Parameter(torch.log(A), requires_grad=True) for _ in range(n_directions)
        ])
        self.D_skip = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner)) for _ in range(n_directions)
        ])

        self.out_proj = nn.Linear(self.d_inner, d_asl, bias=False)
        self.act = nn.SiLU()

    @staticmethod
    def _serialize(x: Tensor, direction: int) -> Tensor:
        """[B, C, H, W] -> [B, L=H*W, C] under one of 4 scan orders.
        Identical to SS2D._serialize in asl_mamba_student.py.
        """
        if direction == 0:
            return x.flatten(2).transpose(1, 2)
        if direction == 1:
            return x.flatten(2).transpose(1, 2).flip(1)
        if direction == 2:
            return x.transpose(2, 3).flatten(2).transpose(1, 2)
        return x.transpose(2, 3).flatten(2).transpose(1, 2).flip(1)

    @staticmethod
    def _deserialize(seq: Tensor, B: int, C: int, H: int, W: int, direction: int) -> Tensor:
        if direction == 1 or direction == 3:
            seq = seq.flip(1)
        if direction <= 1:
            return seq.transpose(1, 2).reshape(B, C, H, W)
        return seq.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)

    @staticmethod
    def _tabs_perm(tissue_seg: Tensor) -> tuple:
        """TABS (Tissue-Aware Bidirectional Scan): compute a per-batch
        permutation that groups pixels of the same dominant tissue class
        contiguously in the 1D scan sequence. Within each class, the
        original row-major order is preserved (stable argsort).

        This makes the SSM state evolve through one tissue class at a
        time, avoiding cross-tissue state contamination during scan.

        Args:
            tissue_seg: [B, n_class, H, W] PV map or logits (4 classes:
                        GM, WM, CSF, BG)
        Returns:
            perm:     [B, L=H*W] long — perm[b, k] = original index of the
                      pixel that should land at sequence position k
            inv_perm: [B, L] long — inverse permutation (for deserialise)
        """
        B, _n, H, W = tissue_seg.shape
        L = H * W
        cls = tissue_seg.argmax(dim=1).reshape(B, L)                # [B, L]
        perm = torch.argsort(cls, dim=1, stable=True)               # [B, L]
        inv_perm = torch.argsort(perm, dim=1)
        return perm, inv_perm

    def forward(
        self,
        x_asl: Tensor,
        t1_cond: Tensor,
        tissue_seg: Optional[Tensor] = None,
        t1_seg_probs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x_asl: [B, H, W, d_asl]
            t1_cond: [B, H, W, d_t1]
            tissue_seg: [B, n_class, H, W] used for TABS scan-ordering only
            t1_seg_probs: unused (legacy parameter, kept for signature compat)
        """
        del t1_seg_probs  # legacy, no longer consumed
        B, H, W, _ = x_asl.shape

        # ASL: u + gate z (both from ASL only)
        uz = self.in_proj_asl(x_asl)                                # [B,H,W, 2*d_inner]
        u, z = uz.chunk(2, dim=-1)
        # T1 alignment
        t1_aligned = self.t1_proj(t1_cond)                          # [B,H,W, d_inner]

        # dwconv on u
        u_nchw = u.permute(0, 3, 1, 2).contiguous()
        u_nchw = self.act(self.dwconv(u_nchw))
        t1_nchw = t1_aligned.permute(0, 3, 1, 2).contiguous()

        # TABS permutation (computed once, applied per direction at row-major
        # serialisation). For non-row-major directions we fall back to plain
        # serialisation — combining TABS with 4-direction scan is left to
        # future work.
        use_tabs = (tissue_seg is not None)
        if use_tabs:
            # Match seg spatial size to current stage if mismatched
            if tissue_seg.shape[-2] != H or tissue_seg.shape[-1] != W:
                tissue_seg = F.adaptive_avg_pool2d(tissue_seg.float(), (H, W))
            perm, inv_perm = self._tabs_perm(tissue_seg)

        out_acc: Tensor = u_nchw.new_zeros(B, self.d_inner, H, W)
        for d_idx in range(self.n_directions):
            u_seq = self._serialize(u_nchw, d_idx)                  # [B, L, d_inner]
            t1_seq = self._serialize(t1_nchw, d_idx)                # [B, L, d_inner]

            # TABS reordering on direction 0 only (row-major).
            applied_tabs = use_tabs and d_idx == 0
            if applied_tabs:
                # `perm` is set inside the `if use_tabs:` branch above. The
                # conjunction `use_tabs and d_idx == 0` guarantees it's bound
                # here; assert silences the static checker.
                assert perm is not None  # type: ignore[possibly-unbound]
                idx = perm.unsqueeze(-1).expand(-1, -1, u_seq.shape[-1])
                u_seq = torch.gather(u_seq, dim=1, index=idx)
                t1_seq = torch.gather(t1_seq, dim=1, index=idx)

            # B, Δ conditioned on (u, t1)
            joint = torch.cat([u_seq, t1_seq], dim=-1)              # [B, L, 2*d_inner]
            if self.use_cada:
                # CADA: ASL base + per-pixel evidence gate × per-channel α adapter.
                # No g_max cap, no norm-balance, no special init. α=0 at step 0
                # ⇒ T1 contribution is exactly zero. Training learns where + how
                # much to use T1 from data, bounded only structurally by
                # V=ASL (the C-readout below is u_seq only).
                cada_mod: CADA = self.cada_gates[d_idx]  # type: ignore[assignment]
                BD_asl = self.x_proj_BD_asl[d_idx](u_seq)           # [B, L, d_state+1]
                B_asl_part, dt_asl_part = BD_asl.split([self.d_state, 1], dim=-1)
                if self.cada_lr:
                    # CADA-LR: B gets a T1-conditioned low-rank adapter acting on
                    # an ASL projection (ΔB = U_B(s ⊙ V_B u), zero-init U_B). The Δ
                    # residual is a plain zero-init projection of (u,t1); softplus
                    # on the discretized step keeps Δ>0, so no gate/α is needed
                    # (2026-06-23: dropped g_D + α_D).
                    dB = cada_mod(u_seq, t1_seq)                   # [B,L,d_state]
                    B_combined = B_asl_part + dB
                    dt_t1_part = self.x_proj_dt_t1[d_idx](joint)    # [B, L, 1]; 0 at init
                    dt_combined = dt_asl_part + dt_t1_part
                else:
                    BD_t1 = self.x_proj_BD[d_idx](joint)            # [B, L, d_state+1]
                    g_B, g_D = cada_mod(u_seq, t1_seq)             # [B,L,G_B], [B,L,1]
                    B_t1_part, dt_t1_part = BD_t1.split([self.d_state, 1], dim=-1)
                    # Broadcast group gates → state channels and apply per-channel α.
                    reps = self.d_state // self.cada_n_groups_B
                    g_B_full = g_B.repeat_interleave(reps, dim=-1)  # [B, L, d_state]
                    alpha_B_full = cada_mod.alpha_B.repeat_interleave(reps)  # [d_state]
                    B_combined = B_asl_part + (g_B_full * alpha_B_full) * B_t1_part
                    dt_combined = dt_asl_part + (g_D * cada_mod.alpha_D) * dt_t1_part
                BD = torch.cat([B_combined, dt_combined], dim=-1)
            elif self.t1_gated_bd:
                # Legacy scalar-gate path (v42g P2). Kept for backward compat
                # and as a fallback when CADA is disabled.
                BD_asl = self.x_proj_BD_asl[d_idx](u_seq)           # [B, L, d_state+1]
                BD_t1 = self.x_proj_BD[d_idx](joint)                # [B, L, d_state+1]
                gate = torch.sigmoid(self.t1_gate_bd[d_idx])
                BD = BD_asl + gate * BD_t1
            else:
                BD = self.x_proj_BD[d_idx](joint)                   # [B, L, d_state+1]
            B_, dt_raw = BD.split([self.d_state, 1], dim=-1)
            delta = F.softplus(self.dt_proj[d_idx](dt_raw))         # [B, L, d_inner]
            # C from ASL stream only (V=ASL invariant)
            C_ = self.x_proj_C[d_idx](u_seq)                        # [B, L, d_state]
            A = -torch.exp(self.A_log[d_idx])                       # [d_inner, N]
            y = selective_scan_pytorch(
                u=u_seq, delta=delta, A=A,
                B_=B_, C_=C_, D_skip=self.D_skip[d_idx],
            )                                                       # [B, L, d_inner]

            # Inverse TABS permutation back to row-major before deserialise
            if applied_tabs:
                idx_inv = inv_perm.unsqueeze(-1).expand(-1, -1, y.shape[-1])
                y = torch.gather(y, dim=1, index=idx_inv)

            out_acc = out_acc + self._deserialize(y, B, self.d_inner, H, W, d_idx)
        out_acc = out_acc / float(self.n_directions)

        # gate
        out_nhwc = out_acc.permute(0, 2, 3, 1).contiguous()
        out_nhwc = out_nhwc * self.act(z)
        return self.out_proj(out_nhwc)                              # [B, H, W, d_asl]


# -----------------------------------------------------------------------------
# Pre-norm MoSSM block (MoSSM + FFN)
# -----------------------------------------------------------------------------

__doc_cmam__ = """\
CMAM — Cross-Modal Attentive Memory (v42 main contribution, READER pattern).

Conceptual role:
    MoSSM is the *memory writer* — it scans x_asl along a learned 1D path
    and produces a state-derived output map `mossm_mem` that summarises
    local recurrent context. CMAM is the *memory reader* — it lets each
    output position non-locally pool from that memory, using T1 features
    as the addressing key.

Data flow (Q, K, V come from THREE DISTINCT sources):
    Q = W_q · query_src      ← usually x_asl (the block input)
    K = W_k · key_src        ← T1 features (addressing oracle)
    V = W_v · value_src      ← Mamba's memory map `mossm_mem` (ASL-derived)

V=ASL invariant — strengthened:
    V is drawn from `value_src`, which is `mossm_mem` ⊂ ASL feature space
    (MoSSM already enforces C=ASL only → mossm_mem is anti-T1-hallucination
    guaranteed). So V here is doubly-filtered ASL information. T1 cannot
    contribute pixel values to attention output through any path.

Anchor:
    - A2Mamba (arXiv 2025): attention re-aggregates SSM hidden state map,
      V drawn from those same hidden states. CMAM extends this to the
      cross-modal setting (T1 as K, ASL state as V), with V=ASL invariant.

(The legacy noise-variance BLUE-Attn weighting was removed 2026-06-21 along
with the rest of the Var_T side-channel.)
"""


class AnatomyKeyedMemoryReader(nn.Module):  # AKMR — paper name (was CMAM)
    __doc__ = __doc_cmam__

    def __init__(
        self,
        d_asl: int,
        d_t1: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        k_rank_div: int = 1,
    ) -> None:
        super().__init__()
        while d_asl % n_heads != 0 and n_heads > 1:
            n_heads //= 2
        self.n_heads = n_heads
        # v42g (P3, 2026-05-18): low-rank K bottleneck. When k_rank_div > 1,
        # the T1 key is first compressed to d_t1 // k_rank_div channels before
        # being passed to the cross-attention. The probe showed CMAM K=t1_feat
        # carried full anatomical detail through to attention pattern; a low-
        # rank bottleneck physically restricts the channel-dim information K
        # can carry, weakening anatomy-based addressing.
        self.k_rank_div = max(1, int(k_rank_div))
        kdim_eff = max(1, d_t1 // self.k_rank_div) if self.k_rank_div > 1 else d_t1
        # Three independent norms because Q, K, V come from three different
        # tensors (x_asl, t1_cond, mossm_mem). Sharing norms would be wrong.
        self.norm_q = nn.LayerNorm(d_asl)
        self.norm_k = nn.LayerNorm(d_t1)
        self.norm_v = nn.LayerNorm(d_asl)
        if self.k_rank_div > 1:
            self.k_compress = nn.Linear(d_t1, kdim_eff, bias=False)
        else:
            self.k_compress = None
        # K from T1 (different channel dim), V from ASL state map.
        # nn.MultiheadAttention's kdim/vdim allow the dim asymmetry; V is
        # internally projected from vdim=d_asl → embed_dim=d_asl.
        self.attn = nn.MultiheadAttention(
            embed_dim=d_asl, num_heads=n_heads,
            kdim=kdim_eff, vdim=d_asl,
            dropout=dropout, batch_first=True,
        )
        # Zero-init the attention output projection (ControlNet-style): CMAM
        # starts at exactly 0 contribution and ramps in via the residual
        # `x_asl + global_readout`. This REPLACES the old scalar sigmoid gate
        # (`sigmoid(gate)·out`), which was expressiveness-redundant with this
        # very out_proj (a global scalar = scaling out_proj.weight/bias) and only
        # served a warm-start role — now done exactly (0 vs sigmoid(-2)≈0.12) with
        # one fewer parameter. q/k/v in_proj stay random, so the T1-addressed
        # attention pattern still learns from step 0; only the output scale is 0.
        nn.init.zeros_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(
        self,
        query_src: Tensor,                   # [B, H, W, d_asl] — usually x_asl
        key_src: Tensor,                     # [B, H, W, d_t1]  — T1 features
        value_src: Tensor,                   # [B, H, W, d_asl] — Mamba memory map
    ) -> Tensor:
        B, H, W, _ = query_src.shape
        N = H * W
        q_norm = self.norm_q(query_src).reshape(B, N, -1)        # [B, N, d_asl]
        k_norm = self.norm_k(key_src).reshape(B, N, -1)          # [B, N, d_t1]
        if self.k_compress is not None:                            # v42g P3 low-rank K
            k_norm = self.k_compress(k_norm)                       # [B, N, kdim_eff]
        v_norm = self.norm_v(value_src).reshape(B, N, -1)        # [B, N, d_asl]

        out, _ = self.attn(query=q_norm, key=k_norm, value=v_norm,
                           need_weights=False)
        return out.reshape(B, H, W, -1)


class ASLLocalContextUnit(nn.Module):
    """ASL Local Context Unit (LCU) — paper name (was ASLLocalBranch).

    ASL-only local restoration path for the Hybrid AG-SSM block.

    A depthwise-separable conv block that runs on ASL features ONLY (no T1,
    no noise argument). It bypasses the MoSSM d_state bottleneck so that
    fine, signal-correlated high-frequency ASL detail (that the low-rank SSM
    state would otherwise smooth away) can survive. Because it sees ASL only
    and its output is fused with the (also ASL-valued) MoSSM/CMAM readout, it
    cannot introduce T1 content — the V=ASL / T1-free-content contract holds.

    Input/output layout: NHWC, matching MoSSMBlock internals.
    """

    def __init__(self, dim: int, expand: float = 2.0, kernel_size: int = 3):
        super().__init__()
        hidden = int(dim * expand)
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size, padding=pad,
                      groups=hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
        )

    def forward(self, x_nhwc: Tensor) -> Tensor:
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        y = self.net(x)
        return y.permute(0, 2, 3, 1).contiguous()


class AnatomyGuidedSSMBlock(nn.Module):
    """Anatomy-Guided SSM Block (AG-SSM Block) — paper name (was MoSSMBlock).

    ASL Local Context Unit (LCU) ∥ Anatomy-Conditioned SSM Unit, fused by a
    zero-init passthrough Conv1x1. Pre-norm SSM + optional AKMR memory-read + FFN.
    """

    def __init__(
        self,
        d_asl: int,
        d_t1: int,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
        n_directions: int = 1,
        use_cmam: bool = False,
        cmam_n_heads: int = 4,
        t1_gated_bd: bool = False,
        t1_gate_init: float = -4.0,
        cmam_k_rank_div: int = 1,
        # CADA propagated from MoSSMEncoder2D per-stage settings.
        use_cada: bool = False,
        cada_n_groups_B: int = 4,
        cada_lr: bool = False,
        cada_lr_rank: int = 8,
        cada_lr_bound: float = 4.0,
        # Hybrid CNN-SSM (2026-06-05): optional ASL-only local conv branch.
        use_hybrid_mossm_block: bool = False,
        hybrid_local_expand: float = 2.0,
        hybrid_local_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.norm_asl = nn.LayerNorm(d_asl)
        self.norm_t1 = nn.LayerNorm(d_t1)
        self.mossm = MoSSM2D(
            d_asl, d_t1, d_state=d_state, n_directions=n_directions,
            t1_gated_bd=t1_gated_bd, t1_gate_init=t1_gate_init,
            use_cada=use_cada,
            cada_n_groups_B=cada_n_groups_B,
            cada_lr=cada_lr,
            cada_lr_rank=cada_lr_rank,
            cada_lr_bound=cada_lr_bound,
        )
        self.use_cmam = bool(use_cmam)
        if self.use_cmam:
            self.cmam = CMAM(
                d_asl, d_t1, n_heads=cmam_n_heads,
                k_rank_div=cmam_k_rank_div,
            )
        # Hybrid CNN-SSM: ASL-only local branch fused with the global readout.
        # hybrid_fuse is init'd as global passthrough (local half = 0, global
        # half = identity) so the block reproduces the global-only behaviour at
        # step 0; the local path is learned through the 1x1 fusion weights.
        self.use_hybrid_mossm_block = bool(use_hybrid_mossm_block)
        if self.use_hybrid_mossm_block:
            self.local_branch = ASLLocalBranch(
                d_asl, expand=float(hybrid_local_expand),
                kernel_size=int(hybrid_local_kernel),
            )
            self.hybrid_fuse = nn.Conv2d(2 * d_asl, d_asl, 1, bias=True)
            nn.init.zeros_(self.hybrid_fuse.weight)
            if self.hybrid_fuse.bias is not None:
                nn.init.zeros_(self.hybrid_fuse.bias)
            with torch.no_grad():
                for c in range(d_asl):
                    # input ch [0:d_asl)=local, [d_asl:2*d_asl)=global readout
                    self.hybrid_fuse.weight[c, d_asl + c, 0, 0] = 1.0
        self.norm2 = nn.LayerNorm(d_asl)
        hidden = int(d_asl * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_asl, hidden), nn.GELU(),
            nn.Linear(hidden, d_asl),
        )

    def forward(
        self,
        x_asl: Tensor,
        t1_cond: Tensor,
        tissue_seg: Optional[Tensor] = None,
        t1_seg_probs: Optional[Tensor] = None,
    ) -> Tensor:
        # Reader-pattern integration of MoSSM (memory writer) and CMAM
        # (memory reader). MoSSM produces a state-derived memory map; CMAM
        # reads from THAT memory with T1 as the addressing key. A single
        # residual update wraps the whole memory-write-then-read step —
        # avoiding the sublayer-stacking pattern that would treat CMAM as
        # an independent attention block.
        x_norm = self.norm_asl(x_asl)
        mossm_mem = self.mossm(
            x_norm, self.norm_t1(t1_cond),
            tissue_seg=tissue_seg,
            t1_seg_probs=t1_seg_probs,
        )                                                # [B, H, W, d_asl]
        if self.use_cmam:
            # CMAM: Q from x_asl (what we want to refine), K from T1
            # (anatomical addressing oracle), V from Mamba's memory map.
            # V=ASL invariant strengthened — V is doubly-filtered ASL.
            global_readout = self.cmam(
                query_src=x_asl, key_src=t1_cond, value_src=mossm_mem,
            )                                            # [B, H, W, d_asl]
        else:
            global_readout = mossm_mem                   # plain MoSSM memory
        if self.use_hybrid_mossm_block:
            # ASL-only local branch fused with the (ASL-valued) global readout.
            # At init hybrid_fuse passes the global readout through unchanged,
            # so this reduces to `x_asl + global_readout` exactly.
            local_mem = self.local_branch(x_norm)        # [B, H, W, d_asl] ASL-only
            fused_nchw = torch.cat([
                local_mem.permute(0, 3, 1, 2).contiguous(),
                global_readout.permute(0, 3, 1, 2).contiguous(),
            ], dim=1)                                    # [B, 2*d_asl, H, W]
            fused = self.hybrid_fuse(fused_nchw).permute(0, 2, 3, 1).contiguous()
            x_asl = x_asl + fused                        # single residual
        else:
            x_asl = x_asl + global_readout               # single residual
        x_asl = x_asl + self.mlp(self.norm2(x_asl))
        return x_asl


# -----------------------------------------------------------------------------
# MoSSM hierarchical encoder — drop-in replacement for ConvEncoder2D
# -----------------------------------------------------------------------------

class MoSSMEncoder2D(nn.Module):
    """Hierarchical MoSSM encoder.

    Matches ConvEncoder2D's output interface so it can be swapped in directly
    inside ASLT1Denoiser:
        forward(x, t1_skips) -> (feat_map, vec, asl_skips)
    where asl_skips is a list of `depth` feature maps at successively
    coarser scales, identical in convention to ConvEncoder2D's skips.

    Channel layout matches ConvEncoder2D: [base_ch * 2^i] for i in 0..depth-1.

    `t1_skips` are the multi-scale T1 features (from the frozen T1 encoder).
    At stage s, the s-th t1_skip is used to modulate B and Δ in every MoSSM
    block of that stage. Spatial dimensions are assumed already matched by
    the parallel ConvEncoder2D running on the T1 input.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        t1_channels_per_scale: Optional[List[int]] = None,
        d_state: int = 16,
        n_blocks_per_scale: int = 2,
        n_directions: Union[int, Sequence[int]] = 1,
        groups: int = 8,
        use_cmam: bool = True,
        cmam_max_tokens: int = 1024,
        cmam_n_heads: int = 4,
        hw_per_scale: Optional[List[int]] = None,
        t1_gated_bd: bool = False,
        t1_gate_init: float = -4.0,
        cmam_k_rank_div: int = 1,
        # CADA (2026-05-25): per-stage activation list (length=depth). Stages
        # with cada_active_stages[s]=True replace the scalar t1_gated_bd at
        # stage s with the cap-free / zero-init-α adapter; others fall back to
        # the scalar gate (or unconditional joint BD if t1_gated_bd=False).
        cada_active_stages: Optional[List[bool]] = None,
        cada_n_groups_B: int = 4,
        cada_lr: bool = False,
        cada_lr_rank: int = 8,
        cada_lr_bound: float = 4.0,
        # Hybrid CNN-SSM (2026-06-05): ASL-only local branch per block.
        use_hybrid_mossm_block: bool = False,
        hybrid_local_expand: float = 2.0,
        hybrid_local_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.depth = depth
        # Stage-adaptive scan directions: int → same for all stages; or a
        # length-`depth` sequence, e.g. [1,1,2,2] = light raster at high-res,
        # bidirectional at low-res. See ASL_dmvae/docs/archive/v2_mossm/hybrid_mossm_stage_adaptive_implementation.md (RETIRED).
        if isinstance(n_directions, int):
            self.n_directions_per_stage = [int(n_directions)] * depth
        else:
            self.n_directions_per_stage = [int(v) for v in n_directions]
            assert len(self.n_directions_per_stage) == depth, (
                len(self.n_directions_per_stage), depth)
        for _v in self.n_directions_per_stage:
            assert _v in (1, 2, 4), _v
        self.channels = [base_ch * (2 ** i) for i in range(depth)]
        if t1_channels_per_scale is None:
            t1_channels_per_scale = list(self.channels)  # parallel ConvEncoder2D shape
        assert len(t1_channels_per_scale) == depth, (
            len(t1_channels_per_scale), depth
        )
        self.t1_channels_per_scale = list(t1_channels_per_scale)
        self.cmam_max_tokens = int(cmam_max_tokens)

        # CMAM is enabled only at stages where H*W <= cmam_max_tokens (to bound
        # attention's O(N^2) memory). Default 1024 → 32×32 and smaller scales.
        # `hw_per_scale` lets caller declare the actual stage resolutions; if
        # absent, we conservatively enable CMAM only at the deepest 1-2 stages.
        if hw_per_scale is None:
            # Conservative default: enable at last stage (bottleneck) only
            cmam_per_stage = [False] * depth
            if use_cmam:
                cmam_per_stage[-1] = True
                # also enable at depth-2 if its resolution likely fits
                if depth >= 2:
                    cmam_per_stage[-2] = True
        else:
            assert len(hw_per_scale) == depth, (len(hw_per_scale), depth)
            cmam_per_stage = [
                use_cmam and (hw_per_scale[s] * hw_per_scale[s] <= self.cmam_max_tokens)
                for s in range(depth)
            ]
        self.cmam_per_stage = list(cmam_per_stage)
        # Default CADA activation: disabled. Caller-side default in
        # ASLT1Denoiser sets all-stages when use_cada=True.
        if cada_active_stages is None:
            cada_active_stages = [False] * depth
        assert len(cada_active_stages) == depth, (
            len(cada_active_stages), depth
        )
        self.cada_active_stages = list(cada_active_stages)
        self.cada_n_groups_B = int(cada_n_groups_B)

        # Stem: in_ch -> channels[0]; no spatial change.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, self.channels[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(groups, self.channels[0]), self.channels[0]),
            nn.SiLU(),
        )

        # Per-stage MoSSM block stacks (CMAM + CADA toggled per-stage)
        self.stages = nn.ModuleList([
            nn.ModuleList([
                MoSSMBlock(
                    d_asl=self.channels[s],
                    d_t1=self.t1_channels_per_scale[s],
                    d_state=d_state,
                    n_directions=self.n_directions_per_stage[s],
                    use_cmam=cmam_per_stage[s],
                    cmam_n_heads=cmam_n_heads,
                    # The scalar gate is disabled at any stage where the CADA
                    # gate is enabled — the per-pixel CADA gate subsumes it.
                    t1_gated_bd=(t1_gated_bd
                                 and not self.cada_active_stages[s]),
                    t1_gate_init=t1_gate_init,
                    cmam_k_rank_div=cmam_k_rank_div,
                    use_cada=self.cada_active_stages[s],
                    cada_n_groups_B=int(cada_n_groups_B),
                    cada_lr=bool(cada_lr),
                    cada_lr_rank=int(cada_lr_rank),
                    cada_lr_bound=float(cada_lr_bound),
                    use_hybrid_mossm_block=bool(use_hybrid_mossm_block),
                    hybrid_local_expand=float(hybrid_local_expand),
                    hybrid_local_kernel=int(hybrid_local_kernel),
                )
                for _ in range(n_blocks_per_scale)
            ])
            for s in range(depth)
        ])

        # Inter-stage 2× downsample with channel growth (matches ConvEncoder2D depth)
        self.downs = nn.ModuleList()
        for s in range(depth - 1):
            self.downs.append(nn.Sequential(
                nn.Conv2d(self.channels[s], self.channels[s + 1],
                          kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(min(groups, self.channels[s + 1]), self.channels[s + 1]),
                nn.SiLU(),
            ))

        self.out_ch = self.channels[-1]

    def forward(
        self,
        x: Tensor,
        t1_skips: List[Tensor],
        tissue_seg: Optional[Tensor] = None,
        t1_seg_probs: Optional[Tensor] = None,
    ):
        """
        Args:
            x: ASL aggregate [B, 1, H, W]
            t1_skips: list of [B, C_s, H_s, W_s] for s in 0..depth-1
            tissue_seg: optional [B, n_class, H_full, W_full] — TABS scan
                        reorder signal. Adaptive-pooled inside MoSSM2D.
            t1_seg_probs: unused (legacy parameter, kept for signature compat).
        """
        assert len(t1_skips) == self.depth, (len(t1_skips), self.depth)
        skips: List[Tensor] = []
        x = self.stem(x)                                          # [B, C0, H, W]
        for s in range(self.depth):
            t1_s = t1_skips[s]                                    # [B, C_t1_s, H_s, W_s]
            assert t1_s.shape[-2:] == x.shape[-2:], (
                "MoSSM stage", s, "x spatial", x.shape[-2:],
                "t1 spatial", t1_s.shape[-2:],
            )
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
            t1_nhwc = t1_s.permute(0, 2, 3, 1).contiguous()
            for blk in self.stages[s]:
                x_nhwc = blk(
                    x_nhwc, t1_nhwc,
                    tissue_seg=tissue_seg,
                    t1_seg_probs=t1_seg_probs,
                )
            x = x_nhwc.permute(0, 3, 1, 2).contiguous()           # back to NCHW
            skips.append(x)
            if s < self.depth - 1:
                x = self.downs[s](x)
        feat = x
        # `vec` (global-avg-pool) is part of the legacy ConvEncoder2D 3-tuple
        # interface but is discarded by every consumer (ASLT1Denoiser unpacks it
        # as `_`); return None instead of computing a dead pooled vector (#3).
        return feat, None, skips


# --- backward-compat aliases (old class names -> new paper-aligned names) -----
# Renamed 2026-06-27 for paper consistency (see ASL_dmvae/docs/archive/v2_mossm/cig_net.md naming table).
# Submodule attribute names (self.mossm / self.cmam / self.cada / self.local_branch)
# and ALL CLI flags (--cada_lr, --use_cmam, --mossm_*, ...) are UNCHANGED, so
# existing checkpoints (incl. the frozen stage1 T1 prior) and every slurm/local
# script load unaffected; these aliases keep external imports resolving.
CADA = DynamicsAdapter                  # paper: LRDA (low-rank mode)
MoSSM2D = AnatomyConditionedSSM2D       # paper: AC-SSM Unit (global SSM branch)
CMAM = AnatomyKeyedMemoryReader         # paper: AKMR
ASLLocalBranch = ASLLocalContextUnit    # paper: LCU
MoSSMBlock = AnatomyGuidedSSMBlock      # paper: AG-SSM Block
