"""PVC residual decomposition head (opt-in, --pvc_residual).

Design (d), architecture level. The reconstruction is assembled as

    ŷ = E + w ⊙ r̂          (no_wgate ablation: ŷ = E + r̂)

where
  • E = P · m̄   is a CLOSED-FORM image-space partial-volume (PVC) base: a per-image
    ridge (shrinkage) fit of a per-tissue ΔM level m̄∈R^C to the input aggregate,
    softly mixed back by the PV composition P (GM/WM/CSF/BG). E is piecewise-tissue-
    constant (no high frequency) — the "tissue-explained" perfusion (Asllani 2008,
    Chappell 2011). This is PV's genuine, non-affine job (per-voxel un-mixing, not a
    per-tissue global rescale).
  • r̂ = the ASL decoder output, which the N2N loss on ŷ trains to be the RESIDUAL
    (deviation from the tissue expectation — where anomalies / lesions live). The
    decoder is unchanged and T1-free; it never sees E (V=ASL at the module boundary).
  • w = a cross-frame REPRODUCIBILITY map (high where the ΔM frames agree, low where a
    single frame disagrees = motion/label noise). It keeps reproducible deviations and
    suppresses single-frame noise ⇒ where the ASL is unreliable the output falls back
    to the tissue prior E. This is the lever that counteracts the near-MMSE over-
    smoothing of small reproducible anomalies (E0.3 retention).

V=ASL: m̄ is estimated FROM the ASL aggregate (P only forms the linear-mixture design
matrix), so E ∈ span(ASL values) — no T1 value/content term. w∈(0,1] only attenuates
an ASL quantity. §3 note (knowing, opt-in relaxation): E carries the PV TISSUE
BOUNDARIES into the output (piecewise-constant, no high-freq), a mismatch-T1-monitored
relaxation of "no T1-shaped structure". Off by default ⇒ baselines byte-unaffected.
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _norm01(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Per-sample spatial standardisation (zero-mean/unit-std) — keeps only the
    RELATIVE spatial pattern of the dispersion so a0/bw set a fixed global operating
    point (identical scope + rationale as ec_guidance._norm01)."""
    B = x.shape[0]
    flat = x.reshape(B, -1)
    mu = flat.mean(dim=1, keepdim=True)
    sd = flat.std(dim=1, keepdim=True) + eps
    return ((flat - mu) / sd).reshape_as(x)


def pvc_base_fixed(agg1: Tensor, pv: Tensor, tau: float = 0.5) -> Tensor:
    """Closed-form image-space PVC base E = P·m̄ with a FIXED ridge (no learnable
    params), for the loss-level DW-N2N path: m̄ = (PᵀP + τI)⁻¹ Pᵀx, E = P·m̄.
    E is used ONLY to compute the deviation map d=|agg−E| that weights the loss; it
    is never added to the output. agg1 [B,1,H,W], pv [B,C,H,W] soft composition."""
    B, _, H, W = agg1.shape
    C = pv.shape[1]
    orig_dtype = agg1.dtype
    with torch.autocast(device_type=agg1.device.type, enabled=False):
        x = agg1.reshape(B, 1, H * W).transpose(1, 2).float()               # [B,N,1]
        P = pv.clamp_min(0.0).reshape(B, C, H * W).transpose(1, 2).float()   # [B,N,C]
        eye = torch.eye(C, device=agg1.device, dtype=torch.float32)
        A = torch.einsum("bni,bnj->bij", P, P) + (tau + 1e-4) * eye.unsqueeze(0)  # [B,C,C]
        b = torch.einsum("bni,bnd->bid", P, x)                              # [B,C,1]
        mbar = torch.linalg.solve(A, b)                                     # [B,C,1]
        E = torch.einsum("bnc,bcd->bnd", P, mbar).transpose(1, 2).reshape(B, 1, H, W)
    return E.to(orig_dtype)


class PVCResidualHead(nn.Module):
    def __init__(self, pv_ch: int = 4) -> None:
        super().__init__()
        self.pv_ch = int(pv_ch)
        # Ridge shrinkage: τ = softplus(ρ) (ρ init 0 ⇒ τ≈0.69, well-conditioned solve).
        self.rho = nn.Parameter(torch.zeros(self.pv_ch))
        # Prior per-tissue ΔM level m0 (ASL-scale; init 0 ⇒ shrink toward 0).
        self.m0 = nn.Parameter(torch.zeros(self.pv_ch))
        # Reproducibility map w = σ(aw0 − softplus(bw)·norm01(dispersion)). Same near-
        # identity init as ec_guidance (aw0=4, bw=-4 ⇒ slope≈0.018 ⇒ w≈σ(4)≈0.98
        # near-uniform at init ⇒ ŷ≈E+r̂ before the dispersion structure is learned).
        self.aw0 = nn.Parameter(torch.tensor(4.0))
        self.bw = nn.Parameter(torch.tensor(-4.0))

    def repro(self, set_a: Tensor, lengths: Optional[Tensor] = None,
              mask: Optional[Tensor] = None) -> Tensor:
        """[B,T,C,H,W] → reproducibility map w [B,1,H,W] (high = frames agree)."""
        B, T, C, H, W = set_a.shape
        if mask is not None:
            valid = (~mask.bool()).float()
        elif lengths is not None:
            valid = (torch.arange(T, device=set_a.device)[None] < lengths[:, None]).float()
        else:
            valid = torch.ones(B, T, device=set_a.device)
        v = valid.view(B, T, 1, 1, 1)
        n = v.sum(1).clamp_min(1.0)                               # [B,1,1,1]
        xbar = (set_a * v).sum(1) / n                             # [B,C,H,W] consensus
        dev = ((set_a - xbar.unsqueeze(1)).abs() * v).sum(1) / n  # [B,C,H,W]
        D = dev.mean(1, keepdim=True)                             # [B,1,H,W] cross-frame dispersion
        return torch.sigmoid(self.aw0 - F.softplus(self.bw) * _norm01(D))

    def pvc_base(self, agg1: Tensor, pv: Tensor) -> Tensor:
        """Closed-form PVC base E = P·m̄, m̄ = (PᵀP + diag τ)⁻¹(Pᵀx + τ⊙m0).
        agg1 [B,1,H,W] (center-slice ΔM aggregate), pv [B,C,H,W] soft composition."""
        B, _, H, W = agg1.shape
        C = pv.shape[1]
        orig_dtype = agg1.dtype
        # fp32 for the SPD solve (fp16-fragile under autocast — matches _c_sem_bayes).
        with torch.autocast(device_type=agg1.device.type, enabled=False):
            x = agg1.reshape(B, 1, H * W).transpose(1, 2).float()          # [B,N,1]
            P = pv.clamp_min(0.0).reshape(B, C, H * W).transpose(1, 2).float()  # [B,N,C]
            tau = F.softplus(self.rho.float())                             # [C]
            eye = torch.eye(C, device=agg1.device, dtype=torch.float32)
            A = (torch.einsum("bni,bnj->bij", P, P)
                 + torch.diag_embed(tau).unsqueeze(0) + 1e-4 * eye)         # [B,C,C] SPD
            b = (torch.einsum("bni,bnd->bid", P, x)
                 + (tau * self.m0.float()).view(1, C, 1))                   # [B,C,1]
            mbar = torch.linalg.solve(A, b)                                 # [B,C,1] per-tissue ΔM level
            E = torch.einsum("bnc,bcd->bnd", P, mbar)                       # [B,N,1]
            E = E.transpose(1, 2).reshape(B, 1, H, W)
        return E.to(orig_dtype)

    def forward(self, agg1: Tensor, pv: Tensor, recon: Tensor, set_a: Tensor,
                lengths: Optional[Tensor] = None, mask: Optional[Tensor] = None,
                no_wgate: bool = False) -> Dict[str, Tensor]:
        E = self.pvc_base(agg1, pv)                                # [B,1,H,W] tissue-explained base
        if no_wgate:
            w = torch.ones_like(E)
        else:
            w = self.repro(set_a, lengths, mask)                   # [B,1,H,W] reproducibility
        yhat = E + w * recon                                       # recon ← residual (trained via N2N on ŷ)
        return {"recon": yhat, "base": E, "w": w}
