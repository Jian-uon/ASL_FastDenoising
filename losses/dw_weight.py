# -*- coding: utf-8 -*-
"""DW-N2N anomaly-aware deviation weighting — reference implementation.

Builds the per-pixel loss-weight map ``d ∈ [0,1]`` ("candidate reproducible
anomaly") that DW-N2N feeds into ``ASLN2NLoss`` via ``dev_weight`` (hooks
``up_dev`` / ``relax_smooth`` already exist in ``losses/asl_n2n_loss.py``).

Design doc: ASL_dmvae/docs/dwn2n_design.md.  Key properties:

  * CLOSED-FORM, ZERO learnable parameters, fully ``@torch.no_grad()`` —
    cannot collapse, cannot be gamed by the recon loss (the RGSF/Var_T lesson).
  * Derived from set_a + the FROZEN PV only ⇒ independent of set_b ⇒ the
    per-pixel N2N minimiser is unchanged (unbiasedness preserved); only the
    loss geometry (which pixels matter more) changes.
  * Absolute, pre-registered thresholds (z0/z1) — deliberately NOT per-batch
    normalised (the Norm01 lesson: per-sample standardisation throws away the
    absolute magnitude that distinguishes "quiet slice" from "anomalous slice").
  * Acts at FULL image resolution (unlike the γ gate's H/4 work grid).

Pipeline (each step has one job):
  1. Tissue expectation  E = P·m̄,  m̄ = (PᵀP + diag τ)⁻¹ (Pᵀx + τ⊙m0)
       — shrinkage ridge toward dataset-level per-tissue prior means m0;
         τ_c = prior evidence in PIXEL-EQUIVALENTS (data-rich classes are
         untouched, near-empty classes fall back to the prior).
  2. Noise calibration   z = |agg − E| / √(Var[mean] + floor)
       — Var[mean] from unbiased frame-to-frame variance / n; makes z
         comparable across frame budgets (z≈1 under pure noise).
  3. Reproducibility gate w_rep from even/odd half-mean agreement
       — a large z driven by a single bad frame is an artifact, not an
         anomaly; halves disagreeing ⇒ gate down.
  4. Fixed squash        d = clamp((z̃ − z0)/(z1 − z0), 0, 1) · w_rep · brain
       — z̃ is 3×3-box-smoothed z (kills single-pixel flicker).

Monitors returned with the map (log to TB; whitelist keys: ``dw_mean``,
``dw_active_frac``, ``dw_boundary_frac``):
  * boundary-band mass fraction — if most weight sits on GM/WM boundaries the
    map has degenerated into a covert boundary sharpener (abort criterion);
  * mean weight and active fraction — sanity that the map is sparse.

The other two protocol monitors (mismatched-T1 leak re-test on the DW-trained
arm; uMSE non-regression) are eval-side — see ASL_dmvae/docs/dwn2n_design.md §5.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from .asl_n2n_loss import estimate_input_noise_var
except Exception:  # pragma: no cover
    from asl_n2n_loss import estimate_input_noise_var  # type: ignore


# ---------------------------------------------------------------------------
# Step 1 — shrinkage-ridge tissue expectation E = P·m̄  (extends pvc_base_fixed
# with a dataset-level prior mean m0; fp32 + autocast off for the SPD solve).
# ---------------------------------------------------------------------------

def pvc_expectation(agg1: Tensor, pv: Tensor,
                    m0: Optional[Tensor] = None,
                    tau: float = 64.0) -> Tensor:
    """Closed-form tissue-expectation image E [B,1,H,W].

    agg1 [B,1,H,W] center-slice ΔM aggregate; pv [B,C,H,W] frozen softmax
    composition. m̄ = (PᵀP + diag τ)⁻¹ (Pᵀx + τ⊙m0). ``tau`` is interpreted as
    prior evidence in pixel-equivalents (default 64 ≈ an 8×8 patch): classes
    with ≫τ pixels of mass are data-driven, near-empty classes shrink to m0.
    m0=None ⇒ m0=0 with a small τ (0.5) — the legacy pvc_base_fixed behaviour;
    run scripts/estimate_dw_prior.py once to get the real m0 (see design doc).
    """
    B, _, H, W = agg1.shape
    C = pv.shape[1]
    orig_dtype = agg1.dtype
    tau_eff = float(tau) if m0 is not None else 0.5
    with torch.autocast(device_type=agg1.device.type, enabled=False):
        x = agg1.reshape(B, 1, H * W).transpose(1, 2).float()               # [B,N,1]
        P = pv.clamp_min(0.0).reshape(B, C, H * W).transpose(1, 2).float()  # [B,N,C]
        eye = torch.eye(C, device=agg1.device, dtype=torch.float32)
        A = torch.einsum("bni,bnj->bij", P, P) + (tau_eff + 1e-4) * eye     # [B,C,C]
        b = torch.einsum("bni,bnd->bid", P, x)                              # [B,C,1]
        if m0 is not None:
            b = b + tau_eff * m0.to(b).float().view(1, C, 1)
        mbar = torch.linalg.solve(A, b)                                     # [B,C,1]
        E = torch.einsum("bnc,bcd->bnd", P, mbar).transpose(1, 2).reshape(B, 1, H, W)
    return E.to(orig_dtype)


# ---------------------------------------------------------------------------
# Step 3 helper — deterministic even/odd half-means (mirrors the model's
# _even_odd_halfmeans; local copy so the loss package has no model import).
# ---------------------------------------------------------------------------

def _halfmeans(frames: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """frames [B,T,1,H,W], valid [B,T] → m1,m2 [B,1,H,W], n1,n2 [B,1,1,1], ok [B,1,1,1]."""
    B, T = frames.shape[0], frames.shape[1]
    idx = torch.arange(T, device=frames.device)
    even = valid * (idx % 2 == 0).float().view(1, T)
    odd = valid * (idx % 2 == 1).float().view(1, T)

    def wmean(w: Tensor) -> Tuple[Tensor, Tensor]:
        n = w.sum(1).view(B, 1, 1, 1)
        m = (frames * w.view(B, T, 1, 1, 1)).sum(1) / n.clamp_min(1e-6)
        return m, n

    m1, n1 = wmean(even)
    m2, n2 = wmean(odd)
    ok = ((n1 >= 0.5) & (n2 >= 0.5)).float()
    return m1, m2, n1, n2, ok


def _valid_from(frames: Tensor, lengths: Optional[Tensor],
                mask: Optional[Tensor]) -> Tensor:
    B, T = frames.shape[0], frames.shape[1]
    if mask is not None:
        return (~mask.bool()).float()
    if lengths is not None:
        return (torch.arange(T, device=frames.device)[None] < lengths[:, None]).float()
    return torch.ones(B, T, device=frames.device)


# ---------------------------------------------------------------------------
# The weight map
# ---------------------------------------------------------------------------

@torch.no_grad()
def dw_weight_map(
    set_a_center: Tensor,            # [B,T,1,H,W] CENTER-slice frames (2.5-D: pass set_a[:, :, c0:c1])
    agg_center: Tensor,              # [B,1,H,W]   model aggregate, center slice (outputs["agg"])
    pv: Tensor,                      # [B,C,H,W]   softmax(frozen t1_seg logits)
    brain_mask: Tensor,              # [B,1,H,W]   ORIGINAL-subject t1>thr mask
    lengths: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    m0: Optional[Tensor] = None,     # [C] dataset-level per-tissue prior means (None ⇒ legacy fallback)
    tau: float = 64.0,
    z0: float = 1.0,                 # noise floor: z ≤ z0 → weight 0   (PRE-REGISTERED, absolute)
    z1: float = 4.0,                 # saturation:  z ≥ z1 → weight 1
    rep_z0: float = 1.5,             # half-agreement: gate starts closing
    rep_z1: float = 4.0,             #                 gate fully closed
    var_floor: float = 1e-4,
) -> Tuple[Tensor, Dict[str, float]]:
    """Return (d [B,1,H,W] ∈ [0,1], monitor stats). Zero learnable params.

    d is high only where ALL THREE hold: deviates from the tissue expectation
    (step 1), beyond the frame-noise floor (step 2), and reproducible across
    the even/odd frame halves (step 3). Everything derives from set_a + frozen
    PV ⇒ ⊥ set_b ⇒ N2N-unbiased.
    """
    B, T = set_a_center.shape[0], set_a_center.shape[1]
    valid = _valid_from(set_a_center, lengths, mask)
    bm = brain_mask.float()

    # -- step 1: tissue expectation
    E = pvc_expectation(agg_center, pv, m0=m0, tau=tau)                     # [B,1,H,W]

    # -- step 2: noise-calibrated deviation z
    len_eff = valid.sum(1).to(torch.long)
    var_mean = estimate_input_noise_var(set_a_center, len_a=len_eff)        # Var[mean(set_a)]
    z = (agg_center - E).abs() / (var_mean.clamp_min(var_floor)).sqrt()     # [B,1,H,W]

    # -- step 3: reproducibility gate (even/odd half agreement)
    m1, m2, n1, n2, ok = _halfmeans(set_a_center, valid)
    var_frame = var_mean * valid.sum(1).view(B, 1, 1, 1).clamp_min(1.0)     # per-frame variance
    sd_diff = (var_frame * (1.0 / n1.clamp_min(1.0) + 1.0 / n2.clamp_min(1.0))).clamp_min(var_floor).sqrt()
    z_rep = (m1 - m2).abs() / sd_diff
    w_rep = ((rep_z1 - z_rep) / max(rep_z1 - rep_z0, 1e-6)).clamp(0.0, 1.0)
    w_rep = ok * w_rep + (1.0 - ok)                                         # degenerate halves → ungated

    # -- step 4: smooth + fixed absolute squash
    z_s = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)               # kill 1-px flicker
    d = ((z_s - z0) / max(z1 - z0, 1e-6)).clamp(0.0, 1.0) * w_rep * bm

    # -- monitors (TB keys: dw_mean / dw_active_frac / dw_boundary_frac)
    nb = bm.sum().clamp_min(1.0)
    gm, wm = pv[:, 0:1], pv[:, 1:2]                                         # channel order GM,WM,CSF,BG (load-bearing)
    band = ((gm * wm).clamp_min(0.0).sqrt() > 0.15).float() * bm            # GM/WM boundary band
    mass = d.sum().clamp_min(1e-6)
    stats = {
        "dw_mean": float((d.sum() / nb).item()),
        "dw_active_frac": float((((d > 0.05).float() * bm).sum() / nb).item()),
        "dw_boundary_frac": float(((d * band).sum() / mass).item()),
    }
    return d, stats


# ---------------------------------------------------------------------------
# One-time offline prior estimation (m0). Run over the TRAIN split only.
# ---------------------------------------------------------------------------

@torch.no_grad()
def accumulate_m0(agg_center: Tensor, pv: Tensor,
                  state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
    """Streaming PV-weighted per-tissue mean of the aggregate:
        m0_c = Σ_i p_ic · x_i / Σ_i p_ic   accumulated over batches.
    Call per train batch, then ``finalize_m0(state)``. Uses the SAME per-sample
    affine-normalised space the loss runs in (agg from the standard pack)."""
    x = agg_center.float()
    P = pv.float().clamp_min(0.0)
    num = torch.einsum("bchw,bdhw->c", P, x)                                # [C]
    den = P.sum(dim=(0, 2, 3))                                              # [C]
    if state is None:
        return {"num": num, "den": den}
    return {"num": state["num"] + num, "den": state["den"] + den}


def finalize_m0(state: Dict[str, Tensor]) -> Tensor:
    return state["num"] / state["den"].clamp_min(1.0)                       # [C]


if __name__ == "__main__":  # synthetic smoke: 3 classes must separate (design doc §1)
    torch.manual_seed(0)
    B, T, H, W, C = 2, 6, 32, 32, 4
    pv = torch.softmax(torch.randn(B, C, H, W), dim=1)
    setA = torch.randn(B, T, 1, H, W) * 0.1                 # background noise
    setA[:, :, :, 10:14, 10:14] += 2.0                      # REPRODUCIBLE anomaly (every frame)
    setA[:, 0, :, 20:24, 20:24] += 3.0                      # SINGLE-FRAME artifact (frame 0 only)
    agg = setA.mean(1)
    brain = torch.ones(B, 1, H, W)
    d, stats = dw_weight_map(setA, agg, pv, brain, lengths=torch.tensor([T, T]))
    an = d[:, :, 10:14, 10:14].mean().item()
    ar = d[:, :, 20:24, 20:24].mean().item()
    bg = d[:, :, 0:4, 0:4].mean().item()
    print(f"d range [{d.min():.3f},{d.max():.3f}]  NaN={bool(torch.isnan(d).any())}")
    print(f"  reproducible-anomaly d ≈ {an:.3f}  (should be high)")
    print(f"  single-frame-artifact d ≈ {ar:.3f}  (should be ~0, gated by reproducibility)")
    print(f"  background            d ≈ {bg:.3f}  (should be ~0, below noise floor)")
    print(f"  monitors: {{{', '.join(f'{k}={v:.3f}' for k, v in stats.items())}}}")
    assert an > 0.5 and ar < 0.2 and bg < 0.2 and not torch.isnan(d).any(), "smoke FAILED"
    st = None
    for _ in range(3):
        st = accumulate_m0(agg, pv, st)
    print(f"  m0 (finalize) = {[round(float(v), 4) for v in finalize_m0(st)]}")
    print("smoke OK")
