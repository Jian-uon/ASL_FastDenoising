# -*- coding: utf-8 -*-
"""Post-hoc ΔM → CBF quantification (ASL consensus single-compartment model).

This module is a DETERMINISTIC, NETWORK-EXTERNAL physical transform applied to
the denoised ΔM *after* it has left the denoiser. It must NEVER be imported by
the training runner or the model — it consumes only an ASL-derived ΔM volume, an
external M0 reference, and scalar sequence parameters, so the project's
content-isolation invariants (V=ASL, decoder T1-free, self-supervised /
no-clean-GT) are untouched. CBF never enters checkpoint selection.

Sign convention (project-pinned, D1 2026-07-22): ΔM = control − label, i.e.
POSITIVE perfusion (GM ΔM > 0). The quantifier assumes this convention; a
negative GM ΔM signals a data/sign error and is surfaced via QC (see
`qc_cbf_ranges`), never silently abs()'d. Physically CBF ≥ 0, so voxels with
negative ΔM are clamped to 0 (with `clamp_negative=True`) rather than sign-flipped.

Single-PLD PCASL CBF — Alsop et al. 2015, MRM 73:102-116, Eq. (1):

                  6000 · λ · ΔM · exp(PLD / T1b)
    CBF  =  ──────────────────────────────────────────────    [mL/100g/min]
              2 · α · T1b · M0 · ( 1 − exp(−τ / T1b) )

Symbols (units): λ blood–brain partition coefficient [mL/g]; PLD post-labeling
delay [s]; τ label (bolus) duration [s]; T1b longitudinal relaxation of arterial
blood [s]; α NET labeling efficiency = α_label · α_BS (see `net_alpha` — the
background-suppression factor is applied ONCE as a net multiplier, NOT per pulse);
M0 tissue equilibrium magnetisation on the SAME intensity scale as ΔM; 6000 =
60 s/min · 100 g. Note T1b appears THREE times (numerator exp, the standalone
denominator factor, and the bolus term) — all three are kept.

For 3D single-shot readouts PLD is spatially uniform (`slice_timing=None`). For
2D multi-slice readouts pass a per-slice delay so PLD_eff(z) = PLD + slice_timing(z).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

__all__ = ["net_alpha", "denorm_deltam", "dm_to_cbf", "qc_cbf_ranges"]

Number = Union[float, np.ndarray]


def net_alpha(alpha_label: float, alpha_bs_per_pulse: float, n_bs_pulses: int) -> float:
    """Net labeling efficiency α = α_label · α_BS^(n_bs_pulses).

    Background suppression attenuates the labeled signal ONCE per inversion pulse,
    so the net factor is a power, not a per-voxel iterated product. Typical values:
    α_label ≈ 0.85 (PCASL), α_bs_per_pulse ≈ 0.93; with 2–4 BS pulses this gives a
    net α_BS of ~0.86–0.75. (Guards against the "×0.75 per pulse" error, which
    would over-suppress and inflate CBF ~2×.)
    """
    if not (0.0 < alpha_label <= 1.0):
        raise ValueError(f"alpha_label out of range: {alpha_label}")
    if not (0.0 < alpha_bs_per_pulse <= 1.0):
        raise ValueError(f"alpha_bs_per_pulse out of range: {alpha_bs_per_pulse}")
    if n_bs_pulses < 0:
        raise ValueError(f"n_bs_pulses must be >= 0: {n_bs_pulses}")
    return float(alpha_label) * float(alpha_bs_per_pulse) ** int(n_bs_pulses)


def denorm_deltam(pwi_norm: np.ndarray, lo: float, scale: float) -> np.ndarray:
    """Invert the set-A robust-affine normalisation to recover raw-intensity ΔM.

    The inference/training normalisation is x_norm = (x − lo) / scale
    (`dataio.data_classes._affine_from_frames`). Because it contains a subtractive
    offset `lo`, it is NOT a pure scaling and must be inverted before CBF: the
    quantifier needs ΔM on the acquisition intensity scale, matching M0.
    """
    return np.asarray(pwi_norm, dtype=np.float64) * float(scale) + float(lo)


def dm_to_cbf(
    dm_raw: np.ndarray,
    m0: np.ndarray,
    *,
    pld: float,
    ld: float,
    alpha: float,
    t1_blood: float,
    lam: float = 0.90,
    m0_scale: float = 1.0,
    slice_timing: Optional[Number] = None,
    mask: Optional[np.ndarray] = None,
    m0_floor: float = 0.0,
    eps: float = 1e-6,
    clamp_negative: bool = True,
) -> dict:
    """Single-PLD PCASL ΔM → CBF (Alsop 2015, Eq. 1).

    Parameters
    ----------
    dm_raw       : [H,W,Z] RAW (de-normalised, acquisition-intensity) ΔM = control−label.
    m0           : [H,W,Z] tissue M0 / PD reference on the SAME intensity scale as dm_raw.
    pld, ld      : post-labeling delay and label (bolus) duration [s].
    alpha        : NET labeling efficiency α_label·α_BS (see `net_alpha`).
    t1_blood     : arterial blood T1 [s] (field-dependent: ~1.65 at 3T, ~2.2 at 7T).
    lam          : blood–brain partition coefficient [mL/g] (default 0.90).
    m0_scale     : multiplicative rescale to put M0 on the ΔM scale (vendor gain / TR
                   correction handled upstream; default 1.0).
    slice_timing : None for 3D readouts; else per-slice delay added to PLD. Scalar,
                   a length-Z 1-D array, or an array broadcastable to dm_raw.
    mask         : optional [H,W,Z] brain mask; CBF is zeroed outside it.
    m0_floor     : voxels with m0*m0_scale <= m0_floor are treated as invalid (CBF=0).
    clamp_negative : clamp physically-impossible negative CBF to 0 (default True).

    Returns
    -------
    dict with:
      "cbf" : [H,W,Z] float64 CBF in mL/100g/min (0 where invalid/outside mask).
      "qc"  : dict of {neg_frac, m0_invalid_frac, n_valid, pld_eff_min/max}.
    """
    dm = np.asarray(dm_raw, dtype=np.float64)
    m0e = np.asarray(m0, dtype=np.float64) * float(m0_scale)
    if dm.shape != m0e.shape:
        raise ValueError(f"dm_raw {dm.shape} and m0 {m0e.shape} must have the same shape")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha (net) out of range: {alpha}")
    if t1_blood <= 0 or ld <= 0 or pld < 0 or lam <= 0:
        raise ValueError("t1_blood/ld/lam must be > 0 and pld >= 0")

    # Effective per-voxel PLD (3D: uniform; 2D: + slice timing along z).
    if slice_timing is None:
        pld_eff = np.float64(pld)
    else:
        st = np.asarray(slice_timing, dtype=np.float64)
        if st.ndim == 1 and st.shape[0] == dm.shape[-1]:
            st = st.reshape((1,) * (dm.ndim - 1) + (dm.shape[-1],))
        pld_eff = pld + st

    bolus = 1.0 - np.exp(-float(ld) / float(t1_blood))          # (1 − e^{−τ/T1b})
    numer = 6000.0 * float(lam) * dm * np.exp(pld_eff / float(t1_blood))
    denom = 2.0 * float(alpha) * float(t1_blood) * np.maximum(m0e, eps) * bolus
    cbf = numer / denom

    valid = m0e > float(m0_floor)
    if mask is not None:
        valid = valid & (np.asarray(mask) > 0.5)

    neg_frac = float((cbf[valid] < 0).mean()) if valid.any() else float("nan")
    if clamp_negative:
        cbf = np.clip(cbf, 0.0, None)
    cbf = np.where(valid, cbf, 0.0)

    qc = {
        "neg_frac": neg_frac,                                   # ΔM/CBF sign health inside valid region
        "m0_invalid_frac": float((~(m0e > float(m0_floor))).mean()),
        "n_valid": int(valid.sum()),
        "pld_eff_min": float(np.min(pld_eff)) if np.ndim(pld_eff) else float(pld_eff),
        "pld_eff_max": float(np.max(pld_eff)) if np.ndim(pld_eff) else float(pld_eff),
    }
    return {"cbf": cbf, "qc": qc}


# Physiological reference ranges for a healthy adult at rest (single-compartment,
# GM-dominant). Used ONLY as a self-validation gate for the quantifier + M0
# plumbing (Alsop 2015 / Zhang 2018 typical values) — never for model selection.
_GM_CBF_RANGE = (40.0, 70.0)      # mL/100g/min
_WM_CBF_RANGE = (20.0, 30.0)
_GMWM_RATIO_RANGE = (2.5, 3.5)


def qc_cbf_ranges(cbf: np.ndarray, gm_mask: np.ndarray, wm_mask: np.ndarray) -> dict:
    """Report GM/WM mean CBF and GM/WM ratio against healthy-adult resting ranges.

    A hard `pass` requires all three in range; use it to catch a wrong constant,
    an M0-convention bug, or a sign flip before trusting any downstream metric.
    """
    gm = np.asarray(gm_mask) > 0.5
    wm = np.asarray(wm_mask) > 0.5
    gm_mean = float(cbf[gm].mean()) if gm.any() else float("nan")
    wm_mean = float(cbf[wm].mean()) if wm.any() else float("nan")
    ratio = gm_mean / wm_mean if wm_mean not in (0.0, float("nan")) and wm_mean != 0 else float("nan")
    ok_gm = _GM_CBF_RANGE[0] <= gm_mean <= _GM_CBF_RANGE[1]
    ok_wm = _WM_CBF_RANGE[0] <= wm_mean <= _WM_CBF_RANGE[1]
    ok_ratio = _GMWM_RATIO_RANGE[0] <= ratio <= _GMWM_RATIO_RANGE[1]
    return {
        "gm_cbf": gm_mean, "wm_cbf": wm_mean, "gm_wm_ratio": ratio,
        "gm_in_range": bool(ok_gm), "wm_in_range": bool(ok_wm), "ratio_in_range": bool(ok_ratio),
        "pass": bool(ok_gm and ok_wm and ok_ratio),
    }
