# -*- coding: utf-8 -*-
"""Unit tests for utils/cbf.py — the post-hoc ΔM → CBF quantifier.

Gate G1 (plan §4.4 / §7 P0): synthesize ΔM analytically from a KNOWN CBF via the
algebraic inverse of Alsop-2015 Eq. 1, feed it back through dm_to_cbf, and require
recovery of the known CBF to <1e-6 relative. The inverse is written out here with
explicit constants so a fat-fingered constant in cbf.py (e.g. 6000→600) is caught.

Runnable either way:
    python tests/test_cbf.py          # plain asserts, prints OK
    pytest tests/test_cbf.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.cbf import net_alpha, denorm_deltam, dm_to_cbf, qc_cbf_ranges  # noqa: E402


# --- reference (independent) forward model: CBF -> ΔM ----------------------
def _cbf_to_dm(cbf, m0, *, pld, ld, alpha, t1_blood, lam=0.90, slice_timing=0.0):
    """Algebraic inverse of Alsop-2015 Eq.1, written out with explicit constants."""
    bolus = 1.0 - np.exp(-ld / t1_blood)
    pld_eff = pld + slice_timing
    denom_wo_m0 = 2.0 * alpha * t1_blood * bolus
    numer_const = 6000.0 * lam * np.exp(pld_eff / t1_blood)
    return cbf * (denom_wo_m0 * m0) / numer_const


P = dict(pld=2.0, ld=1.8, alpha=0.85, t1_blood=1.65, lam=0.90)   # 3T PCASL


def test_roundtrip_constant():
    cbf_true = np.full((8, 8, 5), 60.0)
    m0 = np.full((8, 8, 5), 1000.0)
    dm = _cbf_to_dm(cbf_true, m0, **P)
    out = dm_to_cbf(dm, m0, **P)
    assert np.allclose(out["cbf"], cbf_true, rtol=1e-6, atol=1e-6)
    assert out["qc"]["neg_frac"] == 0.0


def test_roundtrip_spatially_varying():
    rng = np.random.default_rng(0)
    cbf_true = rng.uniform(15.0, 80.0, size=(10, 12, 6))
    m0 = rng.uniform(700.0, 1300.0, size=(10, 12, 6))
    dm = _cbf_to_dm(cbf_true, m0, **P)
    out = dm_to_cbf(dm, m0, **P)
    assert np.allclose(out["cbf"], cbf_true, rtol=1e-6, atol=1e-6)


def test_slice_timing_2d_readout():
    Z = 6
    cbf_true = np.full((5, 5, Z), 50.0)
    m0 = np.full((5, 5, Z), 1000.0)
    st = np.arange(Z, dtype=float) * 0.04                        # 40 ms/slice
    dm = np.stack([_cbf_to_dm(cbf_true[..., z], m0[..., z], slice_timing=st[z], **P)
                   for z in range(Z)], axis=-1)
    out = dm_to_cbf(dm, m0, slice_timing=st, **P)
    assert np.allclose(out["cbf"], cbf_true, rtol=1e-6, atol=1e-6)
    assert out["qc"]["pld_eff_max"] > out["qc"]["pld_eff_min"]


def test_net_alpha_is_power_not_per_pulse():
    # 4 BS pulses at 0.93/pulse -> net ~0.75; NOT 0.75 per pulse.
    a = net_alpha(0.85, 0.93, 4)
    assert abs(a - 0.85 * 0.93 ** 4) < 1e-12
    assert 0.60 < a < 0.66                                       # ~0.636


def test_negative_dm_is_flagged_and_clamped():
    m0 = np.full((6, 6, 3), 1000.0)
    dm = _cbf_to_dm(np.full_like(m0, 50.0), m0, **P)
    dm[:3] *= -1.0                                               # inject a sign-flipped region
    out = dm_to_cbf(dm, m0, **P, clamp_negative=True)
    assert out["qc"]["neg_frac"] > 0.4                           # ~half flagged negative
    assert (out["cbf"] >= 0).all()                              # physically clamped, not abs()'d


def test_m0_guard_zeros_invalid_voxels():
    m0 = np.full((6, 6, 3), 1000.0)
    m0[:, :, 0] = 0.0                                            # air / invalid slice
    dm = _cbf_to_dm(np.full_like(m0, 50.0), np.full_like(m0, 1000.0), **P)
    out = dm_to_cbf(dm, m0, **P, m0_floor=1.0)
    assert np.all(out["cbf"][:, :, 0] == 0.0)
    assert out["qc"]["m0_invalid_frac"] > 0.0


def test_mask_zeros_outside_brain():
    cbf_true = np.full((6, 6, 4), 55.0)
    m0 = np.full((6, 6, 4), 1000.0)
    dm = _cbf_to_dm(cbf_true, m0, **P)
    mask = np.zeros((6, 6, 4)); mask[1:5, 1:5, :] = 1.0
    out = dm_to_cbf(dm, m0, mask=mask, **P)
    assert np.all(out["cbf"][mask < 0.5] == 0.0)
    assert np.allclose(out["cbf"][mask > 0.5], 55.0, rtol=1e-6)


def test_denorm_roundtrip():
    rng = np.random.default_rng(1)
    dm_raw = rng.uniform(-5, 30, size=(4, 4, 4))
    lo, scale = -3.1, 27.4
    norm = (dm_raw - lo) / scale
    assert np.allclose(denorm_deltam(norm, lo, scale), dm_raw, rtol=1e-9, atol=1e-9)


def test_qc_ranges_pass_and_fail():
    cbf = np.zeros((10, 10, 4))
    gm = np.zeros_like(cbf); wm = np.zeros_like(cbf)
    gm[:, :5] = 1.0; wm[:, 5:] = 1.0
    cbf[gm > 0.5] = 55.0; cbf[wm > 0.5] = 22.0                   # ratio 2.5
    q = qc_cbf_ranges(cbf, gm, wm)
    assert q["pass"] and q["gm_in_range"] and q["wm_in_range"] and q["ratio_in_range"]
    cbf[gm > 0.5] = 120.0                                        # unphysical GM
    assert not qc_cbf_ranges(cbf, gm, wm)["pass"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nALL {len(fns)} CBF TESTS PASSED")
