# -*- coding: utf-8 -*-
"""Unit tests for the two Medical Physics metrics added 2026-08-26.

`snr_difference_image` (Dietrich et al., JMRI 2007) and `mutual_information`. Both are
reported metrics, never selection criteria, so what has to hold is that they measure
what their names say — checked here against cases with a known answer.

Runnable either way:
    python tests/test_snr_mi.py       # plain asserts, prints OK
    pytest tests/test_snr_mi.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import mutual_information, snr_difference_image  # noqa: E402

B, HW = 3, 48


def _roi(frac=0.4):
    r = torch.zeros(B, 1, HW, HW)
    k = int(HW * frac)
    r[:, :, :k, :k] = 1.0
    return r


# --- SNR --------------------------------------------------------------------
def test_snr_recovers_a_known_value():
    """signal s with i.i.d. noise sigma in each of the two reconstructions =>
    difference has std sigma*sqrt(2), so the estimator must return s/sigma."""
    torch.manual_seed(0)
    roi = _roi()
    s, sigma = 4.0, 0.5
    base = torch.full((B, 1, HW, HW), s)
    a = base + torch.randn(B, 1, HW, HW) * sigma
    b = base + torch.randn(B, 1, HW, HW) * sigma
    got = snr_difference_image(a, b, roi)
    assert abs(got - s / sigma) / (s / sigma) < 0.12, got


def test_snr_scales_with_noise():
    torch.manual_seed(0)
    roi = _roi()
    base = torch.full((B, 1, HW, HW), 4.0)
    out = []
    for sigma in (0.25, 0.5, 1.0):
        a = base + torch.randn(B, 1, HW, HW) * sigma
        b = base + torch.randn(B, 1, HW, HW) * sigma
        out.append(snr_difference_image(a, b, roi))
    assert out[0] > out[1] > out[2], out          # more noise -> lower SNR
    assert abs(out[0] / out[1] - 2.0) < 0.35, out  # and roughly inversely


def test_snr_is_independent_of_the_background():
    """The background is identically zero under --premask_asl_inputs, which is why the
    textbook mean/std_background estimator is unusable. This one must not care."""
    torch.manual_seed(0)
    roi = _roi()
    base = torch.full((B, 1, HW, HW), 4.0)
    a = base + torch.randn(B, 1, HW, HW) * 0.5
    b = base + torch.randn(B, 1, HW, HW) * 0.5
    outside = 1.0 - roi
    got_zero_bg = snr_difference_image(a * roi, b * roi, roi)
    got_junk_bg = snr_difference_image(a * roi + outside * 99.0,
                                       b * roi + outside * -7.0, roi)
    assert abs(got_zero_bg - got_junk_bg) < 1e-4


def test_snr_skips_tiny_rois():
    roi = torch.zeros(B, 1, HW, HW)
    roi[0, 0, 0, 0] = 1.0                      # one voxel: no usable std
    x = torch.rand(B, 1, HW, HW)
    assert math.isnan(snr_difference_image(x, x + 0.1, roi))


# --- MI ---------------------------------------------------------------------
def test_mi_of_an_image_with_itself_is_its_entropy():
    """MI(x, x) = H(x); for values spread over all `bins` bins that is log2(bins)."""
    bins = 32
    x = torch.linspace(0, 1, HW * HW).view(1, 1, HW, HW).repeat(B, 1, 1, 1)
    mi = mutual_information(x, x, bins=bins)
    assert abs(mi - math.log2(bins)) < 0.35, mi


def test_mi_is_near_zero_for_independent_images():
    torch.manual_seed(0)
    x = torch.rand(B, 1, HW, HW)
    y = torch.rand(B, 1, HW, HW)
    mi = mutual_information(x, y, bins=16)
    assert mi < 0.6, mi                         # finite-sample bias keeps it above 0


def test_mi_is_invariant_to_a_per_image_affine():
    """Binning is per sample over its own range, so a rescale must not move MI. This is
    what makes MI comparable across arms whose outputs have different global scales."""
    torch.manual_seed(0)
    x = torch.rand(B, 1, HW, HW)
    y = torch.rand(B, 1, HW, HW)
    base = mutual_information(x, y, bins=32)
    scaled = mutual_information(x * 7.5 - 3.0, y * 0.2 + 11.0, bins=32)
    assert abs(base - scaled) < 1e-4


def test_mi_ranks_dependence():
    """MI must order strongly > weakly > not dependent.

    Note the floor: with bins=32 over ~2.3k voxels, two INDEPENDENT images still score
    ~0.35 bits from finite-sample bias, so an absolute MI value means little. Only
    comparisons at matched bins and matched voxel count are interpretable -- which is
    exactly how MI(pred, T1) gets used across arms in the paper.

    The noise on the weak pair is kept at the scale of the signal. Pushing it far above
    (sigma=2 on data in [0,1]) makes the per-sample min/max binning collapse onto a few
    central bins, and the measured MI then falls BELOW the independent-pair floor -- an
    artefact of the histogram, not of the dependence.
    """
    torch.manual_seed(0)
    x = torch.rand(B, 1, HW, HW)
    strong = x + torch.randn_like(x) * 0.05
    weak = x + torch.randn_like(x) * 0.5
    indep = torch.rand(B, 1, HW, HW)
    a = mutual_information(x, strong, bins=32)
    b = mutual_information(x, weak, bins=32)
    c = mutual_information(x, indep, bins=32)
    assert a > b > c, (a, b, c)
    assert c < 0.6, c                            # the bias floor, not real dependence


def test_mi_respects_the_mask():
    torch.manual_seed(0)
    x = torch.rand(B, 1, HW, HW)
    y = torch.rand(B, 1, HW, HW)
    mask = _roi()
    inside = y * mask + x * (1 - mask)          # identical to x OUTSIDE the mask only
    mi_masked = mutual_information(x, inside, mask=mask, bins=32)
    mi_all = mutual_information(x, inside, bins=32)
    assert mi_all > mi_masked, (mi_all, mi_masked)


def test_mi_handles_a_constant_image():
    x = torch.rand(B, 1, HW, HW)
    const = torch.full((B, 1, HW, HW), 0.3)
    assert mutual_information(x, const, bins=32) < 1e-4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} SNR/MI tests passed.")
