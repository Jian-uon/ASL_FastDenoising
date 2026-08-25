# -*- coding: utf-8 -*-
"""Unit tests for utils/cbf_metrics.py (CBF-space agreement / reproducibility).

Synthetic cases with known answers. Runnable as:
    python tests/test_cbf_metrics.py    (asserts, prints OK)
    pytest tests/test_cbf_metrics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.cbf_metrics import (  # noqa: E402
    icc_2_1, icc_agreement, bland_altman, within_subject,
    wscv_ratio_ni, bootstrap_ci, mean_diff_ci,
)


def test_icc_identical_columns_is_one():
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 80, size=40)
    assert abs(icc_agreement(x, x) - 1.0) < 1e-9


def test_icc_independent_near_zero():
    rng = np.random.default_rng(1)
    a = rng.uniform(20, 80, size=300)
    b = rng.uniform(20, 80, size=300)
    assert abs(icc_agreement(a, b)) < 0.2


def test_icc_high_agreement():
    rng = np.random.default_rng(2)
    x = rng.uniform(20, 80, size=60)
    y = x + rng.normal(0, 1.0, size=60)            # tiny noise vs a ~17 SD signal
    assert icc_agreement(x, y) > 0.95


def test_bland_altman_recovers_bias_no_prop():
    rng = np.random.default_rng(3)
    a = rng.uniform(40, 70, size=50)
    b = a + 5.0 + rng.normal(0, 0.5, size=50)       # constant +5 offset on b => diff ~ -5
    ba = bland_altman(a, b)
    assert abs(ba["bias"] + 5.0) < 0.4
    assert abs(ba["prop_slope"]) < 0.1              # no proportional bias
    assert ba["loa_low"] < ba["bias"] < ba["loa_high"]


def test_within_subject_analytic_sw():
    a = np.array([10.0, 10.0, 10.0, 10.0])
    b = np.array([8.0, 12.0, 8.0, 12.0])            # a-b = [2,-2,2,-2] -> sw = sqrt(4/2)=sqrt(2)
    ws = within_subject(a, b)
    assert abs(ws["sw"] - np.sqrt(2.0)) < 1e-9
    assert abs(ws["rc"] - 2.77 * np.sqrt(2.0)) < 1e-9


def test_wscv_ratio_ni_fails_when_accel_noisier():
    rng = np.random.default_rng(4)
    base = 50 + rng.normal(0, 5, size=40)
    full_t = base + rng.normal(0, 0.3, size=40); full_r = base + rng.normal(0, 0.3, size=40)
    acc_t = base + rng.normal(0, 1.5, size=40); acc_r = base + rng.normal(0, 1.5, size=40)
    out = wscv_ratio_ni(acc_t, acc_r, full_t, full_r, margin=1.15, n_boot=1000, seed=7)
    assert out["ratio"] > 1.15
    assert out["non_inferior"] is False


def test_wscv_ratio_ni_passes_when_accel_cleaner():
    rng = np.random.default_rng(5)
    base = 50 + rng.normal(0, 5, size=60)
    full_t = base + rng.normal(0, 0.6, size=60); full_r = base + rng.normal(0, 0.6, size=60)
    acc_t = base + rng.normal(0, 0.3, size=60); acc_r = base + rng.normal(0, 0.3, size=60)
    out = wscv_ratio_ni(acc_t, acc_r, full_t, full_r, margin=1.15, n_boot=1000, seed=7)
    assert out["ratio"] < 1.0
    assert out["non_inferior"] is True


def test_bootstrap_ci_degenerate_constant():
    lo, hi = bootstrap_ci(np.full(30, 7.0), lambda d: float(d.mean()), n_boot=500, seed=0)
    assert lo == hi == 7.0


def test_bootstrap_ci_brackets_mean():
    data = np.arange(100).astype(float)
    lo, hi = bootstrap_ci(data, lambda d: float(d.mean()), n_boot=1500, method="percentile", seed=0)
    assert lo < 49.5 < hi
    assert (hi - lo) < 20                            # reasonably tight for n=100


def test_mean_diff_ci_constant():
    out = mean_diff_ci(np.full(30, 10.0), np.full(30, 7.0), n_boot=500, seed=0)
    assert abs(out["diff"] - 3.0) < 1e-9
    assert out["ci_low"] <= 3.0 <= out["ci_high"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nALL {len(fns)} CBF-METRICS TESTS PASSED")
