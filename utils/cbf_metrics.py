# -*- coding: utf-8 -*-
"""CBF-space agreement / reproducibility metrics for the accelerated-ASL evaluation.

Pure numpy + math (no torch, no scipy) — a POST-HOC evaluation layer, never imported
by the training runner. Together with `utils/cbf.py` it is the whole CBF-space
evaluation stack (plan ASL_dmvae/docs/rad_ai_route2_plan.md §4; Paper-1 experiments E1/E2/E3).

Analysis unit is the SUBJECT (or subject×session×ROI) — feed ONE scalar per subject
(e.g. a regional CBF / rCBF value), NOT per-slice / per-voxel values. Every CI is a
SUBJECT-level bootstrap (BCa by default, Efron & Tibshirani 1993).

Provided:
  icc_2_1 / icc_agreement  — ICC(2,1) two-way random, single-measure, absolute
                             agreement (accelerated vs full).                 [E1]
  bland_altman             — bias, 95% limits of agreement, proportional bias. [E1]
  within_subject           — within-subject SD (sw), repeatability coefficient
                             RC = 2.77·sw, and dimensionless wsCV (test-retest).[E2]
  wscv_ratio_ni            — primary NI endpoint: wsCV(accel) / wsCV(full) with a
                             BCa CI and a pre-set non-inferiority margin.       [E2]
  bootstrap_ci             — subject-level percentile or BCa bootstrap CI.
  mean_diff_ci             — unpaired mean difference + CI (generalization gap). [E3]
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = [
    "icc_2_1", "icc_agreement", "bland_altman", "within_subject",
    "wscv_ratio_ni", "bootstrap_ci", "mean_diff_ci",
]


# --------------------------------------------------------------------------- #
# Normal CDF / inverse-CDF (self-contained, for BCa) — no scipy dependency.
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, err < 1e-9)."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


# --------------------------------------------------------------------------- #
# ICC(2,1): two-way random effects, single measurement, absolute agreement.
# --------------------------------------------------------------------------- #
def icc_2_1(matrix: np.ndarray) -> float:
    """ICC(2,1) (Shrout & Fleiss 1979) for an [n_subjects, k_measurements] matrix.

    Two-way random, single-rater, absolute-agreement — the correct ICC for
    'does accelerated reproduce full' (penalizes both scale and offset shifts).
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2:
        raise ValueError("icc_2_1 expects an [n>=2, k>=2] matrix")
    n, k = m.shape
    grand = m.mean()
    ssr = k * np.sum((m.mean(1) - grand) ** 2)              # between subjects (rows)
    ssc = n * np.sum((m.mean(0) - grand) ** 2)              # between measurements (cols)
    sst = np.sum((m - grand) ** 2)
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    return float((msr - mse) / denom) if denom != 0 else float("nan")


def icc_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1) for two paired per-subject arrays (e.g. accelerated vs full CBF)."""
    return icc_2_1(np.column_stack([np.asarray(a, float), np.asarray(b, float)]))


# --------------------------------------------------------------------------- #
# Bland-Altman agreement.
# --------------------------------------------------------------------------- #
def bland_altman(a: np.ndarray, b: np.ndarray) -> dict:
    """Bland-Altman of paired per-subject measurements (diff = a − b).

    Returns bias, SD of differences, 95% limits of agreement, and the
    proportional-bias slope (regression of difference on mean).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = a - b
    mean = (a + b) / 2.0
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if diff.size > 1 else float("nan")
    slope = float(np.polyfit(mean, diff, 1)[0]) if diff.size > 1 else float("nan")
    return {
        "bias": bias, "sd_diff": sd,
        "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
        "prop_slope": slope, "n": int(a.size),
    }


# --------------------------------------------------------------------------- #
# Test-retest repeatability.
# --------------------------------------------------------------------------- #
def _wscv(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mp = (a + b) / 2.0
    return float(np.sqrt(np.mean(((a - b) / mp) ** 2) / 2.0))


def within_subject(a: np.ndarray, b: np.ndarray) -> dict:
    """Test-retest repeatability from paired scans a (test) and b (retest).

    sw   = within-subject SD = sqrt(mean((a−b)^2)/2)
    RC   = repeatability coefficient = 2.77·sw (Bland-Altman)
    wsCV = dimensionless within-subject CV = sqrt(mean(((a−b)/mean)^2)/2)
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    sw = float(np.sqrt(np.mean(d ** 2) / 2.0))
    return {"sw": sw, "rc": 2.77 * sw, "wscv": _wscv(a, b), "mean": float(((a + b) / 2.0).mean())}


# --------------------------------------------------------------------------- #
# Subject-level bootstrap CI (percentile or BCa).
# --------------------------------------------------------------------------- #
def bootstrap_ci(data: np.ndarray, statfn: Callable[[np.ndarray], float], *,
                 n_boot: int = 10000, ci: float = 0.95, method: str = "bca",
                 seed: int = 0) -> tuple:
    """Subject-level bootstrap CI. `data` is resampled along axis 0 (subjects);
    `statfn(data)` returns a scalar. method: 'bca' (default) or 'percentile'."""
    data = np.asarray(data, float)
    n = data.shape[0]
    theta_hat = float(statfn(data))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = np.array([statfn(data[ix]) for ix in idx], dtype=float)
    reps = reps[np.isfinite(reps)]
    alpha = 1.0 - ci
    if reps.size == 0 or np.std(reps) < 1e-12:             # degenerate / zero-variance
        return (theta_hat, theta_hat)
    if method == "percentile":
        return (float(np.percentile(reps, 100 * alpha / 2)),
                float(np.percentile(reps, 100 * (1 - alpha / 2))))
    # BCa
    z0 = _norm_ppf(np.clip(np.mean(reps < theta_hat), 1e-6, 1 - 1e-6))
    jack = np.array([statfn(np.delete(data, i, axis=0)) for i in range(n)], dtype=float)
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2)) ** 1.5
    acc = num / den if den != 0 else 0.0
    zl, zu = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)
    a1 = _norm_cdf(z0 + (z0 + zl) / (1 - acc * (z0 + zl)))
    a2 = _norm_cdf(z0 + (z0 + zu) / (1 - acc * (z0 + zu)))
    return (float(np.percentile(reps, 100 * a1)), float(np.percentile(reps, 100 * a2)))


def wscv_ratio_ni(accel_test: np.ndarray, accel_retest: np.ndarray,
                  full_test: np.ndarray, full_retest: np.ndarray, *,
                  margin: float = 1.15, n_boot: int = 10000, ci: float = 0.95,
                  seed: int = 0) -> dict:
    """PRIMARY non-inferiority endpoint (plan §5.3 / STAT-C3): variance-matched
    comparison of test-retest wsCV under accelerated vs full acquisition.

    ratio = wsCV(accel) / wsCV(full); non-inferior iff the CI UPPER bound < margin
    (i.e. going from full to accelerated adds no more within-subject variability
    than a rescan of the full acquisition itself, within `margin`).
    """
    data = np.column_stack([accel_test, accel_retest, full_test, full_retest]).astype(float)

    def ratio(d):
        wf = _wscv(d[:, 2], d[:, 3])
        return _wscv(d[:, 0], d[:, 1]) / wf if wf > 0 else np.nan

    r_hat = float(ratio(data))
    lo, hi = bootstrap_ci(data, ratio, n_boot=n_boot, ci=ci, method="bca", seed=seed)
    return {"ratio": r_hat, "ci_low": lo, "ci_high": hi, "margin": float(margin),
            "non_inferior": bool(hi < margin)}


def mean_diff_ci(a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 0) -> dict:
    """Unpaired mean difference (mean(a) − mean(b)) with a bootstrap CI.

    Use for a generalization gap: a = held-out-site metric values, b = pooled;
    the gap is non-significant when the CI brackets 0.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff_hat = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    reps = np.array([a[rng.integers(0, a.size, a.size)].mean()
                     - b[rng.integers(0, b.size, b.size)].mean() for _ in range(n_boot)])
    alpha = 1.0 - ci
    return {"diff": diff_hat,
            "ci_low": float(np.percentile(reps, 100 * alpha / 2)),
            "ci_high": float(np.percentile(reps, 100 * (1 - alpha / 2)))}
