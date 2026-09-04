# -*- coding: utf-8 -*-
"""Sharpness / structure metrics for ASL denoising.

All metrics operate on [B,1,H,W] tensors and return scalar floats.
Brain mask is [B,1,H,W] in {0, 1}.

Recommended for distinguishing "blurry-but-PSNR-high" predictions from
sharper ones. PSNR/L1 alone over-reward smoothness.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Sobel + Laplacian kernels (cached per device/dtype)
# ---------------------------------------------------------------------------

_SOBEL_X: Optional[Tensor] = None
_SOBEL_Y: Optional[Tensor] = None
_LAPLACIAN: Optional[Tensor] = None


def _ensure_sobel(device, dtype):
    global _SOBEL_X, _SOBEL_Y
    if (_SOBEL_X is None or _SOBEL_X.device != device or _SOBEL_X.dtype != dtype):
        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=dtype, device=device) / 8.0
        ky = kx.transpose(0, 1).contiguous()
        _SOBEL_X = kx.view(1, 1, 3, 3)
        _SOBEL_Y = ky.view(1, 1, 3, 3)
    return _SOBEL_X, _SOBEL_Y


def _ensure_laplacian(device, dtype):
    global _LAPLACIAN
    if (_LAPLACIAN is None or _LAPLACIAN.device != device or _LAPLACIAN.dtype != dtype):
        k = torch.tensor([[0.0,  1.0, 0.0],
                          [1.0, -4.0, 1.0],
                          [0.0,  1.0, 0.0]], dtype=dtype, device=device)
        _LAPLACIAN = k.view(1, 1, 3, 3)
    return _LAPLACIAN


# ---------------------------------------------------------------------------
# GMSD (Gradient Magnitude Similarity Deviation)
# ---------------------------------------------------------------------------

def gmsd(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None,
         c: float = 0.0026) -> float:
    """Gradient Magnitude Similarity Deviation (Xue et al. 2014).

    Lower = more similar gradient structure (less blurring).
    Insensitive to small intensity offsets, sensitive to high-frequency loss.
    """
    sx, sy = _ensure_sobel(pred.device, pred.dtype)
    gx_p = F.conv2d(pred,   sx, padding=1)
    gy_p = F.conv2d(pred,   sy, padding=1)
    gx_t = F.conv2d(target, sx, padding=1)
    gy_t = F.conv2d(target, sy, padding=1)
    gm_p = (gx_p ** 2 + gy_p ** 2).sqrt()
    gm_t = (gx_t ** 2 + gy_t ** 2).sqrt()
    similarity = (2.0 * gm_p * gm_t + c) / (gm_p ** 2 + gm_t ** 2 + c)   # [B,1,H,W]
    if mask is not None:
        m = mask.float()
        # weighted mean and std under mask
        n = m.sum().clamp_min(1.0)
        mean = (similarity * m).sum() / n
        var = ((similarity - mean) ** 2 * m).sum() / n
    else:
        mean = similarity.mean()
        var = similarity.var()
    return float(var.sqrt().item())


# Partial-volume threshold defining a region that is meant to be ONE tissue. At 2 mm a
# cortical ribbon is one to two voxels wide, so pv > 0.5 mixes the tissues into both regions
# and dilutes every contrast between them. 0.7 is the ASL convention.
#
# This is NOT the threshold for the union of gray and white matter, which marks brain tissue
# and is the denominator of relative CBF: there the aim is coverage, not purity, and it stays
# at 0.5. Anything importing this constant is asking for a single-tissue region.
PV_TISSUE = 0.7

# CSF stands in for the noise floor, so the requirement on its ROI is purity: a voxel that
# still holds tissue carries perfusion and would inflate the very quantity used as noise.
# Purity is enforced by the PV threshold rather than by eroding the mask. Measured on this
# dataset, a 2-voxel erosion of csf > PV_TISSUE leaves a median of 30 voxels in a whole
# 52-slice volume -- under the 64-voxel floor for 92% of evaluation batches, which turns the
# SNR column into NaN. csf > PV_CSF_PURE keeps ~3.6k voxels and 71% of batches, and a voxel
# that is 90% CSF by partial volume is not a boundary voxel to begin with.
PV_CSF_PURE = 0.9

# ---------------------------------------------------------------------------
# GM/WM contrast ratio error
# ---------------------------------------------------------------------------

def gm_wm_contrast_error(pred: Tensor, target: Tensor,
                         gm: Tensor, wm: Tensor,
                         pv_threshold: float = PV_TISSUE) -> float:
    """Absolute error of GM/WM signal ratio.

    Computes mean(signal in GM) / mean(signal in WM) for both pred and target,
    then returns |ratio_pred - ratio_target|. Lower = better contrast preservation.

    Inputs:
      pred, target: [B,1,H,W]
      gm, wm: [B,1,H,W] partial volume maps (probability)
      pv_threshold: pixels with PV > threshold are considered tissue.
    """
    gm_mask = (gm > pv_threshold).float()
    wm_mask = (wm > pv_threshold).float()

    p_gm = (pred   * gm_mask).sum() / gm_mask.sum().clamp_min(1.0)
    p_wm = (pred   * wm_mask).sum() / wm_mask.sum().clamp_min(1.0)
    t_gm = (target * gm_mask).sum() / gm_mask.sum().clamp_min(1.0)
    t_wm = (target * wm_mask).sum() / wm_mask.sum().clamp_min(1.0)

    p_ratio = (p_gm + 1e-8) / (p_wm.abs() + 1e-8)
    t_ratio = (t_gm + 1e-8) / (t_wm.abs() + 1e-8)
    return float((p_ratio - t_ratio).abs().item())


# ---------------------------------------------------------------------------
# Laplacian variance (no-reference sharpness indicator)
# ---------------------------------------------------------------------------

def laplacian_variance(image: Tensor, mask: Optional[Tensor] = None) -> float:
    """Variance of Laplacian filter response. Higher = sharper.

    No-reference: just measures the high-frequency content of the image itself.
    Use to compare pred's intrinsic sharpness against target's.
    """
    lap = _ensure_laplacian(image.device, image.dtype)
    response = F.conv2d(image, lap, padding=1)               # [B,1,H,W]
    if mask is not None:
        m = mask.float()
        n = m.sum().clamp_min(1.0)
        mean = (response * m).sum() / n
        var = ((response - mean) ** 2 * m).sum() / n
    else:
        var = response.var()
    return float(var.item())


# ---------------------------------------------------------------------------
# High-Frequency Error Norm (HFEN) — Ravishankar 2011
# ---------------------------------------------------------------------------

_LOG_KERNEL: Optional[Tensor] = None


def _ensure_log(device, dtype, sigma: float = 1.5, ksize: int = 9):
    global _LOG_KERNEL
    if (_LOG_KERNEL is None or _LOG_KERNEL.device != device
            or _LOG_KERNEL.dtype != dtype):
        ax = torch.arange(ksize, dtype=torch.float64) - (ksize - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        s2 = sigma * sigma
        g = torch.exp(-(xx * xx + yy * yy) / (2.0 * s2))
        log = ((xx * xx + yy * yy - 2.0 * s2) / (s2 * s2)) * g
        log = log - log.mean()  # zero-mean → high-pass
        _LOG_KERNEL = log.to(dtype=dtype, device=device).view(1, 1, ksize, ksize)
    return _LOG_KERNEL


def hfen(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None,
         sigma: float = 1.5) -> float:
    """High-Frequency Error Norm = ||LoG(pred) - LoG(target)||_2 / ||LoG(target)||_2.

    Ravishankar & Bresler (2011). Standard MRI denoising sharpness metric;
    measures error in the high-frequency band only — robust to ref residual
    noise compared with PSNR.
    """
    k = _ensure_log(pred.device, pred.dtype, sigma=sigma, ksize=9)
    lp_p = F.conv2d(pred,   k, padding=4)
    lp_t = F.conv2d(target, k, padding=4)
    if mask is not None:
        diff = (lp_p - lp_t) * mask
        den  = lp_t * mask
    else:
        diff = lp_p - lp_t
        den  = lp_t
    num = float(diff.pow(2).sum().sqrt().item())
    den_norm = float(den.pow(2).sum().sqrt().clamp_min(1e-8).item())
    return num / den_norm


# ---------------------------------------------------------------------------
# No-reference image quality: TG / IE / GE
# ---------------------------------------------------------------------------

def tenengrad(image: Tensor, mask: Optional[Tensor] = None) -> float:
    """Tenengrad — mean Sobel-gradient magnitude.

    Classic autofocus / sharpness metric (Krotkov 1987). Robust to noise
    compared with Laplacian-based variance. Higher = sharper.

    Returned as the mean of `sqrt(Sx^2 + Sy^2)` inside the (optional) mask.
    """
    sx, sy = _ensure_sobel(image.device, image.dtype)
    gx = F.conv2d(image, sx, padding=1)
    gy = F.conv2d(image, sy, padding=1)
    gmag = (gx * gx + gy * gy).clamp_min(1e-12).sqrt()           # [B,1,H,W]
    if mask is not None:
        m = mask.float()
        n = m.sum().clamp_min(1.0)
        return float(((gmag * m).sum() / n).item())
    return float(gmag.mean().item())


def _entropy_from_hist(values: Tensor, mask: Optional[Tensor],
                      bins: int, vmin: float, vmax: float) -> float:
    """Shared helper: histogram → Shannon entropy (in nats).

    `values` and `mask` are [B,1,H,W]. Histogram is computed over the masked
    region (whole image if mask=None), pooled across the batch.
    """
    v = values.flatten()
    if mask is not None:
        m = mask.float().flatten()
        v = v[m > 0.5]
    if v.numel() == 0:
        return 0.0
    hist = torch.histc(v.float(), bins=bins, min=vmin, max=vmax)
    p = hist / hist.sum().clamp_min(1.0)
    p = p[p > 0]                                                   # avoid log(0)
    return float(-(p * p.log()).sum().item())


def image_entropy(image: Tensor, mask: Optional[Tensor] = None,
                  bins: int = 64, vmin: float = 0.0, vmax: float = 1.0) -> float:
    """Shannon entropy of intensity histogram (nats).

    Higher = more diverse intensity distribution. For ASL: a healthy
    perfusion image should have higher IE than an over-smoothed one
    (which collapses toward a uniform value).
    """
    return _entropy_from_hist(image, mask, bins=bins, vmin=vmin, vmax=vmax)


def entropy_focus_criterion(image: Tensor, mask: Optional[Tensor] = None,
                            eps: float = 1e-12) -> float:
    """Entropy Focus Criterion (Atkinson et al. IEEE TMI 1997).

    EFC = -Σ p_i log p_i  where  p_i = x_i² / Σ x_j²

    Designed for MRI motion / blur detection: a sharp, in-focus image has
    pixel² energy concentrated on edges → low EFC; a blurred / motion-
    smeared image spreads energy uniformly → high EFC. Lower = sharper.

    Returned in nats. Computed within the (optional) mask.
    """
    x = image.flatten() if mask is None else image[mask > 0.5].flatten()
    if x.numel() == 0:
        return 0.0
    x2 = x.float() * x.float()
    denom = x2.sum().clamp_min(eps)
    p = x2 / denom
    p = p[p > eps]
    return float(-(p * p.log()).sum().item())


def bright_tail_ratio(image: Tensor, mask: Optional[Tensor] = None,
                      p_hi: float = 99.0, p_lo: float = 50.0,
                      eps: float = 1e-8) -> float:
    """p99 / |p50| of brain intensities — flags focal hot-pixel / hallucination
    spikes. A faithful PWI has a bounded ratio; a model inventing bright focal
    structure (or amplifying noise spikes) shows an inflated bright tail.
    Higher = heavier bright tail. Computed within the (optional) mask.
    """
    x = image.flatten() if mask is None else image[mask > 0.5].flatten()
    if x.numel() < 4:
        return float("nan")
    x = x.float()
    hi = torch.quantile(x, p_hi / 100.0)
    lo = torch.quantile(x, p_lo / 100.0)
    return float(hi / (lo.abs() + eps))


def cnr(image: Tensor, gm_mask: Tensor, wm_mask: Tensor,
        threshold: float = PV_TISSUE, eps: float = 1e-8) -> float:
    """Contrast-to-Noise Ratio between GM and WM.

        CNR = |μ_GM − μ_WM| / σ_WM

    σ_WM is the within-WM std as the noise reference (assumes WM should be
    relatively uniform; over-smoothing decreases σ_WM but also decreases
    contrast, so CNR is not artificially inflated by smoothing alone).

    Args:
        image: [B,1,H,W] (or [B,C,H,W]) tensor
        gm_mask, wm_mask: same shape, soft PV masks in [0,1]
        threshold: PV threshold; PV_TISSUE (0.7) is the ASL convention.
    """
    gm = (gm_mask > threshold).float()
    wm = (wm_mask > threshold).float()
    n_gm = gm.sum().clamp_min(1.0)
    n_wm = wm.sum().clamp_min(1.0)
    mu_gm = (image * gm).sum() / n_gm
    mu_wm = (image * wm).sum() / n_wm
    var_wm = (((image - mu_wm) ** 2) * wm).sum() / n_wm
    sigma_wm = var_wm.sqrt().clamp_min(eps)
    return float(((mu_gm - mu_wm).abs() / sigma_wm).item())


def scov(image: Tensor, mask: Tensor, threshold: float = PV_TISSUE,
         eps: float = 1e-8) -> float:
    """Spatial Coefficient of Variation = std(x)/mean(x) inside `mask`.

    For ASL: report on a grey-matter PV mask thresholded at PV_TISSUE to exclude
    partial-volume voxels. Lower = more uniform (less noise / motion).
    (This docstring previously credited a Wang 2003 MRM paper; that attribution was
    wrong and had been copied into the manuscript.)

    Note: scale-invariant under a GLOBAL rescaling, which is why rCBF and CBF give
    identical values. It is NOT invariant between PWI and CBF: the conversion divides
    by the M0 MAP, not by a scalar, and M0 is far from uniform at 7T. Measured on the
    12-repetition average of 24 subjects, GM M0 has a spatial CoV of 0.34, and sCoV on
    CBF runs 65% above sCoV on the PWI (0.69 -> 1.12 on average, per-subject range -4%
    to +225%, subject rank correlation only 0.58). This docstring previously claimed the
    two agreed to within 2%; that assumed the conversion was a linear rescaling and was
    wrong. Values computed on PWI are therefore NOT comparable with published GM sCoV
    figures, which are defined on CBF (Mutsaerts et al., J Cereb Blood Flow Metab
    2017;37(9):3184-3192 — the ASL community standard).
    """
    if image.numel() == 0 or mask.numel() == 0:
        return 0.0
    m = (mask > threshold).float().flatten()
    x = image.flatten()
    n = m.sum().clamp_min(1.0)
    mean = (x * m).sum() / n
    var = (((x - mean) ** 2) * m).sum() / n
    return float((var.sqrt() / mean.abs().clamp_min(eps)).item())


def mi_nmi(a: Tensor, b: Tensor, mask: Optional[Tensor] = None,
           bins: int = 32, eps: float = 1e-12) -> tuple:
    """Mutual information (nats) and normalized MI between two images.

        MI  = H(A) + H(B) − H(A,B)
        NMI = 2·MI / (H(A) + H(B))            ∈ [0, 1]  (arithmetic-mean norm)

    Histogram-based over brain-masked voxels. Each image is min-max scaled to
    [0,1] inside the mask before binning, so MI is intensity-scale invariant.
    Returns (mi, nmi) floats; (nan, nan) if too few voxels or a constant image.

    Cross-modal use: MI(recon, T1) = information shared with anatomy. Interpret
    against MI(union, T1): recon ≈ union ⇒ natural anatomical correlation;
    recon ≫ union ⇒ excess T1 dependence (a leak signal, like t1sim).
    """
    if mask is not None:
        m = (mask > 0.5).flatten()
        av = a.flatten()[m]
        bv = b.flatten()[m]
    else:
        av = a.flatten()
        bv = b.flatten()
    if av.numel() < 16:
        return float("nan"), float("nan")

    def _binize(v: Tensor):
        vmin = v.min()
        vmax = v.max()
        if float(vmax - vmin) < eps:
            return None
        return ((v - vmin) / (vmax - vmin) * (bins - 1)).round().long().clamp_(0, bins - 1)

    ai = _binize(av)
    bi = _binize(bv)
    if ai is None or bi is None:
        return float("nan"), float("nan")

    joint = torch.zeros(bins * bins, device=av.device, dtype=torch.float64)
    lin = (ai * bins + bi).to(torch.long)
    joint.scatter_add_(0, lin, torch.ones_like(lin, dtype=torch.float64))
    pxy = (joint / joint.sum().clamp_min(1.0)).view(bins, bins)
    px = pxy.sum(1)
    py = pxy.sum(0)
    nz = pxy > 0
    h_xy = -(pxy[nz] * pxy[nz].log()).sum()
    h_x = -(px[px > 0] * px[px > 0].log()).sum()
    h_y = -(py[py > 0] * py[py > 0].log()).sum()
    mi = (h_x + h_y - h_xy).clamp_min(0.0)
    nmi = 2.0 * mi / (h_x + h_y).clamp_min(eps)
    return float(mi), float(nmi)


def upsnr_components(pred: Tensor, set_b: Tensor, lengths_b: Optional[Tensor] = None,
                     mask: Optional[Tensor] = None) -> tuple:
    """Return per-batch raw sums for **pooled** uPSNR aggregation.

    Returns:
        (sum_sq_err, sum_var_corr, n_pixels) — all floats, all summed over the
        masked region. The pooled uMSE across an evaluation set is:
            uMSE = (Σ sq_err − Σ var_corr) / Σ n_pixels
        and uPSNR = 10·log10(M² / max(uMSE, eps)). Pooling avoids the
        log-of-mean ≠ mean-of-log artefact that blows up per-batch uPSNR
        variance.
    """
    if set_b.ndim != 5:
        raise ValueError(f"Expected set_b [B,T,1,H,W], got {tuple(set_b.shape)}")
    B, T, C, H, W = set_b.shape

    if lengths_b is not None:
        T_eff = int(lengths_b.min().item())
    else:
        T_eff = T
    n_per = T_eff // 3
    if n_per < 1:
        return 0.0, 0.0, 0.0, 0.0

    a = set_b[:, 0:n_per].mean(dim=1)
    b = set_b[:, n_per:2 * n_per].mean(dim=1)
    c = set_b[:, 2 * n_per:3 * n_per].mean(dim=1)

    sq_err   = (a - pred) ** 2
    var_corr = 0.5 * (b - c) ** 2
    bc_l1    = (b - c).abs()              # noise-floor L1 (data-driven τ_C basis)

    if mask is not None:
        m = mask.float()
        n = float(m.sum().item())
        ssq = float((sq_err * m).sum().item())
        svc = float((var_corr * m).sum().item())
        nfl1 = float((bc_l1 * m).sum().item())
    else:
        n = float(sq_err.numel())
        ssq = float(sq_err.sum().item())
        svc = float(var_corr.sum().item())
        nfl1 = float(bc_l1.sum().item())
    # Return tuple kept backward-compat for 3-arg callers: ssq, svc, n always
    # at positions 0–2; noise_floor_l1_sum appended at position 3.
    return ssq, svc, n, nfl1


def unsupervised_psnr(pred: Tensor, set_b: Tensor, lengths_b: Optional[Tensor] = None,
                      mask: Optional[Tensor] = None, data_range: float = 1.0,
                      eps: float = 1e-8) -> tuple:
    """Unsupervised PSNR (Marcos-Morales et al., ICML 2023, arxiv:2210.05553).

    Asymptotically unbiased estimator of PSNR-to-clean using only noisy data:

        uMSE = mean[(a - f(y))^2 - 0.5 (b - c)^2]
        uPSNR = 10 · log10(M^2 / uMSE)

    where (a, b, c) are three INDEPENDENT noisy realisations of the same
    underlying signal. The (b-c)^2 / 2 term is an unbiased estimate of the
    noise variance carried in `a` and exactly cancels the noise contribution
    in the first term, so E[uMSE] = MSE(f(y), x_clean).

    For ASL: we split the held-out set_b into 3 equal sub-groups and use
    their per-pixel means as a, b, c. set_b is genuinely independent of
    set_a (the denoiser's input), so a, b, c are independent of f(y).

    Args:
        pred:     [B,1,H,W] denoiser output f(set_a)
        set_b:    [B,T_b,1,H,W] held-out frames (NOT seen by the model)
        lengths_b: [B] number of valid frames per sample (else use full T_b)
        mask:     [B,1,H,W] brain mask
        data_range: signal range M (default 1.0 for normalized PWI)

    Returns:
        (uPSNR, uMSE) — scalars, in dB and squared-intensity units.
        uMSE is the SIGNED unbiased risk estimate (may legitimately be negative from
        finite-sample noise; do NOT clamp it — a small/negative uMSE is meaningful).
        uPSNR is only defined for a POSITIVE risk: if uMSE <= eps we return NaN for
        uPSNR (clamping a ~0/negative uMSE up to eps would fabricate a ~120 dB uPSNR).
    """
    if set_b.ndim != 5:
        raise ValueError(f"Expected set_b [B,T,1,H,W], got {tuple(set_b.shape)}")
    B, T, C, H, W = set_b.shape

    # Use minimum T across batch (or lengths_b if provided)
    if lengths_b is not None:
        T_eff = int(lengths_b.min().item())
    else:
        T_eff = T
    n_per = T_eff // 3
    if n_per < 1:
        # Not enough frames for 3-way split; fall back to NaN
        return float("nan"), float("nan")

    a = set_b[:, 0:n_per].mean(dim=1)                      # [B,1,H,W]
    b = set_b[:, n_per:2 * n_per].mean(dim=1)
    c = set_b[:, 2 * n_per:3 * n_per].mean(dim=1)

    sq_err   = (a - pred) ** 2                             # [B,1,H,W]
    var_corr = 0.5 * (b - c) ** 2                          # [B,1,H,W]

    if mask is not None:
        m = mask.float()
        n = m.sum().clamp_min(1.0)
        umse = (((sq_err - var_corr) * m).sum() / n)
    else:
        umse = (sq_err - var_corr).mean()

    # Keep uMSE SIGNED (a legitimate risk estimate; negative = finite-sample noise).
    # uPSNR is only meaningful for a positive risk — otherwise NaN, never a clamped
    # ~120 dB artefact.
    if float(umse.item()) > eps:
        upsnr = float((10.0 * torch.log10((data_range ** 2) / umse)).item())
    else:
        upsnr = float("nan")
    return upsnr, float(umse.item())


def gradient_entropy(image: Tensor, mask: Optional[Tensor] = None,
                     bins: int = 64) -> float:
    """Shannon entropy of gradient-magnitude histogram (nats).

    Captures textural richness: a flat image has near-zero GE; a noisy
    image has high GE; a well-denoised image has GE between.
    Use as no-reference companion to TG and IE.
    """
    sx, sy = _ensure_sobel(image.device, image.dtype)
    gx = F.conv2d(image, sx, padding=1)
    gy = F.conv2d(image, sy, padding=1)
    gmag = (gx * gx + gy * gy).clamp_min(1e-12).sqrt()
    # Use the 99th percentile under mask as upper bound for stable binning
    if mask is not None:
        m = mask.float()
        flat_g = gmag[m > 0.5]
    else:
        flat_g = gmag.flatten()
    if flat_g.numel() == 0:
        return 0.0
    vmax = float(flat_g.float().quantile(0.99).clamp_min(1e-6).item())
    return _entropy_from_hist(gmag, mask, bins=bins, vmin=0.0, vmax=vmax)


# ---------------------------------------------------------------------------
# Reference-free DETAIL-QUALITY metrics (2026-07) — the two primary detail
# metrics for the CIG bake-off. Both are reference-free, T1-free, and NOT gamed
# by noise (unlike raw lapvar/tenengrad, which reward re-introduced noise).
#
#   1. tissue_csf_hf_ratio  — high-frequency SNR: HF energy in tissue (real
#      texture + noise) over HF energy in CSF interior (ΔM≈0 ⇒ noise floor).
#      ≫1 = structured texture; ≈1 = the "high-freq" is just noise. Single pass.
#   2. hf_consistency       — split-half reproducibility: high-freq agreement of
#      two independently-denoised disjoint frame subsets. Real texture is
#      reproducible across independent noise realizations; noise/hallucination is
#      not. Needs two model outputs (orchestrated in the eval loop).
# ---------------------------------------------------------------------------

def _laplacian_response(img: Tensor) -> Tensor:
    lap = _ensure_laplacian(img.device, img.dtype)
    return F.conv2d(img, lap, padding=1)                       # [B,1,H,W] signed HF


def _hf_power(img: Tensor, mask: Tensor, mode: str = "grad") -> float:
    """Mean high-frequency power inside `mask`. mode='grad' (Sobel, noise-robust)
    or 'lap' (Laplacian)."""
    m = (mask > 0.5).float()
    if mode == "lap":
        r = _laplacian_response(img)
        p = r * r
    else:
        sx, sy = _ensure_sobel(img.device, img.dtype)
        gx = F.conv2d(img, sx, padding=1)
        gy = F.conv2d(img, sy, padding=1)
        p = gx * gx + gy * gy
    n = m.sum().clamp_min(1.0)
    return float(((p * m).sum() / n).item())


def _erode(mask: Tensor, iters: int) -> Tensor:
    """Binary erosion by `iters` (a voxel survives only if its whole
    (2*iters+1) neighbourhood is inside the mask)."""
    m = (mask > 0.5).float()
    if iters <= 0:
        return m
    k = 2 * int(iters) + 1
    return 1.0 - F.max_pool2d(1.0 - m, kernel_size=k, stride=1, padding=int(iters))


def tissue_csf_hf_ratio(pred: Tensor, tissue_mask: Tensor, csf_mask: Tensor,
                        erode_iters: int = 2, mode: str = "grad",
                        threshold: float = 0.5, min_csf_vox: int = 64,
                        eps: float = 1e-8) -> tuple:
    """High-frequency tissue-vs-noise ratio. Returns (ratio, tissue_hf_power).

    CSF ΔM≈0, so HF in the ERODED CSF interior is a pure in-image noise floor;
    HF in tissue (GM∪WM) is real texture + noise. ratio ≫ 1 ⇒ structured detail;
    ratio ≈ 1 ⇒ the extra high-freq is just noise (a noisy "sharp" image such as a
    naive frame mean scores ≈1). Report the pair: ratio flags real-vs-noise,
    tissue_hf_power flags over-smoothing (ratio high but power≈0 = smooth).

    CSF is ERODED (default 2 px) so ventricle/sulcal boundary edges (real
    structural gradients) do NOT contaminate the noise-floor estimate. Batch-
    pooled; returns nan ratio if the eroded CSF has < min_csf_vox voxels.
    """
    t = (tissue_mask > threshold).float()
    c = _erode((csf_mask > threshold).float(), erode_iters)
    e_t = _hf_power(pred, t, mode)
    if float(c.sum().item()) < int(min_csf_vox):
        return float("nan"), e_t
    e_c = _hf_power(pred, c, mode)
    return e_t / max(e_c, eps), e_t


def hf_consistency(pred_a: Tensor, pred_b: Tensor, mask: Optional[Tensor] = None,
                   mode: str = "lap", threshold: float = 0.5,
                   eps: float = 1e-8) -> tuple:
    """Split-half high-frequency reproducibility. Returns (corr, consistent_hf_power).

    pred_a, pred_b are two reconstructions from DISJOINT frame subsets (independent
    noise realizations). Correlate their high-frequency (Laplacian) content inside
    `mask` (brain). Real texture is reproducible ⇒ corr high; noise/hallucination
    is not ⇒ corr ≈ 0. `consistent_hf_power` = covariance = the amount of
    *reproducible* high-freq energy (≈0 for both pure noise AND over-smoothing;
    high only when texture is present AND reproducible).
    """
    if mode == "lap":
        ha = _laplacian_response(pred_a)
        hb = _laplacian_response(pred_b)
    else:
        sx, sy = _ensure_sobel(pred_a.device, pred_a.dtype)
        ha = (F.conv2d(pred_a, sx, padding=1) ** 2 + F.conv2d(pred_a, sy, padding=1) ** 2).sqrt()
        hb = (F.conv2d(pred_b, sx, padding=1) ** 2 + F.conv2d(pred_b, sy, padding=1) ** 2).sqrt()
    if mask is not None:
        sel = (mask > threshold).flatten()
        a = ha.flatten()[sel]
        b = hb.flatten()[sel]
    else:
        a = ha.flatten()
        b = hb.flatten()
    if a.numel() < 16:
        return float("nan"), float("nan")
    a = a - a.mean()
    b = b - b.mean()
    va = (a * a).mean()
    vb = (b * b).mean()
    cov = (a * b).mean()
    corr = float((cov / (va.sqrt() * vb.sqrt()).clamp_min(eps)).item())
    return corr, float(cov.item())


# ---------------------------------------------------------------------------
# Medical Physics additions (2026-08-26): difference-image SNR + mutual information
# ---------------------------------------------------------------------------

def snr_difference_image(recon_a: Tensor, recon_b: Tensor, roi: Tensor,
                         threshold: float = 0.5, eps: float = 1e-8) -> float:
    """SNR by the two-acquisition difference method (Dietrich et al., JMRI 2007).

    ``recon_a`` and ``recon_b`` are two reconstructions of the SAME subject built
    from DISJOINT frame subsets, so their difference contains noise only::

        signal = mean_ROI[ (a + b) / 2 ]
        sigma  = std_ROI [  a - b ] / sqrt(2)
        SNR    = signal / sigma

    Why not the textbook ``mean_ROI / std_background``: this pipeline runs with
    ``--premask_asl_inputs``, so the background is identically zero and its standard
    deviation is 0 — the background estimator is unusable here. The difference method
    needs no background at all, estimates the noise from the data itself, and is the
    accepted approach for accelerated acquisitions where the noise is non-stationary.

    The ROI is normally the grey-matter partial-volume map thresholded at >0.5
    (``gm_asl.nii.gz``); GM is where ASL signal actually lives, so a whole-brain ROI
    would dilute the numerator with WM and CSF.

    Args:
        recon_a, recon_b: [B,1,H,W] reconstructions from disjoint frame subsets
        roi:              [B,1,H,W] soft PV map in [0,1] (grey matter by default)
        threshold:        ROI is ``roi > threshold``
        eps:              guards the division

    Returns:
        SNR averaged over the batch (float). Samples whose ROI holds fewer than two
        voxels are skipped — a single voxel has no usable standard deviation.
    """
    if recon_a.shape != recon_b.shape or recon_a.shape != roi.shape:
        raise ValueError(f"shape mismatch: {tuple(recon_a.shape)} / "
                         f"{tuple(recon_b.shape)} / {tuple(roi.shape)}")
    m = (roi > threshold)
    mean_img = 0.5 * (recon_a + recon_b)
    diff_img = recon_a - recon_b
    vals = []
    for i in range(recon_a.shape[0]):
        sel = m[i]
        if int(sel.sum()) < 2:
            continue
        signal = mean_img[i][sel].mean()
        # unbiased std of the difference; /sqrt(2) because the difference of two
        # independent estimates has twice the variance of a single one
        sigma = diff_img[i][sel].std(unbiased=True) / (2.0 ** 0.5)
        vals.append(signal / sigma.clamp_min(eps))
    if not vals:
        return float("nan")
    return float(torch.stack(vals).mean())


def mutual_information(x: Tensor, y: Tensor, mask: Optional[Tensor] = None,
                       bins: int = 64, eps: float = 1e-12) -> float:
    """Mutual information between two images, in BITS, over the masked region.

    ``MI = sum_ij p_ij * log2( p_ij / (p_i * p_j) )`` from the joint intensity
    histogram. Each sample is binned over its OWN min/max range, so MI is invariant
    to a per-image affine rescale — which matters here because the ASL intensities
    are normalised per sample by an affine estimated from the visible frames.

    Two very different quantities are computed with this one function; say which one
    a reported number is:

      * ``MI(pred, reference)`` — a similarity metric. It is measured against the
        noisy 12-NEX average, so like SSIM and psnr_ref it is a **biased** reference
        metric and belongs in the supplementary set, never in a selection rule.
      * ``MI(pred, T1)`` — a **leakage** metric: how much anatomical information the
        perfusion output shares with the guidance image. The window fusion is a
        convex combination of ASL values, so T1 content cannot enter the output by
        construction; this is the empirical counterpart of that claim. Interpret it
        as a comparison across arms (does MI(pred,T1) rise when T1 keys are used?),
        not as an absolute number — perfusion and anatomy are genuinely correlated,
        so even a T1-free reconstruction has a non-zero value.

    Args:
        x, y:  [B,1,H,W]
        mask:  [B,1,H,W] in {0,1}; None uses every voxel
        bins:  joint-histogram resolution per axis

    Returns:
        MI in bits, averaged over the batch.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    vals = []
    for i in range(x.shape[0]):
        xi, yi = x[i].reshape(-1).float(), y[i].reshape(-1).float()
        if mask is not None:
            sel = mask[i].reshape(-1) > 0.5
            xi, yi = xi[sel], yi[sel]
        if xi.numel() < 2:
            continue
        # per-sample min/max binning => invariant to an affine rescale of either image
        def _idx(v: Tensor) -> Tensor:
            lo, hi = v.min(), v.max()
            if (hi - lo) < eps:
                return torch.zeros_like(v, dtype=torch.long)
            return ((v - lo) / (hi - lo) * (bins - 1)).round().long().clamp(0, bins - 1)
        xb, yb = _idx(xi), _idx(yi)
        joint = torch.zeros(bins * bins, device=x.device, dtype=torch.float32)
        joint.scatter_add_(0, xb * bins + yb, torch.ones_like(xb, dtype=torch.float32))
        joint = (joint / joint.sum().clamp_min(eps)).view(bins, bins)
        px = joint.sum(1, keepdim=True)
        py = joint.sum(0, keepdim=True)
        nz = joint > 0
        mi = (joint[nz] * torch.log2(joint[nz] / (px @ py)[nz].clamp_min(eps))).sum()
        vals.append(mi.clamp_min(0.0))
    if not vals:
        return float("nan")
    return float(torch.stack(vals).mean())
