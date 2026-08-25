from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossWeights:
    w_n2n: float = 1.0
    w_t1: float = 0.03
    w_cos: float = 0.0            # disabled (v35+); cosine dissim was redundant once gradient + SSIM are in
    w_tv: float = 0.02            # masked total variation (interior gradients only)
    # NOTE: w_bg removed in v35 — backgrounds are handled by masking ASL inputs in
    # the runner before forward, which makes an explicit background-suppression
    # term redundant.
    w_grad: float = 0.5           # gradient/edge loss — preserves high-frequency detail under noisy targets
    w_ssim: float = 0.2           # 1-SSIM loss — captures local structure/contrast (complements per-pixel L1)
    w_seg: float = 0.1            # T1 GM/WM/CSF segmentation auxiliary loss (multi-task)
    w_contrast: float = 0.3       # contrast-aware boundary weighting on N2N L1 (tissue-aware)
    w_sure: float = 0.0           # Monte-Carlo SURE regularizer (no-GT denoising estimate)
    # L_sharp hinge (Plan B, 2026-05-22). Pushes lapvar(recon)/lapvar(target)
    # above tau_sharp_ratio when the model collapses into over-smooth attractors.
    # 0 disables; recommended 0.05 with tau≈0.35 matching constrained selection τS.
    w_sharp: float = 0.0
    tau_sharp_ratio: float = 0.35
    # DW-N2N (loss-level residual preservation, 2026-08). Deviation-weighted N2N: E=P·m̄
    # (tissue expectation) → d=|agg−E| → d_norm∈[0,1]; up-weight the N2N L1 by
    # (1+up_dev·d_norm) and relax the edge/grad smoothing by (1−relax_smooth·d_norm) at
    # candidate anomalies. E is NEVER added to the output (strict V=ASL, no T1 texture);
    # it only shapes the loss. d_norm is set_a/PV-derived ⇒ ⊥ set_b ⇒ N2N-unbiased. 0 = off.
    up_dev: float = 0.0
    relax_smooth: float = 0.0
    contrast_boost: float = 4.0   # boundary-region weight multiplier in contrast_weighted_l1
    sure_eps: float = 1e-3        # SURE divergence finite-difference step
    sure_warmup_steps: int = 30   # ramp w_sure 0 → target over first N steps
    sure_anneal_start: int = 0    # after this step, decay w_sure → sure_anneal_to (0 disables)
    sure_anneal_to: float = 0.0   # target w_sure after anneal completes
    mask_threshold: float = 0.05  # T1 intensity threshold for brain mask


def cosine_dissimilarity(x: Tensor, y: Tensor) -> Tensor:
    x_flat = x.flatten(1)
    y_flat = y.flatten(1)
    return 1.0 - F.cosine_similarity(x_flat, y_flat, dim=1).mean()


def total_variation_loss(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Total variation. If mask is given, only count gradients between two
    in-mask pixels — avoids the artificial huge edge at the brain mask boundary
    that dominates `TV(pred * mask)`."""
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    if mask is not None:
        m_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        m_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        loss_y = (dy * m_y).sum() / m_y.sum().clamp_min(1.0)
        loss_x = (dx * m_x).sum() / m_x.sum().clamp_min(1.0)
        return 0.5 * (loss_y + loss_x)
    return dy.mean() + dx.mean()


def _masked_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """L1 averaged over masked pixels (mask: [B,1,H,W] in {0,1})."""
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


def heteroscedastic_nll(mu: Tensor, logvar: Tensor, target: Tensor, mask: Tensor,
                        logvar_min: float = -8.0, logvar_max: float = 4.0) -> Tensor:
    """Per-pixel heteroscedastic Gaussian NLL (Kendall & Gal NeurIPS 2017).

        NLL = (target - μ)^2 / (2 σ²) + 0.5 · log σ²

    `logvar` parameterises log(σ²); clamped to [logvar_min, logvar_max] for
    numerical stability (σ² ∈ [3.4e-4, 54.6]). Constant 0.5·log(2π) is dropped
    (irrelevant for optimisation). Averaged inside the brain mask.
    """
    logvar = logvar.clamp(min=logvar_min, max=logvar_max)
    inv_var = torch.exp(-logvar)
    sq_err = (mu - target) ** 2
    nll = 0.5 * (sq_err * inv_var + logvar)
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    w_2d = g.unsqueeze(1) @ g.unsqueeze(0)            # [W, W]
    return w_2d.unsqueeze(0).unsqueeze(0)             # [1, 1, W, W]


_SSIM_WINDOW: Optional[Tensor] = None


def ssim_loss(pred: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    """1 - SSIM with 11x11 Gaussian window. Apply mask to pred/target beforehand."""
    global _SSIM_WINDOW
    if _SSIM_WINDOW is None or _SSIM_WINDOW.device != pred.device or _SSIM_WINDOW.dtype != pred.dtype:
        _SSIM_WINDOW = _gaussian_window(11, 1.5).to(device=pred.device, dtype=pred.dtype)
    w = _SSIM_WINDOW
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_p = F.conv2d(pred,   w, padding=5)
    mu_t = F.conv2d(target, w, padding=5)
    mu_p2 = mu_p * mu_p
    mu_t2 = mu_t * mu_t
    mu_pt = mu_p * mu_t
    sig_p2 = F.conv2d(pred * pred,     w, padding=5) - mu_p2
    sig_t2 = F.conv2d(target * target, w, padding=5) - mu_t2
    sig_pt = F.conv2d(pred * target,   w, padding=5) - mu_pt
    num   = (2 * mu_pt + c1) * (2 * sig_pt + c2)
    denom = (mu_p2 + mu_t2 + c1) * (sig_p2 + sig_t2 + c2)
    ssim_map = num / denom.clamp_min(1e-8)
    return 1.0 - ssim_map.mean()


def pv_l1_loss(seg_logits: Tensor, gm: Tensor, wm: Tensor, csf: Tensor,
               brain_mask: Tensor) -> Tensor:
    """3-class per-channel L1 (GM/WM/CSF only). Masked to brain.

    Better matched to soft PV targets than soft cross-entropy:
    - No entropy floor (PV CE bottoms out at H(target) ≈ 0.7)
    - Independent regression of GM / WM / CSF allows non-mutually-exclusive predictions
    - Naturally interpretable (pixel-wise PV error)
    """
    p = torch.sigmoid(seg_logits)                       # [B,3,H,W]
    target = torch.cat([gm, wm, csf], dim=1)            # [B,3,H,W]
    diff = (p - target).abs() * brain_mask              # broadcast brain_mask
    return diff.sum() / (brain_mask.sum().clamp_min(1.0) * 3.0)


def pv_l1_loss_4cls(seg_logits: Tensor, gm: Tensor, wm: Tensor, csf: Tensor,
                    brain_mask: Tensor = None) -> Tensor:
    """4-class per-channel L1 (GM/WM/CSF/BG). Evaluated over the full image.

    BG target is derived as 1 - clamp(GM+WM+CSF, 0, 1). The 4 classes are
    mutually exclusive and exhaustive everywhere, so we drop the brain mask
    (training BG class needs out-of-brain pixels).

    seg_logits: [B,4,H,W] in order GM, WM, CSF, BG.
    """
    p = torch.sigmoid(seg_logits)                                    # [B,4,H,W]
    bg = (1.0 - (gm + wm + csf)).clamp(0.0, 1.0)
    target = torch.cat([gm, wm, csf, bg], dim=1)                     # [B,4,H,W]
    return (p - target).abs().mean()


def _fd_grad(x: Tensor):
    """Forward finite-difference spatial gradient. Returns (gx, gy) with shapes
    [B,C,H,W-1] and [B,C,H-1,W]."""
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def pv_seg_probs(seg_logits: Tensor, softmax: bool = False) -> Tensor:
    """4-class PV prediction from logits. softmax=True -> the 4 classes sum to 1 per
    voxel (a simplex, matching the PV target); softmax=False -> independent per-class
    sigmoids (legacy, sum not constrained). Single source of truth so loss / metric /
    viz agree. seg_logits: [B,4,H,W] = GM,WM,CSF,BG."""
    return torch.softmax(seg_logits, dim=1) if softmax else torch.sigmoid(seg_logits)


def pv_l1_loss_4cls_sharp(seg_logits: Tensor, gm: Tensor, wm: Tensor, csf: Tensor,
                          brain_mask: Tensor,
                          w_dice: float = 0.5, w_part: float = 0.1,
                          brain_boost: float = 3.0, softmax: bool = False,
                          eps: float = 1e-6) -> Tensor:
    """Sharpened 4-class PV seg loss — cleans up plain ``pv_l1_loss_4cls`` on soft
    FAST PVE labels WITHOUT gradient matching (PV is physically fractional at
    interfaces, so matching boundary gradients would fight the soft targets and
    amplify high-freq noise):

      L = L1_brainweighted                       # focus capacity inside brain (BG still
                                                  #   supervised out-of-brain at weight 1)
        + w_dice · (1 - softDice(GM,WM,CSF))      # overlap-sensitive, no L1 hedging
        + w_part · |sum_c p_c - 1|_brain          # (sigmoid only) push onto the simplex

    ``softmax=True``: prediction = softmax over the 4 classes ⇒ per-voxel probs SUM
    TO 1 by construction (matches the simplex PV target, enforces class competition —
    rescues thin classes like CSF that independent sigmoids let collapse). The
    partition term is then a no-op and is dropped. ``softmax=False``: independent
    per-class sigmoids (legacy) + the soft partition penalty. seg_logits: [B,4,H,W].
    """
    p = pv_seg_probs(seg_logits, softmax=softmax)                    # [B,4,H,W]
    bg = (1.0 - (gm + wm + csf)).clamp(0.0, 1.0)
    target = torch.cat([gm, wm, csf, bg], dim=1)                     # [B,4,H,W]
    bm = brain_mask                                                  # [B,1,H,W]

    # brain-weighted L1 over all 4 classes
    w = 1.0 + brain_boost * bm
    l1 = ((p - target).abs() * w).sum() / (w.sum().clamp_min(1.0) * 4.0)

    # soft-Dice over the 3 brain tissues (brain-masked)
    p3 = p[:, :3] * bm; t3 = target[:, :3] * bm
    inter = (p3 * t3).sum(dim=(0, 2, 3))
    denom = p3.sum(dim=(0, 2, 3)) + t3.sum(dim=(0, 2, 3))
    dice = 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()

    loss = l1 + w_dice * dice
    if not softmax:   # softmax already sums to 1; partition penalty only meaningful for sigmoid
        part = ((p.sum(dim=1, keepdim=True) - 1.0).abs() * bm).sum() / bm.sum().clamp_min(1.0)
        loss = loss + w_part * part
    return loss


def soft_seg_loss(seg_logits: Tensor, gm: Tensor, wm: Tensor, csf: Tensor,
                  brain_mask: Tensor) -> Tensor:
    """Soft cross-entropy between predicted GM/WM/CSF probabilities and partial-volume targets.

    seg_logits: [B,3,H,W] (logits, channels = GM, WM, CSF)
    gm/wm/csf: [B,1,H,W] in [0,1] (partial volume maps; sum may be < 1 outside brain)
    brain_mask: [B,1,H,W] in {0,1}; only contribute inside brain.
    """
    log_p = F.log_softmax(seg_logits, dim=1)            # [B,3,H,W]
    target = torch.cat([gm, wm, csf], dim=1)            # [B,3,H,W]
    target_sum = target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    target = target / target_sum                        # row-normalise so it's a valid distribution
    ce_per_px = -(target * log_p).sum(dim=1, keepdim=True)  # [B,1,H,W]
    return (ce_per_px * brain_mask).sum() / brain_mask.sum().clamp_min(1.0)


def contrast_weighted_l1(pred: Tensor, target: Tensor, brain_mask: Tensor,
                         gm: Tensor, wm: Tensor, boost: float = 4.0) -> Tensor:
    """L1 with extra weight at GM/WM boundary pixels.

    Weight = 1 + boost * sqrt(gm * wm)  — peaks where both GM and WM are present
    (tissue boundary), elsewhere ≈ 1. Steers the model to preserve cortical contrast
    rather than smooth it out under noisy N2N targets.
    """
    boundary = (gm * wm).clamp_min(0).sqrt()            # [B,1,H,W]
    w = (1.0 + boost * boundary) * brain_mask
    return ((pred - target).abs() * w).sum() / w.sum().clamp_min(1.0)


def estimate_input_noise_var(set_a: Tensor, len_a: Optional[Tensor] = None) -> Tensor:
    """Estimate per-pixel noise variance of mean(setA).

    Uses unbiased frame-to-frame variance (1/(N-1) factor) divided by N (since
    mean(N indep observations) has variance σ²/N).

    set_a: [B, T, 1, H, W]
    Returns: [B, 1, H, W]
    """
    B, T = set_a.size(0), set_a.size(1)
    device = set_a.device
    if len_a is not None:
        valid = (torch.arange(T, device=device).unsqueeze(0) < len_a.unsqueeze(1)).float()
    else:
        valid = torch.ones(B, T, device=device)
    w = valid.view(B, T, 1, 1, 1)
    n = w.sum(dim=1).clamp_min(2.0)                                      # [B,1,H,W]
    mu = (set_a * w).sum(dim=1) / n                                      # [B,1,H,W]
    var_frame = ((set_a - mu.unsqueeze(1)) ** 2 * w).sum(dim=1) / (n - 1.0).clamp_min(1.0)
    sigma2_input = var_frame / n                                         # variance of mean
    return sigma2_input


def mc_sure_term(model, set_a: Tensor, set_b: Tensor, t1: Tensor,
                 brain_mask: Tensor,
                 len_a: Optional[Tensor] = None,
                 len_b: Optional[Tensor] = None,
                 mask_a: Optional[Tensor] = None,
                 mask_b: Optional[Tensor] = None,
                 eps: float = 1e-3) -> Tensor:
    """Monte-Carlo SURE term: estimates MSE(f(y), clean_signal) without clean GT.

    SURE = ||f(y) - y||² - σ²·N + 2·σ²·div(f)(y)
    where y = mean(set_a), σ² = var of input pixel, div(f) ≈ (z, f(y+εz)-f(y)) / ε.

    We approximate by perturbing set_a frames (pre-aggregator) and computing
    output divergence on asl_recon. Per-pixel σ² is estimated from frame variance.

    NOTE: only the perturbed forward needs gradient — original forward is detached
    in the divergence term but its prediction-vs-mean L2 still has grad.
    """
    sigma2 = estimate_input_noise_var(set_a, len_a)                      # [B,1,H,W]
    n_per_slice = brain_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)           # [B]

    # Original forward
    out_orig = model(set_a, set_b, t1, len_a=len_a, len_b=len_b,
                     mask_a=mask_a, mask_b=mask_b)
    f_y = out_orig["asl_recon"]                                          # [B,1,H,W]

    # mean(set_a) — y in SURE formulation
    if len_a is not None:
        v = (torch.arange(set_a.size(1), device=set_a.device).unsqueeze(0)
             < len_a.unsqueeze(1)).float().view(set_a.size(0), set_a.size(1), 1, 1, 1)
        y = (set_a * v).sum(dim=1) / v.sum(dim=1).clamp_min(1.0)
    else:
        y = set_a.mean(dim=1)

    # Data fidelity ||f(y) - y||² inside brain mask
    fid = ((f_y - y) ** 2 * brain_mask).sum(dim=(1, 2, 3))               # [B]

    # MC divergence — broadcast a single z ~ N(0,I) on y-space to all frames so
    # mean(set_a + ε·z_b) = y + ε·z exactly. Old impl used independent z_frames
    # per frame, which made z_y = mean(z_frames) have cov I/N → div underestimated
    # by factor N (= n_frames), causing SURE to lose its regularising power.
    z = torch.randn_like(y)                                              # [B,1,H,W], cov=I on y-space
    z_b = z.unsqueeze(1).expand_as(set_a)                                # broadcast to all frames
    f_pert = model(set_a + eps * z_b, set_b, t1,
                   len_a=len_a, len_b=len_b, mask_a=mask_a, mask_b=mask_b)["asl_recon"]
    div_per_px = z * (f_pert - f_y) / max(eps, 1e-12)                    # [B,1,H,W]

    # SURE per slice (within brain mask):
    sigma_sum = (sigma2 * brain_mask).sum(dim=(1, 2, 3))                 # Σ σ²
    div_sum = (sigma2 * div_per_px * brain_mask).sum(dim=(1, 2, 3))      # Σ σ²·div
    sure = (fid - sigma_sum + 2.0 * div_sum) / n_per_slice
    return sure.mean()


_LAP_KERNEL: Optional[Tensor] = None


def _ensure_laplacian_kernel(device, dtype) -> Tensor:
    global _LAP_KERNEL
    if (_LAP_KERNEL is None or _LAP_KERNEL.device != device
            or _LAP_KERNEL.dtype != dtype):
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=device, dtype=dtype)
        _LAP_KERNEL = k.view(1, 1, 3, 3)
    return _LAP_KERNEL


def laplacian_variance_t(image: Tensor, mask: Tensor) -> Tensor:
    """Differentiable laplacian variance inside `mask`. Returns a scalar tensor.

    Mirrors `utils.metrics.laplacian_variance` formula but keeps the autograd
    graph alive so it can be used inside a training loss.
    """
    lap = _ensure_laplacian_kernel(image.device, image.dtype)
    resp = F.conv2d(image, lap, padding=1)                           # [B,1,H,W]
    m = mask.float()
    n = m.sum().clamp_min(1.0)
    mean = (resp * m).sum() / n
    var = ((resp - mean) ** 2 * m).sum() / n
    return var


def sharpness_hinge_loss(pred: Tensor, target: Tensor, mask: Tensor,
                         tau_ratio: float) -> Tensor:
    """Hinge loss enforcing lapvar(pred) / lapvar(target_detached) ≥ tau_ratio.

    Penalises over-smooth collapse (the failure mode that drove lapvR plateau at
    ~0.33 in the previous training run). target's lapvar is detached so the
    pressure only flows back through pred.
    """
    lap_pred = laplacian_variance_t(pred, mask)
    lap_target = laplacian_variance_t(target, mask).detach()
    ratio = lap_pred / lap_target.clamp_min(1e-8)
    return F.relu(tau_ratio - ratio) ** 2


def gradient_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    L1 between image gradients (edge-preservation loss). Compatible with N2N:
    E[grad(noisy_target)] = grad(true_signal) when noise has zero spatial mean.
    """
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    m_y = mask[:, :, 1:, :]
    m_x = mask[:, :, :, 1:]
    loss_y = ((dy_p - dy_t).abs() * m_y).sum() / m_y.sum().clamp_min(1.0)
    loss_x = ((dx_p - dx_t).abs() * m_x).sum() / m_x.sum().clamp_min(1.0)
    return 0.5 * (loss_y + loss_x)


def _masked_l1_weighted(pred: Tensor, target: Tensor, mask: Tensor, w: Tensor) -> Tensor:
    """Masked L1 with an extra per-pixel weight ``w`` (>=0). DW-N2N up-weights the
    reconstruction at candidate anomalies (where the ASL deviates from tissue
    expectation). ``w`` is computed from set_a + the frozen PV only, so it is
    independent of the set_b target and the N2N minimiser stays unbiased."""
    mw = mask * w
    return ((pred - target).abs() * mw).sum() / mw.sum().clamp_min(1.0)


def gradient_l1_weighted(pred: Tensor, target: Tensor, mask: Tensor, s: Tensor) -> Tensor:
    """``gradient_l1`` with a per-pixel smoothing weight ``s`` in [0,1]. ``s<1``
    RELAXES the edge/smoothing pressure at those pixels so a reproducible focal
    deviation is not smoothed away (DW-N2N smoothing relaxation)."""
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    m_y = mask[:, :, 1:, :] * s[:, :, 1:, :]
    m_x = mask[:, :, :, 1:] * s[:, :, :, 1:]
    loss_y = ((dy_p - dy_t).abs() * m_y).sum() / m_y.sum().clamp_min(1.0)
    loss_x = ((dx_p - dx_t).abs() * m_x).sum() / m_x.sum().clamp_min(1.0)
    return 0.5 * (loss_y + loss_x)


class ASLN2NLoss(nn.Module):
    """
    Loss for the ASL T1 denoiser trained with Noise2Noise.

    Single-direction N2N:
      - set_a → aggregator → encoder → decoder → asl_recon
      - target = direct unweighted mean(set_b)
      - loss_n2n = L1(asl_recon, mean(set_b))

    Weight mapping from config:
      w_n2n         -> w_n2n     (N2N reconstruction)
      w_anat_roi    -> w_t1      (T1 reconstruction fidelity)
      lambda_cos    -> w_cos     (cosine dissimilarity on N2N output)
      w_tv          -> w_tv      (0 by default)
    """

    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights

    @classmethod
    def from_config(cls, cfg) -> "ASLN2NLoss":
        p = cfg.asl_denoiser_train_params
        # NOTE: w_bg removed (backgrounds handled by input masking in runner).
        # If an old config still defines w_bg it is silently ignored.
        weights = LossWeights(
            w_n2n=float(getattr(p, "w_n2n", 1.0)),
            w_t1=float(getattr(p, "w_anat_roi", 0.03)),
            w_cos=float(getattr(p, "lambda_cos", 0.0)),
            w_tv=float(getattr(p, "w_tv", 0.02)),
            w_grad=float(getattr(p, "w_grad", 0.5)),
            w_ssim=float(getattr(p, "w_ssim", 0.2)),
            w_seg=float(getattr(p, "w_seg", 0.1)),
            w_contrast=float(getattr(p, "w_contrast", 0.3)),
            w_sure=float(getattr(p, "w_sure", 0.0)),
            w_sharp=float(getattr(p, "w_sharp", 0.0)),
            tau_sharp_ratio=float(getattr(p, "tau_sharp_ratio", 0.35)),
            contrast_boost=float(getattr(p, "contrast_boost", 4.0)),
            sure_eps=float(getattr(p, "sure_eps", 1e-3)),
            sure_warmup_steps=int(getattr(p, "sure_warmup_steps", 30)),
            sure_anneal_start=int(getattr(p, "sure_anneal_start", 0)),
            sure_anneal_to=float(getattr(p, "sure_anneal_to", 0.0)),
        )
        return cls(weights)

    def forward(
        self,
        outputs: Dict[str, Tensor],
        t1_target: Tensor,
        gm: Optional[Tensor] = None,
        wm: Optional[Tensor] = None,
        csf: Optional[Tensor] = None,
        **_kwargs,
    ) -> Tuple[Tensor, Dict[str, float]]:
        target   = outputs["target"].detach()
        pred     = outputs["asl_recon"]
        logvar   = outputs.get("asl_logvar")    # v37+: optional [B,1,H,W]
        t1_recon = outputs.get("t1_recon")
        t1_seg   = outputs.get("t1_seg")

        # Brain mask: focus loss inside the head, exclude near-zero background.
        mask = (t1_target > self.weights.mask_threshold).float()

        # DW-N2N: optional per-pixel normalized deviation d_norm∈[0,1] (from the runner:
        # |agg − P·m̄| scaled to [0,1]). Up-weights the N2N L1 and relaxes the edge/grad
        # smoothing at candidate anomalies; never enters pred (V=ASL preserved).
        dev_weight = _kwargs.get("dev_weight")

        # Main N2N reconstruction.
        # If a heteroscedastic head is present (asl_logvar not None), use the
        # Gaussian NLL — this trains both the prediction μ and per-pixel σ²
        # (uncertainty map) jointly. Otherwise fall back to masked L1 (v34/v35/v36).
        if logvar is not None:
            loss_n2n = heteroscedastic_nll(pred, logvar, target, mask)
        elif dev_weight is not None and self.weights.up_dev > 0.0:
            loss_n2n = _masked_l1_weighted(
                pred, target, mask, 1.0 + self.weights.up_dev * dev_weight)
        else:
            loss_n2n = _masked_l1(pred, target, mask)

        # T1 reconstruction fidelity (masked L1) — only if t1_recon is provided
        if t1_recon is not None:
            loss_t1 = _masked_l1(t1_recon, t1_target, mask)
        else:
            loss_t1 = pred.new_zeros(())

        # Total variation on prediction, **interior gradients only** (no spurious
        # large gradient at brain mask boundary).
        loss_tv = total_variation_loss(pred, mask)

        # Gradient/edge loss — preserves high-frequency detail under noisy N2N targets.
        # DW-N2N: relax this smoothing pressure at candidate anomalies (s<1 there).
        if dev_weight is not None and self.weights.relax_smooth > 0.0:
            ss = (1.0 - self.weights.relax_smooth * dev_weight).clamp_min(0.0)
            loss_grad = gradient_l1_weighted(pred, target, mask, ss)
        else:
            loss_grad = gradient_l1(pred, target, mask)

        # SSIM loss on masked images — captures local contrast / structure
        loss_ssim = ssim_loss(pred * mask, target * mask)

        # Multi-task: T1 segmentation auxiliary + contrast-weighted L1 on ASL recon
        loss_seg = pred.new_zeros(())
        loss_contrast = pred.new_zeros(())
        if t1_seg is not None and gm is not None and wm is not None and csf is not None:
            # Auto-detect: 4 channels = (GM/WM/CSF/BG) → pv_l1_4cls; else legacy 3-class CE
            if t1_seg.shape[1] == 4:
                loss_seg = pv_l1_loss_4cls(t1_seg, gm, wm, csf)
            else:
                loss_seg = soft_seg_loss(t1_seg, gm, wm, csf, mask)
            loss_contrast = contrast_weighted_l1(
                pred, target, mask, gm, wm, boost=self.weights.contrast_boost
            )

        # L_sharp hinge: only computed if w_sharp > 0. Pulls pred lapvar back up
        # when the ratio against (noisy) target lapvar drops below tau_sharp_ratio.
        # See `sharpness_hinge_loss` docstring.
        if self.weights.w_sharp > 0.0:
            loss_sharp = sharpness_hinge_loss(
                pred, target, mask, self.weights.tau_sharp_ratio
            )
        else:
            loss_sharp = pred.new_zeros(())

        # NOTE: cosine-dissim term and bg-suppression term were removed in v35.
        # bg is handled by masking ASL inputs in the runner; cos_dissim was
        # redundant with gradient + SSIM losses.
        total = (
            self.weights.w_n2n      * loss_n2n
            + self.weights.w_t1       * loss_t1
            + self.weights.w_tv       * loss_tv
            + self.weights.w_grad     * loss_grad
            + self.weights.w_ssim     * loss_ssim
            + self.weights.w_seg      * loss_seg
            + self.weights.w_contrast * loss_contrast
            + self.weights.w_sharp    * loss_sharp
        )

        stats = {
            "loss":           float(total.detach().item()),
            "loss_n2n":       float(loss_n2n.detach().item()),
            "loss_t1":        float(loss_t1.detach().item()),
            "loss_tv":        float(loss_tv.detach().item()),
            "loss_grad":      float(loss_grad.detach().item()),
            "loss_ssim":      float(loss_ssim.detach().item()),
            "loss_seg":       float(loss_seg.detach().item()),
            "loss_contrast":  float(loss_contrast.detach().item()),
            "loss_sharp":     float(loss_sharp.detach().item()),
        }
        return total, stats
