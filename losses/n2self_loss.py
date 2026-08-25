from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def make_j_invariant_input(x: Tensor, mask_ratio: float = 0.05) -> Tuple[Tensor, Tensor]:
    """Produce a J-invariant input for Noise2Self (Batson & Royer 2019).

    A random subset of pixels is replaced with the local mean (3x3 neighbourhood
    excluding centre, approximated by full-window 3x3 mean) so the network
    cannot use the original pixel value at masked positions.

    Args:
        x: [B,1,H,W]
        mask_ratio: fraction of pixels to mask per image

    Returns:
        x_masked: [B,1,H,W] — same as x except masked positions replaced
        mask:     [B,1,H,W] in {0,1}, 1 = masked (loss-evaluated) position
    """
    B, C, H, W = x.shape
    mask = (torch.rand(B, 1, H, W, device=x.device) < mask_ratio).float()
    # Local mean from 3x3 average pool (replacement values for masked pixels)
    local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    x_masked = x * (1.0 - mask) + local_mean * mask
    return x_masked, mask


def j_invariant_l1(pred: Tensor, target: Tensor, mask: Tensor, brain_mask: Tensor) -> Tensor:
    """L1 evaluated only at masked (J-invariant) positions inside brain mask."""
    sel = mask * brain_mask
    return ((pred - target).abs() * sel).sum() / sel.sum().clamp_min(1.0)
