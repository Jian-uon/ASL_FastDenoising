"""Standalone T1 branch: encoder + 4-channel decoder (recon + GM/WM/CSF).

Built from the same ConvEncoder2D / ConvDecoderWithSkips2D blocks used by
ASLT1Denoiser, so its state_dict can be transferred to the full model's
`t1_encoder` and `t1_decoder` after pretraining.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

try:
    from .blocks import ConvDecoderWithSkips2D, ConvEncoder2D
except Exception:
    from blocks import ConvDecoderWithSkips2D, ConvEncoder2D  # type: ignore


class T1Branch(nn.Module):
    """T1 encoder + U-Net decoder, pretrained on one of two pretext tasks.

    ``task='seg'`` (default): 4-channel head = GM/WM/CSF/BG logits (PV
    segmentation). BG target = 1 - clamp(GM+WM+CSF, 0, 1). Tissue-class-abstracted
    features — the original stage-1 (CLAUDE.md §3).

    ``task='recon'``: 1-channel head reconstructs the T1 image (autoencoder
    pretext). Appearance-preserving anatomical features; no segmentation labels
    needed. The V=ASL content-isolation invariant is UNAFFECTED (T1 features are
    still only Keys/gates downstream, never Values).

    Only the decoder head width (4 vs 1) differs between tasks; the encoder
    (which is what stage-2 consumes as multi-scale guidance) is identical.
    """

    def __init__(
        self,
        hw: int = 128,
        in_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        skip_dropout: float = 0.3,
        recon_activation: Optional[nn.Module] = None,
        use_wavelet: bool = False,
        task: str = "seg",
    ) -> None:
        super().__init__()
        self.task = str(task)
        if self.task not in ("seg", "recon"):
            raise ValueError(f"T1Branch task must be 'seg' or 'recon', got {task!r}")
        out_ch = 1 if self.task == "recon" else 4
        self.t1_encoder = ConvEncoder2D(in_ch=in_ch, base_ch=base_ch, depth=depth, use_wavelet=use_wavelet)
        self.t1_decoder = ConvDecoderWithSkips2D(
            in_ch=self.t1_encoder.out_ch, out_ch=out_ch, out_hw=hw,
            base_ch=base_ch, depth=depth,
            skip_dropout=skip_dropout, final_activation=None,
            use_wavelet=use_wavelet,
        )
        self.recon_activation = recon_activation

    def forward(self, t1: Tensor) -> Dict[str, Tensor]:
        feat, _, skips = self.t1_encoder(t1)
        out = self.t1_decoder(feat, skips)
        if self.task == "recon":
            if self.recon_activation is not None:
                out = self.recon_activation(out)
            return {"t1_recon": out}                    # [B,1,H,W]
        return {"t1_seg": out}                          # [B,4,H,W] — GM/WM/CSF/BG
