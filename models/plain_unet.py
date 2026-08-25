from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

try:
    from .blocks import ConvDecoderWithSkips2D, ConvEncoder2D
except Exception:
    from blocks import ConvDecoderWithSkips2D, ConvEncoder2D  # type: ignore


class PlainUNet2D(nn.Module):
    """Single-modality U-Net used by trainable baselines (sup / n2n / n2self).

    No SetTransformer, no T1, no cross-modal fusion.
    Input  : [B, in_ch, H, W] (already aggregated mean image)
    Output : [B, out_ch, H, W]
    """

    def __init__(
        self,
        hw: int = 128,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        skip_dropout: float = 0.3,
        final_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder2D(in_ch=in_ch, base_ch=base_ch, depth=depth)
        self.decoder = ConvDecoderWithSkips2D(
            in_ch=self.encoder.out_ch, out_ch=out_ch, out_hw=hw,
            base_ch=base_ch, depth=depth,
            skip_dropout=skip_dropout, final_activation=final_activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        feat, _, skips = self.encoder(x)
        return self.decoder(feat, skips)
