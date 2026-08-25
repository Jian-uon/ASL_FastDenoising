from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# v42j (2026-05-19): NAFNet-style decoder fusion (Chen et al. ECCV 2022 winner).
#
# Standard U-Net concat-fuse pattern  `Conv1x1(cat[up, skip])`  has been the
# noise leak path in v42a..v42i: the high-res ASL skip carries stem(agg) noise
# directly to the decoder output. In self-supervised regime (N2N target also
# noisy), L1+grad loss can't tell "real edge" from "set_a noise edge" → noise
# is preserved.
#
# NAFNet replaces the conv+activation fuse with a SimpleGate-based block:
#     conv1x1(C → 2C) → DWConv3x3 → SimpleGate (chunk × multiply) → SCA → conv1x1
#     residual with β=0 init → identity at init (trainability invariant).
#
# Empirically NAFNet beats Restormer / Uformer / HINet on SIDD/GoPro with
# fewer params; the SimpleGate's multiplicative non-linearity + SCA's channel
# gating together act as a strong implicit denoising filter on the fused
# feature.
# -----------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm applied per-pixel over channel dim. Required by NAFBlock."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W] → normalise over C
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """NAFNet's SimpleGate (Chen et al. ECCV 2022 §3.2): chunk(x,2) → x1 * x2.
    Replaces GELU/ReLU; halves channel dim."""

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class MDTA(nn.Module):
    """Restormer's Multi-Dconv head Transposed Attention (Zamir et al. CVPR 2022).

    Channel-axis self-attention: Q@Kᵀ produces a (C/h)×(C/h) gain matrix per head,
    cost O(C²·HW) instead of O((HW)²·C). The DConv on QKV injects local spatial
    inductive bias before the channel attention; L2-norm on Q/K + learnable
    temperature stabilises the softmax. Used here as a drop-in replacement for
    NAFNet's SCA at decoder fusion points (v42k)."""

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = True) -> None:
        super().__init__()
        # Round heads down to a divisor of dim so view() is exact.
        h = int(num_heads)
        while h > 1 and dim % h != 0:
            h -= 1
        self.num_heads = h
        self.temperature = nn.Parameter(torch.ones(h, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        hd = c // self.num_heads
        q = q.reshape(b, self.num_heads, hd, h * w)
        k = k.reshape(b, self.num_heads, hd, h * w)
        v = v.reshape(b, self.num_heads, hd, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, h, hd, hd]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class NAFBlock(nn.Module):
    """Single NAFNet block: pre-norm → conv → DWConv → SimpleGate → SCA → conv,
    plus an FFN with same gating; both halves residual with β/γ=0 init
    (identity at start → trainability invariant).

    v42k (2026-05-19): `use_mdta=True` swaps the SCA gain for Restormer's MDTA
    in the spatial-mixing branch. β=0 outer residual still gates the whole
    branch to zero at init, so trainability invariant holds even with MDTA's
    non-zero internal projections.
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2,
                 use_mdta: bool = False, mdta_heads: int = 4) -> None:
        super().__init__()
        dw_ch = c * dw_expand
        ffn_ch = c * ffn_expand
        mid_ch = dw_ch // 2
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_ch, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                               groups=dw_ch, bias=True)  # depthwise
        self.sg = SimpleGate()
        self.use_mdta = bool(use_mdta)
        if self.use_mdta:
            self.mdta = MDTA(mid_ch, num_heads=mdta_heads)
            self.sca = None
        else:
            # SCA: Simplified Channel Attention (NAFNet's SE replacement)
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_ch, mid_ch, kernel_size=1, bias=True),
            )
            self.mdta = None
        self.conv3 = nn.Conv2d(mid_ch, c, kernel_size=1, bias=True)

        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_ch, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, bias=True)

        # NAFNet's gated residual: β, γ initialised to 0 → block is identity at init.
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # Block 1: spatial mixing with SimpleGate + (SCA | MDTA)
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        if self.use_mdta:
            y = self.mdta(y)
        else:
            y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        # Block 2: FFN with SimpleGate
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        x = x + y * self.gamma
        return x


class NAFFusionBlock(nn.Module):
    """Decoder fusion: project cat(up_x, skip) → out_ch via 1×1, then one
    NAFBlock for refinement. Replaces the v42a..v42i `Conv1x1 + GN + SiLU`
    fuse pattern. Trainability invariant: NAFBlock(β=γ=0 init) is identity →
    NAFFusionBlock ≈ Conv1x1 at init.

    v42k: `use_mdta=True` activates Restormer MDTA inside the inner NAFBlock."""

    def __init__(self, in_ch: int, out_ch: int, use_mdta: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.naf = NAFBlock(out_ch, use_mdta=use_mdta)

    def forward(self, x: Tensor) -> Tensor:
        return self.naf(self.proj(x))


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        num_groups = min(groups, out_ch)
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            activation if activation is not None else nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 8) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, stride=stride, groups=groups)
        self.conv2 = ConvNormAct(out_ch, out_ch, stride=1, groups=groups, activation=nn.Identity())
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, stride=2, groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        self.conv = ResidualBlock(in_ch, out_ch, stride=1, groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


# -----------------------------------------------------------------------------
# Wavelet down/up blocks — replace stride-2 conv / bilinear interp with DWT/IDWT
# -----------------------------------------------------------------------------
# The DWT is information-preserving (orthogonal Haar): no aliasing on downsample,
# no interpolation artefacts on upsample. The 4 sub-bands (LL, LH, HL, HH) are
# treated as channel-stacked features for downstream conv processing. Improves
# high-frequency detail retention vs. stride-conv pooling for low-resolution
# medical images such as 128x128 ASL.

class WaveletDownBlock(nn.Module):
    """DWT-based 2x downsample. in_ch -> out_ch with H,W -> H/2, W/2.

    Forward: x (in_ch) -> Haar DWT -> [LL, LH, HL, HH] (4*in_ch) at H/2 x W/2
             -> ResidualBlock(4*in_ch -> out_ch).
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, wave: str = "haar") -> None:
        super().__init__()
        from pytorch_wavelets import DWTForward
        self.dwt = DWTForward(J=1, mode="zero", wave=wave)
        self.conv = ResidualBlock(in_ch * 4, out_ch, stride=1, groups=groups)

    def forward(self, x: Tensor) -> Tensor:
        yl, yh = self.dwt(x)
        # yl: [B, C, H/2, W/2]; yh[0]: [B, C, 3, H/2, W/2] (LH, HL, HH)
        h = yh[0]
        b, c, three, hh, ww = h.shape
        h = h.reshape(b, c * three, hh, ww)
        out = torch.cat([yl, h], dim=1)  # [B, 4C, H/2, W/2]
        return self.conv(out)


class WaveletUpBlock(nn.Module):
    """IDWT-based 2x upsample. in_ch -> out_ch with H,W -> 2H, 2W.

    Forward: x (in_ch) -> 1x1 conv -> 4*out_ch -> split LL + 3 bands ->
             IDWT -> out_ch at 2H x 2W -> ResidualBlock for refinement.
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, wave: str = "haar") -> None:
        super().__init__()
        from pytorch_wavelets import DWTInverse
        # Project to 4*out_ch (LL + LH + HL + HH bands)
        self.proj = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, bias=False)
        self.idwt = DWTInverse(mode="zero", wave=wave)
        self.refine = ResidualBlock(out_ch, out_ch, stride=1, groups=groups)
        self.out_ch = out_ch

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # [B, 4*out_ch, H, W]
        c = self.out_ch
        b, _, hh, ww = x.shape
        yl = x[:, :c]                                          # [B, c, H, W]
        yh = x[:, c:].reshape(b, c, 3, hh, ww)                 # [B, c, 3, H, W]
        out = self.idwt((yl, [yh]))                            # [B, c, 2H, 2W]
        return self.refine(out)


class ConvEncoder2D(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32, depth: int = 4,
                 groups: int = 8, use_wavelet: bool = False) -> None:
        super().__init__()
        channels = [base_ch * (2 ** i) for i in range(depth)]
        self.stem = ResidualBlock(in_ch, channels[0], stride=1, groups=groups)
        self.downs = nn.ModuleList()
        prev = channels[0]
        DownCls = WaveletDownBlock if use_wavelet else DownBlock
        for ch in channels[1:]:
            self.downs.append(DownCls(prev, ch, groups=groups))
            prev = ch
        self.out_ch = prev
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]:
        skips = []
        feat = self.stem(x)
        skips.append(feat)
        for blk in self.downs:
            feat = blk(feat)
            skips.append(feat)
        vec = self.pool(feat).flatten(1)
        return feat, vec, skips


class ConvDecoder2D(nn.Module):
    """Decoder takes a bottleneck feature map [B, in_ch, H', W'] and upsamples to image."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 1,
        out_hw: int = 128,
        base_ch: int = 32,
        depth: int = 4,
        groups: int = 8,
        final_activation: Optional[nn.Module] = None,
        use_wavelet: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.out_hw = int(out_hw)
        channels = [base_ch * (2 ** i) for i in range(depth)]
        bottleneck_ch = channels[-1]
        bottleneck_hw = self.out_hw // (2 ** (depth - 1))
        if bottleneck_hw < 2:
            raise ValueError(f"Output HW={out_hw} is too small for depth={depth}.")
        # Optional 1x1 adapter if input channel count differs from bottleneck channel.
        self.adapter = (
            nn.Conv2d(in_ch, bottleneck_ch, kernel_size=1, bias=False)
            if in_ch != bottleneck_ch else nn.Identity()
        )
        UpCls = WaveletUpBlock if use_wavelet else UpBlock
        ups = []
        ch = bottleneck_ch
        for next_ch in reversed(channels[:-1]):
            ups.append(UpCls(ch, next_ch, groups=groups))
            ch = next_ch
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        self.final_activation = final_activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.adapter(x)
        for blk in self.ups:
            x = blk(x)
        x = self.head(x)
        if x.shape[-1] != self.out_hw or x.shape[-2] != self.out_hw:
            x = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="bilinear", align_corners=False)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class ConvDecoderWithSkips2D(nn.Module):
    """
    Decoder taking a bottleneck feature map plus intra-modal U-Net skip connections.
    Used by both T1 (intra-modal only) and ASL branches.

    Optional multi-scale T1 cross-fusion (use_t1_cross_fusion=True, ASL only):
      After each up-block + same-modality skip fusion, optionally cross-attend to
      the T1 skip at the same scale, with tissue-class similarity bias from t1_seg.
      Anti-hallucination preserved (V=ASL inside cross-attention). Applied only
      at scales with H*W <= attn_max_tokens (default 1024) to bound memory —
      attention is O(N^2) so 64x64 (N=4096) and above are skipped by default.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 1,
        out_hw: int = 128,
        base_ch: int = 32,
        depth: int = 4,
        groups: int = 8,
        final_activation: Optional[nn.Module] = None,
        skip_dropout: float = 0.3,
        use_wavelet: bool = False,
        use_t1_cross_fusion: bool = False,
        t1_attn_heads: int = 4,
        t1_attn_dropout: float = 0.2,
        t1_attn_max_tokens: int = 1024,
        use_film: bool = False,
        film_n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.out_hw = int(out_hw)
        channels = [base_ch * (2 ** i) for i in range(depth)]
        bottleneck_ch = channels[-1]
        bottleneck_hw = self.out_hw // (2 ** (depth - 1))
        if bottleneck_hw < 2:
            raise ValueError(f"Output HW={out_hw} is too small for depth={depth}.")
        self.adapter = (
            nn.Conv2d(in_ch, bottleneck_ch, kernel_size=1, bias=False)
            if in_ch != bottleneck_ch else nn.Identity()
        )
        UpCls = WaveletUpBlock if use_wavelet else UpBlock
        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.t1_fusions = nn.ModuleList() if use_t1_cross_fusion else None
        self.films = nn.ModuleList() if use_film else None
        ch = bottleneck_ch
        # Decoder level i corresponds to spatial size out_hw / 2^(depth - 2 - i)
        # depth=4, out_hw=128: levels produce 32, 64, 128
        for i, next_ch in enumerate(reversed(channels[:-1])):
            self.ups.append(UpCls(ch, next_ch, groups=groups))
            num_g = min(groups, next_ch)
            while next_ch % num_g != 0 and num_g > 1:
                num_g -= 1
            self.fusions.append(nn.Sequential(
                nn.Conv2d(next_ch * 2, next_ch, kernel_size=1, bias=False),
                nn.GroupNorm(num_g, next_ch),
                nn.SiLU(inplace=True),
            ))
            if use_t1_cross_fusion:
                level_hw = self.out_hw // (2 ** (depth - 2 - i))
                if level_hw * level_hw <= int(t1_attn_max_tokens):
                    heads = int(t1_attn_heads)
                    while next_ch % heads != 0 and heads > 1:
                        heads //= 2
                    self.t1_fusions.append(CrossModalFusion(
                        in_ch=next_ch, n_heads=heads, dropout=float(t1_attn_dropout),
                    ))
                else:
                    self.t1_fusions.append(nn.Identity())
            if use_film:
                self.films.append(TissueFiLM(n_classes=int(film_n_classes), ch=next_ch))
            ch = next_ch
        self.head = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        self.final_activation = final_activation
        self.skip_drop = nn.Dropout2d(p=skip_dropout)
        self.use_t1_cross_fusion = bool(use_t1_cross_fusion)
        self.use_film = bool(use_film)

    def forward(
        self,
        x: Tensor,
        skips: List[Tensor],
        t1_skips: Optional[List[Tensor]] = None,
        t1_seg: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.adapter(x)
        n = len(skips) - 1
        for i, (up, fuse) in enumerate(zip(self.ups, self.fusions)):
            x = up(x)
            skip = self.skip_drop(skips[n - 1 - i])
            x = fuse(torch.cat([x, skip], dim=1))
            # FiLM tissue modulation BEFORE cross-fusion (modulates the asl
            # feature stream that cross-attn then queries against T1).
            if self.use_film and self.films is not None and t1_seg is not None:
                x = self.films[i](x, t1_seg)
            if self.use_t1_cross_fusion and self.t1_fusions is not None and t1_skips is not None:
                fusion = self.t1_fusions[i]
                if not isinstance(fusion, nn.Identity):
                    t1_skip = t1_skips[n - 1 - i]
                    x = fusion(x, t1_skip, tissue_seg=t1_seg)
        x = self.head(x)
        if x.shape[-1] != self.out_hw or x.shape[-2] != self.out_hw:
            x = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="bilinear", align_corners=False)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class FrameReliabilityAggregator(nn.Module):
    """Frame Reliability Aggregator (FRA) — paper name (was SetTransformerAggregator).

    Replaces LinearFrameAggregator.
    Multi-head self-attention across T noisy ASL frames identifies consistent signal.
    Output: weighted sum in image space — same interface as LinearFrameAggregator.
    """

    def __init__(self, in_ch: int = 1, hidden_ch: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        assert hidden_ch % n_heads == 0, f"hidden_ch={hidden_ch} must be divisible by n_heads={n_heads}"
        # Per-frame lightweight CNN -> global pool -> D-dim feature
        self.frame_enc = nn.Sequential(
            ConvNormAct(in_ch, hidden_ch // 2, kernel_size=3, stride=2),
            ConvNormAct(hidden_ch // 2, hidden_ch, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_ch,
            nhead=n_heads,
            dim_feedforward=hidden_ch * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        # Predicts per-frame log-variance; weights = softmax(-log_var) is the
        # BLUE / inverse-variance-weighted estimator under a Gaussian noise model.
        self.log_var_head = nn.Linear(hidden_ch, 1)

    @staticmethod
    def _mask_from_lengths(lengths: Tensor, t: int, device) -> Tensor:
        return torch.arange(t, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

    def forward(
        self,
        frames: Tensor,
        lengths: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape

        flat = frames.reshape(b * t, c, h, w)
        feat = self.frame_enc(flat).flatten(1)  # [B*T, hidden_ch]
        feat = feat.view(b, t, -1)              # [B, T, hidden_ch]

        if mask is not None and lengths is not None:
            raise ValueError("Provide either mask or lengths, not both.")
        if mask is not None:
            invalid = mask.bool()
        elif lengths is not None:
            invalid = self._mask_from_lengths(lengths.to(frames.device), t, frames.device)
        else:
            invalid = torch.zeros(b, t, dtype=torch.bool, device=frames.device)

        feat = self.transformer(feat, src_key_padding_mask=invalid)  # [B, T, hidden_ch]

        # Predict per-frame log-variance; weights = softmax(-log_var) (BLUE).
        log_var = self.log_var_head(feat).squeeze(-1)               # [B, T]
        logits = (-log_var).masked_fill(invalid, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights.masked_fill(invalid, 0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        agg = (frames * weights[:, :, None, None, None]).sum(dim=1)  # [B, C, H, W]
        return agg, weights


class SpatialVaryingFrameWeighting(nn.Module):
    """v37+ aggregator — per-pixel BLUE-style frame weighting.

    Replaces SetTransformerAggregator's scalar per-frame weighting [B,T] with
    a spatially-varying per-pixel weighting [B,T,H,W]. Motion artifacts in ASL
    are usually local (a frame can be reliable in some regions but corrupted
    in others); a single scalar per frame can't capture this.

    Anti-lesion-suppression safe-by-design:
      - The log-variance head sees ONLY the per-frame deviation
        ``frames[t] - mean_T(frames)``, never the raw signal value.
      - Therefore weights cannot depend on "how unusual the absolute signal
        looks" (which would suppress real-but-atypical perfusion in lesions).
      - Weights only depend on temporal consistency: a frame inconsistent
        with the others at (h,w) gets low weight; a consistent frame keeps
        its weight regardless of the absolute signal level.
      - Lesion behaviour: all frames consistently show abnormal CBF →
        small per-frame deviation → high weight → lesion preserved.

    Returns the same ``(agg, weights)`` interface as SetTransformerAggregator,
    with ``weights`` shape extended to ``[B, T, H, W]``.
    """

    def __init__(self, in_ch: int = 1, hidden: int = 32) -> None:
        super().__init__()
        # Encoder takes ONLY the deviation (frame - mean), not raw frames.
        # 2 conv layers, GroupNorm-SiLU. ~40k params total.
        num_g = min(8, hidden)
        while hidden % num_g != 0 and num_g > 1:
            num_g -= 1
        self.dev_enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_g, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_g, hidden),
            nn.SiLU(inplace=True),
        )
        # Per-pixel per-frame log-variance head
        self.log_var_head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(
        self,
        frames: Tensor,
        lengths: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if frames.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(frames.shape)}")
        B, T, C, H, W = frames.shape

        # Build validity mask over T frames — same convention as SetTransformer.
        if mask is not None and lengths is not None:
            raise ValueError("Provide either mask or lengths, not both.")
        if mask is not None:
            invalid = mask.bool()                                              # [B, T]
        elif lengths is not None:
            invalid = (torch.arange(T, device=frames.device).unsqueeze(0)
                       >= lengths.to(frames.device).unsqueeze(1))
        else:
            invalid = torch.zeros(B, T, dtype=torch.bool, device=frames.device)
        valid = (~invalid).float()                                             # [B, T]

        # Per-pixel temporal mean over valid frames.
        v = valid.view(B, T, 1, 1, 1)
        n = v.sum(dim=1).clamp_min(1.0)                                        # [B, 1, 1, 1]
        mu_pixel = (frames * v).sum(dim=1) / n                                 # [B, C, H, W]

        # Per-frame DEVIATION (not raw signal). This is the only feature the
        # log-variance head ever sees → safe-by-design vs lesion suppression.
        dev = frames - mu_pixel.unsqueeze(1)                                   # [B, T, C, H, W]
        dev_flat = dev.reshape(B * T, C, H, W)
        feat = self.dev_enc(dev_flat)                                          # [B*T, hidden, H, W]
        log_var = self.log_var_head(feat).reshape(B, T, 1, H, W)               # [B, T, 1, H, W]

        # Mask invalid frames: set log_var to large positive → weight ≈ 0.
        log_var = log_var.masked_fill(invalid.view(B, T, 1, 1, 1), 1e6)

        # Per-pixel softmax over T (BLUE: weights = softmax(-log_var)).
        weights = torch.softmax(-log_var, dim=1)                               # [B, T, 1, H, W]
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        agg = (frames * weights).sum(dim=1)                                    # [B, C, H, W]
        return agg, weights.squeeze(2)                                         # [B, T, H, W]


class TissueFiLM(nn.Module):
    """v37+ Per-channel FiLM modulation conditioned on T1 tissue PV.

    γ, β are derived from globally-pooled tissue-class fractions
    (sigmoid(seg).mean over H,W → 4 numbers per sample), then expanded to
    per-channel scalars. Applied uniformly across spatial dimensions:

        out = γ[None, :, None, None] * x + β[None, :, None, None]

    Safety properties:
      - Per-channel only (NOT per-pixel) → cannot single out specific pixels
        to suppress / amplify. A lesion at (h,w) is treated identically to its
        same-channel neighbours.
      - Conditioning is on **global** tissue composition (whole-image GM
        fraction etc.), so the modulation cannot encode "edit this region".
      - γ is shifted to start near 1.0 (identity init); β starts at 0.

    Use-case: complements V=ASL cross-attention. Cross-attn says "look at
    these spatial neighbours"; FiLM says "for a GM-rich slice, emphasise
    these channels".
    """

    def __init__(self, n_classes: int = 4, ch: int = 128, hidden: int = 32) -> None:
        super().__init__()
        self.gamma_mlp = nn.Sequential(
            nn.Linear(n_classes, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, ch),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(n_classes, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, ch),
        )
        # Identity init: γ → 1.0, β → 0.0 at start
        nn.init.zeros_(self.gamma_mlp[-1].weight)
        nn.init.zeros_(self.gamma_mlp[-1].bias)
        nn.init.zeros_(self.beta_mlp[-1].weight)
        nn.init.zeros_(self.beta_mlp[-1].bias)

    def forward(self, x: Tensor, tissue_seg: Tensor) -> Tensor:
        # tissue_seg: [B, n_classes, H, W] (logits or sigmoid'd PV)
        seg_p = torch.sigmoid(tissue_seg).mean(dim=(2, 3))   # [B, n_classes]
        gamma = 1.0 + self.gamma_mlp(seg_p)                   # [B, ch], near-1 at init
        beta = self.beta_mlp(seg_p)                           # [B, ch], near-0 at init
        return gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)


class CrossModalFusion(nn.Module):
    """
    ASL cross-attends to T1 (or self-attends), with optional tissue-class
    similarity bias.

    Two modes (controlled by `t1_as_key`):

    1. `t1_as_key=True` (default, v37/v38/v39 behaviour) — cross-attention:
       Q from ASL, K from T1, V from ASL. T1 grayscale dictates attention
       pattern but values come from ASL. Used when the encoder is the
       vanilla ConvEncoder2D and the bottleneck is the only place T1
       features touch the ASL branch.

    2. `t1_as_key=False` (v42, when MoSSMEncoder2D is on) — ASL self-attention
       with tissue-class grouping bias:
         Q = K = V = ASL bottleneck features
         attention logits get the `tau * cos(seg_i, seg_j)` bias
       T1 contributes NEITHER values NOR feature keys; it only provides the
       structural grouping signal via tissue_seg. This is the principled
       complement to MoSSM's per-stage T1 modulation: MoSSM handles T1 as
       a state-dynamics modulator (encoder-side, multi-scale); this block
       handles global ASL self-attention with structural tissue grouping
       (bottleneck-only, non-local). Roles are clearly separated.

    Tissue-class bias (when tissue_seg is provided):
      - attn_logits += tau * cos(sigmoid(seg_i), sigmoid(seg_j))
      - seg is the 4-class GM/WM/CSF/BG PV map. Cosine similarity ≈ 1 within the
        same tissue class, ≈ 0 across classes.
      - Forces ASL features to pool within the same tissue class, preventing
        cross-tissue feature averaging (the dominant over-smoothing failure mode).
      - tau is learnable. Setting tissue_seg=None falls back to vanilla
        attention (cross or self depending on t1_as_key).

    Gate (output blending):
      - gate ∈ [0, gate_max]: residual blending of attention output with raw ASL.
      - gate_init=0.3, gate_max=1.0 (was 0.0/0.5 before v34 — fixes "gate→0 collapse"
        seen in v32/v33 where attn was effectively disabled).
    """

    def __init__(self, in_ch: int, n_heads: int = 4, dropout: float = 0.2,
                 gate_init: float = 0.3, gate_max: float = 1.0,
                 tau_init: float = 2.0, t1_as_key: bool = True,
                 use_blue_attn: bool = False,
                 blue_lambda_init: float = 1.0) -> None:
        super().__init__()
        assert in_ch % n_heads == 0, f"in_ch={in_ch} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.t1_as_key = bool(t1_as_key)
        self.use_blue_attn = bool(use_blue_attn)
        self.attn = nn.MultiheadAttention(in_ch, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(in_ch)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.gate_max = float(gate_max)
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        # BLUE-Attn (v42 N4): learnable noise-bias strength. λ multiplies the
        # per-pixel noise-variance penalty added to attention logits.
        if self.use_blue_attn:
            self.blue_lambda = nn.Parameter(torch.tensor(float(blue_lambda_init)))

    def forward(self, asl_map: Tensor, t1_map: Tensor,
                tissue_seg: Optional[Tensor] = None,
                noise_var: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            asl_map:   [B, C, H, W]   ASL bottleneck features
            t1_map:    [B, C, H, W]   T1 bottleneck features (used only when t1_as_key=True)
            tissue_seg: optional [B, n_class, H', W'] tissue PV map for grouping bias
            noise_var: optional [B, 1, H', W'] per-pixel noise variance (e.g.
                       Var_T(set_b)). Used for BLUE-Attn: adds
                       -λ * (var_i + var_j) to attention logits, downweighting
                       noisy pixels in both Q and K roles. This is the spatial
                       analogue of SVFW's per-pixel BLUE temporal weighting.
        """
        B, C, H, W = asl_map.shape
        N = H * W
        q = asl_map.flatten(2).transpose(1, 2)    # [B, N, C]
        if self.t1_as_key:
            k = t1_map.flatten(2).transpose(1, 2)  # [B, N, C] — cross-attn
        else:
            k = q                                  # [B, N, C] — self-attn (v42)
        v = q                                     # V = ASL — output stays in ASL space

        attn_mask: Optional[Tensor] = None
        if tissue_seg is not None:
            # Resize seg to match attention spatial size, then compute pairwise similarity.
            if tissue_seg.shape[-2] != H or tissue_seg.shape[-1] != W:
                tissue_seg = F.adaptive_avg_pool2d(tissue_seg, (H, W))
            seg_p = torch.sigmoid(tissue_seg).flatten(2).transpose(1, 2)  # [B, N, 4]
            seg_n = F.normalize(seg_p, dim=-1, eps=1e-6)
            sim = seg_n @ seg_n.transpose(-1, -2)                          # [B, N, N], in [0, 1]
            bias = self.tau * sim                                          # [B, N, N]
            attn_mask = bias

        # BLUE-Attn: noise-variance penalty (v42 N4).
        # Variance-normalised by batch mean so λ stays in O(1) range.
        if self.use_blue_attn and noise_var is not None:
            if noise_var.shape[-2] != H or noise_var.shape[-1] != W:
                noise_var = F.adaptive_avg_pool2d(noise_var, (H, W))
            var_flat = noise_var.flatten(2).squeeze(1)                      # [B, N]
            var_norm = var_flat / (var_flat.mean(dim=1, keepdim=True).clamp_min(1e-6))
            blue_pen = -self.blue_lambda * (var_norm.unsqueeze(2) + var_norm.unsqueeze(1))  # [B, N, N]
            attn_mask = blue_pen if attn_mask is None else attn_mask + blue_pen

        if attn_mask is not None:
            # nn.MultiheadAttention with batch_first expects 3D mask shape (B*nheads, L, S).
            attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_heads, N, N).reshape(B * self.n_heads, N, N)

        attn_out, _ = self.attn(q, k, v, attn_mask=attn_mask)
        gate = self.gate.clamp(0.0, self.gate_max)
        fused = self.norm(q + gate * attn_out)
        return fused.transpose(1, 2).reshape(B, C, H, W)


# -----------------------------------------------------------------------------
# T1GuidedCoarseHead + ASLDetailDecoder (refactor 2026-05-15)
# -----------------------------------------------------------------------------
# These two modules replace the previous "ASLT1Denoiser.cross_fusion +
# ConvDecoderWithSkips2D(use_t1_cross_fusion=True)" combo on the ASL path,
# splitting the decoder into:
#
#   1. T1GuidedCoarseHead — owns L0 (bottleneck 16x16) + L1 (32x32) operations
#      where T1 cross-attention is active. Adapter + optional CMF0 at L0;
#      Up 16->32 + ASL skip merge + optional CMF1 at L1.
#
#   2. ASLDetailDecoder — owns L2 (64x64) + L3 (128x128). Pure ASL U-Net
#      refinement: up-block + ASL skip + 1x1 fuse. No T1 input by construction
#      => structural guarantee that fine spatial detail cannot carry T1 grayscale.
#
# Computationally identical to the previous combined design; this is a
# refactor for module-boundary clarity and stronger anti-hallucination claim
# (V=ASL invariant + T1-free detail path are now type-level).
# T1 decoder still uses ConvDecoderWithSkips2D directly (intra-modal only).


class T1GuidedCoarseHead(nn.Module):
    """Coarse-scale (16x16 + 32x32) cross-modal fusion head for the ASL path.

    Forward shape pipeline (depth=4, base_ch=32, asl_hw=128):
      input:
        asl_feat_bot: [B, in_ch, 16, 16]              ASL encoder bottleneck
        t1_feat_bot:  [B, bottleneck_ch, 16, 16]      T1 encoder bottleneck (CMF0 K)
        asl_skip_l1:  [B, l1_ch, 32, 32]              ASL encoder skip at L1
        t1_skip_l1:   [B, l1_ch, 32, 32]   optional   T1 encoder skip at L1 (CMF1 K)
        tissue_seg:   [B, n_class, H', W'] optional   PV logits for similarity bias
        noise_var:    [B, 1, H', W']       optional   per-pixel variance for BLUE-Attn
      output:
        feat_l1:      [B, l1_ch, 32, 32]              pure-ASL fused feature

    Both CMF0 and CMF1 keep V=ASL (anti-hallucination). CMF0 may be disabled
    (use_cmf0=False) when an upstream encoder already performs cross-modal
    fusion at the bottleneck (e.g. v42 CMAM-in-MoSSM). CMF1 may be disabled
    (use_cmf1=False) for the no-cross-fusion baseline; in that case the head
    degenerates to adapter + (optional CMF0) + Up + ASL-skip merge.
    """

    def __init__(
        self,
        in_ch: int,
        bottleneck_ch: int,
        l1_ch: int,
        use_cmf0: bool = True,
        use_cmf1: bool = True,
        cmf0_t1_as_key: bool = True,
        cmf0_use_blue: bool = False,
        cmf_n_heads: int = 4,
        cmf_dropout: float = 0.2,
        use_wavelet: bool = False,
        groups: int = 8,
        skip_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.adapter = (
            nn.Conv2d(in_ch, bottleneck_ch, kernel_size=1, bias=False)
            if in_ch != bottleneck_ch else nn.Identity()
        )
        self.use_cmf0 = bool(use_cmf0)
        if self.use_cmf0:
            heads = int(cmf_n_heads)
            while bottleneck_ch % heads != 0 and heads > 1:
                heads //= 2
            self.cmf0 = CrossModalFusion(
                in_ch=bottleneck_ch, n_heads=heads, dropout=float(cmf_dropout),
                t1_as_key=bool(cmf0_t1_as_key), use_blue_attn=bool(cmf0_use_blue),
            )
        UpCls = WaveletUpBlock if use_wavelet else UpBlock
        self.up = UpCls(bottleneck_ch, l1_ch, groups=groups)
        num_g = min(groups, l1_ch)
        while l1_ch % num_g != 0 and num_g > 1:
            num_g -= 1
        self.fuse = nn.Sequential(
            nn.Conv2d(l1_ch * 2, l1_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_g, l1_ch),
            nn.SiLU(inplace=True),
        )
        self.use_cmf1 = bool(use_cmf1)
        if self.use_cmf1:
            heads = int(cmf_n_heads)
            while l1_ch % heads != 0 and heads > 1:
                heads //= 2
            self.cmf1 = CrossModalFusion(
                in_ch=l1_ch, n_heads=heads, dropout=float(cmf_dropout),
                t1_as_key=True, use_blue_attn=False,
            )
        self.skip_drop = nn.Dropout2d(p=float(skip_dropout))

    def forward(
        self,
        asl_feat_bot: Tensor,
        t1_feat_bot: Tensor,
        asl_skip_l1: Tensor,
        t1_skip_l1: Optional[Tensor] = None,
        tissue_seg: Optional[Tensor] = None,
        noise_var: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.adapter(asl_feat_bot)
        if self.use_cmf0:
            x = self.cmf0(x, t1_feat_bot, tissue_seg=tissue_seg, noise_var=noise_var)
        x = self.up(x)
        skip = self.skip_drop(asl_skip_l1)
        x = self.fuse(torch.cat([x, skip], dim=1))
        if self.use_cmf1 and t1_skip_l1 is not None:
            x = self.cmf1(x, t1_skip_l1, tissue_seg=tissue_seg)
        return x


class ContentOnlyDecoder(nn.Module):
    """Content-Only Reconstruction Decoder (CORD) — paper name (was NAFDecoder).

    RETIRED 2026-07-16 (with the MoSSM/CIG-Net-v2 line). NOT the main-line
    decoder: the CIG-VSS + EC-LRDA main method uses the plain T1-free
    ``ConvDecoderWithSkips2D`` (see docs/cig_vss.md §3.5). This class is only
    reachable via ``--use_naf_fusion``, which the runner now rejects. Kept in
    place (blocks.py shares NAF fusion helpers) so old v2 ckpts stay readable
    from git history; do not describe the main model as using SimpleGate/MDTA.

    Unified ASL decoder using NAFNet SimpleGate fusion at every level.

    v42j (2026-05-19): replaces (T1GuidedCoarseHead + ASLDetailDecoder) — a
    single pure-ASL decoder spanning all up-sample levels from the bottleneck
    to the output resolution. No T1 fusion at any level; no ResHead anchor.

    For depth=4, base_ch=32, out_hw=128 this owns:
      L0  UpBlock 16 → 32,  NAFFusion(cat[up, skip_32])
      L1  UpBlock 32 → 64,  NAFFusion(cat[up, skip_64])
      L2  UpBlock 64 → 128, NAFFusion(cat[up, skip_128])
      head: 3x3 conv → out_ch

    Each NAFFusionBlock is `Conv1x1(2C→C) → NAFBlock(C)`; NAFBlock has β/γ=0
    init so the block is identity at start. Compared to the previous Conv1x1
    + GN + SiLU fuse, the NAFBlock adds:
      - per-channel LayerNorm (instead of GroupNorm)
      - depthwise 3x3 conv on the projected feature
      - SimpleGate non-linearity (chunk·multiply, replaces SiLU)
      - Simplified Channel Attention (gate by per-channel global average)
      - Residual with gated scale (β, γ)

    Trainability invariant: NAFBlock(β=γ=0 init) → output = Conv1x1(input).
    Equivalent to "Conv1x1 fuse without activation" at step 0 → safe degenerate.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 1,
        out_hw: int = 128,
        base_ch: int = 32,
        depth: int = 4,
        groups: int = 8,
        skip_dropout: float = 0.3,
        final_activation: Optional[nn.Module] = None,
        use_wavelet: bool = False,
        use_rgsf: bool = False,
        use_mdta: bool = False,
        rgsf_a_init: float = 4.0,
        rgsf_b_raw_init: float = -3.0,
        rgsf_var_zscore: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.out_hw = int(out_hw)
        # Decoder channels: from deepest 256ch → 128 → 64 → 32 (for depth=4)
        channels = [base_ch * (2 ** i) for i in range(depth)]
        decoder_chs = list(reversed(channels[:-1]))  # e.g. [128, 64, 32]
        if not decoder_chs:
            raise ValueError(f"depth={depth} too small for NAFDecoder.")
        UpCls = WaveletUpBlock if use_wavelet else UpBlock
        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.use_mdta = bool(use_mdta)
        ch = int(in_ch)
        for next_ch in decoder_chs:
            self.ups.append(UpCls(ch, next_ch, groups=groups))
            self.fusions.append(NAFFusionBlock(in_ch=next_ch * 2, out_ch=next_ch,
                                               use_mdta=self.use_mdta))
            ch = next_ch
        self.head = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        self.final_activation = final_activation
        self.skip_drop = nn.Dropout2d(p=float(skip_dropout))

        # RGSF (carry over from ASLDetailDecoder). Inits are configurable since
        # Phase 2 Run B (2026-05-22): empirical probe showed mean gate ≈ 0.96
        # under default a=4 / b_raw=-3 (softplus≈0.05), i.e. near-identity skip
        # giving Var_T(set_a) almost no influence. Lower a and raise b_raw to
        # move the gate's working point into a region where log_nv discriminates
        # between clean and corrupt patches.
        self.use_rgsf = bool(use_rgsf)
        self.rgsf_var_zscore = bool(rgsf_var_zscore)
        if self.use_rgsf:
            self.rgsf_a = nn.Parameter(
                torch.full((len(self.ups),), float(rgsf_a_init))
            )
            self.rgsf_b_raw = nn.Parameter(
                torch.full((len(self.ups),), float(rgsf_b_raw_init))
            )

    def forward(
        self,
        x: Tensor,
        asl_skips_reversed: List[Tensor],
        noise_var: Optional[Tensor] = None,
    ) -> Tensor:
        """asl_skips_reversed: ordered coarse-first → fine-last.
        For depth=4: [skip_32, skip_64, skip_128]."""
        if len(asl_skips_reversed) != len(self.ups):
            raise ValueError(
                f"NAFDecoder expected {len(self.ups)} skips, got {len(asl_skips_reversed)}"
            )
        if self.use_rgsf and noise_var is None:
            raise ValueError("NAFDecoder(use_rgsf=True) requires noise_var")
        for i, (up, fuse) in enumerate(zip(self.ups, self.fusions)):
            x = up(x)
            skip = asl_skips_reversed[i]
            if self.use_rgsf:
                nv = F.adaptive_avg_pool2d(noise_var, output_size=skip.shape[-2:])
                log_nv = torch.log(nv.clamp_min(1e-6))
                if self.rgsf_var_zscore:
                    # Standardise log(noise_var) per-batch so the gate's working
                    # point is independent of absolute Var_T scale (Phase 2 Run B).
                    mu = log_nv.mean(dim=(1, 2, 3), keepdim=True)
                    sd = log_nv.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
                    log_nv = (log_nv - mu) / sd
                b_s = F.softplus(self.rgsf_b_raw[i])
                w_s = torch.sigmoid(self.rgsf_a[i] - b_s * log_nv)
                skip = skip * w_s
            skip = self.skip_drop(skip)
            x = fuse(torch.cat([x, skip], dim=1))
        x = self.head(x)
        if x.shape[-1] != self.out_hw or x.shape[-2] != self.out_hw:
            x = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="bilinear", align_corners=False)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class ASLDetailDecoder(nn.Module):
    """Detail decoder for the ASL path: 32x32 -> 128x128, pure ASL.

    Takes the L1 feature produced by T1GuidedCoarseHead and refines through
    the remaining decoder levels using ASL encoder skip connections only.
    By construction this module has no T1 input -> structural guarantee that
    fine spatial detail cannot carry T1 grayscale (V=ASL extended to the
    whole detail path).

    For depth=4, base_ch=32, out_hw=128 this owns:
      L2: UpBlock 32->64,   cat(asl_skip_64),  1x1 conv + GN + SiLU
      L3: UpBlock 64->128,  cat(asl_skip_128), 1x1 conv + GN + SiLU
      head: 3x3 conv -> out_ch
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 1,
        out_hw: int = 128,
        base_ch: int = 32,
        depth: int = 4,
        groups: int = 8,
        skip_dropout: float = 0.3,
        final_activation: Optional[nn.Module] = None,
        use_wavelet: bool = False,
        use_rgsf: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.out_hw = int(out_hw)
        channels = [base_ch * (2 ** i) for i in range(depth)]
        remaining = list(reversed(channels[:-1]))[1:]  # e.g. depth=4 -> [64, 32]
        if not remaining:
            raise ValueError(f"depth={depth} leaves no detail levels for ASLDetailDecoder.")
        UpCls = WaveletUpBlock if use_wavelet else UpBlock
        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        ch = int(in_ch)
        for next_ch in remaining:
            self.ups.append(UpCls(ch, next_ch, groups=groups))
            num_g = min(groups, next_ch)
            while next_ch % num_g != 0 and num_g > 1:
                num_g -= 1
            self.fusions.append(nn.Sequential(
                nn.Conv2d(next_ch * 2, next_ch, kernel_size=1, bias=False),
                nn.GroupNorm(num_g, next_ch),
                nn.SiLU(inplace=True),
            ))
            ch = next_ch
        self.head = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        self.final_activation = final_activation
        self.skip_drop = nn.Dropout2d(p=float(skip_dropout))

        # Reliability-Gated Skip Fusion (RGSF): per-scale sigmoid gate on ASL
        # skip, parameterised by per-pixel noise variance.
        #   b_s = softplus(b_raw_s)  ← enforces b_s ≥ 0 structurally
        #   w_s = sigmoid(a_s - b_s * log(noise_var + eps))
        # 2026-05-18 bug fix: v42f probe found b_s learned negative values
        # (-0.07, -0.13), i.e. opposite sign to the design intent — RGSF was
        # effectively suppressing skip in *low*-noise pixels. softplus(b_raw)
        # makes b_s ≥ 0 a structural property. Init b_raw = -10 → b_s ≈ 5e-5
        # so gate ≈ sigmoid(a_s) ≈ 0.98 (identity skip at init, unchanged).
        self.use_rgsf = bool(use_rgsf)
        if self.use_rgsf:
            self.rgsf_a = nn.Parameter(torch.full((len(self.ups),), 4.0))
            # b_raw init: −3.0 → softplus(b_raw) ≈ 0.0486, gives identity-ish skip
            # (w ≈ σ(a − 0.05·log nv) ≈ σ(4) ≈ 0.98 for typical log nv ≈ −7) while keeping
            # gradient flow through softplus alive. Prior init −10.0 placed b_raw in the
            # softplus saturation region (softplus'(−10) ≈ 4.5e-5), effectively dead-locking
            # the parameter; the empirical probe (2026-05-20) showed b_raw barely moved
            # after 60 training steps. New init unlocks the variance-aware attenuation
            # path while preserving safe identity initialisation.
            self.rgsf_b_raw = nn.Parameter(torch.full((len(self.ups),), -3.0))

    def forward(
        self,
        x: Tensor,
        asl_skips_detail: List[Tensor],
        noise_var: Optional[Tensor] = None,
    ) -> Tensor:
        """asl_skips_detail ordered high-res-last (depth=4: [skip_64, skip_128]).
        noise_var: optional [B,1,H,W] per-pixel input-frame variance; required if
        use_rgsf=True.
        """
        if len(asl_skips_detail) != len(self.ups):
            raise ValueError(
                f"ASLDetailDecoder expected {len(self.ups)} skips, got {len(asl_skips_detail)}"
            )
        if self.use_rgsf and noise_var is None:
            raise ValueError("ASLDetailDecoder(use_rgsf=True) requires noise_var")
        for i, (up, fuse) in enumerate(zip(self.ups, self.fusions)):
            x = up(x)
            skip = asl_skips_detail[i]
            if self.use_rgsf:
                # Pool noise variance to this skip's spatial size; gate the skip.
                nv = F.adaptive_avg_pool2d(noise_var, output_size=skip.shape[-2:])
                log_nv = torch.log(nv.clamp_min(1e-6))
                b_s = F.softplus(self.rgsf_b_raw[i])
                w_s = torch.sigmoid(self.rgsf_a[i] - b_s * log_nv)
                skip = skip * w_s
            skip = self.skip_drop(skip)
            x = fuse(torch.cat([x, skip], dim=1))
        x = self.head(x)
        if x.shape[-1] != self.out_hw or x.shape[-2] != self.out_hw:
            x = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="bilinear", align_corners=False)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


# --- backward-compat aliases (old class name -> new paper-aligned name) ------
# Renamed 2026-06-27 for paper consistency (see docs/archive/v2_mossm/cig_net.md naming table).
# Submodule attribute names and CLI flags are UNCHANGED, so existing checkpoints
# and slurm scripts load unaffected; these aliases keep every external import
# (figure/eval scripts that still say `SetTransformerAggregator` / `NAFDecoder`)
# resolving to the renamed class.
SetTransformerAggregator = FrameReliabilityAggregator   # FRA
NAFDecoder = ContentOnlyDecoder                          # CORD
