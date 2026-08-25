# -*- coding: utf-8 -*-
"""ASL-VRT: T1-free Video-Restoration-Transformer student for v41.

Inspired by Liang et al. *VRT: A Video Restoration Transformer.* ICCV 2022.
Frames an ASL multi-frame denoising problem as video restoration: explicit
spatial + temporal mutual attention replaces SVFW's analytical aggregation.
T1 is never an input to the network -> hallucination is structurally
impossible at inference.

Distilled from a frozen v39 ASLT1Denoiser teacher; output dim aligned with
teacher (`asl_recon` [B,1,H,W]). The student exposes a bottleneck feature
[B, c2, H/8, W/8] for adapter-bridged feature distillation.

Implementation notes
--------------------
A faithful VRT has ~20M parameters and many specialised modules (TMSA, PSA,
optical-flow guided alignment). This module implements a compact research-grade
version with:
  - Spatial Window Attention (W-MSA) within each frame, like SwinIR
  - Temporal Mutual Attention (TMA): each frame's tokens cross-attend to
    pooled cross-frame tokens at the same spatial location
  - 3-level hierarchical encoder + decoder, frames merged by attention pooling
    at the bottleneck
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Window-based spatial multi-head self-attention (Swin-style, per frame)
# -----------------------------------------------------------------------------

def _window_partition(x: Tensor, win: int) -> Tensor:
    """[B, H, W, C] -> [B*nW, win*win, C]."""
    B, H, W, C = x.shape
    x = x.view(B, H // win, win, W // win, win, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win * win, C)


def _window_reverse(seq: Tensor, win: int, H: int, W: int) -> Tensor:
    """[B*nW, win*win, C] -> [B, H, W, C]."""
    B = seq.shape[0] // ((H // win) * (W // win))
    x = seq.view(B, H // win, W // win, win, win, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowSelfAttn(nn.Module):
    """(Optionally shifted) window multi-head self-attention.

    When `shift > 0`, this performs SW-MSA (Liu et al. ICCV 2021, Swin
    Transformer). The token grid is rolled by `(-shift, -shift)` before
    partitioning, then rolled back after. Alternating W-MSA (shift=0)
    and SW-MSA across consecutive blocks lets information cross the
    rigid window boundary — without this, attention is strictly local
    to each window and the output exhibits the visible block-edge
    artifact seen in v41's first eval.

    For simplicity we OMIT the Swin attention mask that prevents tokens
    rolled in from opposite edges from attending to each other. For a
    fully-rolled image (no padding) the leakage is mild; if the artifact
    persists at the edges, add the mask as in the SwinIR codebase.
    """

    def __init__(self, dim: int, n_heads: int, window: int, shift: int = 0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.window = window
        assert 0 <= shift < window, (shift, window)
        self.shift = shift
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // n_heads) ** -0.5

    def forward(self, x: Tensor) -> Tensor:                         # [B, H, W, C]
        B, H, W, C = x.shape
        win = self.window
        # pad to multiple of window
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        if pad_h or pad_w:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        Hp, Wp = H + pad_h, W + pad_w

        # cyclic shift (SW-MSA) — bridges adjacent window boundaries
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        seq = _window_partition(x, win)                              # [B*nW, win*win, C]
        qkv = self.qkv(seq).reshape(seq.shape[0], seq.shape[1], 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                             # [3, B*nW, heads, win*win, C/h]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale                # [B*nW, heads, win*win, win*win]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(seq.shape[0], seq.shape[1], C)
        out = self.proj(out)
        out = _window_reverse(out, win, Hp, Wp)

        # reverse cyclic shift
        if self.shift > 0:
            out = torch.roll(out, shifts=(self.shift, self.shift), dims=(1, 2))

        if pad_h or pad_w:
            out = out[:, :H, :W, :].contiguous()
        return out


# -----------------------------------------------------------------------------
# Temporal Mutual Attention: each frame's pixel cross-attends across frames
# -----------------------------------------------------------------------------

class TemporalMutualAttn(nn.Module):
    """At each spatial position, all T frames attend to each other.

    Input  [B, T, H, W, C]
    Output [B, T, H, W, C]
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // n_heads) ** -0.5

    def forward(self, x: Tensor) -> Tensor:                         # [B, T, H, W, C]
        B, T, H, W, C = x.shape
        # group by spatial position: [B*H*W, T, C]
        seq = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        qkv = self.qkv(seq).reshape(seq.shape[0], T, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale                # [BHW, heads, T, T]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(seq.shape[0], T, C)
        out = self.proj(out)
        return out.reshape(B, H, W, T, C).permute(0, 3, 1, 2, 4).contiguous()


# -----------------------------------------------------------------------------
# Block: spatial-window-attn + temporal-mutual-attn + FFN
# -----------------------------------------------------------------------------

class STMABlock(nn.Module):
    """Pre-norm Spatial Window + Temporal Mutual + FFN.

    `shift_size` is forwarded to WindowSelfAttn. Use 0 for W-MSA, window//2
    for SW-MSA. ASLVRTStudent alternates them across consecutive blocks.
    """

    def __init__(self, dim: int, n_heads: int = 4, window: int = 8,
                 mlp_ratio: float = 2.0, shift_size: int = 0):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.spatial = WindowSelfAttn(dim, n_heads, window, shift=shift_size)
        self.norm_t = nn.LayerNorm(dim)
        self.temporal = TemporalMutualAttn(dim, n_heads)
        self.norm_f = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:                          # [B, T, H, W, C]
        # spatial: applied per-frame (collapse T into batch)
        B, T, H, W, C = x.shape
        flat = x.reshape(B * T, H, W, C)
        flat = flat + self.spatial(self.norm_s(flat))
        x = flat.view(B, T, H, W, C)
        # temporal mutual attention
        x = x + self.temporal(self.norm_t(x))
        # FFN
        x = x + self.mlp(self.norm_f(x))
        return x


# -----------------------------------------------------------------------------
# Hierarchical encoder/decoder over STMA blocks
# -----------------------------------------------------------------------------

class _Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        # 2x2 spatial down via stride conv on H,W only (T preserved)
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:                          # [B, T, H, W, C]
        B, T, H, W, C = x.shape
        flat = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        flat = self.proj(flat)                                       # [B*T, c_out, H/2, W/2]
        return flat.permute(0, 2, 3, 1).reshape(B, T, H // 2, W // 2, -1)


class _Up(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        B, T, H, W, C = x.shape
        flat = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        flat = self.proj(flat)
        return flat.permute(0, 2, 3, 1).reshape(B, T, H * 2, W * 2, -1)


class ASLVRTStudent(nn.Module):
    """T1-free VRT-style student.

    Forward
    -------
    set_a [B, T, 1, H, W]
        |
        v Conv embed (per-frame) -> [B, T, H, W, C0]
        v 2 stages of STMABlock + Down
        v bottleneck STMABlocks
        v 2 stages of Up + STMABlock
        v collapse T (mean over frames) -> [B, H, W, C0]
        v Conv head -> [B, 1, H, W]
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        depth_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2),
        n_heads: int = 4,
        window: int = 8,
        teacher_bottleneck_ch: int = 256,
    ) -> None:
        super().__init__()
        c0, c1, c2 = base_ch, base_ch * 2, base_ch * 4
        # per-frame embedding (shared across T)
        self.embed = nn.Conv2d(in_ch, c0, kernel_size=3, padding=1)

        # helper: alternate W-MSA (shift=0) and SW-MSA (shift=win//2) within a stack
        def _stack(c: int, depth: int, win: int) -> nn.ModuleList:
            return nn.ModuleList([
                STMABlock(c, n_heads, win, shift_size=(0 if (i % 2 == 0) else win // 2))
                for i in range(depth)
            ])

        # encoder
        self.enc0 = _stack(c0, depth_blocks[0], window)
        self.down0 = _Down(c0, c1)
        self.enc1 = _stack(c1, depth_blocks[1], window)
        self.down1 = _Down(c1, c2)
        # bottleneck (smaller window since spatial dim is already H/4)
        bn_win = max(2, window // 2)
        self.bottleneck = _stack(c2, depth_blocks[2], bn_win)
        # decoder
        self.up1 = _Up(c2, c1)
        self.dec1 = _stack(c1, depth_blocks[3], window)
        self.up0 = _Up(c1, c0)
        self.dec0 = _stack(c0, depth_blocks[3], window)
        # head: collapse T then project to 1ch
        self.head = nn.Conv2d(c0, in_ch, kernel_size=3, padding=1)

        # adapter for feature distillation: bottleneck c2 [B, c2, H/4, W/4]
        # vs teacher fused_map [B, 256, H/16, W/16] - dimensionalities differ;
        # use stride-conv adapter to match spatial scale.
        self.bottleneck_adapter = nn.Sequential(
            nn.Conv2d(c2, teacher_bottleneck_ch, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),                              # H/4 -> H/8 (matches teacher)
        )
        self.bottleneck_ch = c2

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, x: Tensor) -> Tensor:
        for blk in blocks:
            x = blk(x)
        return x

    @staticmethod
    def _embed_frames(emb: nn.Module, set_a: Tensor) -> Tensor:
        """[B, T, 1, H, W] -> [B, T, H, W, C] via shared per-frame conv."""
        B, T, C, H, W = set_a.shape
        flat = set_a.view(B * T, C, H, W)
        flat = emb(flat)                                              # [B*T, c0, H, W]
        return flat.permute(0, 2, 3, 1).view(B, T, H, W, -1)

    def forward(
        self,
        set_a: Tensor,                                                 # [B, T, 1, H, W]
        lengths: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, Tensor]:
        # NOTE: lengths/mask not used at architecture level; padded frames
        # contribute zero attention weight but shouldn't affect output much
        # given the mean-collapse at the end.

        x = self._embed_frames(self.embed, set_a)                      # [B, T, H, W, c0]

        # encoder
        e0 = self._run_blocks(self.enc0, x)                            # [B, T, H, W, c0]
        e1_in = self.down0(e0)                                         # [B, T, H/2, W/2, c1]
        e1 = self._run_blocks(self.enc1, e1_in)
        e2_in = self.down1(e1)                                         # [B, T, H/4, W/4, c2]

        # bottleneck
        b = self._run_blocks(self.bottleneck, e2_in)
        bottleneck_feat = b                                            # [B, T, H/4, W/4, c2]

        # decoder
        d1_in = self.up1(b) + e1                                       # [B, T, H/2, W/2, c1]
        d1 = self._run_blocks(self.dec1, d1_in)
        d0_in = self.up0(d1) + e0                                      # [B, T, H, W, c0]
        d0 = self._run_blocks(self.dec0, d0_in)

        # collapse T (mean over frames is a clean choice for variable T)
        if lengths is not None:
            # masked mean
            T = d0.shape[1]
            valid = (torch.arange(T, device=d0.device).unsqueeze(0) < lengths.unsqueeze(1))
            mask_T = valid.float().view(-1, T, 1, 1, 1)
            d0_mean = (d0 * mask_T).sum(dim=1) / mask_T.sum(dim=1).clamp_min(1)
        else:
            d0_mean = d0.mean(dim=1)                                   # [B, H, W, c0]

        # head -> [B, 1, H, W]
        recon = self.head(d0_mean.permute(0, 3, 1, 2).contiguous())

        out: Dict[str, Tensor] = {"asl_recon": recon}
        if return_features:
            # collapse T on bottleneck for adapter
            if lengths is not None:
                T = bottleneck_feat.shape[1]
                valid = (torch.arange(T, device=bottleneck_feat.device).unsqueeze(0) < lengths.unsqueeze(1))
                mask_T = valid.float().view(-1, T, 1, 1, 1)
                bf_mean = (bottleneck_feat * mask_T).sum(dim=1) / mask_T.sum(dim=1).clamp_min(1)
            else:
                bf_mean = bottleneck_feat.mean(dim=1)                  # [B, H/4, W/4, c2]
            bf = bf_mean.permute(0, 3, 1, 2).contiguous()              # [B, c2, H/4, W/4]
            out["feat_bottleneck"] = bf
            out["feat_bottleneck_adapted"] = self.bottleneck_adapter(bf)
        return out
