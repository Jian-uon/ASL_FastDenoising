# -*- coding: utf-8 -*-
"""SwinIR-style Swin-Transformer denoiser as a drop-in trainable baseline.

A compact (SwinIR-light-sized) reimplementation of SwinIR (Liang et al., ICCVW
2021) for single-image restoration, wired to the SAME I/O contract as
``PlainUNet2D`` so the baseline trainer / eval loaders treat it identically:

    Input  : [B, in_ch,  H, W]  (already aggregated mean image, in_ch=1)
    Output : [B, out_ch, H, W]  (out_ch=1), global-residual (predicts input+residual)

No T1, no cross-modal fusion, no SetTransformer -- a pure ASL-only single-image
denoiser used only as an external architecture baseline (§4.1.2 method (iv);
recent-two-years reference, trained supervised vs the 12-NEX mean). The
reference method is SwinIR (Shou et al. adapted a Swin-Transformer denoiser to
3D ASL in MRM 2024); we retrain the architecture under our own regime/data so
the comparison is on equal footing.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _window_partition(x: Tensor, ws: int) -> Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(windows: Tensor, ws: int, H: int, W: int) -> Tensor:
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window multi-head self-attention with a relative-position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(
            [torch.arange(window_size), torch.arange(window_size)], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinLayer(nn.Module):
    """One Swin Transformer layer (W-MSA or SW-MSA + MLP), pre-norm residual."""

    def __init__(self, dim: int, num_heads: int, window_size: int,
                 shift_size: int, mlp_ratio: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _attn_mask(self, H: int, W: int, device) -> Tensor | None:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        spans = (slice(0, -self.window_size),
                 slice(-self.window_size, -self.shift_size),
                 slice(-self.shift_size, None))
        for h in spans:
            for w in spans:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mw = _window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))
        xw = _window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        xw = self.attn(xw, self._attn_mask(H, W, x.device))
        xw = xw.view(-1, self.window_size, self.window_size, C)
        x = _window_reverse(xw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):
    """Residual Swin Transformer Block: depth STLs + a 3x3 conv, residual-added."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SwinLayer(dim, num_heads, window_size,
                      0 if (i % 2 == 0) else window_size // 2, mlp_ratio)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        shortcut = x
        for layer in self.layers:
            x = layer(x, H, W)
        B, L, C = x.shape
        xc = self.conv(x.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)
        return xc + shortcut


class SwinIR2D(nn.Module):
    """SwinIR-light single-image denoiser (global-residual). Drop-in for PlainUNet2D."""

    def __init__(
        self,
        hw: int = 128,
        in_ch: int = 1,
        out_ch: int = 1,
        embed_dim: int = 60,
        depths: Sequence[int] = (6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        **kwargs,  # tolerate/ignore PlainUNet-style extras (base_ch, skip_dropout, ...)
    ) -> None:
        super().__init__()
        assert len(depths) == len(num_heads), "depths and num_heads must match"
        self.window_size = int(window_size)
        self.in_ch, self.out_ch = int(in_ch), int(out_ch)

        self.conv_first = nn.Conv2d(in_ch, embed_dim, 3, 1, 1)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            RSTB(embed_dim, depths[i], num_heads[i], self.window_size, mlp_ratio)
            for i in range(len(depths))])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, out_ch, 3, 1, 1)

    def _pad(self, x: Tensor) -> Tensor:
        ws = self.window_size
        h, w = x.shape[-2:]
        ph, pw = (ws - h % ws) % ws, (ws - w % ws) % ws
        return F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x

    def forward(self, x: Tensor) -> Tensor:
        H0, W0 = x.shape[-2:]
        inp = self._pad(x)
        H, W = inp.shape[-2:]
        feat = self.conv_first(inp)                     # [B, C, H, W]
        B, C = feat.shape[0], feat.shape[1]
        t = self.embed_norm(feat.flatten(2).transpose(1, 2))
        for layer in self.layers:
            t = layer(t, H, W)
        body = self.norm(t).transpose(1, 2).view(B, C, H, W)
        body = self.conv_after_body(body) + feat
        out = self.conv_last(body) + inp[:, : self.out_ch]   # global residual (in_ch==out_ch)
        return out[..., :H0, :W0]
