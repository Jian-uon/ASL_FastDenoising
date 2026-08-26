# -*- coding: utf-8 -*-
"""Unit tests for the multi-scale window cross-fusion (docs/multiscale_window_design.md).

These lock the four properties the design rests on, so a refactor cannot quietly
break them:

  P1  window_fusion_levels=0 builds NOTHING -> no new parameters and the decoder
      is still T1-free at the signature level (the free ablation baseline).
  P2  Maximum principle: with V unprojected and no output projection, every fused
      value is a convex combination of the SAME channel's ASL values inside its
      window, so it cannot leave [min_window, max_window]. This is what makes
      "V=ASL" structural rather than a naming convention.
  P3  The gate is sigmoid-parameterised, so the attention parameters actually
      receive gradient. A clamped gate at 0 gives d(out)/d(theta) = 0 exactly and
      the module would be permanently dead -- that bug is what P3 guards.
  P4  k_source='asl' is genuinely T1-free (the control arm), and k_source='t1' is
      genuinely wired (changing T1 changes the output).

Runnable either way:
    python tests/test_window_fusion.py      # plain asserts, prints OK
    pytest tests/test_window_fusion.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.asl_t1_model import ASLT1Denoiser  # noqa: E402
from models.blocks import WindowCrossFusion, _window_partition  # noqa: E402

HW = 128
ARCH = dict(asl_hw=HW, t1_hw=HW, base_ch=32, depth=4, use_t1_cross_fusion=True,
            t1_attn_max_tokens=1024, t1_task="recon", use_t1_decoder=False)


def _model(**kw):
    torch.manual_seed(0)
    return ASLT1Denoiser(**{**ARCH, **kw}).eval()


def _inputs(b=2, t=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, t, 1, HW, HW, generator=g),
            torch.rand(b, 1, HW, HW, generator=g))


# --- P1 -------------------------------------------------------------------
def test_levels0_builds_nothing():
    m = _model(window_fusion_levels=0)
    names = [n for n, _ in m.named_parameters() if "window_fusion" in n]
    assert not names, f"levels=0 must add no parameters, got {names}"
    assert all(not isinstance(f, WindowCrossFusion) for f in m.asl_decoder.window_fusions)


def test_levels0_decoder_is_t1_free_in_the_detail_path():
    """The detail decoder must ignore t1_skips_detail entirely when levels=0."""
    m = _model(window_fusion_levels=0)
    x = torch.rand(2, 128, 32, 32)
    asl = [torch.rand(2, 64, 64, 64), torch.rand(2, 32, HW, HW)]
    t1a = [torch.rand(2, 64, 64, 64), torch.rand(2, 32, HW, HW)]
    with torch.no_grad():
        y_none = m.asl_decoder(x, asl, t1_skips_detail=None)
        y_rand = m.asl_decoder(x, asl, t1_skips_detail=t1a)
        y_zero = m.asl_decoder(x, asl, t1_skips_detail=[torch.zeros_like(s) for s in t1a])
    assert torch.equal(y_none, y_rand) and torch.equal(y_none, y_zero)


def test_levels_count_and_shift():
    m = _model(window_fusion_levels=2, window_size=8)
    fus = m.asl_decoder.window_fusions
    assert all(isinstance(f, WindowCrossFusion) for f in fus)
    # coarser level unshifted, finest level shifted by ws//2 so seams miss the output
    assert fus[0].shift == 0 and fus[1].shift == 4
    m1 = _model(window_fusion_levels=1)
    assert not isinstance(m1.asl_decoder.window_fusions[0], WindowCrossFusion)
    assert isinstance(m1.asl_decoder.window_fusions[1], WindowCrossFusion)


# --- P2 -------------------------------------------------------------------
def test_maximum_principle():
    """Per channel, the fused value stays inside its own window's ASL range."""
    for ws, shift, ch, hw in ((8, 0, 16, 64), (8, 4, 16, 64), (4, 2, 8, 50)):
        torch.manual_seed(0)
        m = WindowCrossFusion(ch, ch, window_size=ws, shift=shift)
        with torch.no_grad():
            m.gate_logit.fill_(20.0)                       # gate -> 1, the worst case
        x = torch.randn(2, ch, hw, hw)
        t1 = torch.randn(2, ch, hw, hw)
        with torch.no_grad():
            y = m(x, t1)
        # window-local bounds, computed independently of the module's internals
        pad = (ws - (hw + shift) % ws) % ws
        xt = torch.nn.functional.pad(x.permute(0, 2, 3, 1), (0, 0, shift, pad, shift, pad),
                                     value=float("nan"))
        xw = _window_partition(xt, ws).reshape(-1, ws * ws, ch)
        lo = xw.nan_to_num(float("inf")).min(dim=1).values
        hi = xw.nan_to_num(float("-inf")).max(dim=1).values
        yt = torch.nn.functional.pad(y.permute(0, 2, 3, 1), (0, 0, shift, pad, shift, pad))
        yw = _window_partition(yt, ws).reshape(-1, ws * ws, ch)
        valid = ~xw.isnan()
        n = ws * ws
        lo = lo.unsqueeze(1).expand(-1, n, -1)
        hi = hi.unsqueeze(1).expand(-1, n, -1)
        assert bool((yw[valid] >= lo[valid] - 1e-4).all()), f"below window min (ws={ws})"
        assert bool((yw[valid] <= hi[valid] + 1e-4).all()), f"above window max (ws={ws})"


def test_gate_is_bounded_open_interval():
    m = WindowCrossFusion(8, 8, window_size=4)
    for v in (-50.0, -3.0, 0.0, 50.0):
        with torch.no_grad():
            m.gate_logit.fill_(v)
        g = float(torch.sigmoid(m.gate_logit))
        assert 0.0 <= g <= 1.0, g
    with torch.no_grad():
        m.gate_logit.fill_(-3.0)
    assert abs(float(torch.sigmoid(m.gate_logit)) - 0.0474) < 1e-3


def test_no_value_or_output_projection():
    """W_v / W_o would silently destroy P2; fail loudly if someone adds one."""
    m = WindowCrossFusion(8, 8, window_size=4)
    names = {n for n, _ in m.named_parameters()}
    assert not {n for n in names if n.startswith(("wv", "proj_out", "o."))}, names


# --- P3 -------------------------------------------------------------------
def test_attention_params_receive_gradient():
    m = WindowCrossFusion(16, 16, window_size=8, gate_init=-3.0)
    x = torch.rand(2, 16, 32, 32, requires_grad=True)
    m(x, torch.rand(2, 16, 32, 32)).square().mean().backward()
    for name in ("wq.weight", "wk.weight", "proj_t1.weight", "gate_logit"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and float(g.abs().sum()) > 0.0, f"{name} got no gradient"


# --- P4 -------------------------------------------------------------------
def test_k_source_asl_is_t1_free():
    m = WindowCrossFusion(16, 16, window_size=8, k_source="asl")
    assert m.proj_t1 is None
    x = torch.rand(2, 16, 32, 32)
    with torch.no_grad():
        a = m(x, torch.rand(2, 16, 32, 32))
        b = m(x, torch.rand(2, 16, 32, 32) * 100.0)
        c = m(x, None)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_k_source_t1_is_actually_wired():
    m = _model(window_fusion_levels=2)
    with torch.no_grad():
        for f in m.asl_decoder.window_fusions:
            f.gate_logit.fill_(2.0)                        # make the path visible
        x, t1 = _inputs()
        y1 = m(x, x, t1)["asl_recon"]
        y2 = m(x, x, torch.rand_like(t1))["asl_recon"]
    assert not torch.equal(y1, y2), "T1 keys do not reach the output"


# --- integration ----------------------------------------------------------
def test_full_model_forward_and_stats():
    m = _model(window_fusion_levels=2)
    x, t1 = _inputs()
    with torch.no_grad():
        out = m(x, x, t1)
        inf = m.infer_from_subset(x[:, :2], t1)
    assert out["asl_recon"].shape == (2, 1, HW, HW)
    assert inf["asl_recon"].shape == (2, 1, HW, HW)
    st = m.window_stats()
    assert set(st) == {f"wf{i}_{k}" for i in (0, 1) for k in ("gate", "entropy", "delta")}, st
    assert all(v == v for v in st.values())                # no NaN
    assert 0.0 < st["wf0_gate"] < 1.0


def test_param_overhead_is_small():
    base = sum(p.numel() for p in _model(window_fusion_levels=0).parameters())
    full = sum(p.numel() for p in _model(window_fusion_levels=2).parameters())
    assert full > base
    assert (full - base) / base < 0.01, f"window fusion added {(full-base)/base:.2%}"


def test_arch_roundtrip_defaults_to_off():
    """Checkpoints written before 2026-08-26 have no window_* keys in arch."""
    legacy = dict(ARCH)
    m = ASLT1Denoiser(**legacy)
    assert m.window_fusion_levels == 0
    assert all(not isinstance(f, WindowCrossFusion) for f in m.asl_decoder.window_fusions)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} window-fusion tests passed.")
