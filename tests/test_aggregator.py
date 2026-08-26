# -*- coding: utf-8 -*-
"""Unit tests for VarianceFrameAggregator — the closed-form replacement for FRA.

The trained FRA's learnt policy, read off the training probes, is "veto the injected bad
frame, 1/N on every good frame" (``probe_agg_w_bad`` 0.0005 -> 0.0 while
``probe_agg_w_normal_per_frame`` sat at 1/8, 1/9, 1/7). These lock that the closed form
implements that policy, and that the properties it relies on hold:

  P1  it is a convex combination: weights are non-negative, sum to 1, and padded frames
      get exactly 0 (so a short set_a is aggregated over its valid frames only)
  P2  no corrupted frame  -> uniform weights (the plain mean, which is the right answer
      when the frames really are i.i.d.)
  P3  one corrupted frame -> that frame is vetoed and the rest stay uniform
  P4  tau is the single knob AND a readout: tau -> 0 gives exactly the uniform mean
  P5  brain premasking cannot change the weights (background is 0 in every frame, so the
      common factor cancels in the softmax) — this is why the module needs no mask input
  P6  --aggregator defaults to 'fra', so old checkpoints and old runs are unaffected

Runnable either way:
    python tests/test_aggregator.py       # plain asserts, prints OK
    pytest tests/test_aggregator.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.asl_t1_model import ASLT1Denoiser  # noqa: E402
from models.blocks import FrameReliabilityAggregator, VarianceFrameAggregator  # noqa: E402

B, T, HW = 4, 8, 32
ARCH = dict(asl_hw=128, t1_hw=128, base_ch=32, depth=4, use_t1_cross_fusion=True,
            t1_attn_max_tokens=1024, t1_task="recon", use_t1_decoder=False)


def _frames(seed=0, bad_at=None, bad_sigma=2.0):
    g = torch.Generator().manual_seed(seed)
    f = torch.rand(B, T, 1, HW, HW, generator=g) * 0.3
    if bad_at is not None:
        f[:, bad_at] += torch.randn(B, 1, HW, HW, generator=g) * bad_sigma
    return f


# --- P1 ---------------------------------------------------------------------
def test_weights_are_a_convex_combination():
    w = VarianceFrameAggregator()(_frames(bad_at=2))[1]
    assert bool((w >= 0).all())
    assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-5)


def test_padding_is_excluded():
    agg = VarianceFrameAggregator()
    f = _frames(bad_at=2)
    lens = torch.tensor([T, 5, 3, 2])
    w = agg(f, lengths=lens)[1]
    assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-5)
    for i, n in enumerate(lens.tolist()):
        assert bool((w[i, n:] == 0).all()), f"sample {i} put weight on padding"
    # two valid frames, neither corrupted -> 0.5 / 0.5
    assert torch.allclose(w[3, :2], torch.full((2,), 0.5), atol=1e-3)


def test_mask_and_lengths_agree():
    agg = VarianceFrameAggregator()
    f = _frames(bad_at=1)
    lens = torch.tensor([T, 6, 4, 2])
    mask = torch.arange(T)[None, :] >= lens[:, None]      # True = padding
    assert torch.allclose(agg(f, lengths=lens)[1], agg(f, mask=mask)[1], atol=1e-6)


# --- P2 / P3 ----------------------------------------------------------------
def test_uniform_when_no_frame_is_corrupted():
    w = VarianceFrameAggregator()(_frames())[1]
    assert (w - 1.0 / T).abs().max() < 0.02, w


def test_vetoes_a_corrupted_frame_and_keeps_the_rest_uniform():
    w = VarianceFrameAggregator()(_frames(bad_at=2))[1]
    w_bad = w[:, 2].mean()
    others = torch.cat([w[:, :2], w[:, 3:]], dim=1)
    assert w_bad < 0.01, f"bad frame kept weight {w_bad:.4f} (uniform would be {1/T:.4f})"
    assert (others - 1.0 / (T - 1)).abs().max() < 0.02, others
    # ... and the same policy the trained FRA converged to
    assert w_bad < 0.1 * (1.0 / T)


# --- P4 ---------------------------------------------------------------------
def test_tau_zero_is_the_plain_mean():
    f = _frames(bad_at=2)
    agg = VarianceFrameAggregator(tau_init=0.0)
    out, w = agg(f)
    assert (w - 1.0 / T).abs().max() < 1e-5
    assert torch.allclose(out, f.mean(1), atol=1e-5)


def test_tau_is_learnable_and_gets_gradient():
    agg = VarianceFrameAggregator()
    assert sum(p.numel() for p in agg.parameters()) == 1
    agg(_frames(bad_at=2))[0].square().mean().backward()
    assert agg.tau.grad is not None and float(agg.tau.grad.abs()) > 0.0


# --- P5 ---------------------------------------------------------------------
def test_brain_premask_does_not_change_the_weights():
    """Background is 0 in every frame, so masked and unmasked variance differ only by a
    common factor, which cancels in the softmax. This is why no mask input is needed."""
    agg = VarianceFrameAggregator()
    f = _frames(bad_at=2)
    brain = torch.zeros(1, 1, 1, HW, HW)
    brain[..., 4:28, 4:28] = 1.0
    assert torch.allclose(agg(f * brain)[1], agg(f * brain * 1.0)[1], atol=1e-6)
    w_full = agg(f * brain)[1]
    # explicitly restricting the variance to the brain box must give the same weights
    dev = (f * brain) - (f * brain).mean(1, keepdim=True)
    s2 = (dev.pow(2) * brain).sum((2, 3, 4)) / brain.sum().clamp_min(1.0)
    w_boxed = torch.softmax(-torch.log(s2.clamp_min(1e-8)), dim=1)
    assert (w_full - w_boxed).abs().max() < 0.05, (w_full[0], w_boxed[0])


# --- P6 + integration -------------------------------------------------------
def test_default_is_still_fra():
    torch.manual_seed(0)
    m = ASLT1Denoiser(**ARCH)
    assert m.aggregator_kind == "fra"
    assert isinstance(m.aggregator, FrameReliabilityAggregator)


def test_var_arm_builds_and_runs():
    torch.manual_seed(0)
    m = ASLT1Denoiser(**ARCH, aggregator="var", window_fusion_levels=2).eval()
    assert isinstance(m.aggregator, VarianceFrameAggregator)
    assert sum(p.numel() for p in m.aggregator.parameters()) == 1
    g = torch.Generator().manual_seed(1)
    x = torch.rand(2, 5, 1, 128, 128, generator=g)
    t1 = torch.rand(2, 1, 128, 128, generator=g)
    with torch.no_grad():
        assert m(x, x, t1)["asl_recon"].shape == (2, 1, 128, 128)
        assert m.infer_from_subset(x[:, :2], t1)["asl_recon"].shape == (2, 1, 128, 128)


def test_var_arm_is_smaller_than_fra():
    torch.manual_seed(0)
    n_fra = sum(p.numel() for p in ASLT1Denoiser(**ARCH).parameters())
    torch.manual_seed(0)
    n_var = sum(p.numel() for p in ASLT1Denoiser(**ARCH, aggregator="var").parameters())
    assert n_fra - n_var > 80_000, (n_fra, n_var)


def test_svfw_and_var_are_mutually_exclusive():
    try:
        ASLT1Denoiser(**ARCH, aggregator="var", use_svfw=True)
    except ValueError as e:
        assert "v42i_drop_svfw" in str(e)
    else:
        raise AssertionError("per-pixel SVFW + closed-form var should be refused")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} aggregator tests passed.")
