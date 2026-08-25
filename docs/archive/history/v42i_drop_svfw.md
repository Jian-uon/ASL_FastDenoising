# v42i — Drop SVFW, Restore SetTransformer Aggregator

> **Date**: 2026-05-19
> **Branch**: master
> **Parent**: v42h (= v42g + loss/optimizer tweaks)
> **Motivation**: scripts/bypass_svfw_probe.py confirmed SVFW *amplifies* per-pixel noise at T_a=6 frames. Zero-train inference with SVFW replaced by uniform mean produced 49% lower lapvar in recon. The aggregator itself was the dominant structural noise source.

## 1. Discovery summary

[scripts/decompose_recon.py](../scripts/decompose_recon.py) on v42h `best.pth` revealed:

| Quantity | A direct mean | SVFW agg | recon |
|---|---|---|---|
| lapvar | 0.095 | **0.377** | 0.024 |
| σ_residual | 0.098 | **0.187** | 0.057 |

SVFW agg was **4.4× noisier than uniform direct mean** — the per-pixel BLUE weighting was producing patchwork artefacts. Reasoning: with T_a=6 frames, the log-variance head cannot stably estimate per-pixel σ²_t (deviation D_t = Y_t − μ has same noise std as raw frame). softmax(-log σ²) then concentrates weight on a single random frame per pixel → output is a patchwork of different frames' noise realisations, **worse than uniform averaging**.

[scripts/bypass_svfw_probe.py](../scripts/bypass_svfw_probe.py) tested the zero-train fix: monkey-patch `model.aggregate_asl` to return `set_a.mean(1)` instead of SVFW output. Results across 4 val subjects:

| | SVFW agg → recon | uniform agg → recon | improvement |
|---|---|---|---|
| recon lapvar (mean) | 0.047 | **0.024** | **−49%** |
| recon σ_res (mean) | 0.062 | **0.052** | −16% |
| subj 2 (worst case) | lv=0.148 | lv=0.064 | **−57%** |

Every subject improved. Caveat: model was trained on SVFW aggregator, so feeding uniform mean is mildly OOD — retraining without SVFW is expected to do even better.

## 2. v42i = v42h minus `--use_svfw`

Single change. The fallback path in [models/asl_t1_model.py:124-130](../models/asl_t1_model.py#L124-L130) already constructs `SetTransformerAggregator` when `use_svfw=False`:

```python
if bool(use_svfw):
    self.aggregator = SpatialVaryingFrameWeighting(in_ch=in_ch, hidden=32)
else:
    hidden_ch = max(64, base_ch * 2)
    hidden_ch = (hidden_ch // 4) * 4  # divisible by 4 heads
    self.aggregator = SetTransformerAggregator(in_ch=in_ch, hidden_ch=hidden_ch, n_heads=4)
```

So v42i requires no model code changes — only dropping the flag in [scripts/auto_chain_v42_wsl.sh](../scripts/auto_chain_v42_wsl.sh).

## 3. SetTransformer aggregator overview

[models/blocks.py](../models/blocks.py) `SetTransformerAggregator` (Lee et al. ICML 2019):

```
Input: frames [B, T, 1, H, W]
   │
   ├─→ per-frame feature encoder (Conv2d stack) → frame embeddings [B, T, C]
   │
   ├─→ Multi-head self-attention over T (frames "talk to each other")
   │      ↓
   │   per-frame logits [B, T, 1]
   │      ↓
   │   softmax over T (with optional length-mask)
   │      ↓
   │   weights [B, T, 1, 1, 1] — SCALAR per frame, broadcast over H,W
   │
   └─→ agg = Σ_t w_t · frames[t]   (same w for every pixel of frame t)
```

**Key contrast vs SVFW**: weights are *per-frame scalar*, broadcast identically to every pixel. Cannot produce per-pixel patchwork. At init `w ≈ 1/T` (uniform); training may learn to down-weight specific frames but cannot redistribute weight pixel-wise.

Params: ~10k. At T_a=6 it can in principle suppress bad frames as a whole (e.g. motion-corrupted frame gets w → 0), which is exactly what `--bad_frame_p` augmentation was designed to exercise.

## 4. Complete v42i pipeline

```
set_a [B,T_A,1,H,W] ─→ SetTransformerAggregator ─→ agg [B,1,H,W]
                              (per-frame scalar)
t1 [B,1,H,W] ─→ T1 encoder ─→ t1_feat + t1_skips
              ─→ T1 decoder ─→ t1_seg [B,4,H,W]
                               │
                               ▼
              t1_skips_lp = T1LowPass(k=7, n_iter=2)(t1_skips)   ← P1 anti-leak
                               │
agg ─→ MoSSM-ASL Encoder (no TABS)
       per stage:
         B_i = W_B_asl(u_i) + σ(g_B)·W_B[u_i; t1_lp,i]    ← P2 gated B/Δ residual
         Δ_i = softplus(W_Δ_asl(u_i) + σ(g_Δ)·W_Δ[u_i; t1_lp,i])
         C_i = W_C(u_i)                                   ← V=ASL #1
       at stages s where H·W ≤ 1024 (32×32 + 16×16): CMAM
         K = Bottleneck(t1_feat_lp)  d_t1 → d_t1/4 → kdim   ← P3 low-rank K
         V = mossm_mem                                       ← V=ASL #2
         + BLUE attention (key bias + query gate)
       → feat_map [B,256,16,16] + asl_skips
                               │
                               ▼
              T1GuidedCoarseHead (L0+L1; T1-free in default)
                               │
                               ▼
              ASLDetailDecoder (L2+L3; pure ASL)
                  RGSF: w_s = σ(a_s − softplus(b_raw_s) · log Var_T(set_a))   ← softplus fix
                  → raw_recon
                               ▼
              ResHead-with-cap: asl_recon = agg + 0.5·tanh(raw_recon)
                               ▼
                       asl_recon [B,1,H,W]

Target = mean(set_b)        N2N
Loss   = 0.3·L_n2n + 0.3·L_grad + 0.3·L_ssim + 0.05·L_contrast + 0.5·L_bdcyc
                              (CC-5 cycle, stop-grad)
Anti-leak training:
  - T1 dropout p=0.15      (15% steps see zero T1)
  - J-invariant masking p=0.5
  - Bad frame injection p=0.5
```

## 5. Diff against v42h

| Component | v42h | **v42i** |
|---|---|---|
| **Aggregator** | SVFW per-pixel BLUE | **SetTransformer (per-frame scalar)** |
| `--use_svfw` flag | enabled | **removed** |
| Everything else | — | unchanged |

Carryover from v42g (anti-T1-leak):
- T1 low-pass k=7, iter=2 on `t1_skips`
- MoSSM gated B/Δ residual (init σ(-4)≈0.018)
- CMAM K low-rank bottleneck (k_rank_div=4)
- no_tabs
- `w_contrast 0.5 → 0.05`

Carryover from v42h (loss/opt):
- `w_grad 0.7 → 0.3` (halved — was a secondary noise source)
- `lr 5e-5 → 2e-5`, `weight_decay 3e-4 → 1e-3`
- `--warmup_steps 30`, `--grad_clip 0.5`
- `--jinv_p 0.50`, `--bad_frame_p 0.5`
- `max_steps 1500 → 500`, `best_min_step 200 → 80`
- `swa_start_step 100 → 80`, `early_stop_patience 80 → 50`

Other unchanged: RGSF with `b=softplus(b_raw)` (v42g bug fix); ResHead-with-cap (δ_max=0.5); CC-5 with `w_bdcyc=0.5`; T1 dropout p=0.15.

## 6. Expected outcome

| Metric | v42h | **v42i target** |
|---|---|---|
| recon lapvar @ best | 0.047 | **≤ 0.024** (zero-train probe baseline) |
| σ_res @ best | 0.062 | **≤ 0.052** |
| best upsnr_cyc step | 85 | 100–200 |
| best upsnr_cyc | 20.55 | ≥ 20.5 |
| visible noise from step 5 | severe | **substantially reduced** |
| T1 leak (probe ratio) | TBD | preserve v42g level (~8% target) |

## 7. SVFW post-mortem (lessons for future aggregator design)

The SVFW failure adds a third entry to our "retired-with-postmortem" list alongside NAR and AMD (see [output_space_constraints.md](output_space_constraints.md)):

| Module | Intended mechanism | Why it failed | Detection method |
|---|---|---|---|
| NAR (v42b) | per-pixel σ-clipped residual head | σ̂ from 6 frames unstable; hit clamp floor → degenerate to baseline | dark-dots visual + probe |
| AMD (v42c) | anchor-memory feature gating | attention collapsed to uniform → learnt per-channel bias | attention-dist probe |
| **SVFW (v42i)** | per-pixel BLUE temporal aggregation | per-pixel σ̂² unstable at T=6 → softmax picks single frame per pixel → output is patchwork; **noise *amplifier* not reducer** | lapvar comparison vs uniform mean |

**Common lesson**: any module that estimates *per-pixel* statistics from a small number of frames (T<10) is at high risk of estimator-variance collapse. Per-frame scalars (SetTransformer / RobustFrameAggregator) are structurally safer at small T.

## 8. CLI (v42i full)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -u runners/asl_t1_guided_runner_dmvae_n2n.py \
  --config config/wsl_asl_2d_home_v37.yml --exp /mnt/c/tmp/asl_exp \
  --name run_full_v42i --base_ch 32 --depth 4 --seed 42 \
  \
  --use_mossm_encoder --mossm_blocks_per_scale 2 --mossm_n_directions 1 --mossm_d_state 16 \
  --no_tabs \
  --t1_lowpass_kernel 7 --t1_lowpass_iter 2 \
  --mossm_t1_gated --cmam_k_rank_div 4 \
  --use_rgsf --use_reshead --reshead_delta_max 0.5 \
  --use_bdcyc --w_bdcyc 0.5 \
  --t1_dropout_p 0.15 \
  --warmup_steps 30 --grad_clip 0.5 \
  \
  --bad_frame_p 0.5 --jinv_p 0.50 \
  --use_swa --swa_start_step 80 \
  --max_steps 500 --lr_scheduler cosine --lr_min 2e-6 \
  --best_criterion upsnr_cyc --best_min_step 80 \
  --early_stop_patience 50 --early_stop_min_evals 100 \
  --save_every 20 --save_images --log_images 10 \
  \
  --init_t1_from /mnt/c/tmp/asl_exp/logs/stage1_t1_300step/checkpoints/latest.pth \
  --freeze_t1
```

Launch wrapper: [scripts/auto_chain_v42_wsl.sh](../scripts/auto_chain_v42_wsl.sh).

## 9. One-sentence summary

> v42i removes the v37-era SVFW per-pixel BLUE aggregator after probe-confirmation that it amplifies frame noise at T_a=6 (4.4× higher lapvar than uniform mean, patchwork pattern from unstable per-pixel σ² estimation), falling back to the per-frame scalar SetTransformer aggregator while preserving all v42g/v42h anti-T1-leakage and convergence-delaying changes.

## 10. File summary

| File | Change |
|---|---|
| [scripts/auto_chain_v42_wsl.sh](../scripts/auto_chain_v42_wsl.sh) | Removed `--use_svfw` from both smoke and full launch |
| [docs/v42i_drop_svfw.md](v42i_drop_svfw.md) | **this file** |
| [docs/README.md](README.md) | Index updated; v42i is current main method |
| [scripts/decompose_recon.py](../scripts/decompose_recon.py) | New: recon decomposition probe |
| [scripts/bypass_svfw_probe.py](../scripts/bypass_svfw_probe.py) | New: zero-train SVFW bypass probe |

No model code changed for v42i; only flag removed.
