# ASL_FastDenoising — Project Guide for Claude

Accelerated 7T ASL perfusion imaging by self-supervised denoising with anatomical (T1) guidance.

> **Provenance.** Split out of `ASL_denoising/ASL_dmvae` at commit `6d5dc5a` (2026-08-25). That
> repo continues to host a **different, unrelated paper line** (CIG-VSS + EC-LRDA, Mamba/VSS
> backbone, targeting KBS). **Nothing in this repo should reference or reuse that line's claims**
> — see §8. The two share only the dataset and the generic training/eval plumbing.

---

## 1. Goal

Reconstruct a high-SNR **Perfusion Weighted Image (PWI)** from a *small subset* of ASL
control–label difference frames (ΔM), using a co-registered T1w image as anatomical guidance.
7T single-PLD ASL, 12 NEX per subject; the aim is 12-NEX-comparable quality from far fewer frames
(sweep 2–8 at inference) — i.e. **scan-time acceleration**.

**Network output is ΔM (PWI), not CBF.** CBF (ml/100g/min) is available as a *post-hoc,
network-external* transform of the denoised ΔM via the consensus single-compartment model
([utils/cbf.py](utils/cbf.py)) using an external M0 + scalar sequence params. It never enters the
network, the loss, or checkpoint selection.

**Self-supervised regime — there is no clean ground truth.** Even the 12-NEX union is
noise-contaminated (motion, failed acquisitions, labelling-efficiency drift). Training is
**Noise2Noise** (Lehtinen ICML 2018): set_a → model → mean(set_b) as a noisy target. Any auxiliary
loss against a "reference" image is itself biased and is diagnostic only.

**Target venues.** CCR2026 abstract **submitted 2026-08-25** (title: *Self-Supervised Denoising for
Accelerated 7T ASL Perfusion MRI*). Journal target: **BSPC** (Biomedical Signal Processing and
Control) — application-oriented, no ASL paper ever published there (open niche). The paper
materials (`docs/v35_paper/`) are **local only — kept out of this public repo**; see AGENTS.md.

---

## 2. Data Characteristics & Pitfalls

Per subject, under `dataset.root_path / <subject_id> / <raw_dir>/`:

| File | Shape | Content |
|------|-------|---------|
| `asldata_diff.nii.gz` | `[H, W, Z, T]` | T ASL difference frames (**ΔM = control − label**, POSITIVE perfusion) |
| `t1_in_asl.nii.gz` | `[H, W, Z]` | T1w image registered to ASL space |
| `gm/wm/csf_asl.nii.gz` | `[H, W, Z]` | PV maps (**not used by the default no-seg arm**; only the `+seg` ablation) |
| `m0.nii.gz` | `[H, W, Z]` | M0 for post-hoc CBF |

329 subjects. 2D regime: each valid z-slice is one sample; per epoch the T frames are randomly
partitioned into set A (model input) and set B (N2N target).

**Pitfalls that bite repeatedly:**

- **The "GT" is noisy.** Anything that looks like a clean target (12-NEX mean, supervised L1
  against it) inherits motion artefacts and labelling variance. PSNR-vs-reference rewards noise
  mimicry — see §4.
- **`psnr_ref` peaks early then drops.** That is the noise-mimicry signature. Never use it for
  early stopping or selection.
- **T1 is sharper than PWI can ever be.** Letting T1 pixel content into the output produces
  visually impressive but anatomically *invented* structure. How much injection is allowed is now
  a **design parameter**, not a taboo — see §3.
- **Bad / mismatched frames exist** (motion, labelling failure). A naive temporal mean cannot
  detect them; the aggregator must learn non-uniform weighting. `bad_frame_p` injects a high-σ
  frame into set_a during training to force this.
- **N2N target ≠ clean.** `mean(set_b)` is a noisy estimate. Driving loss too hard against it
  makes the model over-smooth or mimic residual noise.
- **2D slicing discards through-plane context.** Known limitation of this line (the 2.5-D variant
  belongs to the other repo's backbone and is not wired here).
- **Brain mask is T1-derived (`t1 > 0.05`).** Loss is masked to brain. For mismatched-T1 tests use
  the *original* subject's mask, never the mismatched T1's.

---

## 3. Architecture (V35 line) and its constraints

```
frames [B,T,1,H,W] ──► FRA (frame reliability aggregator) ──► agg [B,1,H,W]
                          per-frame log-variance → softmax(−log σ²)  (BLUE weighting)
                                        │
agg ──► ASL ConvEncoder2D ──────────────┼──► skips {128, 64, 32, 16}
T1  ──► T1  ConvEncoder2D ──────────────┘──► skips {128, 64, 32, 16}
                                        │
        CMF0 @16×16 : Q=ASL, K=T1, V=ASL        ┐ T1GuidedCoarseHead
        Up 16→32 + ASL skip merge                │ (models/blocks.py:818)
        CMF1 @32×32 : Q=x,   K=T1, V=x          ┘
                                        │
        ASLDetailDecoder 32→64→128 (ASL skips only) ──► ŷ (PWI)
```

Code: `FrameReliabilityAggregator` [blocks.py:479](models/blocks.py#L479) · `CrossModalFusion`
[blocks.py:684](models/blocks.py#L684) · `T1GuidedCoarseHead` [blocks.py:818](models/blocks.py#L818)
· `ASLDetailDecoder` [blocks.py:1038](models/blocks.py#L1038). Selected by `--use_t1_cross_fusion`.
**3.52M params** in the default no-seg arm — 4.18M when the T1 decoder head is built
(`--t1_task seg`, `w_anat_roi > 0`, or `--keep_t1_decoder`).

### Default arm: NO-SEG (2026-08-25)

`--t1_task recon` ⇒ the T1 branch is a **pure encoder trained jointly from scratch**, receiving
gradient *only* through the cross-attention K-path. No stage-1 pretrain, no segmentation labels —
**training is fully label-free** (needs only ASL frames + T1). Under `recon`, `t1_seg_logits` is
None, so the tissue-similarity attention bias, `loss_seg` and `loss_contrast` are all inert (they
share the same guard). `T1_TASK=seg` re-enables the **`+seg` ablation arm** (`w_seg=1.0`;
`w_seg=0.1` was tried first and failed — the diluted gradient loses the shared T1 encoder to the
denoising objective and the seg stayed striped noise).

**Dead weight removed (2026-08-25).** Under `recon` with `w_anat_roi=0` the T1 *decoder* head got
zero gradient and its output was discarded, and `--premask_asl_inputs` re-ran the entire T1 branch
every step only to throw that 1-channel output away (the soft brain mask needs the 4-class seg
head). Both are gone: the runner now builds the model with `use_t1_decoder=False` (**4.18M → 3.52M
params, −16%**) and the pre-mask falls straight through to `t1 > 0.05` in `recon` mode. `asl_recon`
is **bit-identical** (verified against the old arch with shared weights); the T1 **encoder** is
untouched — it is still the CMF0/CMF1 K-path. The head is auto-kept when `--t1_task seg` or
`w_anat_roi > 0`, and `--keep_t1_decoder` forces it back (only needed for the T1-recon val panel);
`t1_task='seg'` + `use_t1_decoder=False` raises. Pre-change checkpoints still `--resume`:
`t1_decoder.*` keys are dropped with a log line, while any *other* unexpected key still fails
loudly ([runner `_filter_ckpt_state`](runners/asl_t1_guided_runner_dmvae_n2n.py#L1523)). Note the
param-init RNG stream shifts, so a fresh `--seed 42` run is no longer step-for-step comparable to
pre-change runs.

### Constraints — status as of 2026-08-25

| Constraint | Status |
|---|---|
| **V=ASL** (T1 may route attention but not inject pixel content) | **HOLDS — and is now STRUCTURAL, not a convention** (2026-08-26). The fine scales gained [multi-scale window cross-fusion](docs/multiscale_window_design.md): `Q=ASL, K=T1, V=ASL unprojected`, no output projection, gate `g=σ(a)∈(0,1)`. The fused output is therefore a **convex combination** of the ASL values inside each window — a maximum principle `min_win(x) ≤ x' ≤ max_win(x)`, locked by `tests/test_window_fusion.py`. T1 decides *what gets averaged with what* and cannot contribute content. Adding a `W_v` or output projection would silently destroy this. Injection-style designs (BAI) were considered and **dropped** — see the design doc §6. |
| Decoder fine scales T1-free | ⚑ **RELAXED, deliberately** (2026-08-26). `ASLDetailDecoder` takes T1 detail skips when `--window_fusion_levels > 0`; at 0 the modules are not constructed and the decoder cannot see T1 at the signature level (bit-exact to the older arch, old ckpts load). Because V=ASL is structural here, this does **not** open a content path — but the mismatched-T1 / E0.3-lesion battery in §6 is still an `--window_fusion_levels 0` measurement and must be re-run per arm. Watch **over-smoothing** (lapvar / EFC), not leakage: the module can only average. |
| Best ckpt / early-stop never use L1, `psnr_ref`, `psnr_b` | **HOLDS — do not violate.** |
| Single-direction N2N (only set_a through the model; no symmetric/round-trip losses on B) | **HOLDS.** |
| No clean-GT loss term | **HOLDS.** Anything supervising against the 12-NEX union is diagnostic only, weight 0. |
| Label-free training (no PV/seg labels in the default arm) | **HOLDS** — it is now a selling point. |

---

## 4. Validation & Model Selection

- **Primary metric (selection + reporting): `uMSE` / `uPSNR`** (unbiased linear-scale risk,
  Marcos-Morales 2023), SSIM secondary. In-loop: `--best_criterion umse`.
- **Supplementary (reported, never selected on):** `CNR`/`cnr_ref` (Wang GM-WM contrast),
  `sCoV_GM/WM`, `lapvar`, `EFC`, `hfen`, `gmsd`, and the biased `psnr_ref` / `psnr_b`.
- **Operating point:** the in-loop best is **gated** by `best_min_step` (falls back to
  `sure_anneal_start`, i.e. step 200 in the current config), which can miss an earlier global
  optimum. **Always confirm with post-hoc selection** over the periodic snapshots:
  `scripts/eval_select_ckpt.py --metric umse --save_selected <run>/checkpoints/best_umse_posthoc.pth`.
  On `run_v35_joint_wseg1_seed42` the post-hoc pick (step 50, uMSE 0.0053) beat the in-loop
  `best_umse.pth` (step ~205, 0.0078) by 32%.
- **⚠ CNR can be fooled by sharpening.** If T1 injection (§7) raises CNR while uMSE worsens, that
  is anatomical hallucination, not real contrast. Report both; let uMSE arbitrate.
- **Sanity checks before claiming a result:** mismatched-T1 leakage; n_frames sweep must degrade
  gracefully 12 → 2; injected-lesion retention.

Metric formulas: [docs/validation_metrics.md](docs/validation_metrics.md).

---

## 5. Training Regime

- **Single stage.** No stage-1 pretrain in the default arm (§3).
- **Augmentation.** `bad_frame_p` (high-σ frame into set_a, forces non-uniform aggregation);
  `jinv_p` (J-invariant masking, Batson & Royer 2019).
- **EMA always on**; validation runs on EMA weights (`--ema_decay`, default 0.9999).
- **Loss recipe** (config): `w_n2n 0.7, w_grad 0.8, w_ssim 0.2, w_contrast 0.5, w_tv 0.2,
  w_sure 0.02` (SURE warmup 60 / anneal 200 → 0.005). `w_seg 1.0` and `w_contrast` fire only in
  the `+seg` arm.
- **`--premask_asl_inputs` is required** for this line (it replaces the removed `w_bg` background
  term; default-off in the runner since 2026-07).
- **Seed 42** by default; full determinism not guaranteed (CUDA + multi-worker loader).
- **Runtime:** ~3 min/step (627 batches) on an RTX 5070 Ti; a 500-step run early-stops around
  step 300 ⇒ ~15 h.

**Launch (local):**
```bash
python runners/asl_t1_guided_runner_dmvae_n2n.py \
  --config env/local/configs/win_asl_2d_home_v35_joint.yml --exp D:/tmp/asl_exp \
  --name run_v35_joint_noseg_seed42 --base_ch 32 --depth 4 \
  --use_t1_cross_fusion --t1_attn_max_tokens 1024 --t1_task recon --premask_asl_inputs \
  --bad_frame_p 0.3 --save_every 50 --save_images --log_images 10 \
  --early_stop_patience 20 --early_stop_min_evals 60 \
  --best_criterion umse --save_per_metric_best
```

**Launch (Tianhe HPC):** `git pull && yhbatch env/hpc/slurm/submit_v35_joint.sh`
(knobs: `SEED=1`, `T1_TASK=seg` for the ablation arm, `MAX_STEPS`, `EXTRA="--resume"`).

---

## 6. Results so far (`run_v35_joint_wseg1_seed42`, the `+seg` arm, seed 42)

Operating point = `best_umse_posthoc.pth` (step 50). Subject-level pooled, held-out split:

| Metric | Value |
|---|---|
| uMSE / uPSNR | 0.0053 / 22.77 dB |
| **CNR vs 12-frame average** | **0.729 vs 0.510 (+43%)** |
| sCoV GM / WM | 0.390 / 0.403 |
| n-frames sweep (2→8) | CNR 0.721–0.728 — essentially flat |
| Mismatched-T1 leakage | L1 0.0285, SSIM 0.931 |
| Injected 4σ lesions (GM/WM) | visible, spillover 0.025–0.027 |
| rCBF agreement vs 12-frame | ICC 0.989 (n=2) → 0.999 (n=8), BA bias ≈ 0 |

The **no-seg** arm (`run_v35_joint_noseg_seed42`, now the default) reached uMSE 0.0073 / CNR 0.735
by step 20 — equal or better than the seg arm at the same point — but the run was interrupted and
needs to be completed (`--resume`) before the comparison is final.

---

## 7. Open work

1. **Finish the no-seg run** and run the full protocol on it (post-hoc selection → mismatch → E0.3
   injection → sweep → CBF), then confirm it as the main arm.
2. **T1-effectiveness check** (important): with no seg supervision the T1 encoder's only gradient
   is the attention-K path — a known inertness pattern in this codebase. Compare the same
   checkpoint under T1-zeroed / T1-permuted inference against a PlainUNet-N2N baseline. If T1 is
   effectively dormant, the "anatomical guidance" story is empty regardless of metrics.
3. **Multi-scale window cross-fusion** (the method contribution) — **implemented and smoke-tested
   2026-08-26**; spec, measured constants and arms in
   **[docs/multiscale_window_design.md](docs/multiscale_window_design.md)**. The coarse scales keep
   CMF0/CMF1 (global, 16²/32²); the decoder's 64²/128² gain the same tissue-guided attention *at the
   feature's own resolution*, bounded by `ws×ws` windows instead of by pooling — pooling to a fixed
   token budget would demote both fine levels to the 32² guidance resolution CMF1 already has, and
   32² cannot resolve a 1–2 px cortical ribbon. `Q=ASL, K=T1, V=ASL unprojected` ⇒ convex combination
   ⇒ V=ASL is structural (§3). +17.2K params (+0.49%), +47% GPU step time but only ~+7% wall clock
   (training is loader-bound). Flags: `--window_fusion_levels {0,1,2} --window_size --window_k_source
   {t1,asl}`; `0` builds nothing (bit-exact old arch, old ckpts load). **Runs to do:** A1 (`2 t1`,
   main), A3 (`2 asl`, the T1-free control — *A1 − A3 is the net effect of anatomical guidance*, and
   it subsumes open question 2), A4 (`1`, cheapest). Watch `window/wf*_entropy` in TB: pinned at
   ln(ws²) means the grouping never got learnt and the module is a box blur. Risk profile is
   **over-smoothing, not hallucination** — this module can only average, so arbitrate on uMSE with
   lapvar/EFC as the guard, not on leakage. Injection-style designs (BAI) were specced and dropped:
   [docs/wavelet_bai_design.md](docs/wavelet_bai_design.md) (superseded).
4. **Baselines to train:** PlainUNet-N2N (doubles as the no-T1 lower bound), SwinIR-N2N,
   Nb2Nb/N2V, naive T1-concat (needs a small conv-path fix), plus classical BM3D/AONLM and plain
   temporal averaging (eval-only).
5. **Multi-seed** (seeds 1, 2) — everything so far is n=1.
6. **External validation:** OSIPI synthetic DRO (has clean GT ⇒ the only unbiased external
   PSNR/SSIM anchor) and the dai in-vivo 3T control set (needs `flirt struc2asl` adaptation).

Full figure/table checklist and the training matrix: `docs/v35_paper/experiment_plan.md`
(**local only — not in this public repo**).

### Closed questions — do not re-explore

- **Auxiliary T1 reconstruction (`w_anat_roi > 0`, the "B1" arm): REJECTED 2026-08-31 on
  evidence.** Restoring the T1 decoder head gives the T1 encoder a second, self-supervised
  gradient besides the cross-attention K-path, addressing the inertness risk in §7 item 2
  without costing the label-free property (the target is the input T1). It was tried at
  `w_anat_roi 0.03` and came out **worse than the encoder-only arm**, so the encoder-only arm
  (`w_anat_roi = 0`, T1 decoder dropped, 3.45M params) remains the reference model.
  ⟨TBD: paste the uMSE/CNR comparison from the server so this entry carries its evidence.⟩
  The plumbing stays in place — `--w_anat_roi` on the runner, `W_ANAT` on
  `submit_v35_joint.sh`, the B1 rows in the matrix — so the arm is one flag away if a reviewer
  asks, but it should not be re-run speculatively.
  *Consequence:* **Figure 1 must be corrected.** It draws the T1 decoder, a "Reconstructed T1w"
  output and an MSE loss on it. None of the three exists in the reference model, the loss is a
  masked L1 rather than an MSE, and the figure contradicts its own caption ("The T1 path has no
  decoder"). It was left standing only while this arm might have won.

- **Band-Limited Guidance (frequency-domain content guard): REJECTED 2026-08-25 on evidence.**
  A DWT decomposition of the mismatched-T1 leakage ([scripts/leakage_spectrum.py](scripts/leakage_spectrum.py))
  found only 12.8% of leakage energy in the detail subbands (below the pre-registered 20% GO
  threshold), and the imprint test was negative: correlation of the high-frequency leakage with the
  *wrong* T1's edges was +0.032, **below** the null baseline of +0.098. Conclusion: the
  high-frequency leakage is diffuse numerical perturbation, **not** T1 structural imprint, so a
  frequency guard would remove something that is neither large nor anatomically shaped.
  *Positive by-product for the paper:* leakage in this architecture has no anatomical morphology —
  a stronger safety statement than the L1 number alone.
- **`w_seg = 0.1`** for the seg arm: too weak, seg never converged. Use 1.0.

---

## 8. Isolation from the CIG-VSS line (source repo)

The source repo's other paper (KBS) owns these — **never claim, reuse, or cite them as this
project's contributions**: soft-PV bilinear conditioning (EC-LRDA), Bayesian/`c_sem` evidence
gates, FSL/PV conditioning sources, CondPyramid/TF-DM, the VSS/Mamba backbone, and the
scan-direction-dropout MC uncertainty / hallucination-index / conformal-on-set_b pillar.

The two papers share the dataset and the generic Noise2Noise + uMSE plumbing, so when both are
submitted they must **cross-cite and state the methodological difference** explicitly.

Retired code from other lines (MoSSM, VMamba, NAFDecoder, SVFW, EC-LRDA …) is **still present in
`models/`** and must not be deleted: `models/asl_t1_model.py` imports those classes unconditionally
at module scope, and every checkpoint's `arch` dict replays all constructor kwargs
(`ASLT1Denoiser(**arch)` has no `**kwargs` catch-all), so removing them breaks loading of existing
checkpoints and the eval scripts that rebuild from them.

---

## 9. Repo Layout

```
config/         conf_data.py — config dataclasses
dataio/         MONAI-based dataset / loader / generator
models/         Model classes + building blocks (blocks.py holds FRA / CMF / heads)
losses/         N2N, SSIM, contrast, J-invariant, SURE …
runners/        Training entrypoint (asl_t1_guided_runner_dmvae_n2n.py) + baselines
scripts/        Eval + analysis tooling (selection, mismatch, injection, CBF, spectrum …)
utils/          CBF, EMA, collate, metrics, batch prep
env/local/      Local (Windows) config
env/hpc/        Tianhe: env.sh + slurm/submit_v35_joint.sh + server config
docs/           validation_metrics.md, v37_legacy.md, multiscale_window_design.md
                (v35_paper/ + patent record + related_work are LOCAL ONLY — see AGENTS.md)
```

**Reading order for new context:** this file → [docs/validation_metrics.md](docs/validation_metrics.md)
→ [docs/multiscale_window_design.md](docs/multiscale_window_design.md). If you have the local
working copy, `docs/v35_paper/README.md` and `docs/v35_paper/experiment_plan.md` come first —
they are kept out of this public repo (AGENTS.md explains what else is).

**Windows vs WSL/HPC:** this line is conv-only — no mamba-ssm CUDA kernels needed, so it trains
fine on Windows GPU. (The `monai` conda env at `D:/softwares/anaconda/envs/monai` has torch 2.9 +
cu128 and pytorch_wavelets.)
