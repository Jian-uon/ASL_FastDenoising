# ASL_FastDenoising

Accelerated 7T arterial spin labeling (ASL) perfusion imaging by **self-supervised denoising with
anatomical (T1) guidance** — recovering 12-NEX-comparable perfusion image quality from a small
subset of the acquired control–label difference frames.

- **No clean ground truth**: trained with Noise2Noise on disjoint frame subsets.
- **Label-free by default**: the T1 branch is a pure encoder trained jointly from scratch — no
  segmentation labels, no pretraining stage. Only ASL frames + a co-registered T1 are required.
- **Anatomical guidance through cross-attention** at coarse scales (16×16, 32×32).

## Status

| | |
|---|---|
| Main run | `run_v35_joint_noseg_seed42` (no-seg arm; `+seg` arm available as ablation) |
| Best result so far | uMSE 0.0053 · uPSNR 22.8 dB · **CNR 0.729 vs 0.510 for the full-frame average (+43%)** |
| Degradation | flat CNR from 8 down to 2 input frames |
| rCBF agreement (post-hoc, vs full-frame) | ICC ≥ 0.989, near-zero bias |
| Conference | CCR2026 abstract submitted (2026-08-25) |
| Journal target | BSPC (Biomedical Signal Processing and Control) |

## Quick start

```bash
# local (Windows / single GPU)
python runners/asl_t1_guided_runner_dmvae_n2n.py \
  --config env/local/configs/win_asl_2d_home_v35_joint.yml --exp D:/tmp/asl_exp \
  --name run_v35_joint_noseg_seed42 --base_ch 32 --depth 4 \
  --use_t1_cross_fusion --t1_attn_max_tokens 1024 --t1_task recon --premask_asl_inputs \
  --bad_frame_p 0.3 --save_every 50 --save_images --log_images 10 \
  --early_stop_patience 20 --early_stop_min_evals 60 \
  --best_criterion umse --save_per_metric_best

# HPC (Tianhe / SLURM)
git pull && yhbatch env/hpc/slurm/submit_v35_joint.sh     # SEED=1 | T1_TASK=seg | EXTRA="--resume"
```

Post-training, always confirm the operating point with post-hoc selection (the in-loop best is
step-gated and can miss an earlier optimum):

```bash
python scripts/eval_select_ckpt.py --ckpts <run>/checkpoints/step*.pth \
  --runner_args "<same flags as training>" --metric umse \
  --output <run>/selection_umse.json --save_selected <run>/checkpoints/best_umse_posthoc.pth
```

Evaluation chain: `scripts/test_mismatched_t1.py` (T1 leakage) · `scripts/test_signal_injection.py`
(lesion retention) · `scripts/eval_select_ckpt.py --nframe_sweep` (frame-budget degradation) ·
`scripts/eval_cbf.py` (post-hoc CBF agreement) · `scripts/leakage_spectrum.py` (leakage band/imprint
analysis).

## Documentation

Start with [CLAUDE.md](CLAUDE.md) — architecture, constraints, validation policy, open work, and
the decisions that must not be re-litigated. Paper materials (`docs/v35_paper/`) and the patent
drafts are kept **out of this public repo** — see AGENTS.md.

## Provenance

Split out of `ASL_denoising/ASL_dmvae` at commit `6d5dc5a` (2026-08-25) so this line can be worked
on without entangling the CIG-VSS + EC-LRDA paper line that remains in that repository. The two
share the dataset and generic training/eval plumbing only; see CLAUDE.md §8 for the isolation
rules. Retired code from other architecture lines is intentionally retained under `models/` —
deleting it breaks checkpoint loading (see CLAUDE.md §8).
