# Server Run Guide — CIG-Net v2 (ASL_dmvae)

Reproduce CIG-Net v2 training/eval on the HPC. The repo holds **code only** —
data, the stage-1 T1 checkpoint, and all run outputs live **outside** the repo
and are set via the variables in §3.

---

## Tianhe (天河) quick start

The cluster is **Tianhe**: SLURM-derived scheduler with `yh*` wrappers
(`yhbatch` submit, `yhq` queue, `yhcancel <id>` cancel, `yhrun` = srun),
partition **`gpu`**, env activated with `source activate`. The ready-made job
scripts in [`slurm/`](slurm/) already encode all of this.

```bash
# 0. clone / update the repo (login node; uses the SSH key — see §0):
sh env/hpc/sync_repo.sh              # clones to $REPO if missing, else git pull
cd /fs1/home/duancaohui/jian/projects/ASL_dmvae

# 1. build the conda env (run on a GPU node so torch.cuda + the mamba build work):
yhrun -p gpu -N 1 --gpus-per-node=1 --cpus-per-gpu=8 --pty /bin/bash
conda create -n asl-mamba python=3.11 -y      # create the env (install_env.sh also does this, idempotently)
sh env/hpc/install_env.sh                      # edit the CUDA wheel line inside for Tianhe's CUDA

# 2. point the config at the data (already set to the Tianhe path):
grep root_path env/hpc/configs/server_v37.yml
#   -> /fs1/home/duancaohui/jian/data/7T_ASL_denoising   (edit if yours differs)
#   also set REPO/EXP at the top of the slurm/*.sh if the repo/output dirs differ.

# 3. submit the single-seed pipeline FROM THE REPO ROOT:
sh env/hpc/slurm/submit_all.sh       # smoke -> p1 -> {p2, p2c} -> p3 -> p4
#   or one phase at a time:  yhbatch env/hpc/slurm/smoke.sh   (see submit_all.sh header)

# watch:  yhq        live log:  tail -f env/hpc/slurm/logs/<phase>-<jobid>.out
```

Default seed is **42** (single-seed-first). Add seeds later: `SEED=1 yhbatch
env/hpc/slurm/phase1_t1.sh` (and the rest). Reference paths in the scripts:
`REPO=/fs1/home/duancaohui/jian/projects/ASL_dmvae`, `EXP=/fs1/home/duancaohui/jian/exp`,
data `=/fs1/home/duancaohui/jian/data/7T_ASL_denoising` — all overridable by env var.

> §1–§2 below are the generic/manual env-build notes that `env/hpc/install_env.sh`
> automates; §5's manual `python -u runners/...` commands still work if you prefer
> running outside the SLURM scripts (prefix with `yhrun` on a GPU allocation).

---

## 0. Get the code

Private GitHub repo: `git@github.com:Jian-uon/ASL_dmvae.git`. The server needs an
SSH key that is registered on the GitHub account (Settings → SSH and GPG keys).

```bash
# one-time on the server, if no key yet:
ssh-keygen -t ed25519 -C "server"            # then add ~/.ssh/id_ed25519.pub to GitHub
ssh -T git@github.com                          # expect: "Hi Jian-uon! ..."

git clone git@github.com:Jian-uon/ASL_dmvae.git
cd ASL_dmvae
```

> HTTPS alternative (no SSH key): `git clone https://github.com/Jian-uon/ASL_dmvae.git`
> — for a private repo this prompts for a GitHub username + a Personal Access
> Token (Settings → Developer settings → Tokens), not your password.

## 1. System prerequisites

- NVIDIA driver with CUDA 13.0 (`nvidia-smi` shows the H100 and CUDA 13.0).
- CUDA toolkit (`nvcc`) is required to **build** mamba-ssm/causal-conv1d; its
  CUDA **major** must match the PyTorch wheel's CUDA major (see §2). Check with
  `nvcc --version`.
- `git`, `wget`, a C/C++ toolchain (`build-essential`) for wheels that compile.
- Miniconda/Anaconda (recommended) or `python3.10–3.12` + `venv`.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git wget
```

## 2. Python environment

```bash
# conda env
conda create -n asl-mamba python=3.11 -y
conda activate asl-mamba

# PyTorch FIRST. Server CUDA is 13.0 → install the cu130 wheel (H100 = sm_90).
# If the cu130 index is not up yet, cu128 also RUNS on a 13.0 driver (CUDA is
# backward-compatible and the wheel bundles its own runtime) — but then read the
# mamba build note below. Check current tags: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
#   fallback if cu130 unavailable:  --index-url https://download.pytorch.org/whl/cu128

# then the rest
cd ASL_dmvae
pip install -r requirements.txt

# selective-scan CUDA kernels (REQUIRED for speed; they COMPILE CUDA extensions).
# mamba-ssm's build checks that the toolchain CUDA major == torch.version.cuda major:
#   * torch cu130  + system nvcc 13.0     -> builds cleanly (recommended).
#   * torch cu128  -> the system 13.0 nvcc mismatches; install a CUDA 12.x
#       toolkit for the build and point CUDA_HOME at it, e.g.
#       `conda install -c nvidia cuda-toolkit=12.8` then `export CUDA_HOME=$CONDA_PREFIX`.
export TORCH_CUDA_ARCH_LIST="9.0"          # H100 = sm_90 (skip building other archs)
pip install ninja
pip install "causal-conv1d>=1.4" "mamba-ssm>=2.2" --no-build-isolation

# sanity: True + H100, and the CUDA scan kernel importable
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba_ssm CUDA kernel OK')"
```

> **Speed note.** `selective_scan_pytorch` (used by **both** `SS2D` and the
> MoSSM encoder) **auto-dispatches to the `mamba_ssm` CUDA kernel** when it is
> importable and the tensor is on CUDA — no code change needed. With mamba_ssm
> installed (above) every GPU run uses the fast kernel (50–100× on long
> sequences); without it the code silently falls back to the slow pure-PyTorch
> loop. So if the import check above fails, **fix the build before training** —
> otherwise runs are 50–100× slower with no error. All server runs use the
> kernel uniformly, keeping multi-seed results internally consistent.

## 3. Paths (set once per shell)

```bash
export REPO=$PWD                              # repo root (run from inside ASL_dmvae)
export PY=python                              # conda env python
export EXP=/data/asl_exp                      # <-- run outputs root (edit me)
export DATA=/data/ASL_denoising_dataset/data  # <-- dataset root (edit me)
export CONFIG=env/hpc/configs/server_v37.yml  # HPC config (own data path; see below)
```

`env/hpc/configs/server_v37.yml` is the server-side config (a copy of the local
`env/local/configs/wsl_asl_2d_home_v37.yml` differing only in `root_path`). Point
it at your data root — either edit the `root_path:` line directly, or:

```bash
sed -i "s#^  root_path:.*#  root_path: \"$DATA\"#" "$CONFIG"
grep root_path "$CONFIG"     # verify
```

Dataset layout expected: `$DATA/sub-<id>/raw/{asldata_diff.nii.gz, t1_in_asl.nii.gz, ...}`.

## 4. What to copy over (NOT in git)

| Item | Size | Note |
|------|------|------|
| Dataset (`$DATA`) | ~12 GB (329 subj) | `rsync -avP` from the source machine |
| Stage-1 T1 ckpt | small | optional — you can **retrain** it on the server (§5, Phase 1) instead of copying |

---

## 5. Task list

`$SEED` drives multi-seed. Core rigor = **3 seeds** (`42 1 2`). `MAX_STEPS=200`
captures the full useful checkpoint range (selection band ≤ step 125); use 500
for strict parity with the original schedule.

### Phase 0 — smoke test (always run first)
Confirms the env/data/paths and measures steps/sec.
```bash
$PY -u runners/train_t1.py --config $CONFIG --exp $EXP \
  --name smoke_t1 --base_ch 32 --depth 4 --seed 42 --max_steps 5 --save_every 5
```
If this completes and writes `$EXP/logs/smoke_t1/`, the pipeline is good.

### Phase 1 — Stage-1 T1 pretraining (run ONCE; produces the frozen prior)
Stage-2 loads `$EXP/logs/stage1_t1_300step/checkpoints/latest.pth`.
```bash
$PY -u runners/train_t1.py --config $CONFIG --exp $EXP \
  --name stage1_t1_300step --base_ch 32 --depth 4 --seed 42 \
  --w_seg 1.0 --seg_loss pv_l1_4cls --best_criterion seg \
  --max_steps 300 --save_every 100 --save_images
```

### Phase 2 — Stage-2 training × 3 seeds (the core rigor sweep = 9 runs)

Shared backbone recipe (no CADA — that is added per-model below):
```bash
COMMON="--config $CONFIG --exp $EXP --base_ch 32 --depth 4 \
  --use_mossm_encoder --mossm_blocks_per_scale 2 --mossm_n_directions 1 --mossm_d_state 16 \
  --no_tabs --cmam_k_rank_div 4 --use_rgsf --use_naf_fusion --use_mdta_fusion \
  --use_hybrid_mossm_block --mossm_n_directions_by_stage 1,1,2,2 --hybrid_local_expand 2.0 --hybrid_local_kernel 3 \
  --no_noise_var \
  --use_bdcyc --w_bdcyc 0.05 --ema_decay 0.997 --ema_start_step 40 \
  --adam_beta1 0.9 --adam_beta2 0.99 --warmup_steps 40 --grad_clip 0.5 \
  --bad_frame_p 0.5 --jinv_p 0.50 \
  --init_t1_from $EXP/logs/stage1_t1_300step/checkpoints/latest.pth --freeze_t1 \
  --save_every 25 --save_images --log_images 10 \
  --max_steps 200 --lr_scheduler cosine --lr_min 2e-6 \
  --best_criterion umse --best_min_step 40 --score_ema_alpha 0.3 --early_stop_patience 0"

CADA_LR="--use_cada --cada_stages 0,1,2,3 --cada_n_groups_B 4 --cada_lr --cada_lr_rank 8 --cada_lr_bound 4.0"

for SEED in 42 1 2; do
  # (a) CIG-Net v2 (main model)
  $PY -u runners/asl_t1_guided_runner_dmvae_n2n.py $COMMON $CADA_LR \
     --seed $SEED --name cignet_v2_seed$SEED

  # (b) ASL-only-MoSSM  (same arch + params, T1 input zeroed → isolates T1 info)
  $PY -u runners/asl_t1_guided_runner_dmvae_n2n.py $COMMON $CADA_LR --zero_t1 \
     --seed $SEED --name baseline_aslonly_seed$SEED

  # (c) naive-T1-concat (no CADA/CMAM; T1 as an unguarded input channel = strawman)
  $PY -u runners/asl_t1_guided_runner_dmvae_n2n.py $COMMON --no_cmam --naive_t1_concat \
     --seed $SEED --name baseline_naiveconcat_seed$SEED
done
```
> Crash-recovery: re-running the same command with `--resume` appended continues
> from the latest checkpoint (loses ≤ `save_every`=25 steps).

### Phase 3 — Best-CNR final-operating-point selection (per seed, all 3 core models)
Rule (updated 2026-06-27): the final operating-point checkpoint is simply the
**single saved checkpoint with the highest GM-WM CNR over the whole run**
(GLOBAL argmax CNR) — there is **no uMSE fidelity band and no checkpoint
averaging (no SWA)**. Run the **same one command for all 3 core models**
(matched selection — this also removes the old ours=CNR-SWA-vs-baseline=best-uMSE
selection-asymmetry confound). `eval_select_ckpt.py` evaluates every checkpoint
and copies the argmax-CNR one verbatim:
```bash
# per run: evaluate every ckpt, copy the global-max-CNR ckpt verbatim -> best_cnr.pth
$PY -u scripts/eval_select_ckpt.py --metric cnr --save_selected <run>/best_cnr.pth
```
> Honest trade-off: GLOBAL max-CNR can land on the degrading tail where uMSE
> rises above the old 1-SE bar and lapvar/hfen rise (more texture/noise). This
> is deliberate and user-accepted — **CNR is now the sole final-operating-point
> criterion; uMSE is no longer a guardrail on the final pick** (still reported as
> a diagnostic/supplementary). In-loop early-stop/best-ckpt still defaults to
> `--best_criterion umse`; selection still never uses L1 / psnr_ref / psnr_b.
> On `run_wd1e4_probe` the global-max-CNR pick is **step150** (CNR 0.6529, uMSE
> 0.00793) per `feasibility_full.json`.
>
> The old SWA helpers (`scripts/select_cnr_primary.py`,
> `scripts/build_swa_feasible.py`, `scripts/archive/build_swa_cnr_primary.sh`) remain on
> disk but are **retained/deprecated** — no longer wired into the pipeline.

### Phase 4 — Evaluation
- **Comparison table (matched selection across all methods, one val pass):**
  `scripts/run_comparison_v3.sh` (edit paths/seed; it pairs per-sample → valid Wilcoxon).
- **mismatched-T1 safety gate** (use the *original* subject brain mask `t1>0.05`,
  not the mismatched-T1 mask): `MODEL=hybrid_nonv bash scripts/run_mismatched_t1.sh`
  (and `MODEL=aslonly`, `MODEL=naive`).
- **n-frame sweep** (2/4/6/8/12): the comparison script takes `--n_frames`.

> The eval/SWA scripts in `scripts/` still hardcode the original machine's
> `cd /mnt/e/...`, `PY=/home/jian/...`, and `--exp /mnt/d/tmp/asl_exp`. Change
> those three constants (or `sed`) to your server's `$REPO`, `$PY`, `$EXP`
> before running. The training commands in Phase 1–2 above are already
> fully parameterized and need no edits.

---

## 6. Runtime estimate

The original **~4 min/step** figure was on the **pure-PyTorch fallback** (no
mamba_ssm installed). With the CUDA kernel (§2), the scan is 50–100× faster at
these sequence lengths, so expect dramatically lower per-step time on H100 —
**run the Phase-0 smoke and read the actual steps/sec** before scheduling the
9-run sweep. If GPU memory allows, the three models per seed can run
concurrently on one H100 (80 GB) or across GPUs to cut wall-clock further.

## 7. Outputs

Everything lands under `$EXP/logs/<run_name>/` (checkpoints, `stdout.txt`,
`val_images/`) and `$EXP/tensorboard/<run_name>/`. Selection metrics print on
each `VAL`/`BEST` line; see `docs/validation_metrics.md §9` for the metric suite
and the selection rule — ASL-QC metrics (CNR primary) lead reporting, and the
final operating point is the global max-CNR checkpoint (`best_cnr.pth`, Phase 3);
uMSE/uPSNR is reported as a diagnostic and is no longer a guardrail on the final
pick.
