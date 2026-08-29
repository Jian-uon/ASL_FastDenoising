#!/bin/sh
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8 -J v35_joint
#SBATCH -o env/hpc/slurm/logs/v35j-%j.out
# NO `-e`: stderr is merged into the .out on purpose. The runner logs through
# Python logging and tqdm, BOTH of which write to stderr, so with separate files
# the .out holds only the shell echoes and looks empty while training is fine --
# which cost an afternoon of debugging on 2026-08-26. One file, chronological.
#SBATCH --time=72:00:00           # 500 epochs conv backbone (~3.3 min/epoch on RTX 5070 Ti => ~28h; margin for slower nodes)
# Tianhe job — V35-joint (user decision 2026-08-24): revive the V35 line
# (--use_t1_cross_fusion: FRA aggregator + multi-scale tissue cross-attn,
# CMF0@16x16 + CMF1@32x32, V=ASL, T1-free detail decoder) but WITHOUT the
# stage-1 T1 pretrain — T1 encoder/decoder train JOINTLY with the ASL branch
# from scratch. Signed-off exception to CLAUDE.md §3 "Stage-2 freezes the T1
# branch" (joint training historically destabilised both branches; the local
# probe run_v35_joint_seed42 on the Windows box is the stability gate — only
# launch this after that probe looks sane).
#
# Launch-critical flags (each fixes a known V35 fault or joint-training need):
#   --t1_task recon (DEFAULT since 2026-08-25) = NO-SEG main: pure T1-encoder
#                            joint training, label-free (see T1_TASK block below);
#                            T1_TASK=seg = the "+seg" ablation arm
#   --best_criterion umse    fault (a): composite_v2 was removed from argparse
#   --premask_asl_inputs     V35 dependency: replaces w_bg (default-off since 2026-07)
#   NO --init_t1_from / --freeze_t1   = joint training
#   No --amp: numerics kept identical to the local probe.
#
# Usage — SUBMIT FROM THE REPO ROOT:
#   sh env/hpc/sync_repo.sh                   # config/scripts must be pushed first
#   yhbatch env/hpc/slurm/submit_v35_joint.sh
#
# ⚠ The `#SBATCH -o` path above is relative to the SUBMISSION directory, and Slurm
#   creates that file BEFORE this script runs (so before the `cd "$REPO"` below).
#   Submitting from elsewhere, or a missing env/hpc/slurm/logs/, makes the job die at
#   launch with no .out, no .err and nothing in yhq — the diagnostics have nowhere to
#   go. The directory is kept in git via .gitkeep; `mkdir -p env/hpc/slurm/logs` if in
#   doubt. (The bare `logs/` rule in .gitignore used to swallow it — fixed 2026-08-26.)
# Knobs: SEED=1 | MAX_STEPS=500 | EVAL_EVERY=5 | SAVE_EVERY=50 | EXTRA="--resume"
#        WIN_LEVELS=2 WIN_K=t1|asl   window cross-fusion arms A1 / A3 (see below)
#        W_ANAT=0.03                 T1-reconstruction loss weight; >0 restores the T1
#                                    decoder head, which is the architecture Figure 1 draws
set -eo pipefail
# Unconditional banner BEFORE anything that can fail (cd, source env.sh). A job that
# dies silently with an empty .out used to give no clue where; now the last line
# printed tells you exactly how far it got. DEBUG=1 additionally turns on `set -x`.
echo "[job] v35_joint $(date +'%F %T') host=$(hostname) jobid=${SLURM_JOB_ID:-?} pwd=$PWD"
if [ "${DEBUG:-0}" = "1" ]; then set -x; fi

REPO=${REPO:-/fs1/home/duancaohui/jian/projects/ASL_FastDenoising}   # <-- MUST be its own clone, separate from ASL_dmvae
cd "$REPO"
CONFIG=${CONFIG:-env/hpc/configs/server_v35_joint.yml}         # V35-joint loss recipe + server data root
source env/hpc/env.sh          # CUDA module + conda + offline vars + EXP/PYTHONPATH/MASTER_PORT
nvidia-smi -L || true

RUNNER=runners/asl_t1_guided_runner_dmvae_n2n.py
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-500}
EVAL_EVERY=${EVAL_EVERY:-5}    # v35 historical cadence (heavy; EVAL_EVERY=10 halves val overhead)
SAVE_EVERY=${SAVE_EVERY:-50}
EXTRA="${EXTRA:-}"
# ⚑ DEFAULT = NO-SEG (2026-08-25 user decision): --t1_task recon + w_anat_roi=0
# in the config ⇒ the T1 branch is a PURE ENCODER trained jointly through the
# cross-attention K-path only — no stage-1, no seg supervision, and (because
# loss_seg/loss_contrast share the t1_seg guard) fully LABEL-FREE training.
# The CMF tissue-similarity bias is inert under recon (t1_seg_logits=None).
# T1_TASK=seg re-enables the "+seg" ABLATION arm (w_seg=1.0 fires again; name
# gets a _seg suffix so the two arms never clobber each other).
T1_TASK=${T1_TASK:-recon}
STAG=$([ "$T1_TASK" = seg ] && echo "_seg" || echo "")
# Multi-scale window cross-fusion (2026-08-26, docs/multiscale_window_design.md).
# WIN_LEVELS=0 (default) builds nothing -> bit-exactly the pre-2026-08-26 arch, so
# existing runs and ckpts are unaffected. 2 = equip 64x64 + 128x128, 1 = 128x128 only.
# WIN_K picks the attention KEY source:
#   t1  = anatomy-grouped cross-attention                   -> arm A1 (main)
#   asl = self-attention control, fine scales stay T1-free  -> arm A3
# A1 minus A3 isolates the FINE-SCALE keys only: both arms keep the coarse-scale T1
# guidance (CMF0/CMF1), so it is not the net effect of anatomy overall, and the two
# differ by the key projection, so it is not parameter-matched either. Q=ASL and
# V=ASL-unprojected either way, so the fused output stays a convex combination.
WIN_LEVELS=${WIN_LEVELS:-0}
WIN_K=${WIN_K:-t1}
# Free-form tag appended to the run name, for arms that differ only by EXTRA flags
# (e.g. NAME_SUFFIX=_agguniform EXTRA="--agg_tau_init 0 --freeze_agg_tau").
NAME_SUFFIX=${NAME_SUFFIX:-}
# T1-reconstruction auxiliary loss (arm B1). At 0 the T1 branch is a pure encoder whose only
# gradient arrives through the cross-attention keys -- nothing then constrains its features to
# represent anatomy, which is the inertness risk CLAUDE.md notes. Above 0 the runner restores
# the decoder head (+0.67M params) and the encoder gets a second, self-supervised gradient:
# the target is the input T1, so training stays label-free. The guidance path itself is
# unchanged -- K is still W_k applied to T1 features, V is still unprojected ASL -- so this
# alters what the T1 encoder learns, not how it can influence the output.
W_ANAT=${W_ANAT:-}
if [ -n "$W_ANAT" ]; then
  ANAT_FLAGS="--w_anat_roi $W_ANAT"
  ATAG="_t1dec$(echo "$W_ANAT" | tr '.' 'p')"   # 0.03 -> 0p03, reversible by submit_select.sh
else
  ANAT_FLAGS=""
  ATAG=""
fi
if [ "$WIN_LEVELS" -gt 0 ]; then
  WIN_FLAGS="--window_fusion_levels $WIN_LEVELS --window_k_source $WIN_K"
  WTAG="_win${WIN_LEVELS}${WIN_K}"
else
  WIN_FLAGS=""
  WTAG=""
fi
# best-ckpt gate: keep the runner default (best_min_step=-1 -> falls back to
# sure_anneal_start=200 from the config), same as the local probe.

echo "=== [v35_joint] seed=$SEED max_steps=$MAX_STEPS eval_every=$EVAL_EVERY t1_task=$T1_TASK win=${WIN_LEVELS}/${WIN_K} w_anat_roi=${W_ANAT:-config} (FRA + joint T1$STAG, no stage-1) ==="
yhrun torchrun --nnodes=1 --nproc_per_node=1 --master_port="$MASTER_PORT" $RUNNER \
  --config "$CONFIG" --exp "$EXP" --base_ch 32 --depth 4 \
  --use_t1_cross_fusion --t1_attn_max_tokens 1024 --t1_task $T1_TASK \
  --premask_asl_inputs \
  --bad_frame_p 0.3 --save_every "$SAVE_EVERY" --save_images --log_images 10 \
  --max_steps "$MAX_STEPS" --eval_every "$EVAL_EVERY" \
  --early_stop_patience 20 --early_stop_min_evals 60 \
  --best_criterion umse --save_per_metric_best \
  $WIN_FLAGS $ANAT_FLAGS \
  --seed "$SEED" --name run_v35_joint${STAG}${WTAG}${ATAG}${NAME_SUFFIX}_seed$SEED $EXTRA

echo "[v35_joint] done -> $EXP/logs/run_v35_joint${STAG}${WTAG}${ATAG}${NAME_SUFFIX}_seed$SEED"
