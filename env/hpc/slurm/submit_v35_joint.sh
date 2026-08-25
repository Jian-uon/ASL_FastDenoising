#!/bin/sh
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8 -J v35_joint
#SBATCH -o env/hpc/slurm/logs/v35j-%j.out
#SBATCH -e env/hpc/slurm/logs/v35j-%j.err
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
# Usage:
#   git pull                                  # config/scripts must be pushed first
#   yhbatch env/hpc/slurm/submit_v35_joint.sh
# Knobs: SEED=1 | MAX_STEPS=500 | EVAL_EVERY=5 | SAVE_EVERY=50 | EXTRA="--resume"
set -eo pipefail

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
# best-ckpt gate: keep the runner default (best_min_step=-1 -> falls back to
# sure_anneal_start=200 from the config), same as the local probe.

echo "=== [v35_joint] seed=$SEED max_steps=$MAX_STEPS eval_every=$EVAL_EVERY t1_task=$T1_TASK (FRA + joint T1$STAG, no stage-1) ==="
yhrun torchrun --nnodes=1 --nproc_per_node=1 --master_port="$MASTER_PORT" $RUNNER \
  --config "$CONFIG" --exp "$EXP" --base_ch 32 --depth 4 \
  --use_t1_cross_fusion --t1_attn_max_tokens 1024 --t1_task $T1_TASK \
  --premask_asl_inputs \
  --bad_frame_p 0.3 --save_every "$SAVE_EVERY" --save_images --log_images 10 \
  --max_steps "$MAX_STEPS" --eval_every "$EVAL_EVERY" \
  --early_stop_patience 20 --early_stop_min_evals 60 \
  --best_criterion umse --save_per_metric_best \
  --seed "$SEED" --name run_v35_joint${STAG}_seed$SEED $EXTRA

echo "[v35_joint] done -> $EXP/logs/run_v35_joint${STAG}_seed$SEED"
