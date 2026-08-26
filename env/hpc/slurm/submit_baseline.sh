#!/bin/sh
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8 -J v35_base
#SBATCH -o env/hpc/slurm/logs/base-%j.out
#SBATCH -e env/hpc/slurm/logs/base-%j.err
#SBATCH --time=72:00:00
# Tianhe job — the external architecture baselines for the Medical Physics paper.
# Same Noise2Noise regime and the same data/config as the main arms, so the
# comparison is on equal footing; only the backbone differs.
#
#   ARCH=plainunet  PlainUNet2D — doubles as the no-T1 lower bound
#   ARCH=swinir     SwinIR-light Transformer denoiser (recent-architecture reference)
#
# Usage — SUBMIT FROM THE REPO ROOT (see submit_v35_joint.sh for why that matters):
#   ARCH=plainunet yhbatch env/hpc/slurm/submit_baseline.sh
#   ARCH=swinir    yhbatch env/hpc/slurm/submit_baseline.sh
#
# Knobs: SEED | MAX_STEPS | SAVE_EVERY | EXTRA="--resume"
set -eo pipefail

REPO=${REPO:-/fs1/home/duancaohui/jian/projects/ASL_FastDenoising}
cd "$REPO"
CONFIG=${CONFIG:-env/hpc/configs/server_v35_joint.yml}
source env/hpc/env.sh
nvidia-smi -L || true

RUNNER=runners/train_baseline.py
ARCH=${ARCH:-plainunet}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-500}
SAVE_EVERY=${SAVE_EVERY:-50}
EXTRA="${EXTRA:-}"

echo "=== [baseline] arch=$ARCH mode=n2n seed=$SEED max_steps=$MAX_STEPS ==="
yhrun torchrun --nnodes=1 --nproc_per_node=1 --master_port="$MASTER_PORT" $RUNNER \
  --mode n2n --arch "$ARCH" \
  --config "$CONFIG" --exp "$EXP" --base_ch 32 --depth 4 \
  --save_every "$SAVE_EVERY" --save_images --log_images 10 \
  --max_steps "$MAX_STEPS" \
  --early_stop_patience 20 --early_stop_min_evals 60 \
  --seed "$SEED" --name run_base_${ARCH}_n2n_seed$SEED $EXTRA

echo "[baseline] done -> $EXP/logs/run_base_${ARCH}_n2n_seed$SEED"
