#!/bin/sh
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8 -J v35_all
#SBATCH -o env/hpc/slurm/logs/all-%j.out
# NO `-e`: stderr merged into the .out on purpose (same reason as submit_v35_joint.sh).
#SBATCH --time=36:00:00
# ===========================================================================
# Everything after training, in order, in ONE job:
#
#   1. submit_select.sh   post-hoc operating point on validation uMSE, all 5 runs
#   2. submit_eval.sh     sweep -> rCBF -> montage -> figures
#
# One job rather than two chained ones, because step 2 reads the
# best_umse_posthoc.pth files step 1 writes: a single allocation makes that
# ordering a fact instead of a dependency that has to be spelled correctly.
# Both sub-scripts run `set -eu` and name the step they died at, so a failure
# stops the sequence here rather than feeding half-selected checkpoints into
# the comparison.
#
# Usage (from the REPO ROOT, after `git pull`):
#   yhbatch env/hpc/slurm/run_all.sh
#   tail -f env/hpc/slurm/logs/all-<jobid>.out
#
# Knobs pass straight through to the two sub-scripts, e.g.
#   K_MONTAGE="2 4" yhbatch env/hpc/slurm/run_all.sh
#   SKIP_SELECT=1 yhbatch env/hpc/slurm/run_all.sh    # selection already done
# ===========================================================================
set -eu

REPO=${REPO:-/fs1/home/duancaohui/jian/projects/ASL_FastDenoising}
cd "$REPO"

echo "############################################################"
echo "# 1/2  checkpoint selection            $(date +'%F %T')"
echo "############################################################"
if [ "${SKIP_SELECT:-0}" = "1" ]; then
  echo "[run_all] SKIP_SELECT=1 -> reusing the existing best_umse_posthoc.pth files."
else
  sh env/hpc/slurm/submit_select.sh
fi

echo
echo "############################################################"
echo "# 2/2  evaluation and figures          $(date +'%F %T')"
echo "############################################################"
sh env/hpc/slurm/submit_eval.sh

echo
echo "[run_all] finished $(date +'%F %T')"
