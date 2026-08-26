#!/bin/sh
# ===========================================================================
# Submit the ENTIRE Medical Physics training matrix in one go.
#
# This is a SUBMITTER, not a job: run it on the LOGIN NODE, from the REPO ROOT.
# It queues every run as its own Slurm job, so they start as nodes free up and the
# whole matrix finishes in roughly one run's wall-clock times
# ceil(n_runs / concurrent GPUs).
#
#   cd /fs1/home/duancaohui/jian/projects/ASL_FastDenoising
#   sh env/hpc/sync_repo.sh
#   sh env/hpc/slurm/submit_medphys_matrix.sh          # dry run: prints, submits nothing
#   GO=1 sh env/hpc/slurm/submit_medphys_matrix.sh     # actually submit
#
# Knobs:
#   SEEDS="42 1 2"      seeds for A1 and A3 (the paired claim; do not cut below 3)
#   A0_SEEDS="42 1 2"   seeds for A0        (set to "42" if you are short on time)
#   MAX_STEPS=500
#   GO=1                submit for real (default is a dry run)
#
# The matrix — 12 runs at the default seeds, 10 with A0_SEEDS="42":
#
#   A1  window fusion, T1 keys      x3   the method
#   A3  window fusion, ASL keys     x3   A1 - A3 = the net effect of anatomical
#                                        guidance: same module, same parameter
#                                        count, one flag apart
#   A0  no window fusion            x3   does the module help at all
#   AGG uniform aggregator          x1   does inverse-variance frame weighting help
#                                        (tau frozen at 0 => exactly 1/N per frame)
#   PlainUNet-N2N                   x1   external baseline + no-T1 lower bound
#   SwinIR-N2N                      x1   recent-architecture reference
#
# Everything else in the paper is EVALUATION ONLY and needs no training run:
# temporal averaging, BM3D/AONLM, the n-frames sweep, mismatched-T1 leakage, the
# E0.3 lesion-retention test, rCBF agreement (ICC + Bland-Altman), the complexity
# numbers, and the per-subject paired statistics.
# ===========================================================================
set -eu

SEEDS=${SEEDS:-"42 1 2"}
A0_SEEDS=${A0_SEEDS:-$SEEDS}
MAX_STEPS=${MAX_STEPS:-500}
GO=${GO:-0}

if [ ! -f env/hpc/slurm/submit_v35_joint.sh ]; then
  echo "ERROR: run this from the repo root (env/hpc/slurm/submit_v35_joint.sh not found here)."
  exit 1
fi
mkdir -p env/hpc/slurm/logs      # Slurm opens the -o file before the job script runs

n=0
sub() {  # sub "<label>" "<full shell command>"
  n=$((n + 1))
  printf '  %2d. %s\n' "$n" "$1"
  if [ "$GO" = "1" ]; then
    eval "$2"
  else
    printf '      %s\n' "$2"
  fi
}

JOINT=env/hpc/slurm/submit_v35_joint.sh
BASE=env/hpc/slurm/submit_baseline.sh

echo "Medical Physics training matrix  (GO=$GO, MAX_STEPS=$MAX_STEPS)"
echo "  A1/A3 seeds: $SEEDS      A0 seeds: $A0_SEEDS"
echo

for s in $SEEDS; do
  sub "A1  window + T1 keys      seed=$s" \
      "SEED=$s MAX_STEPS=$MAX_STEPS WIN_LEVELS=2 WIN_K=t1 yhbatch $JOINT"
done
for s in $SEEDS; do
  sub "A3  window + ASL keys     seed=$s" \
      "SEED=$s MAX_STEPS=$MAX_STEPS WIN_LEVELS=2 WIN_K=asl yhbatch $JOINT"
done
for s in $A0_SEEDS; do
  sub "A0  no window fusion      seed=$s" \
      "SEED=$s MAX_STEPS=$MAX_STEPS WIN_LEVELS=0 yhbatch $JOINT"
done
sub "AGG uniform (tau frozen 0) seed=42" \
    "SEED=42 MAX_STEPS=$MAX_STEPS WIN_LEVELS=2 WIN_K=t1 NAME_SUFFIX=_agguniform \
EXTRA='--agg_tau_init 0 --freeze_agg_tau' yhbatch $JOINT"
sub "PlainUNet-N2N             seed=42" \
    "SEED=42 MAX_STEPS=$MAX_STEPS ARCH=plainunet yhbatch $BASE"
sub "SwinIR-N2N                seed=42" \
    "SEED=42 MAX_STEPS=$MAX_STEPS ARCH=swinir yhbatch $BASE"

echo
if [ "$GO" = "1" ]; then
  echo "submitted $n jobs."
  echo "  watch:    yhq -u \$USER"
  echo "  job logs: env/hpc/slurm/logs/{v35j,base}-<jobid>.out"
  echo "  outputs:  \$EXP/logs/<run name>/"
else
  echo "dry run — nothing submitted ($n jobs listed above)."
  echo "re-run with:  GO=1 sh env/hpc/slurm/submit_medphys_matrix.sh"
fi
