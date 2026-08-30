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
#   B1_SEEDS="42 1 2"   seeds for the T1-decoder arm; OFF by default (rejected 2026-08-31,
#                       see CLAUDE.md "Closed questions") -- set it to re-run the arm
#   W_ANAT="0.03"       its T1-reconstruction weight; list several to sweep at seed 42
#   MAX_STEPS=500
#   MIN_EPOCHS=200      a run already trained this far is skipped, not requeued
#   FORCE=1             queue everything regardless
#   GO=1                submit for real (default is a dry run)
#
# The matrix — 12 runs at the default seeds (B1 off), 10 with A0_SEEDS="42":
#
#   A1  window fusion, T1 keys      x3   the method
#   A3  window fusion, ASL keys     x3   A1 - A3 isolates the FINE-SCALE keys: both
#                                        keep coarse-scale T1 guidance, and they
#                                        differ by the key projection, so this is
#                                        neither "all of anatomy" nor parameter-matched
#   B1  A1 + T1 decoder head        x0   OFF: tried at w=0.03 and came out worse than
#                                        encoder-only, so the arm is closed. B1_SEEDS
#                                        re-enables it if a reviewer asks for the evidence.
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
B1_SEEDS=${B1_SEEDS:-""}      # closed arm; opt in explicitly
W_ANAT=${W_ANAT:-"0.03"}
MAX_STEPS=${MAX_STEPS:-500}
GO=${GO:-0}

if [ ! -f env/hpc/slurm/submit_v35_joint.sh ]; then
  echo "ERROR: run this from the repo root (env/hpc/slurm/submit_v35_joint.sh not found here)."
  exit 1
fi
mkdir -p env/hpc/slurm/logs      # Slurm opens the -o file before the job script runs

. env/hpc/slurm/already_trained.sh
n=0
skipped=0
sub() {  # sub "<label>" "<full shell command>" [<run name to check>]
  if [ -n "${3:-}" ] && [ "${FORCE:-0}" != "1" ] \
     && already_trained "$3" "${MIN_EPOCHS:-200}" >/dev/null; then
    skipped=$((skipped + 1))
    printf '   -- %s  [already trained]\n' "$1"
    return
  fi
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
      "SEED=$s MAX_STEPS=$MAX_STEPS WIN_LEVELS=2 WIN_K=t1 yhbatch $JOINT" \
      "run_v35_joint_win2t1_seed$s"
done
for s in $SEEDS; do
  sub "A3  window + ASL keys     seed=$s" \
      "SEED=$s MAX_STEPS=$MAX_STEPS WIN_LEVELS=2 WIN_K=asl yhbatch $JOINT" \
      "run_v35_joint_win2asl_seed$s"
done
for s in $A0_SEEDS; do
  sub "A0  no window fusion      seed=$s" \
      "SEED=$s MAX_STEPS=$MAX_STEPS WIN_LEVELS=0 yhbatch $JOINT" \
      "run_v35_joint_seed$s"
done
# B1: the Figure 1 architecture. The first weight in W_ANAT runs at every B1 seed; any
# further weights run at seed 42 only, which is enough to see whether the arm is sensitive
# to the weight without paying for a full seed set per value.
W1=$(echo $W_ANAT | awk '{print $1}')
for s in $B1_SEEDS; do
  sub "B1  A1 + T1 decoder w=$W1  seed=$s" \
      "SEED=$s MAX_STEPS=$MAX_STEPS WIN_LEVELS=2 WIN_K=t1 W_ANAT=$W1 yhbatch $JOINT" \
      "run_v35_joint_win2t1_t1dec$(echo "$W1" | tr '.' 'p')_seed$s"
done
for w in $W_ANAT; do
  [ "$w" = "$W1" ] && continue
  sub "B1  A1 + T1 decoder w=$w  seed=42" \
      "SEED=42 MAX_STEPS=$MAX_STEPS WIN_LEVELS=2 WIN_K=t1 W_ANAT=$w yhbatch $JOINT"
done
sub "AGG uniform (tau frozen 0) seed=42" \
    "SEED=42 MAX_STEPS=$MAX_STEPS WIN_LEVELS=2 WIN_K=t1 NAME_SUFFIX=_agguniform \
EXTRA='--agg_tau_init 0 --freeze_agg_tau' yhbatch $JOINT"
sub "PlainUNet-N2N             seed=42" \
    "SEED=42 MAX_STEPS=$MAX_STEPS ARCH=plainunet yhbatch $BASE" \
    "run_base_plainunet_n2n_seed42"
sub "SwinIR-N2N                seed=42" \
    "SEED=42 MAX_STEPS=$MAX_STEPS ARCH=swinir yhbatch $BASE" \
    "run_base_swinir_n2n_seed42"

echo
if [ "$GO" = "1" ]; then
  echo "submitted $n jobs ($skipped already trained, skipped)."
  echo "  watch:    yhq -u \$USER"
  echo "  job logs: env/hpc/slurm/logs/{v35j,base}-<jobid>.out"
  echo "  outputs:  \$EXP/logs/<run name>/"
else
  echo "dry run — nothing submitted ($n jobs listed above, $skipped already trained)."
  echo "re-run with:  GO=1 sh env/hpc/slurm/submit_medphys_matrix.sh"
fi
