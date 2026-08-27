#!/bin/sh
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8 -J v35_select
#SBATCH -o env/hpc/slurm/logs/select-%j.out
# NO `-e`: stderr merged into the .out on purpose (same reason as submit_v35_joint.sh).
#SBATCH --time=12:00:00
# ===========================================================================
# Post-hoc operating-point selection on validation uMSE, for every run the
# Medical Physics paper actually uses.
#
# WHY THIS EXISTS
#   The in-loop best is step-gated (`best_min_step` falls back to
#   sure_anneal_start=200), so an earlier global optimum can be missed, and the
#   two baselines are worse off than that: train_baseline.py:389 saves `best`
#   and early-stops on `psnr_ref`, which CLAUDE.md §4 forbids for selection.
#   Every method in Table 1 must therefore be re-selected here, under ONE rule:
#   global argmin pooled validation uMSE over the periodic snapshots.
#
#   Two selectors, because eval_select_ckpt.py rebuilds the MAIN runner and
#   cannot load a train_baseline.py checkpoint:
#     main runs      -> scripts/eval_select_ckpt.py     --metric umse
#     PlainUNet/Swin -> scripts/select_cnr_plainunet.py --metric umse
#   Both write best_umse_posthoc.pth next to the run's other checkpoints.
#
# Usage (submit from the REPO ROOT, after `git pull`):
#   yhbatch env/hpc/slurm/submit_select.sh
#
# Knobs:
#   MAIN_RUNS="run_v35_joint_win2t1_seed42 ..."   the proposed model, one per seed
#   BASE_RUNS="run_base_plainunet_n2n_seed42 ..." train_baseline.py runs
#   (set either to "" to skip that half)
# ===========================================================================
set -eu

REPO=${REPO:-/fs1/home/duancaohui/jian/projects/ASL_FastDenoising}
cd "$REPO"
CONFIG=${CONFIG:-env/hpc/configs/server_v35_joint.yml}
source env/hpc/env.sh

MAIN_RUNS=${MAIN_RUNS:-"run_v35_joint_win2t1_seed42 run_v35_joint_win2t1_seed1 run_v35_joint_win2t1_seed2"}
BASE_RUNS=${BASE_RUNS:-"run_base_plainunet_n2n_seed42 run_base_swinir_n2n_seed42"}

die() { echo "[select] FAILED at: $*" >&2; exit 1; }

for r in $MAIN_RUNS; do
  D="$EXP/logs/$r"
  [ -d "$D/checkpoints" ] || die "$r has no checkpoints/ (did the job die before the first save?)"
  set -- "$D"/checkpoints/step*.pth
  [ -e "$1" ] || die "$r has no step*.pth snapshots"
  [ -e "$D/checkpoints/best_umse.pth" ] && set -- "$@" "$D/checkpoints/best_umse.pth"
  SEED=$(echo "$r" | sed 's/.*_seed//')
  echo "=== [select] $r  ($# candidate checkpoints, seed=$SEED)"

  RA="--config $CONFIG --exp $EXP --name v35_select_tmp --base_ch 32 --depth 4 \
--use_t1_cross_fusion --t1_attn_max_tokens 1024 --t1_task recon --premask_asl_inputs \
--window_fusion_levels 2 --window_k_source t1 --best_criterion umse --seed $SEED"

  yhrun python scripts/eval_select_ckpt.py \
    --ckpts "$@" --runner_args "$RA" --metric umse \
    --output "$D/select_umse.json" \
    --save_selected "$D/checkpoints/best_umse_posthoc.pth" \
    || die "eval_select_ckpt.py on $r"
done

for r in $BASE_RUNS; do
  D="$EXP/logs/$r"
  [ -d "$D/checkpoints" ] || die "$r has no checkpoints/"
  set -- "$D"/checkpoints/step*.pth
  [ -e "$1" ] || die "$r has no step*.pth snapshots"
  [ -e "$D/checkpoints/best.pth" ] && set -- "$@" "$D/checkpoints/best.pth"
  echo "=== [select] $r  ($# candidate checkpoints, baseline)"

  yhrun python scripts/select_cnr_plainunet.py \
    --ckpts "$@" --config "$CONFIG" --split val --base_ch 32 --depth 4 --metric umse \
    --output "$D/select_umse.json" \
    --save_selected "$D/checkpoints/best_umse_posthoc.pth" \
    || die "select_cnr_plainunet.py on $r"
done

echo "[select] done. Selected operating points:"
for r in $MAIN_RUNS $BASE_RUNS; do
  printf '  %-40s %s\n' "$r" "$(python - "$EXP/logs/$r/select_umse.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    # eval_select_ckpt writes min_ckpt; select_cnr_plainunet writes selected.
    print("%s  val uMSE=%.5f" % (d.get("min_ckpt") or d["selected"], d["selected_umse"]))
except Exception as e:
    print("(could not read: %s)" % e)
PY
)"
done
