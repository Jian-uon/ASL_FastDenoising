#!/bin/sh
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8 -J v35_eval
#SBATCH -o env/hpc/slurm/logs/eval-%j.out
# NO `-e`: stderr merged into the .out on purpose (same reason as submit_v35_joint.sh).
#SBATCH --time=24:00:00
# ===========================================================================
# The whole Medical Physics evaluation, on the server, end to end.
#
# Everything runs headless: every plotting path in scripts/ sets MPLBACKEND=Agg,
# and render_frames_sweep_panel.py was written for the server to begin with. The
# dataset is on Tianhe too, so nothing needs to come back except the outputs.
#
# Four phases; PHASE picks one, or `all`:
#
#   sweep  GPU   every method x every k on the SAME materialised test split
#                -> comparison_long.csv    (per method x subject x k)  = Fig. 3
#                -> comparison_summary.csv (per method x k)            = Fig. 4
#                -> comparison_table.md                                = Table 1
#   cbf    GPU   rCBF ICC / Bland-Altman / voxel-wise r vs the 12-repetition map
#   panel  GPU   the qualitative montage at the operating point         = Fig. 2
#   plots  CPU   redraws every figure FROM THE CSVs — no GPU, no re-inference,
#                so re-running it after a caption or axis change costs seconds
#                (plot_degradation.py says so in its own docstring).
#
# Prerequisite: submit_select.sh has written best_umse_posthoc.pth for all five
# runs. This script refuses to start otherwise, because a table mixing post-hoc
# and psnr_ref-selected operating points is not a fair comparison.
#
# Usage (from the REPO ROOT, after `git pull`):
#   yhbatch env/hpc/slurm/submit_eval.sh
#   PHASE=plots sh env/hpc/slurm/submit_eval.sh     # CPU-only, fine on the login node
#
# Knobs: PHASE, OUT, SPLIT, KS, K_CBF, K_MONTAGE, N_SUBJ, UMSE_MAX_K
# ===========================================================================
set -eu

REPO=${REPO:-/fs1/home/duancaohui/jian/projects/ASL_FastDenoising}
cd "$REPO"
CONFIG=${CONFIG:-env/hpc/configs/server_v35_joint.yml}
source env/hpc/env.sh

PHASE=${PHASE:-all}
OUT=${OUT:-$EXP/medphys_eval}
SPLIT=${SPLIT:-test}
KS=${KS:-"2 3 4 5 6 7 8 9"}     # uMSE needs 3 held-out frames => k <= 9
K_MONTAGE=${K_MONTAGE:-"2 4 6 8 12"}   # the montage renders these k as columns, so the operating
                                    # point can be chosen afterwards without re-running this job
N_SUBJ=${N_SUBJ:-20}          # montage slices, spread evenly over the held-out split
UMSE_MAX_K=${UMSE_MAX_K:-5}   # uPSNR is plotted only this far; see the plots phase
K_CBF=${K_CBF:-"2 4 6 8 10"}  # rCBF needs no held-out frames, so it runs past the uMSE limit;
                              # 12 is the reference and is added by --ref_frames
DATA_ROOT=${DATA_ROOT:-/fs1/home/duancaohui/jian/data/7T_ASL_denoising}
mkdir -p "$OUT"

die() { echo "[eval] FAILED at: $*" >&2; exit 1; }
ck()  { p="$EXP/logs/$1/checkpoints/best_umse_posthoc.pth"
        [ -f "$p" ] || die "$1 has no best_umse_posthoc.pth — run submit_select.sh first"
        echo "$p"; }

C42=$(ck run_v35_joint_win2t1_seed42)
C1=$(ck  run_v35_joint_win2t1_seed1)
C2=$(ck  run_v35_joint_win2t1_seed2)
CPU=$(ck run_base_plainunet_n2n_seed42)
CSW=$(ck run_base_swinir_n2n_seed42)

ra() {  # ra <seed> [<levels>] [<key source>] [<extra flags>]
  # Must reproduce what the run was TRAINED with: the checkpoint is loaded into the
  # architecture these flags build, and a mismatch fails to load rather than silently
  # scoring the wrong model.
  _s=$1; _lv=${2:-2}; _ks=${3:-t1}; _ex=${4:-}
  echo "--config $CONFIG --exp $EXP --name v35_eval_tmp --base_ch 32 --depth 4 \
--use_t1_cross_fusion --t1_attn_max_tokens 1024 --t1_task recon --premask_asl_inputs \
--window_fusion_levels $_lv --window_k_source $_ks --best_criterion umse --seed $_s $_ex"
}

# Ablation arms, added when their post-hoc checkpoint exists. They finish at different
# times, and a sweep covering the arms that are ready beats one that refuses to start.
ARMS=""
add_arm() {   # add_arm <label> <run name> <runner args>
  _p="$EXP/logs/$2/checkpoints/best_umse_posthoc.pth"
  if [ -f "$_p" ]; then
    ARMS="$ARMS --extra_runner '$1::$_p::$3'"
    echo "[eval] + $1"
  else
    echo "[eval] - $1  (no best_umse_posthoc.pth yet, skipped)"
  fi
}

for s in 42 1 2; do
  add_arm "A3_aslkeys_seed$s" "run_v35_joint_win2asl_seed$s" "$(ra $s 2 asl)"
done
for s in 42 1 2; do
  add_arm "A0_nowindow_seed$s" "run_v35_joint_seed$s" "$(ra $s 0 t1)"
done
for s in 42 1 2; do
  add_arm "AGG_uniform_seed$s" "run_v35_joint_win2t1_agguniform_seed$s" \
          "$(ra $s 2 t1 '--agg_tau_init 0 --freeze_agg_tau')"
done

if [ "$PHASE" = all ] || [ "$PHASE" = sweep ]; then
  echo "=== [eval] sweep: split=$SPLIT k in {$KS}"
  # eval, not sh, because the accumulated arm specs carry spaces inside each argument
  eval yhrun python scripts/eval_comparison_table.py \
    --config "$CONFIG" --split "$SPLIT" --seed 42 --slice_context 0 \
    --n_frames $KS --out_dir "$OUT/sweep" \
    --ours "$C42" --ours_runner_args "'$(ra 42)'" --ours_label proposed_seed42 \
    --extra_runner "'proposed_seed1::$C1::$(ra 1)'" \
    --extra_runner "'proposed_seed2::$C2::$(ra 2)'" \
    $ARMS \
    --vanilla "$CPU" --swinir_n2n "$CSW" --include_naive \
    || die "eval_comparison_table.py (sweep)"
fi

if [ "$PHASE" = all ] || [ "$PHASE" = cbf ]; then
  echo "=== [eval] rCBF agreement vs the 12-repetition map"
  # eval_cbf.py enumerates the dataset itself and knows nothing about the split, so the
  # held-out subjects must be handed to it explicitly. Left alone it takes the first N
  # directories alphabetically, which on this dataset is ~83% training subjects.
  TEST_SUBS=$(python scripts/dump_split.py --config "$CONFIG" --split test --sep " ") \
    || die "dump_split.py (test subject list)"
  echo "[eval] rCBF on $(echo $TEST_SUBS | wc -w) held-out subjects"
  # PLD/LD match the protocol reported in the manuscript. They cancel in rCBF, but the
  # numbers must agree with what the paper says was acquired.
  yhrun python scripts/eval_cbf.py \
    --checkpoint "$C42" --config "$CONFIG" --data_root "$DATA_ROOT" \
    --runner_args "$(ra 42)" --n_frames $K_CBF --ref_frames 12 \
    --pld 2.0 --ld 1.8 --subjects $TEST_SUBS --max_subjects 999 \
    --save_maps --rcbf_cmap turbo --rcbf_vmax 2.0 --qc_dpi 300 \
    --out_dir "$OUT/cbf" \
    || die "eval_cbf.py"
fi

if [ "$PHASE" = all ] || [ "$PHASE" = panel ]; then
  echo "=== [eval] qualitative montage at k in {$K_MONTAGE}, $N_SUBJ subjects"
  yhrun python scripts/render_frames_sweep_panel.py \
    --config "$CONFIG" --split "$SPLIT" --seed 42 --slice_context 0 \
    --n_frames $K_MONTAGE --n_subjects "$N_SUBJ" --out_dir "$OUT/panel" \
    --ours "$C42" --ours_runner_args "$(ra 42)" --ours_label "proposed" \
    --vanilla "$CPU" --swinir_n2n "$CSW" --include_naive \
    || die "render_frames_sweep_panel.py"
fi

if [ "$PHASE" = all ] || [ "$PHASE" = plots ]; then
  echo "=== [eval] figures from the CSVs (no GPU)"
  [ -f "$OUT/sweep/comparison_summary.csv" ] || die "no comparison_summary.csv -- run PHASE=sweep first"
  # Collapse the three training runs of the proposed model into one method (band, not three
  # lines) and average the per-batch rows within subject, since slices of one subject are
  # not independent samples.
  python scripts/merge_seeds.py --dir "$OUT/sweep" || die "merge_seeds.py"
  # uMSE subtracts two nearly equal noise terms; once the held-out groups thin out it
  # collapses to zero, so its curve is cut at UMSE_MAX_K and the plot says so.
  python scripts/plot_degradation.py --dir "$OUT/sweep" --umse_max_k "$UMSE_MAX_K" \
    || die "plot_degradation.py"
  python scripts/plot_sweep_boxplots.py \
    --long_csv "$OUT/sweep/comparison_long_merged.csv" \
    --out_dir  "$OUT/sweep/figures" --umse_max_k "$UMSE_MAX_K" \
    || die "plot_sweep_boxplots.py"
  # Absolute uMSE varies far more between subjects than between methods, so boxes of
  # absolute values overlap even where the ranking is consistent. This plots the pairing.
  python scripts/plot_paired_diff.py \
    --long_csv "$OUT/sweep/comparison_long_merged.csv" \
    --out_dir  "$OUT/sweep/figures" --ref proposed --max_k "$UMSE_MAX_K" \
    --exclude naive_mean || die "plot_paired_diff.py"
fi

echo
echo "[eval] done. outputs under $OUT"
echo "  Table 1   $OUT/sweep/comparison_table.md"
echo "  Fig. 2    $OUT/panel/"
echo "  Fig. 3    $OUT/sweep/figures/  (boxplots)"
echo "  Fig. 4    $OUT/sweep/figures/  (degradation_*.png)"
echo "  rCBF      $OUT/cbf/"
[ -f "$OUT/sweep/comparison_table.md" ] && cat "$OUT/sweep/comparison_table.md"
