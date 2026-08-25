#!/usr/bin/env bash
# v35: v34 backbone (multi-scale tissue cross-attn) + loss redesign.
#
# - Same architecture as v34 (--use_t1_cross_fusion).
# - Reuses stage1_t1_padcrop best.pth (non-wavelet, step=1000) as frozen T1.
# - Loss changes (in v35.yml):
#     w_bg removed (input masking in runner replaces it)
#     lambda_cos disabled
#     w_contrast 0.3 enabled
#     w_tv 0.5 enabled (with masked interior-only TV)
#     w_anat_roi 0 (t1_recon dead since v34)
# - Save every 50 steps.
set -u
LOG=C:/tmp/asl_exp/logs/auto_chain_v35.log
echo "[$(date)] chain start" | tee -a "$LOG"

STAGE1_DIR=C:/tmp/asl_exp/logs/stage1_t1_padcrop
if [ ! -f "$STAGE1_DIR/checkpoints/best.pth" ]; then
  echo "[$(date)] ERROR: $STAGE1_DIR/checkpoints/best.pth not found" | tee -a "$LOG"
  exit 1
fi

echo "[$(date)] launching v35 (multi-scale tissue cross-attn + redesigned loss)" | tee -a "$LOG"
KMP_DUPLICATE_LIB_OK=TRUE D:/softwares/anaconda/envs/monai/python.exe -u runners/asl_t1_guided_runner_dmvae_n2n.py \
  --config config/win_asl_2d_home_v35.yml --exp C:/tmp/asl_exp \
  --name run_full_v35e --base_ch 32 --depth 4 \
  --use_t1_cross_fusion --t1_attn_max_tokens 1024 \
  --bad_frame_p 0.3 --save_every 50 --save_images --log_images 10 \
  --early_stop_patience 20 --early_stop_min_evals 60 \
  --best_criterion composite_v2 \
  --init_t1_from $STAGE1_DIR/checkpoints/best.pth \
  --freeze_t1 2>&1 | tee C:/tmp/asl_exp/logs/run_full_v35e_launch.txt

echo "[$(date)] chain complete" | tee -a "$LOG"
