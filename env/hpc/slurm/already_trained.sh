#!/bin/sh
# ===========================================================================
# already_trained <run-name> [min-epochs]
#
# Exit 0 if $EXP/logs/<run-name> already holds a snapshot at or past <min-epochs>,
# non-zero otherwise. Sourced by the job scripts and by the matrix submitter so a
# run that is already finished is not repeated -- 15 to 28 GPU-hours each, and the
# matrix queues fifteen of them.
#
# "Epochs" here are what the runner calls steps: one pass over the 627 batches, saved
# as checkpoints/step%06d.pth every SAVE_EVERY. The highest such number is how far the
# run actually got, which is not the same as how far it was asked to go -- early
# stopping ends most runs well before MAX_STEPS.
#
# A partial run is NOT skipped and NOT resumed automatically: relaunching writes into
# the same directory from scratch, and silently resuming something a previous job left
# in an unknown state is worse than saying so. The caller is told the highest step
# found, so `EXTRA="--resume"` is an informed choice.
# ===========================================================================

# highest step number among a run's periodic snapshots; 0 if there are none
max_step() {
  _d="$EXP/logs/$1/checkpoints"
  [ -d "$_d" ] || { echo 0; return; }
  ls "$_d"/step*.pth 2>/dev/null |
    sed -n 's/.*step0*\([0-9][0-9]*\)\.pth$/\1/p' |
    sort -n | tail -1 | grep . || echo 0
}

already_trained() {
  _name=$1
  _min=${2:-200}
  _got=$(max_step "$_name")
  if [ "$_got" -ge "$_min" ]; then
    echo "[skip] $_name: already trained to step $_got (>= $_min). FORCE=1 to retrain."
    return 0
  fi
  if [ "$_got" -gt 0 ]; then
    echo "[note] $_name: partial run, highest step $_got (< $_min) -- training from scratch."
    echo "       Use EXTRA=\"--resume\" instead if that run should be continued."
  fi
  return 1
}
