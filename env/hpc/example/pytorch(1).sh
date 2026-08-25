#!/bin/sh
#SBATCH -p gpu5 -N 1 --gpus-per-node=1 --cpus-per-gpu=8 -J jobname

source activate envname
which python

nvidia-smi dmon -d 1 -o DT> nvi1-$SLURM_JOB_ID.log &
nvidia-smi -l 1 > nvi2-$SLURM_JOB_ID.log &
free -gs 1 > free-$SLURM_JOB_ID.log &

#yhrun python -u train.py
yhrun torchrun --nnodes=1 --nproc_per_node=1  train.py
