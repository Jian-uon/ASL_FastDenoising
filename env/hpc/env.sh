#!/bin/sh
# ===========================================================================
# 天河 (Tianhe) 共享环境 —— 被 env/hpc/slurm/*.sh 统一 `source`。
#
# 做的事:
#   1. 激活 conda 环境(asl-mamba)。
#   2. 加载 CUDA module(mamba .so 运行期要 libcudart 等),并把 torch 自带的
#      cuDNN/cuBLAS/cudart 放到 LD_LIBRARY_PATH 最前覆盖系统的 —— 否则系统 cuDNN
#      会盖掉 torch 的,报 CUDNN_STATUS_NOT_INITIALIZED。
#   3. 设离线变量 —— **计算节点无外网**,防 mamba-ssm / HuggingFace 联网卡住。
#   4. 导出公共路径变量(EXP / CONFIG / PYTHONPATH),供各 phase 脚本复用。
#
# 约定:调用方先 `cd "$REPO"` 再 `source env/hpc/env.sh`(相对路径才解析得到)。
# 任意变量都可用环境变量覆盖,例如:
#     EXP=/path/exp  CUDA_MODULE=cuda/12.4  yhbatch env/hpc/slurm/smoke.sh
# ===========================================================================

# ---- 1. 公共路径(可被环境变量覆盖)----
export REPO_ROOT="${REPO:-$PWD}"
export EXP="${EXP:-/fs1/home/duancaohui/jian/exp/ASL_FastDenoising}"  # 运行产物根 (logs/, tensorboard/, comparison_*) —— 与 ASL_dmvae 分开,两条线的 run 不会互相覆盖
export ENVNAME="${ENVNAME:-asl-mamba}"                       # conda 环境名
export CONFIG="${CONFIG:-env/hpc/configs/server_v35_joint.yml}"  # HPC config(自带数据 root_path) —— 本仓库只有 V35 线
export PYTHONPATH=".${PYTHONPATH:+:$PYTHONPATH}"             # 仓库根置最前(保留已有 PYTHONPATH),让 scripts/ 能 import 仓库内的包
export PYTHONUNBUFFERED=1                                    # 实时日志:torchrun 启动的 python 没有 -u,靠这个保证 tail -f 能即时看到输出

# torchrun 汇合端口 —— 按 job 唯一,避免多 job 同节点(fan-out 后常见)争抢默认 29500
# 报 EADDRINUSE。派生自 SLURM 作业号(退化到 shell PID),范围 20000–59999;各 phase
# 脚本的 torchrun 都带 --master_port="$MASTER_PORT"。可用环境变量覆盖。
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + ( ${SLURM_JOB_ID:-${SLURM_JOBID:-$$}} % 40000 ) ))}"

# ---- 2. 离线变量(计算节点无外网,防 mamba-ssm / HF 联网卡住)----
# 缓存放到 $EXP 下(仓库外),不污染 git 工作区。
export TORCH_HOME="${TORCH_HOME:-$EXP/cache/torch}"
export HF_HOME="${HF_HOME:-$EXP/cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"  # 多 OpenMP 兜底

# ---- 3. 目录准备 ----
mkdir -p env/hpc/slurm/logs "$EXP/logs" "$TORCH_HOME/hub/checkpoints" "$HF_HOME"

# ---- 4. conda 激活(天河用 source activate;不可用先 `module load anaconda3`)----
source activate "$ENVNAME" 2>/dev/null || conda activate "$ENVNAME"
which python

# ---- 5. CUDA module + 让 torch 自带 cuDNN 优先 ----
# mamba 的 .so 运行期需要 CUDA 库(libcudart 等),用 CUDA module 提供。
module add "${CUDA_MODULE:-CUDA/12.1}" 2>/dev/null \
  || module load "${CUDA_MODULE:-CUDA/12.1}" 2>/dev/null \
  || echo "[env] WARN: 'module add ${CUDA_MODULE:-CUDA/12.1}' 失败(用 'module avail cuda' 查名字)"
# 关键:把 torch wheel 自带的 nvidia/*/lib(cudnn/cublas/cudart...)放到 LD_LIBRARY_PATH 最前,
# 覆盖系统 CUDA module 的同名库 —— 否则系统 cuDNN 会盖掉 torch 的,报
# CUDNN_STATUS_NOT_INITIALIZED(就是之前那个错)。torch 的 cudart 同样覆盖,与 mamba 一致。
_TORCH_NV_LIBS=$(python -c "import sysconfig,glob,os; sp=sysconfig.get_paths()['purelib']; print(':'.join(sorted(glob.glob(os.path.join(sp,'nvidia','*','lib')))))" 2>/dev/null || echo "")
[ -n "$_TORCH_NV_LIBS" ] && export LD_LIBRARY_PATH="$_TORCH_NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ---- 6. 信息打印(每个 job 日志开头都能看到环境是否正确)----
echo "[env] node=$(hostname)  env=$ENVNAME  python=$(command -v python)"
echo "[env] cuda=$(command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -oE 'release [0-9.]+' | head -1 || echo none)  REPO=$REPO_ROOT  EXP=$EXP"
echo "[env] CONFIG=$CONFIG  HF_HUB_OFFLINE=$HF_HUB_OFFLINE  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
