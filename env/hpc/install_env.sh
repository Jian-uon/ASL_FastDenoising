#!/bin/sh
# ===========================================================================
# CIG-Net v2 — Tianhe (天河) one-shot environment install.  SAFE TO RE-RUN.
#
# Builds the `asl-mamba` conda env end-to-end and self-heals a dirty env:
#   1. install the pinned CUDA-12.1 PyTorch stack + project deps,
#   2. PURGE conflicting CUDA-13 nvidia leftovers (the cuDNN-shadow bug),
#   3. verify cuDNN actually initialises on the GPU (fail loudly if not),
#   4. build the selective-scan CUDA kernels (causal-conv1d + mamba-ssm).
# Without those kernels the MoSSM encoder silently falls back to a 50-100x
# slower pure-PyTorch scan, so the build is REQUIRED for usable speed.
#
# RUN ON A GPU NODE (torch.cuda must work for the cuDNN check + arch detect):
#     yhrun -p gpu -N 1 --gpus-per-node=1 --cpus-per-gpu=8 --pty /bin/bash
#     sh env/hpc/install_env.sh
#
# Override via env vars: ENVNAME PYVER TORCH_INDEX PIP_INDEX CUDA_MODULE GCC_MODULE.
# For a different CUDA, change the torch pins in requirements.txt + TORCH_INDEX +
# CUDA_MODULE (keep their majors equal).
# ===========================================================================
set -eu

ENVNAME=${ENVNAME:-asl-mamba}
PYVER=${PYVER:-3.11}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}   # CUDA 12.1 wheels
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}     # full mirror for non-torch deps

# --- 0. conda on PATH ------------------------------------------------------
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not on PATH — 'module load anaconda3' first."; exit 1; }

# --- 1. create + activate the env (idempotent) -----------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENVNAME"; then
  echo "conda env '$ENVNAME' already exists — reusing."
else
  conda create -n "$ENVNAME" python="$PYVER" -y
fi
# shellcheck disable=SC1091
source activate "$ENVNAME"
which python && python --version

# --- 2. purge conflicting CUDA-13 nvidia leftovers -------------------------
# A prior torch upgrade (e.g. to a cu13 build) leaves CUDA-13 nvidia-* packages
# behind — pip never removes them on downgrade. They make torch load the WRONG cuDNN
# (cudnn 9.20 from nvidia-cudnn-cu13 instead of the bundled cu12 9.1.0.70) ->
# CUDNN_STATUS_NOT_INITIALIZED. Drop every nvidia-* wheel that is not a -cu12 build.
PURGED=0
LEFTOVERS=$(pip list --format=freeze 2>/dev/null | grep -i '^nvidia-' | cut -d= -f1 | grep -v -- '-cu12$' || true)
if [ -n "$LEFTOVERS" ]; then
  echo "removing non-cu12 nvidia leftovers:"; echo "$LEFTOVERS"
  echo "$LEFTOVERS" | xargs -r pip uninstall -y
  PURGED=1
fi

# --- 3. PyTorch + project deps ---------------------------------------------
# The cu12 and cu13 nvidia wheels SHARE file paths (e.g. nvidia/cudnn/lib/
# libcudnn.so.9), so the purge above deletes .so files the kept -cu12 packages still
# need -> "ImportError: libcudnn.so.9: cannot open shared object file". Plain
# `pip install` sees torch "satisfied" (metadata intact) and will NOT restore them,
# so force-reinstall the torch CUDA stack whenever we purged or torch won't import.
# (force-reinstall pulls torch's nvidia-*-cu12 deps too, rewriting the deleted .so.)
TORCH_PINS=$(grep -E '^(torch|torchvision)==' requirements.txt | tr '\n' ' ')
if [ "$PURGED" = 1 ] || ! python -c "import torch" 2>/dev/null; then
  echo "installing/repairing torch CUDA stack (force-reinstall): $TORCH_PINS"
  pip install --force-reinstall $TORCH_PINS --index-url "$TORCH_INDEX" --extra-index-url "$PIP_INDEX"
fi
# remaining (non-torch) deps — torch is pinned+present so it's a no-op for it; the
# mirror has monai/nibabel/... that the PyTorch wheel index lacks.
echo "installing project deps from $PIP_INDEX ..."
pip install -r requirements.txt -i "$PIP_INDEX" --extra-index-url "$TORCH_INDEX"
python -c "import torch, torchvision; v=torch.__version__; print('torch', v, '| torchvision', torchvision.__version__); assert '+cu' in v, f'torch lost its CUDA build tag ({v})'"

# --- 4. verify cuDNN actually initialises (fail loudly if broken) ----------
# Prepend torch's bundled nvidia/*/lib so its cuDNN wins even if a system CUDA module
# is loaded in this shell (same trick env/hpc/env.sh uses at job runtime).
NVL=$(python -c "import sysconfig,glob,os; print(':'.join(sorted(glob.glob(os.path.join(sysconfig.get_paths()['purelib'],'nvidia','*','lib')))))" 2>/dev/null || echo "")
[ -n "$NVL" ] && export LD_LIBRARY_PATH="$NVL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  python - <<'PY'
import torch, torch.nn as nn
print("cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version(), "| gpu", torch.cuda.get_device_name(0))
y = nn.Conv2d(3, 8, 3).cuda()(torch.randn(1, 3, 16, 16, device="cuda"))
print("cuDNN conv OK:", tuple(y.shape))
PY
else
  echo "WARNING: no GPU on this node — skipped the cuDNN/conv check (run install on a gpu node)."
fi

# --- 5. selective-scan CUDA kernels (causal-conv1d + mamba-ssm) ------------
# These compile CUDA extensions: need nvcc (CUDA module, major == torch's) + a host
# C++ compiler (GCC module; Tianhe's system gcc is often too old). Override names
# with CUDA_MODULE / GCC_MODULE  (`module avail cuda` / `module avail GCC`).
module add "${GCC_MODULE:-GCC/10.2.0}"  2>/dev/null || module load "${GCC_MODULE:-GCC/10.2.0}"  2>/dev/null || echo "[install] WARN: GCC module '${GCC_MODULE:-GCC/10.2.0}' not loaded (module avail GCC)."
module add "${CUDA_MODULE:-CUDA/12.1}" 2>/dev/null || module load "${CUDA_MODULE:-CUDA/12.1}" 2>/dev/null || echo "[install] WARN: CUDA module '${CUDA_MODULE:-CUDA/12.1}' not loaded (module avail cuda)."

CC=$(python -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "")
if [ -n "$CC" ]; then export TORCH_CUDA_ARCH_LIST="$CC"; echo "TORCH_CUDA_ARCH_LIST=$CC"; else
  echo "WARNING: GPU arch not auto-detected; set e.g. export TORCH_CUDA_ARCH_LIST=9.0 and re-run from here."
fi

TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found — the mamba build WILL fail (the confusing 'bare_metal_version' NameError)."
  echo "       torch CUDA is $TORCH_CUDA;  module avail cuda ; module add CUDA/12.1"
  echo "       (or: CUDA_MODULE=cuda/12.4 sh env/hpc/install_env.sh)"
  exit 1
fi
echo "nvcc:"; nvcc --version | tail -2

pip install ninja
# Pin torch so the mamba/causal-conv1d install (they declare `torch` as a dep) can't
# re-resolve + upgrade it off cu121 (which would ABI-break the just-built
# selective_scan_cuda.so -> "undefined symbol ..c10_cuda_check_implementation").
# --no-cache-dir forces a fresh compile against the current torch.
CONSTR=$(mktemp); pip freeze | grep -E '^(torch|torchvision|triton)==' > "$CONSTR" || true
echo "pinning during mamba build:"; cat "$CONSTR"
pip install --no-cache-dir --no-build-isolation -c "$CONSTR" "causal-conv1d>=1.4" "mamba-ssm>=2.2"
rm -f "$CONSTR"

# --- 6. final sanity -------------------------------------------------------
python -c "import torch; v=torch.__version__; print('torch', v); assert '+cu' in v, 'torch lost its CUDA build tag'"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba_ssm CUDA kernel OK')"
echo
echo "DONE. Env '$ENVNAME' ready."
echo "Next: check the dataset path in env/hpc/configs/server_v37.yml (root_path),"
echo "      then run the smoke test:  yhbatch env/hpc/slurm/smoke.sh"
