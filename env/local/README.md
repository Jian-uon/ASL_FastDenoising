# `env/local/` — Local development (Windows + WSL)

Local box is for **editing code, smoke tests, and debugging** — not the
multi-seed training sweep (that runs on the HPC, see [`../hpc/`](../hpc/)).

```
local/
  configs/
    win_asl_2d_home.yml        # Windows dev / inference (eval_baselines, infer_pwi, test_mismatched_t1)
    win_asl_2d_home_v37.yml    # Windows dev (v37-era flags; distill/eval tooling defaults)
    wsl_asl_2d_home_v37.yml    # WSL training config (mirrors hpc/configs/server_v37.yml except root_path)
  chains/
    auto_chain_cadalr.sh       # CURRENT v2 stage-2 launch chain (HYBRID=1 NONV=1 = main)
    auto_chain_baselines.sh    # ASL-only + naive-T1-concat baselines (same recipe)
```

## Two local sub-environments

### Windows (native) — fast iteration, CPU/build checks
- Interpreter: the base Anaconda Python (CPU PyTorch). Use for import/build
  smoke tests and CPU forward/`infer_from_subset` sanity (e.g. content-guard /
  mismatched-T1 checks), **not** real training.
- **MoSSM CUDA kernels (`mamba-ssm`) do not run on Windows** — the code falls
  back to a pure-PyTorch scan that is 10–20× slower. Anything that exercises the
  MoSSM encoder for real belongs on WSL or the HPC.
- Configs: `configs/win_asl_2d_home*.yml`. Several tools default to these
  (`runners/infer_pwi.py`, `runners/eval_baselines.py`,
  `scripts/test_mismatched_t1.py`, `scripts/eval_iqa_metrics.py`, …).

### WSL — real local training with the CUDA scan
- Conda env `asl-mamba` with `mamba-ssm` + `causal-conv1d` built (see the HPC
  guide [§2](../hpc/README.md) for the build recipe; it is the same on WSL).
- Config: `configs/wsl_asl_2d_home_v37.yml` (edit `dataset.root_path` to your WSL
  data mount). This file mirrors `hpc/configs/server_v37.yml` — keep training
  hyper-parameters in sync between the two.

## Launching a local (WSL) run

The chains `cd` to the repo root themselves; run from anywhere, but the absolute
`cd` line at the top of each chain is hard-coded to this machine — **edit it** if
your repo lives elsewhere.

```bash
# smoke (5 steps):
NAME=run_cadalr_smoke  MAX_STEPS=5   HYBRID=1 NONV=1 bash env/local/chains/auto_chain_cadalr.sh
# full v2 main run:
NAME=run_v2            MAX_STEPS=300 HYBRID=1 NONV=1 bash env/local/chains/auto_chain_cadalr.sh
# baselines (ASL-only + naive concat):
bash env/local/chains/auto_chain_baselines.sh
```

Stage-2 needs a frozen stage-1 T1 checkpoint; the chain expects it at the path
referenced inside the script. If missing, run the stage-1 T1 pretraining first
(`runners/train_t1.py`, or HPC Phase 1).

> Long WSL jobs: hold them foreground via a background Bash invocation; a bare
> `nohup … & disown` inside a one-shot `wsl bash -c` gets torn down. Use the
> `--resume` retry loop for crash recovery. (See memory `reference_wsl_long_job_detach`.)
> Multiline `wsl bash -c '…'` corrupts `\r`-terminated vars — keep it single-line
> or pipe a `tr -d '\r'`-cleaned script. (memory `reference_wsl_crlf_gotcha`.)
