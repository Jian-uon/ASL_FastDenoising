# `env/` — Environment routing hub

This project is **developed locally** (Windows / WSL) and **trained on the HPC**
(SLURM). Anything that differs per environment — configs, launch scripts, setup
docs — lives here, grouped by environment. Shared code (`models/`, `losses/`,
`runners/`, `dataio/`, `utils/`) and shared tools (`scripts/*.py`,
`scripts/run_comparison_v3.sh`, eval/SWA helpers) stay at the repo root and are
called identically from both environments.

## Which environment am I in? → which folder

| I want to… | Environment | Go to | Config | Launcher |
|---|---|---|---|---|
| Edit code, quick CPU smoke, debug on Windows | **local / Windows** | [`local/`](local/) | `local/configs/win_asl_2d_home*.yml` | run `runners/*.py` directly |
| Run a real (slow, pure-PyTorch) train on WSL, or develop the CUDA MoSSM path | **local / WSL** | [`local/`](local/) | `local/configs/wsl_asl_2d_home_v37.yml` | `local/chains/auto_chain_cadalr.sh` |
| Train the multi-seed sweep on the cluster | **HPC / SLURM** | [`hpc/`](hpc/) | `hpc/configs/server_v37.yml` | `hpc/slurm/submit_all.sh` |

Start at the per-environment README:
- [`local/README.md`](local/README.md) — Windows + WSL dev.
- [`hpc/README.md`](hpc/README.md) — full server run guide (env, paths, task list).
  GPU/GRES admin notes: [`hpc/gres_notes.md`](hpc/gres_notes.md).

## Configs: local vs server

`local/configs/wsl_asl_2d_home_v37.yml` and `hpc/configs/server_v37.yml` are the
**same recipe**; they differ **only** in `dataset.root_path` (local data mount vs
server data root). The Windows configs (`win_asl_2d_home*.yml`) are for the
Windows-side dev/inference path. When you change a *training* hyper-parameter,
change it in **both** wsl and server configs (they intentionally mirror).

## What did NOT move (and why)

- **`config/conf_data.py`** stays — it is the config *dataclass* loader (code),
  imported as `from config ...`. Only the `.yml` files moved out.
- **`scripts/*.py` and shared eval shell helpers** (`run_comparison_v3.sh`,
  `run_mismatched_t1.sh`, `eval_select_ckpt.py`, `eval_*_full.sh`) stay —
  they are environment-agnostic tools invoked from both local and HPC (the HPC
  guide calls them in Phase 3–4). Their `--config` default points at
  `env/local/configs/` as a dev fallback; pass `--config` explicitly on the HPC.
  Final-model selection is now **best-CNR** (`eval_select_ckpt.py --metric cnr
  --save_selected <run>/best_cnr.pth`, the global max-CNR checkpoint copied
  verbatim — no uMSE band, no SWA). The old feasible-set SWA helpers
  (`build_swa_cnr_primary.sh`, `build_swa_feasible.py`, `select_cnr_primary.py`)
  remain on disk but are **retained/deprecated** — no longer wired into Phase 3.
- **Legacy** chain scripts (`auto_chain_v31..v45*.sh`, `run_comparison_v2.sh`)
  moved to [`../scripts/archive/`](../scripts/archive/); the old `asl_2d_301.yml`
  to [`../config/archive/`](../config/archive/). Kept only for reading old
  commits/checkpoints — not part of any current workflow.

## Paths are repo-root-relative

All launchers `cd` to the repo root before running, and `--config env/.../*.yml`
is resolved from there. Run scripts from the repo root (or via the launchers,
which handle the `cd`).
