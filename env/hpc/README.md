# Server Run Guide — ASL_FastDenoising (Tianhe 天河)

Everything needed to train **this** line on the HPC. This repo owns one model line:
accelerated 7T ASL perfusion denoising with T1 cross-attention guidance
(`--use_t1_cross_fusion`), trained Noise2Noise with no clean ground truth.

> Not to be confused with **ASL_dmvae**, which hosts the other paper line (CIG-VSS +
> EC-LRDA). The two must live in **separate clones** and separate `$EXP` roots so runs
> never clobber each other — see CLAUDE.md §8.

---

## Quick start

Run everything on the **login node** unless a step says otherwise (compute nodes have
no outbound network).

```sh
# 1. get the code (public repo -> no deploy key, no token)
mkdir -p /fs1/home/duancaohui/jian/projects
cd /fs1/home/duancaohui/jian/projects
git clone https://github.com/Jian-uon/ASL_FastDenoising.git
cd ASL_FastDenoising

# 2. environment: the existing `asl-mamba` env is a SUPERSET of what this line needs.
#    Reuse it — env/hpc/env.sh already defaults to ENVNAME=asl-mamba.
source activate asl-mamba
python -c "import torch, monai, nibabel; print(torch.__version__, torch.cuda.is_available())"
#    Only build a fresh env if that fails:  sh env/hpc/install_env.sh   (on a GPU node)

# 3. check the data path
grep root_path env/hpc/configs/server_v35_joint.yml
#    -> /fs1/home/duancaohui/jian/data/7T_ASL_denoising

# 4. confirm you have the current code (this flag does not exist in older clones)
python runners/asl_t1_guided_runner_dmvae_n2n.py --help | grep window_fusion_levels

# 5. submit FROM THE REPO ROOT (see "Submit from the repo root" below — it matters)
yhbatch env/hpc/slurm/submit_v35_joint.sh                          # A0 baseline
WIN_LEVELS=2 WIN_K=t1  yhbatch env/hpc/slurm/submit_v35_joint.sh   # A1 main
WIN_LEVELS=2 WIN_K=asl yhbatch env/hpc/slurm/submit_v35_joint.sh   # A3 control

# watch
yhq
tail -f env/hpc/slurm/logs/v35j-<jobid>.out
```

Later updates: `sh env/hpc/sync_repo.sh` (fetch + `--ff-only` pull, prints the new HEAD).

---

## The three arms

One script, two env knobs. They write to different run names, so they never collide.

| Arm | Command | Run name | Question |
|---|---|---|---|
| **A0** | *(no knobs)* | `run_v35_joint_seed42` | baseline — bit-exactly the pre-2026-08-26 arch |
| **A1** | `WIN_LEVELS=2 WIN_K=t1` | `run_v35_joint_win2t1_seed42` | main result: anatomy-grouped window attention at 64²/128² |
| **A3** | `WIN_LEVELS=2 WIN_K=asl` | `run_v35_joint_win2asl_seed42` | control: same module, ASL keys, fine scales stay T1-free |
| A4 | `WIN_LEVELS=1 WIN_K=t1` | `run_v35_joint_win1t1_seed42` | level ablation, cheapest |

**A1 minus A3 is the net effect of anatomical guidance** — same module, same parameter
count, one flag apart. Design + decision rules: [../../docs/multiscale_window_design.md](../../docs/multiscale_window_design.md).

Other knobs: `SEED=1`, `MAX_STEPS=500`, `EVAL_EVERY=5`, `SAVE_EVERY=50`,
`T1_TASK=seg` (the `+seg` ablation arm), `EXTRA="--resume"`, `ENVNAME=...`,
`REPO=...`, `EXP=...`.

---

## Submit from the repo root

`submit_v35_joint.sh` declares its Slurm log path **relative** to the submission
directory:

```
#SBATCH -o env/hpc/slurm/logs/v35j-%j.out
```

Slurm creates that file *before* the job script runs (so before the script's own
`cd "$REPO"`). Two consequences:

1. **`yhbatch` must be run from the repo root**, or the path does not resolve.
2. **`env/hpc/slurm/logs/` must exist.** It is kept in git via a `.gitkeep`; if you ever
   see the job vanish with no `.out`, no `.err` and nothing in `yhq`, check that
   directory first — that failure mode produces no diagnostics anywhere, because the
   place the diagnostics would go is the thing that is missing.

```sh
mkdir -p env/hpc/slurm/logs      # harmless; fixes it if the directory is gone
```

---

## Paths

| What | Where | In git? |
|---|---|---|
| Code | `/fs1/home/duancaohui/jian/projects/ASL_FastDenoising` | yes |
| Data | `/fs1/home/duancaohui/jian/data/7T_ASL_denoising` | **no** — copy separately |
| Run outputs (`logs/`, `tensorboard/`) | `$EXP` = `/fs1/home/duancaohui/jian/exp/ASL_FastDenoising` | **no** |
| Slurm job logs | `env/hpc/slurm/logs/v35j-<jobid>.{out,err}` | dir only |

`$EXP` is set by [env.sh](env.sh) and is deliberately separate from ASL_dmvae's.

Data layout per subject (`<root>/<subject_id>/raw/`): `asldata_diff.nii.gz` `[H,W,Z,T]`,
`t1_in_asl.nii.gz`, `m0.nii.gz`, and `gm/wm/csf_asl.nii.gz` (used by the val metrics and
slice filtering; the default no-seg training arm needs no labels).

---

## Environment

This line is **conv-only** — no `mamba-ssm` / `causal-conv1d` needed. The `mamba_ssm`
import is guarded (`try/except` in `models/asl_mamba_student.py`), so the retired
MoSSM/VMamba classes that `models/asl_t1_model.py` imports at module scope still load
without it.

So `asl-mamba` is a superset and reusing it is the low-risk choice. The trade-off worth
knowing: **the two paper lines then share one environment**, so upgrading a package for
one changes the other's reproducibility. Record versions before a submission-grade run:

```sh
pip freeze > env/hpc/freeze_$(date +%Y%m%d).txt
```

To isolate instead: `conda create --clone asl-mamba -n asl-fast`, then
`ENVNAME=asl-fast yhbatch ...` (no code change — `env.sh` reads `$ENVNAME`).

If the env must be rebuilt from scratch, [install_env.sh](install_env.sh) does it in one
shot **on a GPU node** (`torch.cuda` must work for its cuDNN check). Its CUDA notes still
apply: mamba-ssm's build requires the toolchain CUDA major to equal `torch.version.cuda`'s
major — those kernels are optional for *this* line, but the script builds them because the
env is shared.

---

## Runtime

627 batches/epoch (329 subjects, 5015 train slices), 500 epoch cap with early stop
(`--early_stop_patience 20 --early_stop_min_evals 60`, so ≥ step 300).

Measured locally on an RTX 5070 Ti: **~2.4 min/epoch** with `WIN_LEVELS=2`, i.e. ~12 h
to step 300 and ~20 h for the full 500. The `#SBATCH --time=72:00:00` has ample margin;
an H100 should be faster.

---

## After a run

Best-checkpoint selection is **step-gated** in-loop (`best_min_step` falls back to
`sure_anneal_start=200`), so a run that stops earlier has only the periodic
`step0000NN.pth` snapshots. Always confirm the operating point post hoc:

```sh
python scripts/eval_select_ckpt.py --metric umse \
  --save_selected $EXP/logs/<run>/checkpoints/best_umse_posthoc.pth
```

Never select on L1 / `psnr_ref` / `psnr_b` — see CLAUDE.md §4.

Window-fusion probes land in TensorBoard under `window/wf{level}_{gate,entropy,delta}`.
`entropy` pinned at ln(ws²) = 4.159 for ws=8 means the grouping was never learnt and the
module is still a box blur — check that before comparing arms.
