"""Render the EMA reconstruction of the best checkpoint PER validation metric.

Post-hoc, no training-loop coupling — works on any already-trained ASL run.

Pipeline:
  1. Rebuild the runner from the training CLI (`--runner_args`, same string
     eval_select_ckpt.py / phase4d use).
  2. Evaluate EVERY given checkpoint on the full validation metric suite
     (the same `_validate_with_per_unit_umse` that drives selection), on EMA weights.
  3. For each metric with a well-defined "better" direction AND that actually varies
     across checkpoints, pick the argmax/argmin checkpoint.
  4. Render ONE row per DISTINCT winning checkpoint on a fixed validation slice
     (T1 | input mean(setA) | 12-NEX union | EMA recon), captioned with the metrics
     that checkpoint won.  Also write best_per_metric.json + print a table.

Metrics with no project-defined direction (e.g. lapvr ratio, MI-with-union) or that
are constant across checkpoints (fixed-reference quantities) are SKIPPED and logged —
never silently dropped.  Biased-vs-12NEX-union metrics (*_ref) are included but tagged
`[biased-ref]`; the project never SELECTS the final model on them (see
docs/validation_metrics.md, CLAUDE.md §4) — they are here only to visualize what
chasing them looks like.
"""
import argparse
import glob as _glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.eval_select_ckpt import (  # noqa: E402
    _build_runner, _load_model_weights, _validate_with_per_unit_umse,
)
from runners.asl_t1_guided_runner_dmvae_n2n import (  # noqa: E402
    prepare_asl_pair_batch, direct_mean_from_frames,
)

# Project-endorsed self-supervised / ASL-QC metrics with a clear better-direction.
# "min" = lower better (argmin), "max" = higher better (argmax).
_DIRS_PRIMARY = {
    "pooled_umse": "min", "pooled_upsnr": "max", "pooled_upsnr_cyc": "max",
    "pooled_cnr": "max",
    "pooled_scov_gm": "min", "pooled_scov_wm": "min", "pooled_scov_gmwm": "min",
    "pooled_snr_gm": "max", "pooled_snr_wm": "max",
    "pooled_cyc": "min", "pooled_hfcyc": "min",
    "pooled_mlc": "min", "pooled_mlc_aniso": "min", "pooled_mslc": "min",
    "pooled_efc": "min", "pooled_hfen": "min", "pooled_bright_tail": "min",
}
# Biased-vs-12NEX-union (reported, NEVER selected on). Tagged [biased-ref] in output.
_DIRS_SUPP = {
    "pooled_cnr_ref": "max", "pooled_efc_ref": "min",
    "pooled_snr_gm_ref": "max", "pooled_snr_wm_ref": "max",
}
# Deliberately NOT selectable (direction undefined / non-monotone) — logged as skipped.
_SKIP_BY_DESIGN = {"pooled_lapvr", "pooled_mi_union", "pooled_nmi_union"}


def _img(t):
    return t.detach().float().cpu().numpy()[0, 0]


def _freeze(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _short(name: str) -> str:
    return name[len("pooled_"):] if name.startswith("pooled_") else name


def _predict_slice(runner, dev, sample_idx: int):
    """Return the (sample_idx)-th val batch rendered on EMA weights currently loaded."""
    runner.model.eval()
    with torch.no_grad():
        for i, vb in enumerate(runner.val_loader):
            if i < sample_idx:
                continue
            pack = prepare_asl_pair_batch(vb, dev)
            pack = runner._mask_asl_inputs(pack)
            inp = direct_mean_from_frames(pack["setA"], pack["lenA"])
            union = direct_mean_from_frames(
                torch.cat([pack["setA"], pack["setB"]], dim=1),
                pack["lenA"] + pack["lenB"])
            rec = runner._predict_with_ema(
                pack["setA"], pack["t1"], pack["lenA"], pack.get("maskA"))["asl_recon"]
            return {"t1": _img(pack["t1"]), "inp": _img(inp),
                    "uni": _img(union), "rec": _img(rec)}
    raise SystemExit(f"[render] val_loader has < {sample_idx + 1} batches")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Checkpoint paths / globs (e.g. <run>/checkpoints/step*.pth).")
    p.add_argument("--runner_args", required=True,
                   help="Training CLI string (same as eval_select_ckpt / phase4d RA_*).")
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--json_out", default=None,
                   help="Where to write best_per_metric.json (default: alongside --out).")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Which validation batch to render (fixed across ckpts).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_supp", action="store_true",
                   help="Drop the biased-vs-union [biased-ref] metrics.")
    args = p.parse_args()

    # expand globs
    ckpts = []
    for c in args.ckpts:
        hits = sorted(_glob.glob(c))
        ckpts.extend(hits if hits else [c])
    ckpts = [c for c in dict.fromkeys(ckpts) if Path(c).exists()]
    if not ckpts:
        raise SystemExit("[render] no checkpoints matched --ckpts")

    ra = args.runner_args
    if "--name" not in ra:
        ra = ra + " --name render_best_tmp"
    runner, _ = _build_runner(ra)
    dev = runner.device

    dirs = dict(_DIRS_PRIMARY)
    if not args.no_supp:
        dirs.update(_DIRS_SUPP)

    # ---- pass 1: full metric suite per ckpt ----
    table = {}   # ckpt -> {metric: value}; step -> int
    steps = {}
    print(f"[render] evaluating {len(ckpts)} checkpoints on the full metric suite ...")
    for ckpt in ckpts:
        _load_model_weights(runner, ckpt)
        _freeze(args.seed)
        v = _validate_with_per_unit_umse(runner)
        table[ckpt] = {k: float(v[k]) for k in v if k.startswith("pooled_")}
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        steps[ckpt] = int(sd.get("global_step", sd.get("step", -1)))
        print(f"[render]   {Path(ckpt).name} (step {steps[ckpt]}) "
              f"uMSE={table[ckpt].get('pooled_umse', float('nan')):.5f} "
              f"CNR={table[ckpt].get('pooled_cnr', float('nan')):.4f}")

    # ---- selection: best ckpt per metric (skip constant / undefined) ----
    best = {}        # metric -> {"ckpt", "step", "value"}
    skipped = {}     # metric -> reason
    present = set()
    for m in table[ckpts[0]]:
        present.add(m)
    for m in sorted(present):
        if m in _SKIP_BY_DESIGN:
            skipped[m] = "no project-defined better-direction (non-monotone / ref-MI)"
            continue
        if m not in dirs:
            skipped[m] = "unmapped metric — direction not defined here"
            continue
        vals = np.array([table[c].get(m, np.nan) for c in ckpts], dtype=float)
        good = np.isfinite(vals)
        if good.sum() == 0:
            skipped[m] = "all-NaN across checkpoints"
            continue
        vv = vals[good]
        if float(vv.max() - vv.min()) <= 1e-9 * (abs(float(vv.mean())) + 1e-9):
            skipped[m] = "constant across checkpoints (fixed-reference quantity)"
            continue
        idxs = np.where(good)[0]
        pick = idxs[int(np.argmin(vv)) if dirs[m] == "min" else int(np.argmax(vv))]
        ck = ckpts[pick]
        best[m] = {"ckpt": ck, "step": steps[ck], "value": float(vals[pick]),
                   "biased_ref": m in _DIRS_SUPP}

    # ---- report table ----
    print("\n=== best checkpoint per metric ===")
    print(f"{'metric':<22}{'dir':>4}{'best step':>11}{'value':>14}   ckpt")
    for m in sorted(best, key=lambda k: (_DIRS_SUPP.get(k) is not None, k)):
        b = best[m]
        tag = " [biased-ref]" if b["biased_ref"] else ""
        print(f"{_short(m):<22}{dirs[m]:>4}{b['step']:>11}{b['value']:>14.5g}   "
              f"{Path(b['ckpt']).name}{tag}")
    if skipped:
        print("\n--- skipped metrics (not silently dropped) ---")
        for m, why in sorted(skipped.items()):
            print(f"  {_short(m):<22} {why}")

    # ---- winners -> which metrics each won ----
    winners = {}   # ckpt -> [metric,...]
    for m, b in best.items():
        winners.setdefault(b["ckpt"], []).append(m)
    # order winners by step for a readable montage
    winner_ckpts = sorted(winners, key=lambda c: steps[c])

    # ---- pass 2: render one row per distinct winning ckpt on a fixed slice ----
    print(f"\n[render] rendering {len(winner_ckpts)} distinct winning checkpoints ...")
    rows = []
    for ck in winner_ckpts:
        _load_model_weights(runner, ck)
        row = _predict_slice(runner, dev, args.sample_idx)
        mets = winners[ck]
        label = f"step {steps[ck]}\n" + ", ".join(
            _short(m) + ("*" if best[m]["biased_ref"] else "") for m in
            sorted(mets, key=lambda k: (_DIRS_SUPP.get(k) is not None, k)))
        rows.append((label, row))

    cols = [("T1", "t1"), ("input mean(setA)", "inp"),
            ("12-NEX union", "uni"), ("EMA recon", "rec")]
    nr = len(rows)
    fig, axes = plt.subplots(nr, len(cols), figsize=(3 * len(cols), 3 * nr + 0.5))
    if nr == 1:
        axes = axes[None, :]
    for r, (label, row) in enumerate(rows):
        pos = row["uni"][row["uni"] > 0]
        vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
        for c, (title, key) in enumerate(cols):
            ax = axes[r][c]
            if key == "t1":
                ax.imshow(np.rot90(row[key]), cmap="gray")
            else:
                ax.imshow(np.rot90(row[key]), cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
        axes[r][0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=42)
    fig.suptitle("Best checkpoint per validation metric  (* = biased-ref, not selected on)",
                 fontsize=12)
    fig.tight_layout(rect=(0.06, 0, 1, 0.98))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[render] saved figure -> {args.out}")

    json_out = args.json_out or str(Path(args.out).with_suffix(".json"))
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"best": {_short(m): {"step": b["step"], "value": b["value"],
                                        "ckpt": b["ckpt"], "biased_ref": b["biased_ref"]}
                            for m, b in best.items()},
                   "skipped": {_short(m): why for m, why in skipped.items()},
                   "sample_idx": args.sample_idx}, f, indent=2)
    print(f"[render] saved index  -> {json_out}")


if __name__ == "__main__":
    main()
