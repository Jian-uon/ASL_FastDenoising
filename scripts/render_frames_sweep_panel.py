#!/usr/bin/env python
"""Frames-sweep reconstruction montage.

For K example subjects, render a [method ROWS x n_frames COLUMNS] grid of
reconstructions plus the 12-NEX union reference, so that "more frames -> sharper
recon" is directly visible per subject and per method. This is the qualitative
complement to the `sweep` PHASE (which plots metrics-vs-frames, not images).

Reuses eval_comparison_table's proven loading + inference so every method runs in
its NATIVE pipeline on the SAME materialised split and a shared in-brain intensity
window: ours/ablations via the runner machinery (EMA weights, seg-head-masked
input), baselines via load_unet/run_unet. Operating-point ckpt = best_umse.pth.

Usage (server, after the 4-config ladder + baselines are trained/selected):
  python scripts/render_frames_sweep_panel.py --config $CONFIG --split test \
    --slice_context 0 --n_frames 2 4 6 8 12 --n_subjects 4 \
    --out_dir $OUT/frames_montage \
    --ours   $LOGS/cig_vss_ec_csem_bayes_seed42/best_umse.pth \
    --ours_runner_args "$RA_BAYES" --ours_label "SAGE (Bayes)" \
    --extra_runner "M1 (g1)::$LOGS/cig_vss_ec_a1_seed42/best_umse.pth::$RA_A1" \
    --extra_runner "Raw-T1-bil::$LOGS/cig_vss_rawt1_seed42/best_umse.pth::$RA_RAWT1" \
    --extra_runner "M0 (pv0)::$LOGS/cig_vss_pv0_seed42/best_umse.pth::$RA_PV0" \
    --vanilla $LOGS/baseline_n2n_seed42/best_umse.pth --include_naive

RA_* are the SAME structural runner-arg fragments as eval_step400.sh (must match
how each ckpt was trained so it strict-loads). --slice_context MUST match training.
"""
from __future__ import annotations
import argparse
import os
import sys

# Make both the repo root and scripts/ importable, then reuse eval_comparison_table.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root (config/, runners/, utils/)
sys.path.insert(0, _HERE)                    # scripts/ (eval_comparison_table)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import eval_comparison_table as ect          # loaders + infer (build_runner_from, run_ours_runner, ...)


def parse_args():
    p = argparse.ArgumentParser(description="Frames-sweep reconstruction montage (method x n_frames grid per subject).")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slice_context", type=int, default=0, help="MUST match training (0=2D, 2=2.5D 5-slice).")
    p.add_argument("--error_column", action="store_true",
                   help="add the |error| column at the shortest acquisition; a diagnostic, "
                        "and off by default because the paper figure does not carry it.")
    p.add_argument("--dpi", type=int, default=600,
                   help="600 for print; the montage was previously written at 110")
    p.add_argument("--n_frames", type=int, nargs="+", default=[2, 4, 6, 8, 12],
                   help="input frame budgets to draw as columns (2..12 spans setA+setB pool).")
    p.add_argument("--n_subjects", type=int, default=4,
                   help="how many example slices to render (one PNG each).")
    p.add_argument("--first", action="store_true",
                   help="take the first N batches instead of spreading the picks evenly "
                        "across the split. The dataset is flattened per slice and loaded "
                        "unshuffled, so the first N batches are adjacent slices of the "
                        "first subject or two -- fine for a smoke test, useless for "
                        "choosing figure candidates.")
    p.add_argument("--max_samples", type=int, default=0, help="0=all val batches; >0 = smoke subset.")
    p.add_argument("--out_dir", required=True)
    # ours + ablations (runner machinery). --extra_runner repeatable: 'LABEL::CKPT::RUNNER_ARGS'.
    p.add_argument("--ours", default=None)
    p.add_argument("--ours_runner_args", default="")
    p.add_argument("--ours_label", default="ours")
    p.add_argument("--extra_runner", action="append", default=[], help="LABEL::CKPT::RUNNER_ARGS (repeatable)")
    # baselines (PlainUNet / SwinIR loaders); load_unet needs base_ch/depth.
    p.add_argument("--vanilla", default=None)
    p.add_argument("--n2self", default=None)
    p.add_argument("--sup", default=None)
    p.add_argument("--swinir_sup", default=None)
    p.add_argument("--swinir_n2n", default=None,
                   help="SwinIR2D trained with N2N; load_unet dispatches the arch from the "
                        "checkpoint, so this exists only to label the montage row correctly.")
    p.add_argument("--concat", default=None)
    p.add_argument("--include_naive", action="store_true", help="add the naive frame-mean column-set.")
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    return p.parse_args()


def _center_slice(x: torch.Tensor, ctx: int) -> torch.Tensor:
    """2.5D union carries K=2*ctx+1 z-slices as channels; reduce to the center slice
    [B,1,H,W] so it matches the models' center-slice predictions. No-op for 2D."""
    return x if ctx <= 0 else x[:, ctx:ctx + 1]


PAPER_NAME = {
    "naive_mean":  "Repetition\naveraging",
    "vanilla_N2N": "UNet-N2N",
    "SwinIR_N2N":  "SwinIR-N2N",
    "asl_keys":    "ASL keys\n(no T1)",
    "proposed":    "Proposed",
}


SPACER = 0.22      # width of the gap column, as a fraction of one panel


def _save_frames_grid(t1, union, grid, nframes, path, title="", dpi=600,
                      show_error=False):
    """One subject: rows = methods, cols = [12-NEX ref | each n_frames]. Shared
    in-brain window from the reference percentiles (same convention as
    eval_comparison_table._save_panel). No-op-safe on failure."""
    try:
        from scipy.ndimage import binary_fill_holes
        t1_np = t1[0, 0].cpu().numpy()
        u_raw = union[0, 0].cpu().numpy()
        brain = binary_fill_holes((t1_np > 0.05) & (np.abs(u_raw) > 0)).astype(np.float32)
        u = u_raw * brain
        ub = u[brain > 0]
        if ub.size > 0:
            vmin, vmax = float(np.percentile(ub, 1)), float(np.percentile(ub, 99))
        else:
            vmin, vmax = float(u.min()), float(u.max() + 1e-6)
        if not (vmax > vmin):
            vmin, vmax = float(u.min()), float(u.max() + 1e-6)

        methods = list(grid.keys())
        k_res = min(nframes)                       # residual is shown at the shortest acquisition
        res = {m: np.abs(grid[m][k_res][0, 0].cpu().numpy() * brain - u) for m in methods}
        rb = np.concatenate([r[brain > 0] for r in res.values()]) if brain.any() else np.array([0.0])
        rmax = float(np.percentile(rb, 99)) if rb.size else 1.0

        # Column 0 is the acquired image; everything right of the spacer is a reconstruction.
        # The gap says so without a word of caption. The residual column is off by default: it
        # is a diagnostic, and the figure the paper carries does not use it.
        nrow = len(methods)
        ncol = len(nframes) + 1 + (1 if show_error else 0)
        fig = plt.figure(figsize=(2.35 * ncol + 0.7, 2.35 * nrow + 0.95))
        gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1.0, SPACER] + [1.0] * (ncol - 1),
                              left=0.055, right=0.995, top=0.855, bottom=0.01,
                              wspace=0.045, hspace=0.045)

        def cell(r, c):                       # c counts real columns, skipping the spacer
            a = fig.add_subplot(gs[r, c if c == 0 else c + 1])
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            return a

        # The reference column is the union of every acquired repetition. --n_frames ends
        # at that same count in every configuration used here, so it names the column;
        # a montage that stopped short of the full acquisition would mislabel it.
        ref_n = max(nframes)
        # Three rotations, to land in the orientation Figure 5 draws; two figures of the
        # same brain must not disagree about which way is up.
        rot = lambda x: np.rot90(x, 3)
        top = {}
        for r, name in enumerate(methods):
            a = cell(r, 0)
            a.imshow(rot(u), cmap="gray", vmin=vmin, vmax=vmax)
            a.set_ylabel(PAPER_NAME.get(name, name), fontsize=12)
            if r == 0:
                a.set_title("%d repetitions\naveraged" % ref_n, fontsize=12)
                top[0] = a
            for c, nf in enumerate(nframes, start=1):
                a = cell(r, c)
                a.imshow(rot(grid[name][nf][0, 0].cpu().numpy() * brain),
                         cmap="gray", vmin=vmin, vmax=vmax)
                if r == 0:
                    a.set_title("%d repetitions" % nf, fontsize=12)
                    top[c] = a
            if show_error:
                a = cell(r, ncol - 1)
                a.imshow(rot(res[name]), cmap="inferno", vmin=0.0, vmax=rmax)
                if r == 0:
                    a.set_title("|error| at %d rep." % k_res, fontsize=12)
                    top[ncol - 1] = a

        # Which column is the acquired image and which are reconstructions, on the figure
        # rather than only in the caption.
        y = max(t.get_position().y1 for t in top.values()) + 0.075
        last = len(nframes)
        for c0, c1, lab in ((0, 0, "Acquired"), (1, last, "Reconstructed")):
            x0 = top[c0].get_position().x0
            x1 = top[c1].get_position().x1
            fig.add_artist(Line2D([x0, x1], [y, y], color="#333333", lw=1.4))
            fig.text((x0 + x1) / 2, y + 0.008, lab, ha="center", va="bottom",
                     fontsize=14, fontweight="bold")
        # The subject identifier is a patient name; it stays in the file name, off the figure.
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[frames-montage] wrote {path}")
    except Exception as e:
        plt.close("all")
        print(f"[frames-montage] FAILED {path}: {e}")


@torch.no_grad()
def main():
    args = parse_args()
    ect.set_seed(args.seed)
    cfg = ect.Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    ctx = int(args.slice_context)

    print(f"[1/2] materialising '{args.split}' split (seed={args.seed}, slice_context={ctx}) ...")
    val_data = ect.materialise_split(cfg, device, split=args.split, max_samples=args.max_samples,
                                     slice_context=ctx, batch_size=1)
    K = min(args.n_subjects, len(val_data))
    if K == 0:
        raise SystemExit("[frames-montage] empty split — nothing to render.")

    # method(pack, nf, k) -> pred [B,1,H,W] (center slice). Baselines first, ours/ablations after.
    methods = {}
    if args.include_naive:
        methods["naive_mean"] = lambda pack, nf, k: ect.run_naive(
            ect._presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if args.vanilla:
        m = ect.load_unet(args.vanilla, args, device)
        methods["vanilla_N2N"] = lambda pack, nf, k, m=m: ect.run_unet(
            m, ect._presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if args.n2self:
        m = ect.load_unet(args.n2self, args, device)
        methods["Noise2Self"] = lambda pack, nf, k, m=m: ect.run_unet(
            m, ect._presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if args.sup:
        m = ect.load_unet(args.sup, args, device)
        methods["UNet_sup(12NEX)"] = lambda pack, nf, k, m=m: ect.run_unet(
            m, ect._presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if args.swinir_sup:
        m = ect.load_unet(args.swinir_sup, args, device)
        methods["SwinIR_sup"] = lambda pack, nf, k, m=m: ect.run_unet(
            m, ect._presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if getattr(args, "swinir_n2n", None):
        m = ect.load_unet(args.swinir_n2n, args, device)
        methods["SwinIR_N2N"] = lambda pack, nf, k, m=m: ect.run_unet(
            m, ect._presubset_pack(pack, nf, args.seed, k, device), 0, device)
    if args.concat:
        m = ect.load_unet(args.concat, args, device)
        methods["PlainUNet_concat"] = lambda pack, nf, k, m=m: ect.run_unet(
            m, ect._presubset_pack(pack, nf, args.seed, k, device), 0, device)
    for spec in args.extra_runner:
        label, ckpt, ra = spec.split("::", 2)
        runner = ect.build_runner_from(ra, ckpt, label=label)
        methods[label] = lambda pack, nf, k, r=runner: ect.run_ours_runner(
            r, pack, nf, args.seed, k, device, apply_input_mask=True)
    if args.ours:
        runner = ect.build_runner_from(args.ours_runner_args, args.ours, label=args.ours_label)
        methods[args.ours_label] = lambda pack, nf, k, r=runner: ect.run_ours_runner(
            r, pack, nf, args.seed, k, device, apply_input_mask=True)

    if not methods:
        raise SystemExit("[frames-montage] no methods given (need --ours / --extra_runner / --vanilla / ...).")
    print(f"[2/2] rendering {K} slices x {len(methods)} methods x {len(args.n_frames)} frame budgets ...")

    # Even picks across the whole split, so K images span the held-out subjects rather than
    # K neighbouring slices of the first one.
    if args.first or K >= len(val_data):
        picks = list(range(min(K, len(val_data))))
    else:
        picks = [round(i * (len(val_data) - 1) / (K - 1)) for i in range(K)] if K > 1 else [0]
    print(f"[2/2] picks {picks[:6]}{' ...' if len(picks) > 6 else ''} of {len(val_data)} batches")

    for k in picks:
        pack = val_data[k]
        union = ect.direct_mean_from_frames(
            torch.cat([pack["setA"].to(device), pack["setB"].to(device)], dim=1))
        union = _center_slice(union, ctx)
        t1 = pack["t1"].to(device)
        sid = pack.get("subject_id", [f"subj{k:02d}"])
        sid = sid[0] if isinstance(sid, (list, tuple)) else sid
        grid = {}
        for name, fn in methods.items():
            grid[name] = {nf: fn(pack, nf, k) for nf in args.n_frames}
        _save_frames_grid(t1, union, grid, args.n_frames,
                          os.path.join(args.out_dir, f"frames_{k:02d}_{sid}.png"),
                          title=str(sid), dpi=args.dpi, show_error=args.error_column)

    print(f"[frames-montage] done -> {args.out_dir}  ({len(picks)} slices)")


if __name__ == "__main__":
    main()
