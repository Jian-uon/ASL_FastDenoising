"""Render RAW vs EMA reconstructions side-by-side + CNR/sCoV readout.

For a given checkpoint, produces a figure with rows = sample slices, columns =
[T1, input mean(setA), 12-NEX union, EMA recon, RAW recon], and prints a metrics
table (CNR, sCoV-GM, sCoV-WM, lapvR) for EMA recon / RAW recon / union, averaged
over the first --n_cnr batches.

This supports the "option B" presentation decision: see whether the RAW (sharper)
weights give a perceptually better PWI and whether the extra texture *improves*
GM/WM CNR (real contrast) or just raises WM noise.
"""
import argparse
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

from scripts.eval_select_ckpt import _build_runner  # noqa: E402
from runners.asl_t1_guided_runner_dmvae_n2n import (  # noqa: E402
    prepare_asl_pair_batch, direct_mean_from_frames,
)
from utils.metrics import cnr, scov, laplacian_variance  # noqa: E402


def _img(t):
    return t.detach().float().cpu().numpy()[0, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--runner_args", required=True)
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--n_rows", type=int, default=3, help="How many slices to plot.")
    p.add_argument("--n_cnr", type=int, default=12, help="Batches to average metrics over.")
    args = p.parse_args()

    runner, _ = _build_runner(args.runner_args)
    dev = runner.device
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    raw_state = {k: v.to(dev) for k, v in ck["model"].items()}
    ema_state = {k: v.to(dev) for k, v in ck["ema"].items()}

    def predict(pack, state):
        runner.ema.ema_state = state
        return runner._predict_with_ema(pack["setA"], pack["t1"], pack["lenA"], pack.get("maskA"))["asl_recon"]

    acc = {k: 0.0 for k in
           ["cnr_ema", "cnr_raw", "cnr_uni", "scgm_ema", "scgm_raw",
            "scwm_ema", "scwm_raw", "lap_ema", "lap_raw"]}
    n_acc = 0
    plot_rows = []

    runner.model.eval()
    with torch.no_grad():
        for i, vb in enumerate(runner.val_loader):
            if i >= args.n_cnr:
                break
            pack = prepare_asl_pair_batch(vb, dev)
            pack = runner._mask_asl_inputs(pack)
            inp = direct_mean_from_frames(pack["setA"], pack["lenA"])
            union = direct_mean_from_frames(
                torch.cat([pack["setA"], pack["setB"]], dim=1),
                pack["lenA"] + pack["lenB"])
            rec_ema = predict(pack, ema_state)
            rec_raw = predict(pack, raw_state)
            t1 = pack["t1"]
            gm = pack.get("gm"); wm = pack.get("wm")
            mask_b = (t1 > 0.05).float()
            if gm is not None and wm is not None:
                acc["cnr_ema"] += cnr(rec_ema, gm, wm); acc["cnr_raw"] += cnr(rec_raw, gm, wm)
                acc["cnr_uni"] += cnr(union, gm, wm)
                acc["scgm_ema"] += scov(rec_ema, gm); acc["scgm_raw"] += scov(rec_raw, gm)
                acc["scwm_ema"] += scov(rec_ema, wm); acc["scwm_raw"] += scov(rec_raw, wm)
                acc["lap_ema"] += float(laplacian_variance(rec_ema, mask_b) / max(float(laplacian_variance(union, mask_b)), 1e-8))
                acc["lap_raw"] += float(laplacian_variance(rec_raw, mask_b) / max(float(laplacian_variance(union, mask_b)), 1e-8))
                n_acc += 1
            if len(plot_rows) < args.n_rows:
                plot_rows.append({
                    "t1": _img(t1), "inp": _img(inp), "uni": _img(union),
                    "ema": _img(rec_ema), "raw": _img(rec_raw),
                })

    n = max(n_acc, 1)
    print(f"\n=== {Path(args.ckpt).name}  (averaged over {n_acc} batches) ===")
    print(f"{'metric':<10} {'EMA':>10} {'RAW':>10} {'union':>10}")
    print(f"{'CNR':<10} {acc['cnr_ema']/n:>10.3f} {acc['cnr_raw']/n:>10.3f} {acc['cnr_uni']/n:>10.3f}")
    print(f"{'sCoV-GM':<10} {acc['scgm_ema']/n:>10.4f} {acc['scgm_raw']/n:>10.4f} {'-':>10}")
    print(f"{'sCoV-WM':<10} {acc['scwm_ema']/n:>10.4f} {acc['scwm_raw']/n:>10.4f} {'-':>10}")
    print(f"{'lapvR':<10} {acc['lap_ema']/n:>10.3f} {acc['lap_raw']/n:>10.3f} {1.0:>10.3f}")

    # ---- figure ----
    cols = [("T1", "t1"), ("input mean(setA)", "inp"), ("12-NEX union", "uni"),
            ("EMA recon", "ema"), ("RAW recon", "raw")]
    nr = len(plot_rows)
    fig, axes = plt.subplots(nr, len(cols), figsize=(3 * len(cols), 3 * nr))
    if nr == 1:
        axes = axes[None, :]
    for r, row in enumerate(plot_rows):
        vmax = float(np.percentile(row["uni"][row["uni"] > 0], 99)) if (row["uni"] > 0).any() else 1.0
        for c, (title, key) in enumerate(cols):
            ax = axes[r][c]
            if key == "t1":
                ax.imshow(np.rot90(row[key]), cmap="gray")
            else:
                ax.imshow(np.rot90(row[key]), cmap="gray", vmin=0, vmax=vmax)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=11)
    fig.suptitle(f"{Path(args.ckpt).name} — RAW (sharper) vs EMA (smoother)", fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[render] saved {args.out}")


if __name__ == "__main__":
    main()
