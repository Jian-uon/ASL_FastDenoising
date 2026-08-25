# -*- coding: utf-8 -*-
"""Unified evaluation: NLM + 3 trainable baselines + ours, on the same val set.

Materialises the val set once with a fixed seed so every method sees the same
(setA, setB, t1) tuples; then evaluates each method, optionally over an
n_frames sweep. Outputs a long CSV + summary table + paired Wilcoxon stats.

Usage
-----
python runners/eval_baselines.py --config env/local/configs/win_asl_2d_home.yml \
    --out_dir C:/tmp/asl_exp/baselines_eval \
    --ours      C:/tmp/asl_exp/logs/run_full_v17/checkpoints/best.pth \
    --vanilla   C:/tmp/asl_exp/logs/baseline_n2n/checkpoints/best.pth \
    --sup       C:/tmp/asl_exp/logs/baseline_sup/checkpoints/best.pth \
    --n2self    C:/tmp/asl_exp/logs/baseline_n2self/checkpoints/best.pth \
    --include_nlm \
    --n_frames 2 3 4 6 8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.conf_data import Config
from dataio.dataloaders import get_asl_2d_loaders
from models.asl_t1_model import ASLT1Denoiser
from models.plain_unet import PlainUNet2D
from runners.asl_t1_guided_runner_dmvae_n2n import (
    _compute_psnr_ssim,
    direct_mean_from_frames,
    prepare_asl_pair_batch,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser("Unified baseline eval")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"],
                   help="Held-out split for reporting (test = unbiased).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_frames", type=int, nargs="+", default=[0],
                   help="Frame counts to sweep; 0 = use full setA.")
    # Method checkpoints (all optional)
    p.add_argument("--ours",    type=str, default=None, help="ASLT1Denoiser checkpoint (label=ours)")
    p.add_argument("--ours2",   type=str, default=None, help="Second ASLT1Denoiser variant")
    p.add_argument("--ours2_label", type=str, default="ours_v23")
    p.add_argument("--ours3",   type=str, default=None, help="Third ASLT1Denoiser variant")
    p.add_argument("--ours3_label", type=str, default="ours_v24")
    p.add_argument("--ours4",   type=str, default=None, help="Fourth ASLT1Denoiser variant")
    p.add_argument("--ours4_label", type=str, default="ours_v25")
    p.add_argument("--vanilla", type=str, default=None, help="PlainUNet2D (mode=n2n) checkpoint")
    p.add_argument("--sup",     type=str, default=None, help="PlainUNet2D (mode=sup) checkpoint")
    p.add_argument("--n2self",  type=str, default=None, help="PlainUNet2D (mode=n2self) checkpoint")
    p.add_argument("--include_nlm", action="store_true")
    p.add_argument("--include_naive", action="store_true",
                   help="Add naive mean(A) (no denoising) baseline.")
    # Plain U-Net architecture (same for all PlainUNet2D ckpts here)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    # Ours architecture
    p.add_argument("--ours_base_ch", type=int, default=32)
    p.add_argument("--ours_depth", type=int, default=4)
    # NLM params
    p.add_argument("--nlm_patch_size", type=int, default=5)
    p.add_argument("--nlm_patch_distance", type=int, default=6)
    p.add_argument("--nlm_h_factor", type=float, default=0.6)
    # Subset for smoke test
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--save_figures", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------
# Materialise val set once
# ------------------------------------------------------------------

def materialise_split(cfg, device, split: str = "test", max_samples: int = 0, slice_context: int = 0,
                      batch_size: int = None):
    """Return list of dicts with ['setA','setB','t1','lenA','lenB','subject_id'] from split.

    slice_context>0 packs 2*ctx+1 adjacent z-slices per frame as channels (2.5D) —
    MUST match the evaluated model's --slice_context, else setA carries the wrong
    channel count (a 2.5D model expects K=5 channels, a 2D loader gives 1).
    batch_size overrides the config batch size (set 1 for exact per-subject stats,
    since per-batch pooled metrics cannot be split back to subjects)."""
    tp = cfg.asl_denoiser_train_params
    loaders = get_asl_2d_loaders(
        cfg, modes=[split],
        asl_hw=tp.asl_hw, asl_z=tp.asl_z, t1_hw=tp.t1_hw, t1_z=tp.t1_z,
        slice_context=int(slice_context),
    )
    dl = loaders[split]
    if batch_size is not None and int(batch_size) != dl.batch_size:
        from monai.data import DataLoader as _DL
        from utils.utils import collate_varlen as _collate
        dl = _DL(dl.dataset, batch_size=int(batch_size), shuffle=False,
                 num_workers=0, collate_fn=_collate)
    out = []
    for k, vb in enumerate(dl):
        if max_samples and k >= max_samples:
            break
        pack = prepare_asl_pair_batch(vb, device)
        cached = {
            "setA": pack["setA"].cpu(), "setB": pack["setB"].cpu(),
            "t1":   pack["t1"].cpu(),
            "lenA": pack["lenA"].cpu(), "lenB": pack["lenB"].cpu(),
            "subject_id": vb.get("subject_id"),   # per-sample list (from collate); None if absent
        }
        for key in ("gm", "wm", "csf"):
            if key in pack:
                cached[key] = pack[key].cpu()
        out.append(cached)
    return out


def _sub_setA(pack, n_frames: int, device):
    """Take first n_frames of setA. Lengths are clamped to min(actual, n_frames)
    so that batch-padded zeros at positions >= actual_len are NOT treated as real
    frames by the aggregator (this would otherwise zero-bias the mean / aggregator)."""
    setA = pack["setA"].to(device)
    lenA = pack["lenA"].to(device)
    if n_frames > 0:
        setA = setA[:, :n_frames]
        lenA = torch.clamp(lenA, max=n_frames)
    return setA, lenA


# ------------------------------------------------------------------
# Method runners
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Extra metrics: NRMSE + HFEN (high-frequency error norm)
# ------------------------------------------------------------------

_LOG_KERNEL: Optional[torch.Tensor] = None


def _laplacian_of_gaussian(sigma: float = 1.5, ksize: int = 9, device=None, dtype=None) -> torch.Tensor:
    """5x5 LoG-like kernel for HFEN. Standard MRI denoising metric (Ravishankar 2011)."""
    ax = torch.arange(ksize, dtype=torch.float64) - (ksize - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    s2 = sigma * sigma
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * s2))
    log = ((xx * xx + yy * yy - 2.0 * s2) / (s2 * s2)) * g
    log = log - log.mean()  # zero-mean → high-pass
    k = log.to(dtype=dtype or torch.float32, device=device)
    return k.view(1, 1, ksize, ksize)


def hfen(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, sigma: float = 1.5) -> float:
    """High-Frequency Error Norm = ||LoG(pred) - LoG(target)||_2 / ||LoG(target)||_2 (masked)."""
    global _LOG_KERNEL
    if _LOG_KERNEL is None or _LOG_KERNEL.device != pred.device or _LOG_KERNEL.dtype != pred.dtype:
        _LOG_KERNEL = _laplacian_of_gaussian(sigma=sigma, ksize=9, device=pred.device, dtype=pred.dtype)
    lp_p = F.conv2d(pred,   _LOG_KERNEL, padding=4)
    lp_t = F.conv2d(target, _LOG_KERNEL, padding=4)
    diff = (lp_p - lp_t) * mask
    den  = lp_t * mask
    num = float(diff.pow(2).sum().sqrt().item())
    den_norm = float(den.pow(2).sum().sqrt().clamp_min(1e-8).item())
    return num / den_norm


def nrmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Normalised RMSE (masked): sqrt(mean((p-t)^2)) / range(target inside mask)."""
    diff = (pred - target) * mask
    mse = (diff.pow(2).sum() / mask.sum().clamp_min(1.0)).sqrt()
    t_inside = target[mask.bool()] if mask.bool().any() else target.flatten()
    rng = (t_inside.max() - t_inside.min()).clamp_min(1e-8)
    return float((mse / rng).item())


@torch.no_grad()
def run_ours(model, pack, n_frames, device) -> torch.Tensor:
    setA, lenA = _sub_setA(pack, n_frames, device)
    t1 = pack["t1"].to(device)
    out = model.infer_from_subset(setA, t1, lengths=lenA)
    return out["asl_recon"]


def _center_slice_img(x: torch.Tensor) -> torch.Tensor:
    """2.5D: reduce a [B,K,H,W] frame-mean to its CENTER slice [B,1,H,W] (K=2*ctx+1 for a
    slice window; K==1 = plain 2D, returned unchanged). Keeps baseline preds 1-channel."""
    return x if x.shape[1] <= 1 else x[:, x.shape[1] // 2: x.shape[1] // 2 + 1]


@torch.no_grad()
def run_unet(model, pack, n_frames, device) -> torch.Tensor:
    setA, lenA = _sub_setA(pack, n_frames, device)
    meanA = _center_slice_img(direct_mean_from_frames(setA, lenA))
    if getattr(model, "_t1_concat", False):   # unguarded T1-concat baseline: feed [meanA; T1]
        t1 = _center_slice_img(pack["t1"].to(device))
        meanA = torch.cat([meanA, t1], dim=1)
    return model(meanA)


@torch.no_grad()
def run_naive(pack, n_frames, device) -> torch.Tensor:
    setA, lenA = _sub_setA(pack, n_frames, device)
    return _center_slice_img(direct_mean_from_frames(setA, lenA))


def run_nlm(pack, n_frames, device, args) -> torch.Tensor:
    from skimage.restoration import denoise_nl_means, estimate_sigma
    setA, lenA = _sub_setA(pack, n_frames, device)
    meanA = _center_slice_img(direct_mean_from_frames(setA, lenA))
    out = torch.zeros_like(meanA)
    for b in range(meanA.size(0)):
        arr = meanA[b, 0].cpu().numpy()
        sigma = float(estimate_sigma(arr))
        if sigma <= 0:
            sigma = 1e-3
        d = denoise_nl_means(arr, h=args.nlm_h_factor * sigma, sigma=sigma,
                             patch_size=args.nlm_patch_size,
                             patch_distance=args.nlm_patch_distance,
                             fast_mode=True, channel_axis=None)
        out[b, 0] = torch.from_numpy(d.astype(np.float32)).to(device)
    return out


# ------------------------------------------------------------------
# Checkpoint loaders
# ------------------------------------------------------------------

def load_unet(ckpt_path: str, args, device):
    """Load a trainable single-image baseline (PlainUNet2D or SwinIR2D).

    The architecture is dispatched from the checkpoint's stored ``arch``/``arch_cfg``
    (train_baseline.py); pre-SwinIR checkpoints have no such key and default to
    PlainUNet2D, so this is backward-compatible. Both share the same
    ``forward(meanA) -> pred`` contract used by ``run_unet``.
    """
    tp = Config(args.config).asl_denoiser_train_params
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = st.get("arch", "plainunet")
    cfg = dict(st.get("arch_cfg", {}) or {})
    in_ch = int(cfg.pop("in_ch", 1))
    t1_concat = bool(cfg.pop("t1_concat", False))
    if arch == "swinir":
        from models.swinir_2d import SwinIR2D
        m = SwinIR2D(hw=int(tp.asl_hw), in_ch=in_ch, out_ch=1, **cfg).to(device)
    else:
        m = PlainUNet2D(hw=int(tp.asl_hw), in_ch=in_ch, out_ch=1,
                        base_ch=int(cfg.get("base_ch", args.base_ch)),
                        depth=int(cfg.get("depth", args.depth))).to(device)
    m._t1_concat = t1_concat   # eval must feed [meanA; T1] for the unguarded-concat baseline
    sd = st.get("ema") or st["model"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def load_ours(ckpt_path: str, args, device):
    tp = Config(args.config).asl_denoiser_train_params
    m = ASLT1Denoiser(asl_hw=int(tp.asl_hw), t1_hw=int(tp.t1_hw),
                      in_ch=1, base_ch=int(args.ours_base_ch),
                      depth=int(args.ours_depth)).to(device)
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = st.get("ema") or st["model"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    # Tolerate shape mismatches (e.g., older v18 ckpt has T1 head out_ch=1 vs current 4):
    # drop such keys; the affected sub-module gets random init, which is fine
    # because we only consume ASL recon downstream.
    model_sd = m.state_dict()
    dropped = [k for k in list(sd.keys())
               if k in model_sd and sd[k].shape != model_sd[k].shape]
    if dropped:
        print(f"      [load_ours] dropping {len(dropped)} key(s) with shape mismatch "
              f"(e.g., {dropped[:2]})")
    sd = {k: v for k, v in sd.items() if k not in dropped}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/3] Materialising {args.split} set (seed={args.seed}) ...")
    val_data = materialise_split(cfg, device, split=args.split, max_samples=args.max_samples)
    print(f"      {len(val_data)} samples cached from split '{args.split}'.")

    methods: Dict[str, callable] = {}
    if args.include_naive:
        methods["naive"] = lambda pack, nf: run_naive(pack, nf, device)
    if args.include_nlm:
        methods["NLM"] = lambda pack, nf: run_nlm(pack, nf, device, args)
    if args.vanilla:
        m_van = load_unet(args.vanilla, args, device)
        methods["vanilla_N2N"] = lambda pack, nf, m=m_van: run_unet(m, pack, nf, device)
    if args.sup:
        m_sup = load_unet(args.sup, args, device)
        methods["UNet_supervised"] = lambda pack, nf, m=m_sup: run_unet(m, pack, nf, device)
    if args.n2self:
        m_ns = load_unet(args.n2self, args, device)
        methods["Noise2Self"] = lambda pack, nf, m=m_ns: run_unet(m, pack, nf, device)
    if args.ours:
        m_ours = load_ours(args.ours, args, device)
        methods["ours"] = lambda pack, nf, m=m_ours: run_ours(m, pack, nf, device)
    if args.ours2:
        m_ours2 = load_ours(args.ours2, args, device)
        methods[args.ours2_label] = lambda pack, nf, m=m_ours2: run_ours(m, pack, nf, device)
    if args.ours3:
        m_ours3 = load_ours(args.ours3, args, device)
        methods[args.ours3_label] = lambda pack, nf, m=m_ours3: run_ours(m, pack, nf, device)
    if args.ours4:
        m_ours4 = load_ours(args.ours4, args, device)
        methods[args.ours4_label] = lambda pack, nf, m=m_ours4: run_ours(m, pack, nf, device)

    if not methods:
        raise SystemExit("No methods selected; pass --ours/--vanilla/--sup/--n2self/--include_nlm.")

    print(f"[2/3] Methods: {list(methods.keys())}")
    print(f"      n_frames sweep: {args.n_frames}")

    rows: List[Dict] = []
    fig_dir = os.path.join(args.out_dir, "figures")
    if args.save_figures:
        os.makedirs(fig_dir, exist_ok=True)

    from utils.metrics import gmsd, gm_wm_contrast_error, laplacian_variance

    for nf in args.n_frames:
        for k, pack in enumerate(tqdm(val_data, desc=f"n_frames={nf}")):
            t1 = pack["t1"].to(device)
            union = direct_mean_from_frames(
                torch.cat([pack["setA"].to(device), pack["setB"].to(device)], dim=1),
                (pack["lenA"] + pack["lenB"]).to(device),
            )
            mask = (t1 > 0.05).float()
            gm_dev = pack["gm"].to(device) if "gm" in pack else None
            wm_dev = pack["wm"].to(device) if "wm" in pack else None
            ref_lapvar = laplacian_variance(union, mask)

            preds = {}
            for name, fn in methods.items():
                pred = fn(pack, nf)
                preds[name] = pred
                # masked metrics
                valid = mask.sum().clamp_min(1.0)
                l1 = ((pred - union).abs() * mask).sum() / valid
                psnr_ref, ssim_ref = _compute_psnr_ssim(pred * mask, union * mask)
                nrmse_v = nrmse(pred, union, mask)
                hfen_v  = hfen(pred, union, mask)
                gmsd_v  = gmsd(pred, union, mask)
                if gm_dev is not None and wm_dev is not None:
                    gmwm_err_v = gm_wm_contrast_error(pred, union, gm_dev, wm_dev)
                else:
                    gmwm_err_v = float("nan")
                pred_lapvar = laplacian_variance(pred, mask)
                # ratio < 1 = pred smoother than ref; closer to 1 = sharper match
                lapvar_ratio = pred_lapvar / max(ref_lapvar, 1e-8)
                rows.append({
                    "sample": k, "n_frames": nf, "method": name,
                    "l1": float(l1.item()),
                    "psnr": float(psnr_ref), "ssim": float(ssim_ref),
                    "nrmse": nrmse_v, "hfen": hfen_v,
                    "gmsd": gmsd_v, "gmwm_err": gmwm_err_v,
                    "lapvar_pred": pred_lapvar, "lapvar_ref": ref_lapvar,
                    "lapvar_ratio": lapvar_ratio,
                })

            if args.save_figures and k < 4 and nf == args.n_frames[0]:
                try:
                    n_panels = 2 + len(preds)
                    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))
                    axes[0].imshow(t1[0, 0].cpu().numpy(), cmap="gray")
                    axes[0].set_title("T1w", fontsize=8); axes[0].axis("off")
                    axes[1].imshow(union[0, 0].cpu().numpy(), cmap="gray")
                    axes[1].set_title("12-NEX ref", fontsize=8); axes[1].axis("off")
                    for i, (name, p) in enumerate(preds.items(), start=2):
                        axes[i].imshow(p[0, 0].cpu().numpy(), cmap="gray")
                        axes[i].set_title(name, fontsize=8); axes[i].axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_dir, f"sample{k:03d}_nf{nf}.png"),
                                dpi=80, bbox_inches="tight")
                    plt.close(fig)
                except Exception:
                    plt.close("all")

    # --------------------------------------------------------------
    # Output: long CSV
    # --------------------------------------------------------------
    long_csv = os.path.join(args.out_dir, "results_long.csv")
    fields = ["sample", "n_frames", "method",
              "l1", "psnr", "ssim", "nrmse", "hfen",
              "gmsd", "gmwm_err", "lapvar_pred", "lapvar_ref", "lapvar_ratio"]
    with open(long_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"[3/3] Wrote {long_csv}")

    # Summary: method × n_frames × {l1 mean±std, psnr, ssim}
    by = {}
    for r in rows:
        key = (r["method"], r["n_frames"])
        by.setdefault(key, []).append(r)
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_frames",
                    "l1_mean", "l1_std",
                    "psnr_mean", "psnr_std",
                    "ssim_mean", "ssim_std",
                    "nrmse_mean", "nrmse_std",
                    "hfen_mean", "hfen_std",
                    "gmsd_mean", "gmsd_std",
                    "gmwm_err_mean", "gmwm_err_std",
                    "lapvar_ratio_mean", "lapvar_ratio_std",
                    "n"])
        for (m, nf), lst in sorted(by.items()):
            arr = lambda key: np.array([r[key] for r in lst], dtype=float)
            l1, ps, ss = arr("l1"), arr("psnr"), arr("ssim")
            nr, hf = arr("nrmse"), arr("hfen")
            gm_, gw = arr("gmsd"), arr("gmwm_err")
            lr = arr("lapvar_ratio")
            # GM/WM may be NaN when PV maps weren't loaded; use nan-safe stats
            def ms(a):
                a = a[~np.isnan(a)]
                if len(a) == 0:
                    return float("nan"), float("nan")
                return a.mean(), a.std()
            gw_m, gw_s = ms(gw)
            w.writerow([m, nf,
                        f"{l1.mean():.5f}", f"{l1.std():.5f}",
                        f"{ps.mean():.3f}", f"{ps.std():.3f}",
                        f"{ss.mean():.4f}", f"{ss.std():.4f}",
                        f"{nr.mean():.5f}", f"{nr.std():.5f}",
                        f"{hf.mean():.5f}", f"{hf.std():.5f}",
                        f"{gm_.mean():.5f}", f"{gm_.std():.5f}",
                        f"{gw_m:.5f}", f"{gw_s:.5f}",
                        f"{lr.mean():.4f}", f"{lr.std():.4f}",
                        len(lst)])
    print(f"      Wrote {summary_csv}")

    # Paired Wilcoxon: ours vs each baseline (per n_frames, l1 metric)
    if "ours" in methods:
        try:
            from scipy.stats import wilcoxon
            wilcox_csv = os.path.join(args.out_dir, "wilcoxon_ours_vs_others.csv")
            with open(wilcox_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["baseline", "n_frames", "metric", "stat", "p_value", "n"])
                for nf in args.n_frames:
                    ours = sorted([r for r in rows if r["method"] == "ours" and r["n_frames"] == nf],
                                  key=lambda r: r["sample"])
                    for name in methods:
                        if name == "ours":
                            continue
                        bl = sorted([r for r in rows if r["method"] == name and r["n_frames"] == nf],
                                    key=lambda r: r["sample"])
                        for metric in ("l1", "psnr", "ssim", "nrmse", "hfen", "gmsd", "gmwm_err", "lapvar_ratio"):
                            a = np.array([r[metric] for r in ours])
                            b = np.array([r[metric] for r in bl])
                            try:
                                stat, pv = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                                w.writerow([name, nf, metric, f"{stat:.3f}", f"{pv:.3e}", len(a)])
                            except Exception as e:
                                w.writerow([name, nf, metric, "NA", str(e), len(a)])
            print(f"      Wrote {wilcox_csv}")
        except ImportError:
            print("      scipy not available — skipped Wilcoxon test.")


if __name__ == "__main__":
    main()
