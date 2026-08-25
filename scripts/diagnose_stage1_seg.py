# -*- coding: utf-8 -*-
"""diagnose_stage1_seg.py — is the poor T1 PV quality a LABEL problem or a stage-1
UNDERFIT? Runs the trained stage-1 T1 branch EXACTLY as stage-2 sees it (EMA weights,
same val split) and compares its predicted 4-class PV against the FSL PV targets.

Faithful to the pipeline:
  * builds T1Branch from the checkpoint's own base_ch/depth/t1_task/seg_softmax;
  * loads `state["ema"] or state["model"]` (what runner._init_t1_from_ckpt loads);
  * uses get_asl_2d_loaders(...) val split (cache_rate=0, no cohort preload);
  * PV via losses.pv_seg_probs(seg_softmax); brain = (gm+wm+csf) like train_t1.

Reports, pooled over the scored val slices:
  * per-class brain-masked L1 (GM/WM/CSF) + overall (= train_t1's val/seg_brain metric);
  * per-class soft-Dice (2*Σpt / (Σp+Σt) over brain) — pooled, so thin-CSF collapse shows;
  * boundary gradient-L1 (pred vs target gradients) — high ⇒ blur/under-capacity;
and saves side-by-side panels (T1 | pred GM/WM/CSF/BG | FSL GT | |pred−GT| error).

Read:
  * CSF-Dice ≪ GM/WM-Dice, or grad-L1 large vs L1  ⇒ boundary / thin-CSF ⇒ CAPACITY/loss
    (fix stage-1: more T1-branch capacity, tune seg loss) — labels are fine, keep them.
  * all classes uniformly off, pred blurry everywhere ⇒ global UNDERFIT (train longer / lr).
  * pred ≈ FSL but BOTH look wrong ⇒ it's the LABELS (regenerate with a public model).

Usage (local, monai env — T1Branch is plain conv, no mamba needed):
  python scripts/diagnose_stage1_seg.py \
    --ckpt   G:/training/ASL_dmvae/logs/stage1_t1seg_seed42/checkpoints/latest.pth \
    --config env/local/configs/win_asl_2d_home_v37.yml \
    --out_dir G:/training/ASL_dmvae/logs/stage1_t1seg_seed42/seg_diagnosis \
    --max_batches 60 --n_panels 8
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from config.conf_data import Config
from dataio.dataloaders import get_asl_2d_loaders
from losses.asl_n2n_loss import pv_seg_probs, _fd_grad
from models.t1_branch import T1Branch


def parse_args():
    p = argparse.ArgumentParser("stage-1 T1 PV segmentation diagnosis")
    p.add_argument("--ckpt", required=True, help="stage-1 T1Branch checkpoint (latest.pth = what stage-2 loads)")
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max_batches", type=int, default=60, help="0 = all val batches")
    p.add_argument("--n_panels", type=int, default=8)
    return p.parse_args()


def load_t1branch(ckpt_path, cfg, device):
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    tp = cfg.asl_denoiser_train_params
    base_ch = int(st.get("base_ch", 32)); depth = int(st.get("depth", 4))
    seg_softmax = bool(st.get("seg_softmax", True))
    task = str(st.get("t1_task", "seg"))
    if task != "seg":
        raise SystemExit(f"checkpoint t1_task={task!r} — this diagnosis is for the seg prior")
    model = T1Branch(hw=int(tp.t1_hw), in_ch=1, base_ch=base_ch, depth=depth,
                     skip_dropout=0.0, use_wavelet=False, task="seg").to(device)
    # EXACTLY what runner._init_t1_from_ckpt loads: EMA (fallback model), t1_encoder/decoder.
    sd = st.get("ema") or st["model"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    target = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in target and target[k].shape == v.shape:
            target[k] = v.to(target[k].device); loaded += 1
    model.load_state_dict(target)
    frac = loaded / max(len(target), 1)
    print(f"[load] {loaded}/{len(target)} weights (frac={frac:.2f}) from {'EMA' if 'ema' in st else 'model'}; "
          f"base_ch={base_ch} depth={depth} seg_softmax={seg_softmax} step={st.get('step','?')}")
    if frac < 0.9:
        print("[load][WARN] <90% weights matched — arch mismatch? results may be meaningless.")
    model.eval()
    return model, seg_softmax


def _mask_l1(p, t, brain):   # [B,1,H,W]
    return float(((p - t).abs() * brain).sum().item()), float(brain.sum().item())


def _dice_terms(p, t, brain):   # soft-Dice numerator / denominator over brain
    p = p * brain; t = t * brain
    return float(2.0 * (p * t).sum().item()), float((p.sum() + t.sum()).item())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(args.config)
    tp = cfg.asl_denoiser_train_params

    model, seg_softmax = load_t1branch(args.ckpt, cfg, device)
    print(f"[data] building '{args.split}' loader (cache_rate=0, lazy) ...")
    loaders = get_asl_2d_loaders(cfg, modes=["train", "val"],
                                 asl_hw=tp.asl_hw, asl_z=tp.asl_z, t1_hw=tp.t1_hw, t1_z=tp.t1_z,
                                 cache_rate=0.0)
    loader = loaders.get(args.split) or loaders["val"]

    classes = ["GM", "WM", "CSF"]
    l1_num = {c: 0.0 for c in classes}; l1_den = {c: 0.0 for c in classes}
    dice_n = {c: 0.0 for c in classes}; dice_d = {c: 0.0 for c in classes}
    grad_num = 0.0; grad_den = 0.0
    n_slices = 0; saved = 0

    for bi, vb in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        t1 = vb["t1"].to(device)
        gm = vb["gm"].to(device); wm = vb["wm"].to(device); csf = vb["csf"].to(device)
        brain = (gm + wm + csf).clamp(0.0, 1.0)
        with torch.no_grad():
            out = model(t1)
            p = pv_seg_probs(out["t1_seg"], softmax=seg_softmax)   # [B,4,H,W] GM,WM,CSF,BG
        preds = {"GM": p[:, 0:1], "WM": p[:, 1:2], "CSF": p[:, 2:3]}
        tgts = {"GM": gm, "WM": wm, "CSF": csf}
        for c in classes:
            num, den = _mask_l1(preds[c], tgts[c], brain); l1_num[c] += num; l1_den[c] += den
            dn, dd = _dice_terms(preds[c], tgts[c], brain); dice_n[c] += dn; dice_d[c] += dd
        # boundary gradient-L1 (GM/WM/CSF stacked), like train_t1.validate
        p3 = torch.cat([preds["GM"], preds["WM"], preds["CSF"]], dim=1)
        t3 = torch.cat([gm, wm, csf], dim=1)
        gpx, gpy = _fd_grad(p3); gtx, gty = _fd_grad(t3)
        bx, by = brain[..., :, 1:], brain[..., 1:, :]
        grad_num += float(((gpx - gtx).abs() * bx).sum().item()) + float(((gpy - gty).abs() * by).sum().item())
        grad_den += float(bx.sum().item()) * 3.0 + float(by.sum().item()) * 3.0
        n_slices += t1.shape[0]

        if saved < args.n_panels:
            _save_panel(t1, preds, tgts, os.path.join(args.out_dir, f"panel_{saved:02d}.png"), seg_softmax)
            saved += 1

    # pooled metrics
    l1 = {c: (l1_num[c] / max(l1_den[c], 1.0)) for c in classes}
    dice = {c: (dice_n[c] / max(dice_d[c], 1e-6)) for c in classes}
    overall_l1 = sum(l1_num[c] for c in classes) / max(sum(l1_den[c] for c in classes), 1.0)
    grad_l1 = grad_num / max(grad_den, 1.0)

    # Print numbers FIRST (so they always show even if the file write hiccups).
    print("\n=== stage-1 seg diagnosis (pooled over %d slices) ===" % n_slices)
    for c in classes:
        print(f"  {c:3s}  brain-L1={l1[c]:.4f}  soft-Dice={dice[c]:.4f}")
    print(f"  overall brain-L1={overall_l1:.4f}   boundary grad-L1={grad_l1:.4f}")

    md = os.path.join(args.out_dir, "stage1_seg_diagnosis.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Stage-1 T1->PV segmentation diagnosis (pred vs FSL target)\n\n")
        f.write(f"Split=`{args.split}`, {n_slices} slices, EMA weights (what stage-2 freezes). "
                f"brain=(GM+WM+CSF). Metrics pooled over slices.\n\n")
        f.write("| class | brain-L1 (lower better) | soft-Dice (higher better) |\n|---|---|---|\n")
        for c in classes:
            f.write(f"| {c} | {l1[c]:.4f} | {dice[c]:.4f} |\n")
        f.write(f"| **overall** | **{overall_l1:.4f}** | — |\n")
        f.write(f"\nboundary gradient-L1 (pred vs target grads): **{grad_l1:.4f}**\n\n")
        f.write("**Read:** overall-L1 ~ the training `val/seg_brain` log -- compare. "
                "CSF-Dice << GM/WM-Dice or a large grad-L1 => boundary/thin-CSF => CAPACITY/loss "
                "(fix stage-1, keep FSL labels). Uniformly high L1 + blurry panels => global "
                "UNDERFIT (train longer / lr). Pred~FSL but both look wrong in the panels => the "
                "LABELS are the problem (regenerate with a public model).\n")
    print(f"[done] {md} + {saved} panels -> {args.out_dir}")


def _save_panel(t1, preds, tgts, path, seg_softmax):
    try:
        t1_np = t1[0, 0].cpu().numpy()
        gm = tgts["GM"][0, 0].cpu().numpy(); wm = tgts["WM"][0, 0].cpu().numpy(); csf = tgts["CSF"][0, 0].cpu().numpy()
        bg = (1.0 - (gm + wm + csf)).clip(0, 1)
        pg = preds["GM"][0, 0].cpu().numpy(); pw = preds["WM"][0, 0].cpu().numpy(); pc = preds["CSF"][0, 0].cpu().numpy()
        pbg = (1.0 - (pg + pw + pc)).clip(0, 1)
        pred = [pg, pw, pc, pbg]; gt = [gm, wm, csf, bg]; titles = ["GM", "WM", "CSF", "BG"]
        fig, ax = plt.subplots(4, 4, figsize=(12, 12))
        ax[0, 0].imshow(t1_np, cmap="gray"); ax[0, 0].set_title("T1 input", fontsize=8)
        for c in range(1, 4):
            ax[0, c].axis("off")
        for i in range(4):
            ax[1, i].imshow(pred[i], cmap="gray", vmin=0, vmax=1); ax[1, i].set_title(f"pred {titles[i]}", fontsize=8)
            ax[2, i].imshow(gt[i], cmap="gray", vmin=0, vmax=1); ax[2, i].set_title(f"FSL {titles[i]}", fontsize=8)
            ax[3, i].imshow(np.abs(pred[i] - gt[i]), cmap="magma", vmin=0, vmax=0.5)
            ax[3, i].set_title(f"|Δ| {titles[i]}", fontsize=8)
        for a in ax.ravel():
            a.axis("off")
        fig.tight_layout(); fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        plt.close("all"); print(f"[panel] failed: {e!r}")


if __name__ == "__main__":
    main()
