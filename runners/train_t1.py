# -*- coding: utf-8 -*-
"""Stage-1 trainer: T1 branch only.

Loss = w_recon * L1(t1_recon, t1) + w_seg * soft_cross_entropy(t1_seg, GM/WM/CSF).
PV-invalid slices are filtered upstream by ASLTwoSetDataset2DFlat.

Resulting checkpoint can be loaded by the main runner via --init_t1_from / --freeze_t1.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from functools import partial
from typing import Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.conf_data import Config
from dataio.dataloaders import get_asl_2d_loaders
from losses.asl_n2n_loss import (pv_l1_loss_4cls, pv_l1_loss_4cls_sharp,
                                  pv_seg_probs, _fd_grad)
from models.t1_branch import T1Branch
from runners.asl_t1_guided_runner_dmvae_n2n import (
    _FallbackEMAModel,
    _compute_psnr_ssim,
    set_seed,
)

try:
    from utils.training_utils import EMAModel  # type: ignore
except Exception:
    EMAModel = None


# Every stage-1 val metric that can drive a best_<name>.pth, with its optimisation
# direction. seg: seg_loss + per-class/overall brain-L1 (lower=better) + per-class/mean
# soft-Dice (higher=better) + boundary grad-L1 (lower=better). recon: recon_l1.
_METRIC_DIR = {
    "seg_loss": "min", "l1": "min", "l1_gm": "min", "l1_wm": "min", "l1_csf": "min",
    "dice_gm": "max", "dice_wm": "max", "dice_csf": "max", "dice_mean": "max",
    "grad": "min", "recon_l1": "min",
}


def parse_args():
    p = argparse.ArgumentParser("Stage-1 T1 branch trainer")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--exp", type=str, default="./exp")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", type=str, default="info")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--save_images", action="store_true")
    p.add_argument("--log_images", type=int, default=8)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--early_stop_min_evals", type=int, default=30)

    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--skip_dropout", type=float, default=0.3)
    p.add_argument("--use_wavelet", action="store_true",
                   help="Replace stride-2 conv down/up with DWT/IDWT (Haar) for "
                        "information-preserving multiscale processing.")

    p.add_argument("--t1_task", type=str, default="recon", choices=["seg", "recon"],
                   help="Stage-1 pretext: 'recon' = reconstruct the T1 image (autoencoder, "
                        "appearance-preserving features, no labels — DEFAULT); 'seg' = "
                        "4-class GM/WM/CSF/BG PV segmentation (tissue-abstracted features). "
                        "Stage-2 must load with a matching --t1_task.")
    p.add_argument("--w_seg", type=float, default=1.0)
    p.add_argument("--seg_loss", type=str, default="pv_l1_4cls",
                   choices=["pv_l1_4cls", "pv_l1_4cls_sharp"],
                   help="pv_l1_4cls = 4-class (GM/WM/CSF/BG) sigmoid+L1, BG derived, full image. "
                        "pv_l1_4cls_sharp = pv_l1_4cls + soft-Dice + partition (brain-weighted) — "
                        "tightens boundaries via class competition (no gradient matching, which "
                        "would fight soft PV labels). "
                        "(The old 3-class pv_l1 / soft_ce options were removed: the seg head is "
                        "fixed at 4 channels, so 3-class losses raised a channel mismatch.)")
    # pv_l1_4cls_sharp component weights.
    p.add_argument("--w_seg_dice", type=float, default=0.5,
                   help="pv_l1_4cls_sharp: weight of the soft-Dice(GM/WM/CSF) term.")
    p.add_argument("--w_seg_part", type=float, default=0.1,
                   help="pv_l1_4cls_sharp: weight of the partition penalty |sum_c p_c - 1|.")
    p.add_argument("--seg_brain_boost", type=float, default=3.0,
                   help="pv_l1_4cls_sharp: extra L1 weight inside brain (weight = 1 + boost*brain); "
                        "focuses capacity on tissue rather than the dominant BG.")
    p.add_argument("--seg_softmax", action="store_true",
                   help="Predict the 4 PV classes with SOFTMAX (sum-to-1 simplex, matches the PV "
                        "target + enforces class competition -> rescues thin CSF, sharper) instead "
                        "of independent sigmoids. Drops the (now-redundant) partition penalty. "
                        "Applies to loss, selection metric, and val images consistently.")
    p.add_argument("--seg_sel_grad", type=float, default=0.0,
                   help="Model selection: add seg_sel_grad * brain gradient-L1 to the brain L1. "
                        "Default 0 = pure brain L1 (val/seg_brain_grad is still logged as a "
                        "diagnostic). >0 re-enables gradient-aware selection.")
    p.add_argument("--best_criterion", type=str, default="seg",
                   choices=["seg"],
                   help="Model selection metric: lower-better seg loss (segmentation-only branch).")

    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--max_steps", type=int, default=0,
                   help="Override config max_steps. 0 = use config value.")
    return p.parse_args()


def make_loggers(exp_root: str, run_name: str, verbose: str) -> SummaryWriter:
    log_dir = os.path.join(exp_root, "logs", run_name)
    tb_dir = os.path.join(exp_root, "tensorboard", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    level = getattr(logging, verbose.upper(), logging.INFO)
    fmt = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    h1, h2 = logging.StreamHandler(), logging.FileHandler(os.path.join(log_dir, "stdout.txt"))
    for h in (h1, h2):
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h1); root.addHandler(h2)
    root.setLevel(level)
    return SummaryWriter(log_dir=tb_dir)


def _masked_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)


class T1Runner:
    def __init__(self, args):
        self.args = args
        self.cfg = Config(args.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Stage-1 T1 only | Device: {self.device}")

        self.log_dir = os.path.join(args.exp, "logs", args.name)
        self.ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        self.val_img_dir = os.path.join(self.log_dir, "val_images")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if args.save_images:
            os.makedirs(self.val_img_dir, exist_ok=True)

        tp = self.cfg.asl_denoiser_train_params
        self.loaders = get_asl_2d_loaders(
            self.cfg, modes=["train", "val"],
            asl_hw=tp.asl_hw, asl_z=tp.asl_z, t1_hw=tp.t1_hw, t1_z=tp.t1_z,
        )
        self.train_loader = self.loaders["train"]
        self.val_loader = self.loaders["val"]

        self.t1_task = str(getattr(args, "t1_task", "seg"))
        self.model = T1Branch(
            hw=int(tp.t1_hw), in_ch=1,
            base_ch=int(args.base_ch), depth=int(args.depth),
            skip_dropout=float(args.skip_dropout),
            use_wavelet=bool(getattr(args, "use_wavelet", False)),
            task=self.t1_task,
        ).to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay,
        )
        max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(tp.max_steps)
        if args.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps, eta_min=args.lr_min)
        else:
            self.scheduler = None

        ema_cls = EMAModel if EMAModel is not None else _FallbackEMAModel
        self.ema = ema_cls(self.model, update_after_step=0, inv_gamma=1.0,
                           power=2 / 3, min_value=0.0, max_value=0.9999, device=self.device)

        self.global_step = 0
        # Best by configured criterion. For seg_ce, best = lowest (we negate
        # internally so the same `>` comparison logic works).
        self.best_val = -float("inf")
        self.no_improve_count = 0
        self.eval_count = 0
        self._best_higher_better = False  # always seg (lower better)
        # Per-metric best tracker: name -> (best_value, step). Every val metric gets
        # its own best_<name>.pth so the operating point can be chosen post-hoc
        # (e.g. best_dice_csf vs best_l1) without re-running the diagnosis.
        self.best_metrics: Dict[str, Tuple[float, int]] = {}
        if args.seg_loss == "pv_l1_4cls":
            self._seg_fn = pv_l1_loss_4cls
        else:  # pv_l1_4cls_sharp — the only other (4-class) choice
            self._seg_fn = partial(pv_l1_loss_4cls_sharp,
                                   w_dice=float(args.w_seg_dice), w_part=float(args.w_seg_part),
                                   brain_boost=float(args.seg_brain_boost),
                                   softmax=bool(getattr(args, "seg_softmax", False)))
        self.seg_softmax = bool(getattr(args, "seg_softmax", False))

        if args.resume:
            self._try_resume()

    def _unwrap(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _ckpt(self, tag: str) -> str:
        return os.path.join(self.ckpt_dir, f"{tag}.pth")

    def _try_resume(self):
        latest = self._ckpt("latest")
        if not os.path.exists(latest):
            return
        st = torch.load(latest, map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        self.global_step = st.get("step", 0)
        self.best_val = st.get("best_val", -float("inf"))
        if "optimizer" in st:
            self.optimizer.load_state_dict(st["optimizer"])
        if "ema" in st:
            self.ema.ema_state = {k: v.to(self.device) for k, v in st["ema"].items()}
            self.ema.optimization_step = st.get("ema_optimization_step", 0)
        if "scheduler" in st and self.scheduler is not None:
            self.scheduler.load_state_dict(st["scheduler"])
        logging.info(f"Resumed (step={self.global_step}, best psnr={self.best_val:.2f})")

    def _save(self, tag: str, light: bool = False):
        # light=True drops optimizer/scheduler state (best_<metric>.pth are for
        # inference / --init_t1_from, never for --resume) → ~2× smaller, less I/O
        # when many best_<name>.pth are written per eval. model+ema is all
        # _init_t1_from_ckpt / the diagnosis ever load.
        payload = {
            "model": self.model.state_dict(),
            "step": self.global_step,
            "best_val": self.best_val,
            "ema": {k: v.detach().cpu() for k, v in self.ema.ema_state.items()},
            "ema_optimization_step": getattr(self.ema, "optimization_step", 0),
            "optimizer": None if light else self.optimizer.state_dict(),
            "scheduler": None if light or self.scheduler is None else self.scheduler.state_dict(),
            "stage": "t1",
            # Pretext + head config, so stage-2 (--init_t1_from) can assert a match
            # instead of silently loading encoder-only when --t1_task disagrees
            # (recon=1ch vs seg=4ch head → shape-mismatched decoder skipped → frozen
            # random head). base_ch/depth let inference sanity-check the geometry.
            "t1_task": self.t1_task,
            "seg_softmax": bool(getattr(self.args, "seg_softmax", False)),
            "base_ch": int(self.args.base_ch),
            "depth": int(self.args.depth),
        }
        torch.save(payload, self._ckpt(tag))

    @staticmethod
    def _to_dev(batch, device):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _forward_train(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        t1 = batch["t1"]
        out = self._unwrap()(t1)
        if self.t1_task == "recon":
            # Autoencoder pretext: reconstruct the T1 image (full-image L1).
            loss = _masked_l1(out["t1_recon"], t1, torch.ones_like(t1))
            return loss, {"loss": float(loss.detach().item()),
                          "loss_recon": float(loss.detach().item())}
        gm, wm, csf = batch["gm"], batch["wm"], batch["csf"]
        # Brain occupancy from the PV target (soft), NOT t1>0.05: CSF/ventricles are
        # T1-dark, so an intensity threshold under-weights them; (GM+WM+CSF) gives
        # every tissue voxel — incl. partial-volume CSF edges — proper brain-weighted
        # supervision. Out-of-brain BG is still supervised at weight 1 (occupancy=0).
        brain = (gm + wm + csf).clamp(0.0, 1.0)
        loss_seg = self._seg_fn(out["t1_seg"], gm, wm, csf, brain)
        total = self.args.w_seg * loss_seg
        return total, {
            "loss":     float(total.detach().item()),
            "loss_seg": float(loss_seg.detach().item()),
        }

    @torch.no_grad()
    def _predict_with_ema(self, t1: Tensor) -> Dict[str, Tensor]:
        m = self._unwrap(); m.eval()
        self.ema.store(m); self.ema.copy_to(m)
        out = m(t1)
        self.ema.restore(m)
        return out

    def _save_recon_panel(self, t1: Tensor, out: Dict[str, Tensor], idx: int) -> int:
        try:
            inp = t1[0, 0].cpu().numpy()
            rec = out["t1_recon"][0, 0].cpu().numpy()
            dif = (out["t1_recon"][0, 0] - t1[0, 0]).abs().cpu().numpy()
            fig, ax = plt.subplots(1, 3, figsize=(9, 3))
            ax[0].imshow(inp, cmap="gray"); ax[0].set_title("T1 in", fontsize=8); ax[0].axis("off")
            ax[1].imshow(rec, cmap="gray"); ax[1].set_title("T1 recon", fontsize=8); ax[1].axis("off")
            ax[2].imshow(dif, cmap="magma"); ax[2].set_title("|diff|", fontsize=8); ax[2].axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(self.val_img_dir, f"step{self.global_step}_val{idx}.png"),
                        dpi=80, bbox_inches="tight")
            plt.close(fig)
            return 1
        except Exception as e:
            plt.close("all"); logging.warning(f"save img failed: {e}"); return 0

    def _save_seg_panel(self, t1: Tensor, p4, gm, wm, csf, idx: int) -> int:
        try:
            p = p4[0].cpu().numpy()                                          # [4,H,W] pred GM/WM/CSF/BG
            gm_np, wm_np, csf_np = gm[0, 0].cpu().numpy(), wm[0, 0].cpu().numpy(), csf[0, 0].cpu().numpy()
            bg_np = (1.0 - (gm_np + wm_np + csf_np)).clip(0, 1)
            gt = [gm_np, wm_np, csf_np, bg_np]; titles = ["GM", "WM", "CSF", "BG"]
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            axes[0, 0].imshow(t1[0, 0].cpu().numpy(), cmap="gray")
            axes[0, 0].set_title("T1 input", fontsize=8); axes[0, 0].axis("off")
            for c in range(1, 4):
                axes[0, c].axis("off")
            for ci in range(4):
                axes[1, ci].imshow(p[ci], cmap="gray", vmin=0, vmax=1)
                axes[1, ci].set_title(f"pred {titles[ci]}", fontsize=8); axes[1, ci].axis("off")
                axes[2, ci].imshow(gt[ci], cmap="gray", vmin=0, vmax=1)
                axes[2, ci].set_title(f"GT {titles[ci]}", fontsize=8); axes[2, ci].axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(self.val_img_dir, f"step{self.global_step}_val{idx}.png"),
                        dpi=80, bbox_inches="tight")
            plt.close(fig)
            return 1
        except Exception as e:
            plt.close("all"); logging.warning(f"save img failed: {e}"); return 0

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Compute the FULL stage-1 val-metric suite, POOLED over the val split
        (numerator/denominator summed across all slices, so thin-CSF Dice does not
        collapse the way a per-batch mean would). Every returned key is a candidate
        operating point that gets its own best_<key>.pth (min/max via _METRIC_DIR):
          seg:  seg_loss · l1(+l1_gm/wm/csf) · dice_gm/wm/csf/dice_mean · grad
          recon: recon_l1
        Also dumps the first --log_images val panels (unchanged)."""
        self.model.eval()
        saved = 0
        if self.t1_task == "recon":
            rnum = rden = 0.0
            for vb in tqdm(self.val_loader, desc=f"Val (step={self.global_step})", dynamic_ncols=True):
                vb = self._to_dev(vb, self.device); t1 = vb["t1"]
                out = self._predict_with_ema(t1)
                rnum += float((out["t1_recon"] - t1).abs().sum().item())
                rden += float(t1.numel())
                if self.args.save_images and saved < self.args.log_images:
                    saved += self._save_recon_panel(t1, out, saved)
            return {"recon_l1": rnum / max(rden, 1.0)}

        classes = ["gm", "wm", "csf"]
        l1_num = {c: 0.0 for c in classes}; l1_den = 0.0
        d_num = {c: 0.0 for c in classes}; d_den = {c: 0.0 for c in classes}
        g_num = g_den = seg_sum = 0.0; n = 0
        for vb in tqdm(self.val_loader, desc=f"Val (step={self.global_step})", dynamic_ncols=True):
            vb = self._to_dev(vb, self.device)
            t1, gm, wm, csf = vb["t1"], vb["gm"], vb["wm"], vb["csf"]
            out = self._predict_with_ema(t1)
            # Soft PV brain occupancy — matches the training mask (see _forward_train).
            brain = (gm + wm + csf).clamp(0.0, 1.0)
            seg_sum += float(self._seg_fn(out["t1_seg"], gm, wm, csf, brain).item()); n += 1
            p4 = pv_seg_probs(out["t1_seg"], softmax=self.seg_softmax)          # [B,4,H,W]
            p3 = p4[:, :3]; tgt3 = torch.cat([gm, wm, csf], dim=1)              # [B,3,H,W]
            l1_den += float(brain.sum().item())
            for i, c in enumerate(classes):
                pc, tc = p3[:, i:i + 1], tgt3[:, i:i + 1]
                l1_num[c] += float(((pc - tc).abs() * brain).sum().item())     # brain-masked GM/WM/CSF L1
                pm, tm = pc * brain, tc * brain                                 # soft-Dice (pooled)
                d_num[c] += float(2.0 * (pm * tm).sum().item())
                d_den[c] += float((pm.sum() + tm.sum()).item())
            gpx, gpy = _fd_grad(p3); gtx, gty = _fd_grad(tgt3)                  # boundary grad-L1 (pooled)
            bx, by = brain[..., :, 1:], brain[..., 1:, :]
            g_num += float(((gpx - gtx).abs() * bx).sum().item()) + float(((gpy - gty).abs() * by).sum().item())
            g_den += float(bx.sum().item()) * 3.0 + float(by.sum().item()) * 3.0
            if self.args.save_images and saved < self.args.log_images:
                saved += self._save_seg_panel(t1, p4, gm, wm, csf, saved)
        l1_den = max(l1_den, 1.0)
        met = {
            "seg_loss": seg_sum / max(n, 1),
            "l1": sum(l1_num.values()) / (l1_den * 3.0),
            "grad": g_num / max(g_den, 1.0),
        }
        for c in classes:
            met[f"l1_{c}"] = l1_num[c] / l1_den
            met[f"dice_{c}"] = d_num[c] / max(d_den[c], 1e-6)
        met["dice_mean"] = sum(met[f"dice_{c}"] for c in classes) / 3.0
        return met

    def _update_best_metrics(self, metrics: Dict[str, float]) -> list:
        """Save best_<name>.pth for every metric that improved this eval (light ckpts,
        model+ema only). Returns the list of newly-best metric names."""
        newly = []
        for name, val in metrics.items():
            direction = _METRIC_DIR.get(name, "min")
            prev = self.best_metrics.get(name)
            better = prev is None or (val < prev[0] if direction == "min" else val > prev[0])
            if better:
                self.best_metrics[name] = (float(val), int(self.global_step))
                self._save(f"best_{name}", light=True)
                newly.append(name)
        # persist the winners map so the operating point is choosable without re-eval
        with open(os.path.join(self.ckpt_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({k: {"value": v[0], "step": v[1]} for k, v in self.best_metrics.items()},
                      f, indent=2, sort_keys=True)
        return newly

    def train(self, writer: SummaryWriter):
        max_steps = int(self.args.max_steps) if int(self.args.max_steps) > 0 \
                     else int(self.cfg.asl_denoiser_train_params.max_steps)
        eval_every = int(self.cfg.asl_denoiser_train_params.eval_every)

        while self.global_step < max_steps:
            self.global_step += 1
            pbar = tqdm(self.train_loader, desc=f"Train ({self.global_step}/{max_steps})", dynamic_ncols=True)
            self.model.train()
            last = {}
            for batch in pbar:
                batch = self._to_dev(batch, self.device)
                loss, stats = self._forward_train(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.ema.step(self._unwrap())
                last = stats
                pbar.set_postfix(loss=f"{stats['loss']:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

            writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            for k, v in last.items():
                writer.add_scalar(f"train/{k}", v, self.global_step)

            if self.global_step % eval_every == 0 or self.global_step >= max_steps:
                metrics = self.validate()
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, self.global_step)

                self._save("latest")
                if self.args.save_every > 0 and self.global_step % self.args.save_every == 0:
                    self._save(f"step{self.global_step:06d}")

                # Compute + save a best_<name>.pth for EVERY val metric (so the
                # operating point — e.g. best_dice_csf vs best_l1 — is choosable
                # post-hoc without re-running the diagnosis).
                newly = self._update_best_metrics(metrics)

                self.eval_count += 1
                patience = int(self.args.early_stop_patience)
                # `best.pth` (the one stage-2 --init_t1_from loads by default) + early
                # stop track the PRIMARY selection criterion: brain-L1 + seg_sel_grad*grad
                # (recon: recon_l1). This is independent of the per-metric best_*.pth above.
                if self.t1_task == "recon":
                    sel_v = metrics["recon_l1"]
                    brief = f"recon_l1={sel_v:.4f}"
                else:
                    sel_v = metrics["l1"] + float(self.args.seg_sel_grad) * metrics["grad"]
                    brief = (f"l1={metrics['l1']:.4f} dice(gm/wm/csf)="
                             f"{metrics['dice_gm']:.3f}/{metrics['dice_wm']:.3f}/{metrics['dice_csf']:.3f} "
                             f"grad={metrics['grad']:.4f}")
                score = -sel_v
                if score > self.best_val:
                    self.best_val = score
                    self.no_improve_count = 0
                    self._save("best")
                    logging.info(f"[BEST] step={self.global_step} sel={sel_v:.4f} | {brief} "
                                 f"| new best_*: {','.join(newly) if newly else '-'}")
                else:
                    self.no_improve_count += 1
                    tail = f" (best sel {-self.best_val:.4f}"
                    if patience > 0:
                        tail += f", no-improve {self.no_improve_count}/{patience}"
                    tail += ")"
                    logging.info(f"[VAL ] step={self.global_step} sel={sel_v:.4f} | {brief}{tail} "
                                 f"| new best_*: {','.join(newly) if newly else '-'}")

                if (patience > 0
                    and self.eval_count >= int(self.args.early_stop_min_evals)
                    and self.no_improve_count >= patience):
                    logging.info(f"[EARLY STOP] step={self.global_step}, best seg={-self.best_val:.4f}")
                    break

            if self.global_step >= max_steps:
                break
        logging.info(f"Done. best seg_brain={-self.best_val:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    run_log = os.path.join(args.exp, "logs", args.name)
    if os.path.exists(run_log) and not args.resume:
        shutil.rmtree(run_log)
    os.makedirs(run_log, exist_ok=True)
    writer = make_loggers(args.exp, args.name, args.verbose)
    try:
        T1Runner(args).train(writer)
    finally:
        writer.flush(); writer.close()


if __name__ == "__main__":
    sys.exit(main())
