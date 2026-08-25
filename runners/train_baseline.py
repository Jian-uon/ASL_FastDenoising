# -*- coding: utf-8 -*-
"""Unified trainer for plain-UNet baselines: supervised / vanilla-N2N / Noise2Self.

Modes
-----
sup    : input = mean(setA),                 target = mean(setA U setB)  (full-NEX upper bound)
n2n    : input = mean(setA),                 target = mean(setB)          (vanilla N2N w/o T1, w/o SetTrans)
n2self : input = J-invariant masked mean(setA), target = mean(setA)        (single-image self-sup)

Brain mask = (t1 > 0.05) is applied to all loss terms; t1 is NEVER fed to the model.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from typing import Dict, Optional, Tuple

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
from losses.n2self_loss import j_invariant_l1, make_j_invariant_input
from models.plain_unet import PlainUNet2D
from models.swinir_2d import SwinIR2D
from runners.asl_t1_guided_runner_dmvae_n2n import (
    _FallbackEMAModel,
    _compute_psnr_ssim,
    direct_mean_from_frames,
    prepare_asl_pair_batch,
    set_seed,
)

try:
    from utils.training_utils import EMAModel  # type: ignore
except Exception:
    EMAModel = None


def parse_args():
    p = argparse.ArgumentParser("Plain-UNet baseline trainer (sup / n2n / n2self)")
    p.add_argument("--mode", type=str, required=True, choices=["sup", "n2n", "n2self"])
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

    # EMA parity with the main line (submit_min.sh COMMON: --ema_decay 0.997
    # --ema_start_step 40). Validation and best-CNR selection both run on EMA
    # weights, so a baseline smoothed by a different EMA is not selected under
    # the same rule -- which is exactly the selection asymmetry the paper claims
    # to have removed. Defaults keep the historical behaviour of older runs.
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_start_step", type=int, default=0)

    p.add_argument("--arch", type=str, default="plainunet",
                   choices=["plainunet", "swinir"],
                   help="Baseline backbone. 'swinir' = SwinIR-light Transformer denoiser "
                        "(recent-architecture reference); same I/O contract as PlainUNet2D.")
    p.add_argument("--t1_concat", action="store_true",
                   help="Unguarded T1-concat baseline: feed [mean(setA); T1] as a 2-channel "
                        "input (in_ch=2), so T1 flows freely into the value/output path. The "
                        "PlainUNet counterpart of the manuscript A1 (conventional fusion, no "
                        "content guard). T1 is NOT used for the brain mask difference here.")
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--skip_dropout", type=float, default=0.3)
    p.add_argument("--n2self_mask_ratio", type=float, default=0.05)

    # SwinIR-only hyperparameters (ignored when --arch plainunet)
    p.add_argument("--swinir_embed_dim", type=int, default=60)
    p.add_argument("--swinir_depths", type=str, default="6,6,6,6")
    p.add_argument("--swinir_num_heads", type=str, default="6,6,6,6")
    p.add_argument("--swinir_window", type=int, default=8)
    p.add_argument("--swinir_mlp_ratio", type=float, default=2.0)

    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--max_steps", type=int, default=0,
                   help="Override config max_steps (1 step = 1 epoch). 0 = use config value.")
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


class BaselineRunner:
    def __init__(self, args):
        self.args = args
        self.cfg = Config(args.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Mode: {args.mode} | Device: {self.device}")

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

        self.t1_concat = bool(getattr(args, "t1_concat", False))
        in_ch = 2 if self.t1_concat else 1
        if args.arch == "swinir":
            self.arch_cfg = dict(
                embed_dim=int(args.swinir_embed_dim),
                depths=tuple(int(d) for d in str(args.swinir_depths).split(",")),
                num_heads=tuple(int(h) for h in str(args.swinir_num_heads).split(",")),
                window_size=int(args.swinir_window),
                mlp_ratio=float(args.swinir_mlp_ratio),
            )
            self.model = SwinIR2D(
                hw=int(tp.asl_hw), in_ch=in_ch, out_ch=1, **self.arch_cfg,
            ).to(self.device)
        else:
            self.arch_cfg = dict(
                base_ch=int(args.base_ch), depth=int(args.depth),
                skip_dropout=float(args.skip_dropout),
            )
            self.model = PlainUNet2D(
                hw=int(tp.asl_hw), in_ch=in_ch, out_ch=1, **self.arch_cfg,
            ).to(self.device)
        self.arch_cfg["in_ch"] = in_ch
        self.arch_cfg["t1_concat"] = self.t1_concat
        _np = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Arch: {args.arch} | in_ch: {in_ch} | t1_concat: {self.t1_concat} "
                     f"| params: {_np/1e6:.3f}M")
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tp.lr, weight_decay=tp.weight_decay,
        )
        max_steps = int(args.max_steps) if int(getattr(args, "max_steps", 0)) > 0 else int(tp.max_steps)
        if args.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_steps, eta_min=args.lr_min)
        else:
            self.scheduler = None

        ema_cls = EMAModel if EMAModel is not None else _FallbackEMAModel
        self.ema = ema_cls(self.model,
                           update_after_step=int(getattr(args, "ema_start_step", 0)),
                           inv_gamma=1.0, power=2 / 3, min_value=0.0,
                           max_value=float(getattr(args, "ema_decay", 0.9999)),
                           device=self.device)

        self.global_step = 0
        # Best = PSNR vs 12-NEX reference (higher is better) — see main runner
        # for rationale (L1 vs setB diverges from gold-reference under N2N).
        self.best_val = -float("inf")
        self.no_improve_count = 0
        self.eval_count = 0

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
        st = torch.load(latest, map_location=self.device)
        self.model.load_state_dict(st["model"])
        self.global_step = st.get("step", 0)
        self.best_val = st.get("best_val", float("inf"))
        if "optimizer" in st:
            self.optimizer.load_state_dict(st["optimizer"])
        if "ema" in st:
            self.ema.ema_state = {k: v.to(self.device) for k, v in st["ema"].items()}
            self.ema.optimization_step = st.get("ema_optimization_step", 0)
        if "scheduler" in st and self.scheduler is not None:
            self.scheduler.load_state_dict(st["scheduler"])
        logging.info(f"Resumed (step={self.global_step}, best={self.best_val:.6f})")

    def _save(self, tag: str):
        torch.save({
            "model": self.model.state_dict(),
            "step": self.global_step,
            "best_val": self.best_val,
            "ema": {k: v.detach().cpu() for k, v in self.ema.ema_state.items()},
            "ema_optimization_step": getattr(self.ema, "optimization_step", 0),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "mode": self.args.mode,
            "arch": self.args.arch,
            "arch_cfg": self.arch_cfg,
        }, self._ckpt(tag))

    def _t1_channel(self, pack: Dict[str, Tensor]) -> Tensor:
        """T1 as a [B,1,H,W] extra input channel (center slice if 2.5-D)."""
        t1 = pack["t1"]
        if t1.shape[1] > 1:
            c = t1.shape[1] // 2
            t1 = t1[:, c:c + 1]
        return t1

    def _build_io(self, pack: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Return (model_input, target, brain_mask, jmask_or_None)."""
        meanA = direct_mean_from_frames(pack["setA"], pack.get("lenA"))
        brain = (pack["t1"] > 0.05).float()
        if self.args.mode == "n2n":
            meanB = direct_mean_from_frames(pack["setB"], pack.get("lenB"))
            x_in, target, jmask = meanA, meanB.detach(), None
        elif self.args.mode == "sup":
            union = direct_mean_from_frames(
                torch.cat([pack["setA"], pack["setB"]], dim=1),
                pack["lenA"] + pack["lenB"],
            )
            x_in, target, jmask = meanA, union.detach(), None
        else:  # n2self
            x_masked, jmask = make_j_invariant_input(meanA, mask_ratio=self.args.n2self_mask_ratio)
            x_in, target = x_masked, meanA.detach()
        # Unguarded T1-concat (A1 PlainUNet variant): T1 enters as a 2nd input channel.
        if self.t1_concat:
            x_in = torch.cat([x_in, self._t1_channel(pack)], dim=1)
        return x_in, target, brain, jmask

    def _forward_train(self, pack: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        x_in, target, brain, jmask = self._build_io(pack)
        pred = self._unwrap()(x_in)
        if self.args.mode == "n2self":
            loss = j_invariant_l1(pred, target, jmask, brain)
        else:
            loss = _masked_l1(pred, target, brain)
        return loss, {"loss": float(loss.detach().item())}

    @torch.no_grad()
    def _predict_with_ema(self, x: Tensor) -> Tensor:
        m = self._unwrap(); m.eval()
        self.ema.store(m); self.ema.copy_to(m)
        out = m(x)
        self.ema.restore(m)
        return out

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, float, float, float]:
        """Return (l1_b, psnr_b, ssim_b, l1_ref, psnr_ref, ssim_ref)."""
        self.model.eval()
        acc = {k: 0.0 for k in ("l1_b", "psnr_b", "ssim_b", "l1_ref", "psnr_ref", "ssim_ref")}
        n, saved = 0, 0
        for vb in tqdm(self.val_loader, desc=f"Val (step={self.global_step})", dynamic_ncols=True):
            pack = prepare_asl_pair_batch(vb, self.device)
            meanA = direct_mean_from_frames(pack["setA"], pack.get("lenA"))
            meanB = direct_mean_from_frames(pack["setB"], pack.get("lenB"))
            union = direct_mean_from_frames(
                torch.cat([pack["setA"], pack["setB"]], dim=1),
                pack["lenA"] + pack["lenB"],
            )
            x_val = torch.cat([meanA, self._t1_channel(pack)], dim=1) if self.t1_concat else meanA
            out = self._predict_with_ema(x_val)
            l1_b = F.l1_loss(out, meanB).item()
            psnr_b, ssim_b = _compute_psnr_ssim(out, meanB)
            l1_ref = F.l1_loss(out, union).item()
            psnr_ref, ssim_ref = _compute_psnr_ssim(out, union)
            acc["l1_b"] += l1_b; acc["psnr_b"] += psnr_b; acc["ssim_b"] += ssim_b
            acc["l1_ref"] += l1_ref; acc["psnr_ref"] += psnr_ref; acc["ssim_ref"] += ssim_ref
            n += 1

            if self.args.save_images and saved < self.args.log_images:
                try:
                    # Mask the recon (and its diff) to the brain (t1>0.05), the SAME
                    # region the loss/metrics use. The N2N loss is brain-masked, so the
                    # PlainUNet output outside the brain is unconstrained garbage (a gray
                    # halo); showing it raw made the baseline panels inconsistent with the
                    # (seg-head-masked) MoSSM panels. Display only what is evaluated.
                    brain = (pack["t1"][0, 0] > 0.05).float()
                    out_m = (out[0, 0] * brain).cpu().numpy()
                    diff_m = ((out[0, 0] - union[0, 0]).abs() * brain).cpu().numpy()
                    imgs = [
                        (pack["t1"][0, 0].cpu().numpy(),  "T1w"),
                        (union[0, 0].cpu().numpy(),       "12-NEX union"),
                        (meanA[0, 0].cpu().numpy(),       "mean(A) input"),
                        (meanB[0, 0].cpu().numpy(),       "mean(B)"),
                        (out_m,                           f"{self.args.mode} recon"),
                        (diff_m,                          "|recon-union|"),
                    ]
                    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
                    axes = axes.reshape(-1)
                    for i, (im, t) in enumerate(imgs):
                        axes[i].imshow(im, cmap="gray")
                        axes[i].set_title(t, fontsize=8); axes[i].axis("off")
                    fig.tight_layout()
                    fig.savefig(os.path.join(self.val_img_dir,
                        f"step{self.global_step}_val{saved}.png"), dpi=80, bbox_inches="tight")
                    plt.close(fig)
                    saved += 1
                except Exception as e:
                    plt.close("all")
                    logging.warning(f"Save img failed: {e}")
        m = max(1, n)
        return (acc["l1_b"]/m, acc["psnr_b"]/m, acc["ssim_b"]/m,
                acc["l1_ref"]/m, acc["psnr_ref"]/m, acc["ssim_ref"]/m)

    def train(self, writer: SummaryWriter):
        max_steps = int(self.args.max_steps) if int(getattr(self.args, "max_steps", 0)) > 0 \
            else int(self.cfg.asl_denoiser_train_params.max_steps)
        eval_every = int(self.cfg.asl_denoiser_train_params.eval_every)

        while self.global_step < max_steps:
            self.global_step += 1
            pbar = tqdm(self.train_loader, desc=f"Train ({self.global_step}/{max_steps})", dynamic_ncols=True)
            self.model.train()
            last_loss = 0.0
            for batch in pbar:
                pack = prepare_asl_pair_batch(batch, self.device)
                loss, stats = self._forward_train(pack)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.ema.step(self._unwrap())
                last_loss = stats["loss"]
                pbar.set_postfix(loss=f"{last_loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

            writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            writer.add_scalar("train/total", last_loss, self.global_step)

            if self.global_step % eval_every == 0 or self.global_step >= max_steps:
                l1_b, psnr_b, ssim_b, l1_ref, psnr_ref, ssim_ref = self.validate()
                writer.add_scalar("val/l1_vs_B", l1_b, self.global_step)
                writer.add_scalar("val/psnr_vs_B", psnr_b, self.global_step)
                writer.add_scalar("val/ssim_vs_B", ssim_b, self.global_step)
                writer.add_scalar("val/l1_vs_ref", l1_ref, self.global_step)
                writer.add_scalar("val/psnr_vs_ref", psnr_ref, self.global_step)
                writer.add_scalar("val/ssim_vs_ref", ssim_ref, self.global_step)

                self._save("latest")
                if self.args.save_every > 0 and self.global_step % self.args.save_every == 0:
                    self._save(f"step{self.global_step:06d}")

                self.eval_count += 1
                patience = int(self.args.early_stop_patience)
                if psnr_ref > self.best_val:
                    self.best_val = psnr_ref
                    self.no_improve_count = 0
                    self._save("best")
                    logging.info(f"[BEST] step={self.global_step} psnr_ref={psnr_ref:.2f} | "
                                 f"l1_B={l1_b:.4f} ssim_ref={ssim_ref:.4f}")
                else:
                    self.no_improve_count += 1
                    tail = f" (best {self.best_val:.2f}"
                    if patience > 0:
                        tail += f", no-improve {self.no_improve_count}/{patience}"
                    tail += ")"
                    logging.info(f"[VAL ] step={self.global_step} psnr_ref={psnr_ref:.2f} | "
                                 f"l1_B={l1_b:.4f} ssim_ref={ssim_ref:.4f}{tail}")

                if (patience > 0
                    and self.eval_count >= int(self.args.early_stop_min_evals)
                    and self.no_improve_count >= patience):
                    logging.info(f"[EARLY STOP] step={self.global_step}, best psnr_ref={self.best_val:.4f}")
                    break

            if self.global_step >= max_steps:
                break
        logging.info(f"Done. best psnr_ref={self.best_val:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    run_log = os.path.join(args.exp, "logs", args.name)
    if os.path.exists(run_log) and not args.resume:
        shutil.rmtree(run_log)
    os.makedirs(run_log, exist_ok=True)
    writer = make_loggers(args.exp, args.name, args.verbose)
    try:
        BaselineRunner(args).train(writer)
    finally:
        writer.flush(); writer.close()


if __name__ == "__main__":
    sys.exit(main())
