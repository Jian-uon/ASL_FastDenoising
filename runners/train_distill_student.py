# -*- coding: utf-8 -*-
"""Train a T1-free student via distillation from a frozen v39 teacher.

Innovation T5×S4: Oracle teacher (set_a + set_b + real T1) distills into a
T1-free student (set_a only, no T1 input modality). The student is structurally
hallucination-immune — at deployment it has no T1 input path at all.

Supported students:
  - asl_mamba: 4-direction selective state-space Mamba student (v40)
  - asl_vrt:   spatial+temporal mutual attention transformer student (v41)

Distillation losses:
  L_out  = masked L1( student.recon, teacher.asl_recon.detach() )
  L_feat = masked L1( student.feat_bottleneck_adapted,
                      teacher.fused_map.detach() )      [optional, --w_feat>0]
  L_n2n  = masked L1( student.recon, mean(set_b) )      [optional N2N anchor,
                                                         --w_n2n>0]

Usage
-----
    python runners/train_distill_student.py \
      --config env/local/configs/win_asl_2d_home_v37.yml --exp C:/tmp/asl_exp \
      --teacher_ckpt C:/tmp/asl_exp/logs/run_full_v39/checkpoints/swa.pth \
      --student asl_mamba \
      --name run_full_v40_mamba_distill \
      --max_steps 1000 --lr 3e-4 \
      --w_out 1.0 --w_feat 0.3 --w_n2n 0.2 \
      --init_t1_from C:/tmp/asl_exp/logs/stage1_t1_300step/checkpoints/latest.pth \
      --base_ch_teacher 32 --depth_teacher 4 \
      --use_t1_cross_fusion --t1_attn_max_tokens 1024 --use_svfw \
      --freeze_t1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.conf_data import Config
from dataio.dataloaders import get_asl_2d_loaders
from models.asl_t1_model import ASLT1Denoiser
from models.asl_mamba_student import ASLMambaStudent
from models.asl_vrt_student import ASLVRTStudent
from runners.asl_t1_guided_runner_dmvae_n2n import (
    prepare_asl_pair_batch, _compute_psnr_ssim, direct_mean_from_frames,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("train_distill_student")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--exp",    type=str, required=True)
    p.add_argument("--name",   type=str, required=True)
    p.add_argument("--seed",   type=int, default=42)

    # Teacher
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--base_ch_teacher", type=int, default=32)
    p.add_argument("--depth_teacher",   type=int, default=4)
    # The teacher is a v37/v38/v39 ASLT1Denoiser; its construction needs the
    # same flags the runner used. We pass them through here.
    p.add_argument("--use_t1_cross_fusion", action="store_true")
    p.add_argument("--t1_attn_max_tokens",  type=int, default=1024)
    p.add_argument("--use_svfw",   action="store_true")
    p.add_argument("--use_film",   action="store_true")
    p.add_argument("--use_heteroscedastic", action="store_true")
    p.add_argument("--init_t1_from", type=str, default=None)
    p.add_argument("--freeze_t1",    action="store_true")

    # Student
    p.add_argument("--student", type=str, choices=["asl_mamba", "asl_vrt"], required=True)
    p.add_argument("--base_ch_student", type=int, default=24)
    p.add_argument("--depth_blocks", type=int, nargs=4, default=[2, 2, 2, 2])

    # Training
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=0,
                   help="Override config batch size; 0 = use config")
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=1e-2)

    # Distillation losses
    p.add_argument("--w_out",  type=float, default=1.0,
                   help="L1 between student and teacher recon (output-level distill)")
    p.add_argument("--w_feat", type=float, default=0.3,
                   help="L1 between student bottleneck (adapted) and teacher fused_map")
    p.add_argument("--w_n2n",  type=float, default=0.2,
                   help="L1 between student recon and mean(set_b) — N2N anchor")

    # Logging / IO
    p.add_argument("--save_every",    type=int, default=100)
    p.add_argument("--eval_every",    type=int, default=10)
    p.add_argument("--save_images",   action="store_true")
    return p.parse_args()


def _build_teacher(args: argparse.Namespace, cfg: Config, device: torch.device) -> ASLT1Denoiser:
    """Build a v37/v38/v39 ASLT1Denoiser teacher and load the ckpt frozen."""
    asl_hw = cfg.asl_denoiser_train_params.asl_hw
    t1_hw = cfg.asl_denoiser_train_params.t1_hw
    teacher = ASLT1Denoiser(
        asl_hw=asl_hw, t1_hw=t1_hw,
        in_ch=1, base_ch=args.base_ch_teacher, depth=args.depth_teacher,
        use_t1_cross_fusion=args.use_t1_cross_fusion,
        t1_attn_max_tokens=args.t1_attn_max_tokens,
        use_svfw=args.use_svfw,
        use_film=args.use_film,
        use_heteroscedastic=args.use_heteroscedastic,
    ).to(device)
    state = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    sd = state.get("model") or state.get("ema") or state
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    if missing:
        logging.warning(f"teacher missing keys: {missing[:5]} (showing 5)")
    if unexpected:
        logging.warning(f"teacher unexpected keys: {unexpected[:5]}")
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


def _build_student(args: argparse.Namespace, device: torch.device) -> nn.Module:
    teacher_bottleneck_ch = args.base_ch_teacher * (2 ** (args.depth_teacher - 1))
    if args.student == "asl_mamba":
        return ASLMambaStudent(
            in_ch=1, base_ch=args.base_ch_student,
            depth_blocks=tuple(args.depth_blocks),
            teacher_bottleneck_ch=teacher_bottleneck_ch,
        ).to(device)
    if args.student == "asl_vrt":
        return ASLVRTStudent(
            in_ch=1, base_ch=args.base_ch_student,
            depth_blocks=tuple(args.depth_blocks),
            teacher_bottleneck_ch=teacher_bottleneck_ch,
        ).to(device)
    raise ValueError(args.student)


def _masked_l1(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
    denom = mask.sum().clamp_min(1.0)
    return ((a - b).abs() * mask).sum() / denom


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = Config(args.config)

    log_dir = Path(args.exp) / "logs" / args.name
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.exp) / "tensorboard" / args.name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir))
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Override batch_size if requested (config.data_loading.batch_size is the source)
    if args.batch_size > 0:
        cfg.data_loading.batch_size = int(args.batch_size)
    loaders = get_asl_2d_loaders(
        cfg,
        asl_hw=cfg.asl_denoiser_train_params.asl_hw,
        asl_z=cfg.asl_denoiser_train_params.asl_z,
        t1_hw=cfg.asl_denoiser_train_params.t1_hw,
        t1_z=cfg.asl_denoiser_train_params.t1_z,
    )
    train_loader, val_loader = loaders["train"], loaders["val"]

    teacher = _build_teacher(args, cfg, device)
    # Hook the teacher to capture an intermediate "fused" feature for KD.
    # Post-refactor (2026-05-15) the bottleneck CrossModalFusion was moved into
    # T1GuidedCoarseHead; we hook `t1_head` directly which returns feat_l1
    # [B, l1_ch, 32, 32]. Note the shape differs from the pre-refactor
    # cross_fusion output [B, bottleneck_ch, 16, 16] -- the student feat-distill
    # adapter must match this if w_feat > 0.
    teacher_fused_holder: Dict[str, Tensor] = {}
    if args.w_feat > 0.0:
        hook_target = None
        if hasattr(teacher, "t1_head") and teacher.t1_head is not None:
            hook_target = teacher.t1_head
        elif hasattr(teacher, "cross_fusion") and teacher.cross_fusion is not None:
            # Legacy path (pre-refactor ckpts).
            hook_target = teacher.cross_fusion
        if hook_target is not None:
            def _capture_fused(_module, _inputs, output):
                teacher_fused_holder["fused"] = output
            hook_target.register_forward_hook(_capture_fused)
    student = _build_student(args, device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logging.info(f"Student '{args.student}' built: {n_params/1e6:.2f}M trainable params")

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr_min,
    )

    mask_thresh = 0.05
    global_step = 0
    student.train()

    while global_step < args.max_steps:
        for batch in train_loader:
            global_step += 1
            pack = prepare_asl_pair_batch(batch, device)

            # brain mask from real T1 (mask quality from teacher's input)
            brain = (pack["t1"] > mask_thresh).float()

            # Teacher forward (no grad, with set_a + set_b combined as oracle input)
            with torch.no_grad():
                # Oracle: feed teacher all 12 frames as set_a (privileged data).
                set_all = torch.cat([pack["setA"], pack["setB"]], dim=1)
                len_all = (pack.get("lenA") + pack.get("lenB")) if pack.get("lenA") is not None else None
                t_out = teacher(
                    set_a=set_all,
                    set_b=pack["setB"],   # not used by forward path itself but argument required
                    t1=pack["t1"],
                    len_a=len_all,
                    len_b=pack.get("lenB"),
                    mask_a=None, mask_b=pack.get("maskB"),
                )
                teacher_recon = t_out["asl_recon"]
                teacher_fused = teacher_fused_holder.get("fused")

            # Student forward
            s_out = student(
                set_a=pack["setA"],
                lengths=pack.get("lenA"),
                mask=pack.get("maskA"),
                return_features=(args.w_feat > 0.0),
            )
            student_recon = s_out["asl_recon"]

            # Losses
            losses: Dict[str, Tensor] = {}
            losses["L_out"] = _masked_l1(student_recon, teacher_recon, brain)
            total = args.w_out * losses["L_out"]

            if args.w_feat > 0.0 and teacher_fused is not None and "feat_bottleneck_adapted" in s_out:
                # spatial scales should match; if not, project/pool
                tf = teacher_fused
                sf = s_out["feat_bottleneck_adapted"]
                if tf.shape[-2:] != sf.shape[-2:]:
                    sf = F.interpolate(sf, size=tf.shape[-2:], mode="bilinear",
                                        align_corners=False)
                losses["L_feat"] = (sf - tf).abs().mean()
                total = total + args.w_feat * losses["L_feat"]

            if args.w_n2n > 0.0:
                meanB = direct_mean_from_frames(pack["setB"], pack.get("lenB"))
                losses["L_n2n"] = _masked_l1(student_recon, meanB, brain)
                total = total + args.w_n2n * losses["L_n2n"]

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # TB logging
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/total", float(total.item()), global_step)
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", float(v.item()), global_step)

            # Periodic eval
            if global_step % args.eval_every == 0 or global_step >= args.max_steps:
                _validate(student, teacher, val_loader, writer, device,
                          global_step, mask_thresh)

            # Checkpoint
            torch.save({
                "model": student.state_dict(),
                "step": global_step,
                "args": vars(args),
            }, str(ckpt_dir / "latest.pth"))
            if args.save_every > 0 and global_step % args.save_every == 0:
                torch.save({
                    "model": student.state_dict(),
                    "step": global_step,
                    "args": vars(args),
                }, str(ckpt_dir / f"step{global_step:06d}.pth"))

            if global_step >= args.max_steps:
                break

    logging.info(f"Distillation done at step {global_step}.")


@torch.no_grad()
def _validate(student, teacher, val_loader, writer, device, step, mask_thresh):
    student.eval()
    accs: Dict[str, float] = {"l1_b": 0.0, "psnr_b": 0.0, "psnr_ref": 0.0,
                              "l1_to_teacher": 0.0}
    n = 0
    for vb in val_loader:
        pack = prepare_asl_pair_batch(vb, device)
        brain = (pack["t1"] > mask_thresh).float()
        set_all = torch.cat([pack["setA"], pack["setB"]], dim=1)
        len_all = (pack.get("lenA") + pack.get("lenB")) if pack.get("lenA") is not None else None
        t_out = teacher(set_a=set_all, set_b=pack["setB"], t1=pack["t1"],
                        len_a=len_all, len_b=pack.get("lenB"))
        teacher_recon = t_out["asl_recon"]
        s_out = student(set_a=pack["setA"], lengths=pack.get("lenA"),
                         mask=pack.get("maskA"))
        s_recon = s_out["asl_recon"]
        meanB = direct_mean_from_frames(pack["setB"], pack.get("lenB"))
        union = direct_mean_from_frames(
            torch.cat([pack["setA"], pack["setB"]], dim=1),
            (pack.get("lenA") + pack.get("lenB")) if pack.get("lenA") is not None else None,
        )
        accs["l1_b"]    += float(F.l1_loss(s_recon, meanB).item())
        accs["l1_to_teacher"] += float(_masked_l1(s_recon, teacher_recon, brain).item())
        psnr_b, _   = _compute_psnr_ssim(s_recon, meanB)
        psnr_ref, _ = _compute_psnr_ssim(s_recon, union)
        accs["psnr_b"]   += float(psnr_b)
        accs["psnr_ref"] += float(psnr_ref)
        n += 1
    for k in accs:
        accs[k] /= max(1, n)
    writer.add_scalar("val/l1_to_teacher", accs["l1_to_teacher"], step)
    writer.add_scalar("val/l1_b",          accs["l1_b"],          step)
    writer.add_scalar("val/psnr_b",        accs["psnr_b"],        step)
    writer.add_scalar("val/psnr_ref",      accs["psnr_ref"],      step)
    logging.info(
        f"[VAL ] step={step} | l1_to_teacher={accs['l1_to_teacher']:.4f} "
        f"| l1_b={accs['l1_b']:.4f} psnr_b={accs['psnr_b']:.2f} psnr_ref={accs['psnr_ref']:.2f}"
    )
    student.train()


if __name__ == "__main__":
    main()
