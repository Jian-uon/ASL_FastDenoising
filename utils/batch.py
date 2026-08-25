from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


TensorLike = Union[Tensor, Sequence[Tensor]]


def _pad_sequence_tensors(items: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    """Pad a list of [T, C, H, W] tensors into [B, T_max, C, H, W]."""
    if len(items) == 0:
        raise ValueError("Empty tensor list cannot be padded.")
    lengths = torch.tensor([int(x.shape[0]) for x in items], dtype=torch.long)
    t_max = int(lengths.max().item())
    sample = items[0]
    if sample.ndim != 4:
        raise ValueError(f"Expected list items with shape [T,C,H,W], got {tuple(sample.shape)}")
    b = len(items)
    _, c, h, w = sample.shape
    out = sample.new_zeros((b, t_max, c, h, w))
    for i, x in enumerate(items):
        out[i, : x.shape[0]] = x
    return out, lengths



def ensure_batched_frames(x: TensorLike, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    Convert different collate styles into:
      frames: [B, T, C, H, W]
      lengths: [B]

    Accepted inputs:
      - list[Tensor[T,C,H,W]]
      - Tensor[T,C,H,W]            -> B=1
      - Tensor[B,T,C,H,W]
      - Tensor[B,C,H,W]            -> T=1
    """
    if isinstance(x, (list, tuple)):
        return _pad_sequence_tensors(x)

    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported frame container type: {type(x)!r}")

    if x.ndim == 5:
        b, t = x.shape[:2]
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
        return x, lengths

    if x.ndim == 4:
        # Ambiguous: [T,C,H,W] or [B,C,H,W]. We treat leading dim as T for singleton batch.
        frames = x.unsqueeze(0)
        lengths = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        return frames, lengths

    raise ValueError(f"Expected 4D/5D tensor or list of 4D tensors, got shape {tuple(x.shape)}")



def ensure_image_batch(x: TensorLike) -> Tensor:
    """Convert image tensor to [B, C, H, W]."""
    if isinstance(x, (list, tuple)):
        x = torch.stack(list(x), dim=0)
    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported image container type: {type(x)!r}")
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")
    return x



def extract_lengths(batch: Dict[str, Tensor], prefix: str) -> Optional[Tensor]:
    candidates = [
        f"{prefix}_len",
        f"{prefix}_length",
        f"{prefix}_lengths",
        f"len_{prefix}",
        f"length_{prefix}",
        f"lengths_{prefix}",
    ]
    for key in candidates:
        if key in batch:
            return batch[key]
    return None



def prepare_asl_pair_batch(batch: Dict[str, TensorLike], device: torch.device) -> Dict[str, Tensor]:
    """
    Normalize the batch returned by `ASLTwoSetDataset2DFlat` / `collate_varlen` to a stable format.
    Expected semantic keys are setA, setB, t1, and (optionally) gm/wm/csf partial-volume maps.
    """
    set_a, len_a = ensure_batched_frames(batch["setA"], extract_lengths(batch, "setA"))
    set_b, len_b = ensure_batched_frames(batch["setB"], extract_lengths(batch, "setB"))
    t1 = ensure_image_batch(batch["t1"])

    out: Dict[str, Tensor] = {
        "setA": set_a.to(device=device, dtype=torch.float32, non_blocking=True),
        "setB": set_b.to(device=device, dtype=torch.float32, non_blocking=True),
        "lenA": len_a.to(device=device, dtype=torch.long, non_blocking=True),
        "lenB": len_b.to(device=device, dtype=torch.long, non_blocking=True),
        "t1": t1.to(device=device, dtype=torch.float32, non_blocking=True),
    }
    # Forward partial-volume maps if the dataset provides them. Required for
    # loss_contrast (GM/WM-boundary-weighted L1) and loss_seg (PV CE/L1).
    # Without these, the loss code silently zeroes out those terms.
    for k in ("gm", "wm", "csf"):
        if k in batch:
            out[k] = ensure_image_batch(batch[k]).to(
                device=device, dtype=torch.float32, non_blocking=True)
    return out
