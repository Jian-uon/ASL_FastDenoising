# utils.py
from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence


def pad_along_T(tensors, pad_value=0.0):
    """
    输入: list of [T_i, 1, H, W]
    输出: [B, max_T, 1, H, W], mask [B, max_T] (True = padded/无效)
    """
    T_list = [t.shape[0] for t in tensors]
    max_T = max(T_list)
    B = len(tensors)
    _, C, H, W = tensors[0].shape

    flat = [t.view(t.shape[0], -1) for t in tensors]
    padded = pad_sequence(flat, batch_first=True, padding_value=pad_value)  # [B, max_T, C*H*W]
    padded = padded.view(B, max_T, C, H, W).contiguous()

    mask = torch.zeros(B, max_T, dtype=torch.bool)
    for i, Ti in enumerate(T_list):
        if Ti < max_T:
            mask[i, Ti:] = True
    return padded, mask


def collate_varlen(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    输入 item（每个 z-slice）：{'setA': [T,1,H,W], 'setB': [T,1,H,W], 't1': [1,H,W],
                              optional 'gm'/'wm'/'csf': [1,H,W]}
    输出：
      'setA' [B,TAmax,1,H,W], 'maskA' [B,TAmax]
      'setB' [B,TBmax,1,H,W], 'maskB' [B,TBmax]
      't1' [B,1,H,W]; 'gm','wm','csf' [B,1,H,W] when present
    """
    setA_list, setB_list, t1_list = [], [], []
    extras: Dict[str, list] = {}
    sids: list = []
    for it in batch:
        setA_list.append(it["setA"])
        setB_list.append(it["setB"])
        t1_list.append(it["t1"])
        for k in ("gm", "wm", "csf"):
            if k in it:
                extras.setdefault(k, []).append(it[k])
        if "subject_id" in it:
            sids.append(it["subject_id"])

    xA, mA = pad_along_T(setA_list)
    xB, mB = pad_along_T(setB_list)
    out: Dict[str, torch.Tensor] = {
        "setA": xA, "maskA": mA,
        "setB": xB, "maskB": mB,
        "t1": torch.stack(t1_list, dim=0),
    }
    for k, lst in extras.items():
        if len(lst) == len(batch):
            out[k] = torch.stack(lst, dim=0)
    if len(sids) == len(batch):
        # per-sample subject grouping key (list, NOT stacked — keeps it out of the
        # device-move / tensor path; eval reads it to aggregate at subject level).
        out["subject_id"] = sids
    return out


def masked_direct_mean(x_set, mask, eps=1e-8):
    """
    x_set: [B,T,1,H,W], mask: [B,T] (True=pad)
    return: mean [B,1,H,W]
    """
    valid = (~mask).float()                          # [B,T]
    wsum = valid.sum(dim=1).clamp_min(1.0)           # [B]
    w = valid.view(x_set.size(0), x_set.size(1), 1, 1, 1)
    num = (x_set * w).sum(dim=1)                    # [B,1,H,W]
    return num / wsum.view(-1, 1, 1, 1).clamp_min(eps)
