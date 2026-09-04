# data_classes.py
# 载入 4D ASL + 3D T1，构建 2D slice 数据集
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List, Optional, Tuple
import torch
from monai.data import CacheDataset, Dataset, NibabelReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd, Resized,
    ResizeWithPadOrCropd, ScaleIntensityRangePercentilesd,
)


def build_cached_subjects(subjects_list, pre_tf: Compose, cache_rate: float = 0.5, cache_workers: int = 4):
    """对 subject 字典列表做体级预处理并缓存到内存。"""
    return CacheDataset(data=subjects_list, transform=pre_tf, cache_rate=float(cache_rate), num_workers=int(cache_workers))


def get_pre_asl_transform(asl_hw: int = 128,
                          asl_z: int = 48,
                          t1_hw: int = 128,
                          t1_z: int = 48,
                          intensity: str = "percentile", p_lo: int = 1, p_hi: int = 99) -> Compose:
    """把 [H,W,Z,T] 的 ASL 差分转为 [T,H,W,Z]，并与 T1 对齐到统一体素。
    Also loads GM/WM/CSF partial volume maps in ASL space for multi-task supervision."""
    pv_keys = ["gm", "wm", "csf"]
    # m0 rides the same pad/crop as t1 and the PV maps, which is what keeps it voxel-aligned
    # with the reconstruction; it is left on its own intensity scale (no percentile rescale),
    # since sCoV on CBF only needs the ratio dM/M0 and is invariant to M0's global scale.
    aux_keys = pv_keys + ["m0"]
    img_keys = ["asl_diff", "t1"] + aux_keys
    tf = [
        LoadImaged(keys=img_keys, reader=NibabelReader(), image_only=False),
        EnsureChannelFirstd(keys=["asl_diff"], channel_dim=-1),   # [H,W,Z,T] -> [T,H,W,Z]
        EnsureChannelFirstd(keys=["t1"] + aux_keys),               # [H,W,Z] -> [1,H,W,Z]
        EnsureTyped(keys=img_keys, track_meta=True),
        Orientationd(keys=img_keys, axcodes="LPS"),
        # Native data is (96, 112, 52). Trilinear resize to 128 introduces a periodic
        # interpolation alias (栅格 ~3-4 px period) visible in noisy frames and their means.
        # Use pad-or-crop instead: zero-pad H 96→128, W 112→128; center-crop Z 52→48.
        ResizeWithPadOrCropd(keys=["asl_diff"], spatial_size=(asl_hw, asl_hw, asl_z)),
        ResizeWithPadOrCropd(keys=["t1"] + aux_keys, spatial_size=(t1_hw, t1_hw, t1_z)),
    ]
    if intensity == "percentile":
        # T1 is a fixed anatomical volume → per-volume percentile is fine (no A/B leak).
        tf += [ScaleIntensityRangePercentilesd(keys=["t1"], lower=p_lo, upper=p_hi, b_min=0., b_max=1., clip=True)]
        # NOTE (2026-07-14): ASL diff is DELIBERATELY left RAW here. The old
        # per-volume percentile+clip over ALL T frames leaked held-out set-B (and
        # unavailable-at-inference) frames into set-A's scale, and the nonlinear clip
        # biased the Noise2Noise zero-mean target. ASL is now normalised per-sample in
        # ASLTwoSetDataset2DFlat.__getitem__ by an affine estimated from set A ONLY
        # (see _affine_from_frames), applied linearly (no clip) to both A and B.
    # PV maps already in [0,1]; just clip to be safe (no rescaling)
    return Compose(tf)


def _affine_from_frames(frames: torch.Tensor, p_lo: float = 0.01, p_hi: float = 0.99,
                        eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust LINEAR affine to ~[0,1] estimated from the AVAILABLE frames only.

    Returns (lo, scale) s.t. x_norm = (x - lo) / scale. Using ONLY the model-visible
    frames (set A at train, the k input frames at inference) removes the held-out /
    unavailable-frame leak; applying it LINEARLY (no hard clip) preserves the
    Noise2Noise zero-mean assumption (the old clip=True bent it)."""
    v = frames.reshape(-1).float()
    # torch.quantile caps its input at 2**24 (~16.7M) elements and raises
    # "quantile() input tensor is too large" above it — which happens for
    # many-frame inference (>=24 frames * a full volume). Deterministically
    # uniform-stride subsample above the cap: for a robust 1st/99th-percentile
    # scale estimate a strided subsample is effectively unbiased, and set-A at
    # train time is small so this branch never fires there (torch.quantile on GPU).
    _CAP = 16_000_000
    if v.numel() > _CAP:
        v = v[:: (v.numel() // _CAP) + 1]
    lo = torch.quantile(v, p_lo)
    hi = torch.quantile(v, p_hi)
    return lo, (hi - lo).clamp_min(eps)


def _valid_slice_mask(diffs: torch.Tensor, nz_tol: float = 1e-6, min_ratio: float = 0.20,
                      max_frames: Optional[int] = 8) -> torch.Tensor:
    """diffs: [T,H,W,Z]，返回 keep_z: [Z]（True=保留）。"""
    assert diffs.ndim == 4
    import math as _m
    T, H, W, Z = diffs.shape
    step = max(1, _m.ceil(T / (max_frames or T)))
    diffs_ = diffs[::step]
    fg_hwz = (diffs_.abs() > nz_tol).any(dim=0)   # [H,W,Z]
    ratios = fg_hwz.float().mean(dim=(0, 1))       # [Z]
    return ratios >= float(min_ratio)


def _valid_pv_slice_mask(gm: torch.Tensor, wm: torch.Tensor, csf: torch.Tensor,
                         pv_tol: float = 0.1, min_ratio: float = 0.05) -> torch.Tensor:
    """Filter slices where partial-volume maps are mostly empty (likely from
    registration / segmentation failures).

    Slice is kept iff at least `min_ratio` fraction of pixels has GM+WM+CSF > pv_tol.
    Defaults: pv_tol=0.1 (sum of PVs > 10%), min_ratio=0.05 (≥5% of pixels in-slice).
    Tunable via runner CLI later if too aggressive / too lax.

    Inputs are expected as [1,H,W,Z] (channel-first volumes from MONAI loader).
    Returns keep_z: [Z] bool.
    """
    assert gm.ndim == 4 and gm.shape[0] == 1
    pv_sum = (gm[0] + wm[0] + csf[0])           # [H,W,Z]
    fg = (pv_sum > pv_tol).float()
    ratios = fg.mean(dim=(0, 1))                # [Z]
    return ratios >= float(min_ratio)


class ASLTwoSetDataset2DFlat(Dataset):
    """每个样本 = 单 z-slice：setA / setB / t1 / gm / wm / csf。

    Slice keep rule:
      - ASL diff has ≥ asl_min_ratio non-zero pixels (existing)
      - GM+WM+CSF sum has ≥ pv_min_ratio pixels > pv_tol  (new — filters
        registration/segmentation failures)
    """

    def __init__(self, base_subjects: CacheDataset, TA_range=(3, 6), TB_range=(6, 12),
                 pv_filter: bool = True, pv_tol: float = 0.1, pv_min_ratio: float = 0.20,
                 slice_context: int = 0):
        self.base = base_subjects
        self.TA_range, self.TB_range = TA_range, TB_range
        self.slice_context = int(slice_context)   # 2.5D: setA/setB carry 2*ctx+1 z-slices as channels
        self.map: List[Tuple[int, int]] = []
        kept_total = 0
        dropped_pv = 0
        for si in range(len(self.base)):
            item = self.base[si]
            diffs = item["asl_diff"]                 # [T,H,W,Z]
            keep_asl = _valid_slice_mask(diffs.cpu())
            if pv_filter and "gm" in item and "wm" in item and "csf" in item:
                keep_pv = _valid_pv_slice_mask(
                    item["gm"].cpu(), item["wm"].cpu(), item["csf"].cpu(),
                    pv_tol=pv_tol, min_ratio=pv_min_ratio,
                )
                keep = keep_asl & keep_pv
                dropped_pv += int((keep_asl & ~keep_pv).sum().item())
            else:
                keep = keep_asl
            for z in torch.where(keep)[0].tolist():
                self.map.append((si, int(z)))
            kept_total += int(keep.sum().item())
        if pv_filter:
            print(f"[ASLTwoSetDataset2DFlat] kept {kept_total} slices, "
                  f"dropped {dropped_pv} for low PV coverage "
                  f"(pv_tol={pv_tol}, min_ratio={pv_min_ratio}).")

    def __len__(self):
        return len(self.map)

    def __getitem__(self, k: int) -> Dict[str, torch.Tensor]:
        si, z = self.map[k]
        item = self.base[si]
        diffs = item["asl_diff"]               # [T,H,W,Z]
        t1 = item["t1"][:, :, :, z]           # [1,H,W]
        gm  = item["gm"][:, :, :, z]          # [1,H,W]
        wm  = item["wm"][:, :, :, z]          # [1,H,W]
        csf = item["csf"][:, :, :, z]         # [1,H,W]
        m0  = item["m0"][:, :, :, z]          # [1,H,W] — raw scale, evaluation only

        ctx = self.slice_context
        if ctx > 0:                            # 2.5D: z-window (edge-clamped) → [T,K,H,W]
            Zt = diffs.shape[-1]
            zc = [min(max(z + dz, 0), Zt - 1) for dz in range(-ctx, ctx + 1)]
            sl = diffs[..., zc].permute(0, 3, 1, 2).contiguous()   # [T, K=2ctx+1, H, W]
        else:
            sl = diffs[..., z].unsqueeze(1).contiguous()           # [T,1,H,W]  (2D)
        T = int(sl.shape[0])

        if T < 2:
            raise RuntimeError(f"Need at least 2 frames to split, got T={T} for sample {(si, z)}")

        order = torch.randperm(T)
        ta_min = max(1, self.TA_range[0])
        ta_max = min(self.TA_range[1], T - 1)
        TA = 1 if ta_min > ta_max else int(torch.randint(ta_min, ta_max + 1, (1,)).item())

        A_idx = order[:TA]
        B_idx = order[TA:]

        setA = torch.index_select(sl, 0, A_idx).contiguous()  # [TA,K,H,W]  (K=1 when 2D)
        setB = torch.index_select(sl, 0, B_idx).contiguous()  # [TB,K,H,W]

        # Per-sample normalisation from set A ONLY (no held-out/unavailable-frame leak;
        # linear, so the N2N zero-mean target is unbiased). Same affine on B keeps the
        # input and the mean(B) target in one space.
        lo, scale = _affine_from_frames(setA)
        setA = (setA - lo) / scale
        setB = (setB - lo) / scale

        return {
            "setA": setA,
            "setB": setB,
            "t1": t1.contiguous(),
            "gm":  gm.contiguous(),
            "wm":  wm.contiguous(),
            "csf": csf.contiguous(),
            # Left out of every loss and out of the model input. sCoV is reported on the
            # CBF scale, and CBF is proportional to dM/M0 voxel-wise; the proportionality
            # constant drops out of a std/mean ratio, so the raw map is all that is needed.
            "m0": m0.contiguous(),
            # stable per-subject grouping key (base-subject index within this split);
            # surfaced so eval can aggregate the statistical unit at SUBJECT level.
            "subject_id": int(si),
        }
