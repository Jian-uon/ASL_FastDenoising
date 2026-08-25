#!/usr/bin/env python
"""Leakage spectrum analysis — the pre-gate for Band-Limited Guidance (BLG).

QUESTION THIS ANSWERS
---------------------
V35 routes T1 into the ASL path ONLY at 16x16 (CMF0) and 32x32 (CMF1); the
64/128 decoder levels are T1-free by module signature. BLG would go further and
make that a FREQUENCY-DOMAIN guarantee: T1 may only fill the DWT approximation
(low-pass) subband, while every detail (high-pass) subband is ASL-only by
construction.

BLG is only worth building if T1 influence CURRENTLY reaches the detail
subbands. It might not: the learned decoder upsamples from 32x32, and 7T PWI is
low-frequency dominated, so the leakage could be ~entirely low-pass — in which
case BLG would provably remove nothing (and the before/after figure would refute
the paper's own pillar).

WHAT IT MEASURES
----------------
leakage(x) = f(ASL, T1_matched)(x) - f(ASL, T1_mismatched)(x)   [same ASL frames,
same frame-selection seed => the ONLY difference is the T1 input]

A 2-level DWT (sym4) splits each brain-masked slice into
    LL2   32x32 approximation   <- the scale T1 guidance directly touches
    D2    level-2 details       (32-64 band)
    D1    level-1 details       (64-128 band)
and we report the share of leakage ENERGY in the detail subbands (D1+D2), i.e.
the part BLG would structurally remove.

Because absolute shares are hard to read on their own, the same decomposition is
run on the matched OUTPUT itself. The informative quantity is the RATIO:
    detail_share(leakage) / detail_share(output)
    ~1  -> leakage is spread across bands like the image itself (BLG bites)
    <<1 -> leakage is a low-frequency phenomenon (BLG is vacuous)

Brain masking uses the ORIGINAL subject's mask (CLAUDE.md sec.2), apodised
(distance-weighted rolloff) so the mask edge does not inject artificial
high-frequency energy into the DWT. The same apodised mask is applied to both
the leakage and the output, so the edge confound cancels in the ratio.

PRE-REGISTERED DECISION RULE (set before looking at the numbers)
    detail_share(leakage) >= 20%  -> GO   (BLG has real leakage to remove)
    10-20%                        -> MARGINAL (weigh cost vs payoff)
    < 10%                         -> NO-GO (BLG provably removes ~nothing)

Usage (mirrors scripts/test_mismatched_t1.py):
  python scripts/leakage_spectrum.py \
    --checkpoint <run>/checkpoints/best_umse_posthoc.pth \
    --config env/local/configs/win_asl_2d_home_v35_joint.yml \
    --out_dir <run>/eval_leakage_spectrum --max_subjects 16
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pywt
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.conf_data import Config                      # noqa: E402
from dataio.data_generator import DatasetGenerator       # noqa: E402
from dataio.data_classes import (                        # noqa: E402
    get_pre_asl_transform, build_cached_subjects,
)
from monai.data import partition_dataset                 # noqa: E402

# Reuse the loader / inference / pairing logic of the mismatched-T1 gate so the
# leakage measured here is the SAME object that test is scored on.
from scripts.test_mismatched_t1 import (                 # noqa: E402
    load_model, infer_volume,
)

WAVELET = "sym4"
LEVELS = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", default="./exp/leakage_spectrum")
    p.add_argument("--runner_args", default="")
    p.add_argument("--n_frames", type=int, default=6)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_subjects", type=int, default=16)
    p.add_argument("--cache_rate", type=float, default=0.0)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--device", default="")
    return p.parse_args()


def apodised_mask(brain_2d: np.ndarray, rolloff: float = 3.0) -> np.ndarray:
    """Smooth 0..1 mask: binary brain mask with a distance-transform rolloff, so
    the DWT does not see a step edge (which would manufacture detail energy)."""
    from scipy.ndimage import distance_transform_edt, binary_fill_holes
    b = binary_fill_holes(brain_2d.astype(bool))
    if not b.any():
        return np.zeros_like(brain_2d, dtype=np.float64)
    d = distance_transform_edt(b).astype(np.float64)
    return np.clip(d / max(rolloff, 1e-6), 0.0, 1.0)


def hf_reconstruct(img_2d: np.ndarray) -> np.ndarray:
    """Image-space reconstruction of the DETAIL subbands only (approximation
    zeroed) — i.e. exactly the component BLG would make T1-independent."""
    c = pywt.wavedec2(img_2d, WAVELET, level=LEVELS, mode="periodization")
    c_hf = [np.zeros_like(c[0])] + list(c[1:])
    return pywt.waverec2(c_hf, WAVELET, mode="periodization")


def grad_mag(img_2d: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img_2d.astype(np.float64))
    return np.sqrt(gx ** 2 + gy ** 2)


def masked_corr(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    """Pearson correlation over a boolean mask."""
    mm = m.astype(bool)
    if mm.sum() < 32:
        return float("nan")
    x = a[mm].astype(np.float64); y = b[mm].astype(np.float64)
    x = x - x.mean(); y = y - y.mean()
    d = float(np.sqrt((x ** 2).sum() * (y ** 2).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def band_energies(img_2d: np.ndarray) -> Dict[str, float]:
    """2-level DWT band energies (sum of squares) of a single slice."""
    coeffs = pywt.wavedec2(img_2d, WAVELET, level=LEVELS, mode="periodization")
    ll = coeffs[0]
    d2 = coeffs[1]            # (LH2, HL2, HH2) -> 32-64 band
    d1 = coeffs[2]            # (LH1, HL1, HH1) -> 64-128 band
    e_ll = float((ll ** 2).sum())
    e_d2 = float(sum((c ** 2).sum() for c in d2))
    e_d1 = float(sum((c ** 2).sum() for c in d1))
    tot = e_ll + e_d2 + e_d1
    return {"LL": e_ll, "D2": e_d2, "D1": e_d1, "total": tot,
            "detail_share": (e_d2 + e_d1) / tot if tot > 0 else float("nan")}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={device}  out={out_dir}  wavelet={WAVELET} levels={LEVELS}")

    cfg = Config(args.config)
    model = load_model(args, cfg, device)

    tp = cfg.asl_denoiser_train_params
    pre_tf = get_pre_asl_transform(asl_hw=tp.asl_hw, asl_z=tp.asl_z,
                                   t1_hw=tp.t1_hw, t1_z=tp.t1_z)
    subjects = DatasetGenerator(cfg).generate_dataset()
    cached = build_cached_subjects(subjects, pre_tf,
                                   cache_rate=float(args.cache_rate), cache_workers=4)
    splits = partition_dataset(
        data=cached,
        ratios=[cfg.data_loading.train_ratio, cfg.data_loading.val_ratio,
                cfg.data_loading.test_ratio],
        shuffle=cfg.data_loading.shuffle)
    subset = {"train": splits[0], "val": splits[1], "test": splits[2]}[args.split]

    n_sub = min(len(subset), args.max_subjects) if args.max_subjects else len(subset)
    print(f"[INFO] {n_sub} subjects from '{args.split}'")

    # Same pairing rule as the mismatched-T1 gate (shift so nobody maps to self).
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(n_sub)
    for i in range(n_sub):
        if perm[i] == i:
            j = (i + 1) % n_sub
            perm[i], perm[j] = perm[j], perm[i]

    rows: List[Dict] = []
    example = None
    for si in range(n_sub):
        item, item_mis = subset[si], subset[perm[si]]
        asl_vol = item["asl_diff"].cpu()
        t1_m = item["t1"].cpu()
        t1_x = item_mis["t1"].cpu()
        sid = item.get("subject_id", f"sub_{si:03d}")
        print(f"[{si+1}/{n_sub}] {sid}")

        seed_si = args.seed + si          # identical frames for both runs
        pwi_m, _, _, _ = infer_volume(model, asl_vol, t1_m, args.n_frames, device, seed_si)
        pwi_x, _, _, _ = infer_volume(model, asl_vol, t1_x, args.n_frames, device, seed_si)

        brain = (t1_m[0].numpy() > 0.05)
        t1m_np, t1x_np = t1_m[0].numpy(), t1_x[0].numpy()
        Z = pwi_m.shape[2]
        acc = {k: [] for k in ("leak_LL", "leak_D2", "leak_D1", "leak_detail_share",
                               "out_LL", "out_D2", "out_D1", "out_detail_share",
                               "leak_energy", "out_energy",
                               # Imprint test: does the HIGH-FREQUENCY leakage align with the
                               # edges of the WRONG T1? That T1 is another subject's anatomy and
                               # is uncorrelated with this subject's ASL, so any correlation is
                               # T1 imprint by construction (corr_hf_vs_gradT1match is confounded
                               # by genuine cross-modal anatomy agreement; corr_ref_hf_vs_gradT1mis
                               # is the null baseline and should sit near zero).
                               "corr_hf_vs_gradT1mis", "corr_hf_vs_gradT1match",
                               "corr_ref_hf_vs_gradT1mis")}
        for z in range(Z):
            bz = brain[:, :, z]
            if bz.sum() < 64:
                continue
            w = apodised_mask(bz)
            leak = (pwi_m[:, :, z] - pwi_x[:, :, z]) * w
            out = pwi_m[:, :, z] * w
            bl, bo = band_energies(leak), band_energies(out)
            if not (bl["total"] > 0 and bo["total"] > 0):
                continue
            for tag, b in (("leak", bl), ("out", bo)):
                for k in ("LL", "D2", "D1"):
                    acc[f"{tag}_{k}"].append(b[k] / b["total"])
                acc[f"{tag}_detail_share"].append(b["detail_share"])
                acc[f"{tag}_energy"].append(b["total"])
            # --- T1-imprint test on the detail component ---
            leak_hf = np.abs(hf_reconstruct(leak))
            ref_hf = np.abs(hf_reconstruct(out))
            g_mis = grad_mag(t1x_np[:, :, z] * w)
            g_mat = grad_mag(t1m_np[:, :, z] * w)
            acc["corr_hf_vs_gradT1mis"].append(masked_corr(leak_hf, g_mis, bz))
            acc["corr_hf_vs_gradT1match"].append(masked_corr(leak_hf, g_mat, bz))
            acc["corr_ref_hf_vs_gradT1mis"].append(masked_corr(ref_hf, g_mis, bz))
            if example is None and z == Z // 2:
                example = (sid, leak, out, w)

        if not acc["leak_detail_share"]:
            continue
        row = {"subject": sid}
        row.update({k: float(np.nanmean(v)) for k, v in acc.items()})
        row["detail_ratio_leak_over_out"] = (row["leak_detail_share"]
                                             / max(row["out_detail_share"], 1e-12))
        rows.append(row)

    if not rows:
        print("[ERROR] no usable slices")
        return

    import csv
    csv_path = out_dir / "leakage_spectrum.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    def agg(k):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        return float(np.nanmean(v)), float(np.nanstd(v))

    ls_m, ls_s = agg("leak_detail_share")
    os_m, os_s = agg("out_detail_share")
    rr_m, rr_s = agg("detail_ratio_leak_over_out")

    print("\n" + "=" * 68)
    print(f"LEAKAGE SPECTRUM  ({len(rows)} subjects, {WAVELET} {LEVELS}-level DWT)")
    print("=" * 68)
    for k in ("leak_LL", "leak_D2", "leak_D1"):
        m, s = agg(k)
        print(f"  {k:<22}: {m*100:6.2f}% +/- {s*100:.2f}")
    for k in ("out_LL", "out_D2", "out_D1"):
        m, s = agg(k)
        print(f"  {k:<22}: {m*100:6.2f}% +/- {s*100:.2f}")
    print("-" * 68)
    print(f"  detail share  LEAKAGE : {ls_m*100:6.2f}% +/- {ls_s*100:.2f}   <-- BLG would remove this")
    print(f"  detail share  OUTPUT  : {os_m*100:6.2f}% +/- {os_s*100:.2f}")
    print(f"  ratio leak/out        : {rr_m:6.3f} +/- {rr_s:.3f}")
    print("-" * 68)
    print("  T1-IMPRINT TEST on the detail component (Pearson r, brain-masked)")
    for k, lab in (("corr_hf_vs_gradT1mis",     "HF leak  vs |grad wrong-T1|  <-- imprint"),
                   ("corr_hf_vs_gradT1match",   "HF leak  vs |grad right-T1|  (confounded)"),
                   ("corr_ref_hf_vs_gradT1mis", "HF out   vs |grad wrong-T1|  (null base)")):
        m, s = agg(k)
        print(f"    {lab:<44}: {m:+.4f} +/- {s:.4f}")
    verdict = ("GO (BLG has real high-frequency leakage to remove)" if ls_m >= 0.20
               else "MARGINAL (weigh cost vs payoff)" if ls_m >= 0.10
               else "NO-GO (leakage is ~low-frequency; BLG would remove almost nothing)")
    print(f"\n  PRE-REGISTERED VERDICT: {verdict}")
    print("=" * 68)

    # Figure: per-band shares (leakage vs output) + one example leakage map.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    bands = ["LL", "D2", "D1"]
    lm = [agg(f"leak_{b}")[0] * 100 for b in bands]
    om = [agg(f"out_{b}")[0] * 100 for b in bands]
    le = [agg(f"leak_{b}")[1] * 100 for b in bands]
    oe = [agg(f"out_{b}")[1] * 100 for b in bands]
    x = np.arange(len(bands)); wdt = 0.38
    axes[0].bar(x - wdt/2, lm, wdt, yerr=le, capsize=3, label="leakage (matched-mismatched)")
    axes[0].bar(x + wdt/2, om, wdt, yerr=oe, capsize=3, label="output (matched)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(["LL (<=32)", "D2 (32-64)", "D1 (64-128)"])
    axes[0].set_ylabel("share of energy (%)"); axes[0].legend(fontsize=8)
    axes[0].set_title("Band energy distribution")
    axes[1].hist([r["leak_detail_share"] * 100 for r in rows], bins=12)
    axes[1].axvline(20, color="r", ls="--", label="GO threshold 20%")
    axes[1].set_xlabel("leakage detail share (%)"); axes[1].set_ylabel("subjects")
    axes[1].legend(fontsize=8); axes[1].set_title("Per-subject leakage detail share")
    if example is not None:
        sid, leak, out, w = example
        v = float(np.abs(leak).max()) or 1e-6
        im = axes[2].imshow(leak, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[2].set_title(f"leakage map ({sid}, mid slice)"); axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f"T1 leakage spectrum — {Path(args.checkpoint).parent.parent.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "leakage_spectrum.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {csv_path}\n[done] {out_dir/'leakage_spectrum.png'}")


if __name__ == "__main__":
    main()
