"""Multi-checkpoint validation + one-standard-error feasibility set.

Pipeline (Option B):
  1. Build Runner via the same CLI args used to train (passed via --runner_args JSON
     OR positional --config/--name/... here matching the original launch).
  2. For each checkpoint in --ckpts, load EMA weights (validation uses EMA in training).
  3. Run validate() with per-batch uMSE component capture (sum_sq, sum_vc, n).
  4. Aggregate per-subject if subject_id available in batch metadata; else per-batch.
  5. Bootstrap N times over the aggregation units → SE of pooled uMSE at each ckpt.
  6. min_umse + 1·SE = threshold; feasible set = ckpts ≤ threshold.
  7. Dump JSON for build_swa_feasible.py to consume.

Output JSON schema:
{
  "config_path": "...",
  "ckpt_dir": "...",
  "bootstrap_n": 1000,
  "aggregation_level": "subject" or "batch",
  "results": [
      {"ckpt": "step000025.pth", "pooled_umse": 0.123,
       "per_unit": [...], "psnr_b": 23.5, "cyc": 0.04, ...},
      ...
  ],
  "min_umse": 0.118,
  "se_at_min": 0.004,
  "threshold_umse": 0.122,
  "feasible_ckpts": ["step000050.pth", "step000075.pth", ...]
}
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import torch

# Make sibling packages importable
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO))


def _build_runner(runner_args_str: str):
    """Build Runner using the same arg parser as training. Inject --max_steps 1
    + --resume off so __init__ completes without entering training."""
    from runners.asl_t1_guided_runner_dmvae_n2n import parse_args, Runner  # type: ignore

    argv = shlex.split(runner_args_str)
    # Force minimal training-side flags so Runner builds cleanly.
    if "--max_steps" not in argv:
        argv.extend(["--max_steps", "1"])
    if "--resume" in argv:
        argv.remove("--resume")
    # Save original sys.argv, replace, parse, restore
    orig_argv = sys.argv
    try:
        sys.argv = ["scripts/eval_select_ckpt.py"] + argv
        args = parse_args()
    finally:
        sys.argv = orig_argv

    runner = Runner(args)
    return runner, args


def _load_model_weights(runner, ckpt_path: str):
    """Load model + EMA state from ckpt. Validation uses EMA."""
    sd = torch.load(ckpt_path, map_location=runner.device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        runner.model.load_state_dict(sd["model"], strict=True)
    if isinstance(sd, dict) and "ema" in sd:
        runner.ema.ema_state = {k: v.to(runner.device) for k, v in sd["ema"].items()}
        runner.ema.optimization_step = sd.get("ema_optimization_step", 0)


def _grad_mag(x):
    """Gradient magnitude of [B,1,H,W] via forward differences (no padding deps)."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _masked_pearson(a, b, mask):
    """Pearson r between a and b over mask>0.5."""
    m = mask > 0.5
    av = a[m]
    bv = b[m]
    if av.numel() < 2:
        return float("nan")
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = (av.norm() * bv.norm()).clamp_min(1e-8)
    return float((av * bv).sum() / denom)


def _t1_structural_sim(img, t1, mask):
    """Cross-modal T1-leak guardrail: Pearson r between |grad(img)| and |grad(T1)|
    in the brain. Higher = img edges align more with T1 anatomy. Compared against
    the 12-NEX union's own T1 correlation; pred >> ref signals anatomical T1 leak."""
    return _masked_pearson(_grad_mag(img), _grad_mag(t1), mask)


def _masked_mean(img, mask, thr=0.5):
    m = (mask > thr).float()
    return (img * m).sum() / m.sum().clamp_min(1.0)


def _masked_std(img, mask, thr=0.5):
    m = (mask > thr).float()
    n = m.sum().clamp_min(1.0)
    mu = (img * m).sum() / n
    return (((img - mu) ** 2) * m).sum().div(n).clamp_min(1e-12).sqrt()


def _highpass(img, k=9):
    import torch.nn.functional as F
    return img - F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)


def _line_corr(img, brain):
    """MLC: adjacent-line Pearson on the high-pass residual. Returns (vcorr, hcorr)
    = vertical-neighbour and horizontal-neighbour correlation in brain. A coherent
    line/stripe artifact (incl. MoSSM scan-direction bias) shows up as high one-axis
    correlation; isotropic noise → both ≈ 0."""
    r = _highpass(img)
    mv = ((brain[:, :, :-1, :] > 0.5) & (brain[:, :, 1:, :] > 0.5)).float()
    vcorr = _masked_pearson(r[:, :, :-1, :], r[:, :, 1:, :], mv)
    mh = ((brain[:, :, :, :-1] > 0.5) & (brain[:, :, :, 1:] > 0.5)).float()
    hcorr = _masked_pearson(r[:, :, :, :-1], r[:, :, :, 1:], mh)
    return vcorr, hcorr


def _ghost_corr(img, brain):
    """MSLC: Nyquist/ghost detector. Correlation between the image and its half-FOV
    shifted copy, measured where the shifted brain lands OUTSIDE the real brain (the
    N/2 ghost region). High = a half-FOV-shifted brain copy (ghost) is present."""
    best = float("nan")
    H, W = img.shape[-2], img.shape[-1]
    for dim, s in ((2, H // 2), (3, W // 2)):
        shifted = torch.roll(img, shifts=s, dims=dim)
        sbrain = torch.roll(brain, shifts=s, dims=dim)
        region = ((sbrain > 0.5) & (brain <= 0.5)).float()
        if region.sum() < 2:
            continue
        g = _masked_pearson(img, shifted, region)
        if g == g:
            best = g if best != best else max(best, g)
    return best


def _nframe_sweep(runner, n_list=(2, 4, 6, 8), seed=42) -> dict:
    """n-frame inference sweep: forward with a per-sample RANDOM n-subset of the valid
    (setA∪setB) frames for each n, and track within-tissue homogeneity, contrast, and
    mean drift. A faithful denoiser should improve/stay-flat with more frames:
        sCoV_GM(n)  ↓ or flat   (more frames → less noise)
        CNR(n)      not collapsing (≥ ~flat)
        mean(n)     little drift (|mean(n_max)-mean(n_min)| / |mean(n_min)| small)
    Returns per-n pooled values + linear slopes vs n + the mean-drift fraction.

    2026-06-14: replaced the old `all_fr[:, :n]` selection, which (a) took the FIRST n
    frames — confounded with acquisition order / bad-frame ordering (docs/cada_lr_design.md
    §9) — and (b) included setA padding zeros when lenA < maxTA. Now selects n random
    indices from each sample's own valid frames (seeded → reproducible across ckpts).
    """
    import numpy as _np
    from runners.asl_t1_guided_runner_dmvae_n2n import prepare_asl_pair_batch
    from utils.metrics import scov as _scov, cnr as _cnr
    acc = {n: {"scov": [], "cnr": [], "mean": []} for n in n_list}
    with torch.no_grad():
        for bi, vb in enumerate(runner.val_loader):
            pack = prepare_asl_pair_batch(vb, runner.device)
            pack = runner._mask_asl_inputs(pack)
            runner._apply_cond_pv(runner._unwrap(), pack)   # cond_src='fsl' (no-op otherwise)
            t1 = pack["t1"]
            mb = (t1 > 0.05).float()
            setA, setB = pack["setA"], pack["setB"]
            lenA, lenB = pack["lenA"], pack["lenB"]
            B = setA.shape[0]
            tot = lenA + lenB
            tmin = int(tot.min().item())
            gm, wm = pack.get("gm"), pack.get("wm")
            for n in n_list:
                if tmin < n:
                    continue
                sub = torch.zeros(B, n, *setA.shape[2:], device=setA.device, dtype=setA.dtype)
                for i in range(B):
                    la, lb = int(lenA[i].item()), int(lenB[i].item())
                    valid = torch.cat([setA[i, :la], setB[i, :lb]], dim=0)  # [la+lb,1,H,W], no padding
                    g = torch.Generator().manual_seed((int(seed) * 100003 + bi * 1009 + n * 31 + i * 7) & 0x7FFFFFFF)
                    idx = torch.randperm(valid.shape[0], generator=g)[:n].sort().values.to(valid.device)
                    sub[i] = valid[idx]
                lens = torch.full((B,), n, device=setA.device, dtype=lenA.dtype)
                out = runner._predict_with_ema(sub, t1, lens)["asl_recon"]
                if gm is not None:
                    acc[n]["scov"].append(_scov(out, gm, threshold=0.5))
                    if wm is not None:
                        acc[n]["cnr"].append(_cnr(out, gm, wm, threshold=0.5))
                acc[n]["mean"].append(float((out * mb).sum() / mb.sum().clamp_min(1.0)))
    ns = [n for n in n_list if len(acc[n]["mean"]) > 0]

    def _avg(xs):
        return float(_np.mean(xs)) if len(xs) else float("nan")

    scov_v = [_avg(acc[n]["scov"]) for n in ns]
    cnr_v = [_avg(acc[n]["cnr"]) for n in ns]
    mean_v = [_avg(acc[n]["mean"]) for n in ns]

    def _slope(xs, ys):
        xs = _np.asarray(xs, float); ys = _np.asarray(ys, float)
        m = _np.isfinite(ys)
        return float(_np.polyfit(xs[m], ys[m], 1)[0]) if int(m.sum()) >= 2 else float("nan")

    mean_drift = (abs(mean_v[-1] - mean_v[0]) / (abs(mean_v[0]) + 1e-8)
                  if len(mean_v) >= 2 else float("nan"))
    return {
        "n": ns, "scov_gm": scov_v, "cnr": cnr_v, "mean": mean_v,
        "scov_slope": _slope(ns, scov_v),   # ≤0 good
        "cnr_slope": _slope(ns, cnr_v),     # ≥~0 good (no collapse)
        "mean_drift": mean_drift,           # small good
    }


def _slice_radiomic_features(img2d, mask2d, levels: int = 16):
    """Per-slice radiomic-style feature vector over brain voxels.
    first-order (9) + GLCM texture (5). Returns 1-D np.array or None.
    NOTE: skimage first-order+GLCM features, NOT the pyradiomics IBSI set —
    a faithful *radiomic-feature Fréchet distance*, not the published FRD exactly
    (pyradiomics will not build on this py3.10 env).
    """
    import numpy as _np
    from scipy import stats as _st
    from skimage.feature import graycomatrix, graycoprops
    m = mask2d > 0.5
    vox = img2d[m]
    if vox.size < 32:
        return None
    f = [float(vox.mean()), float(vox.std()), float(_st.skew(vox)),
         float(_st.kurtosis(vox)), float(_np.percentile(vox, 10)),
         float(_np.percentile(vox, 50)), float(_np.percentile(vox, 90)),
         float((vox ** 2).mean())]
    h, _ = _np.histogram(vox, bins=levels)
    p = h[h > 0] / max(h.sum(), 1)
    f.append(float(-(p * _np.log(p)).sum()))
    vmin, vmax = float(vox.min()), float(vox.max())
    if vmax - vmin < 1e-8:
        f += [0.0] * 5
        return _np.asarray(f, float)
    q = _np.zeros_like(img2d, dtype=_np.uint8)
    q[m] = _np.clip(((img2d[m] - vmin) / (vmax - vmin) * (levels - 1)).astype(_np.uint8), 0, levels - 1)
    glcm = graycomatrix(q, distances=[1], angles=[0.0, _np.pi / 2],
                        levels=levels, symmetric=True, normed=True)
    for prop in ("contrast", "correlation", "homogeneity", "energy", "ASM"):
        f.append(float(graycoprops(glcm, prop).mean()))
    return _np.asarray(f, float)


def _frd(runner, levels: int = 16) -> dict:
    """Fréchet Radiomic Distance (skimage-feature variant) between the recon
    distribution and the 12-NEX union distribution over all val slices.
    FRD = ||μ_r−μ_u||² + Tr(Σ_r+Σ_u−2√(Σ_rΣ_u)) on standardised features.
    Lower = recon distribution closer to the reference. Also reports FRD(union,
    union-split) ~ 0 sanity is not done here; interpret FRD relative across ckpts.
    """
    import numpy as _np
    from scipy.linalg import sqrtm
    from runners.asl_t1_guided_runner_dmvae_n2n import (
        prepare_asl_pair_batch, direct_mean_from_frames)
    recon_feats, union_feats = [], []
    with torch.no_grad():
        for vb in runner.val_loader:
            pack = prepare_asl_pair_batch(vb, runner.device)
            pack = runner._mask_asl_inputs(pack)
            runner._apply_cond_pv(runner._unwrap(), pack)   # cond_src='fsl' (no-op otherwise)
            t1 = pack["t1"]
            mb = (t1 > 0.05).float()
            out = runner._predict_with_ema(pack["setA"], t1, pack["lenA"], pack.get("maskA"))["asl_recon"]
            _ctx = int(getattr(runner.args, "slice_context", 0))   # 2.5D: center-slice frames for the ref
            if _ctx > 0:
                _n = 2 * _ctx + 1; _ic = pack["setA"].shape[2] // _n; _c0 = _ic * _ctx
                pack["setA"] = pack["setA"][:, :, _c0:_c0 + _ic]
                pack["setB"] = pack["setB"][:, :, _c0:_c0 + _ic]
            union = direct_mean_from_frames(
                torch.cat([pack["setA"], pack["setB"]], dim=1), pack["lenA"] + pack["lenB"])
            o = out.detach().cpu().numpy(); u = union.detach().cpu().numpy()
            mm = mb.detach().cpu().numpy()
            for b in range(o.shape[0]):
                fr = _slice_radiomic_features(o[b, 0], mm[b, 0], levels)
                fu = _slice_radiomic_features(u[b, 0], mm[b, 0], levels)
                if fr is not None and fu is not None:
                    recon_feats.append(fr); union_feats.append(fu)
    if len(recon_feats) < 8:
        return {"frd": float("nan"), "n_slices": len(recon_feats)}
    R = _np.stack(recon_feats); U = _np.stack(union_feats)
    both = _np.concatenate([R, U], 0)
    mu, sd = both.mean(0), both.std(0) + 1e-8
    R = (R - mu) / sd; U = (U - mu) / sd
    mr, mu_u = R.mean(0), U.mean(0)
    cr = _np.cov(R, rowvar=False) + 1e-6 * _np.eye(R.shape[1])
    cu = _np.cov(U, rowvar=False) + 1e-6 * _np.eye(U.shape[1])
    covmean = sqrtm(cr @ cu)
    if _np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mr - mu_u
    frd = float(diff @ diff + _np.trace(cr + cu - 2.0 * covmean))
    return {"frd": frd, "n_slices": len(recon_feats)}


def _validate_with_per_unit_umse(runner) -> dict:
    """Run validate() but capture per-batch (or per-subject) uMSE components in
    addition to the pooled mean. Returns dict with both pooled metrics and a
    list of per-unit dict entries.

    Implementation strategy: monkey-patch the val loop to also write per-batch
    uMSE components to a side accumulator. Rather than rewriting validate(), we
    use a lightweight inline replica focused on the metrics we need.
    """
    from utils.metrics import (upsnr_components, scov, laplacian_variance, cnr, hfen,
                               mi_nmi, entropy_focus_criterion, bright_tail_ratio)
    from runners.asl_t1_guided_runner_dmvae_n2n import (
        prepare_asl_pair_batch, direct_mean_from_frames,
    )
    runner.model.eval()
    per_unit = []
    sum_sq_all = 0.0
    sum_vc_all = 0.0
    n_all = 0.0
    scov_gm_sum = 0.0
    scov_gm_cnt = 0
    scov_wm_sum = 0.0
    scov_wm_cnt = 0
    cnr_sum = 0.0
    cnr_ref_sum = 0.0
    cnr_cnt = 0
    t1sim_sum = 0.0
    t1sim_ref_sum = 0.0
    t1sim_cnt = 0
    # MI/NMI (cross-modal information): recon↔T1, recon↔union, and the
    # union↔T1 baseline (natural anatomy correlation of the noisy reference).
    mi_t1_sum = 0.0; nmi_t1_sum = 0.0
    mi_un_sum = 0.0; nmi_un_sum = 0.0
    mi_ut1_sum = 0.0; nmi_ut1_sum = 0.0
    mi_cnt = 0
    efc_sum = 0.0; efc_ref_sum = 0.0; bright_sum = 0.0; efc_cnt = 0
    snr_gm_sum = 0.0
    snr_wm_sum = 0.0
    snr_gm_ref_sum = 0.0
    snr_wm_ref_sum = 0.0
    snr_cnt = 0
    hfen_sum = 0.0
    hfen_cnt = 0
    cyc_sum = 0.0
    cyc_cnt = 0
    hfcyc_sum = 0.0
    hfcyc_cnt = 0
    mlc_sum = 0.0
    mlc_aniso_sum = 0.0
    mlc_cnt = 0
    mslc_sum = 0.0
    mslc_cnt = 0
    lapvr_sum = 0.0
    lapvr_cnt = 0
    with torch.no_grad():
        for vb in runner.val_loader:
            pack = prepare_asl_pair_batch(vb, runner.device)
            pack = runner._mask_asl_inputs(pack)
            runner._apply_cond_pv(runner._unwrap(), pack)   # cond_src='fsl' (no-op otherwise)
            t1_for_model = pack["t1"]
            pred = runner._predict_with_ema(pack["setA"], t1_for_model, pack["lenA"], pack.get("maskA"))
            out = pred["asl_recon"]
            # 2.5D: the model consumed the full K-slice window above; reduce the frame
            # tensors to the CENTER slice so union/upsnr/scov/cnr below match out [B,1,H,W].
            # Keep the FULL K-slice windows (setA_full/setB_full) for the EXTRA model
            # forwards below (cyc on setA halves, hfcyc on setB) — those re-enter the
            # 2.5D model and need the K-channel input, not the center slice.
            setA_full, setB_full = pack["setA"], pack["setB"]
            _ctx = int(getattr(runner.args, "slice_context", 0))
            if _ctx > 0:
                _n = 2 * _ctx + 1; _ic = pack["setA"].shape[2] // _n; _c0 = _ic * _ctx
                pack["setA"] = pack["setA"][:, :, _c0:_c0 + _ic]
                pack["setB"] = pack["setB"][:, :, _c0:_c0 + _ic]
            # uPSNR components: 3-way split of set_b for unbiased estimator
            # Returns (sum_sq_err, sum_var_corr, n_pixels, nf_l1_sum)
            mask_brain = (pack["t1"] > 0.05).float()
            ssq, svc, n, _ = upsnr_components(
                out, pack["setB"], pack.get("lenB"), mask=mask_brain,
            )
            sum_sq = float(ssq)
            sum_vc = float(svc)
            n = float(n)
            umse_batch = max((sum_sq - sum_vc) / max(n, 1.0), 1e-12)
            # scov_gm + lapvar_ratio — matched to runner.validate() definitions:
            #   union  = mean of ALL frames (setA ∪ setB), the 12-NEX reference
            #   lapvR  = lapvar(recon) / lapvar(union) inside brain mask
            #   scovGM = sCoV of recon inside GM PV mask (>0.5)
            union = direct_mean_from_frames(
                torch.cat([pack["setA"], pack["setB"]], dim=1),
                pack["lenA"] + pack["lenB"],
            )
            ref_lapvar = laplacian_variance(union, mask_brain)
            pred_lapvar = laplacian_variance(out, mask_brain)
            lapvr_batch = pred_lapvar / max(ref_lapvar, 1e-8)
            scov_gm_batch = float("nan")
            if pack.get("gm") is not None:
                scov_gm_batch = scov(out, pack["gm"], threshold=0.5)
                scov_gm_sum += scov_gm_batch
                scov_gm_cnt += 1
            scov_wm_batch = float("nan")
            if pack.get("wm") is not None:
                scov_wm_batch = scov(out, pack["wm"], threshold=0.5)
                scov_wm_sum += scov_wm_batch
                scov_wm_cnt += 1
            cnr_batch = float("nan")
            if pack.get("gm") is not None and pack.get("wm") is not None:
                cnr_batch = cnr(out, pack["gm"], pack["wm"], threshold=0.5)
                cnr_sum += cnr_batch
                cnr_ref_sum += cnr(union, pack["gm"], pack["wm"], threshold=0.5)
                cnr_cnt += 1
            # Cross-modal T1-leak guardrail: gradient-correlation(recon, T1) vs the
            # union's own gradient-correlation(union, T1). pred >> ref ⇒ T1 leak.
            t1sim_batch = _t1_structural_sim(out, t1_for_model, mask_brain)
            t1sim_ref_batch = _t1_structural_sim(union, t1_for_model, mask_brain)
            if t1sim_batch == t1sim_batch:
                t1sim_sum += t1sim_batch
                t1sim_ref_sum += t1sim_ref_batch
                t1sim_cnt += 1
            # MI / NMI (cross-modal mutual information, scale-invariant histogram):
            #   recon↔T1  : info shared with anatomy (high ⇒ anatomy-aligned OR leak)
            #   recon↔union: info shared with the noisy reference perfusion
            #   union↔T1  : baseline — the TRUE reference's natural anatomy MI.
            # recon↔T1 ≈ union↔T1 ⇒ natural; recon↔T1 ≫ union↔T1 ⇒ excess (leak).
            mi_t1_b, nmi_t1_b = mi_nmi(out, t1_for_model, mask_brain)
            mi_un_b, nmi_un_b = mi_nmi(out, union, mask_brain)
            mi_ut1_b, nmi_ut1_b = mi_nmi(union, t1_for_model, mask_brain)
            if mi_t1_b == mi_t1_b:
                mi_t1_sum += mi_t1_b; nmi_t1_sum += nmi_t1_b
                mi_un_sum += mi_un_b; nmi_un_sum += nmi_un_b
                mi_ut1_sum += mi_ut1_b; nmi_ut1_sum += nmi_ut1_b
                mi_cnt += 1
            # EFC (Atkinson 1997, lower=sharper) on recon vs union baseline, and
            # bright_tail_ratio (p99/p50) on recon — focal hot-spot / hallucination.
            efc_b = entropy_focus_criterion(out, mask_brain)
            efc_ref_b = entropy_focus_criterion(union, mask_brain)
            bright_b = bright_tail_ratio(out, mask_brain)
            if efc_b == efc_b:
                efc_sum += efc_b; efc_ref_sum += efc_ref_b
                if bright_b == bright_b:
                    bright_sum += bright_b
                efc_cnt += 1
            # GM/WM SNR with CSF as the noise region (CSF ΔM≈0 ⇒ σ_CSF = noise):
            #   SNR_tissue = μ_tissue / σ_CSF, on the same image. Also computed on
            #   the noisy union as a baseline (recon SNR ≫ union SNR = denoising gain).
            snr_gm_b = snr_wm_b = snr_gm_ref_b = snr_wm_ref_b = float("nan")
            if (pack.get("csf") is not None and pack.get("gm") is not None
                    and pack.get("wm") is not None):
                sig_csf = _masked_std(out, pack["csf"]).clamp_min(1e-8)
                snr_gm_b = float(_masked_mean(out, pack["gm"]) / sig_csf)
                snr_wm_b = float(_masked_mean(out, pack["wm"]) / sig_csf)
                sig_csf_ref = _masked_std(union, pack["csf"]).clamp_min(1e-8)
                snr_gm_ref_b = float(_masked_mean(union, pack["gm"]) / sig_csf_ref)
                snr_wm_ref_b = float(_masked_mean(union, pack["wm"]) / sig_csf_ref)
                snr_gm_sum += snr_gm_b
                snr_wm_sum += snr_wm_b
                snr_gm_ref_sum += snr_gm_ref_b
                snr_wm_ref_sum += snr_wm_ref_b
                snr_cnt += 1
            # HFEN vs the 12-NEX union (reference-based, high-freq band). NOTE:
            # union is noisy, so lower HFEN partly = matching the union's noise →
            # reported for literature alignment, NOT used for selection.
            hfen_b = float(hfen(out, union, mask_brain))
            hfen_sum += hfen_b
            hfen_cnt += 1
            # Subset self-consistency (cyc) for upsnr_cyc: L1 between f(setA[:k])
            # and f(setA[k:2k]) — two extra forwards on disjoint setA halves
            # (matches runner.validate()).
            T_a_min = int(pack["lenA"].min().item())
            if T_a_min >= 4:
                k = T_a_min // 2
                len_a1 = torch.full_like(pack["lenA"], k)
                p1 = runner._predict_with_ema(setA_full[:, :k], t1_for_model, len_a1)["asl_recon"]
                p2 = runner._predict_with_ema(setA_full[:, k:2 * k], t1_for_model, len_a1)["asl_recon"]
                cyc_sum += float(((p1 - p2).abs() * mask_brain).sum()
                                 / mask_brain.sum().clamp_min(1.0))
                cyc_cnt += 1
            # HF-consistency: high-freq error between f(setA) and f(setB), pred-to-pred
            # (noise independent across subsets). Low = high-freq reproducible across
            # subsets = REAL high-freq structure; high = high-freq is noise/unstable.
            pred_b = runner._predict_with_ema(setB_full, t1_for_model,
                                              pack.get("lenB"), pack.get("maskB"))["asl_recon"]
            hfcyc_sum += float(hfen(out, pred_b, mask_brain))
            hfcyc_cnt += 1
            # MLC (line/stripe; high-pass adjacent-line corr) + MSLC (N/2 ghost) on recon.
            vcorr, hcorr = _line_corr(out, mask_brain)
            if vcorr == vcorr and hcorr == hcorr:
                mlc_sum += max(vcorr, hcorr)
                mlc_aniso_sum += abs(vcorr - hcorr)
                mlc_cnt += 1
            mslc_b = _ghost_corr(out, mask_brain)
            if mslc_b == mslc_b:
                mslc_sum += mslc_b
                mslc_cnt += 1
            lapvr_sum += lapvr_batch
            lapvr_cnt += 1
            # Try to recover subject_id from batch metadata
            subj_id = None
            for k in ("subject_id", "case_id"):
                if k in vb and vb[k] is not None:
                    v = vb[k]
                    subj_id = v[0] if isinstance(v, (list, tuple)) else str(v)
                    break
            per_unit.append({
                "subject_id": subj_id,
                "sum_sq": sum_sq,
                "sum_vc": sum_vc,
                "n": n,
                "umse_batch": umse_batch,
                "scov_gm_batch": scov_gm_batch,
                "scov_wm_batch": scov_wm_batch,
                "scov_gmwm_batch": 0.5 * (scov_gm_batch + scov_wm_batch),
                "cnr_batch": cnr_batch,
                "t1sim_batch": t1sim_batch,
                "lapvr_batch": lapvr_batch,
            })
            sum_sq_all += sum_sq
            sum_vc_all += sum_vc
            n_all += n
    # SIGNED pooled uMSE (do NOT clamp to a tiny positive — a clamp here would defeat
    # the `pooled_umse > 0` guard below and fabricate a ~120 dB pooled_upsnr from a
    # near-zero/negative finite-sample risk). Negative = better (lower risk); argmin
    # selection handles it correctly.
    pooled_umse = (sum_sq_all - sum_vc_all) / max(n_all, 1.0)
    pooled_scov_gm = (scov_gm_sum / scov_gm_cnt) if scov_gm_cnt > 0 else float("nan")
    pooled_scov_wm = (scov_wm_sum / scov_wm_cnt) if scov_wm_cnt > 0 else float("nan")
    pooled_lapvr = (lapvr_sum / lapvr_cnt) if lapvr_cnt > 0 else float("nan")
    pooled_cnr = (cnr_sum / cnr_cnt) if cnr_cnt > 0 else float("nan")
    pooled_cnr_ref = (cnr_ref_sum / cnr_cnt) if cnr_cnt > 0 else float("nan")
    pooled_t1sim = (t1sim_sum / t1sim_cnt) if t1sim_cnt > 0 else float("nan")
    pooled_t1sim_ref = (t1sim_ref_sum / t1sim_cnt) if t1sim_cnt > 0 else float("nan")
    _mc = mi_cnt if mi_cnt > 0 else 1
    pooled_mi_t1 = (mi_t1_sum / _mc) if mi_cnt > 0 else float("nan")
    pooled_nmi_t1 = (nmi_t1_sum / _mc) if mi_cnt > 0 else float("nan")
    pooled_mi_union = (mi_un_sum / _mc) if mi_cnt > 0 else float("nan")
    pooled_nmi_union = (nmi_un_sum / _mc) if mi_cnt > 0 else float("nan")
    pooled_mi_ut1 = (mi_ut1_sum / _mc) if mi_cnt > 0 else float("nan")
    pooled_nmi_ut1 = (nmi_ut1_sum / _mc) if mi_cnt > 0 else float("nan")
    pooled_efc = (efc_sum / efc_cnt) if efc_cnt > 0 else float("nan")
    pooled_efc_ref = (efc_ref_sum / efc_cnt) if efc_cnt > 0 else float("nan")
    pooled_bright_tail = (bright_sum / efc_cnt) if efc_cnt > 0 else float("nan")
    pooled_snr_gm = (snr_gm_sum / snr_cnt) if snr_cnt > 0 else float("nan")
    pooled_snr_wm = (snr_wm_sum / snr_cnt) if snr_cnt > 0 else float("nan")
    pooled_snr_gm_ref = (snr_gm_ref_sum / snr_cnt) if snr_cnt > 0 else float("nan")
    pooled_snr_wm_ref = (snr_wm_ref_sum / snr_cnt) if snr_cnt > 0 else float("nan")
    pooled_hfen = (hfen_sum / hfen_cnt) if hfen_cnt > 0 else float("nan")
    pooled_cyc = (cyc_sum / cyc_cnt) if cyc_cnt > 0 else float("nan")
    pooled_hfcyc = (hfcyc_sum / hfcyc_cnt) if hfcyc_cnt > 0 else float("nan")
    pooled_mlc = (mlc_sum / mlc_cnt) if mlc_cnt > 0 else float("nan")
    pooled_mlc_aniso = (mlc_aniso_sum / mlc_cnt) if mlc_cnt > 0 else float("nan")
    pooled_mslc = (mslc_sum / mslc_cnt) if mslc_cnt > 0 else float("nan")
    pooled_upsnr = float(10.0 * np.log10(1.0 / pooled_umse)) if pooled_umse > 0 else float("nan")
    pooled_upsnr_cyc = (pooled_upsnr - 30.0 * pooled_cyc) if pooled_cyc == pooled_cyc else pooled_upsnr
    return {
        "pooled_umse": pooled_umse,
        "pooled_cyc": pooled_cyc,
        "pooled_hfcyc": pooled_hfcyc,
        "pooled_mlc": pooled_mlc,
        "pooled_mlc_aniso": pooled_mlc_aniso,
        "pooled_mslc": pooled_mslc,
        "pooled_upsnr": pooled_upsnr,
        "pooled_upsnr_cyc": pooled_upsnr_cyc,
        "pooled_scov_gm": pooled_scov_gm,
        "pooled_scov_wm": pooled_scov_wm,
        "pooled_scov_gmwm": 0.5 * (pooled_scov_gm + pooled_scov_wm),
        "pooled_cnr": pooled_cnr,
        "pooled_cnr_ref": pooled_cnr_ref,
        "pooled_t1sim": pooled_t1sim,
        "pooled_t1sim_ref": pooled_t1sim_ref,
        "pooled_mi_t1": pooled_mi_t1,
        "pooled_nmi_t1": pooled_nmi_t1,
        "pooled_mi_union": pooled_mi_union,
        "pooled_nmi_union": pooled_nmi_union,
        "pooled_mi_ut1": pooled_mi_ut1,
        "pooled_nmi_ut1": pooled_nmi_ut1,
        "pooled_efc": pooled_efc,
        "pooled_efc_ref": pooled_efc_ref,
        "pooled_bright_tail": pooled_bright_tail,
        "pooled_snr_gm": pooled_snr_gm,
        "pooled_snr_wm": pooled_snr_wm,
        "pooled_snr_gm_ref": pooled_snr_gm_ref,
        "pooled_snr_wm_ref": pooled_snr_wm_ref,
        "pooled_hfen": pooled_hfen,
        "pooled_lapvr": pooled_lapvr,
        "per_unit": per_unit,
    }


def _aggregate_per_subject(per_unit):
    """Group per-unit entries by subject_id → ONE record per subject: uMSE pooled
    over the subject's pixels (sum_sq/sum_vc/n) + mean sCoV/CNR over its slices, so
    the statistical UNIT is the subject (not the batch) for every selection metric
    and downstream SE/bootstrap. Falls back to per-unit if no subject_id present.
    Returns (records, level) where each record is a dict with sid/umse/scov*/cnr.
    NOTE: subject_id may be int 0 (falsy) — test `is not None`, never truthiness.
    """
    have_subjects = any(u.get("subject_id") is not None for u in per_unit)

    def _mean(entries, key):
        vals = [e[key] for e in entries if e.get(key) == e.get(key)]  # drop NaN
        return float(np.mean(vals)) if vals else float("nan")

    if not have_subjects:
        recs = [{"sid": f"unit_{i}", "umse": u["umse_batch"],
                 "scov_gm": u["scov_gm_batch"], "scov_wm": u["scov_wm_batch"],
                 "scov_gmwm": u["scov_gmwm_batch"], "cnr": u["cnr_batch"]}
                for i, u in enumerate(per_unit)]
        return recs, "unit"
    groups = {}
    for u in per_unit:
        sid = u["subject_id"] if u.get("subject_id") is not None else "_unknown"
        groups.setdefault(sid, []).append(u)
    recs = []
    for sid, entries in groups.items():
        s_sq = sum(e["sum_sq"] for e in entries)
        s_vc = sum(e["sum_vc"] for e in entries)
        n = sum(e["n"] for e in entries)
        recs.append({
            "sid": sid,
            "umse": (s_sq - s_vc) / max(n, 1.0),          # SIGNED (no 1e-12 clamp)
            "scov_gm": _mean(entries, "scov_gm_batch"),
            "scov_wm": _mean(entries, "scov_wm_batch"),
            "scov_gmwm": _mean(entries, "scov_gmwm_batch"),
            "cnr": _mean(entries, "cnr_batch"),
        })
    return recs, "subject"


def _bootstrap_se(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> float:
    """Estimate SE of the mean of `values` by resampling with replacement."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan")
    samples = rng.choice(values, size=(n_boot, n), replace=True)
    means = samples.mean(axis=1)
    return float(means.std(ddof=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="List of ckpt paths to evaluate.")
    p.add_argument("--runner_args", required=True,
                   help="Single string of CLI flags to pass to runner's parse_args. "
                        "Use the same flags as the original training launch, MINUS "
                        "--max_steps and --resume (they will be ignored/overridden).")
    p.add_argument("--bootstrap_n", type=int, default=1000)
    p.add_argument("--feasible_se", type=float, default=1.0,
                   help="Threshold = min_metric + feasible_se * SE_at_min.")
    p.add_argument("--metric", choices=["umse", "scov_gm", "scov_umse1se", "cnr", "t1_umse1se"], default="umse",
                   help="Selection metric. 'umse' (lower=better): unbiased risk. "
                        "'scov_gm' (lower): GM sCoV (Wang 2003), 1-SE band on sCoV, "
                        "optional lapvar floor via --select_lapvar_floor. 'scov_umse1se': "
                        "FUSION — feasibility band is uMSE ≤ uMSE_min + feasible_se·SE(uMSE) "
                        "(accuracy guarantee that auto-rejects both noise and "
                        "over-smoothing), then the single best ckpt = argmin sCoV-GM+WM "
                        "(0.5·(sCoV_GM+sCoV_WM)) WITHIN that band. 'cnr' (higher=better): "
                        "GLOBAL max GM-WM CNR over ALL ckpts — NO uMSE fidelity band and "
                        "NO SWA, the single argmax-CNR ckpt is the final model (user "
                        "sign-off 2026-06-27; overrides the CLAUDE.md §4 uMSE-as-bar rule). "
                        "CNR and a cross-modal T1-leak guardrail are reported alongside.")
    p.add_argument("--select_lapvar_floor", type=float, default=0.0,
                   help="For --metric scov_gm: exclude ckpts whose pooled lapvar_ratio "
                        "< floor (over-smooth). 0 = no floor. Recommended ~0.6.")
    p.add_argument("--eval_seed", type=int, default=12345,
                   help="Fixed RNG seed re-applied before EACH checkpoint's val pass so "
                        "every checkpoint is scored on the SAME A/B frame partitions "
                        "(the dataset re-randomises order=randperm(T) per read; without "
                        "this, checkpoint deltas mix with A/B sampling noise). Seeds "
                        "torch/np/random → also determinises DataLoader worker seeds.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument("--save_selected", default=None,
                   help="If set, copy the SELECTED ckpt's file to this path (the final "
                        "operating-point model, e.g. <run>/best_cnr.pth). Used by the "
                        "best-CNR pipeline in place of the post-hoc SWA artefact.")
    p.add_argument("--nframe_sweep", action="store_true",
                   help="Also run the 2/4/6/8-frame inference sweep per ckpt "
                        "(sCoV/CNR/mean-drift vs n; 4x extra forwards). Off by default.")
    p.add_argument("--frd", action="store_true",
                   help="Also compute FRD (Fréchet Radiomic Distance, skimage "
                        "first-order+GLCM features) recon-dist vs union-dist per ckpt.")
    args = p.parse_args()

    print(f"[eval] building Runner with args: {args.runner_args[:200]}...")
    runner, runner_args = _build_runner(args.runner_args)
    # Force batch_size=1 for evaluation: the per-slice metrics are pooled per BATCH,
    # so a multi-subject batch cannot be split back to subjects. bs=1 makes each unit
    # exactly one slice carrying its subject_id → _aggregate_per_subject pools the
    # statistical unit at the SUBJECT level (selection SE / bootstrap over subjects,
    # not batches). Slower, but eval is not perf-critical and correctness matters.
    from monai.data import DataLoader as _DL
    from utils.utils import collate_varlen as _collate
    _vds = runner.val_loader.dataset
    runner.val_loader = _DL(_vds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=_collate)
    print(f"[eval] Runner built. Device={runner.device}; val rebuilt at batch_size=1 "
          f"for per-subject aggregation ({len(_vds)} slices).")
    print(f"[eval] metric={args.metric} lapvar_floor={args.select_lapvar_floor}")

    import random as _random
    def _freeze_eval_rng():
        # Re-freeze the A/B partition RNG so ALL checkpoints see IDENTICAL val inputs.
        s = int(getattr(args, "eval_seed", 12345))
        torch.manual_seed(s); np.random.seed(s); _random.seed(s)

    results = []
    for ckpt_path in args.ckpts:
        ckpt_name = Path(ckpt_path).name
        if not os.path.exists(ckpt_path):
            print(f"[eval] SKIP missing: {ckpt_path}")
            continue
        print(f"[eval] {ckpt_name} loading weights...")
        _load_model_weights(runner, ckpt_path)
        print(f"[eval] {ckpt_name} validating...")
        _freeze_eval_rng()
        v = _validate_with_per_unit_umse(runner)
        if args.nframe_sweep:
            _freeze_eval_rng()
            sw = _nframe_sweep(runner)
            v["nframe_sweep"] = sw
            _pairs = " ".join(
                f"n{n}:sCoV={s:.3f}/CNR={c:.3f}/μ={m:.4f}"
                for n, s, c, m in zip(sw["n"], sw["scov_gm"], sw["cnr"], sw["mean"]))
            print(f"[eval] {ckpt_name} nframe-sweep {_pairs} | "
                  f"sCoV_slope={sw['scov_slope']:+.4f}(≤0 good) "
                  f"CNR_slope={sw['cnr_slope']:+.4f}(≥0 good) "
                  f"mean_drift={sw['mean_drift']:.3f}(small good)")
        if args.frd:
            _freeze_eval_rng()
            fr = _frd(runner)
            v["frd"] = fr
            print(f"[eval] {ckpt_name} FRD={fr['frd']:.4f} (skimage-feat, "
                  f"recon-vs-union dist, lower=closer; n_slices={fr['n_slices']})")
        units, agg_level = _aggregate_per_subject(v["per_unit"])
        unit_umse = np.array([u["umse"] for u in units])
        if args.metric == "scov_gm":
            # SE over per-SUBJECT scov_gm (drop NaN subjects).
            unit_metric = np.array([u["scov_gm"] for u in units
                                    if u["scov_gm"] == u["scov_gm"]])
            pooled_metric = v["pooled_scov_gm"]
        elif args.metric == "cnr":
            # higher=better; SE over per-SUBJECT CNR (drop NaN). pooled_metric = CNR is
            # informational only — selection is a GLOBAL argmax, no 1-SE band (below).
            unit_metric = np.array([u["cnr"] for u in units
                                    if u["cnr"] == u["cnr"]])
            pooled_metric = v["pooled_cnr"]
        else:
            # 'umse' and 'scov_umse1se' both put the 1-SE band on uMSE, so the
            # per-unit metric / SE are computed on uMSE. (scov_umse1se then picks
            # argmin sCoV_GM within that band — handled in the selection block.)
            unit_metric = unit_umse
            pooled_metric = v["pooled_umse"]
        se = _bootstrap_se(unit_metric, n_boot=args.bootstrap_n)
        print(f"[eval] {ckpt_name} {args.metric}={pooled_metric:.5e} SE={se:.5e} | "
              f"uMSE={v['pooled_umse']:.5f} sCoV-GM={v['pooled_scov_gm']:.4f} "
              f"sCoV-WM={v['pooled_scov_wm']:.4f} sCoV-GMWM={v['pooled_scov_gmwm']:.4f} "
              f"CNR={v['pooled_cnr']:.3f}/ref={v['pooled_cnr_ref']:.3f} "
              f"SNR-GM={v['pooled_snr_gm']:.2f} SNR-WM={v['pooled_snr_wm']:.2f} "
              f"HFEN={v['pooled_hfen']:.3f} upsnr_cyc={v['pooled_upsnr_cyc']:.2f} "
              f"HFcyc={v['pooled_hfcyc']:.3f} MLC={v['pooled_mlc']:.3f} "
              f"MLCaniso={v['pooled_mlc_aniso']:.3f} MSLC={v['pooled_mslc']:.3f} "
              f"T1sim={v['pooled_t1sim']:.3f}/ref={v['pooled_t1sim_ref']:.3f} "
              f"MI(T1)={v['pooled_mi_t1']:.3f}/base={v['pooled_mi_ut1']:.3f} "
              f"NMI(T1)={v['pooled_nmi_t1']:.3f}/base={v['pooled_nmi_ut1']:.3f} "
              f"MI(union)={v['pooled_mi_union']:.3f} "
              f"EFC={v['pooled_efc']:.3f}/ref={v['pooled_efc_ref']:.3f} "
              f"bright_tail={v['pooled_bright_tail']:.3f} "
              f"[lapvR={v['pooled_lapvr']:.3f} CV-only]")
        results.append({
            "pooled_cyc": v["pooled_cyc"],
            "pooled_hfcyc": v["pooled_hfcyc"],
            "pooled_mlc": v["pooled_mlc"],
            "pooled_mlc_aniso": v["pooled_mlc_aniso"],
            "pooled_mslc": v["pooled_mslc"],
            "pooled_upsnr": v["pooled_upsnr"],
            "pooled_upsnr_cyc": v["pooled_upsnr_cyc"],
            "ckpt": ckpt_name,
            "ckpt_path": ckpt_path,
            "metric": args.metric,
            "pooled_metric": pooled_metric,
            "pooled_umse": v["pooled_umse"],
            "pooled_scov_gm": v["pooled_scov_gm"],
            "pooled_scov_wm": v["pooled_scov_wm"],
            "pooled_scov_gmwm": v["pooled_scov_gmwm"],
            "pooled_cnr": v["pooled_cnr"],
            "pooled_cnr_ref": v["pooled_cnr_ref"],
            "pooled_snr_gm": v["pooled_snr_gm"],
            "pooled_snr_wm": v["pooled_snr_wm"],
            "pooled_snr_gm_ref": v["pooled_snr_gm_ref"],
            "pooled_snr_wm_ref": v["pooled_snr_wm_ref"],
            "pooled_hfen": v["pooled_hfen"],
            "pooled_t1sim": v["pooled_t1sim"],
            "pooled_t1sim_ref": v["pooled_t1sim_ref"],
            "pooled_lapvr": v["pooled_lapvr"],
            "se": se,
            "n_units": len(units),
            "aggregation_level": agg_level,
        })

    # Apply lapvar floor → candidate set. Applies to ALL metrics (2026-06-08):
    # the scov_umse1se selector minimises sCoV within the uMSE band and so, with
    # no floor, lands on the smoothest (lowest-lapvar) feasible ckpt — an
    # over-smoothing bias. A floor excludes over-smooth ckpts BEFORE the band so
    # the within-band sCoV-argmin cannot pick the degenerate-smooth extreme.
    floor = float(args.select_lapvar_floor)
    if floor > 0.0:
        cand = [r for r in results if r["pooled_lapvr"] >= floor]
        excluded = [r["ckpt"] for r in results if r["pooled_lapvr"] < floor]
        print(f"[eval] lapvar floor {floor}: {len(cand)}/{len(results)} ckpts pass "
              f"(excluded over-smooth: {excluded})")
        if not cand:
            raise SystemExit(f"[eval] no ckpt passes lapvar floor {floor}; lower it.")
    else:
        cand = results

    if args.metric == "cnr":
        # GLOBAL max-CNR final selection — NO uMSE fidelity band, NO SWA.
        # User sign-off 2026-06-27 (informed that this can land on the degrading
        # tail, where uMSE rises above the 1-SE bar while CNR keeps climbing);
        # explicitly overrides the CLAUDE.md §4 'uMSE-as-bar' philosophy and
        # replaces the post-hoc feasible-set SWA with one argmax-CNR checkpoint.
        cnr_cand = [r for r in cand if r["pooled_cnr"] == r["pooled_cnr"]]  # drop NaN CNR
        if not cnr_cand:
            raise SystemExit("[eval] --metric cnr: no ckpt has a finite CNR "
                             "(val set has no GM/WM PV maps?).")
        sel = max(cnr_cand, key=lambda r: r["pooled_cnr"])
        min_ckpt = sel["ckpt"]
        band_min = float(sel["pooled_cnr"])
        se_at_min = float(sel["se"])
        threshold = float("nan")               # no band for a higher-is-better global max
        feasible_results = [sel]
        feasible = [min_ckpt]
        objective_name = "cnr_global_max"
        objective_val = float(sel["pooled_cnr"])
    else:
        # Band is on pooled_metric (= uMSE for umse/scov_umse1se; = sCoV for scov_gm):
        #   threshold = band_min + feasible_se·SE(band_min)
        #   feasible set = ckpts within the band (consumed by build_swa_feasible.py for SWA)
        min_idx_band = int(np.argmin([r["pooled_metric"] for r in cand]))
        band_min = cand[min_idx_band]["pooled_metric"]
        se_at_min = cand[min_idx_band]["se"]
        threshold = band_min + args.feasible_se * se_at_min
        feasible_results = [r for r in cand if r["pooled_metric"] <= threshold]
        feasible = [r["ckpt"] for r in feasible_results]

        # Single best ckpt:
        #   scov_umse1se → argmin sCoV_GM WITHIN the uMSE 1-SE band (the fusion rule)
        #   umse / scov_gm → the band minimum itself
        if args.metric == "scov_umse1se":
            sel = min(feasible_results, key=lambda r: r["pooled_scov_gmwm"])
            min_ckpt = sel["ckpt"]
            objective_name = "scov_gmwm@umse1se"
            objective_val = float(sel["pooled_scov_gmwm"])
        elif args.metric == "t1_umse1se":
            # Structure-priority (user 2026-05-31): within the uMSE 1-SE band, pick the
            # ckpt with the HIGHEST T1-structural-alignment (T1sim); sCoV-GMWM tiebreak.
            sel = max(feasible_results, key=lambda r: (r["pooled_t1sim"], -r["pooled_scov_gmwm"]))
            min_ckpt = sel["ckpt"]
            objective_name = "t1sim@umse1se"
            objective_val = float(sel["pooled_t1sim"])
        else:
            sel = cand[min_idx_band]
            min_ckpt = sel["ckpt"]
            objective_name = args.metric
            objective_val = float(band_min)

    out = {
        "runner_args": args.runner_args,
        "metric": args.metric,
        "select_lapvar_floor": floor,
        "bootstrap_n": args.bootstrap_n,
        "feasible_se": args.feasible_se,
        "aggregation_level": results[0]["aggregation_level"] if results else None,
        "results": results,
        "band_metric": ("cnr" if args.metric == "cnr"
                        else "umse" if args.metric in ("umse", "scov_umse1se", "t1_umse1se")
                        else "scov_gm"),
        "band_min": float(band_min),
        "se_at_min": float(se_at_min),
        "threshold_metric": float(threshold),
        # min_metric kept for backward-compat with build_swa_feasible.py.
        "min_metric": float(band_min),
        "min_ckpt": min_ckpt,
        "objective_name": objective_name,
        "objective_val": objective_val,
        "selected_umse": float(sel["pooled_umse"]),
        "selected_scov_gm": float(sel["pooled_scov_gm"]),
        "selected_scov_wm": float(sel["pooled_scov_wm"]),
        "selected_scov_gmwm": float(sel["pooled_scov_gmwm"]),
        "selected_cnr": float(sel["pooled_cnr"]),
        "selected_cnr_ref": float(sel["pooled_cnr_ref"]),
        "selected_t1sim": float(sel["pooled_t1sim"]),
        "selected_t1sim_ref": float(sel["pooled_t1sim_ref"]),
        "selected_snr_gm": float(sel["pooled_snr_gm"]),
        "selected_snr_wm": float(sel["pooled_snr_wm"]),
        "selected_snr_gm_ref": float(sel["pooled_snr_gm_ref"]),
        "selected_snr_wm_ref": float(sel["pooled_snr_wm_ref"]),
        "selected_hfen": float(sel["pooled_hfen"]),
        "feasible_ckpts": feasible,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[eval] ===== ASL-metric table (all ckpts) =====")
    print(f"{'ckpt':<15}{'uMSE':>8}{'sCoVGW':>8}{'CNR':>6}{'SNRg':>6}{'T1sim':>6}"
          f"{'HFcyc':>7}{'MLC':>6}{'MLCan':>6}{'MSLC':>6}{'upCYC':>7}  flags")
    for r in sorted(results, key=lambda x: x["ckpt"]):
        flags = ("feas" if r["ckpt"] in feasible else "    ") + (" *SEL*" if r["ckpt"] == min_ckpt else "")
        print(f"{r['ckpt']:<15}{r['pooled_umse']:>8.5f}{r['pooled_scov_gmwm']:>8.4f}"
              f"{r['pooled_cnr']:>6.3f}{r['pooled_snr_gm']:>6.2f}{r['pooled_t1sim']:>6.3f}"
              f"{r['pooled_hfcyc']:>7.3f}{r['pooled_mlc']:>6.3f}{r['pooled_mlc_aniso']:>6.3f}"
              f"{r['pooled_mslc']:>6.3f}{r['pooled_upsnr_cyc']:>7.2f}  {flags}")
    print(f"\n[eval] DONE. metric={args.metric}")
    if args.metric == "cnr":
        print(f"[eval] selection = GLOBAL max-CNR (no uMSE band, no SWA): "
              f"CNR={band_min:.5f} (SE={se_at_min:.5e})")
    else:
        print(f"[eval] band ({out['band_metric']}) min={band_min:.5e} (SE={se_at_min:.5e}), "
              f"threshold = min + {args.feasible_se}·SE = {threshold:.5e}")
    print(f"[eval] feasible set ({len(feasible)} ckpts): {feasible}")
    print(f"[eval] SELECTED {min_ckpt}: {objective_name}={objective_val:.5f}")
    print(f"[eval]   uMSE={sel['pooled_umse']:.5f} sCoV-GM={sel['pooled_scov_gm']:.4f} "
          f"sCoV-WM={sel['pooled_scov_wm']:.4f} CNR={sel['pooled_cnr']:.3f}/ref={sel['pooled_cnr_ref']:.3f}")
    print(f"[eval]   SNR-GM={sel['pooled_snr_gm']:.2f} (ref {sel['pooled_snr_gm_ref']:.2f}) "
          f"SNR-WM={sel['pooled_snr_wm']:.2f} (ref {sel['pooled_snr_wm_ref']:.2f}) "
          f"HFEN={sel['pooled_hfen']:.3f} T1sim={sel['pooled_t1sim']:.3f}/ref={sel['pooled_t1sim_ref']:.3f}")
    print(f"[eval] saved -> {args.output}")

    # Materialise the final operating-point model: copy the SELECTED ckpt file
    # verbatim (no averaging) so phase4 can load it like any runner checkpoint.
    if args.save_selected:
        import shutil
        src = sel["ckpt_path"]
        Path(args.save_selected).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, args.save_selected)
        print(f"[eval] copied SELECTED {min_ckpt} ({objective_name}={objective_val:.5f}) "
              f"-> {args.save_selected}")


if __name__ == "__main__":
    main()
