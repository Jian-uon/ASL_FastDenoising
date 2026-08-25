"""Diagnostic: RAW vs EMA weights from the same checkpoint — is EMA over-smoothing?

For each ckpt we evaluate the SHARED validation replica twice: once with the raw
`model` weights, once with the `ema` weights. Both go through the identical data
path (runner._predict_with_ema copies whatever is in runner.ema.ema_state into the
model, infers, restores), so the ONLY variable is the weight set.

Reported per weight set:
  lapvR  = lapvar(recon)/lapvar(noisy union)  — higher = sharper recon
  scovGM = sCoV inside GM PV mask             — higher = more within-tissue texture/noise
  uMSE   = unbiased risk-to-clean             — lower = more accurate

Interpretation:
  raw lapvR >> ema lapvR  → the constant-0.9999 EMA is smoothing the EVALUATED
                            weights; the over-smoothing is an averaging artifact,
                            fixable by lowering EMA decay — NOT a fundamental limit.
  raw ≈ ema (both smooth) → smoothness is intrinsic to the N2N/SNR regime; EMA is
                            not the culprit.
"""
import argparse
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO))

from scripts.eval_select_ckpt import _build_runner, _validate_with_per_unit_umse  # noqa: E402


def _eval_weights(runner, state_dict):
    runner.ema.ema_state = {k: v.to(runner.device) for k, v in state_dict.items()}
    return _validate_with_per_unit_umse(runner)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--runner_args", required=True)
    args = p.parse_args()

    runner, _ = _build_runner(args.runner_args)
    print(f"[diag] runner built. n_val_batches={len(runner.val_loader)}", flush=True)
    print(f"{'ckpt':<18} {'weights':<10} {'lapvR':>8} {'scovGM':>8} {'uMSE':>11}", flush=True)
    print("-" * 60, flush=True)
    for ckpt_path in args.ckpts:
        if not Path(ckpt_path).exists():
            print(f"[diag] SKIP missing {ckpt_path}", flush=True)
            continue
        ck = torch.load(ckpt_path, map_location=runner.device, weights_only=False)
        name = Path(ckpt_path).name
        rows = {}
        for key, label in (("model", "raw"), ("ema", "ema")):
            if not (isinstance(ck, dict) and key in ck):
                print(f"[diag] {name} has no '{key}' key", flush=True)
                continue
            v = _eval_weights(runner, ck[key])
            rows[label] = v
            print(f"{name:<18} {label:<10} {v['pooled_lapvr']:>8.3f} "
                  f"{v['pooled_scov_gm']:>8.4f} {v['pooled_umse']:>11.6f}", flush=True)
        if "raw" in rows and "ema" in rows:
            dl = rows["raw"]["pooled_lapvr"] - rows["ema"]["pooled_lapvr"]
            verdict = "raw SHARPER" if dl > 0.02 else ("≈ equal" if abs(dl) <= 0.02 else "ema sharper")
            print(f"{name:<18} {'ΔlapvR':<10} {dl:>+8.3f}  --> {verdict}", flush=True)
        print("-" * 60, flush=True)


if __name__ == "__main__":
    main()
