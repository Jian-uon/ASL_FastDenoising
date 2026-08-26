"""EC-LRDA guidance: ASL evidence reliability (r_rep) + T1-ASL semantic
compatibility (c_sem). γ = r_rep · c_sem gates the soft-PV bilinear modulation of
the SSM dynamics (B, Δ) inside the VSS encoder.

Two decoupled "legs", each answering a different question:
  • r_rep(x)  — is the ASL evidence at x reproducible across frames? (ASL-intrinsic;
                cross-frame dispersion of a shared per-frame feature. NO scalar
                per-frame q_k, NO K_eff.)
  • c_sem(x)  — is the frozen-PV tissue interpretation at x SUPPORTED by the ASL
                feature? Built from subject-specific ASL tissue prototypes (μ_c is
                pure ASL; PV only does the soft grouping) → expected feature
                f̂ = Σ_c p_c μ_c → compatibility 1−cos(f, f̂).

γ only SCALES the (zero-init, multiplicative, B/Δ-only) modulation, so V=ASL holds:
γ can attenuate the anatomical guidance but can never inject content. γ∈(0,1].

Both branches are supervised by their own signals (L_rep on injected artifact masks,
L_sem as a match>mismatch ranking) and are stop-gradient'd out of the recon path in
the runner, so neither can silently collapse to a constant (the RGSF/Var_T failure).
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _norm01(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Per-sample spatial standardisation (zero-mean/unit-std). This deliberately
    keeps ONLY the RELATIVE spatial pattern of the dispersion/incompatibility and
    DISCARDS the per-sample absolute level (a0/b0 are GLOBAL constants, shared across
    samples — they set a fixed operating point, they do NOT restore a per-sample
    'how (un)reliable overall' scale). That is the intended scope: r_rep/c_sem are
    LOCAL reliability maps supervised by LOCAL patch artifacts (L_rep) and a per-map
    ranking (L_sem), where within-sample relative structure is what matters. It is
    NOT a global evidence-level (n_eff) signal — that was deliberately excluded
    (q_k/K_eff rejected; see ASL_dmvae/docs/cig_vss.md). A future global-strength term, if
    added, must be a separate scalar, not recovered from this normalised input."""
    B = x.shape[0]
    flat = x.reshape(B, -1)
    mu = flat.mean(dim=1, keepdim=True)
    sd = flat.std(dim=1, keepdim=True) + eps
    return ((flat - mu) / sd).reshape_as(x)


class ECGuidance(nn.Module):
    def __init__(self, asl_in_ch: int, pv_ch: int = 4, feat_dim: int = 32,
                 work_div: int = 4, sem_bayes: bool = False,
                 sem_bayes_loo: bool = True, sem_evidence: bool = False) -> None:
        super().__init__()
        self.pv_ch = int(pv_ch)
        self.work_div = max(1, int(work_div))
        g = min(8, feat_dim)
        while feat_dim % g != 0 and g > 1:
            g -= 1
        # shared per-frame encoder (r_rep) — operates on frame DEVIATION-friendly
        # learned features, spatially pooled → higher effective SNR than raw pixels.
        self.frame_enc = nn.Sequential(
            nn.Conv2d(asl_in_ch, feat_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, feat_dim), nn.SiLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, feat_dim), nn.SiLU(inplace=True),
        )
        # aggregate feature (c_sem) — a semantic descriptor of the aggregated ASL.
        self.sem_feat = nn.Sequential(
            nn.Conv2d(asl_in_ch, feat_dim, 3, padding=1, bias=False),
            nn.GroupNorm(g, feat_dim), nn.SiLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
        )
        # monotone reliability maps r = σ(a0 − softplus(β)·norm(D)). β init = -4 ⇒
        # softplus(β)≈0.018 ⇒ the slope on the (standardised) dispersion is ~flat at
        # init, so r,c ≈ σ(a0) ≈ 0.98 near-uniformly ⇒ γ≈1 (near-identity: EC starts
        # ≈ plain soft-PV LRDA, and the guidance is NOT randomly attenuated by the
        # meaningless init features while L_rep/L_sem warm up). softplus(β) keeps the
        # slope ≥0 (more dispersion ⇒ lower reliability, never higher) as it grows.
        self.a0_rep = nn.Parameter(torch.tensor(4.0))
        self.beta_rep = nn.Parameter(torch.tensor(-4.0))
        self.b0_sem = nn.Parameter(torch.tensor(4.0))
        self.beta_sem = nn.Parameter(torch.tensor(-4.0))
        # Bayesian-model-based c_sem (opt-in, --ec_sem_bayes): replaces the plug-in
        # cosine gate with a hierarchical empirical-Bayes PV linear-mixture whose
        # prototypes are a genuine conjugate posterior, feeding a PLUG-IN point-estimate
        # H1-vs-H0 evidence SCORE (not a calibrated posterior probability / marginal-
        # likelihood Bayes factor — see _c_sem_bayes). REUSES b0_sem/beta_sem for the SAME near-identity init
        # (slope≈0.018 ⇒ c≈σ(4)≈0.98 ⇒ γ≈1 while L_sem warms up), adding only the
        # dataset-level ASL prototype M0 and the prior log-precision ρ. Prototypes stay
        # pure-ASL (M0 is an ASL-feature param, PV only forms the design matrix) ⇒ V=ASL
        # preserved. Full design + caveats: ASL_dmvae/docs/backlog_future_ideas.md §1.
        self.sem_bayes = bool(sem_bayes)
        # sem_bayes_loo: RETIRED 2026-08-05 (LOO dropped as empirically inert; the runner
        # always constructs no-LOO now). The LOO forward branch is KEPT only so old LOO
        # checkpoints (arch dict ec_sem_bayes_loo=True) still strict-load and replay their
        # trained forward at eval — no new run sets this True.
        self.sem_bayes_loo = bool(sem_bayes_loo)
        # sem_evidence (--ec_sem_evidence): "no-LOO gate + global evidence" mode.
        #   (i) forces the per-pixel gate to the plain-residual (no-LOO) path (user design);
        #   (ii) ALSO computes a per-image closed-form log Bayes factor E = log p(F|H1) −
        #        log p(F|H0) (PV-tissue mixture vs global-mean null, incl. the Occam
        #        −½d·log|A_ev| complexity term) — the whole-image "is T1/PV trustworthy for
        #        this subject" signal the per-pixel gate self-normalises away. E is
        #        DIAGNOSTIC (detached), stashed on self._last_evidence and returned by
        #        forward(); it does NOT modify the gate. NO new params ⇒ strict-load stays
        #        compatible with existing Bayesian ckpts.
        self.sem_evidence = bool(sem_evidence)
        self._last_evidence = None
        if self.sem_bayes:
            self.sem_M0 = nn.Parameter(torch.zeros(self.pv_ch, feat_dim))
            self.sem_rho = nn.Parameter(torch.zeros(self.pv_ch))  # softplus(0)=0.69 prior precision

    def _down(self, x: Tensor) -> Tensor:
        return x if self.work_div <= 1 else F.avg_pool2d(x, self.work_div)

    def r_rep(self, set_a: Tensor, lengths: Optional[Tensor] = None,
              mask: Optional[Tensor] = None) -> Tensor:
        """[B,T,C,H,W] → r_rep [B,1,h,w] at work resolution."""
        B, T, C, H, W = set_a.shape
        Fk = self.frame_enc(set_a.reshape(B * T, C, H, W))         # [B*T,d,H,W]
        Fk = self._down(Fk)                                        # [B*T,d,h,w]
        d, h, w = Fk.shape[1], Fk.shape[2], Fk.shape[3]
        Fk = Fk.reshape(B, T, d, h, w)
        if mask is not None:
            valid = (~mask.bool()).float()
        elif lengths is not None:
            valid = (torch.arange(T, device=set_a.device)[None] < lengths[:, None]).float()
        else:
            valid = torch.ones(B, T, device=set_a.device)
        v = valid.view(B, T, 1, 1, 1)
        n = v.sum(1).clamp_min(1.0)                                # [B,1,1,1]
        fbar = (Fk * v).sum(1) / n                                 # [B,d,h,w] consensus (all frames, high SNR)
        dev = (Fk - fbar.unsqueeze(1)).abs().mean(2)              # [B,T,h,w] channel-mean deviation
        D = (dev * valid.view(B, T, 1, 1)).sum(1) / n.squeeze(-1) # [B,h,w] cross-frame dispersion
        D = D.unsqueeze(1)                                         # [B,1,h,w]
        return torch.sigmoid(self.a0_rep - F.softplus(self.beta_rep) * _norm01(D))

    def c_sem(self, agg_a: Tensor, pv: Tensor, w: Optional[Tensor] = None) -> Tensor:
        """agg_a [B,C,H,W], pv [B,pv_ch,H,W] softmax → c_sem [B,1,h,w] at work res."""
        if self.sem_bayes:
            return self._c_sem_bayes(agg_a, pv, w)
        f = self._down(self.sem_feat(agg_a))                      # [B,d,h,w]
        pv = self._down(pv)                                       # [B,pv_ch,h,w]
        if w is None:
            w = torch.ones_like(f[:, :1])
        elif w.shape[-2:] != f.shape[-2:]:
            w = F.interpolate(w, size=f.shape[-2:], mode="bilinear", align_corners=False)
        wp = (w * pv).clamp_min(0.0)                              # [B,pv_ch,h,w] reliability-weighted PV
        num = torch.einsum("bchw,bdhw->bcd", wp, f)              # [B,pv_ch,d]
        den = wp.sum(dim=(2, 3)).clamp_min(1e-5).unsqueeze(-1)   # [B,pv_ch,1]
        mu = num / den                                           # [B,pv_ch,d] ASL tissue prototypes (content=ASL)
        fhat = torch.einsum("bchw,bcd->bdhw", pv, mu)           # [B,d,h,w] expected ASL feature
        cos = F.cosine_similarity(f, fhat, dim=1, eps=1e-5).unsqueeze(1)  # [B,1,h,w]
        dsem = 1.0 - cos
        return torch.sigmoid(self.b0_sem - F.softplus(self.beta_sem) * _norm01(dsem))

    def _c_sem_bayes(self, agg_a: Tensor, pv: Tensor,
                     w: Optional[Tensor] = None) -> Tensor:
        """Bayesian c_sem (opt-in). Model each ASL feature as a PV LINEAR MIXTURE of
        subject prototypes: f_i ~ N(p_iᵀ M, σ²I), M∈R^{C×d}, with a shrinkage prior
        M ~ N(M0, diag(τ)⁻¹) (τ=softplus(ρ)). Closed-form conjugate posterior
        M̄ = A⁻¹(PᵀWF + τ⊙M0), A = PᵀWP + diag(τ) — this prototype step is genuine Bayesian
        (≡ ridge). The gate is then a (no-LOO) PLUG-IN, point-estimate evidence SCORE
        b=½(q0/σ0²−q1/σ1²) through a learned sigmoid — NOT a calibrated posterior probability
        or marginal-likelihood Bayes factor (point-estimate M̄, dropped log-normalizer,
        empirical plug-in σ). It scores whether the PV-tissue model (H1) explains f_i better
        than an ASL-only global-mean null (H0), using the plain in-sample residual. (The full
        marginalized log Bayes factor is the diagnostic E only; the LEAVE-ONE-OUT
        branch below is RETIRED — LOO dropped 2026-08-05 as empirically inert — and runs
        only when replaying an old LOO ckpt.) Using ABSOLUTE σ-standardised misfits
        (no Norm01) lets a LOCAL anomaly — where ASL deviates from its tissue-conditional
        expectation — drop the gate and so protect the anomaly. Prototypes are pure ASL
        (M0 is an ASL-feature param; PV only forms the design matrix P) ⇒ V=ASL. σ is
        DETACHED per-sample so the gate cannot be gamed by inflating predictive variance.
        NOTE: per-sample σ ⇒ this is a LOCAL-anomaly detector (its real job, for lesion
        preservation), NOT a global wrong-subject detector (demoted; see
        ASL_dmvae/docs/tmi_experiment_plan.md). H0 straw-man / attribution caveats: backlog §1."""
        f = self._down(self.sem_feat(agg_a))                      # [B,d,h,w]
        pv = self._down(pv)                                       # [B,C,h,w]
        if w is None:
            w = torch.ones_like(f[:, :1])
        elif w.shape[-2:] != f.shape[-2:]:
            w = F.interpolate(w, size=f.shape[-2:], mode="bilinear", align_corners=False)
        Bsz, d, h, wid = f.shape
        C = pv.shape[1]
        orig_dtype = f.dtype
        # fp32 for the linear algebra (solve/inv are fp16-fragile under autocast).
        with torch.autocast(device_type=f.device.type, enabled=False):
            Ff = f.reshape(Bsz, d, h * wid).transpose(1, 2).float()          # [B,N,d]
            P = pv.clamp_min(0.0).reshape(Bsz, C, h * wid).transpose(1, 2).float()  # [B,N,C]
            wv = w.reshape(Bsz, 1, h * wid).transpose(1, 2).clamp_min(0.0).float()  # [B,N,1]
            Pw = P * wv
            tau = F.softplus(self.sem_rho.float())                          # [C]
            eye = torch.eye(C, device=f.device, dtype=torch.float32)
            A = (torch.einsum("bni,bnj->bij", Pw, P)
                 + torch.diag_embed(tau).unsqueeze(0) + 1e-4 * eye)          # [B,C,C] SPD
            Bm = (torch.einsum("bni,bnd->bid", Pw, Ff)
                  + tau.unsqueeze(-1) * self.sem_M0.float())                 # [B,C,d]  (τ⊙M0)
            Mbar = torch.linalg.solve(A, Bm)                                 # [B,C,d]  posterior prototypes
            Fhat = torch.einsum("bnc,bcd->bnd", P, Mbar)                     # [B,N,d]  PV-mixture predictive
            res = Ff - Fhat                                                  # [B,N,d]  plain residual
            # sem_evidence forces the plain-residual (no-LOO) gate (user design).
            if self.sem_bayes_loo and not self.sem_evidence:
                Ainv = torch.linalg.inv(A)                                   # [B,C,C]
                lev = (torch.einsum("bnc,bck->bnk", P, Ainv) * P).sum(-1) * wv.squeeze(-1)  # [B,N] leverage
                res_q = res / (1.0 - lev.clamp(0.0, 1.0 - 1e-3)).unsqueeze(-1)  # leave-one-out
            else:
                res_q = res                                                 # plug-in fit, no LOO
            q1 = res_q.pow(2).mean(-1)                                       # [B,N]  H1 misfit
            fbar = (wv * Ff).sum(1) / wv.sum(1).clamp_min(1e-5)             # [B,d]  ASL-only null mean
            q0 = (Ff - fbar.unsqueeze(1)).pow(2).mean(-1)                    # [B,N]  H0 misfit
            sig1 = res.pow(2).mean(-1).mean(1, keepdim=True).detach().clamp_min(1e-6)  # [B,1] absolute scale
            sig0 = q0.mean(1, keepdim=True).detach().clamp_min(1e-6)                    # [B,1]
            b = 0.5 * (q0 / sig0 - q1 / sig1)                               # [B,N]  H1-better ⇒ +
            b = b.reshape(Bsz, 1, h, wid)
            c = torch.sigmoid(self.b0_sem.float() + F.softplus(self.beta_sem.float()) * b)
            # --- part ②: per-image GLOBAL evidence E (diagnostic, DETACHED) ---
            # closed-form log Bayes factor of H1 (PV-tissue mixture, prior mean M0) vs
            # H0 (global-mean null, m0=0); matrix-determinant-lemma / Woodbury keep it
            # C×C (never forms N×N). σ̂ plug-in. This is the whole-image "is T1/PV
            # trustworthy for this subject" signal the per-pixel gate self-normalises
            # away (wrong-subject / misregistration axis). Shared constants
            # (−Σ log w_i, N·d·log 2π) cancel in the H1−H0 difference. E does NOT alter c.
            if self.sem_evidence:
                Nf = float(h * wid)
                is1 = (1.0 / sig1).squeeze(-1)                              # [B]  σ̂1⁻²
                is0 = (1.0 / sig0).squeeze(-1)                              # [B]  σ̂0⁻²
                A_ev = (torch.diag_embed(tau).unsqueeze(0)
                        + is1.view(Bsz, 1, 1) * torch.einsum("bni,bnj->bij", Pw, P)
                        + 1e-4 * eye)                                       # diag(τ)+σ̂1⁻²PᵀWP  [B,C,C]
                Rev = Ff - torch.einsum("bnc,cd->bnd", P, self.sem_M0.float())  # F − P·M0 (prior mean)  [B,N,d]
                PtWR = torch.einsum("bni,bnd->bid", Pw, Rev)               # [B,C,d]
                quad1 = (is1 * (wv.squeeze(-1) * Rev.pow(2).sum(-1)).sum(-1)
                         - is1.pow(2) * torch.einsum("bid,bid->b", PtWR,
                                                     torch.linalg.solve(A_ev, PtWR)))  # [B]
                logdet1 = float(d) * (Nf * sig1.squeeze(-1).log()
                                      + torch.slogdet(A_ev)[1] - tau.log().sum())      # [B]
                tau0 = torch.as_tensor(1e-3, device=f.device, dtype=torch.float32)
                a0 = (tau0 + is0 * wv.squeeze(-1).sum(-1)).clamp_min(1e-6)  # [B]
                sWf = (wv * Ff).sum(1)                                      # [B,d]  Σ w f  (m0=0)
                quad0 = (is0 * (wv.squeeze(-1) * Ff.pow(2).sum(-1)).sum(-1)
                         - is0.pow(2) * sWf.pow(2).sum(-1) / a0)            # [B]
                logdet0 = float(d) * (Nf * sig0.squeeze(-1).log() + a0.log() - tau0.log())  # [B]
                E = (-0.5 * (quad1 + logdet1)) - (-0.5 * (quad0 + logdet0))  # [B] log p(F|H1)−log p(F|H0)
                self._last_evidence = E.detach()
        return c.to(orig_dtype)

    def forward(self, set_a: Tensor, agg_a: Tensor, pv: Tensor,
                lengths: Optional[Tensor] = None, mask: Optional[Tensor] = None,
                pv_mismatch: Optional[Tensor] = None,
                use_rep: bool = True) -> Dict[str, Tensor]:
        r = self.r_rep(set_a, lengths, mask)                      # [B,1,h,w]
        # DETACHED on purpose. r only *weights* c_sem's tissue-prototype pooling; it
        # must not be trained by L_sem. Without the detach, L_sem -> c_sem -> w=r
        # backprops into a0_rep/beta_rep/frame_enc, so the reproducibility leg would
        # be learned partly from the semantic objective -- the gate's two legs would
        # no longer be independently supervised, which is the property the paper
        # claims ("each leg is instead learned from its own dedicated objective") and
        # the reason the leg-liveness probes are readable as evidence at all. The
        # stop-grad on gamma at the modulation site is a DIFFERENT edge and does not
        # cover this one. Forward values are unchanged; this only removes a backward
        # edge (and the pooling is invariant to a uniform rescale of w anyway).
        w = r.detach() if use_rep else None
        c = self.c_sem(agg_a, pv, w=w)                            # [B,1,h,w]
        gamma = (r * c) if use_rep else c                         # [B,1,h,w]
        out = {"r_rep": r, "c_sem": c, "gamma": gamma}
        if self.sem_bayes and self.sem_evidence and self._last_evidence is not None:
            out["sem_evidence"] = self._last_evidence             # [B] matched-PV log Bayes factor (diagnostic)
        if pv_mismatch is not None:
            out["c_sem_mis"] = self.c_sem(agg_a, pv_mismatch, w=w)
            if self.sem_bayes and self.sem_evidence and self._last_evidence is not None:
                out["sem_evidence_mis"] = self._last_evidence     # [B] mismatched-PV evidence (should be lower)
        return out
