# Paper Draft — BSPC / CMIG submission

> Target: BSPC (IF ~5.1) primary; CMIG (IF ~5.4) backup; MRI journal (IF 2.8) fallback.
> Length: ~10 pages double-column (Elsevier).
> Status: Methods section drafted; other sections outline-only, awaiting experimental results.

---

## Provisional title

**"SVFW-Net: Anti-Hallucination Self-Supervised T1-Guided Denoising for 7T Arterial Spin Labeling Perfusion MRI"**

Alternatives:
- "Lesion-Safe Cross-Modal Self-Supervised Denoising of 7T ASL via Per-Pixel BLUE Frame Weighting"
- "Self-Supervised ASL Perfusion Denoising with Anti-Hallucination Cross-Attention and Spatially-Varying Frame Weighting"

---

## Authors / Affiliations

TODO

---

## Abstract (TODO — write last)

Structure:
1. Background: ASL low-SNR, requires multi-NEX averaging, long acquisition; 7T下B0更inhomogenity
2. Gap: existing DL methods either need clean GT (impractical) or risk T1 hallucination from anatomical priors.
3. Method: SVFW per-pixel BLUE aggregator + V=ASL multi-scale cross-attention + N2N+SURE+J-invariance.
4. Results: PSNR/SSIM/CoV/CBF metrics on N=? subjects vs 4 baselines (NLM/sup/N2N/N2Self).
5. Significance: matches 12-NEX quality from 4 frames; provides per-voxel confidence map; structurally cannot hallucinate T1 anatomy.

Target ~250 words. Write last after numbers settle.

---

## 1. Introduction

### 1.1 Clinical motivation
- ASL = non-invasive perfusion MRI, no contrast agent.
- ΔM (label−control) ~1% of M0 → low SNR → need 8-12 NEX averaging.
- 7T ASL: better SNR than 3T per NEX but T2* shortening and B1 inhomogeneity make individual frames noisier; motion + inflow timing variability → outlier frames common.
- Reducing NEX would shorten scan (5-10 min → 1-2 min) and improve patient compliance.

### 1.2 The denoising problem and existing approaches
- Supervised DL: needs high-NEX clean reference, which is itself noisy (Xie 2020 MRI; Kim 2018 Radiology).
- Self-supervised: Noise2Void / Noise2Self (Krull 2019, Batson 2019) — single-image, ignores frame redundancy.
- Anatomical-prior methods: usually concat T1 with ASL → **T1 grayscale risks bleeding into output** (hallucination).
- Cross-modal Transformers (e.g., Shou 2024 MRM transformer ASL, McMRSR 2022): use K=V from auxiliary modality — content from T1 is *intentionally* injected → unsuitable for denoising.

### 1.3 Our approach (preview)

Three integrated contributions:

1. **SVFW (Spatially-Varying Frame Weighting)** — per-pixel BLUE aggregator that handles spatially-localised motion / outliers; safe-by-design against lesion suppression.
2. **V=ASL multi-scale cross-attention** — T1 features and 4-class GM/WM/CSF/BG segmentation prior guide *where to attend* but values stay in ASL feature space → architecturally cannot leak T1 anatomy.
3. **Fully self-supervised training** — N2N (set_a vs set_b) + Monte-Carlo SURE divergence regulariser + J-invariance blind-spot, no clean GT.

Plus **MC-dropout × SWA-ensemble confidence map** at inference.

### 1.4 Contributions (bulleted summary)

- We propose SVFW, the first per-pixel BLUE aggregator for ASL where the log-variance head sees only frame-mean deviations (`frame − temporal_mean`), not raw signal — provably cannot down-weight a frame for being "too unusual".
- We design V=ASL multi-scale cross-attention with tissue-class similarity bias from a pre-trained frozen T1 segmentation head, providing structural anti-hallucination guarantees.
- We integrate N2N+SURE+J-invariance into a single self-supervised training framework and use psnr_b (Lehtinen 2018 unbiased held-out PSNR) for ckpt selection.
- We provide per-voxel confidence maps via MC-dropout × SWA-ensemble at inference.
- On a 7T ASL dataset (N=?), our method outperforms NLM / supervised U-Net / vanilla N2N / N2Self in PSNR/SSIM/CoV/CBF-MAE; mismatched-T1 sanity check confirms no hallucination.

### 1.5 Paper organization
- §2 Related work; §3 Methods; §4 Experiments; §5 Discussion; §6 Conclusion.

---

## 2. Related Work

### 2.1 ASL denoising
- Classic: NLM, dictionary learning (Wang 2003 sCoV, Gong 2018).
- Supervised CNN (Kim 2018 Radiology, Xie 2020 MRI, Wang 2022 transfer learning for AD).
- Self-supervised: Shou 2024 MRM transformer + KWIA reference (closest method-level competitor).
- 2025 Pediatric multi-delay ASL self-supervised Transformer with KWIA reference (NeuroImage).
- **Gap**: none use per-pixel frame weighting; none use V=ASL anti-hallucination cross-attention; most use M0 not T1 (Guo 2025 MRM).

### 2.2 Self-supervised denoising
- Noise2Noise (Lehtinen ICML 2018).
- Noise2Void (Krull CVPR 2019), Noise2Self (Batson ICML 2019) — J-invariance.
- SURE (Stein 1981; Soltanayev & Chun NeurIPS 2018).
- ENSURE (Aggarwal 2021), UNSURE (ICLR 2024).

### 2.3 Cross-modal attention in medical imaging
- Standard pattern Q from A, K=V from B (MTrans MICCAI 2021, McMRSR TMI 2022, ResViT TMI 2022) — content from auxiliary modality intentionally injected.
- Reference-based SR (TTSR CVPR 2020) — V from reference, suitable for SR not denoising.
- **No prior work uses V=A (target modality) for cross-modal denoising as a structural anti-hallucination constraint.**

### 2.4 Frame weighting / aggregation
- Set Transformer (Lee ICML 2019).
- BLUE (Aitken 1935 / Gauss-Markov).
- Most ASL works use uniform mean.

---

## 3. Methods

### 3.1 Problem formulation

Each subject has T = 12 ΔM frames `{Y_t}`. Each frame:
$$Y_t = X + N_t$$
where $X$ is the underlying clean PWI and $N_t$ is per-frame noise (motion, thermal, inflow timing variation). $N_t$ approximately independent across $t$.

Goal: estimate $\hat{X}$ from a subset of $T_a \le T$ frames `Y_a = {Y_{a_1}, ..., Y_{a_{T_a}}}`, conditioned on the co-registered T1-weighted image $T_1$.

### 3.2 Architecture overview

The network $\hat{X} = f_\theta(\mathbf{Y}_a, T_1)$ has four stages:

1. **SVFW frame aggregator** — `agg = Σ_t w_{t,h,w} · Y_a[t]`, weights $w \in [B,T,H,W]$.
2. **ConvEncoder2D** (4 levels, 32→64→128→256) on agg → bottleneck feature map $F^A_b \in \mathbb{R}^{B \times 256 \times 16 \times 16}$ + skip features.
3. **Multi-scale cross-modal fusion** — at scales 16×16 and 32×32, ASL features cross-attend to T1 features (V=ASL).
4. **ConvDecoderWithSkips2D** — upsamples to PWI output, fusing ASL skips at every level + T1 cross-attention at 32×32.

T1 branch is a separate ConvEncoder2D + ConvDecoder2D pre-trained as a 4-class GM/WM/CSF/BG segmentation head (stage-1, [Method 3.6]) and frozen during ASL training.

(Insert Figure 1: full architecture diagram from [docs/architecture.md](architecture.md) §2.)

### 3.3 Spatially-Varying Frame Weighting (SVFW)

Given input frames $\mathbf{Y}_a \in \mathbb{R}^{B \times T \times 1 \times H \times W}$:

1. Compute per-pixel temporal mean over valid frames:
$$\mu_{h,w} = \frac{1}{|V|} \sum_{t \in V} Y_{t,h,w}$$
where $V$ is the set of valid (non-padded) frame indices.

2. Compute per-frame deviation:
$$D_t = Y_t - \mu \in \mathbb{R}^{B \times 1 \times H \times W}$$

3. A small CNN (2× Conv-GroupNorm-SiLU, ~5k params) maps $D_t$ to a per-pixel log-variance:
$$\log\sigma^2_{t,h,w} = h_\phi(D_t)$$

4. Per-pixel BLUE-style weights:
$$w_{t,h,w} = \frac{\exp(-\log\sigma^2_{t,h,w})}{\sum_{t'} \exp(-\log\sigma^2_{t',h,w})}$$

5. Aggregation:
$$\text{agg}_{h,w} = \sum_t w_{t,h,w} \cdot Y_{t,h,w}$$

**Anti-lesion-suppression property.** The log-variance head $h_\phi$ takes only the *deviation* $D_t = Y_t - \mu$ as input, **never the raw signal $Y_t$**. Consequently, the weighting cannot down-weight a frame for having an unusual *absolute* signal value — it can only down-weight frames that are *temporally inconsistent* with their peers. Lesions, which manifest as consistent perfusion abnormalities across all frames, will have small per-frame deviations and retain full weight.

### 3.4 V=ASL multi-scale cross-attention

A `CrossModalFusion` block at each scale $s \in \{16 \times 16, 32 \times 32\}$:

$$Q^{(s)} = F^{A,(s)}, \quad K^{(s)} = F^{T_1,(s)}, \quad V^{(s)} = F^{A,(s)}$$

with attention bias from tissue-class similarity:
$$\text{bias}_{ij} = \tau \cdot \langle \tilde{S}_i, \tilde{S}_j \rangle$$
where $\tilde{S}_i = \text{normalize}(\sigma(\text{seg}_i))$ is the L2-normalised 4-class softmax-output of the frozen T1 segmentation head at position $i$, and $\tau$ is a learnable scalar temperature.

Attention output:
$$\text{Attn}^{(s)} = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}} + \text{bias}\right) V$$

Residual gate:
$$F'^{(s)} = \text{LayerNorm}\left(F^{A,(s)} + \mathrm{clip}(\gamma^{(s)}, 0, 1) \cdot \text{Attn}^{(s)}\right)$$

with $\gamma^{(s)}$ a learnable per-scale gate (init 0.3).

**Anti-hallucination property.** Because $V = F^A$, the output of cross-attention is a convex combination of ASL features — T1 features can only modulate *which* spatial positions to pool from, never inject T1 content. This is a structural constraint, not a regularisation.

### 3.5 Loss function

The total loss for stage-2 training is:
$$\mathcal{L} = w_{\text{n2n}} \mathcal{L}_{\text{n2n}} + w_{\text{grad}} \mathcal{L}_{\text{grad}} + w_{\text{ssim}} \mathcal{L}_{\text{ssim}} + w_{\text{contrast}} \mathcal{L}_{\text{contrast}} + w_{\text{sure}} \mathcal{L}_{\text{sure}}$$

where:
- $\mathcal{L}_{\text{n2n}} = \|m \odot (\hat{X} - \bar{Y}_b)\|_1 / \|m\|_1$ — N2N L1 against direct mean of held-out frames, masked to brain ($m = (T_1 > 0.05)$).
- $\mathcal{L}_{\text{grad}} = \|m \odot (\nabla \hat{X} - \nabla \bar{Y}_b)\|_1$ — gradient L1 (counters L1 over-smoothing).
- $\mathcal{L}_{\text{ssim}} = 1 - \text{SSIM}(m \odot \hat{X}, m \odot \bar{Y}_b)$.
- $\mathcal{L}_{\text{contrast}} = (1 + 4\sqrt{gm \cdot wm}) \cdot \mathcal{L}_{\text{n2n}}$ — GM/WM PV-boundary-weighted L1.
- $\mathcal{L}_{\text{sure}} = \frac{1}{N} \|\hat{X}(\mathbf{Y}_a) - \mathbf{Y}_a\|^2 + \frac{2\sigma^2}{\epsilon N} \mathbf{u}^\top (f(\mathbf{Y}_a + \epsilon \mathbf{u}) - f(\mathbf{Y}_a)) - \sigma^2$ — Monte-Carlo SURE with $\sigma^2$ estimated from frame variance and $\mathbf{u} \sim \mathcal{N}(0, I)$.

Empirical weights (round-3c): $(w_{\text{n2n}}, w_{\text{grad}}, w_{\text{ssim}}, w_{\text{contrast}}, w_{\text{sure}}) = (0.5, 0.7, 0.3, 0.5, 0.3)$.

**J-invariant blind-spot regulariser.** With probability $p_J = 0.10$ per training step, we randomly mask 10% of input ASL pixels (replaced by neighbour mean) before forward; the loss on masked positions forces neighbour-only inference, removing direct copy-paste of input noise (Krull 2019, Batson 2019).

### 3.6 Two-stage training

**Stage 1: T1 segmentation pre-training.** $T_1 \to (gm, wm, csf, bg)$ logits trained with L1 against partial-volume targets from FAST/FreeSurfer (one-time, single subject pool).

**Stage 2: ASL denoising.** Frozen T1 encoder + decoder used to generate cross-attention features $F^{T_1}$ and tissue-similarity bias; only ASL branch + cross-fusion gates trained.

### 3.7 Model selection and inference

**Primary criterion: upsnr_cyc** = `uPSNR − α · cyc`, α = 30.

*uPSNR* [Marcos-Morales et al. ICML 2023] is a pooled, asymptotically unbiased estimator of PSNR-to-clean. Let $T_b$ denote the held-out set_b length and $n = \lfloor T_b/3 \rfloor$. We split set_b into three disjoint subsets along the frame axis and average each:
$$a = \tfrac{1}{n}\!\sum_{t=1}^{n}\! y^{(b)}_t,\quad b = \tfrac{1}{n}\!\sum_{t=n+1}^{2n}\! y^{(b)}_t,\quad c = \tfrac{1}{n}\!\sum_{t=2n+1}^{3n}\! y^{(b)}_t.$$
Per-pixel residual squares and a variance-correction term are pooled (summed) over the entire validation set within the brain mask:
$$\text{uMSE} = \frac{\sum_{i\in\Omega} \big[(a_i - f(y^{(a)})_i)^2 - \tfrac{1}{2}(b_i - c_i)^2\big]}{\lvert\Omega\rvert},\quad \text{uPSNR} = 10\log_{10}\!\frac{M^2}{\text{uMSE}},\; M=1.$$
The $\tfrac{1}{2}(b-c)^2$ term has the same expectation as the noise variance contained in $(a - f)^2$ — its subtraction makes $\mathbb{E}[\text{uMSE}] = \text{MSE}(f, x_{\text{clean}})$ asymptotically. Pooled accumulation (single log over summed numerator/denominator) is essential: per-batch averaging incurs Jensen-inequality bias of order 1 dB on small batches.

*cyc* (subset consistency) measures input-noise stability. With $k = \lfloor T_a/2 \rfloor$ ($T_a \geq 4$) and the same EMA-averaged weights $f$:
$$\text{cyc} = \frac{1}{\lvert\Omega\rvert}\!\sum_{i\in\Omega}\!\big| f(y^{(a)}_{1:k})_i - f(y^{(a)}_{k+1:2k})_i\big|.$$
A model that overfits set_a noise produces inconsistent predictions on the two disjoint halves (large cyc); a model that recovers only the clean signal has cyc → 0.

The weight $\alpha = 30$ is calibrated from the dataset-level dispersions $\sigma_{\text{uPSNR}} \approx 0.15$ dB and $\sigma_{\text{cyc}} \approx 0.005$, so one $\sigma_{\text{cyc}}$ of stability variation is comparable to one $\sigma_{\text{uPSNR}}$ of fidelity variation. `upsnr_cyc` thus rewards *fidelity to the underlying clean signal* AND *robustness to input noise*, without using a biased reference.

We deliberately do **not** use `psnr_ref` (PSNR vs 12-NEX union) as the criterion: the 12-NEX union shares set_a's noise realisations with the prediction pipeline, so a model that overfits set_a noise can artificially inflate `psnr_ref`. Empirically on our data, `psnr_ref` and `upsnr_cyc` select opposite checkpoints; we trust the unbiased criterion.

**Supplementary metrics** (paper main / supp tables):
- `psnr_ref` / `ssim_ref` (literature alignment),
- `psnr_b` (Lehtinen ICML 2018 — unbiased held-out N2N PSNR with single reference; uPSNR is its low-variance generalisation),
- CNR (GM-WM, Wang convention),
- sCoV-GM / sCoV-WM (Wang 2003 ASL homogeneity),
- EFC (Atkinson 1997 MRI motion),
- lapvar / HFEN / GMSD / TG / IE / GE.

**SWA** (Stochastic Weight Averaging, Izmailov 2018) accumulated from step 200 to end provides an additional checkpoint complementing the single best.

**Inference**: at test time, dropout layers are kept active and $N$ stochastic forward passes are run per checkpoint $c$ over $C$ checkpoints (best.pth + swa.pth, $C=2$). The $N \times C$ samples produce a mean PWI and a per-voxel std map (epistemic uncertainty / confidence map).

### 3.8 Implementation details

| Item | Value |
|---|---|
| Framework | PyTorch 2.x |
| Optimiser | AdamW |
| Learning rate | $3 \times 10^{-4}$ with cosine schedule |
| Weight decay | $10^{-4}$ |
| Batch size | 8 |
| Image size | 128 × 128 (slice) |
| Total parameters | ~4.08M (trainable in stage-2: ~2.20M) |
| Bad-frame injection probability | 0.30 |
| J-invariance probability | 0.10 |
| SWA start step | 200 |
| Random seed | 42 |
| Hardware | single NVIDIA RTX 4090 |
| Training time | TODO |

---

## 4. Experiments

### 4.1 Dataset

(TODO — fill in)
- 7T ASL dataset, subjects = N (sex/age range), single-PLD pCASL, 12 NEX.
- Train / Val / Test split: 0.8 / 0.1 / 0.1.
- Preprocessing: MONAI `Orientationd LPS → Resize → Percentile normalise [0,1]`.
- Brain mask: $T_1 > 0.05$.

### 4.2 Baselines

| Method | Description | Reference |
|---|---|---|
| NLM | Non-local means on slice mean(set_a) | Buades 2005 |
| Supervised U-Net | Same backbone, target = mean(set_a ∪ set_b) | Xie 2020 |
| Vanilla N2N | Same backbone w/o T1, w/o SVFW | Lehtinen 2018 |
| Noise2Self | Same backbone, J-invariant loss only | Batson 2019 |
| **Ours (SVFW-Net)** | Full method | This work |

### 4.3 Evaluation metrics

| Domain | Metric | Note |
|---|---|---|
| PWI | **PSNR, SSIM, NMSE vs 12-NEX union** | Primary — approximate reference, mainstream ASL literature standard |
| PWI | psnr_b (held-out N2N PSNR, Lehtinen 2018) | Supplementary — unbiased diagnostic |
| PWI | Laplacian variance | Sharpness |
| CBF | GM CBF mean, WM CBF mean, GM/WM ratio, sCoV in GM | Physiological sanity |
| CBF | per-ROI Pearson, Bland-Altman | AAL atlas |
| Anti-hallucination | matched/mismatched-T1 L1 ratio | Project-specific |

### 4.4 Main results

(TODO — pending experimental data)
- Table 1: 5 methods × {PSNR, SSIM, NMSE, psnr_b} mean ± std + paired Wilcoxon p-values.
- Figure 2: visual comparison, 1 representative subject × 5 methods, PWI panels.
- Figure 3: same subjects in CBF domain.

### 4.5 Anti-hallucination sanity check

(TODO — run [scripts/test_mismatched_t1.py](../scripts/test_mismatched_t1.py))
- Figure 4: matched vs shuffled-T1 outputs, 4 columns × N rows.
- matched/mismatched L1 ratio table; expect ratio ≪ 1.
- Compare against `--use_t1_cross_fusion` ablation = "V=T1" variant (expect dramatic degradation).

### 4.6 Few-shot evaluation

(TODO — n_frames sweep via [runners/infer_pwi.py](../runners/infer_pwi.py))
- Figure 5a: PWI PSNR vs n_frames {2, 3, 4, 6, 8, 12}, all methods.
- Figure 5b: CBF MAE vs n_frames.
- Quantification: "n=4 ours ≈ n=12 baseline".

### 4.7 Ablation

(TODO — train and evaluate)

| Ablation | Drop |
|---|---|
| − SVFW (back to SetTransformer scalar BLUE) | tests per-pixel necessity |
| − V=ASL (replace with V=T1) | tests anti-hallucination cost |
| − cross-attention entirely (pure ASL branch) | tests T1 prior value |
| − 32×32 fusion (only 16×16 bottleneck) | tests multi-scale necessity |
| − SURE (`w_sure=0`) | tests divergence regulariser value |
| − J-invariance (`p_J=0`) | tests blind-spot regulariser value |
| − bad-frame injection (`p_bad=0`) | tests outlier robustness on injected-corruption val |

Table 2: Ablation table on full val set.

### 4.8 Uncertainty calibration

- Figure 6: confidence map overlay on PWI.
- Calibration: spatial Pearson(σ_pred, |recon − ref|).
- Confidence-weighted ROI CBF aggregation.

### 4.9 Reproducibility

- Intra-subject ICC over set_a sub-sampling.
- Bland-Altman vs 12-NEX CBF.
- Left/right hemisphere CBF correlation.

### 4.10 Failure cases

- Worst-5 subjects discussion (motion, low SNR, atypical anatomy).

---

## 5. Discussion

### 5.1 Why per-pixel BLUE matters
- Motion is locally heterogeneous: a frame may be reliable in one region and corrupted elsewhere. Scalar weighting wastes information from partially-good frames.
- SVFW's deviation-only input prevents lesion suppression, addressing a known concern with adaptive aggregation methods.

### 5.2 Anti-hallucination as a structural property
- Loss-based anti-hallucination (e.g., adversarial T1 mismatch loss) can fail under distribution shift.
- V=ASL is a hard constraint of the architecture: T1 *cannot* contribute pixel content regardless of training.
- Mismatched-T1 sanity check confirms this empirically (§4.5).

### 5.3 Limitations
- Single-site 7T dataset; cross-scanner / cross-site validation pending.
- No clinical reader study; no patient cohort.
- No CBF quantification under multi-PLD or dispersion correction.
- Confidence map calibration validated only against held-out reference, not against true noise variance.

### 5.4 Future work
- Multi-site validation; patient (stroke / AD / glioma) data.
- Clinical reader study.
- β-NLL (Seitzer ICLR 2022) for joint μ/σ² training (heteroscedastic was tested and dropped — see ablation).
- Extension to multi-PLD ASL.

---

## 6. Conclusion

(TODO — summarise main findings in 2 paragraphs once results in.)

---

## Acknowledgements / Funding (TODO)

---

## References

(Bibliography — collect from [docs/related_work.md](related_work.md) and [docs/papers/](papers/) as we go. Full BibTeX in supplementary.)

Key references already verified:
1. Lehtinen et al., "Noise2Noise", ICML 2018.
2. Krull et al., "Noise2Void", CVPR 2019.
3. Batson & Royer, "Noise2Self", ICML 2019.
4. Soltanayev & Chun, "Training and Refining Deep Learning Networks for Image Denoising via SURE", NeurIPS 2018.
5. Stein, "Estimation of the Mean of a Multivariate Normal Distribution", Annals of Statistics 1981.
6. Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018 (SWA).
7. Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017.
8. Lee et al., "Set Transformer", ICML 2019.
9. Xie et al., "Denoising arterial spin labeling perfusion MRI with deep machine learning", *Magnetic Resonance Imaging* 2020.
10. Kim et al., "Improving Arterial Spin Labeling by Using Deep Learning", *Radiology* 2018.
11. Gong et al., "Arterial spin labeling MR image denoising and reconstruction using unsupervised deep learning", 2020.
12. Shou et al., "Transformer-based deep learning denoising of single and multi-delay 3D ASL", *MRM* 2024.
13. 2025 NeuroImage paper on pediatric multi-delay ASL self-supervised Transformer denoising.
14. Wang et al., 2003 ASL sCoV.

---

## TODO — Concrete writing tasks

| # | Task | Depends on | ETA |
|---|---|---|---|
| W1 | Fill §1 with subject count and dataset details | dataset confirm | 0.5 day |
| W2 | Write §3 in full (Methods) — DONE in draft, polish | — | 0.5 day |
| W3 | Generate Figure 1 (architecture diagram) | tikz / drawio | 0.5 day |
| W4 | Run §4.4 main results pipeline | v37 best.pth | 1 day |
| W5 | Run §4.5 mismatched-T1 sanity | v37 best.pth | 0.5 day |
| W6 | Run §4.6 few-shot sweep | v37 best.pth | 0.5 day |
| W7 | Train + eval §4.7 ablation (7 variants) | GPU time | 4 days |
| W8 | Run §4.8 uncertainty calibration | v37 best.pth | 0.5 day |
| W9 | Compute CBF + §4.3-§4.6 CBF-domain extensions | M0/PLD info confirmed | 2 days |
| W10 | Write §1, §2, §5, §6, abstract | results in | 2 days |
| W11 | Generate all figures / tables | results in | 2 days |
| W12 | BibTeX cleanup and format | — | 1 day |
