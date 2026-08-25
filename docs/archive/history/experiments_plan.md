# BSPC / CMIG 投稿实验计划与进度

> 最后更新：2026-05-08
> 目标期刊：**BSPC**（IF ~5.1）/ **CMIG**（IF ~5.4）— 信号处理 & 医学影像方法向，对 self-supervised + 数学正则友好。
> 主方法：**v37**（[architecture.md](architecture.md)）= 多尺度 V=ASL cross-attention + SVFW per-pixel BLUE + L1 N2N + Monte-Carlo SURE + J-invariance。
> 次要 fallback：MRI 期刊（IF 2.8，ASL 社区友好）。

---

## 0. 进度概览

| 项 | 状态 | 备注 |
|---|---|---|
| 核心方法实现（v37）| ✅ | SVFW + cross-fusion + loss + SWA + MC dropout 全部代码就绪 |
| Stage 1 T1 branch 训练 | ✅ | `stage1_t1_padcrop` 作为 frozen T1 init |
| **v37 round-3 训练** | 🟡 进行中 | step 25 psnr_b=23.67（已超 v36 BEST 23.11）；早停或 step 500 完成 |
| 4 baseline ckpt（NLM / sup / n2n / n2self）| ✅ | 待用同一 split manifest 重测 |
| Inference + n_frames sweep + MC dropout | ✅ | [runners/infer_pwi.py](../runners/infer_pwi.py) `--mc_n / --mc_ckpts` |
| Mismatched-T1 sanity 脚本 | ✅ | [scripts/test_mismatched_t1.py](../scripts/test_mismatched_t1.py)，待跑 v37 |
| Eval baselines 脚本 | ✅ | [runners/eval_baselines.py](../runners/eval_baselines.py)，待跑 v37 |
| 主结果 / Ablation / Sanity / Few-shot / Uncertainty / Reproducibility / Failure | ⬜ | v37 best.pth 出来后启动（见 §2-§7）|
| **CBF pipeline**（Buxton + BASIL 比对） | ⬜ | 计划新建 [runners/compute_cbf.py](../runners/compute_cbf.py)（见 §3）|
| Paper draft + figures | ⬜ | 实验完成后 1-2 周 |

---

## 1. 数据准备

| 项 | 现状 | 备注 |
|---|---|---|
| Subject 数 | TBD | 需查 DatasetGenerator 输出 |
| Train / Val / Test 划分 | 0.8 / 0.1 / 0.1 | [config/win_asl_2d_home.yml](../config/win_asl_2d_home.yml) |
| Fixed split manifest | 部分 | 评估前 dump 一份 (subject_id, slice_idx, set_a_idx, set_b_idx) 给所有方法用 |
| M0 / PLD / α / T1_blood 元信息 | TBD | CBF 计算 hard prerequisite，先确认 |

---

## 2. 实验清单（投稿 BSPC / CMIG 必做）

### 2.1 主结果（Main Results）

5 方法对比：**ours v37** vs NLM / supervised / vanilla N2N / N2Self。

| # | 实验 | 域 | 输出 |
|---|---|---|---|
| M1 | PSNR / SSIM / NMSE vs 12-NEX union | PWI + CBF | summary table |
| M2 | 跨 subject 配对 Wilcoxon signed-rank | PWI + CBF | p-value 列 |
| M3 | GM CBF mean / WM CBF mean / 比值 / sCoV in GM | CBF | 物理指标 table |
| M4 | 1 个代表 subject × 5 方法可视化 | PWI + CBF | main figure |

### 2.2 Ablation

| # | Ablation | 训练 | 期望发现 |
|---|---|---|---|
| A1 | w/o SVFW（恢复 SetTransformer 标量 BLUE） | 12h | SVFW 在 motion / outlier 主导 subjects 的优势 |
| A2 | w/o V=ASL（V=T1） | 12h | mismatched-T1 退化最大 |
| A3 | w/o cross-attention（pure ASL branch）| 12h | T1 prior 整体价值 |
| A4 | 仅保留 16×16 cross-fusion（去 32×32）| 12h | 单层 fusion 不够（gate→0 collapse）|
| A5 | 仅保留 32×32（去 16×16）| 12h | 32×32 是主贡献的证据 |
| A6 | w/o SURE（w_sure=0）| 12h | divergence regularization 的价值 |
| A7 | w/o J-invariance（jinv_p=0）| 12h | blind-spot 正则的价值 |
| A8 | w/o bad-frame injection | 12h | 在含 outlier frames 的 val 集上对比 |

### 2.3 Anti-hallucination Sanity ⭐ 项目核心卖点

| # | 实验 | 输出 |
|---|---|---|
| S1 | Mismatched-T1（PWI 域）：matched vs shuffled-T1 | 关键 figure |
| S2 | Mismatched-T1（CBF 域）：per-region CBF correlation | table |
| S3 | matched/mismatched L1 比值（应 ≪ 1）| 数字 |
| S4 | A2/A3 ablation 在 mismatched 下的退化对比 | T1 hallucination 的硬证据 figure |

### 2.4 Few-shot Evaluation

| # | 实验 | 输出 |
|---|---|---|
| F1 | n_frames sweep {2,3,4,6,8,12} PWI PSNR/SSIM | curve figure |
| F2 | n_frames sweep CBF MAE | curve figure |
| F3 | 加速比量化（"n=4 ≈ baseline n=12"）| 表格 |

### 2.5 Uncertainty Quantification ⭐ 项目独有

| # | 实验 | 输出 |
|---|---|---|
| U1 | MC dropout × SWA-ensemble confidence map | per-subject confmap NIfTI |
| U2 | Calibration: σ_pred vs \|recon − ref\| spatial Pearson | 数字 + scatter |
| U3 | Confidence-weighted CBF（σ 加权 ROI 聚合）| robustness table |

### 2.6 Reproducibility

| # | 实验 | 输出 |
|---|---|---|
| R1 | Intra-subject ICC（同 subject 不同 set_a 抽样）| ICC > 0.9 table |
| R2 | Bland-Altman（ours vs 12-NEX CBF）| figure |
| R3 | 左右半球对称性（healthy 应高相关）| 数字 |

### 2.7 Failure Case 分析

| # | 实验 | 输出 |
|---|---|---|
| W1 | Worst-5 subjects PSNR 分析 | 1 figure + 半页 discussion |
| W2 | Edge cases（高运动 / 低 SNR）| 表格 |

---

## 3. CBF 定量 pipeline

由用户决策**加 CBF 域分析**——这是 BSPC/CMIG 的实质性加分项（接近 MRM 工作量）。

| 步骤 | 内容 | 工作量 |
|---|---|---|
| 3.1 数据元信息确认 | M0 / PLD / τ / α / T1_blood @ 7T | 1h |
| 3.2 实现 `runners/compute_cbf.py` | pCASL Buxton single-PLD：`CBF = 6000·λ·ΔM·exp(PLD/T1b) / (2·α·T1b·M0·(1−exp(−τ/T1b)))` | 1 天 |
| 3.3 BASIL (FSL `oxford_asl`) 比对 | 抽 5 subjects 验证 GM mean 误差 < 5% | 0.5 天 |
| 3.4 批量算 5 方法 CBF NIfTI | 全 val 集 | 0.5 天 |
| 3.5 ROI 分析 | AAL / Brainnetome atlas，per-ROI CBF | 1 天 |
| 3.6 Bland-Altman + 双侧对称性 | figure + table | 0.5 天 |

**fallback**：如果数据集没有 M0 → 用 control image 做 proxy（精度损失 ~10%）；如果连 control 都缺 → 改报 ΔM-domain 指标，paper 里明确说 "we report ΔM-domain metrics rather than CBF"。

---

## 4. 评价指标

| 类别 | 指标 | 备注 |
|---|---|---|
| **图像质量**（PWI + CBF）| PSNR、SSIM、NMSE vs 12-NEX | 通用 |
| **ASL 物理**（CBF）| GM CBF mean / WM CBF mean / 比值（应 ≈ 2-3）/ sCoV in GM | ASL paper 必报，否则审稿打回 |
| **锐度**（PWI）| Laplacian variance | 训练时已记录 |
| **Self-supervised 诊断**（PWI）| psnr_b（Lehtinen 2018 unbiased hold-out PSNR）| 仅用于训练 trajectory 诊断；论文报为 supplementary |
| **Anti-hallucination** | Mismatched-T1 L1（matched/mismatched 比值） | 独有，强调 |
| **Reproducibility** | ICC（intra-subject）/ Bland-Altman LoA | 临床期刊偏爱 |
| **Uncertainty Calibration** | σ_pred vs error 的 Pearson | 独有 |

---

## 5. Baselines

| Baseline | 实现 | ckpt |
|---|---|---|
| NLM (skimage) | [runners/eval_nlm.py](../runners/eval_nlm.py) | 推理时算，无训练 |
| Supervised U-Net | [runners/train_baseline.py --mode sup](../runners/train_baseline.py) | `baseline_sup/checkpoints/best.pth` |
| Vanilla N2N (无 T1, 无 SetTrans, 无 cross-attn) | [runners/train_baseline.py --mode n2n](../runners/train_baseline.py) | `baseline_n2n/checkpoints/best.pth` |
| Noise2Self（J-invariant） | [runners/train_baseline.py --mode n2self](../runners/train_baseline.py) | `baseline_n2self/checkpoints/best.pth` |

⚠ baseline ckpt 是早期版本训练的，可能需要在 fixed split manifest 上重训。

---

## 6. 实验时间表

```
Week 1（v37 round-3 出 best.pth 后立即开做）:
  Day 1-2:  数据元信息确认 + CBF pipeline 实现 + BASIL 比对
  Day 3:    fixed split manifest dump + 4 baseline 重测
  Day 4-5:  M1-M4 主结果 + S1-S4 mismatched-T1

Week 2（ablation 训练，并行 GPU）:
  6 个 ablation 训练（A1/A2/A3/A6/A7/A8 优先；A4/A5 次之）
  完成后批量 eval

Week 3:
  F1-F3 few-shot sweep
  U1-U3 uncertainty calibration
  R1-R3 reproducibility
  W1-W2 failure cases

Week 4:
  Paper draft + figures 排版 + supplementary
```

**总周期**：4 周（v37 训完后启动）。BSPC/CMIG review cycle ~8-12 周 → 整体 5-6 个月到 acceptance。

---

## 7. 风险与决策点

| 风险 | 现状 | 缓解 |
|---|---|---|
| v37 round-3 训练 silent crash | 已发生 1 次 @ step 30 → 已重启 | 加 GPU 监控 + 周期 ckpt（已配置 save_every=50） |
| psnr_b 没超 v36 BEST | 当前 trajectory 已超过 | — |
| 数据集 M0 缺失 | 需确认 | 用 control image proxy 或改报 ΔM 指标 |
| Subject 数过少（< 10） | 需确认 | BSPC/CMIG 7-10 仍可投，再少要补合成数据 |
| Cross-attention ablation 训练时间过长 | 6 × 12h = 3 天 | 缩 max_steps 到 300（够看趋势）|

---

## 8. 已弃用方向（保留作 ablation）

| 方向 | 弃用原因 | 实现 flag |
|---|---|---|
| Heteroscedastic NLL head | 与 noisy N2N target 不兼容（σ²→max collapse）| `--use_heteroscedastic` |
| TissueFiLM（per-channel γ/β by global tissue PV） | 视觉上 recon 显出 T1-like 沟回 → hallucination 风险 | `--use_film` |
| TV regulariser | 替换为 J-invariance（principled blind-spot）| `w_tv` config |
| composite_v2 best criterion | 替换为 psnr_b（Lehtinen 2018 unbiased hold-out PSNR）| `--best_criterion` flag |
| Wavelet DWT pool/unpool | 与未调优 cross-fusion 失配，psnr 反降 | `--use_wavelet` |

弃用方向**仍然出现在 paper 里**作为 ablation，证明设计选择不是 arbitrary。
