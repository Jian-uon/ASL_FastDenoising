# Validation metrics — what we considered, what we rejected, what we ship

> 本文档记录 v37 项目在"自监督 ASL 去噪、无 clean GT"场景下，对各类
> validation / model-selection 指标的探索过程、量化对比与最终结论。
> 用于论文 supplementary、专利附录与未来项目复用。

> **⚠️ 当前状态（2026-06-27）：§1–§3（指标全表 + composite 教训）仍是 single source
> of truth；但 §4–§7 的 "no-selection + swa-from-step-200" 方案是 v37 历史，已被 v2
> 取代——当前选择哲学、指标分层与报告规则见本文件末尾 [§9](#9-当前-v2-选择哲学与指标分层2026-06-16load-bearing)。
> 核心原则：主导最终操作点选择与效果展示的是 ASL 相关 QC 指标
> （CNR 优先，sCoV / lapvar 辅助）。**user 反复强调，load-bearing。**
> **最终模型选择更新（2026-06-27, user sign-off）：final 操作点 = 全程保存的 ckpt 中
> GM-WM `CNR` 最高者（global argmax CNR，即 `best-CNR selection`），无 uMSE 保真带、无 SWA
> ckpt 平均，产物从 `swa_cnr_primary.pth` 改名为 `best_cnr.pth`。`uMSE`/`uPSNR` 不再做最终
> 选择约束（降为诊断/补充报告项），但 in-loop early-stop 仍默认 `--best_criterion umse`，且
> 选择仍永不使用 L1 / psnr_ref / psnr_b。详见 §9.2。****

---

## 0. 核心约束

ASL 12-NEX 数据无任何 clean ground truth。任何指标必须满足：
- **不依赖 noisy 12-NEX 平均当 GT**（用了就有 noise mimicry 偏差）
- **不依赖单一 hold-out 单帧**（方差太大）
- 优先使用**理论可证 unbiased**或**self-consistent**的统计量

---

## 1. 候选指标全表

| 指标 | 范围 | 类型 | 取值方向 | 文献 / 来源 |
|------|------|------|---------|-------------|
| `psnr_ref` | dB | biased ref | ↑ 越大越好 | vs `mean(set_a ∪ set_b)`（12-NEX 全集） |
| `psnr_b` | dB | unbiased single-ref | ↑ | Lehtinen N2N (ICML 2018) — vs `mean(set_b)` |
| `uPSNR` | dB | unbiased pooled | ↑ | Marcos-Morales et al. (ICML 2023, arXiv:2210.05553) |
| `l1_B` | [0,1] | masked L1 | ↓ | vs `mean(set_b)`（与 psnr_b 同源不同度量）|
| `cyc` (subset_consistency) | [0,1] | self-cons L1 | ↓ | `L1(f(set_a[:k]), f(set_a[k:2k]))` |
| `CNR_pred / CNR_ref` | unitless | image stats | match ref | Wang convention：`|μ_GM − μ_WM| / σ_WM` |
| `sCoV-GM / sCoV-WM` | unitless | image stats | ↓（接近 ref）| Wang 2003 ASL homogeneity |
| `EFC` (Entropy Focus Criterion) | unitless | image stats | ↓ | Atkinson 1997 IEEE TMI（运动伪影代理）|
| `lapvar`, `hfen`, `gmsd`, `tg`, `ie`, `ge` | various | image stats | various | sharpness / high-freq energy proxies |
| `sure` (MC-Stein) | scalar | unbiased risk | 接近 0 | Ramani et al. IEEE TIP 2008 |

---

## 2. 在本数据集上的实测动态范围（v37, run_full_v37 训练 305 step）

| 指标 | 起点 → 终点 | 总变动 | 动态范围（max-min/std）| 单调性 |
|------|-------------|--------|------------------------|--------|
| `psnr_ref` | 25.26 → 9.06 | −16.2 dB | ~10 dB（noisy） | 单调下降（解耦 noise） |
| `psnr_b` | 21.84 → ~22 | ±1 dB | ~1 dB | 早期↑后期回落 |
| `uPSNR` | 17.5 → 23.7（peak）→ 22.0 | +6.2 dB | ~6 dB | ↑后回落 |
| `l1_B` | 0.28 → 0.045 | −0.24 | ~5× | 单调↓ |
| `cyc` | 0.13 → 0.04 | −0.09 (~3×) | ~3× | 单调↓ |
| `CNR_pred` | 0.41 → 0.65 | +0.24 | 稳定后 ~0% 波动 | 早期↑稳定 |
| `CNR_ref` | 0.46（不变） | — | <1% | 数据集常数 |
| `sCoV-GM ratio` | — | ~2% 波动 | ~2% | 几乎平 |
| `sCoV-WM ratio` | — | ~4% 波动 | ~4% | 几乎平 |
| `EFC` | — | ~0.2% 波动 | <0.5% | 接近 noise floor |
| `hfen` | 0.80 → 0.58 | −0.22 | ~0.2 | 早期↓稳定 |
| `lapvar` | 1.0 → 0.28 | −0.7 | ~3× | 早期↓稳定 |

**结论**：
- `EFC / sCoV / CNR / hfen / lapvar` 在 step ~50 后基本饱和，**动态范围太小**（<5%），
  无法区分 step 100 vs step 1000 的细微差别。
- `psnr_ref` 动态范围大但**反向**——它在持续奖励对 noise reference 的 mimicry。
- `uPSNR / l1_B / cyc` 才是真正"训练越久指标越好"的几个量。
- `psnr_b` 范围太小（±1 dB），方差大于 trend，难以做 model selection。

---

## 3. 我们尝试的 composite criterion：`upsnr_cyc`

### 3.1 设计

`score = uPSNR − α · cyc`，α=30。

**动机**：
- `uPSNR` 捕捉 fidelity（unbiased to clean）
- `cyc` 捕捉 input-stability（小 = 模型不依赖输入噪声实例）
- α=30 来自 `σ_uPSNR / σ_cyc ≈ 0.15 / 0.005`（让两项 1σ 相当）

### 3.2 为什么这是一个**坏想法**（事后总结）

| 问题 | 说明 |
|------|------|
| **量纲不齐** | uPSNR 在 dB（log），cyc 在 [0,1]（线性）。线性相加无 well-defined 单位 |
| **可被 low-variance 解钻空子** | 模型把输出推向接近常数 → cyc 自然小，但 uPSNR 的 variance correction 项 `½(b−c)²` 同时把 noise 抵消 → composite 看起来上涨；实际 `l1_B` 暴涨到 0.23（5×正常值）、`psnr_ref` 跌 16 dB 都没被准则发现 |
| **非文献标准** | 自监督去噪文献里没人这么做，reviewer 难以接受 |
| **α 校准依赖经验** | σ 估计本身有方差；不同数据集 α 要重调 |

### 3.3 实证证据：上一次 305-step 训练

```
step 200: upsnr_cyc=21.22 (BEST gated past) | l1_B=0.231 psnr_ref=8.52
step 205: upsnr_cyc=22.15 (BEST)            | l1_B=0.228 psnr_ref=8.61 hfen=0.578
step 305: upsnr_cyc=20.73                   | l1_B=0.??  psnr_ref=9.06
```

`l1_B = 0.23` 在 [0,1] 归一化数据上表示**像素 L1 误差 0.23**——绝不可能是好模型。
但 composite criterion 选了它当 best.pth。

诊断查 TensorBoard：分项 `uPSNR ≈ 23.5 dB`（理论最优），`cyc ≈ 0.044`（很小），
`CNR_pred ≈ 0.65`（高于 ref 0.46，无 over-smoothing）。说明模型**结构上**没问题，
是 **`psnr_ref` / `l1_B` 的归一化口径与 uPSNR / CNR 不一致**——但 composite 对
此盲。

---

## 4. 最终方案：**no-selection mode**（Self2Self / N2N2 convention）

### 4.1 决策

- `--best_criterion none`：**不选 best ckpt，不做早停**
- 训练跑满 `max_steps=1000`（cosine LR schedule，eta_min=5e-6）
- 论文报告 `swa.pth`（SWA 平均，from step 200）作为 paper-grade 模型
- `latest.pth` 作为 supplementary

### 4.2 为什么这是更好的选择

| 理由 | 说明 |
|------|------|
| **避免 selection bias** | 任何基于 noisy 量的 selection 都引入 bias；不选就没这个问题 |
| **文献先例** | Self2Self (CVPR 2020)、Neighbor2Neighbor (CVPR 2021) 都是固定 step + SWA/avg |
| **SWA 自带正则** | Izmailov UAI 2018，SWA 平均通常优于任何单 ckpt，自动平滑掉 val noise |
| **论文叙述简洁** | "trained for N steps with cosine LR; no validation-based selection" — reviewer 立即接受 |
| **cosine LR 自然降温** | 训练末段 LR → 5e-6，模型自动停止 noise fit，无需早停 |

### 4.3 在论文里怎么写

> *Model selection.* Following Self2Self (Quan et al., CVPR 2020) and
> Neighbor2Neighbor (Huang et al., CVPR 2021), we avoid validation-based
> checkpoint selection because no clean reference is available in our
> self-supervised setting and any reference-based selection criterion
> introduces noise-mimicry bias. We train for a fixed budget of 1000 steps
> with a cosine learning-rate schedule (3e-4 → 5e-6) and report the
> Stochastic Weight Average (SWA, Izmailov et al., UAI 2018) of weights
> from step 200 to the end of training as the paper-grade model.

### 4.4 实施

实现于 [runners/asl_t1_guided_runner_dmvae_n2n.py](../runners/asl_t1_guided_runner_dmvae_n2n.py)
`--best_criterion none` 选项：跳过 best ckpt 选择 + 早停逻辑，仅记录每 eval
事件的全部指标到 stdout / TensorBoard，最后保存 `latest.pth` + `swa.pth`。

启动脚本：`scripts/archive/auto_chain_v37_multiseed.sh`（v37 线的历史脚本，**未随本仓库迁移**，留在
源仓库 `ASL_dmvae`）。本仓库的对应入口是 [env/hpc/slurm/submit_v35_joint.sh](../env/hpc/slurm/submit_v35_joint.sh)。

---

## 5. 论文里如何报告这些指标

| 指标 | 用途 | 表格位置 |
|------|------|---------|
| `uPSNR` | **主指标**，与 baseline 对比 fidelity | Table 1 主结果（mean ± std over val subjects）|
| `psnr_b` | 对照（Lehtinen N2N held-out PSNR 文献对齐）| Table 1 同行 |
| `psnr_ref` | 文献对齐（Xie 2020、Shou 2024 习惯报告 vs 12-NEX）| Table 1 同行，**括号注 biased**|
| `cyc` | 模型稳定性 | Table 2 stability column |
| `CNR_pred / CNR_ref` | 解剖对比度保真 | Table 2 contrast column |
| `sCoV-GM / WM` | ASL 标准灌注均匀性 | Table 2 |
| `EFC, hfen, lapvar, gmsd, tg, ie, ge` | 高频内容 / 锐度 paper supplementary 矩阵 | Supp. Table S1 |
| 训练曲线 | uPSNR / cyc / CNR / l1_B 随 step 变化 | Supp. Figure S1（4 subplot）|
| 不同选择准则结果 | psnr_ref-best / psnr_b-best / uPSNR-best / SWA 各自的最终 metrics | Supp. Table S2（消融）|

**关键写法**：主结果不出现 composite，selection 用 `swa.pth`，把 composite
作为 Supp. S2 的一行 ablation（"Composite criterion (uPSNR − 30·cyc)" 一行 +
解释为何 unstable）。

---

## 6. 后处理：从 ckpt 池事后选模型（论文实验设计）

`auto_chain_v37_multiseed.sh` 的产物：

```
run_full_v37_long/checkpoints/
├── latest.pth          (step 1000)
├── swa.pth             (avg from step 200)
├── step000050.pth, step000100.pth, ..., step001000.pth   (每 50 step)
```

[scripts/eval_iqa_metrics.py](../scripts/eval_iqa_metrics.py) 已支持遍历 ckpt
列表输出全部指标。运行：

```bash
python scripts/eval_iqa_metrics.py \
    --config env/local/configs/win_asl_2d_home_v37.yml \
    --exp C:/tmp/asl_exp --name run_full_v37_long \
    --base_ch 32 --depth 4 --use_t1_cross_fusion --use_svfw \
    --init_t1_from C:/tmp/asl_exp/logs/stage1_t1_300step/checkpoints/latest.pth \
    --freeze_t1 \
    --eval_ckpts swa.pth latest.pth step000200.pth step000400.pth \
                  step000600.pth step000800.pth step001000.pth
```

输出每 ckpt 的 uPSNR / cyc / CNR / psnr_b / sCoV / lapvar 全表，写到论文
Supp. Table S2。

---

## 7. 时间线（决策日志）

| 日期 | 事件 |
|------|------|
| 2026-05-08 | v37 上线 (SVFW + multi-scale fusion)，criterion=`psnr_b` |
| 2026-05-08 | 调研 EFC / sCoV，加入指标但不作 best 选择 |
| 2026-05-08 | 测试 uPSNR (Marcos-Morales 2023)、cyc、CNR；动态范围分析 |
| 2026-05-08 | 设计 `upsnr_cyc = uPSNR − 30·cyc` composite，启用为 best criterion |
| 2026-05-09 | 305-step 训练完成；composite best (step 205) 在 `l1_B=0.23, psnr_ref=8.5` 退化区取到 22.15 高分；diagnose 出 composite 盲区 |
| 2026-05-10 | 切换到 `--best_criterion none` (Self2Self/N2N2 convention)；开始 1000-step 长 cosine 训练 |

---

## 8. 经验教训

1. **任何 self-sup composite criterion 都要设计 sanity-gate**——单纯线性相加
   会在某个分项暴跌时被另一项掩盖；如果一定要 composite，用 `min(项1, 项2)`
   或硬门槛（"项2 < threshold 才允许进 best"）。
2. **量纲不一致的相加 = 隐式权重**。dB + linear L1 + ratio 加在一起几乎一定
   会 collapse 到某个分项主导。
3. **No-selection 的简洁性是真的优势**——少一个准则讨论，少一个 reviewer
   吐槽点，论文叙述紧。
4. **训练曲线本身就是 supplementary 表的内容**。不需要选 best 就能写出
   "uPSNR 在 step 600 达 24.1 dB" 等定量陈述。
5. **SWA 几乎总是值得开**：成本极小（一份额外 ckpt），增益稳定（0.1–0.3 dB
   uPSNR）。

---

## 9. 当前 v2 选择哲学与指标分层（2026-06-16, load-bearing）

> 取代 §4–§7 的 v37 "no-selection" 方案。**核心原则（user 反复强调）：真正主导模型选择
> 与效果展示的是 **ASL 相关 QC 指标**——`CNR`（GM-WM 对比，优先），辅以 `sCoV-GM/WM`
> （灌注均匀性）、`lapvar`（锐度）。** 理由：`uMSE`/`uPSNR` 在 step≈50 之后动态范围太小
> （§2，<5%），单独无法分辨 step100 vs step1000 的模型优劣。区分"好模型"靠的是 ASL QC 指标。
>
> **更新（2026-06-27, user sign-off）：最终操作点选择改为 `best-CNR selection`——全程 ckpt 中
> GM-WM `CNR` 全局最高者（global argmax CNR），不再用 `uMSE` 1-SE 保真带、也不再做 SWA。
> 因此 `uMSE` 不再是最终选择的 *bar* / 可行性约束，仅作诊断与补充报告（in-loop early-stop 仍以
> `umse` 为默认准则）。详见 §9.2。**

### 9.1 指标分层（按"能否据此选模型"）

**A — 可据此选择 / early-stop（自监督，无偏或 self-consistent）。`uMSE` 是这里的 bar：**

| 指标 | 方向 | 出处 / 含义 | 角色 |
|------|------|-------------|------|
| `umse` | ↓ | 无偏 MSE，Marcos-Morales ICML2023：`E[(a−pred)²]−½E[(b−c)²]` | **in-loop bar**（argmin） |
| `uPSNR` | ↑ | `umse` 的 dB 形式 | 同上，dB 报告 |
| `upsnr_cyc` | ↑ | `uPSNR − 30·cyc`，composite | **弃用**（§3：被 low-variance 钻空子） |
| `constrained_umse` | ↓ | uMSE 在 cyc/lapvar 约束下 | 可选 |
| `cyc` | ↓ | subset consistency `L1(f(A[:k]),f(A[k:2k]))` | 约束/诊断项 |
| `sure` | →0 | MC-Stein 无偏风险，Ramani 2008 | 诊断 |

**B — ASL QC 指标（每个 VAL 行都报；主导选择 + 主要效果展示；final 操作点目标）：**

| 指标 | 方向 | 含义 |
|------|------|------|
| **`CNR` / `cnr_ref`** | ↑ | GM-WM 对比 `|μ_GM−μ_WM|/σ_WM`（Wang）—— **主导指标 + final 操作点目标（global argmax CNR）** |
| `sCoV-GM` / `sCoV-WM` | ↓ | Wang 2003 组织内灌注均匀性 |
| `lapvar` / `lapvar_ratio` | — | 锐度 / 高频内容（相对噪声参考） |
| `snr_gm` / `snr_wm` | ↑ | 组织 SNR |

**C — 只报告、从不据此选（biased reference / 文献对齐）：**

| 指标 | 含义 |
|------|------|
| `psnr_ref` | vs 12-NEX union —— **有偏**，奖励 noise mimicry（早峰后跌），报告须括注 biased |
| `psnr_b` | Lehtinen N2N hold-out，vs `mean(set_b)` |
| `l1_B` | masked L1 vs `mean(set_b)` |
| `ssim_ref` / `EFC`(Atkinson 1997) / `hfen` / `gmsd` / `tg`/`ie`/`ge` | 锐度 / 高频 proxy + 文献对齐 |

> **硬约束（§3 hard-constraint）**：best-ckpt / early-stop **永不**用 `L1 / psnr_ref / psnr_b`。

**D — Sanity gate（不是连续选择指标，是发表前必过的关卡）：**
- **mismatched-T1**：`match_vs_mismatch_l1`（T1 泄漏 / 安全）+ psnr/ssim。⚠️ 历史上该指标曾为全图未-mask；
  按 §2 必须用**原 subject** 的 `t1>0.05` 脑 mask（绝不能用错配 T1 的 mask）——
  [scripts/test_mismatched_t1.py](../scripts/test_mismatched_t1.py) 的 `compute_metrics` 已按此实现。
- **n-frame sweep**：2–8 帧的 uMSE/CNR graceful degradation 曲线。

### 9.2 选择规则

- **in-loop best**：`--best_criterion umse`（argmin，仍是默认 early-stop/best-ckpt 准则），`best_min_step 40` burn-in，`score_ema_alpha 0.3`。存 `best.pth`。
- **final 操作点模型（2026-06-27, user sign-off — `best-CNR selection`）**：取全程保存 ckpt 中 **GM-WM `CNR` 全局最高者**（global argmax CNR）。**没有 `uMSE` 保真带，也不做任何 SWA / ckpt 平均**。由
  ```
  scripts/eval_select_ckpt.py --metric cnr --save_selected <run>/best_cnr.pth
  ```
  产出（遍历每个 ckpt，把 argmax-CNR 的 ckpt 原样拷贝）。产物从 `swa_cnr_primary.pth` 改名为 `best_cnr.pth`。HPC phase3 现对全部 3 个核心模型跑这一条命令做 **matched selection**（也消除了旧 "ours=CNR-SWA vs baseline=best-uMSE" 的选择不对称 confound）。
  - **取代** 旧规则（post-hoc CNR-primary feasible-set SWA：在 `uMSE` 1-SE 带内以 max-CNR 取高-CNR 成员做 SWA → `swa_cnr_primary.pth`，uMSE 当保真 bar；保守替代 `swa_feasible.pth` = 同带内 min-sCoV + lapvar floor）。旧 SWA 脚本（`scripts/select_cnr_primary.py`、`scripts/build_swa_feasible.py`、`scripts/archive/build_swa_cnr_primary.sh`）仍留在磁盘但已**不再接入 pipeline（retained/deprecated）**。
  - **诚实后果（不软化）**：global max-CNR 可能落在退化尾段——此处 `uMSE` 升过旧 1-SE bar，`lapvar`/`hfen` 也升高（更多纹理/噪声）。这是**有意、user 接受的权衡**：`CNR` 现在是最终操作点的**唯一**判据，`uMSE` 不再对 final pick 起守门作用。在 `run_wd1e4_probe` 上，global-max-CNR 落在 **step150（CNR 0.6529，uMSE 0.00793，源自 feasibility_full.json）**，而旧 SWA pick 为 {step80, step100}。
- **never** L1 / psnr_ref / psnr_b。

### 9.3 报告规则（效果展示时把相关指标都报出来）

- **主表**：ASL QC（`CNR`、`sCoV-GM`、`sCoV-WM`、`lapvar`、`snr_gm/wm`）+ `uMSE`（诊断/补充，不再是最终选择约束，更不是 winner 指标）。
- **补充表**：`uMSE` / `uPSNR`（诊断）、`psnr_ref`（括注 biased）、`psnr_b`、`EFC`、`hfen`、`gmsd`、`ssim_ref`。
- **sanity**：mismatched-T1 安全表 + n-frame 退化曲线。
- 不把 `uMSE` 当 headline winner——headline 是 ASL QC（CNR 优先）；自 2026-06-27 起 `uMSE` 也不再是 final pick 的守门约束（见 §9.2）。
