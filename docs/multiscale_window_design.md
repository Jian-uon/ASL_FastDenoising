# 多尺度窗口交叉融合 (Multi-scale Window Cross-Fusion)

> 状态：**已实现并跑通**（2026-08-26）。代码：[`WindowCrossFusion`](../models/blocks.py) ·
> [`ASLDetailDecoder`](../models/blocks.py) · [`ASLT1Denoiser`](../models/asl_t1_model.py) ·
> [runner CLI](../runners/asl_t1_guided_runner_dmvae_n2n.py) · 属性测试
> [`tests/test_window_fusion.py`](../tests/test_window_fusion.py)（12 项，全绿）。
>
> 取代 [wavelet_bai_design.md](wavelet_bai_design.md)（BAI 注入方案，2026-08-26 放弃，见 §6）。

---

## 1. 一句话

CMF0/CMF1 已经在 16²/32² 上做全局的组织引导注意力；本模块把**同一件事**接到解码器的
64²/128²，但在**特征自身的分辨率**上做，用 `ws×ws` 窗口而不是池化来控制开销。
`Q=ASL, K=T1, V=ASL 且不投影` ⇒ 融合输出是 ASL 值的**凸组合**，T1 决定平均谁，注入不了内容。

## 2. 为什么是窗口，不是池化

把每层的 token 池化到固定 1024（=32²）会把两个新层**降级**到 32² 的引导分辨率——那正是
CMF1 已有的带宽。四层挤在同一格，等于同一份粗解剖分组用了四次。

| 层 | 特征分辨率 | 池化方案的引导带宽 | 窗口方案 | 原图感受野 |
|---|---|---|---|---|
| 16² | 16² | 16² | 全局（CMF0，不动） | 全脑 |
| 32² | 32² | 32² | 全局（CMF1，不动） | 全脑 |
| 64² | 64² | **32²**（降级 2×） | **64²** | 8×8 窗 = 16 px |
| 128² | 128² | **32²**（降级 4×） | **128²** | 8×8 窗 = 8 px |

数据是原生 96×112 pad 到 128²（[dataio 用 pad-or-crop 不用 resize](../dataio/data_classes.py)），
所以 1 px = 1 个原生体素，皮层灰质带只有 1–2 px 宽。**32² 的引导分辨率分辨不出 GM/WM 边界**，
而边界正是最需要分组、也最容易平均错的地方。

同算力实测对比（RTX 5070 Ti，B=8，T=6，fwd+bwd，交错基准取中位）：

| 配置 | 时间 | 峰值显存 | 参数 | 引导分辨率 |
|---|---|---|---|---|
| baseline (CMF0+CMF1) | 41.9 ms | 1324 MiB | 3.516M | — |
| 池化-全局 @64,128 | ~58 ms | 2079 MiB | +26.0K | 32² |
| **窗口 ws=8 @64,128** | **61.6 ms (+47%)** | **2073 MiB** | **+17.2K** | **逐像素** |
| 只加 128² | 59.0 ms (+41%) | 1993 MiB | +4.0K | 逐像素 |

成本几乎全在 128² 层（单独 +41%），64² 层只贵 6 个百分点。墙钟影响远小于此：训练是数据加载
主导（~3 min/step vs ~20 ms/batch × 627），实测量级 **+7%**。

## 3. 模块设计

```
输入  x [B,C,H,W]  ASL 解码特征（up + skip fuse 之后）
      t [B,C_t1,H,W]  T1 编码器同尺度 skip
      seg 可选，仅 +seg 臂

  k_src = proj_t1(t)              1×1 conv（k_source='asl' 时 k_src = x，且不建 proj_t1）
  x, k  ← pad-based shift（不用 roll ⇒ 没有环绕，不需要 SwinIR 的 attention mask）
  xw, kw = window_partition(·, ws)
  Q = W_q(xw)   K = W_k(kw)   V = xw          ← V 不投影，也没有输出投影
  A = softmax( QKᵀ/√d + 相对位置偏置 [+ τ·cos(seg_i,seg_j)] [+ padding 掩码 −inf] )
  x' = x + reverse( g·(A·V − xw) )            g = sigmoid(a)
```

| 参数（L2, C=64） | 数量 | | 参数（L3, C=32） | 数量 |
|---|---|---|---|---|
| `proj_t1` 1×1 | 4,096 | | `proj_t1` | 1,024 |
| `wq` / `wk` | 4,096 ×2 | | `wq` / `wk` | 1,024 ×2 |
| `rel_bias` (15²,4) | 900 | | 同 | 900 |
| `gate_logit`, `tau` | 2 | | 同 | 2 |
| **小计** | **13,190** | | **小计** | **3,974** |

合计 **+17,164（3.516M → 3.533M，+0.49%）**。

### 三个设计要点

**① V 不投影、没有输出投影 —— 这不是省事，是性质的前提。**
A 行随机 + V 恒等 ⇒ `x' = ((1−g)I + gA)·x` 仍是行随机矩阵作用在 x 上 ⇒ 每个输出值都是**同一通道
在该窗口内 ASL 值的凸组合**，因此有极值原理：`min_窗口(x) ≤ x' ≤ max_窗口(x)`。加一个 `W_v` 或
`W_o` 这条就没了。测试 `test_maximum_principle` / `test_no_value_or_output_projection` 锁住它。

两个推论：
- **T1 侧不需要脑掩膜**（与注入式设计相反）：脑外的 T1 只能在窗口内误导权重，放不进内容。
- **这个模块只会平均，不会锐化**。风险画像是**过平滑**而不是幻觉 ⇒ 主查 lapvar / EFC，不是泄漏。

**② gate 必须 sigmoid 参数化，不能 clamp 在 0。**
`∂x'/∂θ = g·∂Δ/∂θ`，g=0 时注意力参数梯度**恒为零**；clamp 在边界梯度也是 0，模块会永久死掉。
实测：clamp 版 gate_init=0 → `grad|wq| = 0.000e+00`；sigmoid 版 a=−3 → `1.9e-4`，g=0.047，
特征扰动 2.35%。逐位恒等的 baseline 由 **`window_fusion_levels=0` 完全不构造模块**提供。

**③ 移位用 padding 而不是 `torch.roll`。**
没有环绕，就不需要 SwinIR 的 shift attention mask；padding 位置由 key 掩码置 −inf 排除。
最细一层（128²）移 `ws//2`（窗口接缝会直接出现在输出里），粗一层不移。

## 4. 接口

```bash
--window_fusion_levels {0,1,2}   # 从最细层往下数：0=关（不构造），1=仅128²，2=64²+128²
--window_size 8
--window_heads 4
--window_gate_init -3.0
--window_k_source {t1,asl}       # asl = 自注意力对照臂，解码器保持 T1-free
```

进 `denoiser_kwargs` ⇒ 自动写进 ckpt 的 `arch` ⇒ eval 脚本 `ASLT1Denoiser(**arch)` 无需改。
**旧 ckpt 没有 `window_*` 键 ⇒ 默认 0 ⇒ 严格加载不受影响**（`test_arch_roundtrip_defaults_to_off`）。

每 epoch 记录到 TensorBoard `window/wf{层}_{gate,entropy,delta}`：
- `gate` = σ(a)，模型自己要了多少引导
- `entropy` = 注意力熵；**贴着 ln(窗口 token 数) 说明分组没学起来，模块还是个 box blur**
- `delta` = ‖x'−x‖/‖x‖

## 5. 实验矩阵

| Arm | 配置 | 回答什么 |
|---|---|---|
| A0 | `--window_fusion_levels 0` | **免费**，逐位等于改动前的模型 |
| **A1** | `2 --window_k_source t1` | 主结果 |
| A2 | 2 层但池化-全局（需另实现） | 引导分辨率是否重要 |
| **A3** | `2 --window_k_source asl` | **必做**：增益来自解剖，还是仅仅多了非局部平均？ |
| A4 | `--window_fusion_levels 1` | 层数消融，也最便宜 |

判据：uMSE 改善 ≥3% **且** lapvar 相对 baseline 跌幅 <10%。**A1 − A3 就是解剖引导的净效应**——
同一份代码、同一个参数量，只换 K 的来源，比 T1-zeroed / T1-permuted 对照干净得多。

数据先验（10 受试者 80 层实测，两组不相交 k 帧各自平均的逐带相关性 → SNR = r/(1−r)）：

| k 帧 | 全带 | LL(<32²) | 中频 | 高频(>64) |
|---|---|---|---|---|
| 2 | 0.23 | 0.46 | 0.21 | **0.11** |
| 6 | 0.74 | 1.57 | 0.66 | **0.38** |

在加速工作点（2 帧）**高频只有约 10% 的方差是真信号**。所以细尺度上 ASL skip 本身就是噪声，
解码器缺的不是内容而是"该平均谁"的规则；也说明用 ASL 自身算细尺度分组权重（A3）先天吃亏——
但 patch/特征级相似度能平均掉部分噪声，所以仍然是个经验问题，必须实测。

## 6. 已放弃的备选

- **BAI / Wavelet-BAI（加性注入）**：2026-08-26 放弃。理由是任务需要的是方差下降而不是引入内容，
  而注入式设计要付出"必须脑掩膜、失去极值原理、泄漏必须按 α 重测"的代价。原设计与其实测常数
  保留在 [wavelet_bai_design.md](wavelet_bai_design.md)（其中 BLG 的否决证据仍然有效）。
- **Band-Limited Guidance**：更早已被证据否决（detail 子带只占泄漏 12.8%，且与错配 T1 边缘的
  相关性 +0.032 低于 null +0.098）。
- **K 混入 ASL 证据（T1+ASL 联合 key）**：与 CIG-VSS 线的 RKMR 重合，CLAUDE.md §8 要求隔离。
- **BLUE-Attn**（帧间方差进 attention logits）：repo 内 2026-06-21 已标记 inert-to-harmful 并移除。

## 7. 冒烟验证（2026-08-26）

21 受试者子集（`sub-C*`），2 epoch，三个臂全部跑通训练 + 验证 + ckpt：

```
levels=2 k=t1 : umse 0.580 -> 0.532   gate 0.0475/0.0478  entropy 4.149->4.122 / 3.995
levels=2 k=asl: umse 0.626 -> 0.587   gate 0.0476/0.0478  entropy 4.154 / 3.993
levels=0      : umse 0.656 -> 0.517   （无 window 探针，符合预期）
```
（2 epoch、5 个验证层，**这些 umse 之间不可比**，只用于确认管路通畅。）ckpt 里 12 个 window
参数同时出现在 `model` 和 `ema` 中；`gate_logit` 已开始移动（−3.000 / −2.991），确认梯度在流。
