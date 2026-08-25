# ASL_dmvae 主方法架构（v37）

> Last updated: 2026-05-14
> 模型：`ASLT1Denoiser`（[models/asl_t1_model.py](../models/asl_t1_model.py)）
> 总参数 **~4.08M**（stage-2 trainable ~2.20M，T1 branch frozen ~1.88M）

## 0. 摘要

T1-guided 7T ASL 去噪。Self-supervised Noise2Noise + J-invariant masking。**Structural bound on T1 value injection**：T1 仅作为 cross-attention 的 Key + tissue-PV similarity bias 进入 ASL 路径，**通过 Value projection 直接注入 T1 像素灰度的路径被代数排除**；但 T1 仍可经 attention routing / tissue PV / mask 等路径**间接**影响输出，因此完全 T1-invariance 必须由 mismatched-T1 等经验测试补充验证。

**v37 创新（投稿 contribution）**：
- **SVFW**（Spatial-Varying Frame Weighting）：**BLUE-inspired learned** per-pixel 帧权重（替代标量帧权重）。log_var head 仅看 `frame − temporal_mean` deviation，never 看 raw signal → **基于绝对强度的压制**被代数排除（病灶 + 配准抖动 / motion 共现的边界情形仍可能被压制，见 §5.1）。**注**：log_var 是 learned head，没有真值监督，因此是 *BLUE-inspired* 而不是严格 BLUE。
- **V=ASL 设计的 Multi-scale Cross-Attention**（继承 v34 / v36，v37 调权后稳定）：T1 decides where；V 始终留在 ASL feature space（结构上限：**无 T1 value injection**，但 attention pattern 仍由 T1 影响）。Tissue-PV similarity bias（**sigmoid-based, multi-label PV vector**，不是 softmax categorical）让 cross-attention 偏向 GM/WM/CSF 同 PV class 之间 pooling。
- **Loss 重平衡**（**SURE 已删除，2026-05-14**）：w_n2n 0.5 / w_grad 0.5 / w_ssim 0.1 / w_contrast 0.3 + J-invariant masking。SURE 旧版本走的是 Stein iid Gaussian 假设，与 ASL 噪声实际模型不符；权重设为 0，相关函数保留于 [losses/asl_n2n_loss.py:mc_sure_term](../losses/asl_n2n_loss.py) 仅作 ablation。

---

## 1. 目标与约束

**任务**：从 7T 单 PLD ASL 12-NEX 中**任意子集**（n=2/3/4/6 帧）的 label-control 差值帧 ΔM，恢复**高 SNR PWI**（perfusion weighted image）。**不**做 Buxton 动力学定量（不是 CBF）。

**核心设计约束**：
1. **No clean GT**：12-NEX union 仍含运动 / failed acquisition / 重建伪影 → 任何 GT-based loss 都被噪声污染。
2. **Self-supervised**：N2N + J-invariant masking。
3. **No direct T1-value path**（**结构上限**）：T1 不通过任何 Value projection 直接为 PWI 贡献像素灰度——但 T1 仍可经 attention pattern、tissue-PV bias、brain mask、`loss_contrast` 边界加权间接影响输出。**完全 T1-invariance 不构成结构性保证**，需 mismatched-T1 + atlas-mask 等经验测试补足。

---

## 2. 顶层数据流

```
                 ┌───── set_a [B,T,1,H,W]               ┌── t1 [B,1,H,W]
                 │  (label-control 差值帧子集，T 可变)  │
                 ▼                                      ▼
        ┌──────────────────────┐              ┌──────────────────┐
        │ SVFW                 │              │ ConvEncoder2D    │
        │ (per-pixel BLUE)     │              │  (T1 branch)     │ ← FROZEN (stage-2)
        │  log_var head sees   │              │  - 4 levels      │
        │  ONLY deviation =    │              │  - 32→64→128→256 │
        │  frame − temp_mean   │              │                  │
        │  weights[B,T,1,H,W]  │              │                  │
        └──────────┬───────────┘              └──┬───────────┬───┘
                   │                             │           │
                   ▼ agg [B,1,H,W]               ▼           ▼ t1_skips (4 levels)
                                       t1_feat_map [B,256,16,16]
        ┌──────────────────────┐                  │           │
        │ ConvEncoder2D        │                  │           │
        │   (ASL branch)       │                  │           ▼
        │   - 4 levels         │                  │   ┌──────────────────┐
        │   - 32→64→128→256    │                  │   │ ConvDecoder      │ ← FROZEN
        └──────────┬───────────┘                  │   │   WithSkips2D    │
                   │                              │   │   (T1 branch)    │
                   ▼ asl_feat_map [B,256,16,16]   │   │   out_ch=4       │
                  + asl_skips                     │   │  (GM/WM/CSF/BG)  │
                                                  │   └────────┬─────────┘
                                                  │            │
                                                  │            ▼ t1_seg [B,4,128,128]
            ┌─────────────────────────────────────┘            │
            ▼                                                  │
   ┌──────────────────────┐                                    │
   │ T1GuidedCoarseHead   │  ◄────── tissue_seg ──────────────┤
   │  (coarse fusion)     │  (downsampled per-scale, similarity bias)
   │  L0 (16×16):         │
   │    adapter + CMF0    │  Q=ASL, K=T1, V=ASL  (gate~0.09)
   │  L1 (32×32):         │
   │    Up + asl_skip_32  │
   │    + CMF1            │  Q=ASL, K=t1_skip_32, V=ASL  (gate~0.31)
   │  → feat_l1  (pure-ASL; T1 path ends here)
   └──────────┬───────────┘
              │ feat_l1 [B,128,32,32]
              ▼
   ┌──────────────────────┐
   │ ASLDetailDecoder     │  L2/L3 pure ASL U-Net
   │  (32→64→128)         │  no T1 input by construction
   │  - 2 up blocks       │  ASL skips at each level
   │  - head 3×3          │
   └──────────┬───────────┘
              │
              ▼ asl_recon [B,1,H,W]   ── PWI 输出

N2N target: mean(set_b) (direct unweighted mean of held-out frames)
J-invariance regulariser: blind-spot mask on input set_a (p=0.10)
```

---

## 3. 各组件细节

### 3.1 SVFW — Spatial-Varying Frame Weighting

[models/blocks.py:SpatialVaryingFrameWeighting](../models/blocks.py)

**输入**：`frames [B, T, 1, H, W]`，T 可变，可带 length / mask 表示有效帧。
**输出**：
- `agg [B, 1, H, W]`：per-pixel BLUE 加权聚合
- `weights [B, T, H, W]`：per-pixel per-frame 权重（squeeze C 维）

**前向**：
```python
v = valid.view(B,T,1,1,1)
mu_pixel = (frames * v).sum(1) / v.sum(1).clamp_min(1)  # [B,1,H,W] temp mean
dev = frames - mu_pixel.unsqueeze(1)                     # [B,T,1,H,W] deviation
feat = dev_enc(dev_flat)                                 # 2 conv-GN-SiLU
log_var = log_var_head(feat).reshape(B, T, 1, H, W)
log_var = log_var.masked_fill(invalid, 1e6)
weights = softmax(-log_var, dim=1)                       # BLUE per-pixel
agg = (frames * weights).sum(dim=1)                      # [B,1,H,W]
```

**Safe-by-design — 严格表述**：
- log_var head 永远只接 `dev = frame − mean_T(frames)`，**从不接 raw signal**。
- 因此模型**不能基于绝对像素值大小**作为 down-weight 依据；信号"反常但帧间一致"的病灶不会被压制。
- **边界条件（要承认）**：如果病灶 ROI 同时伴随局部 motion / 配准抖动 / 部分 NEX 失败，该 ROI 的 dev 仍会变大，SVFW 仍可能 down-weight。换句话说 "SVFW 不基于绝对强度压制信号"是严格结构 claim；"SVFW 不压制病灶"是带条件的，**需 lesion-ROI 实验验证**（lesion 强度保留 + SVFW weight inside/outside lesion）。
- per-pixel 让 motion artifact（同一帧内某些像素 OK、另一些 corrupted）能被局部处理，标量帧权重做不到。

**"BLUE" 表述谨慎**：log_var head 是 *learned*，没有针对真实噪声方差的 supervised target，所以严格说这是 **BLUE-inspired learned frame weighting**，不是数学上的严格 BLUE。要把 paper 故事走到 "principled BLUE estimator"，需补 calibration 实验（synthetic bad-frame 下 predicted log_var 与真实 corrupted region 对齐度，以及按 predicted variance 分 bin 后的 residual scaling）。

参数量：~5k（vs SetTransformer ~86k，小 17×）。

### 3.2 ConvEncoder2D

[models/blocks.py:ConvEncoder2D](../models/blocks.py)

ASL 和 T1 共享**架构**，**独立权重**。

base_ch=32, depth=4：
```
ResidualBlock(in=1 → 32, s=1)        [B,32,128,128]   ← skip[0]
DownBlock(32 → 64, s=2)              [B,64,64,64]      ← skip[1]
DownBlock(64 → 128, s=2)             [B,128,32,32]     ← skip[2]
DownBlock(128 → 256, s=2)            [B,256,16,16]     ← skip[3] = bottleneck
```

每个 ResidualBlock = `Conv-GN-SiLU → Conv-GN → +skip → SiLU`，groups=8 GroupNorm。

参数量：1.22M（ASL）+ 1.22M（T1，frozen）。

### 3.3 CrossModalFusion (Tissue-Gated, V=ASL)

[models/blocks.py:CrossModalFusion](../models/blocks.py)

CrossModalFusion 是 v37 反幻觉论述的核心组件。一次调用接收一对同空间分辨率的 ASL / T1 feature map，输出与 ASL 同形状的 fused feature map。**T1 只通过 attention pattern（K 来源 + PV similarity bias）影响输出，永远不进 V projection** —— 这是"无 T1 value injection"结构上限的代数来源。

#### 3.3.1 构造参数

| 参数 | default | 含义 |
|------|--------:|------|
| `in_ch` | 256 (bottleneck) / 128 (32×32 scale) | 输入 channel 维度；Q / K / V 共用 |
| `n_heads` | 4 | multi-head 数；要求 `in_ch % n_heads == 0` |
| `dropout` | 0.2 | attention dropout（仅训练）|
| `gate_init` | 0.3 | 残差 gate 初始值（pre-clamp）|
| `gate_max` | 1.0 | gate 上限（v34 前曾用 0.5，发现 gate→0 collapse 改为 1.0）|
| `tau_init` | 2.0 | tissue-PV similarity bias 系数初值，learnable |
| `t1_as_key` | True | 是否 K=T1；False 时 K=Q (退化为 pure ASL self-attn) |
| `use_blue_attn` | False | 是否启用噪声方差 bias（v42 N4 实验路径）|

构造时只创建：
- `nn.MultiheadAttention(in_ch, n_heads, dropout, batch_first=True)`
- `nn.LayerNorm(in_ch)`
- 标量参数 `gate`, `tau`，（可选）`blue_lambda`

#### 3.3.2 Forward 详细伪代码

```python
def forward(asl_map [B,C,H,W], t1_map [B,C,H,W],
            tissue_seg [B,4,H',W']=None, noise_var [B,1,H',W']=None):
    B, C, H, W = asl_map.shape
    N = H * W

    # 1) 把 [B,C,H,W] 拍平到序列形式 [B, N, C]，准备做 attention
    Q = asl_map.flatten(2).transpose(1, 2)              # [B, N, C]
    K = t1_map.flatten(2).transpose(1, 2) if t1_as_key  # [B, N, C]
        else Q                                          # else self-attn
    V = Q                                               # ⚠ V = ASL  (anti-T1-value-injection)

    # 2) 构造可选 attention bias / mask
    attn_mask = None

    # 2a) tissue-PV similarity bias：让同 PV 位置之间更易 attend
    if tissue_seg is not None:
        if (H', W') != (H, W):                           # downsample seg to match attention scale
            tissue_seg = F.adaptive_avg_pool2d(tissue_seg, (H, W))
        seg_p = sigmoid(tissue_seg).flatten(2).transpose(1, 2)  # [B, N, 4]  (sigmoid → PV vector, multi-label)
        seg_n = F.normalize(seg_p, dim=-1, eps=1e-6)            # unit PV vector per position
        sim   = seg_n @ seg_n.transpose(-1, -2)                 # [B, N, N], in [0, 1]
        bias  = tau * sim                                       # learnable scale
        attn_mask = bias                                        # additive logits bias

    # 2b) (optional, v42 path) BLUE noise bias — see §3.3.4
    if use_blue_attn and noise_var is not None:
        ...                                                     # adds -λ·(σ²_i + σ²_j); v37 default off

    # 3) MultiheadAttention 要求 attn_mask 形状是 [B*nh, N, N]
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1).expand(B, n_heads, N, N) \
                              .reshape(B * n_heads, N, N)        # critical reshape

    # 4) scaled dot-product attention
    #    softmax((QK^T)/√d + attn_mask) · V
    attn_out, _ = self.attn(Q, K, V, attn_mask=attn_mask)        # [B, N, C]

    # 5) Residual + LayerNorm
    gate  = clamp(self.gate, 0, gate_max)                        # learnable scalar
    fused = LayerNorm(Q + gate * attn_out)                       # [B, N, C]

    return fused.transpose(1, 2).reshape(B, C, H, W)             # back to [B, C, H, W]
```

每一步对应实现见 [models/blocks.py:578-629](../models/blocks.py).

#### 3.3.3 三个机制的角色拆解

**机制 1 — Q / K / V 非对称设计（结构反幻觉）**

| 项 | 来源 | 角色 |
|----|------|------|
| **Q** | ASL feature | "我（这个 ASL 位置）想要什么" |
| **K** | T1 feature | "T1 上各位置对应的解剖签名" |
| **V** | **ASL feature**（注意：**不**是 T1）| "我能借鉴的内容（始终是 ASL）" |

标准 cross-attn 是 K=V 同源（要么 K=V=T1，要么 K=V=ASL）。本设计**故意拆开**：T1 决定从哪些位置取信号（attention pattern），但取到的**值**始终从 ASL 拿。代数上，输出 attn_out 是 ASL feature 的线性组合（系数由 T1 决定）—— **`∂asl_recon/∂t1` 只能经 attention coefficients 流，永远不经 V projection 流**。

类比：learnable Non-Local Means，T1 是相似度 oracle，pooling target 是 ASL。

**机制 2 — Tissue-PV similarity bias（PV-aware grouping）**

T1 decoder 输出 `tissue_seg ∈ [B, 4, H, W]` 是 GM/WM/CSF/BG 四类 PV logits（stage-1 用 per-channel sigmoid + L1 训练，**不是 softmax categorical**——4 个 channel 不强制互斥，反映 partial volume）。

- 经 sigmoid 得 PV 向量 `seg_p ∈ [0,1]^4`，每个位置一个 4 维向量
- L2 归一 → unit vector
- N×N pairwise cosine similarity 反映"两个位置的 PV 组成有多接近"，在 `[0, 1]` 范围
- 乘 learnable scalar `tau`（init 2.0，训练后稳定 ≈ 2.0），加到 attention logits 上 → 同 PV 类位置 logit 更高、softmax 后权重更大

效果：**ASL feature 更倾向在 PV 相近位置间聚合**（同 GM 之间互相借鉴、同 WM 之间互相借鉴），抑制 cross-tissue 平滑（GM-WM 边界模糊是典型 over-smoothing 失败模式）。

**严格表述**："tissue partial-volume similarity bias"，**不是** "tissue-class similarity bias"。前者承认 PV 向量是 multi-label 多维概率；后者暗示 softmax 互斥类别，与 stage-1 训练形式不一致。

**机制 3 — Per-scale residual gate（scale-specific 调节）**

```
fused = LayerNorm(Q + clamp(gate, 0, gate_max) · attn_out)
                  └──────────┬──────────┘
                       不带 attention pattern 的纯 ASL feature
```

- `gate` 是每个 fusion module 独立的 learnable 标量，初始 0.3，clamp 到 `[0, 1.0]`
- gate → 0 退化为纯 ASL 路径（fused ≈ LN(Q)），attention 不参与
- gate = 1 是 attention residual full contribution

**gate 值不能解释成"T1 注入比例"**：
- `attn_out` 仍是 ASL feature 的线性组合（V=ASL），gate 是这个 attention-mixed ASL feature 的标量权重
- LayerNorm 再做 mean/var 归一，破坏 gate 作为"比例"的可解释性
- 经验上 16×16 gate ≈ 0.09，32×32 gate ≈ 0.31，但这只是 "attention residual 在该尺度的强度"，**不**等于 "9% / 31% T1 影响"。要量化 T1 真实影响必须用 mismatched-T1 sensitivity 测试（match vs mismatch 的 L1 差异）。

#### 3.3.4 attn_mask 形状坑（已确认实现正确）

`nn.MultiheadAttention(batch_first=True)` 的 `attn_mask` 必须是：
- `[N, N]` — broadcast 到所有 (batch, head)，或
- `[B * n_heads, N, N]` — 每个 (batch, head) 独立 mask

但 v37 想要的是 **per-batch** mask（tissue_seg 不同 → bias 不同），所以必须先 expand 到 `[B, n_heads, N, N]` 再 reshape 到 `[B*n_heads, N, N]`：

```python
attn_mask = attn_mask.unsqueeze(1).expand(B, n_heads, N, N).reshape(B * n_heads, N, N)
```

如果省略这一步、直接传 `[B, N, N]`，PyTorch 会 silent-broadcast 或报错。当前实现在 [blocks.py:624](../models/blocks.py#L624) 是正确的（2026-05-15 verified）。

#### 3.3.5 参数量

每个 CrossModalFusion 模块：
- `MultiheadAttention(in_ch=256, n_heads=4)`：4 × (256² 投影矩阵) + output projection = **256² × 4 ≈ 263k**
- `LayerNorm(256)`：512
- gate + tau (+ blue_lambda)：2–3 标量

≈ **264k / 模块**。v37 共 2 个 fusion 模块（bottleneck + 32×32 scale），合计 ~528k。

### 3.4 Multi-scale 应用：两个尺度

cross-modal fusion 在 v37 中应用于**两个尺度**，**refactor 2026-05-15 之后这两个 CMF 都搬进 `T1GuidedCoarseHead` 模块内**（之前 16×16 在 `ASLT1Denoiser.cross_fusion`、32×32 在 `asl_decoder.t1_fusions[0]`，分成两处不直观）：

| 尺度 | 位置 | 模块 | tokens N | gate(step≈25 实测) |
|---|---|---|---:|---:|
| **A. 16×16**（粗）| `T1GuidedCoarseHead` L0 | `t1_head.cmf0` | 256 | ~0.09 |
| **B. 32×32**（中）| `T1GuidedCoarseHead` L1（Up + asl_skip merge 之后）| `t1_head.cmf1` | 1024 | ~0.31 |
| 64×64 / 128×128 | `ASLDetailDecoder` | **不加**（结构上无 T1 入参）| 4096 / 16384 | — |

> **gate 值的正确解释**：这两个 gate 数值是 attention residual branch 的标量缩放系数（在 LayerNorm 之前），**不是**"T1 信息占输出的百分比"。LayerNorm 会重新标准化 feature 尺度，破坏 gate 作为"比例"的可解释性。要量化 T1 对输出的真实影响，应该用 mismatched-T1 sensitivity 测试或 attention entropy / sparsity 指标。

**为何不加 64×64+**：
- **内存**：attention QK^T 是 (4096×4096)，显存爆；
- **设计**：高频层最危险——T1 sulcal 细节直接进 PWI 等于 hallucination 风险（即便 V=ASL 也有 indirect 风险）；
- **实测**：32×32 gate 收敛到 0.31（vs 16×16 收敛到 0.09），attention residual 在 32×32 尺度贡献更显著，但**不能直接断言"T1 信息主要来自 32×32"** —— 这只是 gate 数值，真正的 T1 影响要从 mismatched-T1 实验测。

`t1_attn_max_tokens=1024` 这个 CLI 参数控制阈值，超过的 decoder 层走 Identity。

### 3.5 ConvDecoderWithSkips2D / ASLDetailDecoder / T1GuidedCoarseHead

**Refactor 2026-05-15**：ASL 路径的 decoder 被沿 T1 影响边界拆成两部分：
- `T1GuidedCoarseHead`（L0 16×16 + L1 32×32）—— 拥有 adapter + CMF₀ + Up + asl_skip merge + CMF₁，所有 T1 cross-attention 都在这里发生；输出 `feat_l1 [B,128,32,32]` 是纯 ASL-derived。
- `ASLDetailDecoder`（L2 64×64 + L3 128×128）—— 只剩 2 个 up-level + head，**forward 签名里没有 T1 入参**，结构上禁止 T1 灰度在细节层注入。

T1 decoder 不受影响，继续用原 `ConvDecoderWithSkips2D`（intra-modal skips only）。

| 用途 | 模块 | out_ch | 输出 | T1 cross-fusion 路径 |
|---|---|---:|---|---|
| **T1 decoder** | `ConvDecoderWithSkips2D` | 4 | GM/WM/CSF/BG PV logits | 不启用（intra-modal skips only）|
| **ASL coarse head** | `T1GuidedCoarseHead` | — (中间 feature) | `feat_l1 [B,128,32,32]` | **CMF₀ + CMF₁**（V=ASL，tissue bias）|
| **ASL detail decoder** | `ASLDetailDecoder` | 1 | PWI | **不可能**（forward 签名无 T1）|

下面 §3.5.1–§3.5.6 介绍 `ConvDecoderWithSkips2D` 通用结构（T1 decoder 仍走该路径；ASL coarse head 的 Up + skip merge 复用同样的算子，只是被拆出来组装在 `T1GuidedCoarseHead` 里）。

| 用途（旧框架视角） | out_ch | 输出 | use_t1_cross_fusion | use_film |
|---|---:|---|---|---|
| **T1 decoder** | 4 | GM/WM/CSF/BG PV logits | False（intra-modal skips only）| False |
| **ASL decoder（旧）** | 1 | PWI | True（multi-scale T1 fusion 注入）| optional |

#### 3.5.1 通道阶梯（depth=4, base_ch=32, out_hw=128）

```
channels        = [32, 64, 128, 256]
bottleneck_ch   = 256, bottleneck_hw = 16
decoder spatial = 16 → 32 → 64 → 128   (反向 encoder)
decoder ch      = 256 → 128 → 64 → 32 → out_ch
levels (i=0,1,2): 32×32, 64×64, 128×128
```

#### 3.5.2 构造模块清单

1. **`adapter` (1×1 Conv)**：把外部传入的 bottleneck feature `[B, in_ch, 16, 16]` 投影到 `[B, 256, 16, 16]`；若 `in_ch == 256` 则 `Identity()`。ASL decoder 接 cross-fusion 输出（256），所以 adapter 退化；T1 decoder 接 encoder bottleneck（256），也退化。
2. **`ups: ModuleList[UpBlock]`**（3 个）：每个 `UpBlock(in_ch, out_ch)` = `F.interpolate(scale=2, bilinear) + ResidualBlock(in_ch→out_ch)`。
3. **`fusions: ModuleList[Sequential]`**（3 个）：每个是 `Conv1×1(2·next_ch → next_ch) + GroupNorm + SiLU`，处理 own-modality skip concatenation。
4. **`t1_fusions: ModuleList[CrossModalFusion or Identity]`**（仅 ASL，3 个）：每 level 配一个；若 `level_hw² > t1_attn_max_tokens` 则用 `nn.Identity()` 跳过（默认阈值 1024 → 32×32 启用，64×64 / 128×128 跳过）。
5. **`films: ModuleList[TissueFiLM]`**（v37 default off）：per-channel γ/β modulation by global tissue PV，3 个 levels 各一。
6. **`head` (Conv 3×3, padding=1)**：最后一层 `[B, 32, 128, 128] → [B, out_ch, 128, 128]`，padding 保证 spatial 不变。
7. **`final_activation`**（可选）：T1 decoder 不接（输出 logits，sigmoid 在 loss 里做）；ASL decoder 也不接（输出值域不限制）。
8. **`skip_drop = Dropout2d(p=skip_dropout=0.3)`**：所有 levels 共用一个 Dropout2d 实例，apply 到 own-modality skip。

#### 3.5.3 Forward 详细伪代码

```python
def forward(x [B, in_ch, 16, 16],
            skips: List[[B, c_l, h_l, w_l]] = [feat0, feat1, feat2, feat3],
            t1_skips: Optional[List[Tensor]] = None,
            t1_seg:  Optional[[B, 4, 128, 128]] = None) -> [B, out_ch, 128, 128]:

    x = adapter(x)                                            # [B, 256, 16, 16]

    n = len(skips) - 1                                        # n = 3
    for i, (up, fuse) in enumerate(zip(ups, fusions)):        # i = 0, 1, 2
        # ── (a) Upsample ──────────────────────────────────────
        x = up(x)                                             # 2× bilinear + ResidualBlock
                                                              # i=0: [B,128,32,32]
                                                              # i=1: [B, 64,64,64]
                                                              # i=2: [B, 32,128,128]

        # ── (b) Own-modality skip fusion ─────────────────────
        skip = skip_drop(skips[n - 1 - i])                    # Dropout2d on intra-modal skip
        x = fuse(torch.cat([x, skip], dim=1))                 # Conv1×1 + GN + SiLU

        # ── (c) Optional FiLM (v37 default off) ──────────────
        if use_film and t1_seg is not None:
            x = films[i](x, t1_seg)                           # γ/β by global PV

        # ── (d) Optional ASL-only T1 cross-fusion ────────────
        if use_t1_cross_fusion and t1_skips is not None:
            fusion = t1_fusions[i]
            if not isinstance(fusion, nn.Identity):           # i=0 (32×32, N=1024) ✓
                t1_skip = t1_skips[n - 1 - i]                 # same scale T1 skip
                x = fusion(x, t1_skip, tissue_seg=t1_seg)
            # else: i=1 (64×64, N=4096), i=2 (128×128) — skipped

    x = head(x)                                                # 3×3 Conv → out_ch
    if x.shape[-1] != out_hw:                                  # fallback safety
        x = F.interpolate(x, size=(out_hw, out_hw), mode='bilinear')
    if final_activation:
        x = final_activation(x)
    return x
```

#### 3.5.4 关键设计细节

**(a) Skip 来源严格 intra-modal**
ASL decoder 的 `skips` 来自 ASL encoder（feat0..feat3），T1 decoder 的 `skips` 来自 T1 encoder。**两个 modality 的 skip 永不交叉**——这是结构反幻觉的另一道屏障：T1 的高频细节（沟回 / 灰白质边界）不会通过 skip 路径绕过 cross-fusion 的 V=ASL 约束直接进入 ASL decoder。
对应代码：[asl_t1_model.py](../models/asl_t1_model.py) 把 `t1_skips` 只传给 `asl_decoder` 作为 cross-fusion 的 Key 来源，**不**传给 `asl_decoder.fusions`。

**(b) Skip dropout (p=0.3) 的位置和动机**
Dropout2d 应用在 own-modality skip 上、**在 concat 之前**：
```python
skip = skip_drop(skips[n - 1 - i])   # Dropout2d on intra-modal skip
x = fuse(torch.cat([x, skip], dim=1))
```
- 不在 upsampled feature 上 drop，因为那是经过 cross-fusion 学习过的高阶 representation
- skip 携带最高频的 frame noise（encoder 浅层未经平滑），drop 抑制 decoder 直接复制输入 noise → 鼓励 head 用 deep feature 做最终重建
- 推理阶段 dropout 自动 inactive（除非 MC dropout），所以不损失推理质量

**(c) T1 cross-fusion 注入时机：fuse 之后**
顺序是 `up → fuse(own-modality skip) → cross_fusion(T1 skip)`。这样 attention 的 Q 是"已经融合了同模态高频信息"的 ASL feature，而不是裸的 upsampled feature。
若反过来（cross-fusion 在 fuse 之前），attention 会基于低分辨率 upsampled feature 做 pairwise 相似度，损失精度。

**(d) 多尺度 cross-fusion 由 `t1_attn_max_tokens` gate**
构造时按 `level_hw² <= t1_attn_max_tokens` 决定该 level 用 `CrossModalFusion` 还是 `nn.Identity`：
- 32×32 (N=1024) → enabled ✓
- 64×64 (N=4096) → Identity（attention O(N²) 显存爆）
- 128×128 (N=16384) → Identity

这意味着 T1 cross-fusion **物理上不可能**注入 64×64 以上的高频层，硬性截断 T1 sulcal detail injection 风险（即便代数上 V=ASL 已禁止 T1 value path，物理截断是双重保险）。

**(e) Head 是 Conv 3×3 不是 1×1**
最后一层是 `Conv2d(32, out_ch, kernel_size=3, padding=1)` —— 3×3 给输出层一个轻量空间整合（vs 1×1 纯通道投影）。这对 ASL decoder 的 PWI 输出有轻度空间 smoothing 效果，但比加额外 ResidualBlock 便宜得多。

**(f) `final_activation` 在 v37 留空**
- T1 decoder 输出 logits，sigmoid 在 `pv_l1_loss_4cls` 内部做（避免 logit 被 clamp 到 [0,1] 失去训练梯度幅度）
- ASL decoder 输出 PWI 值域不强制（normalized [0,1] 是 input 阶段的事；output 自由对应 N2N target 的实际值域）

#### 3.5.5 参数量分解（v37 default ASL decoder）

| 部件 | 参数量 |
|------|------:|
| adapter (Identity, 因 in_ch=256) | 0 |
| 3 × UpBlock (ResidualBlock 256→128, 128→64, 64→32) | ~470k |
| 3 × fusion (Conv1×1 2C→C + GN + SiLU) | ~50k |
| t1_fusions[0] (32×32, CrossModalFusion in_ch=128) | ~66k |
| t1_fusions[1], t1_fusions[2] (Identity) | 0 |
| head (Conv 3×3, 32→1) | 0.3k |
| **ASL decoder 总计** | **~735k**（trainable in stage 2）|
| **T1 decoder 总计**（无 cross-fusion，out_ch=4） | **~669k**（**frozen** in stage 2）|

#### 3.5.6 与一般 U-Net decoder 的差异（summary）

| 维度 | 一般 U-Net decoder | v37 ConvDecoderWithSkips2D |
|------|--------------------|----------------------------|
| Upsample | bilinear / transpose conv | bilinear + ResidualBlock |
| Skip concat | concat + Conv | Dropout2d → concat → Conv1×1 + GN + SiLU |
| Cross-modal | n/a | optional CrossModalFusion **after own-modality fuse**, scale-gated by `t1_attn_max_tokens` |
| Modality 隔离 | skip 可跨模态 | **严格 intra-modal skip** （anti-T1-hallucination） |
| Head | Conv 1×1 | **Conv 3×3 padding=1** |
| Final activation | sigmoid / tanh / identity | identity（值域自由）|

### 3.6 T1 Branch（Stage-1 单独训练，Stage-2 冻结）

[models/t1_branch.py](../models/t1_branch.py)

**Stage 1**：用 `runners/train_t1.py` 单独训 `T1Branch = t1_encoder + t1_decoder`：
- 输入：t1
- 输出：4-class PV logits（GM/WM/CSF/BG）
- Loss：`pv_l1_loss_4cls`（per-channel sigmoid + L1）
- BG target = `1 − clamp(GM+WM+CSF, 0, 1)`，4 类构成全图覆盖分布

**Stage 2**：在 ASLT1Denoiser 中 `--init_t1_from + --freeze_t1`：
- t1_encoder + t1_decoder 加载 stage-1 权重，冻结
- 每 forward 重算 t1_seg（拿到 per-sample tissue prior）
- 仅用作 CrossModalFusion 的 K（feature）和 attention bias（seg）来源，**不进 ASL 输出**

当前使用：`stage1_t1_padcrop/checkpoints/best.pth`（pad-or-crop pipeline，无 wavelet）。

---

## 4. 训练 Loss（v37 round-3，**SURE 已移除 2026-05-14**）

总 loss = N2N 主项 + 3 个结构辅助项 + J-invariant masking regulariser：

| 项 | weight | 公式 | 物理作用 |
|---|---:|---|---|
| `loss_n2n` | **0.5** | `L1(asl_recon, mean(set_b)) * mask` | 主 N2N 重建（noisy target 不让单项主导）|
| `loss_grad` | **0.5** | `L1(∇asl_recon, ∇mean(set_b)) * mask` | 边缘对齐 |
| `loss_contrast` | **0.3** | `L1 weighted by (1 + 4·√(gm·wm)) * mask` | T1 PV 边界加权 N2N L1（**注意**：见下警告）|
| `loss_ssim` | **0.1** | `1 − SSIM(asl_recon*mask, target*mask)` | 局部结构 / 对比度 |

已禁用（保留 flag 仅 ablation 用）：`loss_tv`、`loss_cos`、`loss_t1`、`loss_seg`、**`loss_sure`**。

> **⚠️ `loss_contrast` 的 anti-hallucination 风险（待 ablation）**：即便结构上 V=ASL 防止 T1-value injection，loss 在 GM/WM 边界加权 = 训练信号告诉模型"在 T1 边界附近更努力拟合"，这可能让 PWI 输出贴 T1 解剖边界、削弱"T1 只作相似度先验"的论述。投稿前必须报告 `--w_contrast=0` ablation + mismatched-T1 下的 T1 sulcal imprint 检测；如果发现 T1 imprint 显著，应降权或移除。

> **❌ `loss_sure` 移除原因（2026-05-14）**：Monte-Carlo SURE 假设噪声为 iid Gaussian 且 σ² 可估计，但 ASL 差值图实际噪声不严格满足这些前提：
> 1. **σ² 来源不明**：v37 实现按 mean(set_a) 的 within-subject 全局方差估计，但每 voxel 噪声很可能各异；
> 2. **噪声相关性**：ASL 重建后噪声有空间相关性（cardiac pulsation, partial volume effects），违反 iid；
> 3. **divergence 项**：MC Stein divergence 的有限差分对模型结构敏感，加入 SVFW 后未做独立性检查。
>
> 严谨做法：把 SURE 当 *regularizer*（不是 exact unbiased risk estimator），并跑 `N2N-only / +SURE / +Jinv / +SURE+Jinv` 4 路 ablation 证明它的贡献。当前删除是为简化 loss 配方；相关工具函数保留于 [losses/asl_n2n_loss.py:mc_sure_term](../losses/asl_n2n_loss.py) 供后续 ablation 调用。

**J-invariant masking regulariser** [Krull NeurIPS 2019 / Batson ICML 2019]：训练时以概率 `--jinv_p`（默认 0.10）随机 mask ASL 输入像素（替换为邻居均值），loss 仅在 masked 位置计算。Blind-spot 强迫模型从邻居推断，无法靠记住 noise 减小 loss。

**Best ckpt criterion**：`upsnr_cyc` — 纯 self-supervised composite，无 GT 也无 biased reference：
$$\text{score} = \text{uPSNR} - \alpha \cdot \text{cyc}, \quad \alpha = 30$$

**uPSNR 算法**（Marcos-Morales ICML 2023, arXiv:2210.05553；实现 [utils/metrics.py:upsnr_components](../utils/metrics.py)；pooled 累积在 [runners/asl_t1_guided_runner_dmvae_n2n.py:_pool_upsnr](../runners/asl_t1_guided_runner_dmvae_n2n.py)）：

1. 将 hold-out 的 `set_b` 沿 T 维做**三路不交集划分**，每段长度 `n = ⌊T_b/3⌋`，分别取均值得到三张图：
   `a = mean(set_b[0:n])`，`b = mean(set_b[n:2n])`，`c = mean(set_b[2n:3n])`。
2. 在脑 mask 内计算两项 per-pixel 量：
   - 残差平方 `sq_err = (a − f(set_a))²`
   - 方差校正 `var_corr = ½ (b − c)²`
3. **Pooled 累积**（关键，避免 log-of-mean ≠ mean-of-log 的 batch-平均偏差）：跨 val set 累加 `Σ sq_err`、`Σ var_corr`、`Σ N_pixels`，最后单次取对数：
   $$\text{uMSE} = \frac{\sum \text{sq\_err} - \sum \text{var\_corr}}{\sum N}, \quad \text{uPSNR} = 10\log_{10}\!\frac{M^2}{\text{uMSE}}, \; M=1.0$$
4. `var_corr` 项无偏抵消 `(a−f)²` 中由 set_b 噪声引入的方差，故 `E[uMSE] = MSE(f, x_clean)`，是 PSNR-to-clean 的渐近一致估计。

**cyc 算法**（subset_consistency；实现 [runners/asl_t1_guided_runner_dmvae_n2n.py](../runners/asl_t1_guided_runner_dmvae_n2n.py) val 循环）：

1. 取 `k = ⌊T_a/2⌋`（要求 `T_a ≥ 4`，否则跳过）。
2. 把 `set_a` 沿 T 维切两段不交集 halves：`set_a1 = set_a[:k]`，`set_a2 = set_a[k:2k]`。
3. **同一 EMA 权重**分别 forward：`pred1 = f(set_a1)`、`pred2 = f(set_a2)`。
4. 脑 mask 内取 L1：`cyc = (|pred1 − pred2| · mask).sum() / mask.sum()`。
5. val set 上对 cyc 做样本平均（不需要 pooled，因为是线性平均）。

直觉：模型若过拟合输入帧的具体噪声实例，两个不交集 halves 的 noise 实现不同 → pred1/pred2 差异大 → cyc 大；模型只学到 clean signal 时 cyc → 0。

**α=30 — dataset-specific calibration（不是 derived constant）**：基于本数据集观测 `σ_uPSNR ≈ 0.15 dB`、`σ_cyc ≈ 0.005`，取 `α = σ_uPSNR / σ_cyc ≈ 30`，使 1 个 σ_cyc 的稳定性变化 ≈ 1 个 σ_uPSNR 的 fidelity 变化。这是工程标定，**不是 first-principle 推导值**。换数据集（不同 NEX、不同 PLD、不同 noise level）必须重新标定。投稿前建议补 `α ∈ {10, 20, 30, 50}` 的灵敏度表，证明 ckpt 选择对 α 不敏感（即在合理 α 范围内 best ckpt 是同一个）。

`--best_min_step 200` gating 防 init 噪声选偏。

为什么不用 psnr_ref / psnr_b 当 criterion：
- **psnr_ref** vs 12-NEX union 含 set_a 的噪声，**奖励 noise mimicry**——本数据集上 psnr_ref 与 uPSNR 选出的 best ckpt 完全反向。
- **psnr_b** 用 mean(set_b) 当 reference 已是 hold-out unbiased（Lehtinen 2018），但单个 reference 估计仍有较高方差；**uPSNR 用 3-way split + variance debiasing 进一步减方差**，理论更强。

辅助指标 `psnr_ref` / `psnr_b` 作为 paper 副表与文献对齐。

**SWA**（torch.optim.swa_utils.AveragedModel）：从 `--swa_start_step 200` 累积参数移动平均，结束保存到 `checkpoints/swa.pth`。

**Input masking**：训练 / 验证 forward 之前 `set_a *= (t1>0.05); set_b *= (t1>0.05)`——模型从来看不到脑外噪声，自然学到脑外为 0；bad-frame 注入先做再 mask。

---

## 5. 推理（infer）

[runners/infer_pwi.py](../runners/infer_pwi.py)：NIfTI in → PWI NIfTI out。

| Flag | 用途 |
|---|---|
| `--n_frames {2,3,4,6,12}` | 少帧 sweep |
| `--mc_n N` | MC dropout：N 次随机 forward，输出 mean PWI + 一张 std confidence map（epistemic uncertainty）|
| `--mc_ckpts ckpt1 ckpt2 ...` | Deep-ensemble：与主 ckpt 一起多 ckpt 平均；典型用法 `--mc_ckpts swa.pth` |

stochastic 模式下额外保存 `*_confmap.nii.gz`（per-voxel std）。

---

## 6. 模型尺寸总结

| 子模块 | 参数 | stage-2 状态 |
|---|---:|---|
| SVFW aggregator | ~5k | trainable |
| ASL encoder | 1.22M | trainable |
| T1 encoder | 1.22M | **frozen** |
| CrossModalFusion @ 16×16 | 264k | trainable |
| ASL decoder（含 32×32 t1_fusion） | 735k | trainable |
| T1 decoder | 669k | **frozen** |
| **Total** | **~4.08M** | trainable ~2.20M |

---

## 7. 已弃用变体

下列在 v37 试验中评估并放弃，实现保留为 `--use_*` flag 供 ablation 对照：

| 变体 | 弃用原因 | 实测 |
|---|---|---|
| **Heteroscedastic NLL head**（Kendall&Gal NeurIPS 2017，asl_decoder out_ch 1→2 = (μ, log σ²)）| Kendall&Gal NLL 假设 target 是 clean GT，但 N2N 中 `target = mean(set_b)` 本身是噪声估计 → σ² 一次性吸收所有 noise variance；`∂NLL/∂μ ∝ 1/σ²`，σ² 膨胀后 μ 梯度被压制 | best psnr_b=20.24（v36 23.11）|
| **TissueFiLM**（per-channel γ/β by global tissue PV，asl_decoder per-level）| 即便 per-channel only（不能 locally 编辑像素），val 图视觉上 recon 显出清晰 T1 沟回结构；hallucination 风险在不引入额外约束的情况下无法排除 | round-2 step 400 val 图证据 |

**未来方向**（保留 σ² uncertainty map）：β-NLL（Seitzer ICLR 2022）或解耦 μ/σ² 双头训练（μ 用 L1，σ² 单独 NLL with detach）。

---

## 8. 历史决策（简表）

| 决策 | 时间 | 动机 |
|---|---|---|
| 单向 N2N（仅 setA 入模型，setB 仅作 target） | early | 经典 N2N 形式 |
| Feature-map bottleneck（不 GAP）| early | 保留 16×16 空间信息 |
| **V=ASL（反幻觉）** | early | 防 T1 灰度污染 PWI |
| Bad-frame injection | early | 让 BLUE 权重真的有用 |
| ASL self-skip（不交叉 T1 skip）| early | 防 T1 高频结构泄漏 |
| pad-or-crop pipeline（替代 trilinear）| middle | 修栅格 alias bug |
| 4-class seg（删 T1 recon）| middle | 释放 T1 decoder 容量给 PV |
| Multi-scale Tissue-Gated Cross-Attn | middle | 单点 bottleneck 中 gate→0 collapse 后修复 |
| TV → J-invariance | late | TV 是 hand-crafted 平滑先验，J-invariance 是 principled self-supervised 正则 |
| psnr_b 替换 composite_v2 | late | composite 是 heuristic；psnr_b 是 Lehtinen 2018 给出的无偏 hold-out PSNR |
| 常数 SURE schedule | late | ramp 期 N2N 已让模型过拟合 noise |
| **SVFW 替代 SetTransformer** | **v37** | per-frame 标量无法描述空间局部 corruption |
| **Loss 重平衡（w_n2n 0.5 / w_sure 0.5 / w_ssim 0.1）** | **v37 round-3** | N2N 高权重过平滑；SURE 旧权重 effective 占比 ≈ 1/70 主项 |
| **SURE 整体移除（`w_sure: 0`）** | **2026-05-14** | iid Gaussian + 可估计 σ² 假设与 ASL 噪声模型不匹配；divergence 项对 SVFW 加入后未做独立性检查；保留函数但默认 off |
| **"zero hallucination" claim 弱化** | **2026-05-14** | 改为 "no direct T1-value path"，承认 attention routing / contrast loss / mask 等间接 T1 影响路径；mismatched-T1 经验测试作为补充 |
| **"tissue-class similarity" → "tissue-PV similarity"** | **2026-05-14** | T1 decoder 用 sigmoid + L1 训练 → 4-channel 不互斥，是 PV 概率向量，不是 softmax categorical class |
| **Refactor: T1GuidedCoarseHead + ASLDetailDecoder** | **2026-05-15** | 把 ASL 路径的 bottleneck CMF + L1 CMF 统一收进 `T1GuidedCoarseHead`；剩余 64×64 / 128×128 两级搬进 `ASLDetailDecoder`（forward 无 T1 入参）。bit-identical refactor — 仅模块边界更清晰，anti-hallucination 从"参数惯例"升级为"类型签名硬约束"；旧 ckpt state_dict key 不兼容（需重训）。 |
| **lr / w_n2n / jinv_p 收紧（v42 retry）** | **2026-05-15** | lr 3e-4 → 1e-4，w_n2n 0.5 → 0.3，jinv_p 0.10 → 0.20。原因：v42 step 50 已达 uPSNR=22.22 后单调跌（典型噪声模仿），需要慢收敛 + 减弱噪声目标拉力 + 加强 J-inv 正则把 best 推到 step 150-200。 |
