# SURE — Stein's Unbiased Risk Estimator

> 本文档解释 SURE 的数学原理及在本项目 ASL N2N 训练中的应用，并记录 v25→v27 修复的 N 倍 div 估计 bug。配图见 `docs/figures/`。

---

## 1. 设定：为什么需要 SURE

经典去噪问题：观测到含噪图像

$$
y = x + n, \quad n \sim \mathcal{N}(0, \sigma^2 I)
$$

其中 $x$ 是想恢复的真实信号，$n$ 是高斯噪声。我们训练一个去噪函数 $f_\theta$，希望

$$
\mathrm{MSE} = \mathbb{E}\Vert f(y) - x\Vert^2
$$

**最小**。问题是：**我们没有 $x$**。如果有 $x$ 就直接监督学习了。

**N2N 的解决方案**：用第二份独立观测 $y'$ 当 target

$$
\mathbb{E}\Vert f(y) - y'\Vert^2 = \mathbb{E}\Vert f(y) - x\Vert^2 + \sigma^2
$$

（常数偏移，对优化方向无影响），但需要**两份独立观测**。

**SURE 的解决方案**：**只用 $y$ 自己**也能无偏估计 MSE。

![SURE vs N2N](figures/sure_vs_n2n.png)

---

## 2. SURE 定理 (Stein 1981)

**定理**：对任意几乎处处可微的 $f: \mathbb{R}^N \to \mathbb{R}^N$，若 $y = x + n$，$n \sim \mathcal{N}(0, \sigma^2 I)$，则

$$
\boxed{\;\mathbb{E}\Vert f(y) - x\Vert^2 = \mathbb{E}\Big[\,\Vert f(y) - y\Vert^2 \;-\; N\sigma^2 \;+\; 2\sigma^2 \cdot \mathrm{div}(f)(y)\,\Big]\;}
$$

其中

$$
\mathrm{div}(f)(y) = \sum_{i=1}^N \frac{\partial f_i(y)}{\partial y_i} = \mathrm{tr}(\nabla f(y))
$$

是雅可比矩阵的迹（divergence）。

**右边完全不需要 $x$**，只需 $y$ 和噪声方差 $\sigma^2$。

---

## 3. 三个项的物理意义

把 RHS 重新组织：

| 项 | 含义 | 直觉 |
|---|---|---|
| $\Vert f(y) - y\Vert^2$ | **数据保真**（Residual Sum of Squares） | 输出和输入差多少 |
| $-N\sigma^2$ | 噪声能量校正（常数） | 输入自身就含 $N\sigma^2$ 的噪声能量，要从 RSS 里扣掉 |
| $+2\sigma^2 \mathrm{div}(f)$ | **复杂度惩罚** | $f$ 对输入越敏感，散度越大，惩罚越大 |

**关键洞察**：单看数据保真，identity $f(y) = y$ 就能让 RSS = 0，但显然没去噪。`div(f)` 项正好惩罚这种"输出抄输入"的退化解——identity 的 $\mathrm{div}(f) = N$ 最大，惩罚最严。

数学上 $2\sigma^2\mathrm{div}(f) - N\sigma^2$ 等于**自由度（degrees of freedom）的两倍偏差校正**——这正是 AIC、$C_p$ 统计量的核心思想，Stein 把它从参数空间推广到了任意函数。

### 几何直觉：bias-variance 平面

![Bias-variance tradeoff](figures/sure_bias_variance.png)

横轴是 $\mathrm{div}(f)$（denoiser 对输入的敏感度），纵轴是 loss。三种典型解：

- **左端 (div ≈ 0)**：过度平滑（如 Gaussian blur），RSS 很大，但复杂度项很小 → 总 SURE 高
- **右端 (div ≈ N)**：identity 解，RSS = 0，但复杂度项最大 → 总 SURE 高
- **中间最优点**：SURE 曲线（蓝）逼近真实 MSE（绿，不可观测）的最低点

---

## 4. 证明速写（关键 trick：Stein 引理）

**Stein 引理**：若 $z \sim \mathcal{N}(0, 1)$ 且 $g$ 充分光滑，则

$$
\mathbb{E}[z \cdot g(z)] = \mathbb{E}[g'(z)]
$$

（一维分部积分 + 高斯密度的特殊性 $\phi'(z) = -z\phi(z)$）。

展开 MSE：

$$
\Vert f(y) - x\Vert^2 = \Vert f(y) - y\Vert^2 + 2(f(y)-y)^\top n + \Vert n\Vert^2
$$

对 $n$ 取期望：

- 第三项给 $N\sigma^2$
- 第二项用 Stein 引理对每个分量：

$$
\mathbb{E}[n_i \cdot (f_i(y)-y_i)] = \sigma^2 \,\mathbb{E}\!\left[\frac{\partial (f_i - y_i)}{\partial y_i}\right] = \sigma^2 \mathbb{E}\!\left[\frac{\partial f_i}{\partial y_i} - 1\right]
$$

求和得 $2\sigma^2(\mathrm{div}(f) - N)$，合并：

$$
\mathbb{E}\Vert f-x\Vert^2 = \mathbb{E}\Vert f-y\Vert^2 + 2\sigma^2 \mathrm{div}(f) - 2N\sigma^2 + N\sigma^2 = \mathbb{E}\Vert f-y\Vert^2 - N\sigma^2 + 2\sigma^2\mathrm{div}(f) \quad\square
$$

---

## 5. Monte-Carlo SURE：让散度可计算

$\mathrm{div}(f) = \mathrm{tr}(\nabla f)$ 对深度网络要算 $N$ 次反向传播（$N=128\times128=16384$ 个像素），不可行。

**Monte-Carlo SURE (Ramani et al. 2008)**：用随机投影估迹。

设 $z \sim \mathcal{N}(0, I)$，则

$$
\mathbb{E}[z^\top \nabla f(y)\, z] = \mathrm{tr}(\nabla f(y)) = \mathrm{div}(f)
$$

（迹 = $\mathbb{E}[z^\top A z]$ for $z \sim \mathcal{N}(0,I)$，矩阵 trace 估计经典结果，又名 Hutchinson estimator）

用有限差分逼近 $\nabla f(y)\,z \approx \frac{f(y+\epsilon z) - f(y)}{\epsilon}$：

$$
\boxed{\;\mathrm{div}(f)(y) \approx \frac{1}{\epsilon}\, z^\top \big(f(y + \epsilon z) - f(y)\big), \quad z \sim \mathcal{N}(0, I)\;}
$$

**只需一次额外前向**。$\epsilon$ 取小（典型 $10^{-3}$）平衡偏差（太大 → 非线性误差）和方差（太小 → 数值噪声放大）。

![MC perturbation](figures/sure_mc_perturbation.png)

---

## 6. ASL 场景的特殊处

异质噪声：每个像素的 $\sigma^2_i$ 可能不同。我们从**帧间方差**估计：

$$
\hat\sigma^2_i = \frac{1}{N_{\text{frames}}} \cdot \mathrm{Var}_t[\,\text{setA}_{t,i}\,]
$$

（$N$ 帧的均值，方差降为 $\sigma^2_{\text{frame}}/N$）。

异质噪声下 SURE 形式为

$$
\mathrm{SURE}(f) = \Vert f(y)-y\Vert^2_{\text{brain}} - \sum_i \sigma_i^2 + 2\sum_i \sigma_i^2 \frac{\partial f_i}{\partial y_i}
$$

brain mask 是为了不让脑外像素稀释信号——脑外 $x \approx 0$，去不去噪都一样。

---

## 7. v25→v26 的 N 倍 div 估计 bug

我们的模型不直接吃 $y$，而是吃帧序列 $\{y^{(t)}\}_{t=1}^N$，内部 SetTransformer 聚合成 $y = \mathrm{mean}(y^{(t)})$。

### 旧实现（buggy）

在每个帧上独立加噪 $y^{(t)} \mapsto y^{(t)} + \epsilon z^{(t)}$，$z^{(t)} \sim \mathcal{N}(0,I)$。聚合后：

$$
y' = y + \epsilon \cdot \underbrace{\tfrac{1}{N}\sum_t z^{(t)}}_{\text{协方差} = I/N\,!}
$$

探针向量也用 $\bar z = \mathrm{mean}(z^{(t)})$，但它的协方差是 $I/N$，**不是 SURE 公式假设的 $I$**。

代入 MC 公式：

$$
\bar z^\top \frac{f(y') - f(y)}{\epsilon} \approx \bar z^\top \nabla f \cdot \bar z \;\Longrightarrow\; \mathbb{E}[\cdot] = \mathrm{tr}(\nabla f \cdot \mathrm{Cov}(\bar z)) = \mathrm{div}(f) / N
$$

**散度被低估 $N=6$ 倍** → 复杂度惩罚被低估 → 模型学到过度敏感的解，但训练时显得"loss 在降"，最后崩。

### 修复（v27）

在 $y$-空间直接采 $z \sim \mathcal{N}(0,I)$，再**广播**到所有帧——`set_a + ε·z_broadcast`。这样

$$
\mathrm{mean}(\text{set\_a} + \epsilon z_{\text{broadcast}}) = y + \epsilon z
$$

而探针 $z$ 协方差是 $I$，无偏。代码见 [`losses/asl_n2n_loss.py:mc_sure_term`](../losses/asl_n2n_loss.py)。

### 为什么 v25 没崩、v26 崩了？

v25 用了模糊的 stage1 T1 预训练（buggy 旧 best.pth），ASL decoder 输出本身就钝化，div(f) 真值小，被低估 N 倍后噪声主导也不会发散。v26 改用锐版 T1 后，模型变敏感，div(f) 真值大，N 倍低估让 SURE 几乎完全失效，训练 step 70 后开始发散。

---

## 8. SURE 与 N2N 的关系

二者都是绕开 clean $x$ 的策略，但路径不同：

|  | N2N | SURE |
|---|---|---|
| 需要 | 两份独立观测 $y, y'$ | 一份观测 + $\sigma^2$ |
| 损失 | $\Vert f(y) - y'\Vert^2$ | $\Vert f(y)-y\Vert^2 + 2\sigma^2\mathrm{div}(f) -$ const |
| 正则化项 | 隐式（target 的噪声防止过拟合） | 显式（$\mathrm{div}(f)$ 项） |
| 偏差 | $+\sigma^2$（常数，无所谓） | 0（理论上无偏） |
| 方差 | 低 | 高（MC 估计） |

我们组合使用：**N2N 提供主信号**（`mean(set_b)` 当 target），**SURE 当额外正则项**抑制对 noisy target 的过拟合——12-NEX 平均自身仍含残余噪声，N2N 训得越久越拟合这部分残噪，SURE 的 div 项把这种"过敏感"的解推开。

---

## 9. 实务要点

| 问题 | 解决 |
|---|---|
| MC 估计方差大 | 多采样平均；或用 Hutchinson 的 Rademacher 分布（方差更小） |
| div 项不稳定 | warmup（前 60 step $w_{\text{sure}}=0$）+ clamp；v27_final 已加 |
| 后期 over-smooth collapse | post-warmup anneal：step 200→500，$w_{\text{sure}}$ 从 0.02 退到 0.005 |
| $\sigma^2$ 估计偏 | 用 wavelet-domain MAD 或帧间无偏方差；本项目用后者 |
| 高 SNR 区域 div 主导 | mask 到脑内（脑外 $\sigma\to 0$ 不贡献信号也不贡献 SURE） |

---

## 9b. 为什么 SURE 在 noisy-GT 自监督设定下尤其重要

经典监督设定下，目标 $y_{\text{ref}}$ 是 clean signal，PSNR-vs-ref 直接对应 MSE-to-clean，model selection 没有 ambiguity。但在 N2N / 自监督设定下：

$$
y_{\text{ref}} = x_{\text{clean}} + \eta_{\text{resid}} + \alpha_{\text{artefact}}
$$

PSNR-vs-$y_{\text{ref}}$ 同时奖励两件事：
1. ✅ 学 $x_{\text{clean}}$（真去噪）
2. ❌ 学 $\eta_{\text{resid}}$ 和 $\alpha_{\text{artefact}}$（mimic noise/artefact）

**SURE 显式拒绝路径 (2)**——`div(f)` 项惩罚 $f$ 对 input 的过度敏感，这正是 noise mimicry 的特征。所以 SURE 不只是另一个正则项，它是**针对自监督 PSNR 偏差的对症解药**。

实测表现：
| 方法 | PSNR | 真去噪 vs noise mimicry |
|---|---|---|
| v18 (无 SURE) | 26.26 | 部分通过 mimic noise 推高 PSNR |
| v27 (有 SURE) | 24.39 | 拒绝 mimic, PSNR 低但 GM-WM contrast 准 |

这是为什么我们最终用**复合 best criterion**（PSNR + HFEN + LapVar）而不是单 PSNR ——前者是 mimicry-aware 的。

---

## 10. 论文 method section 写法

> "We augment the N2N loss with a Monte-Carlo SURE term (Stein 1981; Ramani et al. 2008) as a principled, GT-free regulariser. Unlike implicit regularisation from N2N's noisy target, SURE provides an explicit unbiased estimate of the bias-variance tradeoff via the divergence of the denoiser, $\mathrm{tr}(\nabla f)$. We estimate $\sigma^2$ per pixel from frame-to-frame variance and use a single $\mathcal{N}(0,I)$ probe in the input space (broadcast to the frame dimension) to avoid the $1/N$ scaling artefact of frame-level perturbation."

---

## 参考文献

- **Stein, C. M.** (1981). "Estimation of the mean of a multivariate normal distribution." *Annals of Statistics* 9(6), 1135–1151. — SURE 定理原文
- **Ramani, S., Blu, T., Unser, M.** (2008). "Monte-Carlo SURE: A Black-Box Optimization of Regularization Parameters for General Denoising Algorithms." *IEEE TIP* 17(9). — MC-SURE
- **Hutchinson, M. F.** (1990). "A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines." *Communications in Statistics*. — Trace estimator
- **Lehtinen, J. et al.** (2018). "Noise2Noise: Learning Image Restoration without Clean Data." *ICML*. — N2N 原文
- **Soltanayev, S., Chun, S. Y.** (2018). "Training deep learning based denoisers without ground truth data." *NeurIPS*. — DL + SURE 早期工作
