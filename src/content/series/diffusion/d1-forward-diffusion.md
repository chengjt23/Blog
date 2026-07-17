---
title: 前向扩散：如何系统地破坏数据
description: 从离散 Gaussian Markov chain 推导任意时刻边缘、噪声日程、信噪比与终点近似条件。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 1
slug: d1-forward-diffusion
tags:
  - diffusion
  - forward-process
  - noise-schedule
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦前向加噪过程、直接采样公式、schedule 与 terminal SNR，不讨论逆过程训练。
---
生成模型最终要做的是从简单噪声出发，得到复杂数据。但 Diffusion 的第一步偏偏反着来：从真实数据出发，主动把结构破坏掉。

这个方向看起来有些绕。既然目标是生成图像，为什么先研究怎样毁掉图像？为什么不直接训练一个从 Gaussian noise 到数据的映射？又为什么要分成成百上千步，而不是一次加入足够大的噪声？

答案不是“加噪本身很强”，而是**一个设计得当的破坏过程，可以把原本未知的生成问题改造成一系列有监督、可计算的局部逆问题**。前向过程的职责是为这些逆问题规定路径、难度和监督信号。本章只研究路径的前向一侧；下一章再利用它推导 reverse process 和训练目标。

读完本章后，我们应当能够回答：

1. forward Markov chain 到底定义了哪些随机对象；
2. 为什么 $q(x_t\mid x_0)$ 可以不经过前面 $t-1$ 步而直接采样；
3. $\beta_t,\alpha_t,\bar\alpha_t$ 分别控制什么；
4. 为什么 timestep $t$ 不是比较噪声强度的最佳坐标；
5. linear、cosine、log-SNR 和 zero-terminal-SNR 分别在解决什么问题；
6. 为什么 training noise distribution、loss weighting 和 sampling grid 不能全都叫作“noise schedule”。

***

## 1. 为什么不能一次把数据变成噪声

设数据为 $x_0\sim q_{\mathrm{data}}$。最直接的破坏方式是

$$
x_1=\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

它确实一步得到了简单分布，但也一步丢掉了 $x_0$ 的全部信息。此时若要求模型从 $x_1$ 恢复 $x_0$，由于 $x_1$ 与 $x_0$ 独立，模型面对的仍是原始任务：从随机数直接生成整个数据分布。前向过程没有为学习提供任何额外结构。

更有用的做法是把破坏拆成许多小步：

$$
x_0\longrightarrow x_1\longrightarrow\cdots\longrightarrow x_T.
$$

每一步只加入少量噪声，于是相邻状态高度相关。逆转 $x_t\to x_{t-1}$ 通常比一步完成 $\epsilon\to x_0$ 更局部。Sohl-Dickstein et al. 在 2015 年把这种 nonequilibrium forward diffusion、learned reverse process 和可计算的路径变分下界组合成现代 diffusion probabilistic model 的基本框架；DDPM 并不是第一次提出 diffusion model，而是在 2020 年找到了一套更有效的参数化和训练方式（Sohl-Dickstein et al., 2015, Secs. 1--2；Ho et al., 2020, Secs. 2--3）。

这里要避免一个常见误解：

> 小步加噪不会自动让无条件逆分布 $q(x_{t-1}\mid x_t)$ 变得已知。它真正提供的是一个已知的 forward path，以及训练时可利用真实 $x_0$ 构造的解析 posterior $q(x_{t-1}\mid x_t,x_0)$。后者在 [D2](/blog/diffusion/d2-ddpm-objective/) 中完整推导。

因此，前向过程不是生成算法本身，也不需要神经网络。它是我们人为选择的 inference/corruption process，用来定义学习问题。

***

## 2. 先分清四种对象

在写公式之前，先区分四个经常被混为一谈的对象。

### 2.1 一条随机轨迹

固定一个数据点 $x_0$，逐步抽取独立噪声，得到

$$
(x_0,x_1,\ldots,x_T).
$$

这是 forward chain 的一次 realization。换一组噪声，即使起点相同，也会得到另一条轨迹。

### 2.2 一步条件分布

$$
q(x_t\mid x_{t-1})
$$

描述已知前一状态时，下一步如何随机变化。它是 Markov chain 的 transition kernel。

### 2.3 给定原始数据的时刻边缘

$$
q(x_t\mid x_0)
$$

把中间状态 $x_1,\ldots,x_{t-1}$ 积分掉，只问“从这个 $x_0$ 出发，到时刻 $t$ 会在哪里”。DDPM 的高效训练依赖这个量可以直接采样。

### 2.4 整个数据分布在时刻 $t$ 的边缘

$$
q_t(x_t)
=\int q(x_t\mid x_0)q_{\mathrm{data}}(x_0)\,dx_0.
$$

这是所有数据点经过 forward corruption 后形成的总体分布。即使 $q(x_t\mid x_0)$ 是 Gaussian，$q_t(x_t)$ 一般仍不是单个 Gaussian；它是缩放后的数据分布与 Gaussian kernel 的卷积。

这四者的差别会贯穿整个教程。尤其不能因为每个 conditional 是 Gaussian，就断言中间的 data marginal 也是 Gaussian。

***

## 3. DDPM 的 Gaussian forward Markov chain

DDPM 采用离散时间 $t=1,\ldots,T$，定义

$$
q(x_{1:T}\mid x_0)
=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),
$$

其中

$$
\boxed{
q(x_t\mid x_{t-1})
=\mathcal N\!\left(
x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I
\right).
}
$$

$\beta_t\in(0,1)$ 是第 $t$ 步加入的 variance。为了简化后续表达，定义

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,
\qquad
\bar\alpha_0=1.
$$

于是一步采样可以写成 reparameterized form：

$$
\boxed{
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t,
\qquad
\epsilon_t\overset{\mathrm{iid}}\sim\mathcal N(0,I).
}
$$

### 3.1 为什么 signal 前面是平方根

假设某个时刻已经有 $x_{t-1}\sim\mathcal N(0,I)$。由于两个 Gaussian 独立，

$$
\operatorname{Cov}(x_t)
=\alpha_t I+\beta_t I
=(\alpha_t+\beta_t)I
=I.
$$

因此 $\mathcal N(0,I)$ 是每个 transition 的 invariant distribution。平方根保证的是**方差**按 $\alpha_t$ 与 $\beta_t$ 相加，而不是 amplitude 直接相加。

这并不意味着任意数据经过一步就变成标准 Gaussian。它只说明：标准 Gaussian 一旦到达，就会被该 kernel 保持；从复杂数据出发时，反复应用 kernel 才会逐渐忘记初始结构。

### 3.2 $\beta_t$、$\alpha_t$ 与 $\bar\alpha_t$ 的职责

| 记号                                    | 含义                                 | 作用范围                |
| ------------------------------------- | ---------------------------------- | ------------------- |
| $\beta_t$                             | 第 $t$ 步加入的噪声方差                     | one-step transition |
| $\alpha_t=1-\beta_t$                  | 第 $t$ 步保留的 signal power 比例         | one-step transition |
| $\bar\alpha_t=\prod_{s\le t}\alpha_s$ | 从 $x_0$ 到 $x_t$ 累积保留的 signal power | endpoint marginal   |
| $\sqrt{\bar\alpha_t}$                 | 原始样本的 amplitude 系数                 | $q(x_t\mid x_0)$    |
| $1-\bar\alpha_t$                      | 累积噪声方差                             | $q(x_t\mid x_0)$    |

“linear schedule”如果指 $\beta_t$ 线性，并不表示 $\bar\alpha_t$、SNR 或 log-SNR 线性。许多 schedule 争论的混乱就来自没有说明“到底是哪一个量随时间怎样变化”。

***

## 4. 核心推导：任意时刻为何可以直接采样

我们现在完整推导

$$
q(x_t\mid x_0)
=\mathcal N\!\left(
\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I
\right).
$$

这是 D1 的核心结论，也是 DDPM 训练不必真的模拟 $t$ 次 forward steps 的原因。Ho et al. 将其写在 DDPM Eq. (4)（PDF p. 2）。

### 4.1 先展开两步

第一步：

$$
x_1=\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\epsilon_1.
$$

第二步：

$$
\begin{aligned}
x_2
&=\sqrt{\alpha_2}x_1+\sqrt{\beta_2}\epsilon_2\\
&=\sqrt{\alpha_2\alpha_1}x_0
+\sqrt{\alpha_2\beta_1}\epsilon_1
+\sqrt{\beta_2}\epsilon_2.
\end{aligned}
$$

后两项是独立零均值 Gaussian，其总方差为

$$
\alpha_2\beta_1+\beta_2
=\alpha_2(1-\alpha_1)+(1-\alpha_2)
=1-\alpha_1\alpha_2.
$$

所以它们可以合并成一个标准 Gaussian $\epsilon$：

$$
x_2
=\sqrt{\alpha_1\alpha_2}x_0
+\sqrt{1-\alpha_1\alpha_2}\epsilon.
$$

两步的形式已经暗示了结论。

### 4.2 归纳证明

假设在 $t-1$ 时刻

$$
x_{t-1}
=\sqrt{\bar\alpha_{t-1}}x_0
+\sqrt{1-\bar\alpha_{t-1}}\epsilon',
$$

其中 $\epsilon'\sim\mathcal N(0,I)$。代入第 $t$ 步：

$$
\begin{aligned}
x_t
&=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t\\
&=\sqrt{\alpha_t\bar\alpha_{t-1}}x_0
+\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\epsilon'
+\sqrt{\beta_t}\epsilon_t.
\end{aligned}
$$

signal 系数满足

$$
\alpha_t\bar\alpha_{t-1}=\bar\alpha_t.
$$

两项独立噪声的总方差满足

$$
\begin{aligned}
\alpha_t(1-\bar\alpha_{t-1})+\beta_t
&=\alpha_t-\bar\alpha_t+1-\alpha_t\\
&=1-\bar\alpha_t.
\end{aligned}
$$

因此

$$
\boxed{
x_t
=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
}
$$

也即

$$
\boxed{
q(x_t\mid x_0)
=\mathcal N\!\left(
x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I
\right).
}
$$

### 4.3 从整条路径看方差为什么恰好望远镜消去

把递归完全展开：

$$
x_t
=\sqrt{\bar\alpha_t}x_0
+\sum_{s=1}^{t}
\sqrt{\beta_s\prod_{j=s+1}^{t}\alpha_j}\epsilon_s.
$$

噪声总方差系数为

$$
\sum_{s=1}^{t}
\beta_s\prod_{j=s+1}^{t}\alpha_j.
$$

利用 $\beta_s=1-\alpha_s$，第 $s$ 项可以写成

$$
(1-\alpha_s)\prod_{j=s+1}^{t}\alpha_j
=\prod_{j=s+1}^{t}\alpha_j
-\prod_{j=s}^{t}\alpha_j.
$$

求和后中间项全部抵消，只剩

$$
1-\prod_{j=1}^{t}\alpha_j
=1-\bar\alpha_t.
$$

这说明 closed form 不是近似，也不是“因为 $T$ 很大”才成立；对任意合法的离散 $\beta_{1:t}$，它都是精确恒等式。

### 4.4 任意两个时刻之间的 transition

相同推导还给出 $0\le s<t\le T$ 时

$$
\boxed{
q(x_t\mid x_s)
=\mathcal N\!\left(
\sqrt{\frac{\bar\alpha_t}{\bar\alpha_s}}x_s,
\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right)I
\right),
}
$$

前提是 $\bar\alpha_s>0$。这说明相邻一步并不特殊：只要知道累计 signal ratio，就能把多个 forward steps 合成一个 Gaussian transition。

***

## 5. 直接边缘采样不等于采样一条 Markov 轨迹

closed form 允许我们为随机训练时刻 $t$ 直接构造 $x_t$：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

但这里有一个很容易被可视化代码掩盖的区别。

若对多个时刻重复使用同一个 $\epsilon$，每个单独的 $x_t$ 都具有正确的 $q(x_t\mid x_0)$，但联合分布

$$
(x_{t_1},x_{t_2},\ldots)\mid x_0
$$

一般不是原始 Markov chain 的联合分布。它只是一个方便观察“同一个 signal 如何随 noise level 变化”的 coupling。要得到真实 forward trajectory，需要逐步采样独立 $\epsilon_t$，或者按上一节的 $q(x_t\mid x_s)$ 继续采样。

这个差异可以直接从 conditional cross-time covariance 看见。固定 $x_0$，若 $s<t$ 且两个时刻复用同一个 $\epsilon$，则

$$
\operatorname{Cov}_{\mathrm{shared}\ \epsilon}(X_s,X_t\mid x_0)
=\sqrt{(1-\bar\alpha_s)(1-\bar\alpha_t)}I.
$$

对真实 Markov path，由

$$
X_t
=\sqrt{\frac{\bar\alpha_t}{\bar\alpha_s}}X_s
+\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_s}}\eta,
\qquad \eta\perp X_s,
$$

可得

$$
\operatorname{Cov}_{\mathrm{Markov}}(X_s,X_t\mid x_0)
=\sqrt{\frac{\bar\alpha_t}{\bar\alpha_s}}
(1-\bar\alpha_s)I.
$$

两者一般不相等。这是“每个 marginal 都对，但 joint law 不对”的一个明确反例。

下面的图为了让同一批二维点可追踪，故意在各时刻复用了同一组 noise。每个面板的 marginal 正确，但整组面板不应被解释成真实 Markov trajectories。

![二维八峰分布在 cosine forward process 下的直接边缘](/images/diffusion/d1_forward_marginals.png)

图中的颜色只标记原始 mode，不是模型在 noisy state 中可见的标签。到末端，各颜色已经混合，整体接近标准 Gaussian。

***

## 6. “数据变成 Gaussian”究竟是什么意思

### 6.1 条件分布的极限

由 closed form，

$$
q(x_T\mid x_0)
=\mathcal N\!\left(
\sqrt{\bar\alpha_T}x_0,(1-\bar\alpha_T)I
\right).
$$

若 $\bar\alpha_T\to0$，则

$$
\sqrt{\bar\alpha_T}x_0\to0,
\qquad
1-\bar\alpha_T\to1,
$$

所以 conditional 趋近 $\mathcal N(0,I)$。若 $X_0$ 是几乎处处有限的随机向量，那么

$$
X_T
=\sqrt{\bar\alpha_T}X_0
+\sqrt{1-\bar\alpha_T}\epsilon
\xrightarrow{d}\epsilon
$$

可直接由 Slutsky theorem 得到。这里不需要假设原始数据是 Gaussian。

### 6.2 有限时刻的 aggregate marginal 通常不是 Gaussian

对有限 $\bar\alpha_t>0$，

$$
q_t
=\operatorname{Law}
\left(\sqrt{\bar\alpha_t}X_0+\sqrt{1-\bar\alpha_t}\epsilon\right).
$$

若 $q_{\mathrm{data}}$ 是多峰分布，$q_t$ 就是每个 mode 缩向原点并被 Gaussian 平滑后的混合分布。只有当 $X_0$ 本来就是 Gaussian，或 signal coefficient 精确为 0 时，它才必然是单个 Gaussian。

### 6.3 均值和协方差怎样变化

设数据均值和协方差分别为 $\mu_0,\Sigma_0$，并假设噪声独立，则

$$
\mathbb E[X_t]=\sqrt{\bar\alpha_t}\mu_0,
$$

$$
\operatorname{Cov}(X_t)
=\bar\alpha_t\Sigma_0+(1-\bar\alpha_t)I.
$$

因此均值被压向 0，协方差被推向 $I$。但只检查前两阶矩还不足以证明整个分布 Gaussian；多峰性、高阶矩和其他非 Gaussian 结构也必须逐渐消失。

### 6.4 分布看起来不变，也可能仍保留信息

考虑一个特殊例子：若 $X_0\sim\mathcal N(0,I)$，则所有 $X_t$ 的 marginal 都是 $\mathcal N(0,I)$。只看直方图，我们会以为 forward process 什么都没做；但 $X_t$ 与 $X_0$ 的相关性仍在下降。

此时 channel 的 nominal SNR 为

$$
\operatorname{SNR}_t
=\frac{\bar\alpha_t}{1-\bar\alpha_t},
$$

mutual information 可以精确写成

$$
I(X_0;X_t)
=\frac d2\log(1+\operatorname{SNR}_t)
=-\frac d2\log(1-\bar\alpha_t).
$$

当 $\bar\alpha_t\to0$ 时，它趋近 0。这个例子说明：forward diffusion 破坏的是 $X_t$ 对起点的可恢复信息，而不只是让某张 marginal density “看起来更圆”。

对一般数据，Markov chain

$$
X_0\to X_s\to X_t,
\qquad s<t,
$$

由 data processing inequality 给出

$$
I(X_0;X_t)\le I(X_0;X_s).
$$

这为“信息逐步被破坏”提供了不依赖图像直觉的表达。

***

## 7. 从 timestep 转向 SNR 与 log-SNR

### 7.1 $t$ 只是索引，不是噪声强度

两个模型都可以使用 $T=1000$，但在 $t=500$ 保留完全不同的 signal。直接比较 timestep 没有物理意义；应比较 $a(t)$、$b(t)$ 或由它们构成的 noise coordinate。

为了跨离散和连续论文统一记号，先写

$$
q(x_t\mid x_0)
=\mathcal N(a(t)x_0,b(t)^2I).
$$

定义 nominal signal-to-noise ratio：

$$
\boxed{
\operatorname{SNR}(t)=\frac{a(t)^2}{b(t)^2}.
}
$$

在 DDPM 的 variance-preserving (VP) 形式中，

$$
a(t)^2=\bar\alpha_t,\qquad b(t)^2=1-\bar\alpha_t,
$$

所以

$$
\boxed{
\operatorname{SNR}_t
=\frac{\bar\alpha_t}{1-\bar\alpha_t}.
}
$$

严格地说，这是假设数据每个方向具有单位尺度时的 nominal SNR。若数据在某个特征方向上的 variance 为 $\lambda$，该方向的实际 power ratio 是

$$
\frac{\bar\alpha_t\lambda}{1-\bar\alpha_t}.
$$

这也是实现通常先把图像缩放到固定范围、理论分析引入 $\sigma_{\mathrm{data}}$ 的原因。

### 7.2 negative log-SNR

VDM 使用

$$
\boxed{
\gamma(t)=-\log\operatorname{SNR}(t)
=\log\frac{1-\bar\alpha_t}{\bar\alpha_t}.
}
$$

forward 过程中 SNR 下降，故 $\gamma(t)$ 上升。对 VP parameterization，反解为

$$
\bar\alpha(t)=\operatorname{sigmoid}(-\gamma(t)),
\qquad
1-\bar\alpha(t)=\operatorname{sigmoid}(\gamma(t)).
$$

相较于 $\beta_t$，log-SNR 有三个优点：

1. 它直接标记 noisy channel 的 signal/noise 比；
2. VP、VE 和其他 Gaussian corruption 可以在同一坐标中比较；
3. time reparameterization 的作用更清楚：改变 $\gamma(t)$ 的速度，就是改变有限 timesteps 在 noise-level axis 上的密度。

若连续时间 $t\sim U[0,1]$，并令 $\lambda=\gamma(t)$，那么由换元公式

$$
p_\lambda(\lambda)
=p_t(t)\left|\frac{dt}{d\lambda}\right|
=\frac{1}{|\gamma'(t)|}.
$$

所以即使代码“uniformly sample time”，模型看到的 log-SNR 也未必均匀。schedule 已经隐式选择了 training noise-level proposal。

### 7.3 跨论文记号不要机械对齐

| 本章统一记号              | DDPM                                  | VDM                    | 含义                        |
| ------------------- | ------------------------------------- | ---------------------- | ------------------------- |
| $a(t)$              | $\sqrt{\bar\alpha_t}$                 | $\alpha_t$             | signal amplitude          |
| $b(t)^2$            | $1-\bar\alpha_t$                      | $\sigma_t^2$           | cumulative noise variance |
| one-step $\alpha_t$ | $1-\beta_t$                           | 不对应                    | DDPM 单步 signal power      |
| $\gamma(t)$         | $\log((1-\bar\alpha_t)/\bar\alpha_t)$ | $-\log\mathrm{SNR}(t)$ | negative log-SNR          |

VDM 的 $\alpha_t$ 对应 DDPM 的 $\sqrt{\bar\alpha_t}$，不是 DDPM 的 $1-\beta_t$。如果不先做这张映射表，很容易写出量纲不一致的式子。

***

## 8. Noise schedule 的问题-解决方案演进

一个 schedule 至少要接受四项检查：

1. **局部性**：每一步是否足够平滑，使相邻逆问题不过分困难；
2. **覆盖**：有限 steps 是否覆盖了有意义的 noise range；
3. **终点**：$q(x_T\mid x_0)$ 是否与 generation prior 足够一致；
4. **资源分配**：训练和采样是否在重要 noise levels 上投入了足够计算。

不存在脱离数据、目标、parameterization 和 sampler 的“唯一最优 schedule”。下面几种设计分别回应了不同问题。

### 8.1 2015：schedule 已经是模型设计的一部分

Sohl-Dickstein et al. 并没有把 forward rate 当成无关紧要的常数。论文 Section 2.4.1 把 $\beta_{2:T}$ 视为可由 likelihood lower bound 学习的参数，并固定第一步 $\beta_1$ 以减少过拟合。

但论文叙述与同期公开代码需要分开记录：2015 年固定 commit 的 <code>model.py</code> 构造了 baseline schedule 和 perturbation coefficients，却保留了“把这些 coefficients 加入可学习参数”的 TODO；该代码快照实际使用固定 baseline。教程因此只能说“论文框架提出/描述学习 schedule”，不能用这份代码证明所有公开实验确实联合学习了 schedule（Sohl-Dickstein et al., 2015, Sec. 2.4.1；official code <code>model.py:81-108</code>）。

这个早期落差也揭示了一个长期问题：schedule 既是理论路径的一部分，又深受具体实现约束。

### 8.2 DDPM：linear $\beta_t$

原始 DDPM 图像实验使用 $T=1000$，令

$$
\beta_t
=\beta_{\min}
+\frac{t-1}{T-1}(\beta_{\max}-\beta_{\min}),
$$

其中

$$
\beta_{\min}=10^{-4},\qquad \beta_{\max}=0.02.
$$

该选择简单、稳定，并让 terminal SNR 很小。它的局限是：$\beta_t$ 线性并不意味着感知信息或 log-SNR 均匀下降。Improved DDPM 观察到，在 32 和 64 分辨率图像上，linear schedule 的后段 latent 已接近纯噪声，若仍为这些时刻分配大量离散 steps，计算利用并不理想（Nichol & Dhariwal, 2021, Sec. 3.2, Figs. 3--5）。

### 8.3 Improved DDPM：cosine cumulative signal

Improved DDPM 不直接规定 $\beta_t$ 的形状，而是先规定 cumulative signal power：

$$
\bar\alpha_t=\frac{f(t)}{f(0)},
$$

$$
f(t)=\cos^2\!\left(
\frac{t/T+s}{1+s}\frac\pi2
\right),
\qquad s=0.008,
$$

再反算

$$
\beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}.
$$

实际实现把 $\beta_t$ 截断到不超过 0.999，以避免末端奇异。该 schedule 让 $\bar\alpha_t$ 在中段下降得更接近线性，并在两端变化较缓。它在论文的图像实验中优于当时的 linear baseline，但 $\cos^2$ 和 $s=0.008$ 仍是有动机的工程设计，不是由普适最优性定理唯一推出（Nichol & Dhariwal, 2021, Eq. (17), p. 4）。

### 8.4 VDM：把 $\gamma(t)$ 作为可学习单调函数

VDM 直接参数化

$$
\gamma_\eta(t)=-\log\operatorname{SNR}(t)
$$

为单调网络。这样 schedule 不再只是手工超参数，还能用于联合优化 endpoint 或降低 continuous-time loss estimator 的方差（Kingma et al., 2021, Secs. 3.2, 5.3 and App. I）。

VDM 的一个重要结论是：在 continuous time、SNR 严格单调、两端 SNR 相同，并对 denoiser 做相应重缩放时，不同 $(a(t),b(t))$ specification 可以定义等价的 continuous objective 和生成模型，latent 只差平凡尺度变换（Kingma et al., 2021, Sec. 5.1）。

这句话经常被过度简化成“schedule 不重要”。正确边界是：

- continuous objective 的 SNR integration range 可以只由 endpoints 决定；
- uniform-time Monte Carlo estimator 的方差仍依赖 schedule shape；
- finite-step training/sampling 的离散误差仍依赖网格；
- 固定网络容量和优化过程不会因为理论重参数化就自动等价。

### 8.5 读图：同一个名字下其实有不同曲线

![不同 forward schedules 的 one-step variance、累计 signal 和 SNR](/images/diffusion/d1_schedule_comparison.png)

左图使用对数纵轴；红线的最后一步达到 $\beta_T=1$，用于实现 exact zero terminal SNR。右图为绘图把 SNR 下限截到 $10^{-12}$，exact zero 本身对应 $\log\mathrm{SNR}=-\infty$。

配套代码在 float64 下复现了 $T=1000$ 的 terminal 数值：

| schedule                         |        $\bar\alpha_T$ | $\operatorname{SNR}_T$ |
| -------------------------------- | --------------------: | ---------------------: |
| DDPM linear $\beta$              | $4.0358\times10^{-5}$ |  $4.0360\times10^{-5}$ |
| Improved DDPM cosine（clip 0.999） | $2.4288\times10^{-9}$ |  $2.4288\times10^{-9}$ |
| latent scaled-linear             | $4.6601\times10^{-3}$ |  $4.6819\times10^{-3}$ |

这些值与 Lin et al. 2024, Table 1 的量级一致。它们说明“非零”不是一个二元标签：$10^{-9}$ 和 $10^{-3}$ 都不严格为 0，但保留 signal 的程度相差六个数量级。

***

## 9. EDM 之后：不要再让 schedule 承担所有职责

Karras et al. 在 EDM 中把多个 diffusion family 重写到以 additive noise standard deviation $\sigma$ 为坐标的共同框架：

$$
p(x;\sigma)
=p_{\mathrm{data}}*\mathcal N(0,\sigma^2I).
$$

对这种坐标，可以粗略理解为 $a=1,b=\sigma$，因此 nominal SNR 为 $1/\sigma^2$。EDM 最值得带回 D1 的并不是某个具体 FID，而是它把长期纠缠在 “schedule” 一词中的组件拆开了（Karras et al., 2022, Table 1, pp. 2--5）。

| 设计对象                            | 它回答的问题                                      | 典型记号                                  |
| ------------------------------- | ------------------------------------------- | ------------------------------------- |
| forward marginal path           | 每个 noise level 的 corrupted distribution 是什么 | $a(t),b(t)$ 或 $\sigma$                |
| time/noise coordinate           | 用什么参数标记同一条 path                             | $t,\gamma,\sigma$                     |
| training proposal               | 训练时哪些 noise levels 更常出现                     | $p_{\mathrm{train}}(\sigma)$          |
| loss weighting/parameterization | 各 noise level 的误差怎样计入目标                     | $\lambda(\sigma)$、预测 $x_0/\epsilon/v$ |
| sampling grid and solver        | 生成时在哪些点调用网络、怎样积分                            | $\{\sigma_i\}$、Euler/Heun 等           |

EDM 的 training noise distribution 使用 log-normal 形式，而 sampling grid 使用 polynomial spacing：

$$
\sigma_{i<N}
=\left[
\sigma_{\max}^{1/\rho}
+\frac{i}{N-1}
(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho})
\right]^\rho,
\qquad \sigma_N=0.
$$

这表明训练和采样没有理由必须以相同密度分配 noise levels。论文发现 $\rho\approx3$ 较能均衡 local truncation error，但图像 FID 更偏好 5--10，正文使用 $\rho=7$；数值误差均衡与感知质量最优并非同一个目标（Karras et al., 2022, Eq. (5) and App. D.1）。

> **跨文献综合。** 更准确的说法不是“选择一个好 schedule”，而是：先选择希望经过的 noisy marginals，再分别决定训练测度、loss metric、network parameterization 和采样网格。改变 schedule 往往会同时移动这些对象，所以实验有效并不自动说明真正起作用的是哪一个机制。Dieleman 2024 的长文进一步系统化了这一批判。

***

## 10. Terminal SNR：近似 prior 还是精确匹配 prior

### 10.1 标准做法中的 train-inference gap

生成时通常设

$$
p(x_T)=\mathcal N(0,I).
$$

但训练时

$$
x_T
=\sqrt{\bar\alpha_T}x_0
+\sqrt{1-\bar\alpha_T}\epsilon.
$$

只要 $\bar\alpha_T>0$，训练输入就仍含少量 data signal，而从 prior 抽取的 inference input 不含这些 signal。这构成 endpoint distribution mismatch。

传统 DDPM 通过让 $\bar\alpha_T$ 很小来近似解决它；变分目标中的 prior term

$$
D_{\mathrm{KL}}(q(x_T\mid x_0)\Vert p(x_T))
$$

也会显式记录 mismatch。因而“terminal SNR 非零”不等于模型在数学上完全无效，问题在于近似是否足够，以及训练和 sampler 是否以一致方式处理 endpoint。

Lin et al. 指出，Stable Diffusion 2.1 所用 scaled-linear schedule 的 terminal SNR 约为 $4.68\times10^{-3}$，保留 signal 明显多于 DDPM linear 和 clipped cosine。他们将这与模型偏向中等亮度的现象联系起来，并提出 exact zero terminal SNR、v-prediction、从最后 timestep 开始采样和 guidance rescaling 等修正（Lin et al., 2024, Secs. 3--4）。

这组实验主要针对 Stable Diffusion，不能直接上升为“所有非零 terminal SNR 都必然导致同等亮度问题”的定理。

### 10.2 在 $\sqrt{\bar\alpha_t}$ 空间重缩放

令

$$
r_t=\sqrt{\bar\alpha_t}.
$$

Lin et al. 的 rescaling 保留第一个训练时刻的 signal amplitude，并把最后一个移到 0：

$$
r_t'
=r_1\frac{r_t-r_T}{r_1-r_T},
\qquad
\bar\alpha_t'=(r_t')^2.
$$

于是

$$
r_1'=r_1,\qquad r_T'=0,\qquad \bar\alpha_T'=0.
$$

再从 cumulative quantities 恢复

$$
\alpha_1'=\bar\alpha_1',
\qquad
\alpha_t'=\frac{\bar\alpha_t'}{\bar\alpha_{t-1}'},
\qquad
\beta_t'=1-\alpha_t'.
$$

最后一步会得到 $\beta_T'=1$。这精确消除 signal，却也让部分传统公式出现除以 $\sqrt{\alpha_T}$ 或计算 $\log0$ 的数值问题。exact-zero endpoint 必须与 parameterization 和 sampler 一起设计，不能只改一行 betas 后继续假设所有旧公式都非奇异。

### 10.3 为什么 zero SNR 会暴露 $\epsilon$-prediction 的端点退化

当 $\bar\alpha_T=0$ 时，

$$
x_T=\epsilon.
$$

若 target 也是 $\epsilon$，网络在该点只需复制输入，loss 不再提供任何关于 $x_0$ 的监督。Lin et al. 建议使用

$$
v_t
=\sqrt{\bar\alpha_t}\epsilon
-\sqrt{1-\bar\alpha_t}x_0.
$$

在 terminal endpoint，

$$
v_T=-x_0.
$$

于是 conditional model 仍需根据条件信息估计数据，而不是完成 trivial identity task。不同 prediction parameterization 如何对应不同 loss weighting，将在 D2/D3 继续展开。

### 10.4 sampler 必须从训练 endpoint 启动

即使 schedule 已经做到 $\bar\alpha_T=0$，若少步 sampler 的第一个 network evaluation 使用 $t<T$，它仍会把 pure noise 交给一个只在 nonzero-SNR inputs 上训练过的时刻。Lin et al. 因此要求 sampling grid 包含最后训练 timestep。

“包含 endpoint”解决的是 train-inference input mismatch；“怎样安排其余步点”解决的是 finite-step discretization error。两者也不是同一个问题。

***

## 11. 从公式到 PyTorch

完整可运行版本位于 d1\_forward\_diffusion.py（补充材料暂未公开）。它使用 <code>torch.float64</code> 构造 schedules，使用 PyTorch/Matplotlib 生成本章两张图，不包含模型训练。

### 11.1 构造 linear 与 cosine schedules

```python
import math
import torch


def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def cosine_beta_schedule(T, s=0.008, max_beta=0.999):
    grid = torch.linspace(0, 1, T + 1, dtype=torch.float64)
    alpha_bar = torch.cos((grid + s) / (1 + s) * math.pi / 2).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(min=1e-12, max=max_beta)
```

cosine schedule 先定义 $T+1$ 个 cumulative endpoints，再由相邻比值恢复 $T$ 个 one-step betas。

### 11.2 预计算 forward coefficients

```python
betas = cosine_beta_schedule(T=1000)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

sqrt_alpha_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
```

这些数组都使用代码索引 <code>0,...,T-1</code>。代码的 <code>alpha\_bar\[0]</code> 对应论文 $\bar\alpha_1$，不是 $\bar\alpha_0=1$。

### 11.3 为 batch 提取不同 timestep 的系数

```python
def extract(coefficients, t, x):
    # t: [batch], x: [batch, ...]
    values = coefficients.to(device=x.device, dtype=x.dtype)[t]
    return values.reshape(t.shape[0], *((1,) * (x.ndim - 1)))
```

例如图像张量形状为 <code>\[B, C, H, W]</code>，reshape 后的系数为 <code>\[B, 1, 1, 1]</code>，可以沿 channel 和空间维广播。

### 11.4 直接采样 $q(x_t\mid x_0)$

```python
def q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    signal = extract(sqrt_alpha_bar, t, x0) * x0
    corruption = extract(sqrt_one_minus_alpha_bar, t, x0) * noise
    return signal + corruption, noise
```

训练时每个 batch item 可以有不同的 $t$。返回 noise 是因为下一章的模型会把它作为监督 target。

### 11.5 rescale 到 exact zero terminal SNR

```python
def rescale_zero_terminal_snr(betas):
    alpha_bar_sqrt = torch.cumprod(1 - betas, dim=0).sqrt()
    first = alpha_bar_sqrt[0].clone()
    last = alpha_bar_sqrt[-1].clone()

    alpha_bar_sqrt = alpha_bar_sqrt - last
    alpha_bar_sqrt = alpha_bar_sqrt * first / (first - last)
    alpha_bar = alpha_bar_sqrt.square()

    alphas = torch.cat([alpha_bar[:1], alpha_bar[1:] / alpha_bar[:-1]])
    return 1 - alphas
```

实际库代码还应检查输入范围、单调性、dtype 和 exact-zero endpoint。配套脚本已加入这些检查。

### 11.6 最小不变量测试

对每个 schedule，至少检查：

```python
assert torch.all((betas > 0) & (betas <= 1))
assert torch.all(alpha_bar[1:] <= alpha_bar[:-1])
assert torch.allclose(
    sqrt_alpha_bar.square() + sqrt_one_minus_alpha_bar.square(),
    torch.ones_like(alpha_bar),
)
```

对 zero-terminal rescale 还应检查：

```python
assert torch.allclose(new_alpha_bar[0], old_alpha_bar[0])
assert new_alpha_bar[-1] == 0
```

这些检查不能证明 schedule 适合某个生成任务，但可以尽早发现索引、累计乘积和数值实现错误。

***

## 12. 常见错误与边界

### 错误 1：把 $\beta_t$ 当成累计噪声

$\beta_t$ 只是一小步 variance；累计噪声是 $1-\bar\alpha_t$。

### 错误 2：把 $\alpha_t$ 与 $\bar\alpha_t$ 混用

$\alpha_t=1-\beta_t$ 是单步 signal power；$\bar\alpha_t$ 是从数据到当前时刻的累计 signal power。

### 错误 3：认为每个 conditional Gaussian 意味着 marginal Gaussian

$q(x_t\mid x_0)$ 是 Gaussian，不代表对 $x_0$ 混合后的 $q_t(x_t)$ 是 Gaussian。

### 错误 4：用同一个 $\epsilon$ 画多个时刻，就称其为 forward trajectory

这只保证每个时刻的 marginal 正确，不保证联合 Markov law 正确。

### 错误 5：把 $q(x_T\mid x_0)\approx\mathcal N(0,I)$ 写成无条件等号

对固定 $x_0$ 的 conditional，exact equality 需要 signal coefficient 为 0。对 aggregate $q_T$，若数据本身恰为与 kernel 相容的 Gaussian，也可能在非零 signal 下保持 Gaussian；两种等号条件不能混写。

### 错误 6：从 VDM 推出“schedule 完全不重要”

其 equivalence 有 continuous-time、endpoint 和 denoiser transformation 条件；finite discretization、estimator variance 和优化仍会受 schedule 影响。

### 错误 7：把 training proposal、loss weighting 与 sampling grid 都叫 schedule

它们可以相互作用，但数学职责不同。importance correction 还决定了改变 training proposal 是只改变 estimator variance，还是连 objective 一起改变。

### 错误 8：只把 $\beta_T$ 改成 1，就认为 zero-terminal 模型完成了适配

exact-zero endpoint 会影响 prediction target、reverse formula、log variance 和 sampler 起点，需要整体处理。

***

## 13. 技术演进：不是越来越复杂，而是逐步拆清问题

| 年份   | 工作                    | 当时的问题                                          | 关键推进                                                    | 留下的问题                                 |
| ---- | --------------------- | ---------------------------------------------- | ------------------------------------------------------- | ------------------------------------- |
| 2015 | Sohl-Dickstein et al. | 灵活密度难训练、难采样                                    | 用 tractable forward diffusion 定义 learned reverse chain  | 原始图像质量有限；schedule/variance 设计空间大      |
| 2020 | DDPM                  | diffusion 尚未显示强图像生成能力                          | closed-form marginal + 简单 noise prediction 参数化          | linear schedule、fixed variance、慢采样    |
| 2021 | Improved DDPM         | 信息破坏与计算分配不理想                                   | cosine cumulative schedule                              | schedule 仍是经验设计                       |
| 2021 | VDM                   | 不同 diffusion specifications 难统一                | SNR/log-SNR 坐标与 continuous-time equivalence             | finite grid 与 estimator variance 仍需设计 |
| 2022 | EDM                   | schedule、preconditioning、solver、weighting 相互纠缠 | 把 training measure、network scaling 和 sampling grid 拆开   | 最佳资源分配仍依赖任务                           |
| 2024 | Lin et al.            | terminal training input 与 inference prior 不一致  | zero terminal SNR、v-prediction、endpoint-aligned sampler | exact-zero 的数值和普适收益仍需具体评估             |

这一时间线的重点不是“新 schedule 取代旧 schedule”，而是研究问题从“怎样让 forward process 可计算”逐渐转向“怎样在 noise-level axis 上正确分配统计和计算资源”。

***

## 14. 本章给下一章准备了什么

[D2](/blog/diffusion/d2-ddpm-objective/) 从未知的 reverse conditional 出发。到这里，我们已经准备好四个关键工具：

1. forward path factorization

   $$
   q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1});
   $$

2. 任意时刻 direct marginal

   $$
   q(x_t\mid x_0)
   =\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I);
   $$

3. reparameterized training sample

   $$
   x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon;
   $$

4. 一套能比较不同噪声强度的 SNR/log-SNR 坐标。

下一章会证明：给定训练样本 $x_0$ 和 noisy state $x_t$，一步 posterior

$$
q(x_{t-1}\mid x_t,x_0)
$$

也是解析 Gaussian。随后，path ELBO 的每个 reverse KL 会变成一个监督回归问题，最终导出 DDPM 的 noise-prediction objective。

***

## 15. 章节小结

- forward diffusion 是人为规定的 corruption/inference process，不需要神经网络；

- DDPM 使用 variance-preserving Gaussian Markov chain，$\mathcal N(0,I)$ 是其 invariant distribution；

- $\beta_t$ 控制单步噪声，$\bar\alpha_t$ 控制从 $x_0$ 到 $x_t$ 的累计 signal；

- 由于 Gaussian 线性组合封闭，任意时刻都可直接采样：

  $$
  x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon;
  $$

- direct marginal sampling 与真实 Markov trajectory sampling 是不同的联合 coupling；

- 有限时刻的 aggregate data marginal 通常不是 Gaussian；terminal prior 匹配需要足够小或精确为零的 signal coefficient；

- SNR/log-SNR 比 timestep 更适合跨 schedule 比较；

- linear、cosine、learned log-SNR 和 zero-terminal rescaling 解决的是不同层面的问题；

- EDM 之后，更清晰的做法是分别讨论 forward path、training proposal、loss weighting、parameterization 和 sampling grid。

***

## 16. 思考题

1. **一步完全加噪为何没有帮助？** 从 mutual information 和逆条件分布两个角度解释 $\beta_1=1$ 与“很多小步”的区别。

2. **marginal 相同是否意味着过程相同？** 设多个 $x_t$ 都用同一个 $\epsilon$ 直接构造。证明每个 marginal 正确，并分析为什么它们的 cross-time covariance 一般不同于原 Markov chain。

3. **time reparameterization 保持了什么？** 给定严格单调 $\gamma(t)$，换一个单调时间坐标后，哪些 noisy marginals 不变？training proposal、loss estimator variance 和 finite sampling grid 中哪些会改变？

4. **SNR 是否真的与数据无关？** 若数据 covariance 为 $\Sigma_0\ne I$，分析不同 eigen-directions 的实际 signal/noise ratio。一个 scalar SNR 会隐藏什么？

5. **exact zero 的收益与代价。** $\bar\alpha_T=0$ 消除了哪一种 mismatch？它为什么又会使某些 DDPM reverse formulas 数值奇异？设计实现时应在哪些接口上增加 endpoint handling？

6. **cosine 的成功说明了什么？** 区分以下三种论断：cosine 比某个 linear baseline 好；cosine 对所有数据最优；感知信息在 cosine time 中均匀下降。哪些有论文实验证据，哪些还需要额外定义或证明？

7. **训练与采样是否应使用同一 noise-level density？** 结合“学习难度”“perceptual relevance”“误差累积”和“ODE truncation error”，论证二者可能不同的原因。

***

## 17. 本章来源与继续阅读

核心历史和公式优先引用原始论文；教学资料只用于讲解顺序与代码对照。

1. Jascha Sohl-Dickstein et al. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/1503.03585 "官方论文页面")。
2. Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2006.11239 "官方论文页面")。
3. Alex Nichol, Prafulla Dhariwal. *Improved Denoising Diffusion Probabilistic Models*. ICML 2021. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2102.09672 "官方论文页面")。
4. Diederik P. Kingma et al. *Variational Diffusion Models*. NeurIPS 2021. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2107.00630 "官方论文页面")。
5. Tero Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS 2022. 本地论文（补充材料暂未公开）；[D1 相关笔记](https://arxiv.org/abs/2206.00364 "官方论文页面")。
6. Shanchuan Lin et al. *Common Diffusion Noise Schedules and Sample Steps are Flawed*. WACV 2024. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2305.08891 "官方论文页面")。
7. Niels Rogge, Kashif Rasul. *The Annotated Diffusion Model*. Hugging Face, 2022. 官方 Markdown 快照（补充材料暂未公开）；使用边界笔记（补充材料暂未公开）。
8. Sander Dieleman. *Noise Schedules Considered Harmful*. 2024. 本地网页快照（补充材料暂未公开）。
9. Chieh-Hsin Lai et al. *The Principles of Diffusion Models*. arXiv:2510.21890v2, 2026 revision. 固定版本 PDF（补充材料暂未公开）；D0--D2 范围核验（补充材料暂未公开）。

独立代数复核见 DDPM core derivation ledger（补充材料暂未公开）。官方代码版本与 commit 见 code provenance（补充材料暂未公开）。
