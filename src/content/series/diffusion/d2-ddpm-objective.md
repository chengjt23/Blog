---
title: 网络为什么只预测噪声，就能学会生成
description: >-
  从 conditioned posterior 与 path ELBO 出发，解释 DDPM 如何把未知 reverse process 化为可训练的
  noise regression。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: diffusion
order: 2
slug: d2-ddpm-objective
tags:
  - diffusion
  - ddpm
  - elbo
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: >-
  推导 posterior、reverse Gaussian、ELBO 与 noise MSE 的关键链条，并区分完整 likelihood
  objective 与重加权 simple loss。
---
第一章规定了一条从数据走向 Gaussian noise 的前向路径。沿用其中的记号：$X_0\in\mathbb R^d$ 是干净数据，$X_t$ 是时刻 $t$ 的带噪随机变量，$\bar\alpha_t\in[0,1]$ 是累计保留的 signal power，$\varepsilon\sim\mathcal N(0,I)$ 是 $d$ 维标准 Gaussian noise，$I$ 是单位矩阵。给定 $X_0$，我们可以在任意时刻直接构造

$$
X_t
=\sqrt{\bar\alpha_t}X_0
+\sqrt{1-\bar\alpha_t}\,\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I).
$$

记 $q$ 为这条已知前向过程诱导的概率分布。生成需要走相反方向：从 $X_T\sim\mathcal N(0,I)$ 出发，逐步得到 $X_{T-1},\ldots,X_0$。真正需要的条件分布

$$
q(x_{t-1}\mid x_t)
$$

仍然依赖未知的数据分布。既然它不能直接计算，网络究竟凭什么学会逆转？为什么 2020 年的 DDPM 最终把训练目标写成了一个看起来异常简单的问题——预测加入的 Gaussian noise？

这个答案经历了两次关键推进。2015 年，Sohl-Dickstein 等人在 ICML 论文 [*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*](https://arxiv.org/abs/1503.03585 "官方论文页面") 中，用一条 learned reverse chain 和路径空间变分下界，把未知逆过程变成了概率模型优化问题。2020 年，Ho、Jain 和 Abbeel 在 NeurIPS 论文 [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239 "官方论文页面") 中发现：Gaussian forward process 还隐藏着一个可解析的 conditioned posterior，它能把每一步逆向学习继续化成带权回归。

整条逻辑链是

$$
\text{未知的真实 reverse}
\longrightarrow
\text{可解析的 conditioned posterior}
\longrightarrow
\text{path ELBO}
\longrightarrow
\text{逐步 Gaussian KL}
\longrightarrow
\text{带权 noise MSE}.
$$

本章的核心，就是把这五个箭头逐一解释清楚。

## 1. 2015：把未知逆过程改写成一个可学习的概率模型

前向过程由我们规定，因此

$$
q(x_t\mid x_{t-1})
$$

是已知的；但真正的反向条件分布由 Bayes rule 决定：

$$
q(x_{t-1}\mid x_t)
=\frac{
q(x_t\mid x_{t-1})q_{t-1}(x_{t-1})
}{
q_t(x_t)
}.
$$

这里 $q_{t-1}$ 与 $q_t$ 分别是前向过程在时刻 $t-1$ 和 $t$ 的无条件边缘密度。它们由真实数据分布 $q_{\mathrm{data}}$ 经过多步加噪得到，因此能够采样，却通常不能在任意位置精确计算。Forward transition 是 Gaussian，并不能消除这个困难。

Sohl-Dickstein et al. 2015 的选择不是继续求一个不存在的解析解，而是定义新的 reverse generative model：

$$
\boxed{
p_\theta(x_{0:T})
=p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t).
}
$$

这里 $p(x_T)$ 是生成过程的起点，通常取标准 Gaussian $\mathcal N(0,I)$；$p_\theta(x_{t-1}\mid x_t)$ 是由参数 $\theta$ 控制的 learned reverse transition。生成时先采样 $X_T$，再按 $t=T,T-1,\ldots,1$ 逐步采样，最终得到 $X_0$。

式中的 $x_{0:T}$ 是整条轨迹 $(x_0,x_1,\ldots,x_T)$ 的简写，$\theta$ 表示神经网络中需要通过训练学习的全部参数。

在 Gaussian diffusion 中，2015 年论文依据小步或连续极限下的逆过程结构，把每一步模型限制为 Gaussian：

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N\!\left(
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

$\mu_\theta$ 与 $\Sigma_\theta$ 分别是网络给出的 reverse mean 和 covariance。这个选择让采样可执行，却仍然留下训练问题：我们没有真实的 $q(x_{t-1}\mid x_t)$ 可以逐点监督它。

这里必须区分三个外形相近、职责完全不同的对象：

| 分布                          | 是否可直接计算 | 使用场景              | 含义                     |
| --------------------------- | ------: | ----------------- | ---------------------- |
| $q(x_{t-1}\mid x_t)$        |   通常不可以 | 理想目标              | 真实的无条件 reverse         |
| $q(x_{t-1}\mid x_t,x_0)$    |   训练时可以 | posterior teacher | 额外给定原始数据后的前向 posterior |
| $p_\theta(x_{t-1}\mid x_t)$ |    学习得到 | 生成时使用             | 神经网络定义的 reverse model  |

下面的图是本文已有教学实验，不是原论文插图。数据只取 $x_0\in\{-2,2\}$，并观察 $x_2=0$。两条虚线分别是给定不同 $x_0$ 后的 Gaussian posterior；黑线是把未知 $x_0$ 积分掉后得到的真实 reverse mixture。它清楚说明：conditioned posterior 可以是 Gaussian，而无条件 reverse 仍可能是多峰分布。

![Conditioned posterior 与无条件 reverse mixture 的差别](/images/diffusion/d2_reverse_conditionals.png)

2015 年工作的关键贡献，是不再要求逐点知道真实 reverse，而是通过整条路径的 likelihood bound 训练 $p_\theta$。但要让这条思路变成高效的监督学习，还需要找到一个训练时真正可计算的 teacher。

## 2. 2020：给定原始数据后，前一步 posterior 可以精确计算

DDPM 2020 的突破口是：训练时 $x_0$ 来自数据集，因此它不是未知变量。一旦同时给定当前带噪状态 $x_t$ 与原始样本 $x_0$，前一状态的 posterior

$$
q(x_{t-1}\mid x_t,x_0)
$$

可以由两个已知 Gaussian 相乘得到。

沿用第一章的记号，$\beta_t\in(0,1)$ 是第 $t$ 步的 forward noise variance，

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

$\alpha_t$ 是单步保留的 signal power，$\bar\alpha_t$ 是到时刻 $t$ 为止累计保留的 signal power。已知的两项分别是

$$
q(x_t\mid x_{t-1})
=\mathcal N\!\left(
\sqrt{\alpha_t}x_{t-1},
\beta_t I
\right),
$$

以及由第一章 direct marginal 得到的

$$
q(x_{t-1}\mid x_0)
=\mathcal N\!\left(
\sqrt{\bar\alpha_{t-1}}x_0,
(1-\bar\alpha_{t-1})I
\right).
$$

根据 Markov property 与 Bayes rule，在忽略不依赖 $x_{t-1}$ 的归一化常数后，

$$
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})
q(x_{t-1}\mid x_0).
$$

两个 Gaussian 的乘积仍是 Gaussian。计算它们关于 $x_{t-1}$ 的 precision，也就是 covariance 的逆，可得 posterior precision

$$
\begin{aligned}
\tilde\beta_t^{-1}
&=
\frac{\alpha_t}{\beta_t}
+\frac{1}{1-\bar\alpha_{t-1}}\\
&=
\frac{1-\bar\alpha_t}
{\beta_t(1-\bar\alpha_{t-1})}.
\end{aligned}
$$

因此 posterior variance 为

$$
\boxed{
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}
{1-\bar\alpha_t}\beta_t.
}
$$

波浪号用于区分 posterior variance $\tilde\beta_t$ 与 forward one-step variance $\beta_t$。前者表示：已经观察到 $x_t$ 和 $x_0$ 后，我们对 $x_{t-1}$ 还剩多少不确定性。

均值则由 precision-weighted information 相加得到：

$$
\boxed{
\tilde\mu_t(x_t,x_0)
=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t.
}
$$

于是 Ho et al. 2020 的 Eqs. (6)--(7) 给出

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N\!\left(
\tilde\mu_t(x_t,x_0),
\tilde\beta_t I
\right).
}
$$

这个 posterior 只在训练时可用，因为训练数据提供了 $x_0$。生成时并不存在一个真实 $x_0$ 可以输入网络。DDPM 的做法是让只接收 $(x_t,t)$ 的模型，去匹配大量由 $(x_0,x_t)$ 构造出的 posterior teacher。

为什么带有 $x_0$ 的 teacher 能教出不带 $x_0$ 的 reverse？以固定 covariance 的 mean regression 为例，模型在同一个 $x_t$ 上会看到来自不同 $x_0$ 的 posterior mean。平方损失的 population optimum 是

$$
\mu_\theta^*(x_t,t)
=
\mathbb E\!\left[
\tilde\mu_t(X_t,X_0)
\mid X_t=x_t
\right]
=
\mathbb E[X_{t-1}\mid X_t=x_t].
$$

$\mathbb E$ 表示对随机变量取期望；第二个等号来自 iterated expectation，也就是先对 $X_{t-1}$ 在 $(X_t,X_0)$ 条件下求均值，再对未知的 $X_0$ 做平均。模型无法记住训练时的具体 $x_0$，只能学到在所有可能起点上平均后的 reverse mean。Conditioned posterior 提供可计算监督，而对 $X_0$ 的平均把它重新连接到无条件 reverse。

## 3. 2015—2020：path ELBO 把全局 likelihood 拆成逐步匹配

有了 posterior teacher，还需要解释为什么匹配它是在训练一个生成模型，而不只是做局部回归。答案来自 2015 年论文的 path variational bound。Ho et al. 2020 在 Appendix A 重述了这一推导，并明确将它归因于 Sohl-Dickstein et al.。

对一个真实样本 $x_0$，我们希望提高生成模型给它的 likelihood $p_\theta(x_0)$。但这个 likelihood 需要对整条隐变量路径 $x_{1:T}$ 积分：

$$
p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}.
$$

直接计算这个高维积分困难。2015 年的做法是使用已知的 forward path

$$
q(x_{1:T}\mid x_0)
$$

作为 variational proposal。把它乘进再除掉，并对对数使用 Jensen inequality，可得

$$
\begin{aligned}
\log p_\theta(x_0)
&=
\log
\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]\\
&\ge
\mathbb E_q
\left[
\log p_\theta(x_{0:T})
-\log q(x_{1:T}\mid x_0)
\right].
\end{aligned}
$$

右侧是 evidence lower bound，简称 ELBO。它小于等于真实 log-likelihood，因此最大化 ELBO 等价于最小化 negative ELBO，记为 $L_{\mathrm{vlb}}$。

Gaussian forward process 的特殊之处在于，它不仅可以正向分解，也可以借助上一节的 posterior 反向分解：

$$
q(x_{1:T}\mid x_0)
=
q(x_T\mid x_0)
\prod_{t=2}^{T}
q(x_{t-1}\mid x_t,x_0).
$$

将它与 reverse model 的分解代入 ELBO，得到 DDPM 2020 Eq. (5) 的三类项：

$$
\boxed{
\begin{aligned}
L_{\mathrm{vlb}}
={}&
\underbrace{
D_{\mathrm{KL}}
\!\left(q(x_T\mid x_0)\,\Vert\,p(x_T)\right)
}_{L_T}\\
&+
\sum_{t=2}^{T}
\underbrace{
\mathbb E_{q(x_t\mid x_0)}
D_{\mathrm{KL}}
\!\left(
q(x_{t-1}\mid x_t,x_0)
\,\Vert\,
p_\theta(x_{t-1}\mid x_t)
\right)
}_{L_{t-1}}\\
&+
\underbrace{
\mathbb E_{q(x_1\mid x_0)}
\left[-\log p_\theta(x_0\mid x_1)\right]
}_{L_0}.
\end{aligned}
}
$$

$D_{\mathrm{KL}}(q\Vert p)$ 是 Kullback–Leibler divergence，用来衡量两个分布的差异。这里三类项各有明确职责：

- $L_T$ 检查前向终点 $q(x_T\mid x_0)$ 是否接近生成 prior $p(x_T)$；
- 中间的 $L_{t-1}$ 让 learned reverse transition 匹配可解析的 posterior teacher；
- $L_0$ 负责最后一步从 $x_1$ 重建离散或连续数据的 decoder likelihood。

这三个边界不能混写。固定 forward schedule 后，$L_T$ 通常不依赖网络参数 $\theta$，但它并不因此数值为零；当 $t=1$ 时 posterior variance 退化为零，$L_0$ 也不能继续当作普通 Gaussian KL 处理。

这一步完成了最重要的概念转换：训练一个全局生成分布，被拆成了许多局部 reverse transition 的匹配问题。下一步只剩下：Gaussian 与 Gaussian 之间的 KL，为什么最后会变成预测噪声？

## 4. 2020：Gaussian KL 为什么会变成噪声回归

DDPM 2020 先固定 reverse covariance 为

$$
\Sigma_\theta(x_t,t)=\sigma_t^2I,
$$

其中 $\sigma_t^2$ 是模型在第 $t$ 个 reverse step 使用的 variance。它不是 forward variance $\beta_t$，尽管论文实验会选择 $\sigma_t^2=\beta_t$ 或 $\sigma_t^2=\tilde\beta_t$。

当 teacher 与 model covariance 都固定时，Gaussian KL 中与网络 mean 有关的部分是

$$
\frac{1}{2\sigma_t^2}
\left\|
\tilde\mu_t(x_t,x_0)
-\mu_\theta(x_t,t)
\right\|^2.
$$

如果直接让网络预测 posterior mean，这已经是一个合法目标。DDPM 的关键参数化，是进一步用 forward noise $\varepsilon$ 重写这个 mean。

由 direct noising 公式

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\,\varepsilon
$$

可以反解

$$
x_0
=
\frac{
x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon
}{
\sqrt{\bar\alpha_t}
}.
$$

把它代入上一节的 posterior mean 并整理，得到 Ho et al. 2020 Eq. (11) 背后的恒等式：

$$
\tilde\mu_t(x_t,x_0)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\varepsilon
\right).
$$

因此可以让网络输出 noise estimate $\varepsilon_\theta(x_t,t)$，再用完全相同的结构定义 model mean：

$$
\boxed{
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\varepsilon_\theta(x_t,t)
\right).
}
$$

$\varepsilon_\theta$ 的输入只有当前状态 $x_t$ 和时间 $t$，输出维度与数据相同。它不是在生成时读取真实噪声，而是在带噪样本中估计“哪一部分更像 forward noise”。

teacher mean 与 model mean 的差只剩

$$
\tilde\mu_t-\mu_\theta
=
\frac{\beta_t}
{\sqrt{\alpha_t(1-\bar\alpha_t)}}
\left(
\varepsilon_\theta-\varepsilon
\right).
$$

代回 Gaussian KL，得到 Ho et al. 2020 Eq. (12) 的带权 noise MSE：

$$
\boxed{
L_{t-1}^{\mathrm{mean}}
=
\mathbb E
\left[
w_t
\left\|
\varepsilon
-\varepsilon_\theta(x_t,t)
\right\|^2
\right],
\qquad
w_t
=
\frac{\beta_t^2}
{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}.
}
$$

$w_t$ 是由 forward schedule 与 reverse variance 共同决定的 timestep weight。它说明：完整 ELBO 并没有平等看待所有噪声时刻。

但 DDPM 2020 真正用于高质量生成的 Eq. (14) 主动删除了这个权重，使用

$$
\boxed{
L_{\mathrm{simple}}
=
\mathbb E_{
X_0,t,\varepsilon
}
\left[
\left\|
\varepsilon
-\varepsilon_\theta(X_t,t)
\right\|^2
\right],
}
$$

其中

$$
\begin{aligned}
X_0&\sim q_{\mathrm{data}},\\
t&\sim\operatorname{Uniform}\{1,\ldots,T\},\\
\varepsilon&\sim\mathcal N(0,I),\\
X_t&=\sqrt{\bar\alpha_t}X_0
+\sqrt{1-\bar\alpha_t}\,\varepsilon.
\end{aligned}
$$

这不是把 ELBO “化简后去掉常数”，而是改变不同 timestep 的相对权重。DDPM 的实验结果是：完整 VLB 更有利于 codelength，而 $L_{\mathrm{simple}}$ 在其设置中获得更好的 sample quality。这是 2020 年论文的经验发现，不是 likelihood 与视觉质量必然冲突的普适定理。

下图比较两种固定 reverse variance 下的 VLB mean weight 与 $L_{\mathrm{simple}}$ 的常数权重。它直观显示：删除 $w_t$ 会大幅重分配模型在不同噪声尺度上的训练注意力。

![完整 VLB 与 L\_simple 对不同 timestep 的权重差异](/images/diffusion/d2_objective_weights.png)

最后还要解释一个容易误解的地方：网络不是在记忆每个样本随机抽到的具体 $\varepsilon$。平方损失的 population optimum 是

$$
\varepsilon_\theta^*(x_t,t)
=
\mathbb E[
\varepsilon
\mid X_t=x_t
].
$$

也就是说，网络学习的是“在观察到 $x_t$ 后，forward noise 的条件均值”。这个条件均值由数据分布的局部几何决定；一旦它被估计出来，model mean 公式就把它转换成逆向移动方向。Noise prediction 之所以能生成，不是因为噪声包含语义，而是因为**从 noisy observation 中分离噪声，等价于估计数据结构应该朝哪里恢复**。

## 5. 2021：mean、variance 与训练权重开始被分别设计

DDPM 2020 证明了 noise prediction 的有效性，但它仍把 reverse variance 固定为预设值。Nichol 与 Dhariwal 在 ICML 2021 论文 [*Improved Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2102.09672 "官方论文页面") 中指出：$L_{\mathrm{simple}}$ 只训练 mean 或 noise predictor，本身不能给 variance branch 提供正确的 likelihood 信号。

Improved DDPM 让网络额外输出 $v_\theta(x_t,t)$，并在 log-variance 空间插值：

$$
\Sigma_\theta(x_t,t)
=
\exp\!\left(
v_\theta\log\beta_t
+
(1-v_\theta)\log\tilde\beta_t
\right)I.
$$

这不是学习任意 covariance matrix，而是在 $\beta_t I$ 与 $\tilde\beta_t I$ 定义的窄族之间学习 diagonal variance。为了让 noise MSE 继续主导 mean，同时用 likelihood 训练 variance，论文采用

$$
L_{\mathrm{hybrid}}
=
L_{\mathrm{simple}}
+\lambda L_{\mathrm{vlb}},
\qquad
\lambda=0.001,
$$

并在 VLB 分支中对 model mean 使用 stop-gradient，使这部分梯度主要更新 variance。这个设计把“预测均值”和“估计不确定性”从同一个 loss 中拆开。

同年，Kingma、Salimans、Poole 与 Ho 在 NeurIPS 2021 论文 [*Variational Diffusion Models*](https://arxiv.org/abs/2107.00630 "官方论文页面") 中进一步用 SNR 坐标重新解释 objective weight。定义

$$
\gamma(t)
=-\log\operatorname{SNR}(t),
$$

其中 $t\in[0,1]$ 是连续时间，$\gamma(t)$ 是 negative log-SNR。VDM 将连续时间 VLB 的 diffusion term 写成

$$
L_\infty
=
\frac12
\mathbb E_{t,\varepsilon}
\left[
\gamma'(t)
\left\|
\varepsilon-\varepsilon_\theta(X_t,t)
\right\|^2
\right],
$$

其中 $\gamma'(t)$ 是 negative log-SNR 对时间的导数。这个表达说明至少有三件事必须分开：

- objective weighting 决定不同 SNR 区域在目标函数中有多重要；
- parameterization 决定网络预测 noise、data 还是其他等价变量；
- timestep proposal 决定训练时怎样抽样 $t$，在做 importance correction 后可以只改变估计方差而不改变目标。

这条 2015—2021 的演进最终形成了现代 DDPM 的训练与生成闭环。训练阶段从数据构造 $(X_t,\varepsilon)$，让网络学习 $\varepsilon_\theta$；生成阶段不再需要 $x_0$，而是从

$$
X_T\sim\mathcal N(0,I)
$$

开始。对 $t=T,T-1,\ldots,1$，先计算

$$
\mu_\theta(X_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
X_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\varepsilon_\theta(X_t,t)
\right),
$$

再采样

$$
X_{t-1}
=
\mu_\theta(X_t,t)
+\sigma_t Z_t,
\qquad
Z_t
\sim
\begin{cases}
\mathcal N(0,I), & t>1,\\
0, & t=1.
\end{cases}
$$

$Z_t$ 是 reverse sampling noise；它与训练时构造 $X_t$ 使用的 forward noise $\varepsilon$ 不是同一个随机变量。最后一步令 $Z_1=0$，因为此时目标是输出数据，而不是再次把 Gaussian noise 加回结果。

现在可以准确回答本章标题：

> 网络只预测噪声，是因为 Gaussian posterior 的 mean 可以完全由当前状态与 forward noise 表示；预测噪声经过解析公式转换后，就是预测 reverse transition 应该向哪里移动。

2015 年提供了 path-space variational training，2020 年把 posterior、ELBO 与 noise regression 连成一条可计算主链，2021 年则开始把 mean、variance、objective weight 和 estimator 分别设计。下一阶段的问题不再只是“如何训练 DDPM”，而是：这个 noise predictor 在统计上究竟学到了什么，它与数据分布的 score 有什么关系？

## 本章论文索引

| 时间   | 论文                                                                                            | 本章中的作用                                                       |
| ---- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| 2015 | Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, ICML | 定义 learned reverse chain 与 path variational bound            |
| 2020 | Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS                                | 推导 tractable posterior、ELBO 分解与 noise parameterization       |
| 2021 | Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML                  | 学习 reverse variance，并用 hybrid objective 分工训练 mean 与 variance |
| 2021 | Kingma et al., *Variational Diffusion Models*, NeurIPS                                        | 用 SNR/log-SNR 解释 objective weighting 与 timestep estimation   |
