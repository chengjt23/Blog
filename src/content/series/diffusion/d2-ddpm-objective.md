---
title: 已知的加噪过程怎样教会网络反向生成
description: >-
  从 Gaussian forward process 出发，推导 posterior、path ELBO 与 noise-prediction
  objective。
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
  解释 direct noising、reverse Gaussian 和 noise regression 的完整链条，并区分 VLB 与
  simplified loss。
---
第一章留下了一个关键空缺：前向 transition 完全已知，真正的无条件逆过程 $q(x_{t-1}\mid x_t)$ 却依赖未知的数据分布。DDPM 的核心贡献，是绕开对这个分布的直接求解，把它改写成可以用真实样本监督的 Gaussian regression。

这件事能够成立，依赖三个条件同时满足：任意噪声时刻可以直接采样；给定原始数据后，一步 posterior 可以解析计算；反向 Gaussian 的 KL 可以转成均值回归。

## 1. Gaussian 前向过程提供了什么

Ho、Jain 与 Abbeel 在 NeurIPS 2020 论文 [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239 "官方论文页面") 中固定了一条 Gaussian Markov chain。令 $X_0\in\mathbb R^d$ 是数据随机变量，$x_0$ 是一次具体取值。第 $t$ 步加入的方差记为 $\beta_t\in(0,1)$，并定义单步信号保留率

$$
\alpha_t=1-\beta_t.
$$

前向 transition 是

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{\alpha_t}\,x_{t-1},
\beta_t I_d
\right).
$$

由于线性 Gaussian 在复合后仍然是 Gaussian，定义累计信号保留率

$$
\bar\alpha_t
=
\prod_{s=1}^{t}\alpha_s,
$$

便可直接得到任意时刻的条件分布

$$
\boxed{
q(x_t\mid x_0)
=
\mathcal N
\left(
x_t;
\sqrt{\bar\alpha_t}\,x_0,
(1-\bar\alpha_t)I_d
\right).
}
$$

等价地，只需采样一次 $\epsilon\sim\mathcal N(0,I_d)$，就能构造

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}\,x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon.
}
$$

第一项是保留下来的 signal，第二项是累计 noise。若数据已经按每维单位方差标准化，一个常用的信噪比定义为

$$
\operatorname{SNR}(t)
=
\frac{\bar\alpha_t}{1-\bar\alpha_t}.
$$

因此训练不必真的从 $x_0$ 逐步模拟到 $x_t$。随机选一个时刻 $t$，一次 Gaussian perturbation 就能得到对应训练样本。Noise schedule 决定的是 $\bar\alpha_t$ 怎样下降，也就是不同噪声难度在时间轴上怎样分布。

![不同 noise schedule 对累计信号和 SNR 的影响](/images/diffusion/d1_schedule_comparison.png)

## 2. 为什么给定原始数据后，逆向一步可以算

真正未知的是 $q(x_{t-1}\mid x_t)$。但训练时我们知道这条 noisy sample 来自哪个 $x_0$，因此可以先研究条件 posterior

$$
q(x_{t-1}\mid x_t,x_0).
$$

根据 Bayes rule，它与两个已知 Gaussian 的乘积成正比：

$$
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})
q(x_{t-1}\mid x_0).
$$

把两个关于 $x_{t-1}$ 的 Gaussian 指数项合并，得到

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N
\left(
x_{t-1};
\tilde\mu_t(x_t,x_0),
\tilde\beta_t I_d
\right),
}
$$

其中 posterior variance 为

$$
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}
{1-\bar\alpha_t}
\beta_t,
$$

posterior mean 为

$$
\tilde\mu_t(x_t,x_0)
=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t.
$$

$\tilde\mu_t$ 是 $x_0$ 与 $x_t$ 的线性组合，$\tilde\beta_t$ 只由 schedule 决定。这就是 DDPM 能获得监督信号的原因：训练样本给出了 $x_0$，所以真实 posterior 的均值和方差都能计算。

无条件的 $q(x_{t-1}\mid x_t)$ 则需要对所有可能的 $x_0$ 求混合，通常不是单个 Gaussian。下图中每个彩色条件分量都容易写出，黑色 mixture 才是模型需要逼近的未知逆分布。

![给定 x\_0 的 Gaussian posterior 与无条件 reverse mixture](/images/diffusion/d2_reverse_conditionals.png)

## 3. Path ELBO 怎样把生成问题拆成逐步匹配

Sohl-Dickstein 等人在 ICML 2015 论文 [*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*](https://arxiv.org/abs/1503.03585 "官方论文页面") 中已经用变分下界训练 reverse chain。DDPM 沿用这条路径，并把模型写成

$$
p_\theta(x_{0:T})
=
p_T(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t),
$$

其中

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

$\mu_\theta$ 与 $\Sigma_\theta$ 分别是网络给出的 reverse mean 和 covariance。把已知 forward path $q(x_{1:T}\mid x_0)$ 当作 variational posterior，负对数似然的上界可以拆成

$$
\mathcal L_{\mathrm{vlb}}
=
\mathbb E_q
\left[
L_T
+
\sum_{t=2}^{T}L_{t-1}
+
L_0
\right],
$$

其中

$$
L_T
=
\operatorname{KL}
\left(
q(x_T\mid x_0)
\Vert
p_T(x_T)
\right)
$$

检查前向终点是否接近 chosen prior，

$$
L_{t-1}
=
\operatorname{KL}
\left(
q(x_{t-1}\mid x_t,x_0)
\Vert
p_\theta(x_{t-1}\mid x_t)
\right)
$$

逐步匹配真实 conditional posterior 与 learned reverse transition，而

$$
L_0
=
-\log p_\theta(x_0\mid x_1)
$$

负责最后的数据重建。

Path ELBO 的作用不是证明真实无条件逆过程为 Gaussian。它只是把一个难以直接优化的 data likelihood，转成许多具有解析 target 的 conditional Gaussian matching。

## 4. Gaussian KL 为什么会变成噪声预测

DDPM 没有让网络直接输出 $\tilde\mu_t$。由前向采样式可知

$$
x_0
=
\frac{
x_t-\sqrt{1-\bar\alpha_t}\,\epsilon
}
{\sqrt{\bar\alpha_t}}.
$$

把它代入 posterior mean 并整理，可以把真实均值写成

$$
\tilde\mu_t(x_t,x_0)
=
\frac1{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon
\right).
$$

因此 Ho 等人用网络 $\epsilon_\theta(x_t,t)$ 预测加入 $x_t$ 的噪声，并定义 model mean

$$
\boxed{
\mu_\theta(x_t,t)
=
\frac1{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
}
$$

若 reverse variance 固定为 $\sigma_t^2 I_d$，两个 Gaussian 的 KL 只差均值，单步变分项便等价于带权噪声误差：

$$
L_{t-1}
=
\mathbb E
\left[
w_t
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
\right]
+
C_t,
$$

其中 $C_t$ 与 $\theta$ 无关，而时间权重为

$$
w_t
=
\frac{\beta_t^2}
{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}.
$$

DDPM 实际用于高质量生成的 simplified objective 则直接去掉 $w_t$：

$$
\boxed{
\mathcal L_{\mathrm{simple}}
=
\mathbb E_{
x_0,t,\epsilon
}
\left[
\left\|
\epsilon
-
\epsilon_\theta
\left(
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
t
\right)
\right\|^2
\right].
}
$$

这里 $t$ 均匀采自 $\{1,\ldots,T\}$。$\mathcal L_{\mathrm{simple}}$ 与完整 VLB 共享相同的 pointwise regression target，却重新分配了各噪声时刻的权重。把它说成“ELBO 完全等价于无权噪声 MSE”是不准确的。

下图展示了两类 objective 对时间步的不同强调。权重变化不会改变无限容量下每个 $t$ 的条件均值，却会改变有限网络把容量用在哪里。

![完整 VLB 与 simplified noise loss 的 timestep 权重](/images/diffusion/d2_objective_weights.png)

## 5. 从训练目标到反向采样

训练时只需重复三件事：采样 $x_0\sim p_{\mathrm{data}}$，均匀选择 $t$，再采样 $\epsilon\sim\mathcal N(0,I_d)$ 构造 $x_t$。网络根据 $(x_t,t)$ 预测 $\epsilon$，并最小化上面的平方误差。

生成时从 $x_T\sim\mathcal N(0,I_d)$ 开始，按 $t=T,\ldots,1$ 迭代

$$
\boxed{
x_{t-1}
=
\frac1{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right)
+
\sigma_t z,
}
$$

其中 $z\sim\mathcal N(0,I_d)$；最后一步通常令 $z=0$。这就是 DDPM 的 ancestral sampling。网络每次只预测局部 reverse transition，完整样本由许多次调用累积得到。

Nichol 与 Dhariwal 在 ICML 2021 论文 [*Improved Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2102.09672 "官方论文页面") 中指出，DDPM 的 mean、variance 和训练权重不应混成一个问题。他们用 $\mathcal L_{\mathrm{simple}}$ 主导噪声预测，再用小权重 VLB 学习 reverse variance；同时提出 cosine schedule 与 timestep importance sampling。

这些改进没有改变 DDPM 的基本逻辑。已知 Gaussian forward process 负责制造 $(x_0,x_t)$ 训练对，可解析 posterior 提供真实的一步 target，神经网络再把这个 target 投影成只依赖 $(x_t,t)$ 的 reverse transition。

“预测噪声就能生成”并不是一个孤立技巧。它是 Gaussian posterior、path ELBO 与 mean parameterization 共同作用后的结果。

## 文献索引

| 时间   | 论文                                                                                      | 本章采用的内容                                                              |
| ---- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 2015 | Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* | 建立 path variational objective 与 learned reverse chain                |
| 2020 | Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*                           | 给出 direct noising、posterior、noise parameterization 与 simplified loss |
| 2021 | Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*                  | 分离 mean、variance、schedule 与 timestep weighting 的设计                   |
