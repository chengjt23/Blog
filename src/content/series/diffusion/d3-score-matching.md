---
title: Score Matching：去噪器究竟学到了什么
description: >-
  连接 score matching、去噪 score matching、Tweedie 公式、噪声预测与 annealed Langevin
  sampling。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 3
slug: d3-score-matching
tags:
  - diffusion
  - score-matching
  - tweedie
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦 noisy-data score 的统计含义、等价目标和正则性条件，并区分 population identity 与有限训练。
---
D2 把未知 reverse transition 的学习化成了一个异常简单的回归任务：随机选 $t$，合成

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
$$

然后让网络预测这次抽到的 $\epsilon$。但这留下了一个概念断点：网络只看到 $x_t,t$，同一个 $x_t$ 可能由很多 $x_0,\epsilon$ 组合产生，它不可能逐样本还原那次随机噪声。那么 squared-error regression 到底在学习什么？

本章的答案是 noisy aggregate marginal 的 **score**：

$$
\boxed{
s_t(x_t)
=
\nabla_{x_t}\log q_t(x_t).
}
$$

这不是“noise 看起来像 gradient”的类比，而是由 conditional expectation 精确导出的 population identity。它同时把四件事串在一起：

$$
\text{随机去噪监督}
\longrightarrow
\text{marginal score}
\longrightarrow
\text{posterior-mean denoiser}
\longrightarrow
\text{Langevin generation}.
$$

本章仍采用三层阅读方式。第一次阅读可抓住 score 的几何直觉、DSM 中心恒等式和 DDPM 映射；第二次阅读完整跟随 integration by parts、conditional-expectation decomposition 与 Tweedie 推导；研究层则需特别留意 boundary conditions、singular support、non-conservative vector field 和 sampling theorem 的边界。

***

## 1. Score：密度的局部几何

设 $p$ 是 $\mathbb R^d$ 上处处为正且可微的 density。定义

$$
\boxed{
s_p(x)=\nabla_x\log p(x).
}
$$

这里的梯度对数据坐标 $x$ 求导。它不是经典统计学中常见的 parameter score

$$
\nabla_\theta\log p_\theta(x).
$$

二者都被称为 score，却是不同对象：前者描述 density 在样本空间中的局部几何，后者描述 log-likelihood 对模型参数的敏感度。本章只讨论 data-space score。

### 1.1 一维直觉：score 不是 density 本身

在一维，

$$
s_p(x)=\frac{p'(x)}{p(x)}.
$$

所以：

- $s_p(x)>0$：向右移动会提高 log density；
- $s_p(x)<0$：向左移动会提高 log density；
- $s_p(x)=0$：可能位于 mode，也可能位于 local minimum；
- $|s_p(x)|$ 衡量的是 log density 的局部斜率，不是 density 值。

对 Gaussian $p(x)=\mathcal N(x;\mu,\sigma^2)$，

$$
s_p(x)=-\frac{x-\mu}{\sigma^2}.
$$

它是指向均值的线性向量场；variance 越小，恢复方向越强。对 Gaussian mixture，

$$
p(x)=\sum_{k=1}^K\pi_kp_k(x),
$$

$$
\begin{aligned}
s_p(x)
&=
\frac{\sum_k\pi_kp_k(x)s_{p_k}(x)}
{\sum_j\pi_jp_j(x)}\\
&=
\sum_k
\underbrace{
\frac{\pi_kp_k(x)}{p(x)}
}_{r_k(x)}
s_{p_k}(x).
\end{aligned}
$$

所以 mixture score 是 component scores 按 posterior responsibilities 加权，不只是“指向最近 mode”。components 之间的低密度区域也有结构：responsibility 的快速切换会让 score field 急剧弯曲。

![二维 Gaussian mixture 的 density 与 score field。箭头归一化后只显示方向，不能从图中读取 score magnitude。](/images/diffusion/d3_score_field.png)

### 1.2 score 丢掉了什么，又保留了什么

因为

$$
\nabla_x\log(cp(x))
=
\nabla_x\log p(x)
$$

对任意与 $x$ 无关的 $c>0$ 成立，score 不携带整体 normalization constant。但在 connected open domain $\Omega$ 上，若 $p,q>0$、可微且

$$
\nabla\log p=\nabla\log q,
$$

则

$$
\nabla(\log p-\log q)=0.
$$

connectedness 给出

$$
\log p-\log q=C,
\qquad
p=e^Cq.
$$

若二者都是 $\Omega$ 上归一化的 density，则 $C=0$，从而 $p=q$。因此在适当 domain 上，完整 score field 加 normalization 足以确定 density。

这个结论有两个重要边界。

第一，若 support 分成互不连通的 components，常数可以 component-wise 不同。设

$$
p(x)=\pi p_1(x)+(1-\pi)p_2(x)
$$

且 $p_1,p_2$ 的 supports 不相交。在第一个 component 内，

$$
\nabla\log p(x)
=
\nabla\log[\pi p_1(x)]
=
\nabla\log p_1(x),
$$

与 $\pi$ 无关。local score 因而不能识别 disconnected components 的相对质量。

第二，任意 neural vector field $s_\theta(x)$ 未必是某个 scalar log density 的 gradient。若它的 Jacobian 不满足适当对称性，或 line integral 依赖路径，它就是 non-conservative field，不能直接写成 $s_\theta=\nabla\log p_\theta$。score regression 仍可训练这样的 field，但“它自动定义了 globally normalized density”不是无条件结论。

***

## 2. 为什么 score 能绕开 partition function

考虑 energy-based model

$$
p_\theta(x)
=
\frac{\exp[-E_\theta(x)]}{Z(\theta)},
\qquad
Z(\theta)
=
\int\exp[-E_\theta(x)]dx.
$$

maximum likelihood 需要处理 $\nabla_\theta\log Z(\theta)$，其中包含对模型分布的 expectation，通常难以计算。若改对数据坐标求导：

$$
\begin{aligned}
\nabla_x\log p_\theta(x)
&=
\nabla_x[-E_\theta(x)-\log Z(\theta)]\\
&=
-\nabla_xE_\theta(x),
\end{aligned}
$$

因为 $Z(\theta)$ 与 $x$ 无关。Hyvärinen 2005 的 score matching 正是利用这一点：不直接匹配 normalized density，而匹配 model 与 data 的 data-space scores。

但“绕开 normalizer”不等于已经算出了 normalized likelihood。score matching 给出的是一种无需 $Z(\theta)$ 的估计准则；若还要报告 exact likelihood、比较 disconnected component mass 或从 energy 直接归一化，normalization 问题仍可能回来。

***

## 3. 从 explicit 到 implicit score matching

设真实 data density 为 $p_0$，其未知 score 为

$$
s_0(x)=\nabla_x\log p_0(x),
$$

模型输出 $s_\theta(x)$。最直接的 Fisher-divergence objective 是

$$
\boxed{
\mathcal J_{\mathrm{ESM}}(\theta)
=
\frac12
\mathbb E_{p_0}
\left[
\|s_\theta(X)-s_0(X)\|^2
\right].
}
$$

ESM 表示 explicit score matching。问题是我们只有 samples，不知道 $p_0(x)$，也无法直接计算 $s_0(x)$。score matching 的关键不是假装 target 可得，而是用 integration by parts 把它消掉。

### 3.1 完整展开

展开平方：

$$
\begin{aligned}
\mathcal J_{\mathrm{ESM}}
=&
\frac12\mathbb E_{p_0}\|s_\theta(X)\|^2\\
&-
\mathbb E_{p_0}
[s_\theta(X)^\top s_0(X)]\\
&+
\frac12\mathbb E_{p_0}\|s_0(X)\|^2.
\end{aligned}
$$

最后一项与 $\theta$ 无关。对 cross term 的第 $i$ 个坐标：

$$
\begin{aligned}
&-
\int_\Omega
p_0(x)s_{\theta,i}(x)
\partial_i\log p_0(x)dx\\
=&-
\int_\Omega
s_{\theta,i}(x)\partial_ip_0(x)dx.
\end{aligned}
$$

由带 boundary 的 integration by parts，

$$
\boxed{
-\int_\Omega s_{\theta,i}\partial_ip_0\,dx
=
-\int_{\partial\Omega}
p_0s_{\theta,i}n_i\,dS
+
\int_\Omega p_0\partial_is_{\theta,i}\,dx,
}
$$

其中 $n_i$ 是 outward normal 的第 $i$ 个坐标。只有当 surface term 为零时，才能写成

$$
-\mathbb E_{p_0}[s_\theta^\top s_0]
=
\mathbb E_{p_0}[\nabla\cdot s_\theta].
$$

于是得到 implicit score matching：

$$
\boxed{
\mathcal J_{\mathrm{ISM}}(\theta)
=
\mathbb E_{p_0}
\left[
\nabla\cdot s_\theta(X)
+
\frac12\|s_\theta(X)\|^2
\right],
}
$$

并且

$$
\mathcal J_{\mathrm{ESM}}(\theta)
=
\mathcal J_{\mathrm{ISM}}(\theta)
+C_{p_0},
$$

其中 $C_{p_0}$ 不依赖 $\theta$。两者因此有相同 minimizers。

### 3.2 被公式省略的 regularity conditions

足够条件随 domain 和函数类变化，但至少要检查：

- $p_0$ 可微，$s_\theta$ 对 $x$ component-wise 可微；
- 所需 expectations 与 derivatives 可积；
- 在 $\mathbb R^d$ 上，$p_0(x)s_\theta(x)$ 沿无穷远衰减得足够快；
- 在 bounded domain 上，必须显式给 boundary condition，不能默认 surface term 消失；
- derivative、integral 和 expectation 的交换合法。

若真实数据集中在低维 manifold 上，它相对 ambient Lebesgue measure 可能没有 ordinary density，$\nabla_x\log p_0(x)$ 便未必存在。有限数据的 empirical distribution 更是 delta masses 的和。这不是实现细节，而是从 raw data 直接使用 ambient score matching 时的定义问题。

### 3.3 ISM 已可训练，为什么还需要 DSM

ISM 只依赖 data samples 和 $s_\theta$，却要计算

$$
\nabla\cdot s_\theta(x)
=
\operatorname{tr}
\left(
\frac{\partial s_\theta}{\partial x}
\right).
$$

高维神经网络的 full Jacobian trace 昂贵；即使用 sliced/Hutchinson estimator 降低计算量，raw-data singular support 与有限样本局部几何仍然存在。Denoising score matching 采取另一条路线：先用已知 kernel 平滑数据，再让 kernel 自己提供监督 target。

***

## 4. DSM 中心恒等式：随机 conditional target 为什么学到 marginal score

设 clean variable 与 corruption kernel 为

$$
X\sim p_0,
\qquad
Y\mid X=x\sim K(\cdot\mid x).
$$

corrupted marginal 是

$$
p_K(y)
=
\int K(y\mid x)p_0(x)dx.
$$

虽然 $p_0$ 未知，kernel $K$ 是我们设计的，因此 conditional corruption score

$$
u(x,y)
=
\nabla_y\log K(y\mid x)
$$

可以计算。DSM 使用 risk

$$
\boxed{
\mathcal J_{\mathrm{DSM}}(s)
=
\frac12
\mathbb E_{X,Y}
\|s(Y)-u(X,Y)\|^2.
}
$$

表面上它在拟合一个依赖 clean $X$ 的随机 target；生成时模型却只收到 $Y$。关键是 squared-error regression 的 optimum 是 conditional mean。

### 4.1 第一步：conditional target 的均值就是 marginal score

假设可以把对 $y$ 的 differentiation 移入 integral：

$$
\begin{aligned}
\nabla_yp_K(y)
&=
\int p_0(x)\nabla_yK(y\mid x)dx\\
&=
\int p_0(x)K(y\mid x)
\nabla_y\log K(y\mid x)dx.
\end{aligned}
$$

两边除以 $p_K(y)>0$：

$$
\begin{aligned}
\nabla_y\log p_K(y)
&=
\int
\frac{p_0(x)K(y\mid x)}{p_K(y)}
u(x,y)dx\\
&=
\int p(x\mid y)u(x,y)dx.
\end{aligned}
$$

因此

$$
\boxed{
\mathbb E[u(X,Y)\mid Y=y]
=
\nabla_y\log p_K(y).
}
$$

需要准确区分等号两侧：

- $u(X,Y)$ 是给定某个 clean/noisy pair 后的 conditional kernel score；
- $\nabla_y\log p_K(Y)$ 是把未知 clean source 积分掉后的 noisy marginal score；
- 单个 $u(X,Y)$ 通常不等于 marginal score；
- 只有给定 $Y$ 后取 conditional expectation 才相等。

### 4.2 第二步：Pythagorean decomposition

记

$$
\bar u(Y)
=
\mathbb E[u(X,Y)\mid Y]
=
\nabla\log p_K(Y).
$$

在 DSM risk 中加减 $\bar u(Y)$：

$$
\begin{aligned}
\mathbb E\|s(Y)-u(X,Y)\|^2
=&
\mathbb E\|s(Y)-\bar u(Y)\|^2\\
&+
\mathbb E\|u(X,Y)-\bar u(Y)\|^2\\
&+
2\mathbb E[
(s-\bar u)^\top(\bar u-u)].
\end{aligned}
$$

cross term 为零，因为

$$
\begin{aligned}
&\mathbb E[
(s(Y)-\bar u(Y))^\top(\bar u(Y)-u(X,Y))]\\
=&
\mathbb E\left[
(s(Y)-\bar u(Y))^\top
\mathbb E[\bar u(Y)-u(X,Y)\mid Y]
\right]\\
=&0.
\end{aligned}
$$

所以

$$
\boxed{
\mathcal J_{\mathrm{DSM}}(s)
=
\frac12
\mathbb E_{p_K}
\|s(Y)-\nabla\log p_K(Y)\|^2
+C_K,
}
$$

其中

$$
C_K
=
\frac12\mathbb E
\|u(X,Y)-\mathbb E[u(X,Y)\mid Y]\|^2
$$

与 $s$ 无关。Vincent 2011 的 equivalence 因而不是“两个 losses 长得相似”，而是它们只差 irreducible conditional variance。

这就是本章最重要的答案：

> 网络不需要从 $Y$ 恢复那次随机 corruption；MSE 把相互冲突的 conditional targets 平均成了 noisy marginal 的 score。

***

## 5. Gaussian DSM：target、noise 与 score

取 additive Gaussian corruption：

$$
Y=X+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
$$

$$
K_\sigma(y\mid x)
=
(2\pi\sigma^2)^{-d/2}
\exp\left[
-\frac{\|y-x\|^2}{2\sigma^2}
\right].
$$

直接求导：

$$
\boxed{
\nabla_y\log K_\sigma(y\mid x)
=
\frac{x-y}{\sigma^2}.
}
$$

由于 $y=x+\sigma\epsilon$，也可写成

$$
\boxed{
\nabla_y\log K_\sigma(Y\mid X)
=
-\frac{\epsilon}{\sigma}.
}
$$

定义 Gaussian-smoothed marginal

$$
p_\sigma(y)
=
\int p_0(x)K_\sigma(y\mid x)dx
=
(p_0*\mathcal N(0,\sigma^2I))(y).
$$

DSM objective 为

$$
\boxed{
\mathcal L_{\mathrm{DSM}}(\theta;\sigma)
=
\frac12
\mathbb E
\left\|
s_\theta(X+\sigma\epsilon,\sigma)
+\frac{\epsilon}{\sigma}
\right\|^2.
}
$$

其 unrestricted population minimizer 是

$$
s_\theta^*(y,\sigma)
=
\nabla_y\log p_\sigma(y),
$$

不是 original $p_0$ 的 score。

把 loss 乘 $\sigma^2$ 可得

$$
\frac12
\mathbb E
\left\|
\sigma s_\theta(X+\sigma\epsilon,\sigma)
+\epsilon
\right\|^2.
$$

对单个固定 $\sigma$，positive scalar 不改变 population minimizer；但一个 shared network 同时拟合多个 $\sigma$ 时，每个 noise level 前的尺度会改变 finite-capacity optimization emphasis。

下图左侧把每个 sampled pair 的 conditional target 画成浅蓝点。给定相近 $y$ 时，它们仍然随机；黑点是分箱 conditional mean，红线是解析 marginal score。两者随样本增多而对齐，但任何单个浅蓝点通常都不在红线上。

![Gaussian DSM 的随机 conditional targets、marginal score 与 Tweedie posterior mean。](/images/diffusion/d3_dsm_tweedie.png)

***

## 6. Tweedie formula：score 就是 posterior-mean denoising correction

同一个 Gaussian corruption 还给出一条 finite-noise exact identity。对 noisy marginal 求导：

$$
\begin{aligned}
\nabla_yp_\sigma(y)
&=
\int p_0(x)K_\sigma(y\mid x)
\frac{x-y}{\sigma^2}dx\\
&=
\frac1{\sigma^2}
\left[
\int xp_0(x)K_\sigma(y\mid x)dx
-yp_\sigma(y)
\right].
\end{aligned}
$$

利用

$$
p(x\mid y)
=
\frac{p_0(x)K_\sigma(y\mid x)}{p_\sigma(y)},
$$

两边除以 $p_\sigma(y)$：

$$
\nabla_y\log p_\sigma(y)
=
\frac{\mathbb E[X\mid Y=y]-y}{\sigma^2}.
$$

整理得到

$$
\boxed{
\mathbb E[X\mid Y=y]
=
y+\sigma^2\nabla_y\log p_\sigma(y).
}
$$

这通常称为 Tweedie formula；Gaussian 形式在相关文献中也常与 Miyasawa 的早期结果联系。它把 score 的几何解释变成了精确的 Bayes denoising 解释：

$$
\underbrace{\sigma^2s_{p_\sigma}(y)}_{\text{denoising correction}}
=
\underbrace{\mathbb E[X\mid Y=y]-y}_{\text{posterior mean shift}}.
$$

### 6.1 posterior mean 不是逐样本 inverse

平方损失的一般分解为

$$
\begin{aligned}
\mathbb E\|X-r(Y)\|^2
=&
\mathbb E\|X-\mathbb E[X\mid Y]\|^2\\
&+
\mathbb E\|r(Y)-\mathbb E[X\mid Y]\|^2.
\end{aligned}
$$

所以 MSE-optimal denoiser 是

$$
r^*(y)=\mathbb E[X\mid Y=y].
$$

当 posterior $p(x\mid y)$ 不是 point mass 时，同一个 $y$ 对应多个可能 clean sources。posterior mean 是 Bayes estimator，不是把每次 corruption 唯一反演回原样本的函数。在 multimodal posterior 中，它甚至可能落在两个高密度解释之间的低密度处。

### 6.2 finite-noise Tweedie 与 small-noise DAE expansion

上面的 identity 对每个 $\sigma>0$ 精确成立，但它使用 smoothed density score

$$
\nabla\log p_\sigma(y).
$$

Alain & Bengio 2014 一类 denoising-autoencoder 结果在额外 smoothness assumptions 和 $\sigma\to0$ 下给出

$$
r_\sigma(x)
=
x+\sigma^2\nabla\log p_0(x)+o(\sigma^2).
$$

两种表述并排为

$$
\begin{array}{ll}
\text{finite }\sigma:&
r_\sigma(y)=y+\sigma^2\nabla\log p_\sigma(y),\\[3pt]
\sigma\to0:&
r_\sigma(x)=x+\sigma^2\nabla\log p_0(x)+o(\sigma^2).
\end{array}
$$

用 $p_0$ score 替换 finite-noise 公式中的 $p_\sigma$ score，一般是错的；两者只在适当小噪声极限下接近。

***

## 7. 一般 linear Gaussian channel：统一 score、noise 与 clean prediction

D1/D2 使用的 corruption 不只是 $Y=X+\sigma\epsilon$，而是

$$
Z=aX+b\epsilon,
\qquad
a>0,\quad b>0,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

其 conditional score 为

$$
\nabla_z\log p(z\mid x)
=
-\frac{z-ax}{b^2}.
$$

由第 4 节的 conditional-score identity：

$$
\boxed{
\nabla_z\log p_Z(z)
=
\mathbb E
\left[
\frac{aX-z}{b^2}
\middle|Z=z
\right].
}
$$

又因为 $\epsilon=(Z-aX)/b$，所以

$$
\boxed{
\mathbb E[\epsilon\mid Z=z]
=
-b\nabla_z\log p_Z(z).
}
$$

把第一条式子对 $\mathbb E[X\mid Z=z]$ 整理：

$$
\boxed{
\mathbb E[X\mid Z=z]
=
\frac{
z+b^2\nabla_z\log p_Z(z)
}{a}.
}
$$

因此同一个 conditional distribution 可用三种输出语义描述：

| network target   | population optimum            | 转成 marginal score      |
| ---------------- | ----------------------------- | ---------------------- |
| score $s_Z(z)$   | $\nabla_z\log p_Z(z)$         | identity               |
| noise $\epsilon$ | $\mathbb E[\epsilon\mid Z=z]$ | $-\epsilon^*(z)/b$     |
| clean data $X$   | $\mathbb E[X\mid Z=z]$        | $(a\hat x^*(z)-z)/b^2$ |

这些是 population-optimal functions 之间的 algebraic equivalence。它不表示三种 parameterizations 在 finite network、loss scaling、endpoint conditioning 和 numerical precision 上完全相同；这些 optimization design issues 留到 D5 系统讨论。

### 7.1 DDPM noise prediction 正是 score estimation

对 DDPM 时刻 $t$，令

$$
a_t=\sqrt{\bar\alpha_t},
\qquad
b_t=\sqrt{1-\bar\alpha_t},
$$

则

$$
X_t=a_tX_0+b_t\epsilon.
$$

这里 $q_t(x_t)$ 表示对 data distribution $q(x_0)$ 和 forward corruption 都积分后的 aggregate marginal：

$$
q_t(x_t)
=
\int q(x_0)q(x_t\mid x_0)dx_0.
$$

由一般 linear Gaussian identity：

$$
\boxed{
\nabla_{x_t}\log q_t(x_t)
=
-\frac{
\mathbb E[\epsilon\mid X_t=x_t]
}{\sqrt{1-\bar\alpha_t}}.
}
$$

D2 的 noise MSE

$$
\mathbb E
\|\epsilon-\epsilon_\theta(X_t,t)\|^2
$$

具有 Bayes optimum

$$
\epsilon_\theta^*(x_t,t)
=
\mathbb E[\epsilon\mid X_t=x_t,t].
$$

所以

$$
\boxed{
s_\theta(x_t,t)
=
-\frac{
\epsilon_\theta(x_t,t)
}{\sqrt{1-\bar\alpha_t}}
}
$$

是 $q_t$ marginal score estimator。这补上了 D2 第 12 节保留的理论接口。

对应 posterior-mean denoiser 为

$$
\boxed{
\mathbb E[X_0\mid X_t=x_t]
=
\frac{
x_t+(1-\bar\alpha_t)
\nabla_{x_t}\log q_t(x_t)
}{\sqrt{\bar\alpha_t}}.
}
$$

### 7.2 参数化转换也会转换 loss weight

若 $\epsilon_\theta=-b_ts_\theta$，则

$$
\begin{aligned}
\|\epsilon-\epsilon_\theta\|^2
&=
\|\epsilon+b_ts_\theta\|^2\\
&=
b_t^2
\left\|
-\frac{\epsilon}{b_t}-s_\theta
\right\|^2.
\end{aligned}
$$

因此 uniform raw $\epsilon$-MSE 等价于给 raw DSM score error 乘

$$
b_t^2=1-\bar\alpha_t.
$$

“noise predictor 可以转换为 score predictor”是函数层等价；“uniform noise MSE 与 uniform score MSE 是同一个 objective”则不成立。D2 的 VLB weight、这里的 parameterization scale 和 timestep sampling proposal 是三种不同层面的权重。

***

## 8. 为什么需要多个 noise levels

对任意 finite measure $p_0$，与 non-degenerate Gaussian 卷积后

$$
p_\sigma
=
p_0*\mathcal N(0,\sigma^2I),
\qquad
\sigma>0,
$$

在 $\mathbb R^d$ 上处处为正且 infinitely differentiable。因此 Gaussian smoothing：

- 让 finite empirical measure 或 low-dimensional support 获得 ambient density；
- 让 ambient score $\nabla\log p_\sigma$ 良好定义；
- 随 $\sigma$ 增大提高 separated modes 的 overlap；
- 提供从简单大噪声分布逐步回到数据附近的路径。

但 smoothing 也改变了 target distribution，并引入 bias。小 $\sigma$ 更接近数据，score target

$$
-\frac{\epsilon}{\sigma}
$$

的尺度和 variance 却会变大，有限样本下更难回归；大 $\sigma$ 容易连接 modes，却会抹去细节。单个 $\sigma$ 无法同时承担“全局移动”和“局部还原”。

![同一个 mixture 在不同 Gaussian noise scales 下的 density 与 score。大噪声增加 overlap，但 score target 也随之改变。](/images/diffusion/d3_smoothing_path.png)

这推动了 noise-conditioned score network。取一列由大到小的 noise levels：

$$
\sigma_1>\sigma_2>\cdots>\sigma_L>0,
$$

让同一个 network 接收 $(y,\sigma_i)$，拟合每个 smoothed marginal $p_{\sigma_i}$ 的 score：

$$
\boxed{
\mathcal L(\theta)
=
\frac1L\sum_{i=1}^L
\lambda(\sigma_i)
\mathbb E
\left[
\left\|
s_\theta(X+\sigma_i\epsilon,\sigma_i)
+\frac{\epsilon}{\sigma_i}
\right\|^2
\right].
}
$$

若 network function class 在每个 $\sigma_i$ 上不受限制，任意 strictly positive $\lambda(\sigma_i)$ 都保留 pointwise population minimizer。但在 shared finite network 中，weights 会改变各 noise levels 争夺 capacity 和 gradient budget 的方式。

### 8.1 objective weight 与 noise-level proposal 仍要分开

设目标明确包含 $\lambda_i$：

$$
\mathcal L
=
\frac1L\sum_i\lambda_i\mathbb E[\ell_i].
$$

若实际以 probability $r_i>0$ 抽取 noise level，则无偏 estimator 应为

$$
\widehat{\mathcal L}
=
\frac{\lambda_i\ell_i}{Lr_i}.
$$

因此：

- 改 $\lambda_i$：改变 optimization objective；
- 改 $r_i$ 并保留 importance correction：只改变 estimator；
- 改 $r_i$ 却不 correction：有效 objective 也被改变。

这与 D2 区分 timestep objective weight 和 sampling proposal 是同一原则。

***

## 9. 从 score 到 sampling：Langevin dynamics

学到 score 后怎样生成样本？最直接的桥梁是 overdamped Langevin dynamics。对 target density $p$，考虑

$$
\boxed{
dX_\tau
=
c\nabla\log p(X_\tau)d\tau
+
\sqrt{2c}\,dW_\tau,
\qquad c>0.
}
$$

这里用 $\tau$ 表示 sampler 的连续运行时间，避免与 diffusion noise index $t$ 混淆。其 Fokker--Planck equation 为

$$
\partial_\tau\rho
=
-\nabla\cdot(c\rho\nabla\log p)
+c\Delta\rho.
$$

代入 $\rho=p$：

$$
\begin{aligned}
-c\nabla\cdot(p\nabla\log p)+c\Delta p
&=
-c\nabla\cdot(\nabla p)+c\Delta p\\
&=0.
\end{aligned}
$$

所以 $p$ 是 stationary density。

这个计算只证明 stationarity，不自动证明从任意 initial distribution 收敛到 $p$。ergodicity、non-explosion、boundary behavior 和 mixing rate 都需要额外条件。

### 9.1 Euler--Maruyama 与 factor-of-two convention

离散化后可写为

$$
\boxed{
x_{k+1}
=
x_k+\alpha s_p(x_k)
+\sqrt{2\alpha}z_k,
\qquad
z_k\sim\mathcal N(0,I).
}
$$

NCSN 2019 论文的 Algorithm 1 写成

$$
x_{k+1}
=
x_k+\frac{\varepsilon}{2}s_p(x_k)
+\sqrt{\varepsilon}z_k.
$$

令 $\varepsilon=2\alpha$ 便完全相同。NCSN/NCSNv2 官方代码使用前一种形式。看到 drift 前的 $1/2$ 时不应立即判断谁“少乘了一倍”；必须同时检查 noise coefficient 和 step-size definition。

### 9.2 exact score 也不等于 finite-step exact sampling

即使 $s_p$ 完全准确，finite-step unadjusted Langevin algorithm 一般仍有 discretization bias；若 step 太大，chain 甚至可能不稳定。换成 learned score 后又增加 model drift error；若 modes 相隔很远，finite-time mixing 是第三个独立问题。

因此应分开：

1. target density 是否是 continuous dynamics 的 stationary distribution；
2. continuous dynamics 是否 ergodic、mixing 多快；
3. finite discretization 离 continuous dynamics 多远；
4. learned score 离 true score 多远；
5. initialization 是否落在 dynamics 能有效探索的区域。

***

## 10. NCSN：用 multi-noise score 解决 manifold 与 mixing 障碍

Song & Ermon 2019 将前面的数学组织成一个可扩展的生成框架。它面对的不是单一问题，而是互相耦合的三个障碍。

### 10.1 问题一：data score 可能没有良好定义

高维图像通常被认为集中在 ambient space 中的低维 structure 附近；empirical distribution 更是离散样本的和。在这种情况下直接学习 raw $p_0$ 的 ambient score 可能没有定义或数值不稳定。

**方案：** 对数据加入多个尺度的 Gaussian noise，学习处处 smooth 的 $p_{\sigma_i}$ scores。

### 10.2 问题二：低密度区域几乎不给训练信号

score matching 的 expectation 由 data density 加权。模型可在 high-density region 表现良好，却在 modes 之间的 low-density corridors 非常不准；sampling trajectory 恰好可能经过这些区域。disconnected-support 的 relative mass 甚至是 exact local-score non-identifiability。

**方案：** 大噪声 level 增加 components 的 overlap，让全局结构进入有概率质量的区域；随后逐渐减小噪声，恢复局部细节。

### 10.3 问题三：直接在小噪声 target 上 Langevin 难以跨 mode

若从 arbitrary noise initialization 直接使用接近 $p_0$ 的 sharp score，chain 可能困在错误 basin。

**方案：** annealed Langevin dynamics。从最大 $\sigma_1$ 开始，在较平滑的 landscape 上移动，再把上一层结果作为下一层 initialization：

$$
\sigma_1
\longrightarrow
\sigma_2
\longrightarrow
\cdots
\longrightarrow
\sigma_L.
$$

NCSN 2019 使用

$$
\lambda(\sigma_i)=\sigma_i^2,
$$

以抵消 Gaussian target $-\epsilon/\sigma_i$ 的典型 squared norm 随 $1/\sigma_i^2$ 增长的尺度，并在 sampler 中取

$$
\alpha_i
=
\eta\frac{\sigma_i^2}{\sigma_L^2}.
$$

这让 relative step size 随 noise scale 缩放，但它是 sampler design，不是由 DSM consistency 单独推出的唯一选择。

### 10.4 训练与采样算法

训练可以压缩成：

```text
repeat
    x ~ data
    i ~ Uniform({1, ..., L})
    epsilon ~ Normal(0, I)
    y = x + sigma[i] * epsilon
    target = -epsilon / sigma[i]
    loss = sigma[i]^2 * ||score_net(y, sigma[i]) - target||^2
    update theta
```

annealed Langevin 为：

```text
x ~ simple initialization
for i = 1, ..., L                 # sigma: large -> small
    alpha_i = eta * sigma[i]^2 / sigma[L]^2
    repeat M times
        z ~ Normal(0, I)
        x = x + alpha_i * score_net(x, sigma[i])
              + sqrt(2 * alpha_i) * z
return x
```

这两个 loops 使用同一列 noise scales，却承担不同职责：training 学每个 smoothed marginal 的 local field；sampling 试图沿这些 fields 逐层 transport 一个 initial distribution。

### 10.5 局限：smoothing 提供路径，但不证明路径可走完

Gaussian convolution 让 density positive、smooth，并提高 overlap；它并不自动保证：

- finite network 在 low-density transition region 准确；
- 每一层 Langevin 在有限 $M$ steps 内 equilibrate；
- 相邻 noise levels 足够 overlap；
- 最大噪声 target 与 chosen initialization 精确匹配；
- 最小 $\sigma_L>0$ 的 samples 已等于 $p_0$。

这些局限解释了为什么后续工作同时改进 network scale、noise schedule、sampler 和 endpoint denoising。

***

## 11. NCSNv2：从原理正确到更可扩展的 recipe

Song & Ermon 2020 的 improved techniques，即常说的 NCSNv2，不是改变 DSM 中心恒等式，而是修正原始 NCSN 在更高分辨率数据上的 scale 与 optimization 问题。

### 11.1 最大 noise 应与数据尺度匹配

**问题：** 若 $\sigma_1$ 太小，最大噪声 distribution 仍高度多模态，与 simple initialization 不匹配；若盲目复用不同数据集的 absolute scale，noise ladder 的含义会变化。

**方案：** 根据数据 pairwise distances 选择足以覆盖 data scale 的 initial noise，使大噪声 marginal 更接近单一 Gaussian-like cloud。

**边界：** “更 Gaussian-like”是设计目标，不是对任意 finite $\sigma_1$ 精确等于 isotropic Gaussian 的定理。

### 11.2 相邻 noise levels 需要足够 overlap

**问题：** $\sigma_i/\sigma_{i+1}$ 相差过大时，上一级 samples 可能落在下一级的低密度区域，annealing 接口断裂；levels 太密又增加 sampling cost。

**方案：** 用 Gaussian toy model 分析并调节 geometric ratio、steps 和 step size，使 adjacent perturbed distributions 有合理 overlap。

**边界：** toy Gaussian 的 calibration 不能证明真实 high-dimensional multimodal data 上的 optimality。

### 11.3 EMA 与 final denoise

NCSNv2 使用 exponential moving average parameters 改善 evaluation-time score stability。完成最小 noise level 的 Langevin steps 后，再应用

$$
\boxed{
x
\leftarrow
x+\sigma_L^2s_\theta(x,\sigma_L).
}
$$

若 score 精确，这正是 Tweedie formula 给出的

$$
\mathbb E[X_0\mid X_0+\sigma_L\epsilon=x].
$$

它把最小噪声 sample 映射到 posterior mean estimate，减少 residual corruption。但它不是从 $p_{\sigma_L}$ 到 $p_0$ 的 sample-wise exact transport；posterior mean 通常收缩 conditional variance，输出 distribution 未必精确等于 clean data distribution。

### 11.4 历史演进不是“DSM 被 DDPM 取代”

可把这一支的发展写成问题链：

| 时间     | 工作                       | 当时要解决的问题                               | 核心推进                                             | 仍留下什么                              |
| ------ | ------------------------ | -------------------------------------- | ------------------------------------------------ | ---------------------------------- |
| 2005   | Score Matching           | 未归一化 model 的估计                         | Fisher divergence 与 implicit objective           | trace、support 与 sampling           |
| 2011   | Denoising Score Matching | data score 不可直接监督                      | conditional score 与 marginal equivalence         | smoothing bias、noise-scale design  |
| 2014   | DAE analysis             | denoiser 学到的 vector field 是什么          | small-noise score expansion                      | finite-noise 层次需区分                 |
| 2019   | NCSN                     | manifold、low-density 与 Langevin mixing | multi-noise DSM + annealed Langevin              | scaling 与有限步稳定性                    |
| 2020   | NCSNv2                   | 高分辨率扩展                                 | data-scale noise、overlap、EMA、Tweedie denoise     | sampler 仍慢，理论条件仍强                  |
| 2020   | DDPM                     | 高质量 reverse-chain training             | noise prediction + variational route             | 与 score/SDE 的统一待展开                 |
| 2021   | Score SDE                | 离散 NCSN/DDPM 框架分裂                      | continuous-time reverse SDE 统一                   | time reversal 与 solver design      |
| 2023 起 | sampling theory          | 小 score loss 是否保证生成正确                  | initialization/discretization/score error bounds | assumptions 与 neural training 仍有距离 |

DDPM 与 NCSN 从不同目标和离散过程出发，却在第 7 节的 linear Gaussian identity 上相遇。下一章会在 continuous-time SDE 中把这次相遇写成统一的 reverse dynamics。

***

## 12. 配套代码：只用解析 mixture 检查概念

d3\_score\_matching.py（补充材料暂未公开） 不训练 U-Net，也不报告 FID。它用已知 Gaussian mixture 实现并检查：

- clean/smoothed mixture 的 analytic log density 与 score；
- Gaussian corruption 和 conditional DSM target；
- minimal noise-conditioned score interface；
- NCSN-style multi-noise DSM loss；
- exact Tweedie posterior mean；
- general linear Gaussian channel 的 noise/score mapping；
- 两种 Langevin step-size conventions；
- annealed Langevin skeleton 与 NCSNv2 final denoise。

运行：

```bash
# 本地验证脚本暂未公开
```

只执行 identity checks、不重画图片：

```bash
# 本地验证脚本暂未公开
```

### 12.1 DSM 的核心实现

代码中的本质只有四步：

```python
used_sigmas = sigmas[labels]
noisy = clean + used_sigmas[:, None] * noise
target = (clean - noisy) / used_sigmas[:, None].square()
prediction = model(noisy, used_sigmas)
loss = (
    used_sigmas.square()
    * (prediction - target).square().flatten(1).sum(1)
).mean()
```

其中 model(noisy, used\_sigmas) 可以是真实 noise-conditioned neural network；本章用 AnalyticMixtureScore 返回 exact $s_{p_\sigma}$，使公式验证不依赖训练成败。

### 12.2 为什么同时保留解析检查和 Monte Carlo 图

DSM 图中的 binned conditional mean 只是帮助直觉的 finite-sample visualization；真正 equality 由 analytic Gaussian-mixture formulas 检查。脚本还用 autograd 独立微分 log density，验证手写 score，并逐点比较：

$$
\mathbb E[X\mid Y=y]
\stackrel{?}{=}
y+\sigma^2s_{p_\sigma}(y),
$$

$$
\mathbb E[\epsilon\mid Z=z]
\stackrel{?}{=}
-b s_{p_Z}(z).
$$

因此图中的采样噪声不会被误当成理论误差。

***

## 13. 从小 DSM loss 到正确生成：中间还有哪些误差

population DSM consistency 只回答：在理想 function space 和 exact expectation 下，risk minimizer 是什么。真实 sampler 至少还经过

$$
\begin{aligned}
&\text{finite data}
\longrightarrow
\text{finite network}
\longrightarrow
\text{nonconvex optimization}\\
&\longrightarrow
\text{learned score field}
\longrightarrow
\text{finite-step dynamics}
\longrightarrow
\text{generated distribution}.
\end{aligned}
$$

可把误差进一步分层：

1. **Smoothing/endpoint bias**：最小 noise level 仍为 $\sigma_L>0$，或 continuous model 在 $t=0$ 前 early stop；
2. **Statistical error**：finite samples 无法给出 population expectation；
3. **Approximation error**：network class 无法表示所有 time/noise-conditioned scores；
4. **Optimization error**：训练没有到达该 function class 的 best risk；
5. **Low-density error**：$L^2(p_t)$ 很小仍允许 rare regions 中误差很大；
6. **Initialization mismatch**：最大噪声 marginal 与 sampling prior 不同；
7. **Discretization error**：finite steps 偏离 continuous dynamics；
8. **Finite-time mixing error**：即使 dynamics 正确，也没有运行到 equilibrium。

Chen et al. 2023 一类 sampling theory 在 score smoothness、moment、per-time $L^2$ score error 和 discretization 等 assumptions 下，把其中若干项组织成 distribution-distance bound。正确解读是：

> 在明确 assumptions 下，score error 可以传递为 sampling error；不是只要 empirical DSM loss 小，所有 assumptions 就自动满足。

### 13.1 为什么 $L^2(p_t)$ error 不等于 uniform vector-field accuracy

训练风险通常控制

$$
\mathbb E_{X_t\sim p_t}
\|s_\theta(X_t,t)-s_t(X_t)\|^2.
$$

若区域 $A$ 的 $p_t(A)$ 很小，即使 $A$ 内 pointwise error 很大，对 expectation 的贡献也可能很小。但 sampler trajectory 可能因 numerical error 或 inaccurate drift 进入 $A$，随后被错误 field 带走。因此 theorem 往往还需要 Lipschitz、moment、time-integrated error 或 stopping arguments，而不能只看一个 training scalar。

### 13.2 singular target 与 convergence metric

若 clean target 真正位于低维 manifold，而 sampler 在任意 positive noise/time 下具有 full-dimensional density，那么它与 singular target 的 total variation distance 可能始终最大，直到 endpoint 才发生奇异极限。此时 early stopping、Wasserstein 类 weaker metric 或先与 smoothed target 比较可能更合适。D12 将系统讨论 metric 与 theorem；本章只标记这个 boundary。

### 13.3 non-conservative learned field 会怎样

若 $s_\theta$ 不是 gradient field，把它放入

$$
dX=s_\theta(X)d\tau+\sqrt2dW
$$

仍定义了一个 stochastic dynamics（在适当 regularity 下），但其 stationary density 若存在，需要解 stationary Fokker--Planck PDE；不能简单宣称

$$
p_\theta(x)
\propto
\exp\left(\int s_\theta(x)dx\right),
$$

因为 line integral 可能依赖路径。DSM 的 true population target 是 conservative score；finite model approximation 未必严格保留这一结构。

***

## 14. 常见错误与适用边界

### 错误 1：把 data score 写成 parameter score

本章的 score 是 $\nabla_x\log p(x)$，不是 $\nabla_\theta\log p_\theta(x)$。

### 错误 2：说 score matching 直接给出了 normalized likelihood

它消除了 data-space derivative 中的 partition function，提供可训练 objective；normalization 和 exact likelihood 未因此自动可得。

### 错误 3：省略 ISM 的 boundary term

integration-by-parts identity 依赖 domain、decay 和 regularity conditions。在 bounded domain 或 heavy-tail setting 下不能默认 surface integral 为零。

### 错误 4：说 DSM target 就是 marginal score

$$
\frac{X-Y}{\sigma^2}
$$

是 conditional target。只有

$$
\mathbb E\left[
\frac{X-Y}{\sigma^2}
\middle|Y=y
\right]
=
\nabla_y\log p_\sigma(y)
$$

成立。

### 错误 5：说 DSM 学的是原始 data score

finite $\sigma$ 时学的是 smoothed marginal $p_\sigma$ 的 score。逼近 $p_0$ 需要小噪声极限及额外条件。

### 错误 6：把 Tweedie denoiser 当成精确去掉每次 noise

Tweedie 给 posterior mean，是 squared-error Bayes estimator，不是 sample-wise inverse，也不保留 posterior variance。

### 错误 7：混用 finite-noise identity 与 small-noise expansion

$$
y+\sigma^2\nabla\log p_\sigma(y)
$$

是 finite-noise exact formula；用 $p_0$ score 的式子一般只在 $\sigma\to0$ 下渐近成立。

### 错误 8：参数化能互换，所以 losses 完全相同

score/noise/clean outputs 可 algebraically 转换，但平方损失会引入 $a_t,b_t$ dependent scaling；finite optimization measure 可能不同。

### 错误 9：Langevin drift 前差一个 $1/2$ 就是实现错误

必须同时比较 noise coefficient。$x+\alpha s+\sqrt{2\alpha}z$ 与 $x+\varepsilon s/2+\sqrt\varepsilon z$ 在 $\varepsilon=2\alpha$ 下相同。

### 错误 10：Gaussian smoothing 保证 sampler 一定跨 mode

smoothing 改善 support 与 overlap；finite network accuracy、相邻层接口和 finite-time mixing 仍是独立问题。

### 错误 11：小 DSM loss 就是 end-to-end convergence theorem

从 score risk 到 generated distribution 还需要 initialization、regularity、discretization、time truncation 和 metric assumptions。

***

## 15. 本章给后续章节准备了什么

D3 已把 DDPM 的 noise regression 翻译成 noisy marginal score estimation：

$$
x_t
\longrightarrow
\epsilon_\theta(x_t,t)
\longleftrightarrow
s_\theta(x_t,t)
\longleftrightarrow
\mathbb E[x_0\mid x_t].
$$

同时，Langevin dynamics 说明 score 能作为 stochastic sampling drift。但 NCSN 使用一列离散 Gaussian smoothing levels，DDPM 使用 VP-style discrete Markov chain；两者目前仍像不同算法。

[D4](/blog/diffusion/d4-continuous-time-sde/) 将进入 continuous time：

- forward SDE 如何同时包含 VE/NCSN 与 VP/DDPM；
- reverse-time drift 为什么出现 marginal score；
- stochastic reverse SDE 与 deterministic probability-flow ODE 为什么能共享 marginals；
- reverse-time theorem 需要哪些严格条件。

D5 再系统处理 $\epsilon,x_0,v,$ score parameterizations 与 loss weights；D6 讨论 Langevin、ancestral、DDIM 和 modern solvers；D12 回到本章只概述的 score-error sampling theory。

***

## 16. 章节小结

- data-space score 是 $\nabla_x\log p(x)$，描述样本空间中 log density 的局部上升方向；
- score 消除与 $x$ 无关的 partition function，但不自动给出 normalized likelihood；
- 在 connected positive-support domain 上，score 加 normalization 可确定 density；disconnected components 的 relative masses 不由 local score 单独决定；
- explicit Fisher divergence 经过带 boundary term 的 integration by parts，得到不含 unknown data score 的 implicit score matching；
- DSM 用已知 corruption kernel 的 conditional score 作监督，其 conditional expectation 精确等于 corrupted marginal score；
- DSM risk 与 noisy marginal 上的 Fisher divergence 只差不依赖模型的 conditional variance；
- Gaussian DSM target 是 $(X-Y)/\sigma^2=-\epsilon/\sigma$，而 population predictor 是 $\nabla\log p_\sigma(Y)$；
- Tweedie formula 给出 $\mathbb E[X\mid Y=y]=y+\sigma^2\nabla\log p_\sigma(y)$，它是 posterior mean 而不是逐样本 inverse；
- finite-noise Tweedie 使用 $p_\sigma$ score，small-noise DAE expansion 才出现 $p_0$ score；
- 对 linear Gaussian channel，score、noise conditional mean 与 clean conditional mean 可精确转换；DDPM noise prediction 因而是 $q_t$ score estimation；
- 多 noise levels 在 global overlap 与 local detail 之间搭桥，NCSN 用 multi-noise DSM 和 annealed Langevin 把它变成生成算法；
- NCSNv2 改善 data-scale noise、相邻 level overlap、EMA 与 final Tweedie denoise，但不改变 DSM 的 population identity；
- exact stationary-density calculation、finite-time mixing、discretization 和 learned-score error 是不同层次；小 DSM loss 本身不是无条件 sampling convergence theorem。

***

## 17. 思考题

1. **disconnected support 与 mixture weight。** 设 $p_1,p_2$ supports 不相交。证明 $p_\pi=\pi p_1+(1-\pi)p_2$ 在每个 component 内的 score 与 $\pi$ 无关。加入任意 $\sigma>0$ Gaussian noise 后，为什么这个 exact non-identifiability 消失？这是否意味着 finite-sample estimation 和 finite-time mixing 也自动解决？

2. **DSM target variance。** 对 $Y=X+\sigma\epsilon$，利用 conditional variance decomposition 分析

   $$
   -\frac{\epsilon}{\sigma}
   $$

   在给定 $Y=y$ 后的 irreducible variance 如何随 $\sigma$ 变化。为什么 $\lambda(\sigma)=\sigma^2$ 只能校正一部分 scale 问题？

3. **boundary counterexample。** 在 bounded interval 上选择一个不满足 zero-flux boundary 的 $p_0,s_\theta$，显式计算 ESM cross term、ISM divergence term 与 boundary term，说明省略 surface term 会怎样改变 objective。

4. **Tweedie 不是 transport。** 构造一个 symmetric two-component posterior，使 posterior mean 落在两个 modes 之间。若对 $Y\sim p_\sigma$ 逐点应用 Tweedie denoiser，为什么输出 distribution 一般不等于 $p_0$？

5. **parameterization 与 weighting。** 从

   $$
   X_t=a_tX_0+b_t\epsilon
   $$

   推出 score、noise 与 $x_0$ prediction 三种 squared errors 的相互 scale。构造一个 shared scalar model，使不同 noise-level weights 给出不同 finite-model optimum。

6. **non-conservative field。** 在二维构造一个 curl 非零的 vector field。它为什么不能成为 positive smooth density 的 score？若仍将其作为 Langevin drift，stationary Fokker--Planck equation 会要求什么，而不能直接写出什么？

7. **stationarity 与 convergence。** 本章把 $\rho=p$ 代入 Fokker--Planck 只证明了什么？给出至少三项还需验证的条件或误差来源，才能从 arbitrary initialization 得到 finite-step approximate sampling guarantee。

8. **low-density error。** 构造一列 vector fields，使其 $L^2(p)$ error 趋于零，但在一列 probability mass 趋于零的 regions 上 pointwise error 发散。讨论 sampler 是否可能进入这些 regions，以及还需要哪类 regularity control。

9. **annealing ladder。** 若相邻 $\sigma_i,\sigma_{i+1}$ 相差极大，会同时影响 initialization mismatch、score geometry 和 finite mixing。尝试给出一个一维 Gaussian-mixture criterion 来量化“足够 overlap”，并说明它为何不能直接升级为真实图像分布上的 theorem。

***

## 18. 本章来源与继续阅读

核心定义、等价性与历史结论优先回到原论文；作者 blog 只承担教学构图和统一叙事职责。

1. Aapo Hyvärinen. *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR, 2005. 本地论文（补充材料暂未公开）；结构化笔记（补充材料暂未公开）。
2. Pascal Vincent. *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation, 2011. 作者托管论文（补充材料暂未公开）；[结构化笔记](https://doi.org/10.1162/NECO_a_00142 "官方论文页面")。
3. Guillaume Alain, Yoshua Bengio. *What Regularized Auto-Encoders Learn from the Data-Generating Distribution*. JMLR, 2014. 本地论文（补充材料暂未公开）；章节笔记（补充材料暂未公开）。
4. Bradley Efron. *Tweedie's Formula and Selection Bias*. JASA, 2011. PMC HTML 快照（补充材料暂未公开）；[章节笔记](https://doi.org/10.1198/jasa.2011.tm11181 "官方论文页面")。
5. Saeed Saremi, Aapo Hyvärinen. *Neural Empirical Bayes*. JMLR, 2019. 本地论文（补充材料暂未公开）；章节笔记（补充材料暂未公开）。
6. Yang Song, Stefano Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS, 2019. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/1907.05600 "官方论文页面")。
7. Yang Song, Stefano Ermon. *Improved Techniques for Training Score-Based Generative Models*. NeurIPS, 2020. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2006.09011 "官方论文页面")。
8. Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2006.11239 "官方论文页面")。
9. Sitan Chen et al. *Sampling is as Easy as Learning the Score: Theory for Diffusion Models with Minimal Data Assumptions*. ICLR, 2023. 本地论文（补充材料暂未公开）；[章节笔记](https://arxiv.org/abs/2209.11215 "官方论文页面")。
10. Yang Song. *Generative Modeling by Estimating Gradients of the Data Distribution*, author blog, 2021. 本地快照（补充材料暂未公开）；使用边界笔记（补充材料暂未公开）。
11. Sander Dieleman. *Diffusion Models Are Autoencoders*. 2022. 本地快照（补充材料暂未公开）；符号与使用边界（补充材料暂未公开）。只用于 denoising/feature-scale 直觉，不作为 score scaling 公式来源。

完整独立代数复核见 score matching / DSM derivation ledger（补充材料暂未公开），解析 Gaussian-mixture 数值复核见 score\_identity\_checks.py（补充材料暂未公开）。NCSN/NCSNv2 官方实现的固定 commit 与逐文件映射见 code provenance（补充材料暂未公开）。本章 claim-evidence mapping 见 D3 chapter source packet（补充材料暂未公开）。
