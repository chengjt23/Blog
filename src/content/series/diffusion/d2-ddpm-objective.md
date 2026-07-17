---
title: DDPM：从逆过程到训练目标
description: 推导 reverse Gaussian parameterization、前向 posterior、ELBO 分解与噪声预测 MSE 的来源和边界。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
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
scope: 覆盖 DDPM 逆过程、逐步 KL、方差选择和简化目标，不把简化损失等同于完整 likelihood。
---
D1 做了一件看似与生成相反的事：从数据 $x_0$ 出发，沿一条已知 Gaussian Markov chain 逐渐加入噪声。现在我们终于转身，尝试从 $x_T$ 一步步走回 $x_0$。

真正的困难也从这里开始。forward transition

$$
q(x_t\mid x_{t-1})
$$

是我们自己设计的，当然已知；但生成需要的 reverse conditional

$$
q(x_{t-1}\mid x_t)
$$

还要对未知数据分布进行 Bayes 反演，因此不能直接计算。理解 DDPM 的核心不能停在一句“让网络预测噪声”，而要掌握下面这条完整链条及其历史来源：

$$
\boxed{
\text{未知 reverse}
\longrightarrow
\text{可解析的 conditioned posterior}
\longrightarrow
\text{path ELBO}
\longrightarrow
\text{逐步 Gaussian KL}
\longrightarrow
\text{带权 noise MSE}
\longrightarrow
L_{\mathrm{simple}}
}
$$

这条链上最后一个箭头尤其需要警惕：$L_{\mathrm{simple}}$ 是 DDPM 有意选择的重加权 surrogate，不是把原 ELBO “等价化简”后自然剩下的同一个目标。

读完本章后，我们应当能够回答：

1. 为什么 $q(x_{t-1}\mid x_t)$ 未知，而 $q(x_{t-1}\mid x_t,x_0)$ 可解析；
2. 为什么训练时可以使用带 $x_0$ 的 posterior，而生成时网络不需要 $x_0$；
3. path ELBO 怎样分成 prior、逐步 KL 与 decoder 三类项；
4. noise prediction 的 MSE 权重究竟从哪里来；
5. $L_{\mathrm{simple}}$、完整 VLB、hybrid loss 和 timestep importance sampling 分别改变了什么；
6. 训练算法中的每个张量怎样映射回公式；
7. 为什么最后一步不能继续加入 Gaussian noise。

***

## 1. 逆过程为什么仍然未知

沿用 D1 的记号：

$$
q(x_{1:T}\mid x_0)
=\prod_{t=1}^{T}q(x_t\mid x_{t-1}),
$$

$$
q(x_t\mid x_{t-1})
=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_tI),
\qquad
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

任意时刻的 direct marginal 为

$$
q(x_t\mid x_0)
=\mathcal N\!\left(
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
$$

如果已知 aggregate density $q_{t-1}(x_{t-1})$，Bayes rule 会给出

$$
q(x_{t-1}\mid x_t)
=\frac{
q(x_t\mid x_{t-1})q_{t-1}(x_{t-1})
}{
q_t(x_t)
}.
$$

问题在于 $q_{t-1}$ 和 $q_t$ 都由未知数据分布

$$
q_{\mathrm{data}}(x_0)
$$

经 forward process 推送而来。能够从数据集抽样，不等于能够在任意点精确计算其密度，更不等于可以解析地完成上面的 Bayes 反演。

### 1.1 三个 reverse 对象必须分开

本章反复使用三个看起来相似、角色完全不同的分布：

| 对象                          |   是否已知 | 是否在生成时使用 | 角色                       |
| --------------------------- | -----: | -------: | ------------------------ |
| $q(x_{t-1}\mid x_t)$        |   通常未知 |     理想目标 | 真实无条件 reverse            |
| $q(x_{t-1}\mid x_t,x_0)$    |  训练时解析 |        否 | posterior teacher        |
| $p_\theta(x_{t-1}\mid x_t)$ | 通过学习得到 |        是 | reverse generative model |

把第二个对象简称成“真实 reverse”很容易造成误解。它是真实 **conditioned posterior**，不是生成时可以调用的无条件 reverse。

### 1.2 无条件 reverse 为什么通常不是单个 Gaussian

对 $x_0$ 积分：

$$
\boxed{
q(x_{t-1}\mid x_t)
=\int
q(x_{t-1}\mid x_t,x_0)
q(x_0\mid x_t)\,dx_0.
}
$$

后文会证明 integrand 中第一个因子是 Gaussian；但其均值依赖 $x_0$。因此上式一般是由不同均值 Gaussian 组成的 mixture，而 mixture 通常不是单个 Gaussian。

下面的两步一维反例故意使用较大的 $\beta_1=\beta_2=0.2$，以便清楚展示有限大步下的现象。数据只取 $x_0\in\{-2,2\}$，观察到 $x_2=0$ 时，两个 conditioned posterior 都是 Gaussian，但对 $x_0$ 积分后的无条件 reverse 明显是双峰分布：

![Conditioned Gaussian posteriors and their unconditional reverse mixture](/images/diffusion/d2_reverse_conditionals.png)

这张图不是在模拟典型的 1000 步 DDPM，而是在给出一个反例：

> “forward transition 是 Gaussian”与“conditioned posterior 是 Gaussian”都不能推出“任意有限步的无条件 reverse 精确是一个 Gaussian”。

Sohl-Dickstein et al. 讨论了小步或连续极限下 reverse kernel 保持同类形式的依据；DDPM 则直接选择 Gaussian $p_\theta$ 作为可计算的 reverse model。小 $\beta_t$ 使这个局部近似更合理，但不会让未知 aggregate reverse 在任意有限步上自动解析（Sohl-Dickstein et al., 2015, Sec. 2.2；Ho et al., 2020, Sec. 2）。

***

## 2. Reverse generative model：真正负责生成的链

DDPM 定义

$$
\boxed{
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t)
}
$$

并通常取

$$
p(x_T)=\mathcal N(0,I),
$$

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N\!\left(
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

forward $q$ 与 reverse $p_\theta$ 的职责不能颠倒：

- $q$ 是训练时人为规定的 inference/corruption proposal；
- $p_\theta$ 是从 prior 出发生成数据的概率模型；
- $q$ 不含需要学习的去噪网络；
- $p_\theta$ 的误差决定最终生成分布是否接近数据。

若 $\Sigma_\theta=\sigma_{\mathrm{rev},t}^2I$，模型只需输出均值，或输出能确定均值的等价量。这里特意写 $\sigma_{\mathrm{rev},t}^2$，避免把 reverse variance 与 D1 中 cumulative forward noise $b(t)^2$ 混淆。Ho et al. 原文把前者记为 $\sigma_t^2$。

### 2.1 生成不是“把 forward 公式倒过来”

forward sample 写成

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t.
$$

机械移项会得到

$$
x_{t-1}
=\frac{x_t-\sqrt{\beta_t}\epsilon_t}{\sqrt{\alpha_t}}.
$$

但生成时我们不知道 forward 中实际使用的 $\epsilon_t$，而且给定 $x_t$ 时它与 $x_{t-1}$ 并不独立。这个代数恒等式不是 reverse conditional。

可学习的做法是：让网络从 $x_t,t$ 中预测与 reverse mean 等价的统计量，再按照 $p_\theta(x_{t-1}\mid x_t)$ 采样。

***

## 3. 训练时的突破口：给定 $x_0$ 后 posterior 可解析

训练数据提供了 $x_0$。一旦把它加入条件，Markov property 给出

$$
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})
q(x_{t-1}\mid x_0).
$$

右侧两项都是关于 $x_{t-1}$ 的 Gaussian：

$$
q(x_t\mid x_{t-1})
=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_tI),
$$

$$
q(x_{t-1}\mid x_0)
=\mathcal N\!\left(
\sqrt{\bar\alpha_{t-1}}x_0,
(1-\bar\alpha_{t-1})I
\right).
$$

Gaussian 相乘仍是 Gaussian，因此

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=\mathcal N\!\left(
\tilde\mu_t(x_t,x_0),
\tilde\beta_tI
\right).
}
$$

下面不跳过配方步骤。

### 3.1 把 exponent 写成关于 $x_{t-1}$ 的二次型

忽略与 $x_{t-1}$ 无关的常数，

$$
\begin{aligned}
\log q(x_{t-1}\mid x_t,x_0)
={}&
-\frac{1}{2\beta_t}
\left\|x_t-\sqrt{\alpha_t}x_{t-1}\right\|^2\\
&-\frac{1}{2(1-\bar\alpha_{t-1})}
\left\|x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0\right\|^2
+C.
\end{aligned}
$$

展开后，$x_{t-1}$ 的 quadratic coefficient，也就是 precision，为

$$
\tilde\beta_t^{-1}
=
\frac{\alpha_t}{\beta_t}
+\frac{1}{1-\bar\alpha_{t-1}}.
$$

通分并使用

$$
\bar\alpha_t=\alpha_t\bar\alpha_{t-1},
\qquad
\beta_t=1-\alpha_t,
$$

得到

$$
\begin{aligned}
\tilde\beta_t^{-1}
&=
\frac{
\alpha_t(1-\bar\alpha_{t-1})+\beta_t
}{
\beta_t(1-\bar\alpha_{t-1})
}\\
&=
\frac{
1-\alpha_t\bar\alpha_{t-1}
}{
\beta_t(1-\bar\alpha_{t-1})
}\\
&=
\frac{1-\bar\alpha_t}
{\beta_t(1-\bar\alpha_{t-1})}.
\end{aligned}
$$

所以

$$
\boxed{
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}
{1-\bar\alpha_t}\beta_t.
}
$$

### 3.2 information vector 给出 posterior mean

exponent 中关于 $x_{t-1}$ 的 linear coefficient 为

$$
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}
{1-\bar\alpha_{t-1}}x_0.
$$

Gaussian mean 等于 covariance 乘 information vector：

$$
\tilde\mu_t
=\tilde\beta_t
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}
{1-\bar\alpha_{t-1}}x_0
\right).
$$

代入 $\tilde\beta_t$ 并整理：

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

这就是训练时的 posterior teacher。其均值是 $x_0$ 与 $x_t$ 的线性组合，权重由 schedule 完全确定。

### 3.3 三个 sanity checks

第一，$\tilde\beta_t<\beta_t$ 对 $t>1$ 成立，因为

$$
1-\bar\alpha_{t-1}<1-\bar\alpha_t.
$$

额外知道 $x_0$ 后，不确定性应当变小。

第二，当 $\beta_t$ 很小时，posterior mean 接近 $x_t$，符合“局部 reverse step”直觉。

第三，在 $t=1$：

$$
\bar\alpha_0=1,
\qquad
\tilde\beta_1=0,
\qquad
\tilde\mu_1=x_0.
$$

这不是实现错误。给定 $x_1,x_0$ 后，“前一时刻”就是已知的 $x_0$，posterior 自然退化成 point mass。也正因为如此，$t=1$ 必须作为 decoder likelihood $L_0$ 处理，不能把 $\log0$ 直接送入普通 Gaussian KL。

### 3.4 为什么用 conditioned teacher 能学到 unconditional reverse

下面是本文基于 KL chain rule 给出的补充推导。假设相关 regular conditional densities 存在；固定 $x_t$，令 $y=x_{t-1}$。有

$$
\begin{aligned}
&\mathbb E_{q(x_0\mid x_t)}
D_{\mathrm{KL}}\!\left(
q(y\mid x_t,x_0)
\Vert p_\theta(y\mid x_t)
\right)\\
={}&
\mathbb E_{q(x_0,y\mid x_t)}
\log\frac{q(y\mid x_t,x_0)}{p_\theta(y\mid x_t)}.
\end{aligned}
$$

在 log ratio 中乘除 $q(y\mid x_t)$：

$$
\log\frac{q(y\mid x_t,x_0)}{p_\theta(y\mid x_t)}
=
\log\frac{q(y\mid x_t,x_0)}{q(y\mid x_t)}
+
\log\frac{q(y\mid x_t)}{p_\theta(y\mid x_t)}.
$$

分别取期望：

$$
\boxed{
\begin{aligned}
&\mathbb E_{q(x_0\mid x_t)}
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\Vert p_\theta(x_{t-1}\mid x_t)
\right)\\
={}&
I(X_{t-1};X_0\mid X_t=x_t)
+
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t)
\Vert p_\theta(x_{t-1}\mid x_t)
\right).
\end{aligned}
}
$$

第一项不依赖 $\theta$。因此，如果 $p_\theta$ 的分布族足够灵活，最小化 expected conditioned KL 与逼近未知 unconditional reverse 具有同一个最优解。

这解释了 DDPM 最容易令人困惑的一点：

> $x_0$ 只在训练时构造可计算的 teacher；生成时模型仍然只接收 $x_t,t$。

不过，如果 $p_\theta$ 被限制为单个 diagonal Gaussian，它只能在受限模型族中寻找最佳近似。由 total variance，

$$
\operatorname{Var}(X_{t-1}\mid X_t)
=
\tilde\beta_tI
+
\operatorname{Var}\!\left(
\tilde\mu_t(X_t,X_0)\mid X_t
\right).
$$

第二项来自对不同 $x_0$ 的 mixture uncertainty，通常不是 isotropic。这也是 fixed reverse variance 只是一种建模选择，而不是精确真值的另一个证明。

***

## 4. 从数据 likelihood 到 path ELBO

现在有了 posterior teacher，但还缺一个原则来说明为什么要逐步匹配它。这个原则来自 maximum likelihood。

我们希望最大化

$$
\log p_\theta(x_0)
=
\log\int p_\theta(x_{0:T})\,dx_{1:T}.
$$

直接对所有 latent states 积分通常不可行。插入已知 forward path proposal：

$$
\begin{aligned}
\log p_\theta(x_0)
&=
\log\int
q(x_{1:T}\mid x_0)
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\,dx_{1:T}\\
&=
\log
\mathbb E_{q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right].
\end{aligned}
$$

$\log$ 是 concave function，由 Jensen inequality：

$$
\boxed{
\log p_\theta(x_0)
\ge
\mathbb E_q
\left[
\log p_\theta(x_{0:T})
-\log q(x_{1:T}\mid x_0)
\right].
}
$$

右侧是 evidence lower bound，简称 ELBO。定义 negative ELBO

$$
\mathcal L_{\mathrm{vlb}}(x_0)
=
-\mathbb E_q
\left[
\log p_\theta(x_{0:T})
-\log q(x_{1:T}\mid x_0)
\right],
$$

便有

$$
-\log p_\theta(x_0)
\le \mathcal L_{\mathrm{vlb}}(x_0).
$$

所以最小化 $\mathcal L_{\mathrm{vlb}}$ 是在最小化 negative log-likelihood 的一个 upper bound。

### 4.1 forward path 的反向 factorization

forward chain 原本写成

$$
q(x_{1:T}\mid x_0)
=\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
$$

为了与 reverse model 对齐，需要把它反向 factorize。对 $t\ge2$，Bayes rule 给出

$$
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)
=
q(x_{t-1}\mid x_t,x_0)q(x_t\mid x_0).
$$

所以

$$
q(x_t\mid x_{t-1})
=
q(x_{t-1}\mid x_t,x_0)
\frac{q(x_t\mid x_0)}
{q(x_{t-1}\mid x_0)}.
$$

把 $t=2,\ldots,T$ 的式子相乘，marginal ratio 望远镜消去，再乘第一步 $q(x_1\mid x_0)$：

$$
\boxed{
q(x_{1:T}\mid x_0)
=
q(x_T\mid x_0)
\prod_{t=2}^{T}
q(x_{t-1}\mid x_t,x_0).
}
$$

这一步是 ELBO 能够拆成逐步 posterior KL 的关键。

### 4.2 完整分解

将

$$
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
$$

与上面的 reverse factorization 一起代入 negative ELBO。把 $t=1$、$t=2,\ldots,T$ 和 terminal 项分组，得到

$$
\boxed{
\mathcal L_{\mathrm{vlb}}
=L_T+\sum_{t=2}^{T}L_{t-1}+L_0.
}
$$

三类项分别是

$$
\boxed{
L_T
=
D_{\mathrm{KL}}\!\left(
q(x_T\mid x_0)\Vert p(x_T)
\right),
}
$$

$$
\boxed{
L_{t-1}
=
\mathbb E_{q(x_t\mid x_0)}
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\Vert
p_\theta(x_{t-1}\mid x_t)
\right),
\quad t\ge2,
}
$$

$$
\boxed{
L_0
=
\mathbb E_{q(x_1\mid x_0)}
\left[-\log p_\theta(x_0\mid x_1)\right].
}
$$

Sohl-Dickstein et al. 2015 使用 trajectory ratio、conditional entropy 与 prior entropy 组织端点；Ho et al. 2020 Appendix A 使用今天更常见的 $L_T+\sum L_{t-1}+L_0$ 写法。二者来自同一个 path variational argument，不应被写成两个互不相关的目标（Sohl-Dickstein et al., 2015, Eq. 14 and App. B；Ho et al., 2020, Eq. 5 and App. A）。

### 4.3 三类项各自训练什么

**Terminal prior term $L_T$.** forward schedule 固定时，它不依赖 $\theta$，所以不训练 reverse network。但“不训练网络”不等于“数值为零”。若 $p(x_T)=\mathcal N(0,I)$，则

$$
\begin{aligned}
L_T
={}&
\frac12\left[
\bar\alpha_T\|x_0\|^2
-d\bar\alpha_T
-d\log(1-\bar\alpha_T)
\right].
\end{aligned}
$$

只有 terminal signal 足够小或精确为零时，这一项才接近或等于零。

**Intermediate terms $L_{t-1}$.** 这些项直接比较 posterior teacher 与 learned reverse transition，是 DDPM 训练目标的主体。

**Decoder term $L_0$.** 它描述从 $x_1$ 恢复观测 $x_0$ 的 likelihood。图像像素离散时，严格 likelihood 实现通常使用 discretized Gaussian decoder；它不是把 $\tilde\beta_1=0$ 代入普通 Gaussian KL。

### 4.4 ELBO 的 gap 是什么

从标准 variational identity 还可写成

$$
\log p_\theta(x_0)
=
\operatorname{ELBO}(x_0)
+
D_{\mathrm{KL}}\!\left(
q(x_{1:T}\mid x_0)
\Vert
p_\theta(x_{1:T}\mid x_0)
\right).
$$

因此 bound gap 是 forward proposal 与生成模型真实 latent posterior 的 path-space KL。即使每个 training loss 都能计算，ELBO 仍可能不紧；maximum likelihood、variational bound 与 sample quality 也不是三个自动等价的评价对象。

***

## 5. Gaussian KL 怎样变成回归

对 $t\ge2$，teacher 为

$$
q(x_{t-1}\mid x_t,x_0)
=\mathcal N(\tilde\mu_t,\tilde\beta_tI).
$$

先考虑固定 reverse covariance：

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N(
\mu_\theta(x_t,t),
\sigma_{\mathrm{rev},t}^2I
).
$$

对两个 $d$-dimensional isotropic Gaussian，

$$
\begin{aligned}
D_{\mathrm{KL}}
\big(
\mathcal N(m_q,s_q^2I)
\Vert
\mathcal N(m_p,s_p^2I)
\big)
=\frac12\Bigg[
&d\frac{s_q^2}{s_p^2}
+\frac{\|m_p-m_q\|^2}{s_p^2}\\
&-d+d\log\frac{s_p^2}{s_q^2}
\Bigg].
\end{aligned}
$$

当 $s_p^2=\sigma_{\mathrm{rev},t}^2$ 固定时，只有

$$
\boxed{
\frac{1}{2\sigma_{\mathrm{rev},t}^2}
\|\tilde\mu_t-\mu_\theta\|^2
}
$$

依赖 mean network。于是 KL training 变成了 posterior mean regression。

### 5.1 用 forward noise 重写真实 posterior mean

D1 的 reparameterization 是

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

解出 $x_0$：

$$
x_0
=
\frac{
x_t-\sqrt{1-\bar\alpha_t}\epsilon
}{
\sqrt{\bar\alpha_t}
}.
$$

代入 posterior mean

$$
\tilde\mu_t
=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t.
$$

第一项变为

$$
\frac{\beta_t}
{\sqrt{\alpha_t}(1-\bar\alpha_t)}
\left(
x_t-\sqrt{1-\bar\alpha_t}\epsilon
\right),
$$

因为

$$
\frac{\sqrt{\bar\alpha_{t-1}}}
{\sqrt{\bar\alpha_t}}
=\frac{1}{\sqrt{\alpha_t}}.
$$

把两个 $x_t$ coefficient 合并：

$$
\begin{aligned}
&\frac{\beta_t}
{\sqrt{\alpha_t}(1-\bar\alpha_t)}
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}\\
={}&
\frac{
\beta_t+\alpha_t(1-\bar\alpha_{t-1})
}{
\sqrt{\alpha_t}(1-\bar\alpha_t)
}\\
={}&
\frac{1-\bar\alpha_t}
{\sqrt{\alpha_t}(1-\bar\alpha_t)}
=\frac1{\sqrt{\alpha_t}}.
\end{aligned}
$$

所以

$$
\boxed{
\tilde\mu_t
=
\frac1{\sqrt{\alpha_t}}
\left(
x_t
-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon
\right).
}
$$

这一步说明：在训练样本中，posterior mean 与 forward noise $\epsilon$ 一一对应。

### 5.2 让网络预测 noise

DDPM 用同样的 algebraic form 定义 model mean：

$$
\boxed{
\mu_\theta(x_t,t)
=
\frac1{\sqrt{\alpha_t}}
\left(
x_t
-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
}
$$

二者之差为

$$
\tilde\mu_t-\mu_\theta
=
\frac{\beta_t}
{\sqrt{\alpha_t(1-\bar\alpha_t)}}
\left(
\epsilon_\theta-\epsilon
\right).
$$

因此，单个 KL 中与 mean network 有关的精确项是

$$
\boxed{
\mathbb E
\left[
\frac{\beta_t^2}
{2\sigma_{\mathrm{rev},t}^2
\alpha_t(1-\bar\alpha_t)}
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
\right].
}
$$

这就是 noise-prediction MSE 的来源。它不是由“噪声比图像容易预测”这一经验口号推出的，而是 posterior Gaussian KL 在一种特定 mean parameterization 下的精确重写。

### 5.3 两种固定 variance 对应两套权重

Ho et al. 比较两种固定 reverse variance：

$$
\sigma_{\mathrm{rev},t}^2=\beta_t
$$

或

$$
\sigma_{\mathrm{rev},t}^2=\tilde\beta_t.
$$

第一种给出

$$
w_t^{(\beta)}
=
\frac{\beta_t}
{2\alpha_t(1-\bar\alpha_t)}.
$$

第二种在 $t\ge2$ 给出

$$
\begin{aligned}
w_t^{(\tilde\beta)}
&=
\frac{\beta_t^2}
{2\tilde\beta_t\alpha_t(1-\bar\alpha_t)}\\
&=
\frac{\beta_t}
{2\alpha_t(1-\bar\alpha_{t-1})}.
\end{aligned}
$$

两者都随 timestep 变化。下面用 DDPM 的 1000-step linear-$\beta$ schedule 直接画出权重；虚线 $1$ 是 $L_{\mathrm{simple}}$：

![Exact VLB mean weights versus the simple objective](/images/diffusion/d2_objective_weights.png)

图中 $t=1$ 被排除，因为它是 decoder term $L_0$。纵轴使用 log scale；如果只看 linear scale，后期较小的 VLB 权重几乎不可见。

***

## 6. $L_{\mathrm{simple}}$ 到底简化了什么

DDPM 实际用于高质量图像生成的目标是

$$
\boxed{
L_{\mathrm{simple}}
=
\mathbb E_{
t\sim U\{1,\ldots,T\},
x_0\sim q_{\mathrm{data}},
\epsilon\sim\mathcal N(0,I)
}
\left[
\left\|
\epsilon-
\epsilon_\theta(
\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,
t
)
\right\|^2
\right].
}
$$

聚焦于 intermediate mean regression 时，它做了两件事：

1. 用 uniform distribution 采样 timestep；
2. 删除精确 VLB mean term 中随 $t$ 变化的 $w_t$。

相对完整 VLB，它也不再显式包含 terminal prior、decoder likelihood 和 variance-dependent terms；因此下面的“重加权”比较特指 intermediate mean objective。

因此它不是“只差一个对所有样本相同的常数”。它改变了不同 noise levels 的相对训练权重。Ho et al. 明确把它描述为对 variational bound 的简化与重加权，并报告它在实验中改善 sample quality，却不利于 codelength；这是一项实证发现，不是 likelihood 与感知质量必然冲突的普适定理（Ho et al., 2020, Eqs. 12--14 and Sec. 4.2）。

### 6.1 uniform timestep 估计中的全局 $T$

若要无偏估计

$$
\sum_{t=2}^{T}L_{t-1},
$$

从 $t\sim U\{2,\ldots,T\}$ 抽一个时刻后，应乘 $T-1$。若只比较参数 minimizer，这个全局正数可以省略。

但 $w_t$ 不是全局常数，不能同样省略：

$$
\sum_t w_t\ell_t(\theta)
\quad\text{与}\quad
\sum_t\ell_t(\theta)
$$

一般是两个不同目标。

### 6.2 为什么不同权重仍可能有相同的 population predictor

这里有一个需要细分的研究层结论，也是本文为区分“目标不同”与“理想预测器相同”补充的分析。设

$$
\mathcal J_w(f)
=
\mathbb E\left[
w(t)\|\epsilon-f(x_t,t)\|^2
\right],
\qquad w(t)>0.
$$

固定 $x_t,t$ 后，$w(t)$ 是正的常数，因此 conditional risk 的 minimizer 为

$$
\boxed{
f^*(x_t,t)
=\mathbb E[\epsilon\mid x_t,t].
}
$$

所以在无限数据、无限函数容量、每个 $t$ 都得到正权重且达到全局最优的理想条件下，不同纯 timestep 权重具有相同的 pointwise Bayes predictor。

这并不让两个 objective 变成同一个函数：

- finite-capacity 网络在 timesteps 之间共享参数；
- SGD 只运行有限时间；
- 梯度尺度与 estimator variance 不同；
- 完整 VLB 还包含 variance、decoder 与 prior terms；
- 某些权重可能使某些 noise levels 几乎得不到有效优化。

因此，“Bayes optimum 相同”与“有限训练得到相同模型”是两种不同论断。$L_{\mathrm{simple}}$ 的经验成功恰恰发生在后一个层面。

### 6.3 noise、$x_0$ 与 mean prediction 的关系

由

$$
x_t
=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon
$$

可从 predicted noise 得到

$$
\boxed{
\hat x_{0,\theta}
=
\frac{
x_t-\sqrt{1-\bar\alpha_t}
\epsilon_\theta(x_t,t)
}{
\sqrt{\bar\alpha_t}
}.
}
$$

再把 $\hat x_{0,\theta}$ 代入 $\tilde\mu_t(x_t,x_0)$，会得到与 DDPM Eq. 11 相同的 $\mu_\theta$。所以预测 $\epsilon$、预测 $x_0$ 与预测 posterior mean 在 algebra 上可以互相转换。

但转换包含与 noise level 有关的尺度。给定同一种 raw MSE 时，不同 parameterization 隐式对应不同的 weighting 与数值条件；不能只凭“可互相换算”就宣称训练行为完全相同。更系统的 parameterization 比较留给 D5。

***

## 7. 最小训练算法：公式怎样落到张量

DDPM Algorithm 1 可以写成：

```text
repeat
    x0 ~ q_data
    t  ~ Uniform({1, ..., T})
    epsilon ~ Normal(0, I)
    xt = sqrt(alpha_bar[t]) * x0
         + sqrt(1 - alpha_bar[t]) * epsilon
    epsilon_hat = network(xt, t)
    loss = mean((epsilon - epsilon_hat)^2)
    update network parameters
until convergence
```

这里没有采样完整 forward trajectory。D1 已证明 direct marginal 正确，因此一次训练只需一次 Gaussian noise。

### 7.1 数学量与代码变量

| 数学量                | 代码张量                 | 典型 shape    | 说明                        |
| ------------------ | -------------------- | ----------- | ------------------------- |
| $\beta_t$          | `betas`              | `[T]`       | one-step forward variance |
| $\bar\alpha_t$     | `alpha_bar`          | `[T]`       | cumulative signal power   |
| $t$                | `t`                  | `[B]`       | 每个 batch item 可不同         |
| $x_0,x_t,\epsilon$ | `x0, xt, noise`      | `[B,C,H,W]` | 同 shape                   |
| scalar coefficient | `extract(array,t,x)` | `[B,1,1,1]` | 用 broadcasting 乘图像        |
| $\epsilon_\theta$  | `predicted_noise`    | `[B,C,H,W]` | 网络输出                      |

代码索引是 $0,\ldots,T-1$，所以：

$$
\text{code index }0
\quad\Longleftrightarrow\quad
\text{paper step }1.
$$

这不是无关紧要的偏移：posterior variance 的零点和 reverse loop 的 final-step mask 都由它决定。

### 7.2 posterior coefficients

配套代码复用 D1 的 schedule，并新增：

```python
alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
posterior_variance = (
    betas * (1 - alpha_bar_prev) / (1 - alpha_bar)
)
posterior_mean_coef_x0 = (
    betas * alpha_bar_prev.sqrt() / (1 - alpha_bar)
)
posterior_mean_coef_xt = (
    (1 - alpha_bar_prev) * alphas.sqrt() / (1 - alpha_bar)
)
```

batch posterior mean 只是两项 broadcast：

```python
mean = (
    extract(posterior_mean_coef_x0, t, xt) * x0
    + extract(posterior_mean_coef_xt, t, xt) * xt
)
```

### 7.3 simple loss 与 weighted VLB mean term

```python
per_example_mse = (
    (true_noise - predicted_noise).square()
    .flatten(1)
    .mean(1)
)

weight_t = betas.square() / (
    2 * reverse_variance * alphas * (1 - alpha_bar)
)
weighted_mean_term = per_example_mse * weight_t[t]
```

`flatten(1).mean(1)` 只在 non-batch dimensions 上平均，保留每个样本的 loss，便于之后施加 timestep weights 或 importance correction。

配套脚本同时实现并检查：

- $x_0,\epsilon,\mu$ 三种 posterior mean 写法的一致性；
- simple 与 weighted loss 的张量映射；
- learned-range variance 的两个端点；
- timestep importance correction 的无偏性；
- final reverse step 的 noise mask。

运行：

```bash
# 本地验证脚本暂未公开
```

它只执行代数不变量检查并生成两张解释图，不训练 U-Net。

### 7.4 教学实现不等于论文目标

Hugging Face *Annotated Diffusion* 很适合阅读 direct noising、loss 与 sampler 的最小 PyTorch 顺序，但其正文训练使用 Huber loss，并提供 L1/L2/Huber 选项；DDPM Algorithm 1 使用 squared error。Huber 可以作为工程变体，却不能冒充本节从 Gaussian KL 推出的原始目标（Rogge & Rasul, 2022, source lines 741--760 and 931--934）。

***

## 8. Ancestral sampling：从 network output 走回 $x_0$

训练完成后，从

$$
x_T\sim\mathcal N(0,I)
$$

开始，对 $t=T,T-1,\ldots,1$ 重复采样

$$
x_{t-1}\sim
p_\theta(x_{t-1}\mid x_t).
$$

固定 variance 时，DDPM 的一步更新为

$$
\boxed{
x_{t-1}
=
\frac1{\sqrt{\alpha_t}}
\left(
x_t
-\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right)
+
\sigma_{\mathrm{rev},t}z,
}
$$

其中

$$
z\sim\mathcal N(0,I)
\quad\text{if }t>1,
$$

而在 $t=1$ 令

$$
z=0.
$$

伪代码是：

```text
xT ~ Normal(0, I)
for t = T, ..., 1
    epsilon_hat = network(xt, t)
    mean = (xt - beta[t] / sqrt(1-alpha_bar[t]) * epsilon_hat)
           / sqrt(alpha[t])
    if t > 1
        z ~ Normal(0, I)
    else
        z = 0
    x_{t-1} = mean + sigma_rev[t] * z
return x0
```

### 8.1 为什么最后一步不加噪声

在代码 index $0$：

$$
\tilde\beta_1=0.
$$

posterior teacher 已经退化到 $x_0$。若最后仍注入随机噪声，输出会在 decoder 之后被无依据地再次扰动。官方 DDPM 代码用 `nonzero_mask` 在 `t==0` 时关闭 noise；这与“为了画面更稳定而临时设零”不同，它来自端点分布的数学结构（Ho et al., 2020, Algorithm 2 and official `diffusion_utils.py`）。

### 8.2 从 $\mathcal N(0,I)$ 启动仍有条件

采样器假设 initial prior 与训练时 $q(x_T)$ 足够匹配。D1 已说明标准 finite schedule 通常只有

$$
q(x_T\mid x_0)\approx\mathcal N(0,I),
$$

而不是自动精确相等。若训练 terminal SNR 非零，却始终从 pure Gaussian 启动，就存在 endpoint train-inference mismatch。若改成 exact zero terminal SNR，又必须同步处理 $\epsilon$-prediction 的端点退化和 reverse formula 的数值奇异。

***

## 9. Fixed variance 为什么不够：Improved DDPM

$L_{\mathrm{simple}}$ 只训练 $\epsilon_\theta$，不能为 $\Sigma_\theta$ 提供 likelihood 信号。DDPM 2020 的最终模型因而固定 reverse variance 为 $\beta_t$ 或 $\tilde\beta_t$。

这留下两个问题：

1. fixed variance 限制 reverse Gaussian 的表达能力；
2. 直接优化完整 VLB 虽有 likelihood 意义，却在实验中梯度噪声大且 sample quality 较差。

Improved DDPM 将这些问题拆开处理（Nichol & Dhariwal, 2021, Secs. 3.1 and 3.3）。

### 9.1 learned-range variance

网络额外输出 $v_\theta(x_t,t)$，并在 log-variance 空间插值：

$$
\boxed{
\log\Sigma_\theta
=
v_\theta\log\beta_t
+
(1-v_\theta)\log\tilde\beta_t.
}
$$

等价地，

$$
\Sigma_\theta
=
\exp\!\left(
v_\theta\log\beta_t
+
(1-v_\theta)\log\tilde\beta_t
\right).
$$

这里是 log-space interpolation，不是 variance 的线性插值。论文没有硬性把 $v_\theta$ clamp 到 $[0,1]$，虽然这两个端点给出了设计时的参考范围。

它仍然不是“学习任意 full covariance”：官方实现通常输出 per-pixel diagonal log variance，并被 parameterization 限制在两条 schedule-dependent reference curves 附近。

### 9.2 hybrid objective 与 stop-gradient

Improved DDPM 使用

$$
\boxed{
L_{\mathrm{hybrid}}
=
L_{\mathrm{simple}}
+\lambda L_{\mathrm{vlb}},
\qquad
\lambda=0.001.
}
$$

关键实现细节是：VLB branch 中的 mean/noise output 被 stop-gradient，variance output 保留梯度。于是形成显式分工：

- $L_{\mathrm{simple}}$ 主要训练 mean/noise prediction；
- 小权重 $L_{\mathrm{vlb}}$ 主要训练 variance。

若教程代码只写

```python
loss = simple_loss + 0.001 * vlb_loss
```

却不解释 VLB branch 的 detached mean，它就没有忠实表达论文的训练设计。stop-gradient 不是可忽略的性能小技巧，而是目标分工的一部分（Nichol & Dhariwal, 2021, Eq. 16；official `gaussian_diffusion.py`）。

### 9.3 likelihood 与 sample quality 的张力

Improved DDPM 报告：

- 直接优化 resampled VLB 改善 NLL；
- 但对应 FID 明显变差；
- hybrid objective 在二者之间折中。

正确措辞是“在该论文的模型、数据与训练设置中观察到 tradeoff”，而不是“更好 likelihood 必然导致更差 perceptual quality”。FID 与 NLL 测量不同属性，但它们之间不存在由这些实验直接推出的普适反向定理。

***

## 10. 改 objective weight 与改 timestep proposal 是两件事

这是 diffusion 实现中最常见的概念混淆之一。

设目标是 timestep average：

$$
\mathcal J(\theta)
=\frac1T\sum_{t=1}^{T}
\mathbb E[\ell_t(\theta)].
$$

可以不从 uniform distribution 采样，而取任意 $p_t>0$：

$$
t\sim p.
$$

只要使用 importance correction，

$$
\boxed{
\widehat{\mathcal J}
=
\frac{\ell_t}{Tp_t},
}
$$

就仍然无偏：

$$
\mathbb E_{t\sim p}
\left[
\frac{\ell_t}{Tp_t}
\right]
=
\frac1T\sum_{t=1}^{T}
\mathbb E[\ell_t].
$$

所以：

- 改 $p_t$ 并正确 correction：objective 不变，estimator variance 改变；
- 直接把 $\ell_t$ 乘新的权重且不抵消：objective 改变；
- $L_{\mathrm{simple}}$ 相对 VLB 属于第二种；
- Improved DDPM 的 loss-second-moment sampler 属于第一种。

### 10.1 为什么 optimal proposal 与 second moment 有关

为简化记号，令单样本 timestep loss 的 second moment 为

$$
m_t=\mathbb E[\ell_t^2].
$$

无偏 sum estimator $\ell_t/p_t$ 的 second moment 为

$$
\sum_t\frac{m_t}{p_t}.
$$

在约束

$$
\sum_tp_t=1
$$

下，用 Lagrange multiplier：

$$
\mathcal F(p,\lambda)
=
\sum_t\frac{m_t}{p_t}
+\lambda\left(\sum_tp_t-1\right).
$$

令偏导为零：

$$
-\frac{m_t}{p_t^2}+\lambda=0,
$$

所以

$$
\boxed{
p_t\propto\sqrt{\mathbb E[\ell_t^2]}.
}
$$

Improved DDPM 用每个 timestep 最近若干 loss 的 second moment 估计该 proposal，并混入很小的 uniform probability，防止任何 timestep 永远不被采样。这里优化的是 gradient/loss estimator 的方差，不是重新定义 VLB（Nichol & Dhariwal, 2021, Eq. 18 and `resample.py`）。

***

## 11. VDM：在 SNR 坐标中重新看目标权重

D1 已引入统一 Gaussian marginal：

$$
q(z_t\mid x)
=
\mathcal N(a(t)x,b(t)^2I),
$$

$$
\operatorname{SNR}(t)
=\frac{a(t)^2}{b(t)^2},
\qquad
\gamma(t)=-\log\operatorname{SNR}(t).
$$

为了避免跨论文记号误读：

| 本教程/D1      | VDM 原文                 | DDPM                                  |
| ----------- | ---------------------- | ------------------------------------- |
| $a(t)$      | $\alpha_t$             | $\sqrt{\bar\alpha_t}$                 |
| $b(t)^2$    | $\sigma_t^2$           | $1-\bar\alpha_t$                      |
| $\gamma(t)$ | $-\log\mathrm{SNR}(t)$ | $\log((1-\bar\alpha_t)/\bar\alpha_t)$ |

VDM 对相邻 $s<t$ 的 posterior KL 推出 $x$-prediction form：

$$
\boxed{
\frac12
\left(
\operatorname{SNR}(s)-\operatorname{SNR}(t)
\right)
\left\|
x-\hat x_\theta(z_t,t)
\right\|^2.
}
$$

由

$$
\hat x_\theta
=
\frac{z_t-b(t)\hat\epsilon_\theta}{a(t)}
$$

以及

$$
\left\|
x-\hat x_\theta
\right\|^2
=
\frac{1}{\operatorname{SNR}(t)}
\left\|
\epsilon-\hat\epsilon_\theta
\right\|^2,
$$

得到 noise-prediction weight：

$$
\begin{aligned}
&\frac12
\left(
\frac{\operatorname{SNR}(s)}
{\operatorname{SNR}(t)}
-1
\right)\\
={}&
\frac12
\left(
\exp(\gamma(t)-\gamma(s))-1
\right)\\
={}&
\boxed{
\frac12\operatorname{expm1}
\left(
\gamma(t)-\gamma(s)
\right).
}
\end{aligned}
$$

`expm1(u)` 在 $u$ 很小时比直接计算 `exp(u)-1` 更稳定。

### 11.1 连续极限

令 $s=t-\Delta t$。当 $\Delta t\to0$：

$$
\gamma(t)-\gamma(s)
\approx\gamma'(t)\Delta t,
$$

$$
\operatorname{expm1}(\gamma(t)-\gamma(s))
\approx\gamma'(t)\Delta t.
$$

求和趋于积分：

$$
\boxed{
L_\infty
=
\frac12
\int_0^1
\gamma'(t)
\mathbb E
\left[
\|\epsilon-\hat\epsilon_\theta\|^2
\right]dt.
}
$$

令 $\lambda=\gamma(t)$，则 $d\lambda=\gamma'(t)dt$：

$$
L_\infty
=
\frac12
\int_{\gamma(0)}^{\gamma(1)}
\mathbb E
\left[
\|\epsilon-\hat\epsilon_\theta\|^2
\right]d\lambda.
$$

这给出一个非常清楚的解释：

> VLB 选择了 negative-log-SNR 轴上的一种积分 measure；uniform-time $L_{\mathrm{simple}}$ 选择的是另一种 measure。

### 11.2 三层“权重”不要混为一谈

1. **Objective measure**：理论上要对哪些 noise levels 赋多少权重；
2. **Parameterization scale**：预测 $x,\epsilon,$ score 时变量转换引入的尺度；
3. **Monte Carlo proposal**：为了估计既定积分，实际怎样采样 $t$。

VDM 的 continuous-time specification equivalence 要求 SNR 单调可逆、endpoints 相同，并对 denoiser 做相应重缩放。它不意味着 finite-step discretization、network conditioning、Monte Carlo variance 与 sampler error 都不受 schedule 影响（Kingma et al., 2021, Secs. 4--5 and Apps. E, K）。

***

## 12. 从 noise predictor 到 score：只建立接口

给定 $x_0$，forward conditional score 为

$$
\begin{aligned}
\nabla_{x_t}
\log q(x_t\mid x_0)
&=
-\frac{
x_t-\sqrt{\bar\alpha_t}x_0
}{
1-\bar\alpha_t
}\\
&=
-\frac{\epsilon}
{\sqrt{1-\bar\alpha_t}}.
\end{aligned}
$$

因此可把 noise predictor 转成

$$
s_\theta(x_t,t)
=
-\frac{\epsilon_\theta(x_t,t)}
{\sqrt{1-\bar\alpha_t}}.
$$

平方损失的 Bayes optimum 是

$$
\epsilon_\theta^*(x_t,t)
=\mathbb E[\epsilon\mid x_t,t],
$$

而不是训练样本中特定 $\epsilon$ 的确定性逆函数。进一步有

$$
-\frac{
\mathbb E[\epsilon\mid x_t]
}{
\sqrt{1-\bar\alpha_t}
}
=
\nabla_{x_t}\log q_t(x_t).
$$

这表明 noise prediction 最终连接到 aggregate noisy marginal 的 score。为什么 conditional target 的回归会得到 marginal score、该结论与 denoising score matching 有什么关系，见 [D3. Score Matching：去噪器究竟学到了什么](/blog/diffusion/d3-score-matching/)；本章只需要知道它为 reverse model 提供了另一种解释。

***

## 13. 技术演进：每一步解决的是哪个问题

### 13.1 2015：先让 learned reversal 可训练

**问题：** 复杂数据 density 难以直接建模，同时希望支持 sampling 与 likelihood training。

**方案：** Sohl-Dickstein et al. 构造固定 forward diffusion 与 learned reverse chain，用 trajectory variational lower bound 将训练拆成局部 reverse matching。

**局限：** 图像生成质量尚不具竞争力；参数化、schedule 与 reverse variance 仍有很大设计空间；采样需要大量串行步骤。

### 13.2 2020 DDPM：用 noise parameterization 获得高质量样本

**问题：** 原始 diffusion framework 可训练，但怎样选择有效的 reverse mean target 与 weighting？

**方案：** Ho et al. 将 posterior mean 改写为 $\epsilon$-prediction，并采用 uniform-t $L_{\mathrm{simple}}$。

**解决：** diffusion model 在图像合成上表现出当时有竞争力的 sample quality，同时算法极其简洁。

**代价：** 简化目标偏离 exact VLB weighting；likelihood 与 sample quality 指标出现张力；variance 仍固定。

### 13.3 2021 Improved DDPM：把 mean、variance 与 estimator 分工

**问题：** $L_{\mathrm{simple}}$ 不训练 variance；直接 VLB 梯度噪声大；likelihood 与少步采样需要改进。

**方案：** learned-range variance、hybrid loss、mean stop-gradient、loss-second-moment timestep sampling。

**解决：** 更好的 likelihood 与更少的 sampling steps，并保留高质量 mean training。

**局限：** hybrid coefficient 与 variance family 仍是工程设计；sampling 仍然串行。

### 13.4 2021 VDM：从 timestep 重组到 SNR measure

**问题：** schedule、parameterization、离散深度和 likelihood weighting 的关系仍然混杂。

**方案：** 用 SNR/log-SNR 表示 forward marginals、离散 KL 与 continuous VLB。

**解决：** 揭示不同目标是在 noise-level axis 上选择不同 measure，并分清 continuous objective 与 estimator variance。

**局限：** continuous equivalence 不消除 finite discretization、network capacity 与 sampling solver 的影响。

可以把演进压缩为：

| 工作                             | 改变的层                                  | 主要目的                    |
| ------------------------------ | ------------------------------------- | ----------------------- |
| Sohl-Dickstein 2015            | path model + variational training     | 让 reversal 可训练          |
| DDPM 2020 VLB                  | posterior/noise mean parameterization | 可解析 likelihood bound    |
| DDPM $L_{\mathrm{simple}}$     | objective timestep weight             | 提升训练与 sample quality    |
| Improved DDPM learned variance | reverse model family                  | 改善 likelihood/少步采样      |
| Improved DDPM resampling       | estimator proposal                    | 降低 VLB gradient noise   |
| VDM 2021                       | SNR coordinate + continuous objective | 统一 weighting 与 schedule |

***

## 14. 常见错误与适用边界

### 错误 1：把 conditioned posterior 当成生成时的 reverse

$$
q(x_{t-1}\mid x_t,x_0)
\ne
q(x_{t-1}\mid x_t).
$$

前者是 teacher，后者才是理想生成 transition。

### 错误 2：因为 posterior 是 Gaussian，就说真实 reverse 精确是 Gaussian

对 $x_0$ 积分通常得到 Gaussian mixture。单 Gaussian reverse 是小步直觉支持下的模型族选择。

### 错误 3：说 $L_{\mathrm{simple}}$ 就是 ELBO 去掉常数

它删除了随 $t$ 变化的 KL-induced weight，改变 objective measure。

### 错误 4：把 $L_T$ 说成永远等于零

它在固定 schedule 下不依赖 $\theta$，但只有 terminal prior 精确匹配时数值才为零。

### 错误 5：在 $t=1$ 继续使用普通 posterior Gaussian KL

$\tilde\beta_1=0$。该端点应使用 decoder likelihood，并对 log variance 单独处理。

### 错误 6：learned variance 后仍把整个 KL 写成纯 noise MSE

variance ratio 与 log-determinant 仍依赖网络，不能被 mean MSE 吸收。

### 错误 7：hybrid loss 只要相加两个 scalar 就够

Improved DDPM 对 VLB branch 的 mean output 使用 stop-gradient，使其主要训练 variance。

### 错误 8：改变 timestep sampling 就一定改变 objective

带正确 importance correction 时，只改变 estimator；没有 correction 或显式重加权时，才改变目标。

### 错误 9：把 better NLL 与 worse FID 写成定理

这是特定实验设置中的经验 tradeoff，不是对所有模型和数据的普遍定律。

### 错误 10：reverse loop 每一步都加入随机噪声

最后代码步 `t==0` 必须关闭 noise。论文 timestep 与代码 index 相差 1。

***

## 15. 本章给后续章节准备了什么

D2 已经把一个未知 reverse problem 化成了可以优化的神经网络目标：

$$
x_0,\epsilon,t
\longrightarrow
x_t
\longrightarrow
\epsilon_\theta(x_t,t)
\longrightarrow
\mu_\theta(x_t,t)
\longrightarrow
p_\theta(x_{t-1}\mid x_t).
$$

但仍有三个深层问题尚未解决：

1. noise predictor 为什么等价于 noisy data marginal 的 score；
2. 离散 reverse chain 在连续极限下如何变成 reverse-time SDE；
3. 如何用更少 steps、更高阶 solver 或确定性路径完成 sampling。

[D3](/blog/diffusion/d3-score-matching/) 将回答第一个问题，把 DDPM 与 denoising score matching 连接起来。D4 进入 reverse-time SDE 与 probability-flow ODE；D6 再系统讨论 DDPM、DDIM 与现代 solvers。

***

## 16. 章节小结

- 真实无条件 reverse $q(x_{t-1}\mid x_t)$ 依赖未知 aggregate data distribution，通常不能解析；
- 训练时给定 $x_0$ 后，posterior $q(x_{t-1}\mid x_t,x_0)$ 是可解析 Gaussian；
- expected conditioned-posterior KL 与逼近 unconditional reverse 只差一个不依赖模型的 conditional mutual information；
- path ELBO 分成 terminal prior、intermediate KL 与 decoder 三类项；
- 固定 reverse variance 时，每个 intermediate KL 的 mean term 精确化为带 timestep weight 的 noise MSE；
- DDPM $L_{\mathrm{simple}}$ 删除了该 weight，是有意的 surrogate，而不是 unchanged ELBO；
- 不同正 timestep weights 在理想 population limit 下可有相同 Bayes predictor，但有限模型、优化与完整 likelihood 目标仍然不同；
- Improved DDPM 分别用 learned variance、hybrid loss、stop-gradient 和 importance sampling 处理不同问题；
- 改 objective weight 与改 estimator proposal 必须通过 importance correction 区分；
- VDM 将离散权重重写为 log-SNR measure，统一了 discrete VLB 与 continuous objective；
- reverse sampling 从 prior 开始逐步 ancestral sample，最后一步不再加入 Gaussian noise。

***

## 17. 思考题

1. **conditioned teacher 的信息代价。** 证明本章的 conditional-KL decomposition，并解释其中 $I(X_{t-1};X_0\mid X_t)$ 在 $\beta_t\to0$ 时可能怎样变化。它是否意味着 teacher 与 unconditional reverse 变得相同？

2. **Gaussian reverse 的最佳方差。** 利用 total variance，推导在固定 $x_t$ 下用单个 Gaussian 拟合 reverse mixture 时的 moment-matching mean 与 covariance。比较它与 $\tilde\beta_tI$ 的差异。

3. **相同 Bayes predictor，不同训练结果。** 构造一个只有一个共享 scalar parameter、两个 timesteps 的回归例子，使两种 positive timestep weights 得到不同的有限模型 optimum。它为什么不违背本章的 pointwise Bayes 结论？

4. **importance sampling 的边界。** 若 $p_t=0$ 对某个 loss 非零的 timestep 成立，无偏性在哪里失败？Improved DDPM 混入 uniform probability 还解决了哪些 early-training 问题？

5. **endpoint decoder。** 从 $t=1$ 的退化 posterior 出发，解释为什么连续数据上的 Gaussian decoder 与离散 8-bit 图像上的 discretized likelihood 需要不同处理。

6. **预测目标与端点。** 当 terminal SNR 精确为零时，$x_T=\epsilon$。分析 pure $\epsilon$-prediction 在该点的监督为何退化成 identity mapping，以及 $x_0$-prediction 或其他 parameterization 会怎样改变问题。

7. **likelihood 与感知质量。** 给出至少三种可能导致 VLB weighting 与 perceptual sample quality 不一致的机制，并区分哪些是数学事实、哪些只是需要实验检验的假设。

8. **离散到连续。** 从

   $$
   \frac12\operatorname{expm1}(\gamma(t)-\gamma(s))
   \|\epsilon-\hat\epsilon_\theta\|^2
   $$

   出发，写出连续极限需要的 regularity assumptions。若 $\gamma$ 在某点不可微或不是严格单调，哪些步骤需要修改？

***

## 18. 本章来源与继续阅读

核心公式与历史结论优先引用原始论文；博客与教程只承担教学顺序和实现对照。

1. Jascha Sohl-Dickstein et al. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/1503.03585 "官方论文页面")。
2. Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2006.11239 "官方论文页面")。
3. Alex Nichol, Prafulla Dhariwal. *Improved Denoising Diffusion Probabilistic Models*. ICML 2021. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2102.09672 "官方论文页面")。
4. Diederik P. Kingma et al. *Variational Diffusion Models*. NeurIPS 2021. 本地论文（补充材料暂未公开）；[结构化笔记](https://arxiv.org/abs/2107.00630 "官方论文页面")。
5. Niels Rogge, Kashif Rasul. *The Annotated Diffusion Model*. Hugging Face, 2022. 官方 Markdown 快照（补充材料暂未公开）；使用边界笔记（补充材料暂未公开）。
6. Calvin Luo. *Understanding Diffusion Models: A Unified Perspective*. 2022. 本地论文（补充材料暂未公开）。其 HVAE-to-VDM 长推导与本章 posterior/ELBO 交叉核验。
7. Lilian Weng. *What are Diffusion Models?* 2021. 本地网页快照（补充材料暂未公开）。其紧凑推导中的 Gaussian KL、`L_simple` 与 score identity 需要按[独立教程交叉核验及勘误](https://arxiv.org/abs/2208.11970 "官方论文页面")修正后使用。
8. Chieh-Hsin Lai et al. *The Principles of Diffusion Models*. arXiv:2510.21890v2, 2026 revision. 固定版本 PDF（补充材料暂未公开）。Chs. 1--2 提供近期 VAE/HVAE-to-DDPM 教学路线；其版本和 per-step KL/weighting 限定见范围核验（补充材料暂未公开）。

完整独立代数复核见 DDPM core derivation ledger（补充材料暂未公开）。两份独立二手推导的覆盖与勘误见 [D2 tutorial cross-check](https://arxiv.org/abs/2208.11970 "官方论文页面")。官方实现版本与 commit 见 code provenance（补充材料暂未公开）。D2 的 claim-evidence mapping 见 chapter source packet（补充材料暂未公开）。
