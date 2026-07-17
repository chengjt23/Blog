---
title: 参数化、Schedule 与训练设计空间
description: 比较 epsilon、x0、score 与 velocity 参数化，拆分噪声路径、训练采样、损失权重和采样网格。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 5
slug: d5-parameterization-training-design
tags:
  - diffusion
  - parameterization
  - training-design
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦参数化几何、隐式权重、EDM 预条件与训练设计，不把可逆预测量误判为相同优化问题。
---
同一个带噪样本，网络可以预测噪声 $\epsilon$、干净数据 $x_0$、score，或者所谓的 velocity $v$。这些量之间只是线性变换，所以一个自然问题是：既然它们可以互相转换，为什么换一个 prediction target 往往会改变训练稳定性、样本质量和低噪声或高噪声端点的行为？

答案是：**预测值可逆，不代表未经调整的损失相同；population objective 可经加权调成相同，也不代表有限网络、有限 batch 和具体优化器具有相同 dynamics。** 参数化同时改变网络输出的尺度、误差被放大的方向、不同噪声水平收到的隐式权重，以及端点附近的数值条件。

本章还会处理另一个常见混淆：“noise schedule”经常同时指前向加噪路径、训练时如何抽噪声、loss 如何加权，以及采样时怎样布置离散步。它们共享一个噪声坐标，但承担不同数学职责。EDM 的重要贡献之一，正是把这些选择拆成可分别研究的模块。

![参数化的几何、隐式损失尺度与端点条件](/images/diffusion/d5_parameterization_geometry.png)

本章按三层阅读。第一次阅读可以抓住参数化转换表和五类设计轴；第二次阅读跟随所有 MSE 换算、log-SNR 测度与 EDM 二阶矩推导；研究层则要留意 monotonic weighting theorem 的条件、finite-network non-equivalence，以及 EDM2 所揭示的训练期尺度漂移。

***

## 1. 一个反例：输出相同信息，优化问题却不同

考虑统一的 affine Gaussian corruption：

$$
\boxed{
z_t=a_t x_0+b_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
}
$$

给定 $z_t,t$，若网络输出 $\hat\epsilon$，就可以恢复

$$
\hat x_0=\frac{z_t-b_t\hat\epsilon}{a_t}.
$$

假设噪声预测有误差 $\delta_\epsilon=\hat\epsilon-\epsilon$，则干净数据误差为

$$
\hat x_0-x_0=-\frac{b_t}{a_t}\delta_\epsilon.
$$

当 $a_t\to0$ 时，即使 $\delta_\epsilon$ 很小，$x_0$ 误差也会被 $b_t/a_t$ 大幅放大。在纯噪声端点 $a_T=0$ 上，这个反演甚至没有定义。

反过来，若网络直接输出 $\hat x_0$，高噪声处的 target 仍是完整数据，但输入几乎不含关于具体 $x_0$ 的信息。此时最优 MSE predictor 只能趋近条件均值；target 本身没有退化，问题却在统计上高度不确定。

所以“信息等价”至少要拆成三句话：

1. 在非奇异噪声水平，给定同一个 $z_t$，不同预测值可以代数转换；
2. 转换会缩放误差，constant MSE 因而定义不同的 population metric；
3. 即使把 MSE 权重校准到同一 metric，网络 head、梯度尺度、参数共享和优化噪声仍可能不同。

这也是本章的组织原则：先固定 corruption，再讨论 output target；先换算 objective，再讨论训练 proposal 和网络 preconditioning。

***

## 2. 一条加噪公式，七个独立设计决定

定义

$$
\mathrm{SNR}_t=\frac{a_t^2}{b_t^2},
\qquad
\lambda_t=\log\mathrm{SNR}_t.
$$

本章默认 $a_t,b_t\ge0$。Variance Preserving（VP）路径还满足

$$
\boxed{a_t^2+b_t^2=1.}
$$

在 VP 情形，只给定 SNR 就能恢复

$$
a_t^2=\frac{\mathrm{SNR}_t}{1+\mathrm{SNR}_t},
\qquad
b_t^2=\frac{1}{1+\mathrm{SNR}_t}.
$$

但一般 affine path 不由 SNR 单独确定：把 $(a_t,b_t)$ 同时乘一个尺度，SNR 不变，输入方差却改变。讨论 schedule 前，必须先说明自己在谈哪一层。

| 设计层                     | 数学对象                                     | 它决定什么                    |
| ----------------------- | ---------------------------------------- | ------------------------ |
| corruption path         | $t\mapsto(a_t,b_t)$ 或 transition kernels | 数据如何被破坏、端点是什么            |
| path coordinate         | $t,\sigma,\lambda$ 等                     | 如何给同一噪声水平编号              |
| output target           | $x_0,\epsilon,v,s$                       | 网络输出的语义和尺度               |
| objective measure       | $W_x(t)dt$ 或 $w(\lambda)d\lambda$        | population loss 强调哪些噪声区域 |
| training proposal       | $p_{train}(t)$ 或 $p_{train}(\lambda)$    | Monte Carlo 在哪里取样、估计方差多大 |
| network preconditioning | 输入、skip、输出和 noise embedding 的尺度函数        | 网络内部看到的数值范围              |
| sampling grid/solver    | $\{\lambda_i\}$ 与 update rule            | 生成时的离散化误差和 NFE           |

最后一层属于 D6。本章只说明它为何不能与训练 proposal 合并。架构本身、optimizer、EMA 又是更外层的训练系统；EDM2 会让我们看到，初始尺度被规范化后，这一层仍会随训练演化。

### 2.1 EDM 的 $\sigma$ 不是未经换算的 VP $b_t$

EDM 常用 additive-noise coordinate：

$$
x=y+n,
\qquad
n\sim\mathcal N(0,\sigma^2I).
$$

从 VP 表达 $z=ax_0+b\epsilon$ 除以 $a$ 可得

$$
\frac za=x_0+\frac ba\epsilon,
$$

因此在这个特定 rescaling 下，additive noise standard deviation 是 $\sigma=b/a=1/\sqrt{\mathrm{SNR}}$，而不是 VP 中有界的 $b$。不先说明坐标和 rescaling，就把两个 $\sigma$ 或 noise level 直接等同，会同时弄错输入尺度和 loss weight。

***

## 3. $x_0$、$\epsilon$ 与 conditional score 的完整换算

### 3.1 $x_0\leftrightarrow\epsilon$

由

$$
z_t=a_tx_0+b_t\epsilon
$$

直接解线性方程：

$$
\boxed{
\hat\epsilon=\frac{z_t-a_t\hat x}{b_t},
\qquad
\hat x=\frac{z_t-b_t\hat\epsilon}{a_t}.
}
$$

第一式要求 $b_t\ne0$，第二式要求 $a_t\ne0$。这不是代码实现的小问题，而是参数化在 clean/noise endpoints 上具有不同条件数。

固定 $z_t$，将一致的 $\hat x$ 与 $\hat\epsilon$ 比较：

$$
\hat\epsilon-\epsilon
=-\frac{a_t}{b_t}(\hat x-x_0).
$$

平方后得到本章最基本的 loss-scale identity：

$$
\boxed{
\|\hat\epsilon-\epsilon\|^2
=\mathrm{SNR}_t\|\hat x-x_0\|^2.
}
$$

因此 constant epsilon-MSE 并不是 constant $x_0$-MSE。它在 clean、large-SNR 区域赋予更高的等效 $x_0$ 权重，在 low-SNR 区域赋予更低权重。

### 3.2 conditional score 与 marginal score 不能混写

给定 $x_0$，corruption kernel 为

$$
q(z_t\mid x_0)=\mathcal N(a_tx_0,b_t^2I),
$$

所以 sample-wise conditional score target 是

$$
s^\star(z_t,x_0,t)
=\nabla_{z_t}\log q(z_t\mid x_0)
=-\frac{z_t-a_tx_0}{b_t^2}
=-\frac\epsilon{b_t}.
$$

若定义 $\hat s=-\hat\epsilon/b_t$，则

$$
\hat s-s^\star
=\frac{a_t}{b_t^2}(\hat x-x_0),
$$

从而

$$
\boxed{
\|\hat s-s^\star\|^2
=\frac{a_t^2}{b_t^4}\|\hat x-x_0\|^2.
}
$$

在 VP 下，$a^2/b^4=\mathrm{SNR}(1+\mathrm{SNR})$。这解释了图中 score target 的误差尺度为何在 high-SNR 端增长得更快。

这里的 $s^\star(z_t,x_0,t)$ 依赖被抽中的 clean sample，不等于 marginal score $\nabla\log p_t(z_t)$ 的一次 realization。D3 已证明，在平方损失的 population optimum 上，conditional expectation 满足

$$
\mathbb E[s^\star(z_t,x_0,t)\mid z_t]
=\nabla_{z_t}\log p_t(z_t).
$$

本章只使用这个接口，不重复 DSM consistency proof。

***

## 4. $v$-prediction：VP 圆上的旋转，而非任意 path 的通用速度

Salimans 与 Ho 在 progressive distillation 中使用

$$
\boxed{v_t=a_t\epsilon-b_tx_0.}
$$

把 $z_t,v_t$ 联立：

$$
\begin{bmatrix}z_t\\v_t\end{bmatrix}
=
\begin{bmatrix}a_t&b_t\\-b_t&a_t\end{bmatrix}
\begin{bmatrix}x_0\\\epsilon\end{bmatrix}.
$$

矩阵 determinant 是 $a_t^2+b_t^2$，所以一般 inverse 为

$$
\boxed{
x_0=\frac{a_tz_t-b_tv_t}{a_t^2+b_t^2},
\qquad
\epsilon=\frac{b_tz_t+a_tv_t}{a_t^2+b_t^2}.
}
$$

只有 VP normalization 使 denominator 等于 1，才有熟悉的

$$
x_0=a_tz_t-b_tv_t,
\qquad
\epsilon=b_tz_t+a_tv_t.
$$

遗漏 denominator 的公式在 off-VP path 上不是近似，而是错误。配套代码随机生成 $a^2+b^2\ne1$ 的样本：保留 denominator 的最大恢复误差约为 $10^{-15}$，错误省略它后平均绝对 gap 约为 $0.50$。

### 4.1 为什么叫 velocity

在 VP 圆上令

$$
a_\phi=\cos\phi,
\qquad
b_\phi=\sin\phi.
$$

于是

$$
z_\phi=\cos\phi\,x_0+\sin\phi\,\epsilon,
$$

对角坐标 $\phi$ 求导：

$$
\boxed{
\frac{dz_\phi}{d\phi}
=-\sin\phi\,x_0+\cos\phi\,\epsilon
=v_\phi.
}
$$

所以 $v$ 是 VP 圆相对于 **angle** 的切向速度。若实际 time coordinate 是 $t$，则

$$
\frac{dz_t}{dt}=\dot\phi(t)v_{\phi(t)}.
$$

改变时间参数会缩放 velocity target 和它的 MSE。名称中的“velocity”不能掩盖 coordinate dependence。

### 4.2 一般 path velocity 的边界

对任意可微 affine path，真正的 time velocity 是

$$
u_t=\dot a_tx_0+\dot b_t\epsilon.
$$

令

$$
\Delta_t=a_t\dot b_t-b_t\dot a_t.
$$

当 $\Delta_t\ne0$ 时，$(z_t,u_t)$ 才能恢复 endpoints：

$$
x_0=\frac{\dot b_tz_t-b_tu_t}{\Delta_t},
\qquad
\epsilon=\frac{-\dot a_tz_t+a_tu_t}{\Delta_t}.
$$

VP angle 下 $\Delta_\phi=1$，于是 $u_\phi=v_\phi$。本节的目的只是划清术语边界；general velocity training、flow matching 与 rectified flow 的 path/objective 由 D9 展开。

***

## 5. $v$-MSE 与统一的 target-weight 换算表

固定 $z_t$，把一个 $v$-prediction 转成 implied $\hat x$。由一般 inverse 有

$$
\hat x-x_0=-\frac{b_t}{a_t^2+b_t^2}(\hat v-v),
$$

因此

$$
\boxed{
\|\hat v-v\|^2
=\frac{(a_t^2+b_t^2)^2}{b_t^2}
\|\hat x-x_0\|^2.
}
$$

VP 下 $a_t^2+b_t^2=1$，且 $b_t^{-2}=1+\mathrm{SNR}_t$，所以

$$
\boxed{
\|\hat v-v\|^2
=(1+\mathrm{SNR}_t)\|\hat x-x_0\|^2.
}
$$

这意味着 constant $v$-MSE 对 low-SNR reconstruction 仍保留约为 1 的权重，而 constant epsilon-MSE 的等效权重 $\mathrm{SNR}$ 会趋近 0。这是 $v$ 在 noisy endpoint 附近更适合表达 clean direction 的一个代数原因，但不是对所有模型和指标的性能保证。

现在指定一个共同的 $x_0$-metric：

$$
W_x(t)\|\hat x-x_0\|^2.
$$

不同 target 要实现同一 metric，target-space MSE weight 必须为：

| target            | target MSE / $x_0$-MSE | 等效 target-space weight |
| ----------------- | ---------------------: | ---------------------: |
| $x_0$             |                    $1$ |                  $W_x$ |
| $\epsilon$        |         $\mathrm{SNR}$ |     $W_x/\mathrm{SNR}$ |
| conditional score |              $a^2/b^4$ |           $W_xb^4/a^2$ |
| $v$，general       |      $(a^2+b^2)^2/b^2$ |   $W_xb^2/(a^2+b^2)^2$ |
| $v$，VP            |       $1+\mathrm{SNR}$ | $W_x/(1+\mathrm{SNR})$ |
| path velocity $u$ |         $\Delta^2/b^2$ |      $W_xb^2/\Delta^2$ |

表中最后一行同样只是一致 prediction 之间的代数换算，不是在本章提出 flow objective。

### 5.1 “objective 等价”的准确含义

假设两个模型族能够通过固定线性变换一一对应，而且训练分布、weight、输入 conditioning 都同步转换。上表可让它们的 **population weighted MSE 数值** 相同。但真实训练还共享一组参数去拟合所有 $t$，所以以下因素仍会不同：

- output head 的数值范围与初始化；
- target conditional variance；
- 单样本梯度的方向、范数和方差；
- 网络在不同 noise levels 间如何分配有限 capacity；
- normalization、optimizer state、gradient clipping 和 precision；
- inference 时由 output error 到 denoised estimate 的放大倍数。

因此不能从线性可逆推出“所有参数化训练完全相同”，也不能仅凭某个 target 的名字推出它必然更优。

***

## 6. 从 DDPM 到 Min-SNR：权重设计在解决什么问题

![不同权重族，以及同一 Min-SNR metric 在不同 target 空间中的权重](/images/diffusion/d5_weighting_families.png)

### 6.1 DDPM simple loss：一个有意改变 metric 的 surrogate

DDPM 常用

$$
\mathcal L_{simple}
=\mathbb E_{t,x_0,\epsilon}
\|\epsilon-\epsilon_\theta(z_t,t)\|^2.
$$

由第 3 节，constant epsilon-MSE 等价于

$$
W_x(t)=\mathrm{SNR}_t.
$$

它不是“原 ELBO 换一个写法”。D2 已推导 exact variational weight，并解释 Ho 等人为何有意移除它。本章只强调：一旦 target 与 target-space weight 被选定，一个隐式的 $x_0$-metric 就随之确定。

### 6.2 Improved DDPM：不要把四项改进叫成一个 schedule

Improved DDPM 同时讨论了 cosine noise path、learned reverse variance、hybrid objective 和基于历史 loss second moment 的 timestep proposal。它们分别作用于：

- forward corruption 的噪声分配；
- reverse transition covariance head；
- mean objective 与 variational term 的组合；
- Monte Carlo estimator 的抽样方差。

learned variance 不是另一种 mean prediction target。其 hybrid 实现对 VLB 中 mean prediction 使用 stop-gradient，目的正是减少 covariance learning 对 simple mean objective 的干扰。完整 reverse posterior 与 variance interpolation 已在 D2 给出，这里只保留设计轴的位置。

### 6.3 truncated/max-SNR：给 noisy endpoint 保留 reconstruction weight

Salimans--Ho 在少步蒸馏语境中考虑

$$
W_x(t)=\max(\mathrm{SNR}_t,1).
$$

相对于 simple epsilon loss 的 $W_x=\mathrm{SNR}$，它把 low-SNR 区域截在 1，而不是让 clean reconstruction 权重趋于 0。因此更准确的名称是 truncated 或 max-SNR weighting。

### 6.4 P2：乘在既有 baseline 上的 perceptual prioritization

P2 从已有 per-timestep weight $\lambda_t$ 出发，定义

$$
\boxed{
\lambda'_t
=\lambda_t(k+\mathrm{SNR}_t)^{-\gamma}.
}
$$

其动机是：very high-SNR 阶段主要清理难以感知的微小噪声，有限模型容量未必应过度投入该区域。要解释 P2，必须同时写清 baseline $\lambda_t$、target、SNR convention 和 training proposal。单独画 $(k+\mathrm{SNR})^{-\gamma}$ 不能说明最终的有效 $x_0$-metric。

例如图中为了可比，P2 multiplier 乘在 DDPM simple 的 $W_x=\mathrm{SNR}$ 上。改变 baseline 后，曲线含义也随之改变。

### 6.5 Min-SNR：把 noise levels 视为共享参数的多个任务

Min-SNR 的动机是不同 timesteps 共用一个网络，梯度方向可能冲突；large-SNR 区域又可能因 implicit weight 过大主导训练。它在共同 $x_0$-metric 中取

$$
\boxed{
W_x(t)=\min(\mathrm{SNR}_t,\gamma).
}
$$

因此 VP 下各 target 的权重不是同一个公式：

$$
\boxed{
w_{x_0}=\min(\mathrm{SNR},\gamma),
}
$$

$$
\boxed{
w_\epsilon
=\frac{\min(\mathrm{SNR},\gamma)}{\mathrm{SNR}}
=\min\left(1,\frac\gamma{\mathrm{SNR}}\right),
}
$$

$$
\boxed{
w_v
=\frac{\min(\mathrm{SNR},\gamma)}{1+\mathrm{SNR}}.
}
$$

conditional-score target 的等效权重为

$$
w_s=\min(\mathrm{SNR},\gamma)\frac{b^4}{a^2}.
$$

VP 下也可写为 $w_s=\min(\mathrm{SNR},\gamma)/[\mathrm{SNR}(1+\mathrm{SNR})]$。论文常用 $\gamma=5$，这是经验超参数，不是理论唯一解。

### 6.6 三种名字相近的曲线，方向却不同

| 方法                | 形式                                          | 主要作用                              |
| ----------------- | ------------------------------------------- | --------------------------------- |
| max/truncated-SNR | $\max(\mathrm{SNR},c)$                      | 抬高 low-SNR reconstruction 权重      |
| Min-SNR           | $\min(\mathrm{SNR},\gamma)$                 | 截断 high-SNR 的共同 $x_0$-metric      |
| P2                | baseline $\times(k+\mathrm{SNR})^{-\gamma}$ | 相对抑制 high-SNR cleanup，依赖 baseline |

把 P2 写成 Min-SNR，或把 max-SNR 与 Min-SNR 当作同一 clipping，只会掩盖它们解决的问题不同。

***

## 7. Objective、training proposal 与 sampling grid 必须分开

![objective measure、training proposal 与 sampling grid 的职责分离](/images/diffusion/d5_objective_proposal_grid.png)

### 7.1 用 log-SNR 写 continuous objective

令 $\lambda=\log\mathrm{SNR}$，并假设它从 clean endpoint 向 noisy endpoint 单调下降。一个加权 continuous objective 可抽象为

$$
\boxed{
\mathcal L_w
=\frac12
\int_{\lambda_{min}}^{\lambda_{max}}
w(\lambda)\ell_\theta(\lambda)d\lambda,
}
$$

其中

$$
\ell_\theta(\lambda)
=\mathbb E_{x_0,\epsilon}
\|\hat\epsilon_\theta(z_\lambda,\lambda)-\epsilon\|^2.
$$

只要 corruption kernel 在每个 $\lambda$ 上相同、integration endpoints 固定，$t\mapsto\lambda(t)$ 的具体速度不会出现在这个积分中。这就是 continuous schedule-shape invariance 的核心：它是 change of variables，而不是“schedule 永远不重要”。

若从 uniform $t\sim U(0,1)$ 出发，由于 $\lambda(t)$ 递减，诱导的 density 为

$$
p(\lambda)=-\frac{dt}{d\lambda}.
$$

不同 $\lambda(t)$ 会改变 Monte Carlo 在各噪声区域出现的频率。若没有相应 Jacobian/importance correction，实际优化的 measure 也会改变。

### 7.2 任意 proposal 下的无偏估计

从任意正 density $p_{train}(\lambda)$ 抽样，可以写

$$
\mathcal L_w
=\frac12
\mathbb E_{\lambda\sim p_{train}}
\left[
\frac{w(\lambda)}{p_{train}(\lambda)}
\ell_\theta(\lambda)
\right].
$$

单样本 estimator 是

$$
\boxed{
\widehat{\mathcal L}_w
=\frac12
\frac{w(\lambda)}{p_{train}(\lambda)}
\|\hat\epsilon_\theta-\epsilon\|^2.
}
$$

因此：

- 正确 importance correction 下，改变 proposal 不改变 exact population objective；
- proposal 仍改变 loss 和 stochastic-gradient estimator 的方差；
- 改 proposal 却不除以 $p_{train}$，就同时改变了 objective；
- 理想的低方差 proposal 依赖 integrand 或 gradient norm，通常又随模型训练而变化。

配套代码构造一个一维 integrand。uniform 与 adaptive proposal 都能无偏估计同一积分，但 adaptive proposal 的单样本 contribution 标准差约为 uniform 的 $0.45$。这只是 importance sampling identity 的说明，不是图像模型 benchmark。

### 7.3 sampling grid 是另一种离散化

训练完成后，sampler 选择有限点

$$
\lambda_0>\lambda_1>\cdots>\lambda_N
$$

并用 Euler、Heun 或更专门的 solver 更新状态。这组点决定数值求解在哪些区域投入 NFE；它不是 training proposal，也不需要复现训练 density。

同一个训练 objective 可以配多种 sampling grids；同一个 grid 也可用于由不同 proposal 训练的模型。grid、solver order、model error 和 stochasticity 的交互由 D6 讨论。

***

## 8. 研究层：monotonic weighting 为什么有 ELBO 解释

Kingma 与 Gao 将一类 weighted diffusion objective 与 noisy-data augmentation 上的 ELBO 联系起来。下面给 proof skeleton，目标是说明条件在哪里进入，而不是重建论文所有 measure-theoretic 细节。

设 $\mathcal K(t;x)$ 表示从某个噪声时刻到 terminal endpoint 的适当 path-space KL quantity，并把 weighted loss 写成

$$
\mathcal L_w(x)
=-\int_0^1
w(\lambda_t)\frac d{dt}\mathcal K(t;x)dt.
$$

令 $r(t)=w(\lambda_t)$。integration by parts 给出

$$
-\int_0^1 r(t)\mathcal K'(t;x)dt
=-[r(t)\mathcal K(t;x)]_0^1
+\int_0^1r'(t)\mathcal K(t;x)dt.
$$

第一项产生 endpoint/boundary contributions；在模型参数无关部分整理后，第二项是关键。如果 $r(t)$ 随加噪时间单调不减，则

$$
r'(t)\ge0.
$$

适当归一化后，$r'(t)dt$ 构成一个非负 augmentation measure；若边界有剩余质量，还可能在 endpoint 形成 atom。于是 weighted objective 可以解释为：先随机选择一个 Gaussian-noise augmentation level，再计算相应数据的 ELBO，最后取期望，并加上与 score-network 参数无关的常数。

这个结论不能被压缩成“所有 weighting 都是 ELBO”。至少要保留：

1. $w(\lambda_t)$ 相对 forward time 的 monotonicity；若 $\lambda_t$ 递减，它对应 $w$ 对 $\lambda$ 的相反单调方向；
2. boundary terms、normalization、finite endpoints 与 data likelihood conventions；
3. 论文所用 KL/objective identity 的 model-family assumptions；
4. P2、Min-SNR 或 EDM 的任意有效 weight 不会自动满足该 theorem；
5. ELBO interpretation 不推出最佳 FID、感知质量或 finite-network optimization。

这个 theorem 的价值在于把一类 heuristic weighting 重新解释为“对不同噪声增强数据的 likelihood objectives 做混合”。它给出语义边界，而不是替所有 loss design 排序。当前本地来源按 preprint 对待；正式引用时应再次核对版本状态。

***

## 9. EDM：从二阶矩推导 preconditioning

![EDM 的输入、skip、输出规范化，以及 proposal 与 weight 的分离](/images/diffusion/d5_edm_preconditioning.png)

EDM 使用 additive coordinate：

$$
x=y+n,
\qquad
\mathbb E[y]=\mathbb E[n]=0,
$$

$$
\operatorname{Var}(y)=\sigma_{data}^2,
\qquad
\operatorname{Var}(n)=\sigma^2,
\qquad y\perp n.
$$

其 denoiser 写成

$$
\boxed{
D_\theta(x;\sigma)
=c_{skip}(\sigma)x
+c_{out}(\sigma)
F_\theta(c_{in}(\sigma)x;c_{noise}(\sigma)).
}
$$

这不是简单地把 target 从 $\epsilon$ 改成 $x_0$。skip connection、raw network input、raw network output target、loss weight 和 noise conditioning 都被显式设计。

### 9.1 让 network input 具有单位二阶矩

因为

$$
\operatorname{Var}(x)=\sigma_{data}^2+\sigma^2,
$$

取

$$
\boxed{
c_{in}(\sigma)
=\frac1{\sqrt{\sigma^2+\sigma_{data}^2}}
}
$$

使 $c_{in}x$ 的 variance 为 1。这里使用的是数据和噪声独立、零均值、各向同性的二阶矩模型；它没有声称真实图像每个 feature 都严格 Gaussian。

### 9.2 最小方差的线性 skip

先让 skip 分支承担线性可预测部分。最小化

$$
R(c)=\mathbb E\|y-cx\|^2
$$

对每个标量维度求导：

$$
R'(c)=-2\operatorname{Cov}(y,x)+2c\operatorname{Var}(x).
$$

由于 $x=y+n$，

$$
\operatorname{Cov}(y,x)=\operatorname{Var}(y)=\sigma_{data}^2.
$$

令导数为 0：

$$
\boxed{
c_{skip}(\sigma)
=\frac{\sigma_{data}^2}
{\sigma^2+\sigma_{data}^2}.
}
$$

它也是在上述二阶矩假设下，由 noisy observation 对 clean signal 做最佳线性估计的系数。

### 9.3 让 residual target 具有单位二阶矩

把 residual 写为

$$
r=y-c_{skip}x.
$$

代入最优线性回归的 residual variance：

$$
\begin{aligned}
\operatorname{Var}(r)
&=\operatorname{Var}(y)
-\frac{\operatorname{Cov}(y,x)^2}{\operatorname{Var}(x)}\\
&=\sigma_{data}^2
-\frac{\sigma_{data}^4}{\sigma^2+\sigma_{data}^2}\\
&=\frac{\sigma^2\sigma_{data}^2}
{\sigma^2+\sigma_{data}^2}.
\end{aligned}
$$

因此取 residual standard deviation

$$
\boxed{
c_{out}(\sigma)
=\frac{\sigma\sigma_{data}}
{\sqrt{\sigma^2+\sigma_{data}^2}}
}
$$

并让 raw network 拟合

$$
F^\star(x,\sigma)
=\frac{y-c_{skip}x}{c_{out}},
$$

则 $F^\star$ 的 variance 为 1。

### 9.4 让 raw-network error 的有效权重为 1

考虑 denoising loss

$$
\lambda(\sigma)\|D_\theta(x;\sigma)-y\|^2.
$$

由于

$$
D_\theta-y
=c_{out}(F_\theta-F^\star),
$$

loss 变成

$$
\lambda(\sigma)c_{out}(\sigma)^2
\|F_\theta-F^\star\|^2.
$$

令 raw-network MSE 的显式系数为 1，得到

$$
\boxed{
\lambda(\sigma)=\frac1{c_{out}^2}
=\frac{\sigma^2+\sigma_{data}^2}
{(\sigma\sigma_{data})^2}.
}
$$

至此，$c_{in},c_{skip},c_{out},\lambda$ 都可由二阶矩规范化动机推出。EDM 使用的

$$
\boxed{c_{noise}=\tfrac14\log\sigma}
$$

则是 noise-conditioning 的经验尺度选择，不由上述二阶矩唯一决定。

### 9.5 proposal 仍是独立选择

EDM 训练中还选取

$$
\log\sigma\sim\mathcal N(P_{mean},P_{std}^2),
$$

论文默认参数为 $P_{mean}=-1.2,P_{std}=1.2$。这是 training proposal；它与 $\lambda(\sigma)$ 不是同一个函数。前者决定哪些 $\sigma$ 更常进入 batch，后者决定给定 $\sigma$ 后 denoising error 如何加权。

### 9.6 EDM 给出的真正方法论

EDM 不只是一个新 sampler 或一组超参数。它提出一种更可审计的 design-space 分解：

- 用 $\sigma$ 明确 noise coordinate；
- 用 preconditioning 管理网络输入、输出和 skip 尺度；
- 用 $p_{train}(\sigma)$ 管理训练样本分配；
- 用 $\lambda(\sigma)$ 管理 objective；
- 用独立 grid 与 solver 管理 inference。

这种分解比“换了一个 schedule”更准确，也让消融实验能够回答具体组件的问题。

***

## 10. Data scaling、resolution 与实际 log-SNR

设原数据被放大为 $x_0'=c x_0$。若 corruption 仍写为

$$
z=a x_0'+b\epsilon,
$$

从平均 signal power 与 noise power 比较，effective SNR 变成

$$
\mathrm{SNR}'=c^2\frac{a^2}{b^2}=c^2\mathrm{SNR},
$$

所以

$$
\boxed{\lambda'=\lambda+2\log c.}
$$

若文献使用 amplitude ratio 而不是 power ratio，shift 会少一个 factor 2。因此任何“平移 log-SNR”的 recipe 都必须先声明 SNR convention。

图像 resolution 变化更复杂。相同 per-pixel variance 不代表相同的可感知 information destruction：高分辨率图像有不同的 spatial frequency content，encoder latent 的缩放也会改变 signal statistics。Chen 等人的 resolution-dependent noise scheduling 提供经验方案，但它不是只由图像边长推出的普适 theorem。

实践上至少要保持以下对象一致：

- 训练数据的 normalization 与 $\sigma_{data}$ 估计；
- 训练和采样两侧对 latent/data 的 scale convention；
- log-SNR shift 的 amplitude/power 定义；
- pretrained autoencoder 的 latent scaling factor；
- endpoint noise range 与目标参数化。

否则名义上相同的 schedule 会对应不同的实际 corruption difficulty。

***

## 11. Exact-zero terminal SNR：endpoint、target 与 sampler 必须联动

VP terminal endpoint 若满足

$$
a_T=0,
\qquad
b_T=1,
$$

则

$$
\boxed{z_T=\epsilon.}
$$

此时 epsilon target 就是网络输入本身。预测 $\epsilon$ 不再要求从输入中恢复任何 data-dependent quantity；而

$$
\boxed{v_T=-x_0}
$$

仍保留 clean-data target。与此同时，公式

$$
\hat x_0=\frac{z_T-b_T\hat\epsilon}{a_T}
$$

因除以 0 而不可用。

这揭示了 terminal mismatch 的三部分：

1. 训练 path 是否真的到达 zero SNR；
2. chosen target 在 endpoint 是否有非退化语义；
3. sampler 第一步是否按这个 endpoint 和 target 正确初始化、转换与更新。

Lin 等人的建议把 rescaled zero-terminal-SNR schedule、$v$-prediction 和 sampler terminal handling 联动起来。只修改 beta schedule，却继续用依赖 $1/a_T$ 的 epsilon-to-clean conversion，不能称为完整修复。

数值实现还要显式处理 $\mathrm{SNR}=0$ 的 mask/limit。像

$$
w_\epsilon
=\frac{\min(\mathrm{SNR},\gamma)}{\mathrm{SNR}}
$$

在正 SNR 上等于 $\min(1,\gamma/\mathrm{SNR})$，但直接把 $\mathrm{SNR}=0$ 代入前一形式会得到 $0/0$。代数极限正确不代表浮点表达会自动正确。

***

## 12. Self-conditioning：把迭代历史接回网络

标准训练通常从单个 $(z_t,t)$ 预测 $x_0$ 或等价 target；采样时，网络却在一列噪声水平上重复运行，上一时刻已经产生一个 clean estimate。self-conditioning 把这个 estimate 作为额外输入：

$$
\hat x_0^{(k)}
=f_\theta(z_{t_k},\hat x_0^{(k-1)},t_k).
$$

Analog Bits 使用的训练近似是：

1. 约一半样本把 self-condition 设为 0；
2. 其余样本先计算 $\tilde x_0=f_\theta(z_t,0,t)$；
3. 对 $\tilde x_0$ stop-gradient；
4. 再计算并训练 $f_\theta(z_t,\tilde x_0,t)$。

用伪代码表示：

```python
condition = zeros_like(z_t)
if random_uniform() < 0.5:
    preliminary = model(z_t, condition, t)
    condition = stop_gradient(preliminary)
prediction = model(z_t, condition, t)
loss = target_loss(prediction, target)
```

stop-gradient 防止 preliminary pass 形成另一条反向传播路径。论文报告额外训练时间低于 25%，但这是具体实现的经验结果。

self-conditioning 不改变 forward corruption，也不是关于 diffusion state 的新 theorem。它是一种 architecture/training interface，用来缩小“训练时孤立单步预测”与“采样时迭代使用历史估计”之间的差异。把它解释成 learned fixed-point iteration 可以提供直觉，但网络输入和噪声水平每步都变化，因此不能直接套用静态 contraction theorem。

***

## 13. EDM2：为什么初始 normalization 不会自动维持

EDM 的二阶矩推导让 raw input、raw target 和显式 loss coefficient 在初始化语义上较均衡。然而深层网络开始训练后，parameter norms、activation magnitudes、optimizer moments 和 per-noise loss 都会共同演化。**initially normalized 不等于 dynamically balanced。**

EDM2 关注的不是新的 diffusion probability identity，而是一组训练动力学问题：

- residual/concatenation 等层可能累积 activation magnitude；
- learned weights 的 norm 可能增长；
- ordinary weight normalization 下，effective learning rate 会随 weight norm 改变；
- 不同 noise levels 的 raw loss 可能在训练中重新失衡；
- fixed EMA decay 的实际 smoothing horizon 依赖训练长度、batch、architecture 和后续 guidance。

对应方案包括：

1. 对 learned/fixed-function layers 做 expected-magnitude preservation；
2. forced weight normalization，并分析 tangent-space parameter update；
3. 用 noise-dependent learned uncertainty/log-variance 调节 loss balance；
4. 用 power-function EMA profile 与 post-hoc reconstruction，减少预先猜固定 decay 的代价。

这些是 architecture/optimization system 的经验设计和局部分析，不能改写成“EDM2 证明所有 diffusion training 都必须如此”。它们更重要的教学意义是：即使 target-space objective 已经代数校准，训练过程本身仍是一个动态系统。

***

## 14. 从公式到代码：可审计的最小实现

配套脚本不训练图像模型，而是把本章最容易写错的接口做成可执行函数。

| 函数                                                     | 对应公式/职责                               |
| ------------------------------------------------------ | ------------------------------------- |
| `affine_corruption`                                    | $z=ax_0+b\epsilon$                    |
| `epsilon_from_x0`, `x0_from_epsilon`                   | target conversion 与 singularity guard |
| `conditional_score_from_epsilon`                       | $s^\star=-\epsilon/b$                 |
| `v_target`, `endpoints_from_v`                         | 保留 $a^2+b^2$ 的一般 inverse              |
| `path_velocity_target`, `endpoints_from_path_velocity` | general $u$ 与 $\Delta$ boundary       |
| `mse_multipliers`                                      | target-MSE 到 $x_0$-MSE 的倍数            |
| `min_snr_target_weights`, `p2_multiplier`              | 两类不可混同的 weighting                     |
| `importance_estimate`                                  | proposal correction                   |
| `edm_coefficients`, `edm_raw_target`                   | EDM preconditioning 与 raw target      |
| `self_conditioned_prediction`                          | stop-gradient 的数组级接口说明                |

运行：

```powershell
# 本地验证脚本暂未公开
```

脚本固定 seed `20260714`，检查结果包括：

- VP $x_0/\epsilon$ inverse error 约 $8.9\times10^{-16}$；
- epsilon/$v$ MSE scaling error 约 $9.1\times10^{-13}$ 以内；
- score scaling error 约 $4.7\times10^{-10}$；
- general path velocity inverse error低于 $1.7\times10^{-15}$；
- off-circle 省略 denominator 的平均 gap 约 $0.50$；
- Min-SNR 各 target objective equivalence error低于 $1.1\times10^{-14}$；
- EDM input/target unit-variance Monte Carlo error约 $5.1\times10^{-3}$、$2.4\times10^{-3}$；
- $\lambda c_{out}^2=1$ 与 zero-terminal identities 达到浮点精度。

这些 checks 验证代数和二阶矩，不验证 FID、模型收敛速度或方法间的经验排名。

### 14.1 与官方实现的对应边界

本地固定版本代码用于核对公式，不作为可直接训练的仓库镜像：

| 实现                       | 本章核对内容                                       | 许可边界            |
| ------------------------ | -------------------------------------------- | --------------- |
| Progressive Distillation | $v$、log-SNR 与 weight                         | Apache-2.0      |
| P2                       | baseline multiplier                          | MIT             |
| Min-SNR                  | target-specific weights                      | 仓库未发现明确许可，仅本地阅读 |
| EDM                      | proposal、weight、preconditioning              | CC BY-NC-SA 4.0 |
| EDM2                     | magnitude-preserving layers、loss balance、EMA | CC BY-NC-SA 4.0 |

公式说明代码由本教程独立实现；发布教程时不应把受限官方代码打包成自己的可再许可附件。

***

## 15. 技术演进：问题怎样推动设计空间拆分

| 时间                            | 暴露的问题                                                           | 解决方案                                                        | 新局限与下一步                                              |
| ----------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------- |
| 2020 DDPM                     | reverse mean 需要稳定、简单的训练接口                                       | epsilon simple loss                                         | 它改变 ELBO weighting；端点 conditioning 不均                |
| 2021 Improved DDPM            | linear path 浪费步数，variance/likelihood/sample quality 互相牵制        | cosine path、learned variance、hybrid loss、proposal           | 多个设计轴仍常被统称 schedule                                  |
| 2021 VDM                      | 离散 beta conventions 难比较                                         | continuous SNR/log-SNR 与 learned schedule                   | continuous equivalence 不消除 estimator/finite-model 差异 |
| 2022 Salimans--Ho             | low-SNR 下 epsilon-to-clean 转换不稳，少步蒸馏需更平衡 target                 | VP $v$-prediction、max-SNR metric                            | 旋转公式限于 VP；蒸馏另有 teacher/model error                   |
| 2022 P2                       | capacity 可能过度用于 high-SNR imperceptible cleanup                  | baseline 乘 perception-prioritized factor                    | stage hypothesis 与收益是经验性的，且依赖 baseline               |
| 2022 EDM                      | parameterization、training density、loss 和 sampler conventions 混杂 | $\sigma$-coordinate、preconditioning、proposal/weight/grid 拆分 | 初始二阶矩平衡会在训练中漂移                                       |
| 2022 Analog Bits              | sampler 的历史 clean estimate 没有进入单步网络                             | self-conditioning                                           | 额外 forward pass；没有普适改进 theorem                       |
| 2023 Min-SNR                  | timesteps 共享参数时梯度冲突、收敛慢                                         | clipped common $x_0$-metric                                 | $\gamma$ 是经验超参数，冲突解释不是完整优化理论                         |
| 2023 Kingma--Gao              | objective、weight 与 schedule 的语义缺乏统一                             | weighted log-SNR integral、monotonic ELBO interpretation     | theorem 有 monotonicity/boundary 条件；当前按 preprint 引用   |
| 2023--2024 data/endpoint work | resolution scaling 与 terminal train-inference mismatch          | log-SNR shift、normalization、zero-SNR 联动修复                   | recipe 依赖数据、target 和 sampler convention              |
| 2024 EDM2                     | magnitude、effective LR、loss balance 与 EMA 在训练中漂移                | MP design、forced norm、learned balance、post-hoc EMA          | 是 architecture-specific empirical system，不是概率定理      |
| 2024 教学综合                     | “noise schedule”承担太多含义                                          | 按 path、input、target、weight、proposal/grid 拆解                 | blog 用于组织理解，核心 claim 仍回到 primary sources             |

这条历史线不是“新方法不断替代旧方法”，而是研究对象逐渐分层：从一个 beta schedule 和一个 MSE，拆成概率路径、坐标、目标、测度、估计器、网络尺度和数值求解器。2025--2026 的增量检索尚未发现需要强行加入本章主线、且已形成广泛影响的独立 weighting/parameterization 结果；更近的 velocity/flow 演进放在 D9。

***

## 16. 常见错误与边界检查

### 错误 1：$x_0,\epsilon,v$ 可转换，所以 constant MSE 相同

转换会缩放误差。至少要用第 5 节的 target-weight 表校准到同一个 metric。

### 错误 2：所有 target 经加权后训练过程完全相同

population objective 等价不固定网络输出尺度、gradient variance、optimizer state、capacity allocation 或 endpoint conditioning。

### 错误 3：一般 affine path 中仍写 $x_0=az-bv$

一般公式必须除以 $a^2+b^2$。只有 VP circle 才能省略 denominator。

### 错误 4：Salimans--Ho $v$ 是任意 time coordinate 的 velocity

它是 VP angle 的 velocity。相对一般 time 的速度还含 $\dot\phi(t)$，一般 path 则是 $u=\dot ax_0+\dot b\epsilon$。

### 错误 5：conditional score target 就是 marginal score sample

前者依赖 $x_0$。只有对 $x_0\mid z_t$ 取 conditional expectation 后才得到 marginal score。

### 错误 6：P2 就是 Min-SNR

P2 是乘在 baseline 上的 factor；Min-SNR 定义 clipped common $x_0$-metric。两者公式、动机和比较基准都不同。

### 错误 7：max-SNR 与 Min-SNR 只是名字不同

一个抬高 low-SNR floor，一个截断 high-SNR ceiling，作用方向相反。

### 错误 8：training proposal 就是 objective weight

proposal 是抽样 density。做 importance correction 后可以改变 proposal 而保持 objective；不校正才会一并改变 objective。

### 错误 9：training proposal 应复制 sampling grid

前者控制 stochastic-gradient estimator，后者控制 reverse numerical solver。二者没有相等要求。

### 错误 10：continuous schedule invariance 意味着 schedule 不重要

该结论只针对 fixed endpoints、同一 $\lambda$-indexed kernels 与 exact continuous integral。MC variance、finite discretization、network conditioning 和 sampler grid 仍会变化。

### 错误 11：所有 weighting 都有 noisy-data ELBO 解释

Kingma--Gao 结果要求 monotonicity、boundary 和 normalization 条件。不能对任意 P2、Min-SNR 或 EDM effective weight直接套用。

### 错误 12：EDM 的所有系数都由概率模型唯一推出

$c_{in},c_{skip},c_{out},\lambda$ 有明确二阶矩动机；$c_{noise}$、training proposal parameters、$\sigma_{data}$ 估计和网络细节仍含设计选择。

### 错误 13：EDM preconditioning 保证训练全程各层 magnitude 平衡

它规范初始接口。EDM2 正是针对训练中 activation、weight、loss 与 effective LR 的漂移。

### 错误 14：zero-terminal-SNR 只需 rescale schedule

target 语义、singular conversion 和 sampler terminal step 必须一起修改。

### 错误 15：self-conditioning 改变了 forward diffusion

它只改变网络的额外输入和训练/采样接口，forward corruption 不变。

***

## 17. 本章给 D6 与 D9 准备了什么

对 D6，本章固定了三个边界：

- sampling grid 不等于 training proposal；
- output target 需要先转换成 sampler 所需的 denoiser、score 或 drift 接口；
- low-SNR conversion conditioning 会与有限步 solver/model error 相互作用。

[D6](/blog/diffusion/d6-sampling-solvers/) 将在同一 trained model 上比较 DDPM ancestral、DDIM、predictor--corrector、Heun、DPM-Solver 等 update rules，并分析 NFE 与离散误差。本章不提前比较 solver order 或 grid benchmark。

对 D9，本章只建立

$$
u_t=\dot a_tx_0+\dot b_t\epsilon
$$

及其 coordinate dependence，用来防止把 VP $v$ 无条件推广。D9 才会讨论 flow matching、rectified flow、progressive distillation、consistency 与少步生成如何重新选择 path、target 和 objective。

D12 则负责更严格的 end-to-end error 分解与 convergence boundary。本章关于 finite network 的讨论是必要警告，不构成完整优化或采样收敛定理。

***

## 18. 章节小结

1. 在 $z=ax_0+b\epsilon$ 下，$x_0,\epsilon$、conditional score 和 $v$ 可以线性转换，但转换在 endpoints 可能奇异或病态。
2. constant epsilon-MSE 等价于 $\mathrm{SNR}$-weighted $x_0$-MSE；VP constant $v$-MSE 等价于 $(1+\mathrm{SNR})$-weighted $x_0$-MSE。
3. $v=a\epsilon-bx_0$ 的无 denominator inverse 与 angular-velocity 解释只在 VP normalization 下成立。
4. 要比较不同 targets，必须先指定共同 metric，再换算 target-space weights；objective 等价仍不保证 finite-network optimization 等价。
5. DDPM simple、max-SNR、P2 与 Min-SNR 是不同 weighting families，不能按名称混写。
6. corruption path、noise coordinate、objective measure、training proposal、network preconditioning 与 sampling grid 是不同设计轴。
7. fixed-endpoint continuous objective 可写成 log-SNR integral；改变 proposal 并做 importance correction不改 objective，但会改变 estimator variance。
8. monotonic weighting 在特定条件下可解释为 noise-augmented data ELBO 的混合；该结论不覆盖任意非单调 weight 或感知最优性。
9. EDM 的 $c_{in},c_{skip},c_{out}$ 与 $\lambda$ 可由二阶矩规范化逐步推出；proposal 与 noise embedding scale 仍是独立选择。
10. data scaling 会平移 effective log-SNR；zero-terminal-SNR 必须联动 target、conversion 和 sampler。
11. self-conditioning 利用迭代历史，EDM2 则说明初始数值平衡不会自动维持整个训练过程。

***

## 19. 思考题

1. **共同 metric。** 设 VP path 使用 constant score-MSE。将它换算成 $x_0$-metric，并分析 $\mathrm{SNR}\to0$ 与 $\mathrm{SNR}\to\infty$ 两端的权重。若想得到 constant $x_0$-MSE，应给 score loss 乘什么？
2. **off-circle $v$。** 取 $a=2,b=1$，直接计算 $(z,v)$ 到 $(x_0,\epsilon)$ 的 inverse。省略 $a^2+b^2$ 后，恢复值会被怎样系统缩放？
3. **coordinate velocity。** 令 VP angle 为 $\phi(t)=t^2\pi/2$。求 $dz_t/dt$ 与 $v_{\phi(t)}$ 的关系，并写出 constant time-velocity MSE 对 angular-$v$-MSE 的隐式 weight。
4. **proposal variance。** 对积分 $I=\int f(\lambda)d\lambda$，证明在 $f\ge0$ 且可归一化时，零方差 importance proposal 与 $f$ 成正比。若 estimator 换成 gradient vector，为什么最优标量 proposal 不再只由 loss value 决定？
5. **schedule invariance 的反例边界。** 构造两个具有相同 $\lambda$ endpoints、不同 $t\mapsto\lambda(t)$ 的 path parameterizations。说明 exact $d\lambda$ objective 相同，但 uniform-$t$ 未校正 estimator 的 objective 与 variance为何都可不同。
6. **monotonic measure。** 从第 8 节 integration-by-parts 式出发，讨论 $r'(t)$ 在一段区间为负时为何不能解释为 augmentation probability density。是否可能用 signed measure 保留代数 identity？它还保留概率解释吗？
7. **EDM skip。** 不假设 $y,n$ 独立，只给出 $\operatorname{Cov}(y,n)=C$。重新推导最优 scalar $c_{skip}$。原 EDM 系数在哪些额外条件下恢复？
8. **非各向同性数据。** 若 clean covariance 为矩阵 $\Sigma_{data}$，noise covariance 为 $\sigma^2I$，最佳 linear skip 将变成什么矩阵？为什么实际网络仍常使用 scalar preconditioning？
9. **zero-SNR target。** 在 $a_T=0,b_T=1$ 处分别求 $\epsilon,x_0,v$ targets 的 conditional expectation given $z_T$。哪一个 target 的最优 predictor只需复制输入？哪一些只能退化为数据均值？
10. **有限网络非等价。** 设计一个共享单参数的两噪声水平线性模型，使两种经 population value 换算的 target parameterizations 产生不同的 per-sample gradient variance。需要哪些额外重参数化才能让优化轨迹也相同？
11. **self-conditioning。** 把 sampling update 抽象成 $h_{k+1}=F_t(h_k,z_t)$。若想用 fixed-point contraction 解释 self-conditioning，需要对 $F_t$ 加什么条件？真实 diffusion sampling 为什么通常不满足一个固定 map 的假设？
12. **训练期漂移。** 即使每个 noise level 在初始化时满足 unit raw-target variance，Adam 的 second moment、weight norm 与 residual depth如何让 effective update scale 再次依赖 noise level？提出一个只记录统计量、不做大规模实验的诊断方案。

***

## 20. 本章来源与继续阅读

1. Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2006.11239 "官方论文页面")。
2. Alex Nichol, Prafulla Dhariwal. *Improved Denoising Diffusion Probabilistic Models*. ICML, 2021. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2102.09672 "官方论文页面")。
3. Diederik P. Kingma et al. *Variational Diffusion Models*. NeurIPS, 2021. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2107.00630 "官方论文页面")。
4. Tim Salimans, Jonathan Ho. *Progressive Distillation for Fast Sampling of Diffusion Models*. ICLR, 2022. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2202.00512 "官方论文页面")。
5. Jooyoung Choi et al. *Perception Prioritized Training of Diffusion Models*. CVPR, 2022. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2204.00227 "官方论文页面")。
6. Tero Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS, 2022. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2206.00364 "官方论文页面")。
7. Ting Chen et al. *Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning*. ICLR, 2023. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2208.04202 "官方论文页面")。
8. Tiankai Hang et al. *Efficient Diffusion Training via Min-SNR Weighting Strategy*. ICCV, 2023. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2303.09556 "官方论文页面")。
9. Diederik P. Kingma, Ruiqi Gao. *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation*. 2023 preprint，本教程按本地版本状态保守引用。本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2303.00848 "官方论文页面")。
10. Ting Chen. *On the Importance of Noise Scheduling for Diffusion Models*. 2023 preprint，本教程将 resolution recipe 视为经验研究。本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2301.10972 "官方论文页面")。
11. Shanchuan Lin et al. *Common Diffusion Noise Schedules and Sample Steps are Flawed*. WACV, 2024. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2305.08891 "官方论文页面")。
12. Tero Karras et al. *Analyzing and Improving the Training Dynamics of Diffusion Models*. CVPR, 2024. 本地论文（补充材料暂未公开），[结构化笔记](https://arxiv.org/abs/2312.02696 "官方论文页面")。
13. Sander Dieleman. *Noise Schedules Considered Harmful*. 2024. 本地结构化笔记与页面存档入口（补充材料暂未公开）。该 blog 只承担概念组织职责，公式与历史 claim 由上述 primary sources 支撑。

完整独立代数复核见 parameterization/weighting derivation ledger（补充材料暂未公开），数值交叉检查见 parameterization\_weighting\_checks.py（补充材料暂未公开），本章 claim--evidence mapping 见 D5 chapter source packet（补充材料暂未公开）。
