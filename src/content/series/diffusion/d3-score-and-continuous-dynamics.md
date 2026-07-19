---
title: 去噪网络真正学到的是什么
description: >-
  连接 score matching、denoising、reverse-time SDE 与 probability-flow
  ODE，解释去噪网络真正学习的向量场。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: diffusion
order: 3
slug: d3-score-and-continuous-dynamics
tags:
  - diffusion
  - score-matching
  - sde
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦 noisy marginal score 的统计含义与连续时间动力学，并明确相同 marginals 不等于相同 path law。
---
DDPM 把训练写成噪声预测，但“预测噪声”只是参数化方式。给定一个 noisy state，平方回归真正得到的是条件平均噪声；这个条件平均又恰好等价于 noisy marginal density 的 score。

一旦知道每个噪声时刻的 score，反向动力学便不再局限于 DDPM 的离散 Gaussian chain。它可以写成 reverse-time SDE，也可以写成具有相同边缘分布的 probability-flow ODE。

## 1. Score 绕开了密度的归一化常数

对一个具有可微密度 $p(x)$ 的随机变量，score 定义为 log-density 对状态的梯度：

$$
s_p(x)
=
\nabla_x\log p(x).
$$

它是一个与 $x$ 同维的向量，指向 log-density 在局部上升最快的方向。若能量模型写成

$$
p_\theta(x)
=
\frac{\exp[-E_\theta(x)]}{Z_\theta},
$$

其中 $Z_\theta$ 是难以计算的归一化常数，那么

$$
\nabla_x\log p_\theta(x)
=
-\nabla_x E_\theta(x)
$$

不再含 $Z_\theta$。这使我们有可能学习密度的局部几何，而不必先求出完整 normalized density。

Hyvärinen 在 2005 年论文 Estimation of Non-Normalized Statistical Models by Score Matching（补充材料暂未公开） 中提出 score matching。他从模型 score $s_\theta$ 与 data score $s_{\mathrm{data}}$ 的 Fisher divergence 出发：

$$
\mathcal J_{\mathrm{explicit}}(\theta)
=
\frac12
\mathbb E_{p_{\mathrm{data}}}
\left[
\|s_\theta(X)-s_{\mathrm{data}}(X)\|^2
\right].
$$

真实 data score 未知，看起来仍然无法训练。Hyvärinen 通过 integration by parts，在密度可微、相关积分存在且边界项消失时，把它改写成只含 $s_\theta$ 及其 divergence 的 implicit objective。这个结果解决了未知归一化常数，却需要对模型 score 再求一次导数，在高维神经网络中并不便宜。

下图画出了一个平滑密度的 score field。向量描述的是局部概率上升方向，不是从某个噪声点到某张训练图像的唯一连线。

![平滑分布上的 score field](/images/diffusion/d3_score_field.png)

## 2. Denoising target 为什么等于 noisy marginal score

Vincent 在 2011 年论文 [*A Connection Between Score Matching and Denoising Autoencoders*](https://doi.org/10.1162/NECO_a_00142 "官方论文页面") 中证明，score 可以通过 clean-corrupted pairs 学习，而不必计算网络输出对输入的 divergence。

设 clean sample $X_0\sim p_{\mathrm{data}}$，并通过已知 corruption kernel $q_t(x_t\mid x_0)$ 得到 $X_t$。Noisy marginal density 是

$$
p_t(x)
=
\int
q_t(x\mid x_0)
p_{\mathrm{data}}(x_0)\,dx_0.
$$

对 $x$ 求导并把导数移入积分，可得

$$
\nabla_x p_t(x)
=
\int
q_t(x\mid x_0)
p_{\mathrm{data}}(x_0)
\nabla_x\log q_t(x\mid x_0)
\,dx_0.
$$

两边除以 $p_t(x)$，积分中的权重正好变成 posterior $p(x_0\mid x)$，于是

$$
\boxed{
\nabla_x\log p_t(x)
=
\mathbb E
\left[
\nabla_x\log q_t(X_t\mid X_0)
\mid X_t=x
\right].
}
$$

这条 conditional-to-marginal identity 是 denoising score matching 的核心。单个 conditional target 依赖具体 $X_0$，可能很 noisy；平方回归在总体上的最优解，则是给定当前 $X_t=x$ 后的条件平均，也就是 marginal score。

回到 DDPM。定义

$$
a_t=\sqrt{\bar\alpha_t},
\qquad
b_t=\sqrt{1-\bar\alpha_t},
$$

于是

$$
X_t=a_tX_0+b_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I_d).
$$

Gaussian conditional score 为

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{x_t-a_tx_0}{b_t^2}
=
-\frac{\epsilon}{b_t}.
$$

若 $\epsilon^*(x,t)=\mathbb E[\epsilon\mid X_t=x]$ 是 noise MSE 的 population optimum，那么

$$
\boxed{
\nabla_x\log p_t(x)
=
-\frac{\epsilon^*(x,t)}{b_t}.
}
$$

所以 DDPM 的 noise predictor、denoiser 与 score network 在总体极限下可以互相转换。它们训练时使用的权重和数值尺度可能不同，但指向的是同一个 noisy marginal field。

同一恒等式还能给出 posterior denoising mean：

$$
\boxed{
\mathbb E[X_0\mid X_t=x]
=
\frac{x+b_t^2\nabla_x\log p_t(x)}
{a_t}.
}
$$

这类“noisy observation 加上 score correction”通常称为 Tweedie identity；Efron 在 2011 年论文 [*Tweedie's Formula and Selection Bias*](https://doi.org/10.1198/jasa.2011.tm11181 "官方论文页面") 中系统讨论了 Gaussian observation 下的经验 Bayes 形式。

![Conditional denoising target、marginal score 与 posterior mean](/images/diffusion/d3_dsm_tweedie.png)

## 3. 为什么必须在许多噪声尺度上学习

直接在原始数据分布上估计 score 有两个问题。自然图像常集中在高维空间中的低维区域附近，ambient density 可能非常尖锐；不同 modes 之间的低密度区域又几乎没有训练样本，而采样过程恰恰必须穿过这些区域。

Song 与 Ermon 在 NeurIPS 2019 论文 [*Generative Modeling by Estimating Gradients of the Data Distribution*](https://arxiv.org/abs/1907.05600 "官方论文页面") 中提出 Noise Conditional Score Network（NCSN）。他们不只学习一个数据端 score，而是用一列 $\sigma_1>\cdots>\sigma_L>0$ 的 Gaussian perturbation 构造平滑分布 $p_{\sigma_i}$，让同一个网络 $s_\theta(x,\sigma_i)$ 学习每个噪声尺度的 score。

大噪声会把彼此分离的 modes 连接起来，使全局移动更容易；小噪声再把样本拉回数据细节。NCSN 使用 annealed Langevin dynamics 从大噪声逐级走向小噪声。

这与 DDPM 的时间条件网络已经非常接近：两者都在许多噪声水平上学习 denoising direction。差别主要在 forward perturbation、训练权重与采样动力学。2021 年的 SDE 视角正是为了把这些表面不同的设计放到同一套连续时间对象中。

需要保留一个边界：score loss 是按 $p_t$ 加权的平均误差。低 loss 不意味着低密度区域的向量场处处准确，也不自动解决有限步 sampler 的 mixing 和离散化问题。

## 4. Score 怎样决定 reverse-time SDE

设连续时间 forward diffusion 满足

$$
dX_t
=
f(X_t,t)\,dt
+
g(t)\,dW_t,
\qquad
t\in[0,T].
$$

$f(x,t)$ 是 forward drift，$g(t)>0$ 是与状态无关的 scalar diffusion coefficient，$W_t$ 是标准 Brownian motion。记 $p_t(x)$ 为 $X_t$ 的边缘密度，并定义 time-dependent score

$$
s_t(x)
=
\nabla_x\log p_t(x).
$$

Anderson 在 1982 年论文 [*Reverse-Time Diffusion Equation Models*](https://doi.org/10.1016/0304-4149\(82\)90051-5 "官方论文页面") 中给出了反向扩散方程的经典来源。Song、Sohl-Dickstein、Kingma、Kumar、Ermon 与 Poole 在 ICLR 2021 论文 Score-Based Generative Modeling through Stochastic Differential Equations（补充材料暂未公开） 中把它用于生成建模：当时间从 $T$ 向 $0$ 积分，因此 $dt<0$ 时，reverse-time SDE 为

$$
\boxed{
dX_t
=
\left[
f(X_t,t)
-
g(t)^2s_t(X_t)
\right]dt
+
g(t)\,d\bar W_t.
}
$$

$\bar W_t$ 是反向时间中的 Brownian motion。Forward drift 本身不足以反演扩散，因为加噪让许多起点汇入同一 noisy state；额外的 $-g^2s_t$ 项利用当前边缘密度，把概率质量重新推回高密度区域。

实际模型用 $s_\theta(x,t)$ 代替未知 $s_t(x)$。因此 score error 会直接变成 drift error，并在整个反向积分过程中累积。上面的简洁公式还依赖 $g$ 与状态无关；一般 matrix-valued、state-dependent diffusion 需要额外的 diffusion-matrix divergence 项。

## 5. 同一组边缘还可以由确定性 ODE 生成

Song 等人 2021 还构造了 probability-flow ODE：

$$
\boxed{
\frac{dX_t}{dt}
=
f(X_t,t)
-
\frac12g(t)^2s_t(X_t).
}
$$

它与 forward SDE 具有相同的 one-time marginals $\{p_t\}_{t\in[0,T]}$。从 $p_T$ 反向积分这条 ODE，也可以得到 $p_0$；因为动力学是确定性的，还能使用 continuous change of variables 计算模型 likelihood。

系数从 reverse SDE 中的 $1$ 变成 $1/2$，不是随意折中。Fokker–Planck equation 中，SDE diffusion term 对 density evolution 的贡献可以改写为一个 score-dependent transport term；把随机扩散去掉时，只需把这部分补进 drift，便得到相同的边缘演化。

但相同 marginals 不表示相同 path law。Reverse SDE 在给定初值后仍有随机波动，probability-flow ODE 则为每个初值确定一条轨迹；它们的 transition kernels、时间相关性与 quadratic variation 都不同。

![Reverse SDE 与 probability-flow ODE 可以共享边缘分布，却具有不同路径](/images/diffusion/d4_sde_pf_paths.png)

到这里，noise prediction 的含义已经清楚：网络通过 denoising regression 学习 noisy marginal score，而 score 决定反向生成所需的向量场。离散 DDPM、score-based Langevin sampling、reverse SDE 与 probability-flow ODE 因此不再是四套互不相关的方法。

## 文献索引

| 时间   | 论文                                                                                       | 本章采用的内容                                                    |
| ---- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1982 | Anderson, *Reverse-Time Diffusion Equation Models*                                       | 提供 reverse-time diffusion 的经典历史来源                          |
| 2005 | Hyvärinen, *Estimation of Non-Normalized Statistical Models by Score Matching*           | 定义 score matching 并消除未知归一化常数                               |
| 2011 | Vincent, *A Connection Between Score Matching and Denoising Autoencoders*                | 证明 conditional denoising target 与 marginal score 的等价关系     |
| 2011 | Efron, *Tweedie's Formula and Selection Bias*                                            | 给出 Gaussian observation 下的 posterior mean correction       |
| 2019 | Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution*     | 用多噪声 score 与 annealed Langevin 处理 manifold 和 mixing 问题     |
| 2021 | Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* | 统一 score objective、reverse-time SDE 与 probability-flow ODE |
