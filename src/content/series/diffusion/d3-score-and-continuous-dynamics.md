---
title: 去噪器究竟学到了什么，又如何驱动反向扩散
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
第二章得到一个看起来近乎反常的结论：网络只需要从带噪样本 $X_t$ 中预测 forward noise $\varepsilon$，就能构造 reverse transition 的 mean。但这仍然没有回答更深的问题：

> 当网络在许多数据样本上最小化 noise MSE 时，它最终学到的究竟是什么函数？

答案不是“每个样本中实际加入的那一份随机噪声”。给定同一个 $x_t$，它可能由许多不同 $x_0$ 与 $\varepsilon$ 组合产生，网络不可能唯一恢复某一次 forward draw。平方损失最终逼近的是 conditional mean

$$
\mathbb E[\varepsilon\mid X_t=x_t].
$$

这个条件均值与 noisy data density 的 log-gradient 精确对应。统计学把这种对数据变量求导的 log-density gradient 称为 **score**。一旦 score 被估计出来，它不仅能解释 denoising，还能直接进入随机过程的反向 drift。

这条历史线索并非始于 DDPM。2005 年，Hyvärinen 提出 score matching；2011 年，Vincent 证明 denoising 可以训练 score；2019 年，Song 与 Ermon 用多噪声 score network 实现生成；2021 年，Song 等人把 NCSN 与 DDPM 统一进随机微分方程，并将经典的 diffusion time reversal 变成可计算的生成动力学。

整章的逻辑可以压缩为

$$
\text{denoising target}
\longrightarrow
\text{noisy marginal score}
\longrightarrow
\text{reverse-time SDE}
\longrightarrow
\text{probability-flow ODE}.
$$

## 1. 2005：不学习密度本身，而学习密度上升最快的方向

2005 年，Aapo Hyvärinen 在 JMLR 论文 Estimation of Non-Normalized Statistical Models by Score Matching（补充材料暂未公开） 中研究一个经典困难：许多灵活概率模型只能写成未归一化形式

$$
p_\theta(x)
=
\frac{q_\theta(x)}{Z(\theta)}.
$$

这里 $x\in\mathbb R^d$ 是数据，$q_\theta(x)$ 是可以计算的非负函数，$Z(\theta)$ 是让密度积分为 1 的 partition function。Maximum likelihood 需要处理 $Z(\theta)$，但在高维能量模型中它往往难以计算。

Hyvärinen 的关键观察是：如果对数据变量 $x$ 而不是参数 $\theta$ 求导，归一化常数会消失。他将

$$
\boxed{
s_p(x)
=
\nabla_x\log p(x)
}
$$

称为 density score。这里 $\nabla_x$ 是对数据坐标的梯度，$s_p(x)\in\mathbb R^d$ 是一个向量场。它与统计推断中常见的 parameter score $\nabla_\theta\log p_\theta(x)$ 不是同一个对象。

对未归一化模型，

$$
\nabla_x\log p_\theta(x)
=
\nabla_x\log q_\theta(x)
-\nabla_x\log Z(\theta)
=
\nabla_x\log q_\theta(x),
$$

因为 $Z(\theta)$ 不依赖 $x$。Score 因而保留了密度的局部几何，却不需要显式知道归一化常数。

直观上，score 指向 log-density 增长最快的方向。若当前位置落在低密度区，沿 score 移动会靠近局部高密度区域；在 mode 附近，score 的大小趋近零。下图展示一个三峰分布及其 Gaussian-smoothed 版本。红色箭头是 score field：它描述的是局部密度几何，而不是样本从某个确定起点出发的轨迹。

![原分布与平滑分布的 score field](/images/diffusion/d3_score_field.png)

Hyvärinen 2005 的 explicit score-matching objective 希望直接匹配 model score 与 data score：

$$
\mathcal J_{\mathrm{ESM}}(\theta)
=
\frac12
\mathbb E_{X\sim p_{\mathrm{data}}}
\left[
\left\|
s_\theta(X)
-\nabla_x\log p_{\mathrm{data}}(X)
\right\|^2
\right].
$$

$s_\theta$ 是神经网络或能量模型产生的 score field，$\mathbb E$ 表示对真实数据取期望。问题是，真实 data score 仍然未知。Hyvärinen 在 differentiability、integrability 与 boundary term 消失等条件下，通过 integration by parts 将目标改写为

$$
\mathcal J_{\mathrm{ISM}}(\theta)
=
\mathbb E_{p_{\mathrm{data}}}
\left[
\nabla\cdot s_\theta(X)
+\frac12\|s_\theta(X)\|^2
\right]
+C,
$$

其中 $\nabla\cdot s_\theta$ 是向量场的 divergence，常数 $C$ 与 $\theta$ 无关。这使未知 data score 从目标中消失，却引入了对 model score 求 divergence 的计算，通常涉及更高阶自动微分。

2005 年工作解决了“如何绕过 normalization constant”，但还没有给出后来 Diffusion 使用的简单噪声回归。这个接口来自 denoising。

## 2. 2011—2020：denoising target 的条件均值就是 noisy marginal score

2011 年，Pascal Vincent 在 Neural Computation 论文 [*A Connection Between Score Matching and Denoising Autoencoders*](https://doi.org/10.1162/NECO_a_00142 "官方论文页面") 中提出 denoising score matching，简称 DSM。它要解决的问题是：能否只用 clean/noisy sample pairs 训练 score，而不计算上一节的 divergence？

Vincent 原文以有限训练集形成的 empirical distribution 及其 Gaussian smoothing 为对象；下面使用 population density $p_0$ 重写同一恒等式，使它能直接与现代 Diffusion 记号对接。

设干净数据

$$
X\sim p_0,
$$

并加入标准差为 $\sigma>0$ 的 Gaussian noise：

$$
Y=X+\sigma\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I).
$$

$Y$ 的 marginal density 是

$$
p_\sigma(y)
=
\int
\mathcal N(y;x,\sigma^2I)
p_0(x)\,dx.
$$

它是原数据分布与 Gaussian kernel 的卷积。只要 $\sigma>0$，这个平滑分布通常比可能集中在低维流形上的原始分布更规则。

给定 clean sample $x$ 后，corruption kernel 的 conditional score 可以直接计算：

$$
\boxed{
\nabla_y
\log p(y\mid x)
=
\frac{x-y}{\sigma^2}
=
-\frac{\varepsilon}{\sigma}.
}
$$

单个 $(x,y)$ 对给出的 target 是随机的，它并不等于 marginal score $\nabla_y\log p_\sigma(y)$。Vincent 2011 的核心等价来自 conditional expectation：

$$
\boxed{
\mathbb E
\left[
\nabla_y\log p(Y\mid X)
\mid Y=y
\right]
=
\nabla_y\log p_\sigma(y).
}
$$

证明只需对 noisy marginal 求导。假设导数可以移入积分，

$$
\begin{aligned}
\nabla_y p_\sigma(y)
&=
\int p_0(x)\nabla_y p(y\mid x)\,dx\\
&=
\int p_0(x)p(y\mid x)
\nabla_y\log p(y\mid x)\,dx.
\end{aligned}
$$

除以 $p_\sigma(y)$ 后，积分中的权重正是 posterior $p(x\mid y)$，于是得到上面的条件期望恒等式。

因此 DSM 可以最小化

$$
\mathcal J_{\mathrm{DSM}}(\theta)
=
\frac12
\mathbb E
\left[
\left\|
s_\theta(Y,\sigma)
+\frac{\varepsilon}{\sigma}
\right\|^2
\right].
$$

平方损失的最优预测器是 target 在给定输入后的 conditional mean，所以 population optimum 满足

$$
s_\theta^*(y,\sigma)
=
\nabla_y\log p_\sigma(y).
$$

下图左侧展示了这一点：浅蓝色条件 target 含有很大随机性，但在相同 $y$ 附近取平均后，黑点落在红色 marginal score 上。网络学到的不是某一次随机 target，而是这些 target 的条件均值。

![DSM conditional target、marginal score 与 Tweedie identity](/images/diffusion/d3_dsm_tweedie.png)

同一恒等式还能解释 denoiser 的输出。通常称为 Tweedie formula 的 Gaussian posterior-mean identity 可以写成

$$
\boxed{
\mathbb E[X\mid Y=y]
=
y+\sigma^2\nabla_y\log p_\sigma(y).
}
$$

Efron 在 2011 年 JASA 论文 [*Tweedie’s Formula and Selection Bias*](https://doi.org/10.1198/jasa.2011.tm11181 "官方论文页面") 中给出了 normal-means 形式，并记录 Robbins 1956 将它归因于 Tweedie 的通信；Gaussian 多变量文献也常引用 Miyasawa 1961。因此更稳妥的称呼是 Tweedie/Miyasawa identity，而不武断声称单一优先权。

公式说明：posterior-mean denoiser 不只是把 $y$ 拉向某个训练样本，而是在当前 noisy marginal 的 score 方向上移动 $\sigma^2$ 的尺度。上图右侧中，直接计算的 posterior mean 与 $y+\sigma^2s_{p_\sigma}(y)$ 完全重合。

现在回到 Ho、Jain 与 Abbeel 的 NeurIPS 2020 论文 [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239 "官方论文页面")。对时刻 $t$，定义

$$
a_t=\sqrt{\bar\alpha_t},
\qquad
b_t=\sqrt{1-\bar\alpha_t},
$$

于是前向 channel 是

$$
X_t=a_tX_0+b_t\varepsilon.
$$

$q_t$ 表示 $X_t$ 的 aggregate marginal density。将 Vincent 的 conditional-score identity 应用到这个线性 Gaussian channel，可得

$$
\boxed{
\nabla_{x_t}\log q_t(x_t)
=
-\frac{
\mathbb E[\varepsilon\mid X_t=x_t]
}{
b_t
}.
}
$$

第二章已经说明，noise MSE 的 population optimum 是

$$
\varepsilon_\theta^*(x_t,t)
=
\mathbb E[\varepsilon\mid X_t=x_t].
$$

因此 DDPM noise predictor 与 score predictor 只差一个已知尺度：

$$
\boxed{
s_\theta(x_t,t)
=
-\frac{
\varepsilon_\theta(x_t,t)
}{
\sqrt{1-\bar\alpha_t}
}.
}
$$

这回答了本章的第一个问题：**DDPM 的去噪网络实际上在学习每个 noisy marginal $q_t$ 的 score。** Noise prediction、score prediction 与 posterior-mean denoising 不是三个彼此无关的技巧，而是同一个 Gaussian conditional identity 的不同参数化。

## 3. 2019：一组 noise-conditioned score 如何真正生成数据

知道 score 的统计含义，还不等于已经得到生成算法。2019 年，Yang Song 与 Stefano Ermon 在 NeurIPS Oral 论文 [*Generative Modeling by Estimating Gradients of the Data Distribution*](https://arxiv.org/abs/1907.05600 "官方论文页面") 中提出 Noise Conditional Score Network，简称 NCSN，并把多噪声 DSM 与 annealed Langevin dynamics 组合成生成模型。

这篇论文指出，直接学习原始 data score 有两个实际障碍。第一，图像数据可能接近低维 manifold，ambient-space density 甚至未必存在；第二，低密度区域几乎没有训练样本，score error 不受充分约束，但 sampler 恰好要从远离数据的区域进入各个 modes。

NCSN 不只训练一个噪声尺度，而是选择

$$
\sigma_1>\sigma_2>\cdots>\sigma_L>0,
$$

并对每个 $\sigma_i$ 定义 smoothed density

$$
p_{\sigma_i}(y)
=
\int
\mathcal N(y;x,\sigma_i^2I)
p_{\mathrm{data}}(x)\,dx.
$$

一个共享网络接收样本与 noise level：

$$
s_\theta(y,\sigma_i)
\approx
\nabla_y\log p_{\sigma_i}(y).
$$

大噪声让不同 modes 充分重叠，score field 更平滑，便于 sampler 从广泛区域移动；噪声逐渐减小时，目标分布恢复细节。这个过程不是在所有阶段学习同一个 density，而是在学习一条由 Gaussian smoothing 连接起来的 density path。

给定某个 noise level 的 score 后，NCSN 使用 Langevin update

$$
Y_{k+1}
=
Y_k
+\eta_i s_\theta(Y_k,\sigma_i)
+\sqrt{2\eta_i}\,Z_k,
\qquad
Z_k\sim\mathcal N(0,I),
$$

其中 $\eta_i>0$ 是该噪声层的 step size。这里采用常见的 time-rescaled convention；Song 与 Ermon 2019 Eq. (4) 写成 $\epsilon_i s/2+\sqrt{\epsilon_i}Z$，令 $\epsilon_i=2\eta_i$ 即与上式相同。第一项沿 score 提高 log-density，第二项保留随机探索。算法从最大 $\sigma_1$ 开始，在每个噪声层运行若干步，再逐渐下降到 $\sigma_L$。

Song 与 Ermon 并没有发明 Langevin dynamics；他们的 2019 贡献是把 multi-noise score estimation、noise conditioning 与 annealed sampling 组合成可扩展生成系统。有限步、无 Metropolis correction 的实现仍有 discretization bias，而且不同 modes 之间的 mixing 可能很慢。

到这里，NCSN 与 DDPM 已经显露出同一结构：二者都沿一条 noise path 训练 time/noise-conditioned network，也都利用 score-like direction 从噪声回到数据。但一个使用 annealed Langevin，另一个使用 learned reverse Markov chain。2021 年的 SDE 框架正是为了解释它们为何属于同一个连续体系。

## 4. 1982 与 2021：score 为什么出现在 reverse-time SDE 中

Song、Sohl-Dickstein、Kingma、Kumar、Ermon 与 Poole 在 ICLR 2021 Oral 论文 Score-Based Generative Modeling through Stochastic Differential Equations（补充材料暂未公开） 中，将离散 noise levels 改写为连续时间 $t\in[0,T]$。他们从 forward Itô SDE 出发：

$$
\boxed{
dX_t
=
f(X_t,t)\,dt
+g(t)\,dW_t.
}
$$

这里 $X_t\in\mathbb R^d$ 是随机状态，$f(x,t)\in\mathbb R^d$ 是 drift，控制确定性平均运动；$g(t)>0$ 是与状态无关的 scalar diffusion coefficient；$W_t$ 是标准 Wiener process。记 $p_t(x)$ 为 $X_t$ 的 density，并定义 time-dependent score

$$
s_t(x)=\nabla_x\log p_t(x).
$$

在这个框架中，DDPM 的连续极限对应 variance-preserving SDE：

$$
dX_t
=
-\frac12\beta(t)X_t\,dt
+\sqrt{\beta(t)}\,dW_t,
$$

而 NCSN 的 smoothing path 对应 variance-exploding SDE：

$$
dX_t
=
\sqrt{
\frac{d[\sigma(t)^2]}{dt}
}\,dW_t.
$$

两者选择不同的 forward dynamics，却共享同一个训练接口：在各时刻估计 $s_t(x)$。

密度如何随 SDE 演化，可以由 Fokker–Planck equation 描述。在 scalar $g(t)$ 情形下，

$$
\boxed{
\partial_t p_t(x)
=
-\nabla\cdot\!\left(
f(x,t)p_t(x)
\right)
+\frac12g(t)^2\Delta p_t(x).
}
$$

$\nabla\cdot$ 是 divergence，$\Delta$ 是 Laplacian。该方程把 sample path 的随机动态转换成 density 的确定性演化。这里采用的标准推导可见 Särkkä 与 Solin 2019 的专著 Applied Stochastic Differential Equations（补充材料暂未公开）；它需要足够的可微性、可积性与边界条件。

若从 $t=T$ 向 $t=0$ 反向运行，仅把 drift 改成 $-f$ 是错误的。Forward diffusion 不断把概率质量摊开，逆过程必须利用当前 density 的 score 把质量重新聚回高密度区域。

Diffusion time reversal 的经典历史来源是 Brian Anderson 1982 年的论文 [*Reverse-Time Diffusion Equation Models*](https://doi.org/10.1016/0304-4149\(82\)90051-5 "官方论文页面")。Song et al. 2021 在生成建模中采用保持原时间标签、从 $T$ 积分到 0 的写法，此时 $dt<0$：

$$
\boxed{
dX_t
=
\left[
f(X_t,t)
-g(t)^2s_t(X_t)
\right]dt
+g(t)\,d\bar W_t,
\qquad
dt<0.
}
$$

$\bar W_t$ 是 reverse-time Wiener process。与 forward drift 相比，reverse drift 多出

$$
-g(t)^2s_t(x),
$$

它正是由当前 noisy marginal 决定的 density correction。若 score exact，并满足 time-reversal theorem 所需条件，从 $p_T$ 启动该 SDE 会恢复 forward process 的反向 path law。

这个公式的严格性不能只靠形式上的 Fokker–Planck matching。Song et al. 2021 给出生成模型所需的 scalar interface，并引用 Anderson 1982；Cattiaux、Conforti、Gentil 与 Léonard 在 2023 年论文 [*Time Reversal of Diffusion Processes under a Finite Entropy Condition*](https://doi.org/10.1214/22-AIHP1320 "官方论文页面") 中，在 finite-entropy 等条件下处理更低正则的 time reversal。对 state-dependent diffusion matrix，一般公式还会出现 matrix-divergence term，不能机械套用上面的 scalar 形式。

实际模型用 $s_\theta(x,t)$ 代替未知的 $s_t(x)$。因此 DSM 的统计误差不再只是一个 denoising 指标，它会直接变成 reverse drift error，并沿整条生成路径累积。

## 5. 2021：同一组 marginals 还可以由确定性 ODE 实现

Score-SDE 论文还有一个容易被误解的重要结果：同一条 marginal density path 不只对应一个随机过程。Song et al. 2021 构造了 probability-flow ODE。

先把 Fokker–Planck equation 写成 conservation law。因为

$$
\nabla p_t
=
p_t\nabla\log p_t
=
p_t s_t,
$$

SDE 的 density evolution 可以写成

$$
\partial_t p_t
=
-\nabla\cdot
\left[
\left(
f-\frac12g^2s_t
\right)p_t
\right].
$$

另一方面，确定性 ODE

$$
\frac{dZ_t}{dt}
=
v(Z_t,t)
$$

的 density 满足 continuity equation

$$
\partial_t p_t
=
-\nabla\cdot(vp_t).
$$

只要选择相同的 probability current，就得到 Song et al. 2021 Eq. (13)：

$$
\boxed{
\frac{dZ_t}{dt}
=
f(Z_t,t)
-\frac12g(t)^2s_t(Z_t).
}
$$

Reverse-time SDE 中 score 前的系数是 $g^2$，而 probability-flow ODE 中是 $\frac12g^2$。这个二分之一不是经验超参数；它来自把 SDE 的 second-order diffusion flux 完全吸收到 first-order velocity 中。

在 exact score 与足够 regularity 下，forward SDE 与 probability-flow ODE 对每个固定时刻具有相同边缘分布：

$$
\mathcal L(X_t)=\mathcal L(Z_t)=p_t.
$$

但它们不是同一个 path law。一般而言，

$$
\mathcal L(X_s,X_t)
\ne
\mathcal L(Z_s,Z_t).
$$

SDE path 保留随机扰动并具有非零 quadratic variation；ODE 在给定初值后是确定性轨迹，并具有 finite variation。下图上方显示两类完全不同的 sample paths，下方却显示它们在多个固定时刻的 histograms 基本重合。

![随机 SDE path 与确定性 probability-flow ODE 的同边缘、异路径](/images/diffusion/d4_sde_pf_paths.png)

因此，Score-SDE 所说的“等价”应准确写成 **marginally equivalent**，不能升级为 transition kernel 或 conditional trajectory 等价。反向积分 reverse SDE 与反向积分 probability-flow ODE 是两种生成器：前者是 stochastic sampler，后者是 deterministic transport。

Probability-flow ODE 还允许使用 continuous change of variables 计算该 ODE model 的 likelihood，这是 2021 论文的重要扩展；但可计算 likelihood 不表示 DSM 本身已经直接优化 exact maximum likelihood，数值 ODE tolerance、score error 与 divergence estimation 仍会带来误差。

现在可以回答本章标题：

> 去噪器学到的是各噪声时刻的 marginal score；score 既给出 posterior-mean denoising direction，也给出 reverse-time SDE 的密度修正项，并能进一步定义同边缘的 probability-flow ODE。

从 2005 年 score matching，到 2011 年 DSM，再到 2019 年 NCSN 和 2021 年 Score-SDE，研究主线始终是在寻找同一个对象的更可训练、更可采样表示。这条演进最终把去噪目标、密度几何与生成动力学连接成了同一套理论接口。

## 本章论文索引

| 时间   | 论文                                                                                             | 本章中的作用                                               |
| ---- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 1982 | Anderson, *Reverse-Time Diffusion Equation Models*, SPA                                        | Score-SDE 文献追溯 reverse-time diffusion equation 的经典来源 |
| 2005 | Hyvärinen, *Estimation of Non-Normalized Statistical Models by Score Matching*, JMLR           | 定义 data-space score 与 score-matching objective       |
| 2011 | Vincent, *A Connection Between Score Matching and Denoising Autoencoders*, Neural Computation  | 证明 denoising target 的条件均值等于 noisy marginal score     |
| 2011 | Efron, *Tweedie’s Formula and Selection Bias*, JASA                                            | 给出 Gaussian posterior-mean 与 score 的统计学表达及历史说明       |
| 2019 | Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution*, NeurIPS  | 提出 multi-noise NCSN 与 annealed Langevin generation   |
| 2020 | Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS                                 | 将 noise prediction 与 noisy marginal score 连接         |
| 2021 | Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR | 统一 NCSN、DDPM、reverse SDE 与 probability-flow ODE      |
| 2023 | Cattiaux et al., *Time Reversal of Diffusion Processes under a Finite Entropy Condition*, AIHP | 给出较低正则条件下 diffusion time reversal 的严格理论边界            |
