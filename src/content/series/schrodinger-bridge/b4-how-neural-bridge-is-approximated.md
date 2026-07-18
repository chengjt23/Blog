---
title: 高维 Bridge 怎样被神经网络近似
description: 以精确算子为基线，比较 DSB、Bridge Matching 与 simulation-free 方法近似了什么。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 4
slug: b4-how-neural-bridge-is-approximated
tags:
  - schrodinger-bridge
  - deep-learning
  - bridge-matching
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 区分 IPF half-step、Markovian projection 与 conditional regression，并拆分五类近似误差。
---
第三章的 exact IPF 可以写成

$$
P^{(2n+1)}
=
\arg\min_{P_T=\mu_T}
\operatorname{KL}
\left(
P\Vert P^{(2n)}
\right),
$$

$$
P^{(2n+2)}
=
\arg\min_{P_0=\mu_0}
\operatorname{KL}
\left(
P\Vert P^{(2n+1)}
\right).
$$

在有限状态空间中，每个 path law 可以由矩阵或有限 transition kernels 表示，half-step 也能通过精确 scaling 完成。面对高维图像、分子构象和连续场数据，我们无法显式存储对应的 endpoint matrix，更无法直接计算所有 conditional bridges。

现代 neural Schrödinger Bridge 方法并没有改变目标定义。它们保留 exact operator 的外形，再用轨迹采样、conditional expectation 和神经网络近似其中不可计算的部分。

因此阅读任何方法时，都应先问：

$$
\boxed{
\text{它近似的是 IPF half-step、Markovian projection，还是 conditional bridge regression？}
}
$$

方法名称并不能回答这个问题，只有它的 population target 与 exact operator 才能回答。

## 1. 2021：为什么 exact path-space IPF 在高维中难以直接计算

一次 exact terminal half-step 的形式是

$$
\frac{dP^{\mathrm{new}}}{dQ}
=
\frac{d\mu_T}{dQ_T}(X_T).
$$

它把 terminal marginal 换成 $\mu_T$，同时保留给定 $X_T$ 后的 conditional path law。公式简洁，实现却需要解决三个问题。首先，density ratio

$$
\frac{d\mu_T}{dQ_T}
$$

通常未知，样本不能直接给出高维 density ratio。其次，一般 nonlinear diffusion 的 transition density 与 reverse drift 没有解析式。最后，outer IPF、physical-time discretization 与 neural optimization 是三个循环，训练 loss 降低不等于 Bridge iteration 收敛。

因此高维方法通常用 network 表示 time-dependent drift、score、conditional mean 或 velocity。一次 exact projection 被替换为：

$$
\text{sample paths}
\longrightarrow
\text{construct regression targets}
\longrightarrow
\text{fit a neural field}
\longrightarrow
\text{simulate new paths}.
$$

只有在无限数据、足够丰富的函数类与全局优化所构成的总体极限中，regression target 才可能恢复 exact operator。有限网络算法还要额外承担 approximation 与 simulation error。

## 2. 2021：DSB 怎样近似 path-space IPF

De Bortoli、Thornton、Heng 与 Doucet 在 NeurIPS 2021 论文 [*Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling*](https://arxiv.org/abs/2106.01357 "官方论文页面") 中提出 Diffusion Schrödinger Bridge，简称 DSB。它的核心选择是：保留 path-space IPF 的前后向交替结构，用 neural regression 近似每个 half-step 所需的 reverse 或 forward dynamics。

先看一个离散小步 Markov chain。设 forward transition 为

$$
X_{k+1}
=
F_k(X_k)
+
\sqrt{2\gamma_{k+1}}\,Z_{k+1},
\qquad
Z_{k+1}\sim\mathcal N(0,I),
$$

其中 $F_k$ 是当前 forward mean map，$\gamma_{k+1}>0$ 是 step variance。若希望用 Gaussian reverse transition 预测 $X_k$，平方损失的 population optimum 是

$$
\boxed{
B_{k+1}^*(x)
=
\mathbb E
\left[
X_k
\mid
X_{k+1}=x
\right].
}
$$

因此可以从当前 forward process 采样样本对 $(X_k,X_{k+1})$，再用 network $B_{\theta,k+1}(x)$ 回归前一步状态。得到 reverse sampler 后，从 terminal distribution $\mu_T$ 启动并生成 backward paths；随后交换方向，拟合新的 forward maps。

整个外层结构是

$$
\text{simulate forward}
\rightarrow
\text{fit reverse}
\rightarrow
\text{simulate backward}
\rightarrow
\text{fit forward}
\rightarrow\cdots
$$

它对应 exact IPF 中交替满足 terminal 与 initial constraints 的 cycles，但每个 half-step 已不再是矩阵意义下的 exact I-projection。

![Neural half-step 从随机 transition target 学习 conditional mean](/images/bridge/B8_population_regression.png)

DSB 的 exact IPF iterates 与实际 neural algorithm 必须分开：后者还加入 time discretization、finite samples 与 optimization error，因此不会自动继承 exact-IPF 的全部收敛保证。

Vargas et al. 在 2021 年论文 [*Solving Schrödinger Bridges via Maximum Likelihood*](https://doi.org/10.3390/e23091134 "官方论文页面") 中用 reverse-drift MLE 近似 half-bridge；2023 年 [correction](https://doi.org/10.3390/e25020289 "官方论文页面") 澄清，保留 conditional path law 只消去 conditional KL，不能据此推出有限样本 approximate-IPF 收敛。

## 3. 1986—2023：Bridge Matching 怎样近似 Markovian projection

另一类方法不直接模仿 IPF scaling，而从 reference conditional bridges 出发。

设 $\gamma$ 是某个 endpoint coupling，并按它混合 reference bridges：

$$
\Pi
=
\int
R^{xy}
\,\gamma(dx,dy).
$$

$\Pi$ 具有正确的 endpoint coupling，也共享 reference conditional bridges，因此是 reciprocal path law；但任意 $\gamma$ 产生的 mixture 未必 Markov。若希望用只依赖当前状态 $X_t$ 的 neural drift 表示它，就需要把 endpoint-dependent dynamics 投影成 Markov field。

假设给定 endpoints $(X_0,X_T)$ 后，conditional bridge 在时刻 $t$ 的 drift 为

$$
b^{X_0,X_T}(t,X_t).
$$

Markovian projection 的 population drift 定义为

$$
\boxed{
b_M(t,x)
=
\mathbb E_\Pi
\left[
b^{X_0,X_T}(t,X_t)
\mid
X_t=x
\right].
}
$$

对 conditional bridge drift 做平方回归时，population optimum 正是 $b_M$。因此 Bridge Matching 可以通过采样 endpoint pairs、conditional bridges 和中间状态，学习一个只依赖 $(t,x)$ 的 Markov drift。

Gyöngy 在 1986 年论文 [*Mimicking the One-Dimensional Marginal Distributions of Processes Having an Itô Differential*](https://doi.org/10.1007/BF00699039 "官方论文页面") 中建立了经典 mimicking theorem 路线；Brunick 与 Shreve 在 2013 年论文 [*Mimicking an Itô Process by a Solution of a Stochastic Differential Equation*](https://doi.org/10.1214/12-AAP881 "官方论文页面") 中将 projection 扩展到更一般的 path functionals。其关键边界是：逐时刻满足 $\mathcal L_\Pi(X_t)=\mathcal L_M(X_t)$，并不推出两时刻联合分布也相同。

Markovian projection 可以保留 one-time marginals，却通常改变 transition kernels 和完整 path law。

Shi、De Bortoli、Campbell 与 Doucet 在 NeurIPS 2023 论文 [*Diffusion Schrödinger Bridge Matching*](https://doi.org/10.52202/075280-2717 "官方论文页面") 中把 Bridge Matching 与 Iterative Markovian Fitting 结合。其理想化 outer loop 在两类对象间交替：

$$
\text{reciprocal projection}
\longleftrightarrow
\text{Markovian projection}.
$$

Reciprocal projection 恢复相对于 reference 的 conditional bridges，Markovian projection 用当前状态函数逼近 reciprocal law 的 marginals。若两类 projections 精确、相关 densities 正则且 iterations 收敛，fixed point 才与目标 Schrödinger Bridge 对接。

![Bridge Matching 与 Markovian projection](/images/bridge/B9_markovian_projection.png)

这条路线把 endpoint-conditioned dynamics 转成 supervised regression，但 matching one-time marginals、恢复 Markovity 与恢复 exact path law 仍是三种目标。

## 4. 2024：simulation-free 方法省掉了什么，又保留了什么

DSB 与 iterative Bridge Matching 都可能在每轮训练前生成大量 SDE trajectories。Simulation-free 方法试图直接采样 conditional path points 与解析 targets，从而避免在训练 inner loop 中反复求解当前 neural SDE。

Tong、Malkin、Huguet、Zhang、Rector-Brooks、Fatras、Wolf 与 Bengio 在 2024 年论文 [*Simulation-Free Schrödinger Bridges via Score and Flow Matching*](https://arxiv.org/abs/2307.03672 "官方论文页面") 中提出 SF$^2$M。它先获得或近似一个 entropic endpoint coupling

$$
(X_0,X_T)\sim\gamma,
$$

再从给定 endpoints 的 tractable conditional bridge 中采样中间状态 $X_t$。若 conditional target field 为 $Y_t$，平方回归的 population optimum 为

$$
\boxed{
v^*(t,x)
=
\mathbb E
\left[
Y_t
\mid
X_t=x
\right].
}
$$

这与 DSB 和 Bridge Matching 中的 conditional-expectation projection 具有同一统计骨架，区别在于 training samples 的构造方式。SF$^2$M 通过 conditional bridges 直接产生 supervised pairs，减少 current-model trajectory simulation。

![Simulation-free conditional target 与 population field](/images/bridge/B10_sf2m_population_identity.png)

“Simulation-free”需要按范围理解。它通常表示训练 target 不要求反复模拟当前 neural dynamics，并不表示：

- 不需要 endpoint samples；
- 不需要近似或求解 endpoint coupling；
- inference 时不需要 ODE/SDE solver；
- conditional bridge 可以对任意 reference 解析采样；
- finite minibatch regression 自动等于 exact population field。

Liu et al. 在 2024 年论文 [*Generalized Schrödinger Bridge Matching*](https://arxiv.org/abs/2310.02233 "官方论文页面") 中把 matching 扩展到更一般的 stochastic processes 与 constraints；reference family、coupling source 与 projection target 仍须逐项核验。

## 5. 2021—2025：怎样判断一个 neural Bridge 近似了什么

高维方法可以统一放进同一张 operator map：

| 路线                       | 试图近似的 exact object                    | 主要训练接口                                        | 主要新增误差                                               |
| ------------------------ | ------------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| Neural IPF / DSB         | endpoint half-bridge I-projection     | forward/reverse mean、drift 或 score regression | outer-iteration、regression、simulation、discretization |
| Bridge Matching / IMF    | reciprocal law 的 Markovian projection | conditional bridge drift regression           | projection、function approximation、path-law mismatch  |
| Simulation-free matching | conditional target 的 population field | analytically sampled bridge points 与 targets  | coupling、conditional-model、Monte Carlo、regression    |

三类路线可以组合使用，因此评价时应按 operator responsibility 拆开，并区分 projection、regression、sampling、discretization 与 outer-iteration error。较低 training MSE 只约束 regression layer；endpoint samples 或 one-time marginals 正确，不能单独证明 full path law 正确。

现在可以回答本章标题：

> 高维 neural Bridge 不是重新定义 Schrödinger Bridge，而是用函数回归和采样近似 exact IPF、conditional bridge 或 Markovian projection 中的不可计算算子。方法是否仍接近目标 $P^*$，取决于这些近似在总体、有限样本与数值计算三个层面分别有多准确。

## 本章论文索引

| 时间   | 论文                                                                                               | 本章中的作用                                                            |
| ---- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| 1986 | Gyöngy, *Mimicking the One-Dimensional Marginal Distributions...*                                | 建立以 Markov diffusion 匹配 one-time marginals 的经典 projection 路线      |
| 2013 | Brunick & Shreve, *Mimicking an Itô Process by a Solution of a Stochastic Differential Equation* | 扩展 mimicking/Markovian projection 结构                              |
| 2021 | De Bortoli et al., *Diffusion Schrödinger Bridge*                                                | 用 neural forward/reverse regression 近似 path-space IPF             |
| 2021 | Vargas et al., *Solving Schrödinger Bridges via Maximum Likelihood*                              | 将 half-bridge 的 reverse dynamics 写成 maximum-likelihood estimation |
| 2023 | Vargas et al., *Correction: Solving Schrödinger Bridges via Maximum Likelihood*                  | 澄清 conditional KL 与 marginal-ratio KL 的责任边界                       |
| 2023 | Shi et al., *Diffusion Schrödinger Bridge Matching*                                              | 结合 Bridge Matching 与 Iterative Markovian Fitting                  |
| 2024 | Tong et al., *Simulation-Free Schrödinger Bridges via Score and Flow Matching*                   | 用 conditional bridge samples 进行 score/flow population regression  |
| 2024 | Liu et al., *Generalized Schrödinger Bridge Matching*                                            | 将 matching 扩展到更一般的 reference processes 与 constraints              |
