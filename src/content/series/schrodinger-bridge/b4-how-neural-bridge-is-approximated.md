---
title: 高维 Bridge 怎样被神经网络近似
description: 说明高维方法为什么需要回归与采样，并比较 DSB、Bridge Matching 与 simulation-free 路线。
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
scope: >-
  区分 path-space IPF 半步、Markovian projection 与 conditional bridge regression
  的近似边界。
---
第三章里的 IPF 在有限状态空间中并不难：把一个矩阵的行和、列和交替缩放，就能逐步逼近两个目标边缘。高维连续数据没有这样一张可以完整存下来的矩阵。更麻烦的是，Bridge 的每次边缘修正都要求我们保留“给定一个端点后，整条参考路径怎样走”的条件分布。

高维方法没有重新定义 Schrödinger Bridge，而是把无法直接计算的条件路径投影变成样本回归。读论文时，先找清楚它近似了哪一步，通常比记住方法名称更重要。

## 1. 精确的半步投影为什么算不动

设 $Q$ 是当前的一条候选路径律，$Q_T$ 是它在终点时刻 $T$ 的边缘分布。只把终点边缘改成观测到的 $\mu_T$，同时尽量不动其余路径结构，第二章的 KL 投影可以写成

$$
\frac{dP^{\mathrm{new}}}{dQ}(\omega)
=
\frac{d\mu_T}{dQ_T}(X_T(\omega)).
$$

这里的 $X_T(\omega)$ 是路径 $\omega$ 的终点。这个公式看起来只需要一个 density ratio，实际上要做两件事：先从 $\mu_T$ 取一个终点，再从 $Q$ 的条件路径律 $Q(d\omega\mid X_T)$ 中补出整条路径。高维非线性 diffusion 通常既没有可用的 $Q_T$ 密度，也没有可以直接采样的条件桥。

这正是 2021 年 DSB 工作面对的计算障碍。De Bortoli、Thornton、Heng 与 Doucet 在 NeurIPS 2021 论文 [*Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling*](https://arxiv.org/abs/2106.01357 "官方论文页面") 中没有改动这个投影的目标，而是把其中的条件路径信息改写成可从样本估计的时间依赖函数。这样做之后，精确半步变成了“采样—回归—再采样”的近似循环。

从这里开始，函数近似、有限样本和 SDE 离散都会带来误差；训练损失下降不能单独说明 outer IPF 已经收敛。

## 2. DSB：把每一轮半步变成回归

先看最小的离散例子。令 $X_k\in\mathbb R^d$ 表示第 $k$ 个时间格点的状态，$m_k$ 表示当前 forward chain 的均值映射，$\gamma_{k+1}>0$ 表示这一步的噪声尺度，$\xi_{k+1}\sim\mathcal N(0,I_d)$ 是标准高斯噪声，$I_d$ 是 $d$ 维单位矩阵。一次 forward transition 可以写成

$$
X_{k+1}=m_k(X_k)+\sqrt{2\gamma_{k+1}}\,\xi_{k+1}.
$$

如果反向 transition 的协方差固定，最合适的反向均值就是给定 $X_{k+1}=x$ 时上一步状态的条件均值：

$$
r_k^*(x)
=
\mathbb E\left[X_k\mid X_{k+1}=x\right].
$$

原因很简单。对任意候选函数 $r$，平方损失可以按条件期望分解为

$$
\mathbb E\|r(X_{k+1})-X_k\|^2
=
\mathbb E\|r(X_{k+1})-r_k^*(X_{k+1})\|^2
+
\mathbb E\|X_k-r_k^*(X_{k+1})\|^2.
$$

第二项与 $r$ 无关，所以 population optimum 必然是上面的条件均值。DSB 用神经网络 $r_{\theta,k}$ 拟合它，得到一条近似的 reverse chain；再用反向生成的样本拟合 forward chain，如此交替进行。

这套安排保留了 IPF 的方向：先满足一侧端点，再用另一侧的条件动力学修正回来。它没有声称一次回归就得到最终 Bridge。每次 outer iteration 使用的样本分布都会变化，网络拟合的是当前 half-step 的条件对象，而不是一个固定的数据集上的普通 denoiser。

Vargas、Padhy、Nüsken 与 Öktem 在 2021 年论文 [*Solving Schrödinger Bridges via Maximum Likelihood*](https://doi.org/10.3390/e23091134 "官方论文页面") 中从 reverse-drift 的 maximum likelihood 角度写出了类似的半步估计；2023 年的 [correction](https://doi.org/10.3390/e25020289 "官方论文页面") 特别指出，保留 conditional path law 只能消掉 conditional KL，不能把有限样本的训练过程直接当作精确 IPF 的收敛证明。

![DSB 用条件均值回归近似一轮 neural half-step](/images/bridge/B8_population_regression.png)

DSB 不是把高维路径分布显式存下来再做 Sinkhorn，而是用 forward、reverse 两组可模拟的动力学间接实现交替投影。

## 3. Bridge Matching：直接拟合条件桥的平均漂移

还有一条不同的路线：不把每轮 IPF scaling 写成 density ratio，而是直接从端点条件桥生成训练样本。

设 $\gamma$ 是某个起点与终点的耦合，把参考过程在端点 $(x,y)$ 下的条件桥记作 $R^{xy}$。从每一对 $(X_0,X_T)\sim\gamma$ 出发，按 $R^{xy}$ 采样整条路径，得到混合路径律

$$
\Pi_\gamma(d\omega)
=
\int R^{xy}(d\omega)\,\gamma(dx,dy).
$$

这个过程在给定两个端点后仍然使用参考桥，因此保留了 reference 的局部随机性；但把所有端点对混在一起后，它一般只是 reciprocal process，不一定能由一个只看当前 $X_t$ 的 Markov drift 表示。

假设固定端点后的桥在时刻 $t$ 的漂移为 $b^{x,y}(t,X_t)$。若我们要找一个只依赖当前时间和状态的 Markov drift，平方回归的总体最优解是

$$
b_M(t,x)
=
\mathbb E_{\Pi_\gamma}
\left[b^{X_0,X_T}(t,X_t)\mid X_t=x\right].
$$

Gyöngy 在 1986 年论文 [*Mimicking the One-Dimensional Marginal Distributions of Processes Having an Itô Differential*](https://doi.org/10.1007/BF00699039 "官方论文页面") 中证明了这类 Markovian projection 可以匹配给定时刻的边缘分布；Brunick 与 Shreve 在 2013 年论文 [*Mimicking an Itô Process by a Solution of a Stochastic Differential Equation*](https://doi.org/10.1214/12-AAP881 "官方论文页面") 中把结果推广到更一般的过程函数。

这个条件期望说得很具体：网络学到的是条件漂移的平均，而不是端点信息本身。它可以保证 one-time marginals 对得上，却不会自动保留任意两时刻的联合分布，更不能仅凭一个低训练误差声称完整 path law 已经相同。

Shi、De Bortoli、Campbell 与 Doucet 在 2023 年 NeurIPS 论文 [*Diffusion Schrödinger Bridge Matching*](https://doi.org/10.52202/075280-2717 "官方论文页面") 中把这种 Bridge Matching 与 Iterative Markovian Fitting 结合起来：一边用条件桥构造 reciprocal law，一边把它投影成可由神经 SDE 表示的 Markov law。两种投影都精确且迭代收敛时，才有理由把 fixed point 与 Schrödinger Bridge 联系起来。

![Bridge Matching 把端点条件漂移投影成当前状态的函数](/images/bridge/B9_markovian_projection.png)

## 4. Simulation-free 方法省掉了哪一步

DSB 和 iterative Bridge Matching 常常需要在每轮训练前，用当前模型生成许多 SDE 轨迹。2024 年，Tong、Malkin、Huguet、Zhang、Rector-Brooks、Fatras、Wolf 与 Bengio 在 [*Simulation-Free Schrödinger Bridges via Score and Flow Matching*](https://arxiv.org/abs/2307.03672 "官方论文页面") 中提出 SF$^2$M，目标是把这段 inner-loop simulation 换成可以直接采样的 conditional bridge 数据。

设先得到一个端点耦合

$$
(X_0,X_T)\sim\gamma,
$$

再从给定端点的 tractable bridge 采样 $X_t$，并为每个样本构造一个目标场 $Y_t$。网络 $v_\theta(t,x)$ 使用平方损失

$$
\min_\theta
\mathbb E\left[\|v_\theta(t,X_t)-Y_t\|^2\right].
$$

在总体极限中，最优函数仍是条件期望

$$
v^*(t,x)
=
\mathbb E[Y_t\mid X_t=x].
$$

它与 DSB、Markovian projection 共享同一个统计事实：平方回归学到的是给定当前状态后的平均目标。区别在于，SF$^2$M 不必反复模拟当前 neural dynamics 来制造训练 target。

“simulation-free”只描述训练环节省掉了什么，并不表示不需要端点样本，不表示端点耦合自动已知，也不表示推理时可以不用 ODE/SDE solver。Liu 等人在 2024 年论文 [*Generalized Schrödinger Bridge Matching*](https://arxiv.org/abs/2310.02233 "官方论文页面") 中把 matching 扩展到更一般的随机过程与约束；reference 的选择、耦合的来源和条件桥的可采样性，仍然是方法成立的前提。

![Simulation-free 方法中的条件桥样本与回归目标](/images/bridge/B10_sf2m_population_identity.png)

## 5. 神经网络究竟近似到了哪里

DSB 近似 path-space IPF 的半步；Bridge Matching 近似 reciprocal law 的 Markovian projection；simulation-free 方法则把当前模型轨迹换成条件桥样本。三者都把精确 KL 投影拆成了可采样、可回归的步骤，也因此引入了耦合、函数拟合、时间离散和迭代误差。

Endpoint discrepancy 小只能说明终点边缘接近，中间 marginal 接近也只说明快照层面接近。要判断学到的是不是 Schrödinger Bridge，还要检查 transition statistics、路径相关性和相对于 reference 的目标。

## 本章论文索引

| 时间   | 论文                                                                                               | 本章采用的内容                                             |
| ---- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| 1986 | Gyöngy, *Mimicking the One-Dimensional Marginal Distributions...*                                | 说明 Markovian projection 可匹配 one-time marginals      |
| 2013 | Brunick & Shreve, *Mimicking an Itô Process by a Solution of a Stochastic Differential Equation* | 将 mimicking 推广到更一般的过程函数                             |
| 2021 | De Bortoli et al., *Diffusion Schrödinger Bridge...*                                             | 用 neural forward/reverse dynamics 近似 path-space IPF |
| 2021 | Vargas et al., *Solving Schrödinger Bridges via Maximum Likelihood*                              | 从 maximum likelihood 角度估计 half-bridge dynamics      |
| 2023 | Vargas et al., *Correction: Solving Schrödinger Bridges via Maximum Likelihood*                  | 澄清 conditional KL 与有限样本收敛的边界                        |
| 2023 | Shi et al., *Diffusion Schrödinger Bridge Matching*                                              | 结合 Bridge Matching 与 Iterative Markovian Fitting    |
| 2024 | Tong et al., *Simulation-Free Schrödinger Bridges via Score and Flow Matching*                   | 用 conditional bridge samples 减少训练期轨迹模拟              |
| 2024 | Liu et al., *Generalized Schrödinger Bridge Matching*                                            | 扩展 matching 到更一般的 reference 与约束                     |
