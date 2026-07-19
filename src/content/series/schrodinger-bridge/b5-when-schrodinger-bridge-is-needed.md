---
title: 什么时候真正需要 Bridge
description: 从任务真正需要的对象出发，判断 OT、Diffusion、Flow 与 Schrödinger Bridge 哪个更合适。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 5
slug: b5-when-schrodinger-bridge-is-needed
tags:
  - schrodinger-bridge
  - method-selection
  - applications
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 围绕双端约束、可信 reference 和完整随机路径的需求，说明 Bridge 的适用边界与代价。
---
只要看到两个分布，很多文章就会把问题写成“从 A 生成 B”。但这句话没有说明我们要终点样本、端点配对，还是一条有时间相关性的随机轨迹，也没有说明参考动力学究竟是科学模型还是随手选的噪声。

Schrödinger Bridge 的代价，正来自它承诺得更多。它同时约束两个端点，并在给定参考路径律的前提下选择整条随机演化。判断它是否值得，关键不在任务属于图像、生物还是物理，而在我们究竟掌握了哪些信息、需要怎样的结果。

## 1. OT、Diffusion、Flow 和 Bridge 各自在选什么

Kantorovich 在 1942 年论文 *On the Translocation of Masses* 中把 Monge 的搬运问题写成端点 coupling 的优化。给定 $\mu_0,\mu_T$ 和代价 $c(x,y)$，它求的是

$$
\min_{\gamma\in\Pi(\mu_0,\mu_T)}
\int c(x,y)\,\gamma(dx,dy),
$$

其中 $\gamma$ 是起点与终点的联合分布，$\Pi(\mu_0,\mu_T)$ 表示具有指定两侧边缘的所有 coupling。这个问题回答“谁和谁配对、搬运代价多大”，并不规定配对之后的随机波动。Benamou 与 Brenier 在 2000 年论文 [*A Computational Fluid Mechanics Solution to the Monge–Kantorovich Mass Transfer Problem*](https://doi.org/10.1007/s002110050002 "官方论文页面") 中给出 deterministic dynamic formulation，但其中的 velocity 仍是确定性输运场，不是 diffusion 的随机漂移。

Diffusion 走的是另一条路。Ho、Jain 与 Abbeel 在 2020 年论文 [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239 "官方论文页面") 中先固定一条 Gaussian noising chain，再学习逆向去噪；Song 等人在 2021 年 ICLR 论文 Score-Based Generative Modeling through Stochastic Differential Equations（补充材料暂未公开） 中把它写成连续时间 SDE。这里的 forward noise law 是建模者预先选定的，目标通常是从一个简单终点分布生成数据，而不是同时满足两个任意的 hard endpoint marginals。

Flow Matching 也先选路径。Lipman、Chen、Ben-Hamu、Nickel 与 Le 在 2023 年论文 [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747 "官方论文页面") 中先给出 conditional interpolation，再回归生成这条 marginal path 所需的 velocity。训练可以很稳定，但 coupling 和路径形状已经预先选定，不是通过 reference-relative KL 从所有可能路径中筛出来的。

Bridge 把这三件事放在同一个变分问题里：给定两个端点分布和参考路径律 $R$，在满足 $P_0=\mu_0$、$P_T=\mu_T$ 的路径律中最小化 $\operatorname{KL}(P\Vert R)$。所以它选择的不只是 coupling，也包括端点之间的随机路径。

## 2. 只有快照时，不能凭空得到轨迹

假设实验只告诉我们每个时刻的群体分布 $p_t$，并没有告诉我们同一个个体在相邻时刻的位置。给定一个 velocity field $v_t(x)$，这些边缘可以满足 continuity equation

$$
\partial_t p_t(x)
+
\nabla\!\cdot\bigl(p_t(x)v_t(x)\bigr)
=0.
$$

这条连续性方程只约束每个时刻的 density 如何变化，不足以确定样本路径之间的时间相关性。Gyöngy 在 1986 年论文 [*Mimicking the One-Dimensional Marginal Distributions of Processes Having an Itô Differential*](https://doi.org/10.1007/BF00699039 "官方论文页面") 中说明，不同的随机过程可以拥有相同的 one-time marginals，却有不同的 transition kernels 和 path laws。

这不是抽象的数学反例。Weinreb 等人在 2018 年论文 [*Fundamental Limits on Dynamic Inference from Single-Cell Snapshots*](https://doi.org/10.1073/pnas.1714723115 "官方论文页面") 中指出，single-cell snapshots 本身通常不足以识别真实 lineage。Schiebinger 等人的 2019 年 [*Waddington-OT*](https://doi.org/10.1016/j.cell.2019.01.006 "官方论文页面") 选择一个 OT coupling 来连接不同时间的 cell states；Forrow 等人在 2021 年 [*LineageOT*](https://doi.org/10.1038/s41467-021-25133-1 "官方论文页面") 中加入 lineage information 后，得到的 coupling 已不再只是由两张 marginal snapshot 决定。

因此，如果下游任务只需要每个时刻的分布或最后的样本，Flow、Diffusion 或一个简单的 Markovian projection 往往够用。若任务依赖个体之间的对应关系、转移概率或轨迹上的稀有事件，才需要额外的 coupling 和 path-law 假设；Bridge 能提供这种结构，但它不能从没有记录的 lineage 中恢复事实。

![相同 one-time marginals 不代表相同的随机路径](/images/bridge/B12_same_marginals_paths.png)

## 3. 哪些任务确实需要参考路径律

第一类是端点配对有实际含义的转换任务。例如图像复原中，退化图像与清晰图像是成对观测。Liu 等人在 2023 年论文 [*I$^2$SB: Image-to-Image Schrödinger Bridge*](https://arxiv.org/abs/2302.05872 "官方论文页面") 中利用解析的 Gaussian bridge 建立 paired conditional diffusion。这里的 Bridge 价值不在于“两个图像分布之间有一条曲线”，而在于退化机制和目标图像共同限定了条件路径；如果没有成对信息，模型只能依赖额外先验来猜 coupling。

第二类是参考动力学本身就是科学模型。若 $R$ 描述无控制粒子的扩散、分子构象变化或一个可信的 corruption process，那么改变漂移的 KL 可以解释成控制这套系统所付出的信息代价。Chen、Georgiou 与 Pavon 在 2021 年综述 [*Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge*](https://doi.org/10.1137/20M1339982 "官方论文页面") 中系统讨论了这种 stochastic control 解释。此时 reference 不是训练技巧，而是问题的一部分；换一个 $R$，回答的就是另一个问题。

第三类是条件稀有事件。Hartmann 与 Schütte 在 2012 年论文 [*Efficient Rare Event Simulation by Optimal Nonequilibrium Forcing*](https://doi.org/10.1088/1742-5468/2012/11/P11004 "官方论文页面") 中研究有限时间内如何用最小 forcing 提高稀有事件的出现率；Chetrite 与 Touchette 在 2015 年论文 [*Nonequilibrium Markov Processes Conditioned on Large Deviations*](https://doi.org/10.1007/s00023-014-0375-8 "官方论文页面") 中研究长期大偏差条件下的 driven process。这些工作与 Bridge 都有“相对于原动力学做最小改动”的味道，但长期经验量约束、有限时间端点约束和给定 endpoint pair 的 Brownian bridge 不能混为一谈。

## 4. Bridge 最容易被误用的地方

最常见的问题是 reference 选得没有依据。Bridge 可以在数学上精确地投影到 $R$ 上，但如果 $R$ 的 support、噪声方向或漂移与真实系统无关，得到的只是“相对于错误模型最合理”的过程。这个缺陷不会被更大的网络或更低的训练损失修复。

其次是把两个边缘当成了完整监督。它们通常允许许多 endpoint couplings；Bridge 通过 $R_{0T}$ 选择其中一个，这个选择需要 paired data、lineage、干预实验或其他机制证据来检验。只有两张分布图时，coupling 本身往往不可识别。

再次是把终点指标当成路径指标。高维近似还会引入 coupling、条件桥采样、回归、时间离散和 outer iteration 的误差。终点分布接近，不等于中间分布接近；中间快照接近，也不等于 transition law 或 path KL 接近。

最后是可行性和成本。若目标端点要求 reference support 不允许的迁移，$\operatorname{KL}(P\Vert R)=+\infty$，任何神经网络输出的“连接轨迹”都不是原问题的 finite-entropy Bridge。即使问题可行，反复的 forward/backward simulation、coupling estimation 和多轮训练，也可能远贵于一次 Diffusion 或 Flow 训练。

## 5. 先看任务究竟需要什么

先看两个端点是不是都来自真实约束。只有一个 data distribution，另一个只是方便采样的 base prior，Diffusion 或 Flow 通常更直接；两端都来自观测，Bridge 才有明确的双端问题。

再看 reference dynamics 是否有独立含义，以及下游是否需要随机路径。只要 endpoint pairing，可以先看 OT；只要终点样本或 marginal curve，可以先看 Diffusion、Flow；需要双端约束下、相对于 $R$ 的 stochastic trajectories，才应考虑 Schrödinger Bridge。

最后还要能验证新增结构。Endpoint coupling、intermediate marginals、transition statistics 和 reference sensitivity 若都无法检查，复杂的 Bridge 系统也就很难被证伪。

Schrödinger Bridge 不是所有分布变换的总称。它真正解决的是一个更窄、也更有用的问题：

$$
\boxed{
\text{在保留一套有意义的 reference dynamics 的前提下，}
\text{怎样以最小信息改动满足两个端点观测？}
}
$$

没有双端约束、可信 reference 和对完整随机演化的需求，使用 Bridge 往往只是增加复杂度。

## 本章论文索引

| 时间   | 论文                                                                                       | 本章采用的内容                                          |
| ---- | ---------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 1942 | Kantorovich, *On the Translocation of Masses*                                            | 将 transport 写成具有双边缘约束的 coupling 优化               |
| 1986 | Gyöngy, *Mimicking the One-Dimensional Marginal Distributions...*                        | 说明 one-time marginals 不唯一确定 path law             |
| 2000 | Benamou & Brenier, *A Computational Fluid Mechanics Solution...*                         | 给出 deterministic OT 的动态表述                        |
| 2012 | Hartmann & Schütte, *Efficient Rare Event Simulation by Optimal Nonequilibrium Forcing*  | 说明有限时间稀有事件 forcing 与 KL/control 的关系              |
| 2015 | Chetrite & Touchette, *Nonequilibrium Markov Processes Conditioned on Large Deviations*  | 区分长期大偏差条件过程与有限时间 Bridge                          |
| 2018 | Weinreb et al., *Fundamental Limits on Dynamic Inference from Single-Cell Snapshots*     | 说明 snapshots 对真实 lineage 的非识别性                   |
| 2019 | Schiebinger et al., *Waddington-OT*                                                      | 用 OT coupling 推断跨时刻 cell-state transitions       |
| 2020 | Ho et al., *Denoising Diffusion Probabilistic Models*                                    | 提供固定 forward noising law 的 Diffusion 对照          |
| 2021 | Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* | 统一 score diffusion 的连续时间表述                       |
| 2021 | Forrow et al., *LineageOT*                                                               | 说明 lineage information 如何改变 coupling             |
| 2021 | Chen, Georgiou & Pavon, *Stochastic Control Liaisons...*                                 | 说明 meaningful reference 下的 stochastic control 解释 |
| 2023 | Lipman et al., *Flow Matching for Generative Modeling*                                   | 提供 chosen path 与 learned velocity 的 Flow 对照      |
| 2023 | Liu et al., *I$^2$SB*                                                                    | 展示 paired image-to-image conditional Bridge      |
