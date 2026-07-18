---
title: 什么时候真正需要 Bridge
description: 比较 OT、Diffusion、Flow 与 Schrödinger Bridge，给出面向任务的方法选择标准。
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
scope: 从输入、输出、约束和目标函数判断何时需要 reference-aware stochastic path law，并说明失败边界。
---
前四章已经定义了 Schrödinger Bridge 的完整责任链：

$$
(\mu_0,\mu_T,R)
\longrightarrow
P^*
\longrightarrow
\text{Schrödinger factors}
\longrightarrow
\text{exact or learned solver}.
$$

但一个方法具有完整理论，并不表示所有“从分布 A 到分布 B”的任务都应该使用它。很多任务只需要生成终点样本，或者只需要一条方便训练的 marginal path；此时引入 reference path law、双端 constraints 和 repeated projections 可能增加成本，却没有增加任务真正需要的信息。

选择方法前应先问：

> 我们究竟需要一个 endpoint coupling、一条 marginal curve、一个 deterministic velocity，还是一个相对于 reference 最优的 stochastic path law？

这四种输出不同，不能因为都画成“从左到右的箭头”就视为同一个问题。

## 1. 1942—2024：四类方法究竟固定了什么

先把 classical OT、score Diffusion、Flow Matching 与 Schrödinger Bridge 放在同一张表中：

| 方法                 | 基本输入                                             | 被选择或学习的对象                         | 核心约束                                 | 典型输出                               |
| ------------------ | ------------------------------------------------ | --------------------------------- | ------------------------------------ | ---------------------------------- |
| Classical OT       | $\mu_0,\mu_T$ 与 cost $c$                         | endpoint coupling 或 transport map | 两个 endpoint marginals                | 最低 cost pairing                    |
| Score Diffusion    | data law 与 chosen forward noising dynamics       | noisy marginal scores             | forward process 预先固定                 | reverse SDE / probability-flow ODE |
| Flow Matching      | chosen coupling/path 与 conditional interpolation | ODE velocity                      | 生成预先选择的 marginal path                | deterministic flow                 |
| Schrödinger Bridge | $\mu_0,\mu_T,R$                                  | endpoint coupling 与完整 path law    | 双端 marginals + reference-relative KL | stochastic bridge process          |

Kantorovich 在 1942 年论文 *On the Translocation of Masses* 中把 transport 推广为满足双边缘约束的 coupling。其现代形式解决

$$
\min_{\gamma\in\Pi(\mu_0,\mu_T)}
\int c(x,y)\,\gamma(dx,dy),
$$

它选择 endpoint pairing，却不自动给出 stochastic fluctuations。Benamou 与 Brenier 在 2000 年论文 [*A Computational Fluid Mechanics Solution to the Monge–Kantorovich Mass Transfer Problem*](https://doi.org/10.1007/s002110050002 "官方论文页面") 中给出 deterministic dynamic formulation；其中的 velocity 与 controlled diffusion drift 不是同一个对象。

Ho、Jain 与 Abbeel 在 2020 年论文 [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239 "官方论文页面") 中使用固定 Gaussian noising chain。Song et al. 在 ICLR 2021 论文 Score-Based Generative Modeling through Stochastic Differential Equations（补充材料暂未公开） 中将其统一为连续时间 SDE：先指定 forward dynamics，再学习 noisy marginal scores。它不以两个任意 hard endpoint marginals 下的

$$
\min
\left\{
\operatorname{KL}(P\Vert R):
P_0=\mu_0,\,
P_T=\mu_T
\right\}
$$

为定义。

Lipman、Chen、Ben-Hamu、Nickel 与 Le 在 2023 年论文 [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747 "官方论文页面") 中先选择 conditional probability paths，再回归生成其 marginals 的 velocity。Population regression 可以精确恢复 chosen marginal field，但 coupling 与 path design 已经作为输入给定，不由 path-space entropy objective 选择。

Schrödinger Bridge 则把 reference 与两个 endpoints 同时放进 optimization。它不仅要求终点样本正确，还要求 endpoint coupling 与 conditional path fluctuations 相对于 $R$ 具有最小信息改动。

因此，方法之间可以共享 neural network、MSE、score、velocity 或 SDE solver，却仍然解决不同的 optimization problem。

## 2. 2021—2025：什么时候 one-time marginals 已经足够

如果任务只关心每个时刻的 population distribution，那么 Flow Matching 或 probability-flow ODE 可能已经足够。给定 density curve $(p_t)$ 与 velocity $v_t$，continuity equation

$$
\partial_t p_t
+
\nabla\cdot(p_tv_t)
=0
$$

规定 marginals 怎样变化。

但它不唯一确定 stochastic path law。Gyöngy 1986 年论文 [*Mimicking the One-Dimensional Marginal Distributions of Processes Having an Itô Differential*](https://doi.org/10.1007/BF00699039 "官方论文页面") 表明，不同 processes 可以共享 $p_t$，却有不同的 temporal correlations 与 transition kernels。

下图展示两类 sample paths 在多个固定时刻具有相同 histograms，却保留不同的随机轨迹结构。

![相同 one-time marginals 不等于相同 path law](/images/bridge/B12_same_marginals_paths.png)

因此需要 Bridge 的关键信号不是“任务包含两个分布”，而是下面至少一项成立：

- endpoint pairing 本身具有含义；
- 中间 stochastic trajectories 是推断或控制对象；
- reference dynamics 编码真实物理、几何或先验；
- 需要对偏离 reference 的代价作可解释比较；
- downstream task 依赖 transition law，而不只依赖 snapshots。

反过来，如果只需要从 base distribution 生成 data samples，且没有独立 meaningful reference 或第二个 hard endpoint，普通 Diffusion、Flow 或 one-sided transport 通常更直接。

Stochastic interpolants 进一步说明了 coupling 的作用。Albergo、Boffi 与 Vanden-Eijnden 在 2023—2025 年工作 [*Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*](https://arxiv.org/abs/2303.08797 "官方论文页面") 中从 endpoint coupling、interpolation 和 latent noise 构造共享 marginals 的 ODE/SDE family。只有加入额外的 variational objective 与条件，chosen interpolant 才成为 Schrödinger Bridge；任意 stochastic interpolation 本身并不具有 reference-relative optimality。

## 3. 2019—2024：真正适合 Bridge 的任务具有怎样的信息结构

应用不应按领域名称判断，而应按“我们实际观察到了什么”判断。

**双端 paired transformation。** 在 image restoration 中，如果 degraded observation 与 clean target 成对，任务已经提供 conditional endpoint information。Liu et al. 2023 的 [*I$^2$SB*](https://arxiv.org/abs/2302.05872 "官方论文页面") 用 analytic Gaussian bridges 构造 paired conditional diffusion；评价时仍须分开 paired supervision 与 bridge objective 的贡献。

**Population snapshots。** Single-cell experiments 通常只有 marginals，没有真实 lineage。Weinreb et al. 2018 年论文 [*Fundamental Limits on Dynamic Inference from Single-Cell Snapshots*](https://doi.org/10.1073/pnas.1714723115 "官方论文页面") 证明了这种非识别性；[*Waddington-OT*](https://doi.org/10.1016/j.cell.2019.01.006 "官方论文页面") 选择 state coupling，[*LineageOT*](https://doi.org/10.1038/s41467-021-25133-1 "官方论文页面") 则加入 lineage。Bridge 也不能从不足的数据中恢复唯一历史。

**有物理意义的 control。** 若 $R$ 是可信的 uncontrolled dynamics，path KL 可解释为 control energy。Chen、Georgiou 与 Pavon 2021 的 [*Stochastic Control Liaisons*](https://doi.org/10.1137/20M1339982 "官方论文页面") 说明 reference drift、diffusion channel 与 control cost 如何共同定义最小能量 steering。

**Rare-event conditioning。** Hartmann–Schütte 2012 的 [*Efficient Rare Event Simulation by Optimal Nonequilibrium Forcing*](https://doi.org/10.1088/1742-5468/2012/11/P11004 "官方论文页面") 研究 finite-horizon forcing；Chetrite–Touchette 2015 的 [*Nonequilibrium Markov Processes Conditioned on Large Deviations*](https://doi.org/10.1007/s00023-014-0375-8 "官方论文页面") 研究 long-time driven process。二者都不同于 fixed endpoint bridge。

## 4. 从理论到实践：Bridge 最容易在哪些地方失败

Bridge 的额外结构同时带来额外失败模式。

第一是 reference misspecification。$P^*$ 只在“相对于 $R$”的意义下最优。如果 $R$ 的 support、noise geometry 或 drift 与真实系统严重不符，entropy projection 可能得到数学上精确、科学上无意义的路径。

第二是 coupling non-identifiability。两个 endpoint marginals 通常允许许多 couplings。Bridge 用 $R_{0T}$ 选择其中一个，但数据本身未必能验证这个选择。若 endpoint pairing 是科学结论，就需要 lineage、paired observations、interventions 或独立机制证据。

第三是 learned-operator error。高维算法通常同时近似 coupling、conditional bridges、Markovian projection 和 sampler。Endpoint metrics 良好不能证明 path KL、transition kernels 或 intermediate dynamics 正确。

第四是 support 与 feasibility。若 target marginals 要求 reference support 外的 transport，path KL 为 $+\infty$。Neural network 可能仍输出一条连接两端的轨迹，却不再是相对于原 reference 的 finite-entropy Bridge。

第五是计算成本。相比一次 score/velocity training，iterative Bridge 方法可能需要多轮 forward/backward simulation、多个 networks、endpoint coupling estimation 和更复杂的 stopping criteria。

因此评估至少要区分 endpoint、intermediate marginal、path/dynamics、reference-relative objective 与 compute。只报告 sample quality 或 endpoint discrepancy，无法证明任务真正需要的 stochastic dynamics 已被恢复。

## 5. 一套面向任务的方法选择标准

可以用下面的顺序判断是否需要 Schrödinger Bridge。

首先问是否存在两个必须同时满足的 hard endpoint distributions。若只有一个 data distribution 与一个方便选择的 base prior，Diffusion 或 Flow 通常更自然。

其次问 reference dynamics 是否具有独立意义。如果 reference 只是为了让训练方便而临时选择，就很难解释为何需要 reference-relative path optimality；如果它来自物理模型、已知 Markov process、uncontrolled dynamics 或可信 corruption mechanism，Bridge 的建模价值更强。

第三问输出是否需要完整 stochastic path law：

$$
\begin{cases}
\text{只要 endpoint pairing}
&\Rightarrow
\text{优先考虑 OT},\\
\text{只要生成 samples 或 marginal path}
&\Rightarrow
\text{优先考虑 Diffusion / Flow},\\
\text{需要双端约束下的 reference-aware trajectories}
&\Rightarrow
\text{考虑 Schrödinger Bridge}.
\end{cases}
$$

第四问是否有能力验证新增结构。若无法检查 coupling、intermediate marginals、dynamics 或 reference sensitivity，复杂 Bridge 系统可能无法被证伪，也就很难证明它优于更简单方法。

最后问计算预算是否允许 repeated projections 或 conditional bridge estimation。选择 Bridge 不只是换一个 loss，还可能改变训练循环、采样过程和误差账本。

现在可以回答整篇 Blog 的最终问题：

> 当任务同时具有两个真实端点约束、一个有意义的 reference dynamics，并且关心的不只是终点样本而是中间 stochastic evolution 时，Schrödinger Bridge 才提供其他方法没有直接给出的建模对象。

Schrödinger Bridge 的价值不在于把所有分布变换统一成一个名字，而在于明确回答一个更严格的问题：

$$
\boxed{
\text{在尊重 reference dynamics 的前提下，怎样以最小信息改动满足两个端点观测？}
}
$$

## 本章论文索引

| 时间   | 论文                                                                                       | 本章中的作用                                                              |
| ---- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 1942 | Kantorovich, *On the Translocation of Masses*                                            | 将 transport 表述为具有双边缘约束的 coupling optimization                       |
| 1986 | Gyöngy, *Mimicking the One-Dimensional Marginal Distributions...*                        | 说明相同 one-time marginals 不唯一确定完整 path law                            |
| 2000 | Benamou & Brenier, *A Computational Fluid Mechanics Solution...*                         | 给出 quadratic OT 的 deterministic dynamic formulation                 |
| 2012 | Hartmann & Schütte, *Efficient Rare Event Simulation by Optimal Nonequilibrium Forcing*  | 连接 rare-event forcing 与 KL/control 视角                               |
| 2015 | Chetrite & Touchette, *Nonequilibrium Markov Processes Conditioned on Large Deviations*  | 区分 long-time conditioned process 与 driven dynamics                  |
| 2018 | Weinreb et al., *Fundamental Limits on Dynamic Inference from Single-Cell Snapshots*     | 说明 snapshots 对真实 trajectories 的非识别性                                 |
| 2019 | Schiebinger et al., *Waddington-OT*                                                      | 用 OT coupling 推断跨时刻 cell-state transitions                          |
| 2020 | Ho et al., *Denoising Diffusion Probabilistic Models*                                    | 提供固定 forward noising law 的 Diffusion 对照                             |
| 2021 | Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* | 统一 score diffusion 的 forward SDE、reverse SDE 与 probability-flow ODE |
| 2021 | Forrow et al., *LineageOT*                                                               | 说明 lineage information 如何改变 trajectory inference                    |
| 2021 | Chen, Georgiou & Pavon, *Stochastic Control Liaisons...*                                 | 说明 meaningful reference 下的 stochastic control 用途                    |
| 2023 | Lipman et al., *Flow Matching for Generative Modeling*                                   | 提供 chosen path 与 learned velocity 的 Flow 对照                         |
| 2023 | Liu et al., *I$^2$SB*                                                                    | 展示 paired image-to-image conditional Bridge                         |
| 2025 | Albergo, Boffi & Vanden-Eijnden, *Stochastic Interpolants*                               | 区分 chosen interpolation 与 variationally selected Bridge             |
