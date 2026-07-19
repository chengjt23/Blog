---
title: Schrödinger Bridge 是什么
description: 从两次群体观测出发，说明 reference dynamics、随机路径律与 Schrödinger Bridge 要解决的问题。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 1
slug: b1-schrodinger-bridge-overview
tags:
  - schrodinger-bridge
  - path-space
  - relative-entropy
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 介绍双端分布约束、参考路径律，以及它与普通插值和固定端点 bridge 的区别。
---
假设我们在早上和晚上各拍了一张照片。照片里不是一个人，而是一大群粒子：早上的照片告诉我们它们从哪里出发，晚上的照片告诉我们它们最后分布在哪里。我们还知道，在没有晚间观测时，这些粒子大致会遵循怎样的随机动力学。

现在的问题是：中间发生了什么？

只把两张照片叠在一起，再画几条平滑曲线，并不能回答这个问题。许多完全不同的随机过程都可以拥有相同的初始分布和终止分布。Schrödinger Bridge 的特别之处，在于它不凭空挑选一条看起来顺眼的插值，而是尽量保留原有动力学，只做满足两次观测所必需的改动。

这就是整篇 Blog 要讨论的对象。

## 1. 从两次群体观测说起

把初始时刻记为 $0$，终止时刻记为 $T$。粒子群在这两个时刻的分布分别记为 $\mu_0$ 和 $\mu_T$。这里的 $\mu_0,\mu_T$ 都是群体分布，而不是某个粒子的具体位置。

这个区别很重要。假如我们知道一个粒子从 $x$ 出发，最后到达 $y$，问题只是“它在这两个点之间怎样走”。但现在我们只知道一群粒子开始时怎样分布、结束时怎样分布，并不知道哪个起点对应哪个终点。

即使端点配对已经知道，中间轨迹仍然不唯一。粒子可以几乎沿直线移动，也可以剧烈波动后再到达终点；它们可以保留很强的时间相关性，也可以迅速忘掉初始状态。两次群体观测没有包含这些信息。

所以，Schrödinger Bridge 研究的不是两个分布之间的几何连线，而是符合两次观测的完整随机演化。

## 2. “最可能”必须相对于一套动力学

仅仅说“找最可能的演化”仍然不完整。最可能是相对于什么而言？

我们用 $R$ 表示参考路径律，也就是没有施加终端约束时，系统原本遵循的随机动力学。$R$ 可以是 Brownian motion，可以是带漂移的 diffusion，也可以来自一个已有的物理模型或 Markov process。它不仅描述粒子在每个时刻可能出现在哪里，还描述整条路径怎样波动。

选择不同的 $R$，会得到不同的 Bridge。以 Brownian motion 为参考，模型偏好连续而扩散的轨迹；如果参考过程带有稳定漂移，逆着漂移运动就会付出更高代价；如果某些区域在 $R$ 下根本不可达，Bridge 也不能凭空穿过这些区域。

因此，我更愿意把 $R$ 看成问题本身的一部分，而不是训练时附加的正则项。参考动力学选错了，优化仍可能给出一个数学上精确的答案，但那个答案回答的是另一个问题。

## 3. 它与普通插值、Brownian bridge 有什么不同

这三个对象经常画成相似的图，却解决不同的问题。

普通分布插值先规定一条从 $\mu_0$ 到 $\mu_T$ 的边缘分布曲线。线性插值、displacement interpolation 或人为选择的 probability path 都属于这一类。它们告诉我们每个时刻的群体分布，却未必确定多时刻之间的相关性，更没有要求轨迹接近某个参考动力学。

Brownian bridge 则从一个具体起点 $x$ 出发，并条件于时刻 $T$ 到达具体终点 $y$。路径在中间仍然随机，只是在终点被迫汇合。Doob 在 1957 年论文 [*Conditional Brownian Motion and the Boundary Limits of Harmonic Functions*](https://doi.org/10.24033/bsmf.1494 "官方论文页面") 中建立了 conditional Brownian motion 与 $h$-path 的经典理论；Fitzsimmons、Pitman 与 Yor 在 1993 年论文 Markovian Bridges: Construction, Palm Interpretation, and Splicing（补充材料暂未公开） 中把固定端点 bridge 推广到更一般的 Markov process。

![Brownian motion 与 Brownian bridge 的路径、均值和方差](/images/bridge/B1_brownian_bridge.png)

Schrödinger Bridge 面对的是两个分布，而不是一对已经给定的点。它既要决定哪些起点与哪些终点更可能配对，也要保留参考过程在每一对端点之间的随机波动。

这个区别看似只是术语，实际上决定了模型最终给出的是什么：普通插值给出一条边缘分布曲线，Brownian bridge 给出固定端点后的条件过程，Schrödinger Bridge 给出满足双端分布约束的完整路径律。

## 4. 一条公式定义这个问题

把一条完整轨迹看成一个随机对象，它在所有可能轨迹上的概率分布称为路径律（path law）。用 $P$ 表示一个候选路径律，$P_0$ 和 $P_T$ 分别表示它在时刻 $0$ 与 $T$ 的边缘分布。前面的观测要求 $P_0=\mu_0$ 且 $P_T=\mu_T$。

在所有满足这两个条件的路径律中，Schrödinger Bridge 选择相对参考路径律 $R$ 改动最小的一个：

$$
\boxed{
P^*
=
\arg\min_{\substack{
P_0=\mu_0\\
P_T=\mu_T
}}
\operatorname{KL}(P\Vert R).
}
$$

$P^*$ 是最终得到的路径律，$\operatorname{KL}(P\Vert R)$ 是 $P$ 相对于 $R$ 的路径空间相对熵。这里暂时可以把它理解为改变整套随机动力学所需付出的信息代价：候选过程越偏离 $R$ 原本偏好的路径，这个代价越大。

Léonard 在 2014 年综述 [*A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport*](https://doi.org/10.3934/dcds.2014.34.1533 "官方论文页面") 中系统整理了这一现代定义，以及它与 optimal transport、随机控制和大偏差理论的关系。

公式很短，但它已经排除了两个常见误解。首先，$P^*$ 不是一条确定性轨迹，而是整条随机路径的概率分布。其次，优化不是在所有想象得到的轨迹之间追求几何上的“短”，而是在参考模型 $R$ 允许的世界里寻找最小改动。

## 5. 这个问题是怎样出现的

Schrödinger 在 1931 年论文 *Über die Umkehrung der Naturgesetze* 中提出了这个问题，并在 1932 年论文 Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique（补充材料暂未公开） 的 Section VII 再次讨论。1931 年文本可以参照 Chetrite、Muratore-Ginanneschi 与 Schwieger 在 2021 年发表的同行评审英译 [*E. Schrödinger's 1931 Paper “On the Reversal of the Laws of Nature”*](https://doi.org/10.1140/epjh/s13129-021-00032-7 "官方论文页面")。

他考虑的是一次罕见的群体涨落：大量粒子按照已知规律独立迁移，初始分布已经准备好，终点却观察到一个在原动力学下很不寻常的分布。既然这次涨落已经发生，最可能通过怎样的整体迁移产生？

这个问题后来才逐渐获得今天的数学形式。Schrödinger 原文使用的是粒子迁移概率、组合 likelihood 与乘法因子，并没有写出上面的 path-space KL optimization。Föllmer 与 Gantert 在 1997 年论文 Entropy Minimization and Schrodinger Processes in Infinite Dimensions（补充材料暂未公开） 中讨论了无限维路径空间上的熵最小化；Léonard 2014 则把动态问题、静态耦合和 optimal transport 的联系系统地整理到同一框架中。

![Reference migration、条件 likelihood 与 endpoint KL 极限](/images/bridge/B0_migration_likelihood.png)

把今天的路径空间 KL 公式直接归给 Schrödinger，是一种时代错置。更准确的说法是：Schrödinger 提出了罕见端点观测下的最可能迁移问题，后来的概率论把“最可能”严格化为相对于参考路径律的熵投影。

这个定义也有清楚的边界。它不会从两次群体观测中恢复唯一的真实历史；它给出的是在所选参考模型 $R$ 下最合理的解释。一般状态空间中的解是否存在、高维情况下怎样计算，都还需要额外条件和方法。

Schrödinger Bridge 给出的是模型 $R$ 内最可能的演化，不是演化本身的唯一真相。

## 文献索引

| 时间        | 论文                                                                                               | 本章采用的内容                                      |
| --------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| 1931/1932 | Schrödinger, *Über die Umkehrung der Naturgesetze*；*Sur la théorie relativiste de l'électron...* | 提出罕见终端分布下的最可能粒子迁移问题                          |
| 1957      | Doob, *Conditional Brownian Motion and the Boundary Limits of Harmonic Functions*                | 建立 conditional Brownian motion 与 $h$-path 理论 |
| 1993      | Fitzsimmons, Pitman & Yor, *Markovian Bridges*                                                   | 构造一般 Markov process 的固定端点 bridge             |
| 1997      | Föllmer & Gantert, *Entropy Minimization and Schrodinger Processes in Infinite Dimensions*       | 讨论无限维路径空间熵最小化                                |
| 2014      | Léonard, *A Survey of the Schrödinger Problem...*                                                | 整理现代 Schrödinger problem 及其主要联系              |
| 2021      | Chetrite, Muratore-Ginanneschi & Schwieger, *E. Schrödinger's 1931 Paper...*                     | 提供 1931 年原文的同行评审英译与历史评注                      |
