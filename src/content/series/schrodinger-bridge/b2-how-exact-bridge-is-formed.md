---
title: 为什么“最可能演化”会得到这样的解
description: 从端点耦合出发，解释路径空间 KL 如何产生 Schrödinger system 与 Entropic OT 结构。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: schrodinger-bridge
order: 2
slug: b2-how-exact-bridge-is-formed
tags:
  - schrodinger-bridge
  - schrodinger-system
  - entropic-ot
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 覆盖大偏差背景、KL chain rule、dynamic-to-static reduction、Schrödinger factors 与熵正则输运。
---
Schrödinger Bridge 的定义只有一行：在满足两个端点分布的路径律中，选择相对参考过程 $R$ 的 KL 最小者。真正值得追问的是，这一行优化究竟选中了什么。

一个常见但不准确的说法是：它在所有可能轨迹中挑出“最好的一条”。实际上，优化对象始终是概率分布。它先决定哪些起点更可能与哪些终点配对，再决定每对端点之间保留怎样的随机波动。把这两件事拆开，解的结构就不再神秘。

## 1. 两个边缘分布没有规定谁走向谁

设状态空间只有 $A,B$ 两点，初始和终止分布相同：

$$
\mu_0=\mu_T=
\begin{pmatrix}
1/2\\[2pt]
1/2
\end{pmatrix}.
$$

下面两种端点联合分布都符合观测：

$$
\gamma_{\mathrm{stay}}
=
\begin{pmatrix}
1/2 & 0\\
0 & 1/2
\end{pmatrix},
\qquad
\gamma_{\mathrm{swap}}
=
\begin{pmatrix}
0 & 1/2\\
1/2 & 0
\end{pmatrix}.
$$

矩阵的行对应起点，列对应终点。前一个耦合让所有质量留在原状态，后一个让两边完全交换。两者的行和、列和都等于 $\mu_0,\mu_T$，但表达的迁移完全不同。

这个例子简单得近乎刻意，它恰好说明：边缘分布只规定每一端有多少质量，没有规定质量怎样配对。

一般地，用 $P$ 表示一条候选路径律，并记

$$
\gamma=P_{0T}
$$

为随机端点 $(X_0,X_T)$ 在 $P$ 下的联合分布。满足双端约束意味着

$$
\gamma\in\Pi(\mu_0,\mu_T),
$$

其中 $\Pi(\mu_0,\mu_T)$ 是所有第一边缘为 $\mu_0$、第二边缘为 $\mu_T$ 的端点耦合集合。Schrödinger Bridge 必须从这个集合中选出一个 $\gamma^*$。

## 2. 相对熵为什么会出现在这里

Schrödinger 在 1931 年论文 *Über die Umkehrung der Naturgesetze* 和 1932 年论文 Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique（补充材料暂未公开） 中考虑大量独立粒子的罕见迁移。1931 年原文可参照 Chetrite、Muratore-Ginanneschi 与 Schwieger 2021 年的同行评审英译 [*E. Schrödinger's 1931 Paper “On the Reversal of the Laws of Nature”*](https://doi.org/10.1140/epjh/s13129-021-00032-7 "官方论文页面")。

用现代记号，设 $X^1,\ldots,X^N$ 是 $N$ 条相互独立、均服从参考路径律 $R$ 的样本路径。它们的经验路径测度为

$$
L_N
=
\frac1N
\sum_{k=1}^N
\delta_{X^k},
$$

其中 $\delta_{X^k}$ 是集中在第 $k$ 条路径上的 Dirac 测度。Sanov 在 1957 年论文 *On the Probability of Large Deviations of Random Variables* 中建立的经验分布大偏差原理，在相应条件下给出

$$
\Pr(L_N\approx P)
\asymp
\exp\!\left[-N\operatorname{KL}(P\Vert R)\right].
$$

$\asymp$ 表示 $N\to\infty$ 时的指数尺度，不是有限样本下的等式。候选路径律 $P$ 的 KL 越小，参考粒子的经验行为偶然接近它所需付出的指数代价就越低。因此，在已经观察到双端涨落的条件下，KL 最小者正是大粒子数意义下最可能的解释。

Léonard 在 2010 年论文 [*Entropic Projections and Dominating Points*](https://doi.org/10.1051/ps/2009003 "官方论文页面") 中给出了 Gibbs conditioning 的严格框架。

所以，路径空间 KL 不是凭方便挑出的 loss；但把这条现代公式原样归给 Schrödinger，也不符合历史。原文给出了粒子迁移和乘法因子的直觉，路径测度与大偏差语言来自后来的概率论。

## 3. 路径空间 KL 可以按端点拆开

给定端点 $X_0=x,X_T=y$ 后，记 $P^{xy}$ 为 $P$ 的条件路径律。于是

$$
P(d\omega)
=
\int
P^{xy}(d\omega)
\,\gamma(dx,dy).
$$

参考路径律 $R$ 也有同样的分解。记 $R_{0T}$ 为参考端点耦合，$R^{xy}$ 为参考过程在固定端点 $(x,y)$ 后的条件桥：

$$
R(d\omega)
=
\int
R^{xy}(d\omega)
\,R_{0T}(dx,dy).
$$

Föllmer 与 Gantert 在 1997 年论文 Entropy Minimization and Schrodinger Processes in Infinite Dimensions（补充材料暂未公开） 中讨论了无限维熵分解；Léonard 2014 的综述则把端点 disintegration 与动态、静态 Schrödinger problem 放在同一套记号下。

若 $P\ll R$，Radon–Nikodym 导数可以分成端点密度比与条件路径密度比。对一条端点为 $(x,y)$ 的路径 $\omega$，形式上有

$$
\frac{dP}{dR}(\omega)
=
\frac{d\gamma}{dR_{0T}}(x,y)
\frac{dP^{xy}}{dR^{xy}}(\omega).
$$

取对数并在 $P$ 下求期望，得到 KL chain rule：

$$
\boxed{
\operatorname{KL}(P\Vert R)
=
\operatorname{KL}(\gamma\Vert R_{0T})
+
\int
\operatorname{KL}(P^{xy}\Vert R^{xy})
\,\gamma(dx,dy).
}
$$

第一项只计较端点怎样配对，第二项计较固定端点后是否还要改造参考桥。

我认为理解这条公式最简单的方法，是先把 $\gamma$ 固定。此时第一项已经确定，第二项又总是非负，所以最优选择只能是

$$
P^{xy}=R^{xy},
\qquad
\gamma\text{-a.e.}
$$

也就是说，Schrödinger Bridge 不会在每一对端点内部重新发明一套轨迹；它直接保留参考过程的条件桥，只优化端点混合方式。

因此，原来的路径空间问题精确化简为

$$
\boxed{
\inf_{\substack{
P_0=\mu_0\\
P_T=\mu_T
}}
\operatorname{KL}(P\Vert R)
=
\inf_{\gamma\in\Pi(\mu_0,\mu_T)}
\operatorname{KL}(\gamma\Vert R_{0T}).
}
$$

若右侧存在最优耦合 $\gamma^*$，完整路径律由

$$
\boxed{
P^*(d\omega)
=
\int
R^{xy}(d\omega)
\,\gamma^*(dx,dy)
}
$$

重建。这是固定噪声水平下的精确等式，不是 small-noise approximation。

![Path KL 分解为 endpoint coupling 与 conditional bridge 两层](/images/bridge/B2_path_kl_projection.png)

## 4. Schrödinger system 不是额外假设

端点问题已经变成

$$
\gamma^*
=
\arg\min_{\gamma\in\Pi(\mu_0,\mu_T)}
\operatorname{KL}(\gamma\Vert R_{0T}).
$$

先看有限状态情形。设起点有 $m$ 个状态，终点有 $n$ 个状态；用严格为正的矩阵 $Q=(Q_{ij})$ 表示参考端点耦合，其中 $Q_{ij}>0$。目标边缘分别写成 $\mu=(\mu_i)$ 和 $\nu=(\nu_j)$。候选耦合 $\gamma=(\gamma_{ij})$ 需要满足

$$
\sum_j\gamma_{ij}=\mu_i,
\qquad
\sum_i\gamma_{ij}=\nu_j.
$$

优化目标是

$$
\operatorname{KL}(\gamma\Vert Q)
=
\sum_{i,j}
\gamma_{ij}
\log\frac{\gamma_{ij}}{Q_{ij}}.
$$

对行、列约束分别引入 Lagrange multipliers $\alpha_i,\beta_j$。最优点的 stationarity condition 为

$$
\log\frac{\gamma_{ij}}{Q_{ij}}
+1-\alpha_i-\beta_j=0.
$$

指数化后，最优耦合必然具有乘法形式

$$
\boxed{
\gamma^*_{ij}
=
u_iQ_{ij}v_j,
}
$$

其中 $u_i>0$ 只依赖起点状态，$v_j>0$ 只依赖终点状态。代回边缘约束便得到

$$
\boxed{
u_i(Qv)_i=\mu_i,
\qquad
v_j(Q^\top u)_j=\nu_j.
}
$$

这就是有限状态下的 Schrödinger system。它不是为了方便计算而猜出的 ansatz，而是 endpoint KL projection 的一阶最优性条件。

因子本身有尺度自由度。对任意常数 $c>0$，变换

$$
u\mapsto cu,
\qquad
v\mapsto c^{-1}v
$$

不会改变 $\gamma^*$。因此 $u,v$ 只在这一尺度变换下唯一；严格正有限情形中的耦合 $\gamma^*$ 则由 KL 的严格凸性唯一确定。

Schrödinger 在 1932 年原文中已经写出迁移因子与耦合方程。Fortet 在 1940 年论文 Résolution d'un système d'équations de M. Schrödinger（补充材料暂未公开） 中，在特定 positivity、continuity 与 integrability 条件下证明存在性与尺度意义下的唯一性；Jamison 在 1974 年论文 [*Reciprocal Processes*](https://doi.org/10.1007/BF00532864 "官方论文页面") 中，把乘法端点结构与 reciprocal/Markov process 联系起来。

![Schrödinger system、左右 factors 与目标 marginals](/images/bridge/B3_schrodinger_system.png)

严格正矩阵是最干净的教学模型，不是一般存在性定理。参考耦合存在 structural zeros，或者连续空间中的 support、可积性与 dual attainment 失败时，可行性和 factorization 都要重新检查。

## 5. Entropic OT 从哪里进入

Schrödinger problem 与 Entropic OT 的关系来自参考端点耦合的具体形式，并不是一句“Bridge 就是带噪声的 OT”。

取基础耦合

$$
M=\mu_0\otimes\mu_T,
$$

给定 transport cost $c(x,y)$ 和尺度 $\varepsilon>0$，并定义 Gibbs 形式的参考耦合

$$
R_{0T}^{\varepsilon}(dx,dy)
=
\frac1{Z_\varepsilon}
\exp\!\left(-\frac{c(x,y)}{\varepsilon}\right)
M(dx,dy),
$$

其中 $Z_\varepsilon$ 是归一化常数。对任意 $\gamma\in\Pi(\mu_0,\mu_T)$，展开 KL 得到

$$
\varepsilon
\operatorname{KL}
\left(
\gamma\Vert R_{0T}^{\varepsilon}
\right)
=
\int c(x,y)\,\gamma(dx,dy)
+
\varepsilon
\operatorname{KL}(\gamma\Vert M)
+
\varepsilon\log Z_\varepsilon.
$$

最后一项与 $\gamma$ 无关，所以 KL projection 与下面的问题具有相同的最优耦合：

$$
\boxed{
\min_{\gamma\in\Pi(\mu_0,\mu_T)}
\left\{
\int c\,d\gamma
+
\varepsilon
\operatorname{KL}
\left(
\gamma\Vert\mu_0\otimes\mu_T
\right)
\right\}.
}
$$

这就是 Entropic OT 的静态形式。只有当参考端点耦合能相对于某个基础测度写成 Gibbs density 时，才会自然出现“transport cost 加 entropy regularization”。对于任意参考过程，始终正确的对象是 $\operatorname{KL}(\gamma\Vert R_{0T})$，未必存在预先给定的 Euclidean cost。

Mikami 在 2004 年论文 [*Monge's Problem with a Quadratic Cost by the Zero-Noise Limit of h-Path Processes*](https://doi.org/10.1007/s00440-004-0340-4 "官方论文页面") 中研究 Brownian $h$-path 的零噪声极限；Léonard 在 2012 年论文 [*From the Schrödinger Problem to the Monge–Kantorovich Problem*](https://doi.org/10.1016/j.jfa.2011.11.026 "官方论文页面") 中用 large deviations 与 $\Gamma$-convergence 说明适当缩放的 Schrödinger problems 怎样趋向 classical OT。

![Entropic couplings 随 regularization scale 的变化](/images/bridge/B5_entropic_couplings.png)

有限 $\varepsilon$ 下的 KL 恒等式与 $\varepsilon\to0$ 时的收敛是两类结论。后者还需要 tightness、coercivity 和 limiting optimizer uniqueness 等条件，不能用一张越来越尖锐的耦合图代替证明。

## 文献索引

| 时间        | 论文                                                                                               | 本章采用的内容                                    |
| --------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| 1931/1932 | Schrödinger, *Über die Umkehrung der Naturgesetze*；*Sur la théorie relativiste de l'électron...* | 提出罕见端点观测下的最可能迁移问题与乘法因子                     |
| 1940      | Fortet, *Résolution d'un système d'équations de M. Schrödinger*                                  | 建立 Schrödinger system 的存在性路线               |
| 1957      | Sanov, *On the Probability of Large Deviations of Random Variables*                              | 给出相对熵作为经验分布大偏差速率                           |
| 1974      | Jamison, *Reciprocal Processes*                                                                  | 连接乘法端点结构、reciprocal process 与 Markovity    |
| 1997      | Föllmer & Gantert, *Entropy Minimization and Schrodinger Processes in Infinite Dimensions*       | 给出无限维路径熵分解与最优过程结构                          |
| 2004      | Mikami, *Monge's Problem with a Quadratic Cost by the Zero-Noise Limit of h-Path Processes*      | 研究 Brownian Schrödinger problem 的零噪声 OT 极限 |
| 2010      | Léonard, *Entropic Projections and Dominating Points*                                            | 给出 Gibbs conditioning 的严格框架                |
| 2012      | Léonard, *From the Schrödinger Problem to the Monge–Kantorovich Problem*                         | 连接 Schrödinger problem 与 classical OT      |
| 2014      | Léonard, *A Survey of the Schrödinger Problem...*                                                | 统一动态、静态与 transport 表述                      |
| 2021      | Chetrite, Muratore-Ginanneschi & Schwieger, *E. Schrödinger's 1931 Paper...*                     | 提供 1931 原文英译与历史评注                          |
