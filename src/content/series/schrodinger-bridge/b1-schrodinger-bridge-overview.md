---
title: Schrödinger Bridge 的全景图：输入、输出与核心问题
description: 从起点分布、终点分布与参考路径律出发，建立 Schrödinger Bridge 的完整心智模型。
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
scope: 定义三个输入与完整路径律输出，区分端点耦合、条件桥、普通插值和 Schrödinger Bridge。
---
设想这样一个问题：我们在时刻 $0$ 观察到一群粒子的分布，在时刻 $T$ 又观察到它们的分布；同时，我们知道如果没有第二次观测，这些粒子本来会按照怎样的随机动力学运动。现在希望推断：

> 在满足两个端点观测的所有随机演化中，哪一种对原有动力学的改动最小？

这就是 Schrödinger Bridge 的核心。它接收三个输入：

$$
\text{起点分布 }\mu_0,
\qquad
\text{终点分布 }\mu_T,
\qquad
\text{参考路径律 }R,
$$

并输出一个新的 path law $P^*$。这个输出不是一条确定性曲线，也不只是一族中间时刻的 probability densities，而是对整条随机路径的联合分布。

Léonard 在 2014 年综述 [*A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport*](https://doi.org/10.3934/dcds.2014.34.1533 "官方论文页面") 中给出了现代统一入口。整章可以先压缩为一张最小全景图：

$$
\boxed{
(\mu_0,\mu_T,R)
\xrightarrow{\ \text{path-space KL projection}\ }
P^*.
}
$$

理解这张图需要依次回答四件事：为什么两个 endpoint marginals 还不够，reference path law 究竟规定了什么，Brownian bridge 与 Schrödinger Bridge 有何区别，以及 relative entropy 为什么能够表达“最可能的演化”。

## 1. 2014：三个输入怎样定义一个 Bridge

先固定最常见的连续状态空间 $\mathbb R^d$，并令

$$
\Omega=C([0,T],\mathbb R^d)
$$

表示从时刻 $0$ 到 $T$ 的连续路径空间。对任意路径 $\omega\in\Omega$，记

$$
X_t(\omega)=\omega(t)
$$

为时刻 $t$ 的 coordinate map。

Reference path law $R$ 是 $\Omega$ 上的概率测度。它描述没有施加双端约束时，系统本来怎样运动。例如，$R$ 可以是 Brownian motion、带 drift 的 diffusion、离散 Markov chain，或者受几何约束的随机过程。

Candidate path law $P$ 是另一个定义在 $\Omega$ 上的概率测度。记

$$
P_t=(X_t)_\#P
$$

为 $P$ 在时刻 $t$ 的 marginal distribution，其中 pushforward $(X_t)_\#P$ 表示从随机路径中只取时刻 $t$ 的状态。双端观测要求

$$
P_0=\mu_0,
\qquad
P_T=\mu_T.
$$

满足这两个约束的 path laws 通常有无穷多个。Schrödinger Bridge 使用 path-space relative entropy 从中选择对 reference 改动最小者：

$$
\boxed{
P^*
=
\arg\min_{
P\in\mathcal P(\Omega):
P_0=\mu_0,\,
P_T=\mu_T
}
\operatorname{KL}(P\Vert R).
}
$$

这里 $\mathcal P(\Omega)$ 表示 $\Omega$ 上的概率测度集合。若 $P$ 对 $R$ 绝对连续，记为 $P\ll R$，则

$$
\operatorname{KL}(P\Vert R)
=
\int_\Omega
\log\!\left(
\frac{dP}{dR}
\right)dP,
$$

其中 $dP/dR$ 是 Radon–Nikodym derivative。若 $P\not\ll R$，则定义

$$
\operatorname{KL}(P\Vert R)=+\infty.
$$

这个 $+\infty$ 不是形式上的补丁。它表示 candidate 使用了 reference 认为不可能的路径事件，因此无法通过有限的信息改动从 $R$ 得到。

三个输入的职责彼此不同。起点分布 $\mu_0$ 与终点分布 $\mu_T$ 是观测约束；reference law $R$ 是关于中间动力学的先验；relative entropy 则规定“尽可能保留 reference”究竟是什么意思。改变任何一个输入，通常都会改变最终的 $P^*$。

这也解释了为什么 reference 不是计算时随手加入的 noise。两条 reference processes 即使具有相同的起点和终点 marginals，也可能允许不同路径、偏好不同 endpoint pair，并在中间产生不同波动。更换 reference，相当于改变问题本身。

## 2. 2014：为什么两个 endpoint marginals 还不能决定中间演化

第一层缺失信息是 endpoint coupling。

设状态空间只有 $A,B$ 两点，并令初末分布都为

$$
\mu_0=\mu_T=
\begin{pmatrix}
1/2\\[2pt]
1/2
\end{pmatrix}.
$$

下面两张 coupling matrices 的行对应起点 $A,B$，列对应终点 $A,B$：

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

两者的 row marginals 和 column marginals 完全相同，但含义相反。$\gamma_{\mathrm{stay}}$ 让所有质量留在原状态，$\gamma_{\mathrm{swap}}$ 则让两边质量全部交换。只看时刻 $0$ 与 $T$ 的 snapshots，无法判断实际采用了哪种配对。

一般地，记

$$
\gamma=P_{0T}
$$

为 candidate path law $P$ 在两个端点上的 joint distribution。那么

$$
\gamma\in\Pi(\mu_0,\mu_T),
$$

其中 $\Pi(\mu_0,\mu_T)$ 是所有第一边缘为 $\mu_0$、第二边缘为 $\mu_T$ 的 couplings。双端 marginals 只规定 $\gamma$ 的行列和，并没有指定 $\gamma$ 本身。

第二层缺失信息是 conditional path law。即使 $\gamma$ 已经给定，每一对 endpoint $(x,y)$ 之间仍然可以有许多不同的随机运动方式。记 $P^{xy}$ 为给定

$$
X_0=x,
\qquad
X_T=y
$$

后的 conditional path law，则完整过程可以写成

$$
\boxed{
P(d\omega)
=
\int
P^{xy}(d\omega)
\,\gamma(dx,dy).
}
$$

因此，一个 path law 包含两层结构：$\gamma$ 决定总体怎样配对起点与终点，$P^{xy}$ 决定每一对端点之间怎样运动。Endpoint coupling 不是完整 path law，更不是一条确定性 transport map。

对 reference law $R$，相应地记 $R_{0T}$ 为 reference endpoint coupling，记 $R^{xy}$ 为给定端点后的 reference conditional bridge。Föllmer 与 Gantert 1997 年论文 Entropy Minimization and Schrodinger Processes in Infinite Dimensions（补充材料暂未公开） 以及 Léonard 2014 的统一表述给出一个核心结果：最优过程保留 conditional reference bridges，只重新选择 endpoint coupling。于是

$$
\boxed{
P^*(d\omega)
=
\int
R^{xy}(d\omega)
\,\gamma^*(dx,dy).
}
$$

$\gamma^*$ 是使 endpoint KL 相对于 $R_{0T}$ 取到最小值的 coupling。这个结论来自 relative entropy 对 endpoint map 的 chain rule；这里先保留它的结构含义，而不展开证明。

下图把问题分成 endpoint pairing 与 conditional paths 两层。两个 marginals 只固定最外侧的行列约束，Schrödinger Bridge 还要决定内部怎样配对，并用 reference bridges 补全每一对端点之间的随机路径。

![Endpoint coupling、conditional bridges 与完整 path law](/images/bridge/B2_path_kl_projection.png)

## 3. 1957—1993：Brownian bridge 只回答“给定一对端点后怎样走”

Brownian bridge 是理解 $R^{xy}$ 的最简单例子。

设 reference process 是 $\mathbb R^d$ 上扩散强度为 $\varepsilon>0$ 的 Brownian motion：

$$
dX_t
=
\sqrt{\varepsilon}\,dW_t,
\qquad
X_0=x,
$$

其中 $W_t$ 是 $d$ 维标准 Brownian motion，$x\in\mathbb R^d$ 是固定起点。现在再条件于固定终点

$$
X_T=y.
$$

对任意 $0<t<T$，Gaussian conditioning 给出

$$
\boxed{
X_t\mid(X_0=x,X_T=y)
\sim
\mathcal N\!\left(
\left(1-\frac tT\right)x+\frac tT y,\,
\varepsilon\frac{t(T-t)}{T}I
\right),
}
$$

其中 $I$ 是 $d\times d$ 单位矩阵。条件均值沿直线从 $x$ 移向 $y$，条件协方差在中间保持非零，并在 $t\to T$ 时收缩到零。因此 Brownian bridge 仍然是一条随机路径，而不是连接 $x$ 与 $y$ 的确定性线段。

同一个 conditional law 可以写成 SDE：

$$
\boxed{
dX_t
=
\frac{y-X_t}{T-t}\,dt
+
\sqrt{\varepsilon}\,dW_t,
\qquad
0\le t<T.
}
$$

Drift $(y-X_t)/(T-t)$ 在终点附近变得奇异，正是它迫使随机路径在时刻 $T$ 命中 $y$。

Doob 在 1957 年论文 [*Conditional Brownian Motion and the Boundary Limits of Harmonic Functions*](https://doi.org/10.24033/bsmf.1494 "官方论文页面") 中建立了 conditional Brownian motion 与 $h$-path 的经典理论背景。Fitzsimmons、Pitman 与 Yor 在 1993 年发表的 Markovian Bridges: Construction, Palm Interpretation, and Splicing（补充材料暂未公开） 中，在更一般的 positive transition-density 条件下构造 finite-time Markov bridges。

下图比较 unconditioned Brownian motion 与固定终点后的 Brownian bridge。路径在中间仍然分散，却在终点汇合。

![Brownian motion 与 Brownian bridge 的路径、均值和方差](/images/bridge/B1_brownian_bridge.png)

Brownian bridge 的输入是一个确定的 endpoint pair $(x,y)$，输出是 reference conditional law $R^{xy}$。Schrödinger Bridge 的输入则是两个 endpoint distributions $\mu_0,\mu_T$，还必须从所有可能的 endpoint pairings 中选择 $\gamma^*$。因此：

$$
\boxed{
\text{diffusion bridge}
=
\text{conditional building block},
\qquad
\text{Schrödinger Bridge}
=
\text{entropy-optimized mixture}.
}
$$

名字中都含 bridge，并不表示两者解决同一个问题。

## 4. 1931—2017：为什么“最可能”会变成 path-space KL

Schrödinger 的原始问题来自大量粒子的罕见涨落。他在 1931 年论文 *Über die Umkehrung der Naturgesetze* 中提出相关设想，并在 1932 年论文 Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique（补充材料暂未公开） 的 Section VII 中再次陈述。1931 文本可通过 Chetrite、Muratore-Ginanneschi 与 Schwieger 2021 年的同行评审英译 [*E. Schrödinger's 1931 Paper “On the Reversal of the Laws of Nature”*](https://doi.org/10.1140/epjh/s13129-021-00032-7 "官方论文页面") 核对。

原始设定可以概括为：许多粒子按照已知 reference dynamics 独立运动，初始分布已经准备好，却在终点观察到一个在 reference 下极其罕见的分布。问题不是“这个终端分布有多罕见”，而是：

> 在罕见观测已经发生的条件下，哪一种整体路径分布最可能解释它？

Relative entropy 的答案来自大粒子数极限。设 $X^1,\ldots,X^N$ 是 $N$ 条独立 reference paths，每条路径都服从 $R$。定义经验路径测度

$$
L_N
=
\frac1N
\sum_{k=1}^N
\delta_{X^k},
$$

其中 $\delta_{X^k}$ 是集中在第 $k$ 条样本路径上的 Dirac measure。Sanov 在 1957 年论文 *On the Probability of Large Deviations of Random Variables* 中建立了经验分布的大偏差原理；应用到路径样本时，它在相应条件下给出 logarithmic probability scale

$$
\Pr(L_N\approx P)
\asymp
\exp\!\left[
-N\operatorname{KL}(P\Vert R)
\right].
$$

符号 $\asymp$ 表示 $N\to\infty$ 时的指数尺度，不是有限 $N$ 的精确等式。一个 candidate path law $P$ 的 KL 越小，reference particles 的经验 path law 偶然接近它所需付出的指数代价就越低。

在 endpoint constraints 下，最低代价者正是

$$
\arg\min_{P_0=\mu_0,\,P_T=\mu_T}
\operatorname{KL}(P\Vert R).
$$

Léonard 在 2010 年论文 [*Entropic Projections and Dominating Points*](https://doi.org/10.1051/ps/2009003 "官方论文页面") 中给出 Gibbs conditioning 的正式框架：在 good large-deviation principle、连续约束映射与正概率 shrinking neighborhoods 等条件下，条件经验测度会集中到 constrained rate minimizers。

Schrödinger 原始实验的初始粒子位置通常是预先准备的，因此不能把普通 iid Sanov 不加说明地直接套用。Luçon 在 2017 年论文 [*Quenched Large Deviations for Interacting Diffusions in Random Media*](https://doi.org/10.1007/s10955-017-1719-9 "官方论文页面") 的 Proposition 2.2 中，在确定性初始 profile、Feller conditional kernel 等条件下提供相应的大偏差输入。

这里同样需要保留历史边界：Schrödinger 1931/1932 使用粒子迁移、组合 likelihood 与乘法因子，并没有写出今天的 path-space KL optimization。现代 relative-entropy 语言是后来概率论与大偏差理论对原始问题的严格化。

下图展示有限粒子 migration likelihood 如何在大样本下形成 endpoint KL 形状。它是原始粒子直觉与现代 entropy projection 之间的最小桥梁。

![Reference migration、条件 likelihood 与 endpoint KL 极限](/images/bridge/B0_migration_likelihood.png)

## 5. 这一定义解决了什么，又没有承诺什么

现在可以把三个相近但不同的对象放在一起：

| 对象                        | 已知信息                                    | 输出                                  | 没有解决的部分               |
| ------------------------- | --------------------------------------- | ----------------------------------- | --------------------- |
| Brownian/diffusion bridge | reference $R$ 与固定端点 $(x,y)$             | conditional law $R^{xy}$            | 不选择 endpoint coupling |
| marginal interpolation    | endpoint marginals 与 interpolation rule | 一族 one-time distributions $(\mu_t)$ | 不决定多时刻相关性与完整 path law |
| Schrödinger Bridge        | $\mu_0,\mu_T,R$                         | 最优 path law $P^*$                   | 不声称恢复唯一的真实历史          |

Schrödinger Bridge 的价值在于同时保留三个层面：端点观测由 $\mu_0,\mu_T$ 表达，中间动力学由 $R$ 表达，两者之间的最小改动由 $\operatorname{KL}(P\Vert R)$ 表达。

但这一定义本身不自动承诺三件事。第一，一般状态空间中的 optimizer 是否存在，需要 finite-entropy feasibility、support 与 compactness 等条件；第二，公式没有直接给出高维数值算法；第三，$P^*$ 是相对于所选 reference 最合理的解释，而不是脱离模型假设的唯一真实轨迹。

现在可以回答本章标题：

> Schrödinger Bridge 以起点分布、终点分布和 reference path law 为输入，在满足双端约束的所有随机路径分布中，选择相对 reference 的 path-space KL 最小者。它输出的是完整 path law，而不是单条曲线、单个 coupling 或一族彼此独立的中间 marginals。

任何理论解释或计算方法，都应明确自己固定了 $\mu_0,\mu_T,R,P^*$ 中的什么、近似了什么，以及最终是否仍然满足这一定义。

## 本章论文索引

| 时间        | 论文                                                                                               | 本章中的作用                                                        |
| --------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| 1931/1932 | Schrödinger, *Über die Umkehrung der Naturgesetze*；*Sur la théorie relativiste de l'électron...* | 提出罕见终端分布下的最可能粒子演化问题                                           |
| 1957      | Doob, *Conditional Brownian Motion and the Boundary Limits of Harmonic Functions*                | 建立 conditional Brownian motion 与 $h$-path 的经典背景               |
| 1957      | Sanov, *On the Probability of Large Deviations of Random Variables*                              | 使 relative entropy 成为经验分布偏离 reference law 的大偏差速率              |
| 1993      | Fitzsimmons, Pitman & Yor, *Markovian Bridges*                                                   | 严格构造一般 Markov reference 的固定端点 bridge                          |
| 1997      | Föllmer & Gantert, *Entropy Minimization and Schrodinger Processes in Infinite Dimensions*       | 建立无限维 path-space entropy minimization 与 conditional bridge 分解 |
| 2010      | Léonard, *Entropic Projections and Dominating Points*                                            | 给出大偏差约束下 entropy projection 的条件集中框架                           |
| 2014      | Léonard, *A Survey of the Schrödinger Problem...*                                                | 给出 Schrödinger Bridge 的现代统一表述                                 |
| 2017      | Luçon, *Quenched Large Deviations for Interacting Diffusions in Random Media*                    | 为确定性初始 profile 提供带条件的大偏差输入                                    |
| 2021      | Chetrite, Muratore-Ginanneschi & Schwieger, *E. Schrödinger's 1931 Paper...*                     | 提供 1931 原文的同行评审英译与历史评注                                        |
