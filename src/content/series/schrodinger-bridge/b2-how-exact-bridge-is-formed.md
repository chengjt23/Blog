---
title: 一个 Bridge 的精确解是怎样形成的
description: 沿路径空间 KL、端点 KL、Schrödinger factors 与双向动力学，推导精确 Bridge 的结构。
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
  - stochastic-control
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
  覆盖 dynamic-to-static reduction、Schrödinger system、Doob transform、Entropic OT
  与随机控制解释。
---
第一章把 Schrödinger Bridge 压缩成三个输入与一个输出：

$$
(\mu_0,\mu_T,R)
\xrightarrow{\ \text{path-space KL projection}\ }
P^*.
$$

$\mu_0$ 与 $\mu_T$ 是初末分布，$R$ 是 reference path law，$P^*$ 是满足双端约束且相对 $R$ 的 path-space KL 最小的 path law。这个定义很短，却没有告诉我们 $P^*$ 具体长什么样。

真正的结构来自连续发生的四次化简：

$$
\text{path-space KL}
\longrightarrow
\text{endpoint KL}
\longrightarrow
\text{Schrödinger factors}
\longrightarrow
\text{two-sided dynamics}.
$$

第一步把整条路径的优化降到 endpoint coupling；第二步说明最优 coupling 只能是 reference endpoint law 的乘法缩放；第三步把两个 endpoint factors 传播到所有中间时刻；第四步把同一个解分别读成 entropic transport 与 minimum-energy control。

## 1. 1997—2014：路径空间 KL 怎样拆成两层代价

继续令

$$
\Omega=C([0,T],\mathbb R^d)
$$

为连续路径空间。对任意 candidate path law $P$，记

$$
\gamma=P_{0T}
$$

为 $(X_0,X_T)$ 在 $P$ 下的 joint distribution。因为 $P_0=\mu_0$、$P_T=\mu_T$，所以

$$
\gamma\in\Pi(\mu_0,\mu_T).
$$

$\Pi(\mu_0,\mu_T)$ 是所有具有这两个 marginals 的 endpoint couplings。给定一对端点 $(x,y)$ 后，记 $P^{xy}$ 为 $P$ 的 conditional path law，于是

$$
P(d\omega)
=
\int
P^{xy}(d\omega)
\,\gamma(dx,dy).
$$

Reference law $R$ 也可以作同样分解。记 $R_{0T}$ 为 reference endpoint coupling，$R^{xy}$ 为 reference 在给定 $X_0=x,X_T=y$ 后的 conditional bridge，则

$$
R(d\omega)
=
\int
R^{xy}(d\omega)
\,R_{0T}(dx,dy).
$$

Föllmer 与 Gantert 在 1997 年论文 Entropy Minimization and Schrodinger Processes in Infinite Dimensions（补充材料暂未公开） 中给出无限维 entropy decomposition；Léonard 在 2014 年综述 [*A Survey of the Schrödinger Problem and Some of Its Connections with Optimal Transport*](https://doi.org/10.3934/dcds.2014.34.1533 "官方论文页面") 中以 endpoint disintegration 统一整理这一结构。

若 $P\ll R$，则 Radon–Nikodym derivative 可以分成 endpoint density ratio 与 conditional path density ratio。对一条满足 $X_0=x,X_T=y$ 的路径 $\omega$，形式上写成

$$
\frac{dP}{dR}(\omega)
=
\frac{d\gamma}{dR_{0T}}(x,y)
\frac{dP^{xy}}{dR^{xy}}(\omega).
$$

对上式取 logarithm，再在 $P$ 下求期望，就得到 path-space relative entropy 的 chain rule：

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

第一项是改变 endpoint coupling 的代价；第二项是在每一对端点内部改变 conditional path law 的平均代价。两项都非负，并且承担不同职责。

现在固定 endpoint coupling $\gamma$。第一项随之固定，第二项在

$$
P^{xy}=R^{xy},
\qquad
\gamma\text{-a.e.}
$$

时取到最小值 0。因此，在所有具有同一个 $\gamma$ 的 path laws 中，最优选择不是重新设计端点内部的轨迹，而是直接保留 reference conditional bridges。

于是 dynamic Schrödinger problem 精确化为 static endpoint problem：

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

如果右侧存在 minimizer $\gamma^*$，完整 path law 由

$$
\boxed{
P^*(d\omega)
=
\int
R^{xy}(d\omega)
\,\gamma^*(dx,dy)
}
$$

重建。这个等式对每个固定 reference 都是 finite-noise identity，不需要令噪声趋于零，也不是某种近似。

![Path KL 分解为 endpoint coupling 与 conditional bridge 两层](/images/bridge/B2_path_kl_projection.png)

这一步已经回答了一个重要问题：Schrödinger Bridge 的随机性并没有在优化中消失。即使最优 endpoint coupling 已经确定，样本仍然按照每个 $R^{xy}$ 在端点之间波动。

## 2. 1940—1974：endpoint KL 为什么产生 Schrödinger system

Chain rule 把问题降成

$$
\gamma^*
=
\arg\min_{\gamma\in\Pi(\mu_0,\mu_T)}
\operatorname{KL}(\gamma\Vert R_{0T}).
$$

现在先看有限状态且严格正的情形。设起点有 $m$ 个状态，终点有 $n$ 个状态。为避免与 path law $R$ 混淆，记

$$
Q=(Q_{ij})
$$

为 strictly positive reference endpoint probability matrix，即 $Q_{ij}>0$ 且所有元素之和为 1。目标 marginals 记为

$$
\mu_{0,i}>0,
\qquad
\mu_{T,j}>0.
$$

一个可行 coupling $\gamma=(\gamma_{ij})$ 必须满足

$$
\sum_j\gamma_{ij}=\mu_{0,i},
\qquad
\sum_i\gamma_{ij}=\mu_{T,j}.
$$

Static objective 是

$$
\operatorname{KL}(\gamma\Vert Q)
=
\sum_{i,j}
\gamma_{ij}
\log\frac{\gamma_{ij}}{Q_{ij}}.
$$

在这个 strictly positive setup 中，可行集合含有正 coupling，且 $x\log x$ 在 $x\downarrow0$ 时的右导数趋于 $-\infty$，因此 optimizer 落在可行 polytope 的 relative interior。这里可以安全使用普通 Lagrange multipliers；存在 structural zeros 时则不能沿用这一步。

对 row 与 column constraints 引入 Lagrange multipliers $\alpha_i,\beta_j$，再对 $\gamma_{ij}$ 求导，stationarity condition 为

$$
\log\frac{\gamma_{ij}}{Q_{ij}}
+1-\alpha_i-\beta_j=0.
$$

指数化后，所有最优 entries 必须具有乘法形式

$$
\boxed{
\gamma^*_{ij}
=
u_iQ_{ij}v_j,
}
$$

其中 $u_i>0$ 只依赖起点状态，$v_j>0$ 只依赖终点状态。把它代回两个 marginal constraints，可得

$$
\boxed{
u_i(Qv)_i=\mu_{0,i},
\qquad
v_j(Q^\top u)_j=\mu_{T,j}.
}
$$

这就是有限状态下的 Schrödinger system。它不是额外添加的模型假设，而是 endpoint KL projection 的 first-order optimality structure。

Factors 本身存在一个 gauge freedom。对任意常数 $c>0$，变换

$$
u\mapsto cu,
\qquad
v\mapsto c^{-1}v
$$

不会改变 $\gamma^*$。因此 factors 通常只在这一尺度变换下唯一，而 coupling $\gamma^*$ 本身在严格正有限情形下由 KL 的 strict convexity 唯一确定。

Schrödinger 在 1932 年论文 Sur la théorie relativiste de l'électron et l'interprétation de la mécanique quantique（补充材料暂未公开） 中已经写出粒子 migration factors 与 coupled integral equations，但没有证明一般存在唯一性。Fortet 在 1940 年论文 Résolution d'un système d'équations de M. Schrödinger（补充材料暂未公开） 中，在实数区间和特定 positivity、continuity、integrability 条件下建立了存在与 gauge uniqueness 路线。Jamison 在 1974 年论文 [*Reciprocal Processes*](https://doi.org/10.1007/BF00532864 "官方论文页面") 中，则在 continuous strictly positive transition-density 条件下给出相应的 product-coupling theorem。

这些结果不能压缩成“任意 kernel 都有正且唯一的 factors”。当 $Q_{ij}$ 存在 structural zeros，或者一般空间中的 support、integrability 与 dual attainment 失败时，factorization 和可行性都需要重新检查。

![Schrödinger system、左右 factors 与目标 marginals](/images/bridge/B3_schrodinger_system.png)

## 3. 1974—1997：endpoint factors 怎样变成双向动力学

Static coupling 的乘法结构可以直接提升到 path space。把有限状态 factors $u,v$ 写成一般状态空间上的 endpoint functions $f_0,g_T$，则最优 law 相对 reference 的 density 具有形式

$$
\boxed{
\frac{dP^*}{dR}
=
\frac{
f_0(X_0)g_T(X_T)
}{Z},
}
$$

其中 $Z>0$ 是 normalization constant。它可以被吸收到任意一侧 factor 中，因此与有限情形的 gauge freedom 相同。

为了得到中间时刻的结构，令 $r_t$ 表示 reference marginal density，并定义从两端传播的 potentials：

$$
g_t(x)
=
\mathbb E_R
\left[
g_T(X_T)
\mid
X_t=x
\right],
$$

$$
f_t(x)
=
\mathbb E_R
\left[
f_0(X_0)
\mid
X_t=x
\right].
$$

$g_t$ 将终端信息传播到较早时刻，$f_t$ 将初端信息传播到较晚时刻。两者共同给出最优过程的 marginal：

$$
\boxed{
p_t(x)
=
\frac{
r_t(x)f_t(x)g_t(x)
}{Z}.
}
$$

这就是文献中常见的前后向势函数乘积结构。一个 factor 从初端进入，另一个 factor 从末端进入；在任意中间时刻，两个边界信息通过乘积相遇。

若 reference 是 Markov process，最优 forward transition kernel 只需用 $g_t$ 作 Doob transform。对 $0\le s<t\le T$，若 $R_{s,t}(x,dy)$ 是 reference transition kernel，则

$$
\boxed{
P^*_{s,t}(x,dy)
=
R_{s,t}(x,dy)
\frac{g_t(y)}{g_s(x)}.
}
$$

分母 $g_s(x)$ 正是

$$
g_s(x)
=
\int
g_t(y)
R_{s,t}(x,dy),
$$

所以 transformed kernel 自动归一化。反向 kernel 则由 $f_t$ 作对称的 backward transform。前后两个方向重建的是同一个 $P^*$，不是两条彼此独立的生成过程。

在 smooth diffusion 情形，设 reference 满足

$$
dX_t
=
b_t(X_t)\,dt
+
\sigma_t(X_t)\,dW_t,
\qquad
a_t=\sigma_t\sigma_t^\top.
$$

$b_t$ 是 reference drift，$a_t$ 是 diffusion matrix。若 $g_t$ 足够光滑且为正，Doob transform 把 forward drift 改为

$$
\boxed{
b_t^{*,+}(x)
=
b_t(x)
+
a_t(x)\nabla\log g_t(x).
}
$$

类似地，采用反向时间参数时，reverse drift 相对 reference reverse drift 的修正由

$$
a_t(x)\nabla\log f_t(x)
$$

给出。于是两个 endpoint factors 不只控制 marginals，还直接决定最优过程的前向与反向 feedback。

Jamison 1974 的 reciprocal-process 框架说明：任意 reference bridges 的 endpoint mixture 都保留双侧条件结构，因此通常是 reciprocal process；但任意 endpoint mixture 未必 Markov。乘法 density

$$
f_0(X_0)g_T(X_T)
$$

正是使最优 mixture 在相应 positivity 条件下重新成为 Markov process 的关键结构。Föllmer–Gantert 1997 又把这一结构扩展到更一般的 path-space entropy setting。

![Reciprocal endpoint mixture 与 Markov factorization](/images/bridge/B4_reciprocal_markov.png)

## 4. 2004—2021：同一个解为何既是 Entropic OT，又是随机最优控制

到这里已经得到唯一的核心对象 $P^*$。Entropic OT 与随机控制不是在它之外再发明两个问题，而是从 endpoint 与 dynamics 两个方向重新解释同一个解。

先看 static interpretation。取 base coupling

$$
M=\mu_0\otimes\mu_T,
$$

并给定 transport cost $c(x,y)$ 与 regularization scale $\varepsilon>0$。定义 Gibbs endpoint reference

$$
R_{0T}^{\varepsilon}(dx,dy)
=
\frac1{Z_\varepsilon}
\exp\!\left(
-\frac{c(x,y)}{\varepsilon}
\right)
M(dx,dy),
$$

其中 $Z_\varepsilon$ 是 normalization constant。对任意 $\gamma\in\Pi(\mu_0,\mu_T)$，直接展开 relative entropy 得到

$$
\boxed{
\varepsilon
\operatorname{KL}
\left(
\gamma
\Vert
R_{0T}^{\varepsilon}
\right)
=
\int c(x,y)\,\gamma(dx,dy)
+
\varepsilon
\operatorname{KL}(\gamma\Vert M)
+
\varepsilon\log Z_\varepsilon.
}
$$

最后一项与 $\gamma$ 无关。因此，对 Gibbs-form endpoint reference 做 KL projection，与求解

$$
\boxed{
\min_{\gamma\in\Pi(\mu_0,\mu_T)}
\left\{
\int c\,d\gamma
+
\varepsilon
\operatorname{KL}
\left(
\gamma
\Vert
\mu_0\otimes\mu_T
\right)
\right\}
}
$$

具有相同 minimizer。这就是 Schrödinger problem 与 entropic optimal transport 的 static connection。

需要注意，Euclidean transport cost 不是任意 reference 自动附带的对象。只有当 $R_{0T}$ 能相对于选定 base measure 写成 Gibbs density 时，endpoint KL 才能改写为“cost 加 entropy regularization”。Reference structural zeros 对应 support constraints 或 infinite cost。

Mikami 在 2004 年论文 [*Monge's Problem with a Quadratic Cost by the Zero-Noise Limit of h-Path Processes*](https://doi.org/10.1007/s00440-004-0340-4 "官方论文页面") 中研究 Brownian $h$-path 的 zero-noise limit。Léonard 在 2012 年论文 [*From the Schrödinger Problem to the Monge–Kantorovich Problem*](https://doi.org/10.1016/j.jfa.2011.11.026 "官方论文页面") 中，以 large deviations 与 $\Gamma$-convergence 形式说明适当缩放的 Schrödinger problems 怎样趋向 classical OT。这个极限需要 tightness、coercivity 与 optimizer uniqueness 等条件，不能把 finite-$\varepsilon$ identity 简写成“Bridge 就是 OT”。

![Entropic couplings 随 regularization scale 的变化](/images/bridge/B5_entropic_couplings.png)

再看 dynamic interpretation。考虑 diffusion matrix $a_t=\sigma_t\sigma_t^\top$ 保持不变、只通过同一扩散通道用 $a_t\beta_t$ 修正 drift 的过程。Girsanov/Föllmer theory 在 absolute continuity 与 finite-energy 条件下给出

$$
\boxed{
\operatorname{KL}(P\Vert R)
=
\operatorname{KL}(P_0\Vert R_0)
+
\frac12
\mathbb E_P
\int_0^T
\beta_t^\top
a_t(X_t)
\beta_t
\,dt.
}
$$

若 reference initial law 已经等于 $\mu_0$，第一项为零。于是最小化 path-space KL 等价于：在所有能把 $\mu_0$ 驱动到 $\mu_T$ 的 controls 中，选择 quadratic energy 最小者。

对扩散强度为 $\varepsilon$ 的 Brownian reference，

$$
a_t=\varepsilon I.
$$

若用 physical drift correction $u_t=\varepsilon\beta_t$，则

$$
\operatorname{KL}(P\Vert R)
=
\frac1{2\varepsilon}
\mathbb E_P
\int_0^T
\|u_t\|^2\,dt.
$$

第三节已经得到最优 forward drift correction：

$$
\boxed{
u_t^*(x)
=
\varepsilon
\nabla\log g_t(x).
}
$$

因此 terminal Schrödinger potential $g_t$ 同时是 Doob transform 的 factor，也是 minimum-energy control 的价值函数梯度表示。

Lehec 在 2013 年论文 [*Representation Formula for the Entropy and Functional Inequalities*](https://doi.org/10.1214/11-AIHP464 "官方论文页面") 中严格讨论 Wiener-space entropy 与 canonical Föllmer drift。Chen、Georgiou 与 Pavon 在 2021 年综述 [*Stochastic Control Liaisons: Richard Sinkhorn Meets Gaspard Monge on a Schrödinger Bridge*](https://doi.org/10.1137/20M1339982 "官方论文页面") 中，则系统连接 Schrödinger system、stochastic control 与 optimal transport。

从 control PDE 看，Hopf–Cole transform $g_t=e^{\lambda_t}$ 会把 quadratic-control HJB 线性化为 reference backward Kolmogorov equation，因此同一个 potential 会同时出现在 factorization、Doob drift 与 optimal control 中。

## 5. 精确解成立需要什么，又不能推出什么

“精确解”意味着：只要相关 conditional laws、absolute continuity 与 optimizer 存在，path-KL decomposition、factorization 与 transformed drift 在固定 noise level 下就是恒等式，而不是 small-noise approximation。但它们仍有四条必须保留的边界。

第一，有限 strictly positive kernel 只是教学模型；连续空间中的 factorization 还需要 support compatibility、finite entropy、integrability 或 dual attainment 等条件。

第二，KL 与 control energy 的等式只适用于能通过同一 diffusion channel 改变测度的 laws；drift correction 若不在 $\operatorname{Range}(\sigma_t)$ 中，就不能机械套用 Girsanov 公式。

第三，Entropic OT interpretation 依赖 Gibbs-form endpoint reference。对任意 reference process，static problem 始终是

$$
\min_{\gamma\in\Pi(\mu_0,\mu_T)}
\operatorname{KL}(\gamma\Vert R_{0T}),
$$

却未必存在一个预先给定的 Euclidean cost $c(x,y)$。

第四，small-noise convergence 不等于 finite-noise identity。Optimizer values 收敛也不自动推出 couplings 与完整 path laws 收敛，后者还需要 compactness 和 limiting optimizer uniqueness。

现在可以回答本章标题：

> 一个精确 Schrödinger Bridge 先通过 KL chain rule 保留 reference conditional bridges，再通过 endpoint KL projection 产生 Schrödinger factors；这些 factors 沿 reference dynamics 传播，形成同一个 path law 的前向、反向、Entropic OT 与 minimum-energy control 表示。

## 本章论文索引

| 时间   | 论文                                                                                          | 本章中的作用                                                         |
| ---- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 1932 | Schrödinger, *Sur la théorie relativiste de l'électron...*                                  | 写出粒子 migration factors 与 coupled equations 的原始形式               |
| 1940 | Fortet, *Résolution d'un système d'équations de M. Schrödinger*                             | 在特定正则条件下建立 Schrödinger system 的存在与 gauge uniqueness            |
| 1974 | Jamison, *Reciprocal Processes*                                                             | 连接 product endpoint coupling、reciprocal structure 与 Markovity  |
| 1997 | Föllmer & Gantert, *Entropy Minimization and Schrodinger Processes in Infinite Dimensions*  | 给出 path entropy decomposition 与无限维 Schrödinger process 结构      |
| 2004 | Mikami, *Monge's Problem with a Quadratic Cost by the Zero-Noise Limit of h-Path Processes* | 建立 Brownian bridge 到 quadratic OT 的 zero-noise 路线              |
| 2012 | Léonard, *From the Schrödinger Problem to the Monge–Kantorovich Problem*                    | 以 large deviations 与 $\Gamma$-convergence 连接 SB 和 classical OT |
| 2013 | Lehec, *Representation Formula for the Entropy and Functional Inequalities*                 | 建立 Wiener entropy 与 canonical Föllmer drift 的能量表示              |
| 2014 | Léonard, *A Survey of the Schrödinger Problem...*                                           | 统一 dynamic/static reduction、factorization 与 OT 接口              |
| 2021 | Chen, Georgiou & Pavon, *Stochastic Control Liaisons...*                                    | 系统连接 Schrödinger system、stochastic control 与 transport         |
