---
title: 路径空间上的 Schrödinger Problem
description: 在路径测度上定义动态 Schrödinger 问题，并用相对熵分解说明端点耦合与条件桥的职责。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 2
slug: b2-path-space-schrodinger-problem
tags:
  - schrodinger-bridge
  - path-space
  - relative-entropy
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 覆盖 path-law 优化、KL disintegration、可行性和动态到静态的约化。
---
## 1. 为什么优化对象必须是 path law

给定初末分布 `mu_0,mu_T`，只知道每个粒子从哪里出发、最终在哪里出现，并不
知道中间怎样运动。即使进一步给出所有 one-time marginals `(mu_t)`，也通常不能
唯一确定多时刻相关性或整条随机路径。

Schrödinger problem 因此不直接选择一个 drift 或一条 deterministic map，而是在
path laws 中选择：

$$
\min_{P}\ H(P\mid R)\quad\text{subject to}\quad P_0=\mu_0,\;P_T=\mu_T. \tag{1.1}
$$

`R` 是 reference path measure，编码未施加端点信息时的动力学。`P` 是候选
path law。目标方向固定为 `H(P|R)`：我们问“为了满足观测，必须把 reference
改动多少”，不是计算反向 KL。

这个表述同时包含三类信息：

1. endpoint marginals 是观测约束；
2. reference conditional paths 是动力学先验；
3. relative entropy 是在两者之间选择最小改动的准则。

## 2. Measurable setup 与相对熵

令 `Omega` 为 Polish path space，状态空间 `X` 也是 Polish。endpoint map 为

```text
Phi: Omega -> X x X,
Phi(omega)=(X_0(omega),X_T(omega)).                    (2.1)
```

对连续路径和 uniform topology，`Phi` 连续；在一般 Skorokhod space 中，固定时刻
evaluation 不自动处处连续，涉及 LDP 时必须另查 continuity set。

相对熵定义为

```text
H(P|R) = integral log(dP/dR) dP,      if P << R,
       = +infinity,                   otherwise.       (2.2)
```

它是 extended-value functional。`H(P|R)=+infinity` 表示 candidate 使用了
reference 认为不可能的路径事件，或需要无限信息改动。教程中的 chain rule 不做
`infinity-infinity` 运算。

endpoint constraints 用 pushforward 表示：

```text
P_0=(X_0)_#P=mu_0,
P_T=(X_T)_#P=mu_T.                                    (2.3)
```

这与固定 joint endpoint coupling 不同。所有满足 (2.3) 的 `P` 可具有不同

```text
gamma=P_0T=Phi_#P in Pi(mu_0,mu_T).                   (2.4)
```

## 3. Regular conditional bridges

Polish spaces 是 standard Borel，因此 `P` 和 `R` 可相对于 `Phi` disintegrate：

```text
P(domega)=integral P^z(domega) gamma(dz),
R(domega)=integral R^z(domega) R_0T(dz),               (3.1)
```

其中 `z=(x,y)`，`R^z` 是 reference conditional bridge。一般情形下这些版本只需
在 endpoint law 几乎处处定义。B1 的 positive transition-density theorem 可以为
更多 endpoint pairs 选择显式 Markov kernels，但 B2 的 KL decomposition 不依赖
SDE drift 公式。

需要区分：

- `R^{xy}`：给定两个 endpoint 的 reference conditional path law；
- `gamma`：endpoint pairs 在总体中的 joint distribution；
- `P`：把各 conditional bridges 按 `gamma` 混合后的 path law。

更换 `gamma` 不改变每个 `R^{xy}`，却会改变总体的 endpoint correlation，甚至
改变 mixture 是否 Markov。这个区别在 B4 继续展开。

## 4. 路径 KL 的 chain rule

设 `P<<R`。令 `gamma=P_0T`。相对熵对 endpoint map 的 chain rule 是

$$
H(P\mid R)=H(\gamma\mid R_{0T})+\int H(P^{xy}\mid R^{xy})\,\gamma(dx\,dy). \tag{4.1}
$$

[Léonard 2014](https://doi.org/10.3934/dcds.2014.34.1533 "官方论文页面")
Proposition 2.3/Lemma 2.4 与
Föllmer--Gantert 1997
equation (3.41) 给出正式来源；有限完整证明见
`references/notes/derivations/bridge/finite_path_kl_derivations.md`。

式 (4.1) 将信息改动拆成：

1. 改变 endpoint coupling 的代价 `H(gamma|R_0T)`；
2. 在每个 endpoint pair 内改变 conditional paths 的代价。

第二项非负。因此固定 `gamma` 后，最优 choice 是

```text
P^{xy}=R^{xy},       gamma-a.e.                        (4.2)
```

这不是“bridge 看起来合理”的直觉，而是 KL chain rule 的精确结论。

## 5. Dynamic problem 精确降到 static problem

对任意 `gamma<<R_0T`，定义 reference-bridge lift

```text
P_gamma = integral R^{xy} gamma(dxdy).                 (5.1)
```

它满足 `P_gamma,0T=gamma`，且

```text
H(P_gamma|R)=H(gamma|R_0T).                            (5.2)
```

于是

```text
inf_{P_0=mu_0,P_T=mu_T} H(P|R)
 = inf_{gamma in Pi(mu_0,mu_T)} H(gamma|R_0T).          (5.3)
```

左侧称 dynamic Schrödinger problem，右侧称 static Schrödinger problem。这里
`dynamic'' 和 `static'' 指优化变量层级，不是说 endpoint coupling 没有动力学
含义。

若 static minimizer `gamma*` 存在，dynamic optimizer 是

```text
P* = integral R^{xy} gamma*(dxdy).                     (5.4)
```

反过来，任何 dynamic optimizer 的 endpoint pushforward 都是 static optimizer，
并且 conditional paths 必须等于 reference bridges。

式 (5.3) 允许两边同时为 `+infinity`。B2 不声称 optimizer 总存在；support、
finite-entropy feasibility、lower semicontinuity、tightness 和 factor attainment
属于 B3。

## 6. Half-bridge 不是完整 SB

若只要求 terminal marginal `P_T=nu`，且 `nu<<R_T`，I-projection 是

```text
dP*/dR = dnu/dR_T(X_T),
H(P*|R)=H(nu|R_T).                                     (6.1)
```

它保留 `R(.|X_T)`，但 initial marginal 一般由 optimization 自己决定。我们称它为
terminal half-bridge。

[Lehec 2013](https://doi.org/10.1214/11-AIHP464 "官方论文页面") Lemma 10
在 Wiener space 给出该结论，并在更强条件下得到 heat-semigroup drift。该文把
`rho(w_1)gamma(dw)` 称为 Brownian bridge；本教程为避免与 fixed-point Brownian
bridge 或 two-marginal SB 混淆，坚持使用 \`\`terminal-law half-bridge''。

IPF 的一次 marginal correction 常是 half-bridge projection。一次 projection
满足一个边缘，不等于已经解出双端 fixed point。

## 7. 为什么 relative entropy 表示“最可能演化”

相对熵的统计解释必须分层。

### 7.1 iid Sanov template

若 `N` 条 reference paths iid 服从 `R`，经验路径测度

```text
L_N=(1/N) sum_i delta_{X^i}                              (7.1)
```

在 Sanov theorem 条件下满足 rate `H(.|R)` 的大偏差原理。于是满足某个 rare
empirical constraint 的 path laws，以相对熵最小者拥有最低指数代价。

### 7.2 Gibbs conditioning

[Léonard 2010](https://doi.org/10.1051/ps/2009003 "官方论文页面")
Theorem 7.1 假设：input empirical-measure LDP 是 good；constraint map 连续；
closed neighborhoods `C_delta` regular、正概率并缩向目标集合。则先
`N->infinity`、再 `delta->0` 时，条件概率以指数速度集中到 constrained rate
minimizers 附近。严格凸且解唯一时，条件 law 才收敛到该点的 Dirac measure。

### 7.3 fixed-initial boundary

Schrödinger 原始实验把初始粒子位置准备成 deterministic triangular array，通常
不是 iid paths。对应 modified Sanov theorem 由 DG87/CL95 承担；其全文假设仍在
B0 核验中。

因此 B2 的安全表述是：

> iid Sanov 与 Gibbs conditioning 严格解释了为何 constrained entropy projection
> 是“最可能 law”的标准模板；prepared initial profile 需要额外 modified Sanov
> input，不能由有限 migration 或 iid theorem 自动推出。

完整责任表见
`references/notes/bridge/b0_ldp_theorem_responsibility.md`。

## 8. Reference process 不是无关噪声

`R` 决定三件事：

1. 哪些 paths 有正概率，即 feasible support；
2. endpoint prior `R_0T`，从而决定 static entropy cost；
3. 每个 `R^{xy}` 的中间 fluctuation 和 geometry。

两个 references 即使具有相同 endpoint marginals，也可能给出不同 `R_0T`、
不同 conditional bridges 和不同 optimizer。换 reference 不是给同一个问题换
sampler，而是更换模型假设。

只有在 B5 的 Gibbs/heat-kernel 条件下，`H(gamma|R_0T)` 才能改写成某个显式
transport cost 加 entropy regularization。一般 B2 不预设 Euclidean cost。

## 9. 有限路径空间验证

`finite_path_kl_checks.py` 枚举一个三状态、三时刻 reference 的全部 27 条路径。
验证结果为：

- total normalization error `1.11e-16`；
- endpoint KL `0.014622704860669756`；
- reference-bridge lift path KL `0.014622704860669758`；
- dynamic/static reduction error `1.73e-18`；
- 完整 chain-rule error `7.63e-17`；
- perturb conditional bridges 后增加 KL `5.12e-2`；
- terminal half-bridge marginal error `0`。

这段代码证明有限例子的每个恒等式，并检查 KL direction。它不证明 general
Polish disintegration theorem，但能及时发现公式中漏掉 conditional entropy 或
把 endpoint marginals 误当 coupling 的错误。

![路径 KL 投影分解为 endpoint coupling 与 conditional bridge 两层](/images/bridge/B2_path_kl_projection.png)

**图 9.1：** 有限路径空间中的 KL 投影。固定 endpoint coupling 后，保留 reference
conditional bridges 恰好消去条件 KL；任何 bridge 内部扰动都会增加总路径 KL。
图由同一 finite fixture 生成。

## 10. 常见错误

1. **只固定 `P_0,P_T` 却把 `P_0T` 当成已知。** marginals 不决定 coupling。
2. **把 `R^{xy}` 与 SB 混为一谈。** 前者是 point-conditioned reference；后者还
   要优化 endpoint coupling。
3. **由“保留 reference bridges”推出 Markov。** 任意 endpoint mixture 至少可为
   reciprocal，但只有额外 factorization 才是 Markov。
4. **忽略 absolute continuity。** 若 candidate 使用 reference support 外路径，
   entropy 是 infinity。
5. **把 half-bridge 当 full bridge。** 一次 I-projection 只满足一个 constraint。
6. **把有限 mode 或 iid Sanov 当 prepared-initial theorem。** 它们是不同证明层。
7. **从 one-time marginals 推断 path law。** temporal correlation 仍未确定。

## 11. 小结

B2 的核心链条是

```text
path law P
 -> endpoint coupling gamma + conditional paths P^{xy}
 -> KL chain rule
 -> preserve reference bridges
 -> optimize only H(gamma|R_0T).                       (11.1)
```

这个 reduction 同时解释了 SB 的路径性质和计算结构：B3 只需研究 endpoint
factorization，B4 再研究何时 mixture 是 Markov，B7 才设计 alternating
projections。reference dynamics 从未消失；它通过 `R_0T` 和 `R^{xy}` 保留在解中。

## 12. 研究式思考题

1. 若 `R_0T` 的 support 不是 rectangular，哪些 endpoint marginals 根本不可行？
2. 当 static infimum finite 但不 attained 时，dynamic problem 可以出现怎样的
   minimizing sequence？
3. 仅观察一族 `(P_t)`，是否可以恢复 endpoint coupling 或 conditional bridges？
4. 对 jump paths，选择哪种 topology 可使 endpoint evaluation 适合 LDP contraction？
5. 若将 KL 换成 reverse KL 或其他 `f`-divergence，(4.1) 的 conditional lift 和
   static reduction还保留多少？

## 13. 前后章节链接

- B0：原始粒子问题、finite migration 与 fixed-initial LDP 缺口；
- B1：point-conditioned Brownian/Markov/diffusion bridges；
- B3：static optimizer、Schrödinger factors、existence/uniqueness；
- B4：reciprocal mixture、Markov factorization 与 two-sided kernels；
- B5：Gibbs endpoint reference、entropic OT 与 small-noise limit；
- B7：half-bridge alternating projections 与 Sinkhorn/IPF。

一般条件和证明责任见
`references/notes/derivations/bridge/path_space_disintegration_conditions.md`。
