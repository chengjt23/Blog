---
title: Schrödinger 的问题与全局时间线
description: 从 1931 年的稀有涨落问题出发，用有限粒子模型解释 Schrödinger 问题的统计动机与理论时间线。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 0
slug: b0-schrodinger-problem-timeline
tags:
  - schrodinger-bridge
  - history
  - large-deviations
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 覆盖原始问题、有限迁移模型、prepared-initial 大偏差解释及全系列阅读地图。
---
## 1. 问题不是“怎样从噪声生成数据”

Schrödinger 的出发点是一群按已知 reference dynamics 独立运动的粒子。我们已经
观察到它们的初始分布，却又在较晚时刻观察到一个在 reference 下罕见的终端分布。
问题是：

> 在已经知道初末两端群体分布的条件下，哪一种中间演化最可能产生这次罕见涨落？

这个问题同时包含两类对象：

- reference process：如果没有终端条件，粒子本来怎样运动；
- conditional inference：在罕见终端观测已经发生后，怎样重估路径 law。

现代生成建模可以使用相同数学工具，但不是 1931 问题的历史定义。

## 2. 点终点与分布终点是两个问题

[1931 英译文本](https://doi.org/10.1140/epjh/s13129-021-00032-7 "官方论文页面")
明确先讨论单粒子的固定 endpoints，再讨论大量粒子的 endpoint distributions。

固定 `X_0=x,X_T=y` 时，Brownian intermediate density 是

$$
\mathbb{P}(X_t=z\mid X_0=x,X_T=y)=\frac{r_{0,t}(x,z)\,r_{t,T}(z,y)}{r_{0,T}(x,y)}. \tag{2.1}
$$

这是 point-conditioned reference bridge，B1 研究它的一般版本。

分布终点问题只给

$$
P_0=\mu_0,\qquad P_T=\mu_T. \tag{2.2}
$$

它没有给出 `(X_0,X_T)` 的 coupling。许多 endpoint couplings 都具有相同
marginals；Schrödinger problem 还要根据 reference likelihood 在它们之间选择。

## 3. 原始 migration table

把初始与终端空间各分成有限 cells。令

```text
a_i = initial particle count in cell i,
b_j = terminal particle count in cell j,
c_ij = particles moving i -> j,
K_ij = reference transition probability.              (3.1)
```

constraints 为

```text
sum_j c_ij=a_i,
sum_i c_ij=b_j.                                        (3.2)
```

给定初始 counts，每张 migration table 的 reference probability 是 row-wise
multinomial weights 的乘积：

```text
Pr(C=c|a)
 = product_i [a_i!/product_j c_ij!]
   product_ij K_ij^(c_ij).                             (3.3)
```

再条件于 terminal counts `b`，最可能的 table 是 (3.3) 在 constraints (3.2)
上的 mode。

## 4. Stirling 公式产生 endpoint KL

写

```text
gamma_ij=c_ij/N,
mu_i=a_i/N,
R_ij=mu_i K_ij.                                       (4.1)
```

对 factorial 使用 Stirling approximation，并利用固定 row sums，得到

```text
(1/N)log Pr(C=c|a)
 = -H(gamma|R)+o(1),                                  (4.2)

H(gamma|R)=sum_ij gamma_ij log(gamma_ij/R_ij).        (4.3)
```

所以大粒子数下的 endpoint mode 解

```text
minimize H(gamma|R)
subject to gamma in Pi(mu_0,mu_T).                    (4.4)
```

这是原始组合论论证的现代重建。1931/1932 原文没有使用 KL 或 path-space entropy
术语，不能把 (4.4) 的现代 notation 反向归给 Schrödinger。

## 5. Multiplicative table 是一阶条件，不是一般 existence theorem

对 positive finite table，(4.4) 的 Lagrange equations 给出

```text
gamma*_ij=u_i R_ij v_j.                               (5.1)
```

这对应原文中的

```text
c_ij=psi_i g_ij phi_j.                                (5.2)
```

Schrödinger 的文本明确把 factor existence/uniqueness 留作未证明输入。B3 才分别
用 finite proof、Fortet 和 Jamison 处理 positivity、gauge 与 support boundary；
B7 再讨论怎样迭代计算 factors。

## 6. 1931 与 1932 文本怎样关联

Schrödinger 1932
Section VII（printed pp. 296--306）在一篇更广的 lecture paper 中重述相关问题。
它包含 fixed endpoint formula、migration likelihood、multiplicative factors 和
forward/backward density product。

两份文本的数学链一致，但 1932 版本经过压缩、重排和增删，不是 1931 德文论文
的逐句法译。1931 原印本的书目信息已由 DNB/lobid 核验；内容引用使用 2021
同行评审英译，并明确引用翻译者与 CC BY 版本。

## 7. 从 endpoint table 到 path law

finite table 只决定 `(X_0,X_T)`。要回答“中间怎样走”，还需要 reference conditional
bridges `R^{xy}`。给定 endpoint coupling `gamma`，最自然的 lift 是

```text
P_gamma = integral R^{xy} gamma(dxdy).                 (7.1)
```

B2 的 entropy chain rule 证明：固定 `gamma` 后，任何偏离 reference conditional
bridges 的改动都会增加 path KL。因此 dynamic optimizer 保留 `R^{xy}`，只优化
endpoint coupling。

这一步把原始 migration intuition 升级为真正的 path-law statement；它不是仅由
one-time density product 推出的。

## 8. Iid Sanov 只是第一层模板

若 `N` 条 paths iid 服从同一个 `R`，empirical path law

```text
L_N=(1/N)sum_i delta_{X^i}                             (8.1)
```

在 Sanov 条件下以 `H(.|R)` 为 rate function。这解释了 relative entropy 为什么
度量某个 empirical evolution 的指数稀有程度。

但 Schrödinger 的 prepared initial population 通常把每个粒子的初始位置预先固定，
paths 独立却不 identically distributed。普通 iid Sanov 不能直接承担该步骤。

## 9. Fixed-initial profile 的直接 LDP

[Luçon 2017](https://doi.org/10.1007/s10955-017-1719-9 "官方论文页面")
Proposition 2.2 给出需要的 deterministic-profile theorem。设固定序列 `x_i` 满足

```text
(1/N)sum_{i=1}^N delta_{x_i} -> mu_0,                 (9.1)
```

path space 为 continuous Polish space，且

```text
x -> R(.|X_0=x)                                       (9.2)
```

是 Feller kernel。则 empirical path law 满足 good LDP，rate 为

```text
I_mu0(P)
 = H(P|mu_0 R)+iota_{P_0=mu_0}.                       (9.3)
```

这里 `mu_0 R` 表示先按 `mu_0` 选 prepared initial state，再使用 reference
conditional dynamics。对 `P_0=mu_0`，

```text
H(P|R)=H(mu_0|R_0)+H(P|mu_0 R),                       (9.4)
```

第一项在 constraint 中为常数，所以 (9.3) 与双端 path-KL minimization 选择同一
optimizer。

该结论不覆盖任意 non-Feller disintegration、非嵌套 triangular array 或一般
Skorokhod fixed-time topology。

## 10. 从 LDP 到“most likely evolution”

有 rate function 仍不够。还要把终端 constraint 写成 shrinking neighborhoods
`C_delta`，并检查 endpoint map 连续、conditioning event 正概率和 regularity。

[Léonard 2010](https://doi.org/10.1051/ps/2009003 "官方论文页面")
Theorem 7.1 在这些条件下证明：先令 `N->infinity`，再令 `delta->0`，conditional
empirical laws 以指数速度集中到 constrained rate minimizers。若 minimizer 唯一，
conditional law 才集中到该单点。

B0 的 finite positive two-state model 取 total-variation balls，并选择 interior
rational target。对足够大兼容 `N`，至少存在一张整数 migration table；`K_ij>0`
使该 event 有正概率。一般连续模型仍须单独验证 Léonard assumptions。

## 11. 原创 finite-migration 图

![Reference migration、conditional table likelihood 与 KL limit](/images/bridge/B0_migration_likelihood.png)

**图 11.1：** 左两图比较 reference endpoint law 与满足 observed terminal marginal
的 KL-optimal coupling；中图在 `N=80` 精确枚举所有 feasible `2x2` tables，并
与 normalized `exp(-N H(gamma|R))` 比较；右图显示 exact mode/conditional mean
向 continuous KL projection 靠近。图由 `b0_migration_figure.py` 基于 exact
factorials 生成，无 Monte Carlo 或第三方素材。

该图验证 finite combinatorics 和 asymptotic shape，不替代 Luçon/Leonard 的
Polish path-space theorem。

## 12. 五阶段路线，而非一条优先权口号

本教程把后续发展组织为五个可核验阶段：

1. **问题与组合结构：** 1931/1932 fixed endpoints、migration table 与 factors；
2. **经典概率结构：** Fortet 的 stated positive-kernel existence regimes，Jamison 的 reciprocal/Markov framework，Föllmer entropy route；
3. **OT 与控制：** path KL、entropic OT、small-noise limit、Girsanov/HJB；
4. **经典计算：** IPF/Sinkhorn、support conditions 与 stable arithmetic；
5. **现代近似：** neural IPF、bridge matching、simulation-free/generalized methods。

这是一张阅读路线，不是 \`\`谁最先完成整个领域'' 的 claim。具体 theorem、版本和
边界分别由 B1--B14 承担。

## 13. 三个不要混用的 bridge

```text
Brownian/diffusion bridge:
  reference conditioned on point endpoints;

Schrodinger Bridge:
  path-law entropy projection under endpoint marginals;

bridge matching:
  a learned regression/projection method whose exact target must be audited. (13.1)
```

名称相似不等于优化对象、constraint 或 guarantee 相同。

## 14. 常见错误

1. **把 fixed endpoint 当 distribution endpoint。** 后者还要选择 coupling。
2. **把原文写成 path-KL theorem。** KL/LDP 是后来严格化。
3. **用 finite Stirling 计算代替 path-space LDP。** 证明层次不同。
4. **把 iid Sanov 套到 prepared initial sequence。** fixed-initial theorem 需要额外输入。
5. **条件在 exact continuous marginal event 上。** 应使用 regular shrinking sets 和正确极限次序。
6. **说原文证明了 factor existence/uniqueness。** 两份文本都留下了缺口。
7. **用 \`\`first/complete'' 概括历史。** 强优先权需要独立 primary evidence。

## 15. 小结与全篇入口

B0 的严格链条是

```text
finite migration likelihood
 -> endpoint KL by Stirling
 -> fixed-initial path empirical LDP
 -> regular conditional concentration
 -> path-law entropy projection.                      (15.1)
```

接下来：B1 研究 point bridges；B2 定义 path-space problem；B3/B4 给 factors、
reciprocal/Markov dynamics；B5/B6 建立 OT/control connections；B7 给 exact
computation；B8--B10 审计 neural approximations；B11--B14 处理扩展、比较、应用
与开放问题。

完整 LDP 责任表见 `references/notes/bridge/b0_ldp_theorem_responsibility.md`，历史
claim 状态见 `references/notes/bridge/b0_history_ledger.md`。

## 16. 研究式思考题

1. 在 two-state migration model 中固定 initial counts，推导 feasible table 的自由度，并说明为什么 terminal marginal constraint 会选择一个 endpoint coupling 而不是一条 point bridge。
2. finite Stirling rate 与 path empirical LDP 分别控制什么随机对象？列出从前者到后者仍缺的拓扑、指数紧性或条件化责任。
3. 为什么连续状态下不能直接条件于一个通常零概率的 exact empirical marginal event？给出 regular shrinking sets 的替代构造。
4. 比较 Schrödinger 1931/1932 原始文本、后来的 path-KL 表述与现代 neural bridge matching：哪些是历史归属，哪些是后来的数学重构？
5. 若 reference endpoint law 有 structural zeros，finite KL projection、factorization 与条件大偏差叙事分别需要怎样修改？
