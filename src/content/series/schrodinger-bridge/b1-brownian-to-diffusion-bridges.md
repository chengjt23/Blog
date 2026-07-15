---
title: 从 Brownian Bridge 到一般条件扩散
description: 区分点终点条件化与双边缘优化，从 Brownian bridge 推进到 Doob 变换和一般扩散桥。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 1
slug: b1-brownian-to-diffusion-bridges
tags:
  - schrodinger-bridge
  - brownian-motion
  - doob-transform
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦 point-conditioned bridge、Doob 变换与一般扩散条件化，不讨论双边缘熵优化。
---
## 1. 点终点条件不是双边缘优化

本章研究的问题是：给定一个 reference process 和一个点终点 `X_T=y`，条件化
之后中间路径怎样变化？这是 conditional bridge problem，不是 B2 开始的
two-marginal Schrödinger problem。

Brownian bridge 是最清楚的例子。它同时展示：

- Gaussian conditioning 如何改变 mean/covariance；
- pinned transition kernel 如何产生 time-inhomogeneous Markov law；
- Doob `h`-transform 如何把 endpoint information 写进 drift；
- 为什么 bridge 在每个 `t<T` 与 reference locally absolutely continuous，却在
  包含终点的 path sigma-field 上 singular。

这些结构推广到 diffusion 时需要 theorem assumptions，不能只把 Brownian 公式
中的 variance 换成 diffusion matrix。

## 2. Brownian bridge 的 Gaussian 构造

令 `W` 是一维 standard Brownian motion，固定 `X_0=x_0`、`X_T=y`。由于
`(W_t,W_T)` jointly Gaussian，

```text
W_t | W_T=y-x_0
 ~ Normal(t(y-x_0)/T, t(T-t)/T).                       (2.1)
```

整个 path 可在同一个 Brownian trajectory 上构造为

```text
X_t
 = x_0 + (t/T)(y-x_0) + W_t-(t/T)W_T.                 (2.2)
```

于是

```text
E[X_t]=x_0+(t/T)(y-x_0),                              (2.3)
Cov(X_s,X_t)=min(s,t)-st/T.                            (2.4)
```

(2.2) 是 finite-dimensional Gaussian laws 的一致 path construction，不是分别从
每个单时刻 conditional density 独立采样。独立采样会丢掉 (2.4) 的 temporal
covariance。

当 `t->T` 时，variance `t(T-t)/T->0`，而 mean趋于 `y`，所以
`X_t->y` in `L^2`；连续版本给 almost-sure endpoint pinning。

## 3. Pinned transition kernel

Brownian transition density 记为

```text
r(s,x;t,z)=Normal(z;x,t-s).                             (3.1)
```

由 Markov 性和 Bayes rule，对 `s<t<T`

```text
r^y(s,x;t,z)
 = r(s,x;t,z)r(t,z;T,y)/r(s,x;T,y).                    (3.2)
```

配方得到

```text
X_t | X_s=x, X_T=y
 ~ Normal(
    ((T-t)/(T-s))x+((t-s)/(T-s))y,
    (t-s)(T-t)/(T-s)).                                 (3.3)
```

这族 kernels 满足 Chapman--Kolmogorov，是 time-inhomogeneous Markov process。
FPY 1993
Proposition 1 在 dual Markov/positive transition-density setup 下严格构造一般
finite-time bridge、regular conditional law 和 non-homogeneous kernel。

## 4. Space-time Doob transform

定义 terminal likelihood

```text
h(t,x)=r(t,x;T,y).                                      (4.1)
```

Brownian `h` 满足 backward heat equation，且

```text
partial_x log h(t,x)=(y-x)/(T-t).                       (4.2)
```

对 generator `L=(1/2)partial_xx`，space-time `h`-transform 给

```text
L_t^h phi=L phi+(partial_x log h)partial_x phi.         (4.3)
```

因此 bridge SDE 是

```text
dX_t=(y-X_t)/(T-t)dt+dB_t,       t<T.                  (4.4)
```

drift 在终点奇异，但 solution 本身不爆炸；奇异 feedback 正是迫使 conditional
variance 收缩到零的机制。

[Doob 1957](https://doi.org/10.24033/bsmf.1494 "官方论文页面") 是
Green-space/Martin-boundary `h`-path 的 primary source。它提供 transform 背景，
但不能单独承担 fixed finite-time Brownian SDE；本章用 Gaussian/Doob 双路线和
现代 bridge theorem 交叉核验。

## 5. 从 SDE 反推 Gaussian law

令

```text
m_t=x_0+(t/T)(y-x_0),
Y_t=X_t-m_t.                                           (5.1)
```

则

```text
dY_t=-Y_t/(T-t)dt+dB_t.                                (5.2)
```

线性 integrating factor 给

```text
Y_t=(T-t) integral_0^t (T-u)^{-1}dB_u.                 (5.3)
```

对 `s<=t`，Itô isometry：

```text
Cov(Y_s,Y_t)
 =(T-s)(T-t) integral_0^s (T-u)^{-2}du
 =s-st/T.                                              (5.4)
```

与 Gaussian conditioning 完全一致。这说明 (4.4) 不是只匹配 one-time marginals
的另一个 process，而是同一个 Brownian bridge path law。

## 6. Local absolute continuity 与 terminal singularity

对每个 `t<T`，bridge restricted to `F_t` 满足

```text
dP^{x_0->y}/dR^{x_0}|_{F_t}
 = h(t,X_t)/h(0,x_0).                                  (6.1)
```

但在 `F_T` 上，bridge 给 `{X_T=y}` 概率一；unconditioned Brownian motion 对这个
atomless event 给概率零。因此完整 closed-interval laws singular。

两句话必须同时保留：

```text
locally AC for every t<T,
globally singular when the pinned endpoint is included.          (6.2)
```

不能把 finite-time RN derivative 直接令 `t=T`；其 martingale 极限恰可能集中到
reference-null event。

## 7. 一般 diffusion bridge 的 formal drift

令 reference generator 为

```text
A phi=(1/2)a:Hess phi+b dot grad phi,
a=sigma sigma^T.                                       (7.1)
```

若 time-homogeneous diffusion 的 remaining-time transition density `p_tau(x,z)`
足够正则，固定终点 `z`，定义

```text
h(t,x)=p_{T-t}(x,z).                                   (7.2)
```

formal Doob calculation 给

```text
A_t^h phi=A phi+(a grad log h) dot grad phi,            (7.3)
```

对应 SDE

```text
dX_t=[b(X_t)+a(X_t)grad_x log p_{T-t}(X_t,z)]dt
     +sigma(X_t)dB_t.                                  (7.4)
```

(7.4) 只有在 reference martingale problem、density、terminal concentration、
nonexplosion 和 transformed equation well posed 时才是 exact bridge。本节接下来
给出一组可引用的充分条件，而不把 formal calculation 当 theorem。

## 8. Çetin--Danilova exact bridge theorem

[Çetin--Danilova 2016](https://doi.org/10.1016/j.spa.2015.09.015 "官方论文页面")
给出 B1 所需的 exact SDE theorem。

### 8.1 Reference 条件

Assumption 2.1 要求：Borel/local-bounded coefficient 条件、`a` symmetric
nonnegative，以及每个 starting pair 的 local martingale problem well posed。
bounded Hölder coefficients和 uniformly positive definite `a` 是方便 sufficient
regime，不是 theorem 的最小假设。

### 8.2 Density 与 point conditioning

Assumption 2.2 要求 regular transition density、weak continuity、Chapman--
Kolmogorov、near-terminal concentration (2.6) 和 off-target boundedness (2.8)。
Theorem 2.2 还要求 `p(T,x,z)>0`、`m({z})=0`、`h` 为 `C^{1,2}`，以及对应
interior/positivity 条件。

在这些条件下，Theorem 2.2 证明：

- (7.4) 存在 weak solution；
- `X_T=z` almost surely；
- `h>0` 时 weak uniqueness；
- bridge kernel 为

```text
p^z_{s,t}(x,dy)
 = p(t-s,x,y)p(T-t,y,z)/p(T-s,x,z) m(dy).               (8.1)
```

proof p. 20 给出 (6.1) 的 general local RN formula。

### 8.3 Strong realization

Theorem 4.1 在 interior closed subsets 上另加 `b,sigma` locally Lipschitz、density
positivity 和 reference 不离开 interior 等条件，才升级为 unique strong solution。

因此要区分：

```text
conditional path law / weak bridge
    != automatically strong pathwise construction.               (8.2)
```

## 9. 一般终端奇异性不等于统一爆炸率

Çetin--Danilova theorem 证明 continuous bridge reaches `z`。在 `m({z})=0` 时，
closed-interval singularity 也立即成立。

但它不证明任意 diffusion 都有 Brownian 式 asymptotic

```text
a grad log p(T-t,x,z) ~ (z-x)/(T-t).                    (9.1)
```

一般 short-time expansion 依赖 ellipticity、geometry、heat-kernel asymptotics 和
cut-locus 等附加条件。B1 正文只对 Brownian bridge 写 explicit `1/(T-t)`；对一般
diffusion 只陈述 exact score drift与 terminal convergence。

## 10. Exact bridge 与 guided proposal

exact diffusion bridge 需要 inaccessible target score

```text
grad log p(T-t,x,z).                                    (10.1)
```

这正是 simulation difficulty。guided proposal 用 tractable auxiliary transition
density `ptilde`：

```text
dX_t^circ
 =[b(t,X_t^circ)+a(t,X_t^circ)
   grad log ptilde(T-t,X_t^circ,z)]dt
  +sigma(t,X_t^circ)dW_t.                              (10.2)
```

[Schauer et al. 2017](https://doi.org/10.3150/16-BEJ833 "官方论文页面")
给出 finite-time RN relation、full-interval equivalence assumptions 和 likelihood
correction。通常

```text
guided proposal != exact bridge.                       (10.3)
```

命中同一个 endpoint 不足以保证 path laws 相同。terminal covariance mismatch
甚至可使 full-interval laws singular；exact expectation需要 importance/MH/RN
correction及相应 assumptions。

## 11. 与 Schrödinger Bridge 的边界

point bridge 固定一个 endpoint pair `(x,z)`；SB 固定 endpoint distributions 并
优化 coupling/path law。

B2 的 exact SB 可写成

```text
P* = integral R^{xz} gamma*(dxdz).                     (11.1)
```

B1 提供每个 `R^{xz}` 的 conditional intuition/theorem。B3 决定 optimal
`gamma*`。所以

```text
diffusion bridge = conditional building block,
Schrödinger Bridge = entropy-optimized mixture.         (11.2)
```

名字中都含 bridge，不代表问题相同。

## 12. 数值验证与图示

`brownian_bridge_checks.py` 使用 exact Gaussian construction (2.2)，不通过 Euler
离散奇异 SDE。200,000 paths 的结果：

- endpoint error `2.22e-16`；
- max mean error `2.56e-3`；
- max covariance error `3.12e-3`；
- conditional transition mean error `5.46e-4`；
- transition variance error `2.94e-5`。

下图由同一解析 sampler 生成，比较 unconditioned Brownian paths、pinned paths、
mean/covariance band 和终端 variance collapse。

![Brownian motion 与 Brownian bridge 的路径、均值和方差对比](/images/bridge/B1_brownian_bridge.png)

**图 12.1：** Brownian bridge 的解析采样结果。所有 pinned paths 在终点汇合；
经验均值与协方差带同时展示条件过程的线性均值和终端方差塌缩。图为固定 seed 的
原创计算结果，生成脚本见 `references/code/bridge/generate_classic_figures.py`。

## 13. 常见错误

1. **把各时刻 conditional Gaussian 独立采样。** 会破坏 path covariance。
2. **把 Doob 1957 直接当 finite-time SDE theorem。** 它的主线是 Green/Martin
   `h`-paths。
3. **将 formal `a grad log p` 写成无条件 exact bridge。** 必须列 density、
   terminal concentration 和 well-posedness。
4. **把 local AC 延伸到 `F_T`。** pinned endpoint 通常造成 singularity。
5. **声称一般 score 都按 `1/(T-t)` 爆炸。** Brownian explicit rate 不自动推广。
6. **把 guided proposal 称 exact bridge。** auxiliary score需要 likelihood correction。
7. **把 point conditioning 当 two-marginal SB。** 后者还要优化 endpoint coupling。

## 14. 小结

Brownian bridge 把条件过程的四层结构串起来：

```text
Gaussian conditioning
 -> pinned transition kernel
 -> Doob terminal likelihood
 -> singular endpoint drift and exact path law.         (14.1)
```

一般 diffusion 保留同一结构，但 theorem 需要 martingale problem、transition
density、regularity、terminal concentration 与 weak/strong well-posedness 条件。
simulation 方法可以用 guided law 逼近 exact bridge，却不能删除 RN correction
和 full-interval equivalence 的假设。

## 15. 研究式思考题

1. local AC density (6.1) 在 `t->T` 时为何不要求有 `L^1` limit density？
2. 对 hypoelliptic diffusion，transition density正但 `a` degenerate 时，bridge drift
   的可控方向是什么？
3. strong solution 条件比 weak bridge law 多了什么？能否用 Yamada--Watanabe
   路线替代 Theorem 4.1 的局部条件？
4. guided proposal 命中 endpoint且 finite-time equivalent，为何 full path laws
   仍可能 singular？
5. 将 point bridge mixtures 用任意 coupling 混合后，何时 mixture 仍 Markov？

## 16. 前后章节链接

- B0：point endpoint 与 particle distribution constraints 的区别；
- B2：reference conditional bridges 与 path-space KL；
- B3：optimal endpoint coupling/factors；
- B4：arbitrary bridge mixture、reciprocal/Markov 与 two-sided transform；
- B8--B10：bridge simulation、regression 和 matching 的现代近似。

独立 Brownian 推导见
`references/notes/derivations/bridge/brownian_bridge_derivations.md`。
