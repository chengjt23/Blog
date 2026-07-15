---
title: Schrödinger Bridge、Diffusion 与 Flow Matching 的关系
description: >-
  以 coupling、path law、marginal path 和回归场四层接口比较 Diffusion、Flow Matching 与
  Schrödinger Bridge。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 12
slug: b12-diffusion-flow-matching-unification
tags:
  - schrodinger-bridge
  - diffusion
  - flow-matching
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 给出可共享的条件期望与场表示，同时保留 reference、coupling、目标函数和路径律的差异。
---
Score diffusion、Flow Matching、stochastic interpolants 与现代 Schrödinger Bridge
算法看起来都在做同一件事：从一个分布连续地走到另一个分布，并用平方损失回归
某个 time-dependent field。

这个相似性是真的，但只到一定层级为止。它们可以共享 conditional expectation、
continuity equation、score 或 probability-flow 公式，却仍然选择不同的 coupling、
不同的 path object 与不同的 objective。本章的核心判断标准是：

```text
same regression formula
    != same marginal curve
    != same endpoint coupling
    != same path law
    != same variational problem.                              (0.1)
```

## 1. 先问五个对象问题

看到一个“连接两个分布”的方法，先不要从网络结构判断它属于哪一类，而应依次问：

1. 输入是一端 data law，还是两个 hard endpoint marginals？
2. endpoint coupling 是给定、任意采样、学习得到，还是由 objective 选择？
3. 被定义的是 marginal density curve、ODE flow、SDE law，还是 reference 的 bridge mixture？
4. loss 在估计 score/velocity，还是在最小化 path-space relative entropy？
5. 结论属于 theorem identity、exact operator、population regression、有限训练还是离散 sampler？

这五问已经足以拆开大多数表面上的“统一”。

## 2. Score diffusion：先固定 forward noising law

Diffusion D3 已建立 population 接口。给定 corruption kernel，随机 conditional target
的条件期望是 noisy marginal score：

$$
s_t(x)=\nabla\log p_t(x)=\mathbb{E}\!\left[\nabla\log p_t(x\mid X_0)\mid X_t=x\right]. \tag{2.1}
$$

平方损失在 population optimum 恢复 (2.1)，不表示有限网络、有限样本与 optimizer
已经恢复 exact score。

Diffusion D4 再固定 forward SDE

```text
dX_t=f_t(X_t)dt+g_t dW_t.                                    (2.2)
```

在 scalar state-independent diffusion 与相应 time-reversal 条件下，exact marginal
score 决定 reverse SDE；同一 score 还给 probability-flow velocity

$$
v_t(x)=f_t(x)-\frac{g_t^2}{2}\,s_t(x). \tag{2.3}
$$

ODE `dot X_t=v_t(X_t)` 与 forward SDE 共享 one-time marginals。它不恢复原 SDE 的
path law。更重要的是，普通 score diffusion 的基本输入是一条 chosen noising law；
它不以两个 hard endpoint constraints 下的 `min H(P|R)` 为定义。

因此 score SDE 与 SB 可以共享 score/time-reversal 工具，但不是同一个优化问题。

## 3. Flow Matching：先选择 marginal path，再回归 ODE velocity

[Lipman et al. 2023](https://arxiv.org/abs/2210.02747 "官方论文页面")先给
density path `p_t` 与生成它的 velocity `u_t`，定义

```text
L_FM(theta)=E_{t,X_t~p_t}|v_theta(t,X_t)-u_t(X_t)|^2.          (3.1)
```

直接访问 marginal target 通常很难，于是引入 conditional paths `p_t(x|z)`。若
`u_t(x|z)` 生成每条 conditional path，则 Theorem 1 给出

```text
u_t(x)=E[u_t(X_t|Z)|X_t=x].                                  (3.2)
```

Theorem 2 在 positivity 与积分/正则条件下证明 CFM 与 FM objective 相差参数无关
常数，因而 population gradients 相同。

公式 (3.2) 与 score regression 很像，但 random target 不同：一个回归 velocity，
另一个回归 density score。更关键的是，FM 的 path/coupling design 已经先选择了
`p_t`。原始 objective 没有 reference path law，也没有优化

```text
min {H(P|R): P_0=mu_0, P_T=mu_T}.                             (3.3)
```

Flow Matching 的 exact population 结论是“ODE 生成 chosen marginal curve”，不是
“ODE path law 是 Schrödinger Bridge”。Lipman et al. 对 conditional OT paths 也明确
警告：conditional path 的局部 optimality 不推出 marginal/global OT optimality。

## 4. Stochastic interpolants：coupling 是模型输入

[Albergo et al. 2025](https://arxiv.org/abs/2303.08797 "官方论文页面")
从 endpoint coupling `(X_0,X_1)~nu`、latent Gaussian `Z` 与插值函数出发：

```text
X_t^I=I(t,X_0,X_1)+eta(t)Z.                                  (4.1)
```

`nu`、`I` 与 noise schedule `eta` 共同决定 marginal curve `rho_t`。endpoint marginals 本身并不
选择 `nu`。在论文 Assumption 5 的正则条件下，Theorems 6--8 给出

```text
b_t(x)=E[dot X_t^I|X_t^I=x],
s_t(x)=grad log rho_t(x),                                    (4.2)
```

以及相应 population quadratic objectives。若用 diffusion variance convention
`g_t^2=2 epsilon_t`，同一 `rho_t` 可由

```text
ODE:            dot X_t=b_t(X_t),
forward SDE:    dX_t=(b_t+epsilon_t s_t)(X_t)dt
                      +sqrt(2epsilon_t)dW_t,
Nelson original-time backward drift:
                b_t-epsilon_t s_t                             (4.3)
```

实现。这里 `b_t` 专指 current/probability-flow velocity，不是 B1/B4 中常用的
forward SDE drift；reversed-clock drift 还需按 B4/B6 的时间反演约定变号并换时钟。

Corollary 18 给出这些 same-marginal realizations；Remark 20 明确说 interpolant、
ODE 与 forward/backward SDE 是不同 stochastic processes。换言之，统一的是
`rho_t` 与 field conversion，不是 path law。

### 4.1 什么时候 stochastic interpolant 才成为 SB

论文 Section 3.4 另行引入 hydrodynamic SB objective。Assumption 39 要求未知最优
density curve 可被高度正则的 reversible map 映到标准 Gaussian；Theorem 41 再对
interpolant 与 velocity 做额外 max--min optimization，才恢复该 regime 下的 SB。

所以准确关系是

```text
arbitrary stochastic interpolant != SB;
interpolant class + extra variational optimization
                  + Assumption 39 => stated SB solution.       (4.4)
```

原文将 Theorem 41 的数值研究留给 future work，不能把它写成已经验证的通用 solver。

## 5. Schrödinger Bridge：objective 会选择 coupling 与 path law

B2/B5 的定义从 reference path law `R` 出发：

```text
P*=argmin {H(P|R): P_0=mu_0, P_T=mu_T}.                       (5.1)
```

相对熵 chain rule 把 candidate law 分成 endpoint coupling `gamma=P_0T` 与 conditional
paths：

```text
H(P|R)=H(gamma|R_0T)
       +integral H(P^{xy}|R^{xy})gamma(dxdy).                  (5.2)
```

因此 exact optimizer 同时做两件事：选择 distinguished endpoint coupling `gamma*`，
并在 `gamma*`-a.e. endpoints 下保留 reference bridges `R^{xy}`。这比“找到一条
marginal curve”更强，也比“输出正确 endpoint samples”更强。

在 Brownian/nondegenerate diffusion 等 classical Markov setting，SB 还具有 Markov、
reciprocal 与 two-sided drift structure。Flow/score regression 只有在其 target、
coupling 与 exact operator 被证明对应 (5.1) 时，才能继承这些结构。

## 6. 同一 marginal 与 velocity 仍不确定 path law

令 `X_0~N(0,1)`，比较 constant ODE

```text
X_t=X_0                                                        (6.1)
```

与 stationary Ornstein--Uhlenbeck process

```text
dY_t=-2Y_tdt+2dW_t,       Y_0~N(0,1).                         (6.2)
```

二者每个时刻都服从 `N(0,1)`。OU 的 score 为 `-x`，所以 (2.3) 给其 PF/current
velocity `-2x-2(-x)=0`，也与 constant ODE velocity 相同。

但 covariance kernels 是

```text
Cov(X_s,X_t)=1,
Cov(Y_s,Y_t)=exp(-2|t-s|).                                   (6.3)
```

前者路径恒定，后者具有非零 quadratic variation；endpoint couplings、transition
kernels 与完整 path laws 均不同。

![相同 N(0,1) marginals 与零 current velocity 对应不同 ODE/OU path laws](/images/bridge/B12_same_marginals_paths.png)

**图 6.1：** constant ODE 与 stationary OU 共享全部 `N(0,1)` one-time marginals
和零 current velocity，但 covariance kernel、endpoint coupling、quadratic variation
与完整 path law 均不同。

说明代码用 120000 条路径复核 stationary marginals 与 OU covariance。端点协方差
gap 为 `0.868894`，shared PF velocity 误差为 `0`。这不是训练 benchmark，而是
“marginal/PDE equivalence 不可升级成 path-law equivalence”的有限 Gaussian witness。

## 7. DSB：借用 diffusion 工具，不等于普通 DDPM

[De Bortoli et al. 2021](https://arxiv.org/abs/2106.01357 "官方论文页面")
把 classical IPF half-steps 表成 forward/backward diffusions，并用 neural regression
近似难处理的 drift/time-reversal update。这与 score diffusion 共享 SDE simulation、
score-like targets 与 neural parameterization，但两者的 outer problem 不同：

```text
score diffusion: chosen one-sided noising process + reverse modeling;
DSB:             two hard endpoints + alternating path-law KL projections. (7.1)
```

还要区分三层：

```text
exact IPF law update
-> population conditional-mean regression
-> finite neural training and discretized simulation.         (7.2)
```

De Bortoli et al. Propositions 2、4--6 的 exact representation/monotonicity/convergence
不会自动传给有限 Algorithm 1。网络、样本、optimization、outer truncation 与 SDE
discretization 都是额外误差。

## 8. DSBM/IMF：projection 各自保留不同对象

[Shi et al. 2023](https://doi.org/10.52202/075280-2717 "官方论文页面")
从 endpoint coupling `gamma` 与 reference bridges 构造 reciprocal mixture

```text
Pi_gamma=integral Q^{xy}gamma(dxdy).                           (8.1)
```

Markovian projection 的 population drift 是 bridge drift 的 conditional expectation。
它保留每个 one-time marginal，但一般改变 endpoint coupling 与 pinned bridges。
reciprocal projection 则精确保留 endpoint coupling，并把 conditional paths 换回
reference bridges，但可能改变 intermediate marginals 与 Markovity。

| Exact operator        | 保留                                  | 一般改变                               |
| --------------------- | ----------------------------------- | ---------------------------------- |
| Markov projection     | all one-time marginals              | endpoint coupling、bridges、path law |
| reciprocal projection | endpoint coupling、reference bridges | intermediate marginals、Markovity   |

exact IMF 交替两个 operators。在 Shi et al. Appendix C 的 finite-KL、closure、
tightness/coercivity 与 lower-semicontinuity responsibilities 下，Theorem 8 才给相应
asymptotic conclusion。DSBM 用 regression 近似 Markov step；finite learned DSBM
不是 exact IMF theorem 的直接 corollary。

这一节与 Flow Matching 的共同点是 conditional expectation；不同点是 DSBM 的
target 来自 reference bridge mixture，并嵌在 reciprocal/Markov projection sequence 中。

## 9. SF2M：coupling 决定学到哪条 bridge mixture

[Tong et al. 2024](https://arxiv.org/abs/2307.03672 "官方论文页面")
先从 coupling `q(x_0,x_1)` 抽 endpoint pair，再从 analytic conditional bridge 抽
`X_t`。Theorems 3.1--3.2 给

```text
u_t(x)=E[u_t(x|X_0,X_1)|X_t=x],
s_t(x)=E[s_t(x|X_0,X_1)|X_t=x].                              (9.1)
```

Theorem 3.3 说明 global population fit 恢复 **该 coupling 所定义 bridge mixture** 的
marginals。改变 `q` 会改变中间 marginals 与 regression targets，即使两端 marginal
保持不变。

对 Brownian reference，Proposition 3.4 补上决定性条件：使用 exact entropic-OT
endpoint coupling 才恢复 SB。于是

```text
arbitrary q + Brownian bridges -> a reciprocal bridge mixture;
entropic-OT q* + Brownian bridges -> the Brownian SB.          (9.2)
```

“simulation-free”只说明训练不模拟当前 learned process。它仍需要 coupling samples、
analytic bridge samples，并在推理时积分 learned ODE/SDE；实际 minibatch OT、finite
regression 与 solver 又分别引入误差。

## 10. 统一比较表

| 方法                      | 基本输入                             | 被学习/优化对象                      | exact 层结论                    | SB 还缺什么                                      |        |
| ----------------------- | -------------------------------- | ----------------------------- | ---------------------------- | -------------------------------------------- | ------ |
| score SDE               | data + forward SDE               | marginal score                | reverse SDE/PF interfaces    | hard second endpoint + path KL problem       |        |
| Flow Matching           | chosen conditional/marginal path | ODE velocity                  | generates chosen marginals   | reference、coupling optimization、path law     |        |
| stochastic interpolants | coupling + interpolant + noise   | velocity/score/fields         | same-marginal ODE/SDE family | extra variational optimization + assumptions |        |
| SB                      | reference + two marginals        | path law                      | minimizes \`H(P              | R)\`                                         | 已是定义对象 |
| DSB                     | SB + IPF iteration               | alternating drifts/laws       | exact IPF under assumptions  | learned-step stability                       |        |
| DSBM/IMF                | reference bridges + coupling     | Markov/reciprocal projections | exact IMF under assumptions  | learned projection control                   |        |
| SF2M                    | coupling + analytic bridges      | mixture score/velocity        | chosen-mixture marginals     | exact entropic coupling for Brownian SB      |        |

完整 object--constraint--objective--exactness 矩阵见
B12 responsibility ledger（补充材料暂未公开）。

## 11. “等价”必须写出层级

下列说法只有加上限定才成立：

- **CFM 等价于 FM**：population objectives 相差参数无关常数；不表示有限训练相同。
- **SDE 等价于 PF-ODE**：exact fields 下 one-time marginals 相同；path laws 不同。
- **score 与 velocity 可互换**：需固定 density 与 diffusion convention；它们不是同一 field。
- **bridge matching 得到 SB**：需 exact projection/fixed-point assumptions；learned step 另算。
- **stochastic interpolant 可解 SB**：需 Theorem 41 的额外 max--min 与 Assumption 39。
- **SF2M 是 SB**：需 conditional bridges匹配 reference，且 coupling 是对应 entropic optimum。

一个可靠的技术陈述应包含

```text
equivalent at [object level]
under [assumptions]
for [exact/population/learned/discretized layer].              (11.1)
```

## 12. 误差账本与方法选择

从 theorem 到样本至少可能经过：

```text
path/coupling design error
+ conditional-target or bridge-sampling error
+ function-class/statistical error
+ optimization and outer-iteration error
+ ODE/SDE discretization error.                               (12.1)
```

Flow Matching 主要把 modeling responsibility 放在 probability path/coupling design；
score diffusion 放在 forward noising law 与 score estimation；DSB/DSBM 放在 reference、
endpoint constraints 与 repeated path-law projections；SF2M 将训练 simulation 换成 coupling
与 analytic conditional bridge primitives。

因此方法选择不应从“都用 MSE”出发，而应从手中可验证的对象出发：是否真的需要
two-sided constraints？是否关心完整 stochastic dynamics，还是只关心 endpoint samples？
是否有可信 coupling 或 analytic bridge？是否需要 reference-relative path optimality？

## 13. 常见错误

- **“所有连续生成方法只是 parameterization 不同。”** 它们可能优化不同对象。
- **“marginals 相同，所以随机过程相同。”** OU/constant-ODE 反例直接否定。
- **“velocity 学准就恢复了 stochastic drift。”** 还缺 score 与 diffusion coefficient。
- **“任意 coupling 只影响训练方差。”** coupling 可以改变整条 marginal curve。
- **“两端样本正确就是 SB。”** 还缺 reference bridges 与 path-KL optimality。
- **“DSB 是把 DDPM 跑两遍。”** DSB 的 outer object 是双端 path-law KL projection。
- **“simulation-free 不需要模拟。”** 它仍采 conditional bridges，推理仍积分 dynamics。
- **“exact operator theorem 证明 neural training 收敛。”** 中间缺 approximation/stability theorem。

## 14. 小结与思考题

本章真正统一的是责任语言，而不是把所有方法压成一个 loss：conditional expectation
解释 population regression，continuity/Fokker--Planck equations解释 marginal evolution，
reference、coupling 与 variational objective 才决定 path-law claim 能走多远。

1. 给定同一 `rho_t` 与 current velocity，列出还需要哪些量才能恢复 forward SDE drift。
2. 构造两个具有相同 endpoint marginals、不同 coupling 的 Gaussian interpolants，比较中间 variance。
3. 为什么 Markov projection 保留 all one-time marginals仍可能改变 `(X_0,X_T)` joint law？
4. 若 SF2M 使用 minibatch OT coupling，哪些误差不能由 Theorem 3.3 消除？
5. 为一句“DSBM 与 Flow Matching 等价”的表述补齐 object level、assumptions 与 exactness layer。
