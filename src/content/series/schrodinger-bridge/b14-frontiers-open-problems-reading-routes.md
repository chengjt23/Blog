---
title: 理论前沿、开放问题与阅读路线
description: 汇总全系列的保证层级、尚未闭合的理论问题和按背景划分的阅读路线，避免把 population 结果误读为训练保证。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 14
slug: b14-frontiers-open-problems-reading-routes
tags:
  - schrodinger-bridge
  - theory
  - reading-guide
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 覆盖 B0--B13 的保证地图、IMF/IPMF 前沿、开放问题和四条带注释阅读路线。
---
本篇前十三章不断出现“exact”“convergence”“matching”和“equivalent”。这些词只有在
证明对象、假设、metric 与实现层级都写清时才有意义。例如，exact IPF/IMF 的
path-law projection theorem，不会因为实现用了相同 loss，就自动成为有限网络的训练
收敛定理；population conditional expectation 恢复了正确 field，也不表示 Euler
sampler 已恢复同一个 path law。

因此本章不把“经典理论”和“现代算法”各列一份文献清单，而是沿同一条责任链审计：

```text
problem
 -> exact operator
 -> finite outer algorithm
 -> population regression
 -> finite learned estimator
 -> discretized sampler
 -> evaluation.                                                (0.1)
```

每一次向右移动，都需要新的 stability、approximation 或 numerical theorem。没有这座
桥时，正确写法是保留缺口，而不是把左侧 theorem 的名称搬到右侧。

## 1. 五层 guarantee：先问证明控制了什么

一条保证至少应回答四个问题：对象是什么，在哪些假设下成立，用什么 metric，是否
覆盖实际计算。按证明责任，可先分成五层。

| 层级                | 被证明的对象                           | 一个合格陈述必须说明                                                        | 典型失败                                           |
| ----------------- | -------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------- |
| problem           | feasible path law / optimizer    | state/path space、reference、constraints、support、entropy finiteness | 目标不可行，或 optimizer 不存在/不唯一                      |
| exact operator    | projection / transform           | 输入输出 families、KL 方向、invariant、fixed points                        | 只证明 fixed point，却声称全局收敛                        |
| finite algorithm  | exact/inexact iterates           | initialization、停止准则、residual/gap、rate assumptions                 | marginal residual 很小，却 objective gap 未知        |
| learned estimator | population target / finite model | sampling law、function class、统计与优化误差                               | population identity 被写成 finite-network theorem |
| sampler           | discretized path/output          | SDE/ODE scheme、weak/strong/path metric、endpoint 与 path-law error  | endpoint samples 好，却完整路径结构错误                   |

这些层级不是强弱词汇的修辞排序，而是不同的数学对象。B3 的 coupling uniqueness 不
证明 B7 的 finite iteration rate；B7 的 exact cyclic projection convergence 不证明
B8 的 neural regression 收敛；B9 的 exact IMF theorem 也不控制 learned drift 与
time discretization。**任何保证都不会自动向表格下方传播。**

## 2. B0--B12 已闭合到哪里

### 2.1 经典问题链

B0--B7 建立的是 problem 与 exact-operator 基线：

- [B0](/blog/schrodinger-bridge/b0-schrodinger-problem-timeline/)把有限粒子枚举与 continuous Feller setting
  的 conditional LDP 分开；有限 Stirling 计算不是 path-space LDP 的证明。
- [B1](/blog/schrodinger-bridge/b1-brownian-to-diffusion-bridges/)区分 point-conditioned exact bridge、Doob
  transform 与 guided proposal；后者通常仍需 likelihood correction。
- [B2](/blog/schrodinger-bridge/b2-path-space-schrodinger-problem/)用 entropy disintegration 把 path KL 分成
  endpoint coupling 与 pinned conditional laws；任意 endpoint mixture 不一定 Markov。
- [B3](/blog/schrodinger-bridge/b3-static-schrodinger-system/)在正核及明确的 support 条件下给 factorization、
  gauge 与 coupling uniqueness；Lagrange multiplier 计算不能替代一般 existence。
- [B4](/blog/schrodinger-bridge/b4-markov-reciprocal-dynamics/)区分 Markov 与 reciprocal，并限定
  two-sided dynamics/time reversal 的正则条件。
- [B5](/blog/schrodinger-bridge/b5-schrodinger-bridge-entropic-ot/)分开 finite-noise identity、viscous action、
  zero-noise Gamma/minimizer/interpolation limits；Sinkhorn divergence 不是 SB path law。
- [B6](/blog/schrodinger-bridge/b6-stochastic-control-girsanov-follmer/)分开 chosen-control upper bound、
  Föllmer equality、strong realization 与 HJB verification。
- [B7](/blog/schrodinger-bridge/b7-fortet-ipf-sinkhorn/)给 finite exact I-projection、support
  boundary 与 qualitative convergence；log-domain stability 解决 arithmetic 问题，
  不等于 nonasymptotic outer-loop rate。

这条链闭合了“问题如何定义、经典结构怎样出现、exact projection 在什么条件下成立”，
但它没有讨论 finite neural approximation。

### 2.2 从 exact operator 到现代 matching

[B8](/blog/schrodinger-bridge/b8-neural-schrodinger-bridge/)、
[B9](/blog/schrodinger-bridge/b9-bridge-matching-markovian-projection-imf/)和
[B10](/blog/schrodinger-bridge/b10-simulation-free-generalized-matching/)把现代方法拆成四个不能合并的对象：

```text
exact path-law operator
 != population regression optimum
 != finite learned algorithm
 != discretized sampler.                                      (2.1)
```

B8 的 IPF/control/FBSDE identities 解释 population target 来自哪里，但不提供覆盖所有
network、sample、optimization、outer-loop 和 Euler errors 的统一 theorem。B9 的 exact
Markovian projection 保留 one-time marginals，exact reciprocal projection保留 endpoint
coupling/reference bridges；learned DSBM 只是对其中 local operator 的近似。B10 又进一步
说明 SF2M、GSBM、Light/Optimal SBM、ASBM 与 IPMF 替换的是不同 primitive，不能借用
彼此的 convergence claim。

### 2.3 扩展与比较的边界

[B11](/blog/schrodinger-bridge/b11-discrete-multimarginal-meanfield/)已经分别核验 finite CTMC、
positive multi-marginal tensor、soft/unbalanced、mean-field、compact manifold 与 reflected
reference 的代表结论。但“每个扩展族有一个 theorem”不推出“六类扩展可以无条件组合”。

[B12](/blog/schrodinger-bridge/b12-diffusion-flow-matching-unification/)则给出一个重要反例：constant ODE 与
stationary OU 可以共享全部 one-time marginals 和 current velocity，却有不同 covariance
kernel、quadratic variation、endpoint coupling 与 path law。这说明 E7 的 marginal metric
无法单独识别 E1--E2 的 path object。

## 3. E0--E7：从建模到评估的误差栈

对现代算法，本篇统一使用以下顺序：

```text
E0 reference / constraint misspecification
 -> E1 exact problem existence / uniqueness
 -> E2 exact operator or fixed-point characterization
 -> E3 finite outer iteration
 -> E4 approximate regression / projection class
 -> E5 finite sample and optimization
 -> E6 time discretization / numerical arithmetic
 -> E7 evaluation and identifiability.                         (3.1)
```

| 层  | 问题                                                 | 需要的证据                                                                   | 不能由什么替代                        |
| -- | -------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------ |
| E0 | reference、coupling 或 constraints 是否描述了任务           | data-generating assumptions、support 与 sensitivity audit                 | 更大的 network                    |
| E1 | exact optimizer 是否存在且唯一                            | feasibility、finite entropy、compactness/coercivity、strict convexity或相应替代 | algorithm fixed point          |
| E2 | operator 是否真是所称 projection/transform               | KL direction、invariants、Pythagorean identity、well-posedness             | 相同形式的 regression loss          |
| E3 | 截断后的 exact/inexact iteration 离目标多远                 | residual-to-gap bound、rate、perturbation accumulation                    | 只画 loss curve                  |
| E4 | population target 能否由所选 function class 表示          | approximation/stability theorem                                         | universal approximation slogan |
| E5 | finite samples 与 optimizer 是否接近 population optimum | generalization、concentration、optimization error                         | training loss 单值               |
| E6 | 数值积分与 arithmetic 是否保持所需 law                        | weak/strong solver error、KL/Wasserstein/path metric、roundoff analysis   | 减小步长的经验图                       |
| E7 | benchmark 是否识别所声称对象                                | endpoint、coupling、path、cost、likelihood 的分项指标                            | 单一 endpoint FID/Wasserstein    |

几个已核验例子能说明误差不能“跨层抵消”：

1. B7 的 log-domain Sinkhorn 修复的是 E6 underflow，不证明 E3 rate，也不能修复 E1
   infeasibility。
2. Flow Matching/CFM 的 theorem 在 population 层识别 chosen marginal velocity；它不在
   E1 选择 SB objective，也不控制 finite network 的 E5。
3. SF2M 若使用 exact Brownian bridge 与 exact entropic-OT coupling，population object
   可以是 SB；minibatch coupling、finite regression 与 solver 分别重新引入 E0/E4--E6。
4. B13 的 finite rare-event Doob fixture 表明：committor 精确时可得到 zero-variance
   proposal；Hartmann--Schütte、Chetrite--Touchette 与 SFS 又分别说明 finite-horizon
   KL control、long-time driven process 和 finite-step sampler 有不同保证。learned
   committor/drift 的误差仍须在 E4--E6 重新分析。
5. endpoint Wasserstein、sample quality 或 all-time marginal matching 属于 E7 证据，不能
   单独证明 path KL、endpoint joint law 或 reciprocal/Markov structure。

误差栈不是要求每篇论文一次解决全部八层，而是要求结论明确停在哪一层。

## 4. 四种 exactness 不能折叠

现代 bridge 论文中最容易混淆的是下面四个层级。

| 层级                       | 形式                                           | 可合法声称                                                       | 仍然缺少                                          |      |                                             |                                         |
| ------------------------ | -------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------- | ---- | ------------------------------------------- | --------------------------------------- |
| exact operator           | 对 full path laws 做解析 KL projection/transform | invariants、fixed point、在 theorem assumptions 下的 convergence | estimator 如何实现 operator                       |      |                                             |                                         |
| population regression    | \`argmin\_h E                                | Y-h(X)                                                      | ^2=E\[Y                                       | X]\` | rich class/global optimum 下恢复指定 local field | finite representation、samples、optimizer |
| finite learned algorithm | finite network/data/SGD/outer loops          | 只有另有 theorem 时才有 finite guarantee                           | 通常还缺 projection stability 与 error propagation |      |                                             |                                         |
| discretized sampler      | Euler/ODE/SDE numerical paths                | 指定 scheme 和 metric 下的数值结论                                   | continuous learned law 与 exact target 的距离     |      |                                             |                                         |

例如 B9 的 finite-state IMF rate控制 full-law exact iterates，不含 neural drift regression；
B12 的 conditional expectation identity控制 population field，不选择 endpoint coupling；即使
field 被准确估计，离散 sampler 仍需要独立的 SDE/ODE error theorem。用“理论上 exact”
概括这四行，会把三个尚未证明的蕴含隐藏起来。

## 5. 深入一例：有限状态 exact IMF 的 proof map

为了看到一个 convergence claim 实际承担多少责任，考虑
[Sokolov--Korotin 2025](https://arxiv.org/abs/2508.02770 "官方论文页面")
的有限状态结果。固定有限 state space、离散时刻 `t_0<...<t_N`、处处正的 reference
pinned kernel

```text
q(x_1,...,x_{N-1}|x_0,x_N),
```

以及处处正的 endpoint marginals `mu,nu`。从具有这些 marginals 的 reciprocal law
出发，exact IMF 在 full path-law 层交替

```text
reciprocal law
 -> exact Markovian projection
 -> exact reciprocal projection.                              (5.1)
```

令 `p*` 为该 positive finite model 中的唯一 SB，并记

```text
epsilon_q  = min q(x_1,...,x_{N-1}|x_0,x_N),
epsilon_mu = min_x mu(x),
epsilon_nu = min_x nu(x),
m = epsilon_q^(N+2) epsilon_mu epsilon_nu.                      (5.2)
```

论文 Theorem 1 对 exact iterates `p_k` 给出 reverse-KL 几何界

```text
KL(p_k||p*)
 <= (1-m^3/4)^(k-1) KL(p_0||p*).                               (5.3)
```

### 5.1 证明责任链

这一定理的 proof map 可分成四步：

1. finite state 与 strict positivity 使所有 admissible iterates 位于有限维 probability
   simplex 的 relative interior。
2. exact reciprocal/Markov updates 是相应 constraint families 上的 KL minimizations；
   Pythagorean identities 把单步改进写成 KL/Bregman projection gap。
3. (5.2) 为 iterates 的 coordinates 提供 uniform lower bound；在这个 compact interior
   region 上，negative entropy 具有 proof 所需的 strong-convexity/smoothness constants。
4. projection-gap estimate 至少控制当前 KL error 的 `m^3/4`；移项得到 one-step
   contraction，再归纳得到 (5.3)。

这里复述的是已引 theorem 的责任链与常数，不声称给出了超出该 finite-state result 的
独立一般证明。

这里 strict positivity 不是方便记号。若 endpoint 或 transition mass 可为零，`m` 退化，
该 rate theorem 就不可用。`epsilon_q^(N+2)` 也意味着时间网格变细时 bound 可能快速变得
数值上空泛：它是严格的 nonasymptotic theorem，却不是 dimension/time-grid-free guarantee。

### 5.2 这一定理没有证明什么

(5.3) 不控制：

```text
inexact Markovian projection
+ approximate reciprocal/conditional sampler
+ finite sample and optimization
+ time discretization.                                       (5.4)
```

若要把 (5.3) 升级成 learned DSBM/IPMF 的 end-to-end bound，至少需要在同一个 metric 下
给每步 perturbation estimate，并证明这些扰动可求和、被 contraction 吸收，或收敛到一个
可量化 error floor。population regression identity 与 endpoint sample quality 都不提供
这条缺失的 stability bridge。

## 6. IMF/IPMF convergence 结果不能互相替代

| 结果                                                      | 状态（截至 2026-07-15）                          | 机制与结论                                                                                                 | 不能外推到                                                      |
| ------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Shi et al. 2023, Theorem 8                              | NeurIPS 2023 正式论文                          | closure/coercivity、lower semicontinuity 等条件下 exact IMF asymptotic reverse-KL conclusion               | explicit rate；finite learned DSBM                          |
| Sokolov--Korotin 2025, Theorem 1                        | arXiv v2 frontier preprint；未核验正式 venue     | finite/discrete/positive model 的 explicit geometric reverse-KL rate (5.3)                             | continuous diffusion；zeros；learned steps                   |
| Gentiloni Silveri--Conforti--Durmus 2025, Theorems 1--2 | arXiv v1，原文标记 under review                 | Langevin reference、log-concavity/convexity-at-infinity 与 sufficiently large horizon 下 exact IMF rates | arbitrary horizon/data；drift estimation；SDE discretization |
| Kholkin et al. 2026, Theorems 3.2--3.3                  | ICLR 2026 正式论文；本地为 arXiv v5 author version | 特定 Gaussian regime 的 exponential convergence；bounded support 的 weak convergence                       | 一般 distribution/dimension/noise；finite learned IPMF        |

[Gentiloni et al. note](https://arxiv.org/abs/2510.20871 "官方论文页面")
中的 functional-inequality argument 与有限 simplex proof 不是同一个机制；
[Kholkin et al. note](https://arxiv.org/abs/2410.02601 "官方论文页面")研究的四步 IPMF 也不是
(5.1) 的两步 IMF。它们是互补的 regime-specific results，不是可删除彼此假设的证据。

## 7. 2025--2026：正式结果、预印本与 frontier 必须分栏

年份新不等于证据弱，但新工作必须保留版本责任。当前核验到的代表条目如下。

| 工作                                                          | 版本状态                                  | 本篇如何使用                                                                                                         |
| ----------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Hernández--Tangpi 2025 mean-field propagation of chaos      | *SIAM J. Control Optim.* 63(1)，正式论文   | 在 convexity/feasibility 与 weak terminal constraint 边界内使用；hard equality 还需额外 smooth displacement-convex penalty |
| Albergo--Boffi--Vanden-Eijnden 2025 stochastic interpolants | *JMLR* 26，正式论文                        | 使用 same-marginal ODE/SDE 与 population regression；SB 恢复另需 Theorem 41 的 max--min 与 Assumption 39                 |
| 两篇 2025 exact IMF rate                                      | arXiv preprints；其中一篇 under review     | 仅作 theorem-regime 已核验的 frontier rates，不称正式发表或 learned-algorithm guarantee                                      |
| Kholkin et al. 2026 IPMF                                    | ICLR 2026 正式论文                        | 使用 Theorems 3.2--3.3 的受限结论；一般 convergence 仍按论文标记为 conjectural                                                  |
| Owusu Adu et al. 2026 sub-Riemannian SB                     | arXiv v2 `frontier_candidate`，未核验正式发表 | 只进入更新观察表；Eulerian uniqueness、heat-kernel/zero-noise claims 需独立确认，不替代 compact boundaryless baseline             |

这张表还说明两种常见误读：正式发表不表示 theorem 没有狭窄假设；预印本也不表示
可以忽略其已明确证明的 regime。状态决定引用语气，assumptions 决定数学结论。

## 8. 开放问题：只能从已核验缺口反向生成

下面每个问题都对应 B0--B12 的明确边界，而不是泛化的“未来可研究”。

### 8.1 支撑退化与低正则的 existence/factorization

B3 的完整 factorization 主要依赖正核；结构零只在 bounded support-qualified 情形闭合。
B1/B4 的 bridge drift 与 time reversal 也依赖 density、finite entropy 或 well-posedness。
开放问题是：在 singular marginals、zero transition support、degenerate diffusion 或
non-dominated reference 下，怎样同时给出 existence、factorization 与可计算 operator，
且不把 smooth score formula 当成 path-law definition？

### 8.2 Inexact IMF/IPMF 的 nonasymptotic perturbation theory

现有 exact IMF/IPMF rates不含 regression、adversarial projection、conditional sampler
和 time discretization。需要的不是再证明一个 exact fixed point，而是形如

```text
d(P_{k+1},P*) <= rho d(P_k,P*) + C delta_k               (8.1)
```

的 operator stability，其中 `d`、KL direction、`delta_k` 的 estimator meaning 与 support
assumptions 都可验证。随后还需说明 `delta_k` 可求和、趋零，或导致怎样的 error floor。

### 8.3 Coupling learning 与 reference misspecification 的 identifiability

B9 的 aligned pairs 是 data-generating assumption；B10 的 Gaussian fixture 证明同端点
marginals、不同 coupling 会改变中间 targets；B12 又证明相同 marginals 不能识别 path
law。开放问题是：哪些 observations 能区分 coupling error 与 reference error？若只观察
sparse marginals 或 paired noisy data，哪些 path quantities 是 identifiable，哪些只是
modeling choice？

### 8.4 Field error 怎样传播到 endpoint 与 path law

population score/drift/velocity regression通常只控制一个加权 `L2` risk。要得到 SB-specific
结论，需要把它传播到 endpoint KL、endpoint coupling、path KL 或 control cost。不同
diffusion channels、low-density regions 和 terminal singular drift 会怎样放大误差？ODE
likelihood、SDE weak error 与 full path relative entropy 需要不同的 stability tools，不能
用一个 marginal theorem 统一替代。

### 8.5 六类 extension 能否组合

B11 分别有 finite jump、multi-marginal、unbalanced、mean-field、manifold 和 reflection
的代表理论，但没有一个 theorem 同时覆盖这些改变。尤其 nonlinear
`H(P|Gamma(P))` 失去固定 linear reference 的 convex结构，reflection 增加 boundary
local time，countable jumps 还会有 explosion。开放问题是找出哪些经典 entropy projection
结构仍可组合保留，以及哪些组合必须换问题或换 divergence。

### 8.6 General IPMF 与 learned generalized matching

IPMF 目前的证明限于指定 Gaussian 或 bounded-support regimes；GSBM 只有 exact stages
的 monotonicity 与 optimum fixed-point inclusion，没有 global convergence。开放问题应
分成两步：先扩大 exact population operator 的 convergence regime，再分析 finite CondSOC、
adversarial或 potential-family approximation。跳过第一步会让失败无法归因于 E1--E3 还是
E4--E6。

### 8.7 SB-specific evaluation 与可证伪比较

B12 的反例说明 endpoint/marginal metrics 太弱。一个能检验 SB-specific claim 的 protocol
至少要分开 endpoint fit、endpoint coupling、reference-bridge consistency、path/control
cost 与 numerical error。哪些有限统计量能可靠拒绝错误 path law，怎样设计不把 extra
supervision、model size 或 solver budget 误记成 SB gain，仍需要应用域内的独立证据。

## 9. B13 应用审计：E0 与 E7 不能留到实验之后

[B13](/blog/schrodinger-bridge/b13-applications-and-evidence/)完成了 image、single-cell、
linear control、rare-event 与 one-sided sampling 五类代表性证据链。结论不是“SB 在五类
任务中都更好”，而是每类收益必须分解到不同责任：

| 应用                  | E0 的关键输入                                           | exact/problem object                       | E7 能验证什么                                 | 不能归因于 SB 的部分                               |
| ------------------- | -------------------------------------------------- | ------------------------------------------ | ---------------------------------------- | ------------------------------------------ |
| I2SB restoration    | paired clean/degraded endpoints                    | pair-conditioned Dirac-boundary bridge     | output FID/accuracy                      | pairing、image prior 与 finite network       |
| single-cell         | sparse snapshots，加 growth/lineage/alignment 才改变信息集 | chosen coupling/path law under a reference | marginal 或 lineage-derived coupling，必须分开 | 未观测的真实 individual trajectory               |
| covariance steering | known plant/noise/controllability                  | minimum-energy stochastic law              | endpoints、feasibility、energy             | generic nonlinear robotics                 |
| rare event          | event/cost/committor family                        | conditioned or KL-controlled path law      | unbiasedness、weights、variance、event rate | 仅凭 proposal 下事件变常见                         |
| SFS                 | target energy 与 Gaussian reference                 | one-sided Föllmer transport                | `W_2`/posterior sampling quality         | physical trajectory 或通用 two-marginal claim |

single-cell 证据尤其说明 identifiability 不能由更强 optimizer 修复：相同 snapshots 可由
不同 growth、circulation、noise 与 hidden dynamics 产生；lineage/barcode 是额外观测，
不是单纯的 variance reduction。I2SB 的 paired data、LineageOT 的 lineage tree、control
中的 known plant 都必须进入 baseline fairness。

rare-event 证据则把三种常被混称为 Doob/SB 的结论分开：finite exact conditioning 给
constant weights；finite-horizon KL control 的 exact optimum 给 zero-fluctuation identity，
parametric forcing 仍是近似；long-time driven process 只在 LDP、spectral gap 与 convexity
条件下 logarithmically equivalent，通常不共享 finite-time fluctuations。SFS 再增加一层
exact continuous drift 到 Euler/inner-Monte-Carlo error 的传播。

因此第 8.7 节的 benchmark 要求已有应用证据支持：endpoint fit、coupling、reference
consistency、path/control cost、likelihood weights 和 solver error 必须分项报告。inverse/
data assimilation、generic robotics 与 broad molecular/scientific claims 仍被 B13 明确排除，
因为它们尚未形成同等强度的 representative evidence chain。

## 10. 四条 annotated reading routes

四条路线不是四套互不相干的文献。它们在 B2--B8 交汇：probability 提供 path measure，
OT 提供 static coupling geometry，control 提供 drift/energy representation，generative
modeling 才把 exact operators 换成 estimators。

### 10.1 概率与路径测度路线

| 次序     | 必读 checkpoint                                                          | 可暂缓                 | 进入 frontier 的位置                               |
| ------ | ---------------------------------------------------------------------- | ------------------- | --------------------------------------------- |
| B0     | finite migration likelihood；conditional LDP 的 shrinking-set conditions | 完整历史优先权争议           | low-regularity conditional LDP                |
| B1     | finite-time Doob kernel；diffusion bridge weak/strong assumptions       | guided proposal 工程  | singular/degenerate bridge SDE                |
| B2     | standard Borel disintegration；path-entropy chain rule                  | existence 留到 B3     | non-dominated reference                       |
| B3--B4 | factorization；reciprocal vs Markov；two-sided transform                 | numerical scaling   | zero kernels、weak factorization/time reversal |
| B6     | chosen-control upper bound vs Föllmer equality                         | covariance steering | degenerate/time-dependent control channel     |

这条路线的 invariant 是始终区分 endpoint coupling、pinned conditional law 与 full path
measure。没有完成 B2/B4 前，不应把 score 或 drift formula 当作路径对象的定义。

### 10.2 OT 与计算路线

| 次序  | 必读 checkpoint                                            | 可暂缓                       | 进入 frontier 的位置                         |
| --- | -------------------------------------------------------- | ------------------------- | --------------------------------------- |
| B3  | finite positive scaling、gauge、support boundary           | continuous Fortet details | structural zeros/singular marginals     |
| B5  | Gibbs-reference identity；small-noise assumptions         | control derivation留到 B6   | quantitative zero-noise rate            |
| B7  | exact I-projection；Pythagorean identity；support criteria | large-scale GPU solver    | stochastic/inexact scaling              |
| B9  | reciprocal/Markov operators与各自 invariants                | learned DSBM              | exact IMF rates outside current regimes |
| B10 | IPF/IMF/IPMF 的不同 operator sequence                       | benchmark details         | inexact/general IPMF convergence        |

只看 finite Sinkhorn residual 不足以完成这条路线；还要知道 endpoint coupling怎样通过
reference pinned bridges lift 回 dynamic path law。

### 10.3 随机控制路线

| 次序  | 必读 checkpoint                                         | 可暂缓                          | 进入 frontier 的位置                        |
| --- | ----------------------------------------------------- | ---------------------------- | -------------------------------------- |
| B1  | Brownian bridge singular drift 与 terminal limit       | guided proposals             | constrained-domain bridge              |
| B4  | forward/backward drift 与 current/osmotic convention   | reciprocal counterexample可二读 | low-regularity time reversal           |
| B6  | Girsanov、Föllmer、same-channel HJB/Hopf--Cole          | 完整 covariance steering       | degenerate control channels            |
| B8  | classical factors支撑的 FBSDE/population identity        | DeepGSB implementation       | finite-sample/optimization propagation |
| B10 | GSBM exact conditional SOC stages；Theorems 5--6 的精确边界 | spline/importance solver     | global/inexact CondSOC convergence     |

这条路线必须区分 terminal-law half bridge、two-marginal SB 与 general running-cost control。
相同的 quadratic energy 形式不保证它们施加了相同 endpoint constraints。

### 10.4 生成建模路线

| 次序                    | 必读 checkpoint                                                  | 可暂缓                    | 进入 frontier 的位置                          |
| --------------------- | -------------------------------------------------------------- | ---------------------- | ---------------------------------------- |
| Diffusion D3--D4 只读接口 | population score identity；reverse SDE/PF-ODE 的 object boundary | 后续训练工程                 | end-to-end sampler error                 |
| B8                    | exact IPF vs population regression vs finite network           | FBSDE implementation细节 | learned IPF guarantee                    |
| B9                    | marginal mimicking、coupling、exact IMF assumptions              | aligned application    | approximate projection stability         |
| B10                   | SF2M coupling条件；GSBM/ASBM/IPMF responsibility                  | 五套实现的完整复现              | coupling learning与 finite-data guarantee |
| B12                   | object/constraint/objective 表；same-marginals counterexample    | 应用 survey              | identifiable SB-specific benefit         |

这条路线在 population identity 被 finite trained model 替代时进入 frontier。从这里开始，
function class、sample、optimization、outer iteration 与 discretization 必须分别记账。

### 10.5 路线交汇图

```text
probability: B0 -> B1 -> B2 -> B3/B4 -> B6
                         |       |       |
OT:                      B3 -> B5 -> B7 -+
                                      |  |
control:                         B6 -> B8
                                      |  |
generative:                    D3/D4 -> B8 -> B9 -> B10 -> B12
                                              |
                                              +-> B14 guarantees
```

这些箭头表示证明义务，不是主题相似：B5 需要 B2 的 path-KL lift，B8 需要 B7 的 exact
operator baseline，B9/B10 需要 B4 的 path-law distinctions。

## 11. 常见错误

1. **把 existence、uniqueness 与 convergence 合成“有理论保证”。** 三者证明对象不同。
2. **把 fixed point 当成 global attractor。** GSBM 的 optimum fixed-point inclusion 就不
   是任意初始化的 convergence theorem。
3. **把 exact operator 换成相似 loss 后继续引用原 theorem。** population regression、
   finite learned update 与 exact KL projection之间缺 stability bridge。
4. **只写 KL，不写方向。** `KL(P_k||P*)`、`KL(P*||P_k)` 与 projection objective 不能互换。
5. **把 all-time marginals 当 path-law equality。** B9/B12 的 Gaussian/OU fixtures 已给反例。
6. **把 endpoint marginals 当 endpoint coupling。** coupling 决定 reciprocal mixture 与
   intermediate targets。
7. **把 numerical stability 当 algorithmic rate。** log-domain arithmetic 与 outer
   convergence属于 E6 与 E3 两层。
8. **把 formal publication 当 universal theorem。** IPMF 2026 正式发表，但一般 regime
   仍是 conjectural；假设不会因 venue 消失。
9. **把 preprint 当作已解决的 baseline。** 2025 IMF rates 与 2026 sub-Riemannian 工作
   必须保留版本及未闭合责任。
10. **用 endpoint sample metric 证明 SB optimality。** 还需 reference、coupling、path
    structure 与 objective evidence。

## 12. 小结

经典 Schrödinger Bridge 的核心链条在明确 assumptions 下已经相当完整：path-space KL、
static factorization、reciprocal/Markov structure、stochastic control 与 exact IPF 可以互相
回链。现代 frontier 的主要困难不是再写一个相似 regression loss，而是证明下面的纵向
传递：

```text
well-posed exact problem
 -> stable exact projection
 -> controlled finite/inexact iteration
 -> statistically and computationally controlled estimator
 -> path-law controlled numerical sampler
 -> identifiable evaluation.                                  (12.1)
```

有限状态 exact IMF rate 展示了 rigorous nonasymptotic theorem 的完整代价：strict
positivity、有限维 compact interior、明确 KL 方向和 exact projections。它也精确指出
learned bridge 方法仍缺什么——不是“更多实验”，而是共同 metric 下的 per-step perturbation
与跨 E3--E6 的 error propagation。

本章已达到 公开预览版 v0.1，资料门槛为 已完成来源核验。B0--B13 的
problem、operator、extension、comparison 与 application evidence 已接入；仍开放的内容
按第 8 节保留为有来源边界的研究问题，而不是本章未完成的写作占位。

## 13. 研究式思考题

1. 在 finite-state IMF bound (5.3) 中，若某个 endpoint mass 趋近零，rate constant 怎样
   退化？这与 structural-zero model 的“不可用”有什么区别？
2. 假设 inexact operator 满足 (8.1)，分别讨论 `sum_k delta_k<infinity`、
   `delta_k->0` 与 `delta_k<=delta` 时能得到什么结论。
3. 构造两个具有相同 endpoint marginals、不同 endpoint coupling 的 finite path laws；
   哪些 E7 metrics 无法区分它们？
4. 为什么 population drift 的 `L2(rho_t dt)` error 不自动给 path KL error？写出至少三项
   还需检查的条件。
5. 比较 exact IMF、D-IMF 与四步 IPMF 的 operator sequence；哪一个 theorem 可以合法
   转移，哪一个不可以？
6. 为一个 mean-field + reflected + multi-marginal 问题列 E0--E2 清单，指出 B11 的哪些
   单项 theorem 不能直接拼接。
7. 设计一个同时报告 endpoint fit、coupling、path cost、reference consistency 与 solver
   error 的最小 benchmark；哪些结论仍然无法由有限样本识别？
8. 选一条阅读路线，标出第一次从 exact operator 进入 learned estimator 的位置，并为
   那个箭头写出你认为最小可行的 stability theorem。

## 14. 本章核验底稿

- Guarantee 与 E0--E7 matrix（补充材料暂未公开）
- 有限状态 exact IMF proof map（补充材料暂未公开）
- 四条 annotated reading routes（补充材料暂未公开）
