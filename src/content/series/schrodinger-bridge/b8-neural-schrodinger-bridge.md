---
title: 高维神经 Schrödinger Bridge：从 DSB 到 FBSDE
description: 从经典交替投影进入高维神经方法，比较 DSB、FBSDE 参数化、采样回归与有限网络误差。
publishedAt: null
updatedAt: '2026-07-15'
draft: true
type: series-chapter
series: schrodinger-bridge
order: 8
slug: b8-neural-schrodinger-bridge
tags:
  - schrodinger-bridge
  - deep-learning
  - fbsde
authors:
  - preview-author
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 覆盖神经交替拟合的 population target、轨迹采样接口与实现层级误差，不宣称有限训练精确收敛。
---
有限状态空间里，我们可以保存一张转移矩阵，用 Sinkhorn 或 IPF 反复校正
边缘分布。图像、分子构象或高维场数据没有这张矩阵可供保存。现代神经
Schrödinger Bridge 方法的共同策略，是保留经典交替投影的结构，但用轨迹采样和
函数回归表示每次前后向更新。

这句话容易造成一个误解：既然外层写得像 IPF，训练出来的网络就一定是精确
Schrödinger Bridge。事实并非如此。本章始终区分三个对象：

1. 路径测度上的 exact IPF projection；
2. 无限数据、无限函数类下的 population regression 或 FBSDE identity；
3. 有限网络、有限样本、有限优化和离散 SDE 构成的 learned algorithm。

只有第一层自动继承 B7 的 exact projection 几何。第二层需要额外的 time reversal、
Gaussian transition 或 FBSDE 表示定理；第三层还要承担逼近与数值误差。

## 1. 从不可存储的势函数到可学习的 drift

设参考路径律为 `R`，两端目标分布为 `mu_0,mu_T`。经典动态问题是

```text
P* = argmin KL(P || R)
     s.t. P_0=mu_0, P_T=mu_T.                         (1.1)
```

B7 的 exact IPF 从 `P^(0)=R` 出发，交替做

```text
P^(2n+1) = argmin_{P_T=mu_T} KL(P || P^(2n)),
P^(2n+2) = argmin_{P_0=mu_0} KL(P || P^(2n+1)).       (1.2)
```

一次 half-bridge 的精确含义很简单：换掉被约束的端点边缘，同时保留当前路径律在
该端点条件下的 conditional path law。它是路径测度的 I-projection，不是一次
SGD 更新。

在离散时间 Markov 链中，若当前 forward law 写为

```text
p^n(x_0:N)=mu_0(x_0) product_k p^n_{k+1|k}(x_{k+1}|x_k),
```

那么 terminal half-step 可写为

```text
q^n(x_0:N)=mu_T(x_N) product_k p^n_{k|k+1}(x_k|x_{k+1}). (1.3)
```

下一次 initial half-step 再保留 `q^n` 的 forward conditionals。De Bortoli 等人的
Proposition 2 给出这一 exact representation。困难在于，已知 forward transition
并不意味着可以在高维中计算 reverse transition；它依赖未知的中间边缘密度。

神经方法因此不直接存储 Schrödinger potentials，而是用网络表示 reverse/forward
mean、drift、score 或 policy。这样解决的是表示和计算问题，不会自动消除经典问题
的存在性、边界条件和收敛责任。

## 2. DSB：外层 IPF 与内层 mean matching

### 2.1 Gaussian 小步近似

DSB 考虑离散小步 forward transition

```text
X_{k+1} = F_k^n(X_k) + sqrt(2 gamma_{k+1}) Z_{k+1},
F_k^n(x)=x+gamma_{k+1} f_k^n(x).                     (2.1)
```

时间反向后，论文用同协方差 Gaussian transition 近似 reverse kernel，并用
`B_{k+1}^n` 表示其均值。与为每轮重新学习全部 marginal score 相比，Proposition 3
给出直接的 mean-matching objective：

```text
B_{k+1}^n
= argmin_B E ||B(X_{k+1})
  - [X_{k+1}+F_k^n(X_k)-F_k^n(X_{k+1})]||^2.         (2.2)
```

这里的期望取在当前 forward joint law `p^n_{k,k+1}` 下。平方损失的 population
minimizer 是条件期望，所以

```text
B_{k+1}^n(x)
= E[X_{k+1}+F_k^n(X_k)-F_k^n(X_{k+1}) | X_{k+1}=x]. (2.3)
```

用拟合的 backward dynamics 从 `mu_T` 采样后，forward 更新具有对称目标：

```text
F_k^{n+1}
= argmin_F E ||F(X_k)
  - [X_k+B_{k+1}^n(X_{k+1})-B_{k+1}^n(X_k)]||^2.    (2.4)
```

Diffusion D3 提供的是“平方损失最优解等于条件期望”这一步。目标变量为何取
(2.2)/(2.4) 的形式，以及它何时近似 reverse transition，则由 DSB 的 Gaussian
transition 推导承担。不能只引用 generic denoising identity 就跳过这一层。

### 2.2 两层算法

DSB Algorithm 1 可以整理为：

```text
initialize forward map F^0 from the reference dynamics
for outer iteration n:
    simulate forward paths from mu_0 using F^n
    fit backward map B^n to the population target (2.2)
    simulate backward paths from mu_T using B^n
    fit forward map F^(n+1) to the population target (2.4)
return the last forward/backward samplers.                      (2.5)
```

`n` 是 outer IPF iteration，`k` 是 physical-time grid，优化器 step 还需要第三个
索引。把三者都叫 `t` 会让“训练收敛”和“扩散到终点”混成同一件事。

### 2.3 DSB 究竟证明了什么

DSB 的理论必须分开阅读：

- Theorem 1 是 score-based generative model 的采样误差界。它假设 uniform score
  error，并要求数据密度的强光滑和尾部条件；它不是 learned DSB convergence
  theorem。
- Propositions 4--5 研究 exact IPF iterates 的单调性与收敛，并使用额外的
  identification assumptions。
- Proposition 6 在有限熵和扩散正则性条件下，把 continuous exact IPF iterates
  表示成前向与反向 SDE。
- Algorithm 1 再以 Gaussian step、有限数据、网络和梯度下降近似这些对象。

因此合理的结论是：DSB 的外层结构受到 exact IPF 启发，population regression
targets 有明确推导；有限神经算法没有从 exact-IPF theorem 自动获得收敛保证。
论文还明确提醒，若把 forward/backward networks 联合训练，平衡点可能给出一个
连接两端的 bridge，却不一定是相对于指定 reference 的 Schrödinger Bridge。

## 3. Maximum-Likelihood SB：把 reverse drift 当作统计估计问题

Vargas 等人的 IPML 从另一条路线处理 half-bridge。给定一批由当前扩散产生的
连续时间路径，time reversal 决定了反向 drift；于是一次 half-step 可以通过最大
似然拟合反向扩散完成。更新另一端后，再交换方向并重复。

它的重要优点是可以处理 transition density 没有解析式的一般 diffusion prior。
代价是，算法正确性取决于三件不同的事：

1. exact half-bridge 的 disintegration；
2. reverse-drift MLE 的统计一致性；
3. 实际 Euler--Maruyama 样本与连续扩散样本之间的离散化误差。

### 3.1 2023 correction 为什么必须进入主线

原论文 Observation A1 的证明曾错误地说 KL 趋于零。对路径律 `P,Q`，正确分解是

```text
KL(P||Q)
= KL(P_0||Q_0) + E_{P_0} KL(P(.|X_0)||Q(.|X_0)).   (3.1)
```

half-bridge 令 conditional path law 与 reference 相同，只使第二项为零。第一项
仍是端点边缘比值的 KL，并恰好等于 half-bridge minimum。2023 correction
(`10.3390/e25020289`) 修正了这句话，并删除 Lemma A3 中 Equation (A45) 后一条
不清楚的句子。

这个修正不推翻 half-bridge 方法，反而澄清了它为何成立：最优值通常不是零，零掉的
只是条件路径部分。

### 3.2 一致性的精确范围

原论文 Theorem 1 的 reverse-MLE consistency 使用 exact discretized-SDE samples、
compactness/continuity 条件，以及样本数增大和时间步趋零的极限。Appendix D 证明
Euler scheme 的 time-reversal path convergence，但作者明确指出，这还没有证明
对 Euler 样本计算的 MLE 一致。把 stochastic integral 的收敛补齐仍是开放问题。

Appendix F 的 finite-sample IPFP 部分同样不能当作 theorem。它自称 rough sketch，
并把结果标为 Conjecture；其中直接假设 approximate projection 是 Lipschitz，且每轮
projection error 有统一上界。对论文采用的 Gaussian-process drift estimator，这些
假设没有在文中证明。

因此 IPML 的强项是清晰的统计建模路线和对一般 prior 的适应性，而不是已经闭合的
finite-sample neural-IPFP convergence theory。

## 4. SB-FBSDE：从 coupled PDE 到 likelihood identity

B6 已经说明，经典 SB 的正向和反向势函数满足 coupled linear PDE 与 split endpoint
conditions。SB-FBSDE 的贡献是把这组 PDE 沿随机轨迹局部表示，并导出可训练的
likelihood objective。

### 4.1 classical factor 到 FBSDE

设 reference diffusion 为

```text
dX_t = f(t,X_t) dt + g(t) dW_t.                     (4.1)
```

若正势 `Psi` 和反势 `hat_Psi` 是满足 coupled PDE 与两端乘积边界的正经典解，最优
forward/backward drift correction 分别为

```text
g^2 grad log Psi,        -g^2 grad log hat_Psi.      (4.2)
```

论文 Theorem 3 令

```text
Y_t=log Psi(t,X_t),       Z_t=g grad log Psi(t,X_t),
hat_Y_t=log hat_Psi(t,X_t),
hat_Z_t=g grad log hat_Psi(t,X_t),                   (4.3)
```

并在 nonlinear Feynman--Kac 的正则条件下得到 coupled FBSDE。关键边界仍然来自
Schrödinger system，而不是任意给定一个普通 terminal cost。`Y_t+hat_Y_t` 沿最优
路径等于最优中间边缘的 log density。

### 4.2 likelihood equality 与 parameterized objective

Theorem 4 在 exact FBSDE solution 下给出 `log p_0^SB(x_0)` 的 stochastic integral
identity。它把 forward/backward policies 的二次项、divergence 和 reference drift
组合起来。将 `Z,hat_Z` 替换成 networks 后，可以构造 parameterized likelihood
lower bound 或 training objective。

逻辑方向是

```text
classical Schrödinger factors
  -> exact FBSDE solution
  -> exact likelihood identity
  -> parameterized stochastic objective.             (4.4)
```

最后一步不是等价箭头。对任意有限网络，exact likelihood equality 不再自动成立。
还必须处理 function approximation、optimizer、Monte Carlo、divergence estimator
和 SDE discretization。

### 4.3 joint 与 alternating training

论文给出 joint likelihood training 和 alternating/IPF-style training。Appendix
认为 classical IPF convergence analysis 可较容易迁移到 alternating scheme，但把
正式分析留作 future work。这句话是研究方向，不是已证明定理。

SB-FBSDE 的价值在于把 SB optimality 与 likelihood training 建立结构联系，并揭示
SGM 是某些 forward control 退化时的特例。它不证明任何有限网络训练都命中两端
分布或达到 path-space KL optimum。

## 5. DeepGSB：mean-field extension，不是任意 generalized SB

Deep Generalized Schrödinger Bridge 处理带 mean-field interaction 的随机控制和
mean-field games。其端点分布仍是硬约束，但 drift/cost 可以依赖当前群体分布，
从而得到 HJB 与 Fokker--Planck 的 mean-field coupling。

Theorem 2 在 classical regularity 下把 generalized factor system 写为 FBSDE。
Propositions 3--4 引入 temporal-difference objectives，并在 function/FBSDE level
给出 combined objectives 最小化与目标系统成立之间的 characterization。这里的
“最小化”是 population function-level 条件，不表示有限 neural optimizer 一定找到
global minimum。

Appendix A.4.2 给出最重要的边界：加入用于强制 mean-field structure 的 TD
objectives 后，combined alternating objective 不再直接对应 standard IPF。论文用
TRPO 作算法类比，但没有证明 DeepGSB Algorithm 1 的 convergence。

因此本教程把 DeepGSB 称为“mean-field generalized SB-FBSDE solver”。我们不把它
扩张成任意 reference process、任意 path cost、manifold/reflection 或任意 state
constraint 的统一定理。这些推广属于 B11。

## 6. 四条路线的责任对照

| 方法       | exact baseline                | population object                 | learned object                           | 当前理论边界                                             |
| -------- | ----------------------------- | --------------------------------- | ---------------------------------------- | -------------------------------------------------- |
| DSB      | path-law IPF                  | Gaussian mean matching            | alternating drift/mean networks          | exact IPF convergence 不覆盖 finite Algorithm 1       |
| IPML     | half-bridge IPFP              | reverse-drift MLE                 | GP/parametric drift fitting              | EM-sample MLE consistency 与 finite-sample IPFP 未闭合 |
| SB-FBSDE | coupled SB factors/PDE        | exact FBSDE likelihood identity   | joint 或 alternating policy networks      | finite neural convergence 留作 future work           |
| DeepGSB  | mean-field stochastic control | generalized FBSDE + TD identities | alternating neural policies/value fields | combined objective 非标准 IPF，未证算法 convergence        |

比较方法时，先问“它逼近哪一个 exact object”，再问“论文证明到哪一层”。仅比较
网络结构或 FID，会遗漏 reference dynamics、双端约束和 path-law objective 的差别。

## 7. 五层误差栈

神经 SB 的误差至少分为：

```text
outer finite iteration
+ function approximation
+ optimization
+ Monte Carlo/generalization
+ time discretization
+ initialization/reference mismatch.                (7.1)
```

这里的加号只是分类，不是已经证明的 additive bound。

### 7.1 outer finite iteration

即使每个 half-step 都精确，有限轮 IPF 通常仍未到 fixed point。需要用 exact-IPF
theorem 指定 topology、rate 或 residual 的含义。

### 7.2 function 与 optimization error

Population target 可能不在网络类中；即使存在，非凸优化也未必找到 global optimum。
training loss 小只说明经验目标局部拟合良好。

### 7.3 Monte Carlo error

每轮 sampling law 依赖上一轮 fitted model，所以误差不是独立同分布地一次累积。
需要 stability 分析，不能把普通 supervised generalization bound 直接逐轮相加。

### 7.4 discretization error

连续 reverse-time identity、Euler paths 和离散 likelihood 是三个对象。路径强收敛不
自动推出含 stochastic integral 的 estimator consistency。

### 7.5 endpoint 与 path-law diagnostics

有限样本的端点 MMD、分类准确率或图像 FID 可以发现明显 mismatch，但它们不等于
path-space KL、IPF objective gap 或双端约束的严格证明。

## 8. 最小 Gaussian 验证

`references/code/bridge/b8_population_regression_checks.py` 不训练神经网络。它生成
affine Gaussian one-step samples，对 (2.2) 和 (2.4) 的 target 分别计算解析条件
期望与 Monte Carlo OLS，并确认偏离最优斜率会增大 MSE。

运行：

```powershell
python references/code/bridge/b8_population_regression_checks.py
```

固定 seed 的当前结果为：

```text
backward target: intercept_error=2.314e-04 slope_error=1.499e-03
forward target:  intercept_error=2.617e-04 slope_error=3.396e-04
B8 population regression checks passed
```

这个实验只验证 `L2` population target 的代数。它特意不声称 Gaussian step 是 exact
time reversal，也不声称交替 fitted affine maps 收敛到 SB。

![Forward/backward half-step 的 Gaussian population regression](/images/bridge/B8_population_regression.png)

**图 2.1：** 两个方向的 sample targets 与其 conditional-mean population OLS；
虚线故意扰动 slope，产生更高 MSE。图验证 regression target，不承担 finite network、
finite sample outer iteration 或 discretized sampler 的 convergence guarantee。

## 9. 常见错误与概念边界

### 错误 1：把 Proposition 3 叫作 DSB convergence theorem

Proposition 3 标识 population regression target。exact IPF convergence 和 finite
network convergence 是另外两层。

### 错误 2：half-bridge 的 KL minimum 等于零

条件路径 KL 可以为零，端点边缘 KL 通常仍在。ML-SB 2023 correction 专门修正了
这一点。

### 错误 3：Euler path convergence 推出 reverse-MLE consistency

MLE 还包含 likelihood/stochastic-integral functional。连续映射或一致可积性条件需要
独立证明。

### 错误 4：FBSDE identity 对任意网络都成立

exact identity 以 exact FBSDE solution 为前提；network substitution 产生训练目标，
不是免费得到的等式。

### 错误 5：DeepGSB 已证明 standard IPF convergence

论文恰恰指出 TD-combined objective 不直接等于 standard IPF。

### 错误 6：方法名含 Bridge 就已满足双端约束

必须检查有限算法输出是否只近似一轮 projection、只匹配 one-time marginals，还是
真的求解指定 reference 下的 path-space KL problem。

## 10. 小结

高维神经 SB 的核心不是抛弃 classical theory，而是选择可学习的 exact-object
representation：DSB 学 conditional mean，IPML 学 reverse drift，SB-FBSDE 学
forward/backward policies，DeepGSB 再加入 mean-field TD structure。

它们共同揭示了一条技术演进：

```text
stored potentials
  -> sampled paths
  -> population drift/score/policy identities
  -> finite neural approximations.                   (10.1)
```

每向右一步，计算更可扩展，但证明责任也增加。B9 将研究另一条路线：不完整执行每次
IPF，而是在 reciprocal projection 与 Markovian projection 之间做 bridge matching。
那里“一次 matching”和“得到 Schrödinger Bridge”的距离必须继续保持可见。

## 11. 研究式思考题

1. 若每轮 population regression error 都小于 `epsilon_n`，还需要哪一种 operator
   stability，才能把局部误差转化为 outer-iteration error？
2. 构造一个同时具有正确 `t=0,T` marginals、但相对于给定 reference 并非最小 KL
   的 path law。它说明 endpoint diagnostics 缺少什么？
3. ML-SB 中 Euler scheme 的强收敛为什么不足以控制 Girsanov likelihood？尝试列出
   对 stochastic integral 使用连续映射定理所缺的条件。
4. 比较 SB-FBSDE joint training 与 alternating training。哪一种更接近 exact IPF，
   哪一种可能更容易优化？这两件事为何不等价？
5. DeepGSB 的 mean-field interaction 若置零，哪些 FBSDE/TD 项退化？这能否单独
   证明有限训练恢复 classical SB？

## 12. 来源与进一步阅读

- De Bortoli et al. (2021), DSB：exact IPF representation、mean matching 与
  neural algorithm；
- Vargas et al. (2021, corrected 2023)：reverse MLE、IPML 和有限样本边界；
- Chen, Liu and Theodorou (2022)：SB-FBSDE 与 likelihood identity；
- Liu et al. (2022)：mean-field DeepGSB；
- B4：two-sided transform 与 time reversal；
- B6：stochastic control、Girsanov、HJB/Hopf--Cole；
- B7：exact IPF/Sinkhorn 与 convergence responsibility；
- Diffusion D3/D4：conditional-expectation regression 与 reverse-time SDE 接口。
