---
title: 理论保证、评估、记忆与安全边界
description: 审视 score、离散化与分布误差的保证链，并比较 FID、precision-recall、记忆、隐私和安全证据。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 12
slug: d12-theory-evaluation-safety
tags:
  - diffusion
  - theory
  - safety
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦 theorem 假设、metric 偏差和系统风险，不由单一 loss、指标或近邻检查推出完整质量与安全结论。
---
## 1. 这一章问的不是“Diffusion 有没有理论”

训练结束时，我们通常拥有一条下降过的 loss 曲线、若干漂亮样图和一组 FID。最诱人的叙述是：loss 低，所以 score 准；score 准，所以反向过程准；反向过程准，所以样本分布准；FID 低，所以模型好；没搜到训练近邻，所以模型没有记忆；加了 safety checker，所以系统安全。

这条推理链的每一个“所以”都缺少条件。

本章的核心不是罗列 theorem、metric 和风险论文，而是建立一套边界意识：

1. **理论边界**：训练目标究竟控制哪个数学对象，误差怎样传播到生成分布？
2. **评估边界**：一个有限样本统计量究竟看见了质量、覆盖、条件一致性还是系统成本？
3. **记忆边界**：memorization、copying、extraction 和 membership inference 各自在问什么？
4. **安全边界**：偏差审计、concept erasure、水印和模型卡分别能提供哪一层证据？

一句话概括：**任何结论都要同时写出对象、假设、距离、协议和证据等级。**

## 2. 三层阅读路线

这一章可以按三种深度阅读。

**直觉主线**只需抓住四张表：error-source matrix、metric matrix、memorization/privacy distinction、safety evidence layers。读完后，你应当不再用一个数字代替整个系统。

**完整推导主线**会从反向 SDE 的 drift difference 出发，用 Girsanov、data processing 和 Pinsker 建立 score error 到 marginal TV 的代表性链条；再独立推导 empirical FID bias 与 unbiased MMD。

**研究主线**比较 2022--2026 年代表性 convergence 结果的 target、score access、sampler、metric、endpoint 和 complexity，并把正式论文与前沿预印本分开。

## 3. 问题驱动的历史：每个新指标都在修补上一个盲区

| 时期         | 暴露的问题                               | 代表性回应                                              | 新留下的边界                                        |
| ---------- | ----------------------------------- | -------------------------------------------------- | --------------------------------------------- |
| 2017       | 只看样图或 Inception Score 难比较分布         | FID 用 feature 均值/协方差给出标量                           | Gaussian/moment 假设与 finite-sample bias        |
| 2018--2019 | 单一距离混合 fidelity 与 coverage          | KID/MMD、precision/recall                           | kernel、embedding、kNN 密度依赖                     |
| 2022       | resize/compression 足以改变 FID 排名      | CleanFID 固定预处理与实现                                  | 只修复 protocol，不修复 metric 定义                    |
| 2022--2024 | 早期 Diffusion 理论依赖强光滑/凸性条件           | general-data、manifold、near-d-linear convergence    | population score access、early stop、特定 sampler |
| 2023--2024 | text-to-image 不能只看无条件 image feature | HEIM、CMMD、VBench、领域有效性评测                           | benchmark coverage 与 human protocol 仍有限       |
| 2023--2025 | 高质量输出掩盖训练数据记忆和社会风险                  | extraction、membership、bias、unsafe-generation audit | 攻击模型与 detector-dependent 结论                   |
| 2023--2024 | “删除概念”“加水印”被当作安全完成                  | ESD、Ring-A-Bell、Stable Signature                   | suppression、robustness、provenance 不是同一目标      |
| 2024--2026 | 高维最坏界与现实低维结构、局部记忆不吻合                | intrinsic-dimension、PF-ODE、local-coverage frontier | 多项结果仍是预印本或 stylized model                     |

这是一条“问题--解决方案--新盲区”的路线，而不是模型排行榜。

## 4. 先固定七类误差，不让一个 loss 承担全部责任

![Diffusion 从数据到系统的误差源](/images/diffusion/d12_error_source_matrix.png)

从数据分布到最终系统报告，至少经过七个不同接口：

| 误差源                     | 比较对象                                   | 常用控制量                            | 不属于这一项的内容                |
| ----------------------- | -------------------------------------- | -------------------------------- | ------------------------ |
| terminal initialization | 前向终点 $p_T$ 与先验 $\pi$                   | KL、TV、Wasserstein                | score 学习误差               |
| score estimation        | $s_\theta(\cdot,t)$ 与 $\nabla\log p_t$ | time-weighted population $L^2$   | empirical generalization |
| early stopping          | $p_\delta$ 与 $p_0$                     | $W_1/W_2$、弱收敛                    | 奇异端点的 TV 收敛              |
| reverse discretization  | 连续反向 law 与离散 Markov chain              | path KL、global solver error      | network approximation    |
| finite data             | empirical risk 与 population risk       | generalization/sample complexity | optimization 成功          |
| network approximation   | 函数类最优与真实 score                         | approximation norm               | finite precision         |
| implementation/protocol | 数学算法与实际软件/评测                           | checksum、版本、置信区间                 | metric 的统计充分性            |

本章把理论文献中的 **time truncation** 具体记为 early stopping：反向过程只运行到 $t=\delta>0$，比较 $p_\delta$ 与 $p_0$。它不同于 terminal initialization 的有限 $T$，也不同于把连续反向过程离散成有限步。三者分别记为 $\mathcal E_{\mathrm{stop}}$、$\mathcal E_{\mathrm{init}}$ 与 $\mathcal E_{\mathrm{disc}}$，不能共同塞进“采样误差”。

我们可以把它们写成概念账本

$$
\mathcal E_{\mathrm{total}}
\leadsto
\mathcal E_{\mathrm{init}}
+\mathcal E_{\mathrm{score}}
+\mathcal E_{\mathrm{stop}}
+\mathcal E_{\mathrm{disc}}
+\mathcal E_{\mathrm{finite}}
+\mathcal E_{\mathrm{approx}}
+\mathcal E_{\mathrm{impl}},
$$

但这里故意使用 $\leadsto$ 而不是等号或 $\le$。只有为每一项选定兼容的距离并证明三角不等式、data processing 或 coupling 后，这才会成为严格 bound。

## 5. 训练损失低，到底已经知道了什么

以连续时间 denoising score matching 为例，理想 population risk 是

$$
\mathcal L_{\mathrm{pop}}(\theta)
=
\int_\delta^T
\lambda(t)
\mathbb E_{X_t\sim p_t}
\left[
\|s_\theta(X_t,t)-\nabla\log p_t(X_t)\|_2^2
\right]dt.
$$

实际训练优化的却是有限数据、有限时间采样和有限 minibatch 下的估计

$$
\widehat{\mathcal L}_n(\theta)
=
\frac1n\sum_{i=1}^n
\widehat\ell_\theta(x_0^{(i)},t_i,\epsilon_i).
$$

训练曲线低，直接说明的只是 optimizer 找到了一个使当前 empirical estimator 较小的参数。要得到 population score error，还需要处理

$$
\mathcal L_{\mathrm{pop}}(\widehat\theta)
-\inf_{\theta\in\Theta}\mathcal L_{\mathrm{pop}}(\theta),
$$

其中至少混合

$$
\underbrace{\mathcal E_{\mathrm{generalization}}}_{\text{finite data}}
+\underbrace{\mathcal E_{\mathrm{optimization}}}_{\text{未达到经验最优}}
+\underbrace{\mathcal E_{\mathrm{approximation}}}_{\text{函数类不含真实 score}}.
$$

再往后，population score error 仍只是反向过程误差的一项。

## 6. 分布距离不是可互换的单位

后面的 theorem 会使用 KL、TV 和 Wasserstein。先固定定义。

相对熵为

$$
\operatorname{KL}(P\|Q)
=
\begin{cases}
\displaystyle\int \log\frac{dP}{dQ}\,dP,&P\ll Q,\\
+\infty,&\text{otherwise}.
\end{cases}
$$

总变差采用

$$
\operatorname{TV}(P,Q)
=\sup_A|P(A)-Q(A)|
=\frac12\|p-q\|_1
$$

这一 convention。二阶 Wasserstein 为

$$
W_2(P,Q)^2
=\inf_{\gamma\in\Pi(P,Q)}
\mathbb E_{(X,Y)\sim\gamma}\|X-Y\|_2^2.
$$

三者回答的问题不同：

- KL 强，需要绝对连续并对 low-probability mismatch 敏感；
- TV 控制所有有界事件，但对奇异支撑极其严格；
- Wasserstein 允许两个分布支撑不同，强调搬运几何和 coupling。

所以不能只看“大 O 更小”。一个 $W_2$ 结论与一个 TV 结论可能根本不在同一问题上。

## 7. 路径分布、转移核和单时刻边缘再次分开

与 [D4](/blog/diffusion/d4-continuous-time-sde/) 一样，本章同时出现三类对象：

$$
\mathbb P=\operatorname{Law}\bigl((X_t)_{t\in[\delta,T]}\bigr),
\qquad
P_{s,t}(x,\cdot),
\qquad
p_t=\operatorname{Law}(X_t).
$$

路径 KL 控制整条轨迹；终点 marginal 是路径的投影。由 data processing，路径误差可以控制终点误差，反过来通常不成立。两个过程可以拥有相同的一时刻 marginals，却有完全不同的 path law，这也是 Diffusion 与 Schrödinger Bridge 交界处必须保持的边界。

## 8. 代表性推导的起点：前向 SDE 与真实反向 drift

考虑 scalar diffusion coefficient 的前向过程

$$
dX_t=f(X_t,t)\,dt+g(t)\,dW_t,
\qquad X_t\sim p_t.
$$

只在 $t\in[\delta,T]$ 上讨论反向，其中 $\delta>0$ 暂时避开可能奇异的 data endpoint。令正向参数化的反向时间为

$$
\tau=T-t.
$$

真实反向过程 $Y_\tau=X_{T-\tau}$ 的 drift 是

$$
b^*(x,\tau)
=-f(x,T-\tau)
+g(T-\tau)^2\nabla_x\log p_{T-\tau}(x).
$$

用学到的 score 替换真实 score，得到

$$
b^\theta(x,\tau)
=-f(x,T-\tau)
+g(T-\tau)^2s_\theta(x,T-\tau).
$$

两条过程共享 diffusion coefficient $g(T-\tau)$。这是下一步使用 Girsanov 的关键结构。

## 9. Girsanov 之前必须列出的假设

令 $\mathbb P^*$ 是真实反向 path law，$\mathbb P^\theta$ 是近似反向 path law。为了写出二者的 Radon--Nikodym derivative，至少需要：

1. 两个过程在同一时间区间上有共同、非退化的 diffusion coefficient；
2. 初始分布满足相应绝对连续关系；
3. drift difference 经 diffusion coefficient 归一化后平方可积；
4. Novikov 或其他足以保证指数鞅成立的条件；
5. SDE 解的存在性与所用反向时间定理条件。

不能看到两个 drift 相减，就直接把平方积分称作 path KL。

## 10. 归一化 drift difference 为什么带一个 $g(t)$

drift difference 为

$$
b^*(x,\tau)-b^\theta(x,\tau)
=g(T-\tau)^2
\left[
\nabla\log p_{T-\tau}(x)-s_\theta(x,T-\tau)
\right].
$$

Girsanov 使用的是 diffusion-normalized difference：

$$
u(x,\tau)
=
\frac{b^*(x,\tau)-b^\theta(x,\tau)}{g(T-\tau)}
=g(T-\tau)\bigl(s_{T-\tau}(x)-s_\theta(x,T-\tau)\bigr),
$$

其中

$$
s_t(x)=\nabla_x\log p_t(x).
$$

这解释了为什么 path-space 误差中出现 $g(t)^2$ 加权，而不是任意选一个训练权重都能得到同一个 theorem。

## 11. 从 Radon--Nikodym derivative 到 path KL

若真实反向过程从 $p_T$ 开始，近似过程从采样先验 $\pi$ 开始，链式 KL 分成初值和条件路径两部分：

$$
\operatorname{KL}(\mathbb P^*\|\mathbb P^\theta)
=\operatorname{KL}(p_T\|\pi)
+\mathbb E_{X_T\sim p_T}
\operatorname{KL}\bigl(\mathbb P^*(\cdot\mid X_T)\|
\mathbb P^\theta(\cdot\mid Y_0=X_T)\bigr).
$$

Girsanov 给出条件路径部分

$$
\operatorname{KL}\bigl(\mathbb P^*(\cdot\mid X_T)\|
\mathbb P^\theta(\cdot\mid Y_0=X_T)\bigr)
=\frac12
\mathbb E_{\mathbb P^*}
\int_0^{T-\delta}\|u(Y_\tau,\tau)\|_2^2d\tau.
$$

换回正向时间 $t=T-\tau$：

$$
\boxed{
\operatorname{KL}(\mathbb P^*\|\mathbb P^\theta)
=\operatorname{KL}(p_T\|\pi)
+\frac12\int_\delta^T
g(t)^2
\mathbb E_{X_t\sim p_t}
\|s_\theta(X_t,t)-s_t(X_t)\|_2^2dt
}.
$$

注意 expectation 在真实 $p_t$ 下。若改成模型 marginal，KL 方向与证明都要重做。

## 12. 从 path KL 到 marginal KL

取终点投影

$$
F:(Y_\tau)_{\tau\in[0,T-\delta]}\mapsto Y_{T-\delta}.
$$

路径 law 经 $F$ 的 pushforward 分别是真实 $p_\delta$ 和近似 $q_\delta^\theta$。KL 的 data-processing inequality 给出

$$
\operatorname{KL}(p_\delta\|q_\delta^\theta)
\le
\operatorname{KL}(\mathbb P^*\|\mathbb P^\theta).
$$

这是“整条路径近，终点也近”；它没有说终点近能恢复正确路径，也没有加入数值离散误差。

## 13. Pinsker 给出代表性 TV bound

在本章的 TV convention 下，Pinsker 为

$$
\operatorname{TV}(P,Q)
\le
\sqrt{\frac12\operatorname{KL}(P\|Q)}.
$$

代入上一节得到

$$
\boxed{
\operatorname{TV}(p_\delta,q_\delta^\theta)
\le
\sqrt{
\frac12\operatorname{KL}(p_T\|\pi)
+\frac14\int_\delta^T
g(t)^2
\mathbb E_{p_t}\|s_\theta-s_t\|_2^2dt
}
}.
$$

这条式子非常有教育意义，因为它清楚列出：

- terminal initialization；
- continuous-time population score error；
- data-processing 后的 marginal error。

同样重要的是，它**没有**自动包含 empirical generalization、network approximation、early-stop $p_0\leftrightarrow p_\delta$ 和数值 discretization。

## 14. 为什么“$T$ 足够大”仍不是零初始化误差

以 VP/OU forward process 为例，随着 $T$ 增大，$p_T$ 逐渐接近标准 Gaussian，但有限 $T$ 一般只有

$$
\operatorname{KL}(p_T\|\pi)>0.
$$

初始化项取决于 forward mixing、data tails 和所选距离。代码里从 $\pi$ 采样并不等于理论上已经从 $p_T$ 采样。把 $p_T=\pi$ 当作精确等式，只能是显式模型假设或极限简化。

## 15. Early stopping 的 Wasserstein coupling

现在处理上面故意留下的 $p_0\leftrightarrow p_\delta$。先看最简单的 VE smoothing：

$$
X_\delta=X_0+\sigma_\delta Z,
\qquad Z\sim\mathcal N(0,I_d).
$$

使用这一本身就是合法 coupling 的联合分布：

$$
W_2(p_0,p_\delta)^2
\le
\mathbb E\|X_\delta-X_0\|_2^2
=\sigma_\delta^2\mathbb E\|Z\|_2^2
=d\sigma_\delta^2.
$$

因此

$$
\boxed{W_2(p_0,p_\delta)\le\sqrt d\,\sigma_\delta}.
$$

这说明小噪声在 Wasserstein 意义下可以回到数据分布。对于 VP channel，还会多出 signal attenuation 引起的 coupling term。

## 16. 流形端点下，TV 可以永远等于 1

设 $p_0$ 完全集中在低维流形 $\mathcal M\subset\mathbb R^d$ 上：

$$
p_0(\mathcal M)=1.
$$

只要 $\sigma_\delta>0$，Gaussian smoothing 后的 $p_\delta$ 对 Lebesgue 测度有密度，而低维流形的 Lebesgue 测度为零，于是

$$
p_\delta(\mathcal M)=0.
$$

因此

$$
\operatorname{TV}(p_0,p_\delta)
\ge|p_0(\mathcal M)-p_\delta(\mathcal M)|=1.
$$

又因为 TV 最大值就是 1，得到

$$
\boxed{\operatorname{TV}(p_0,p_\delta)=1,\qquad \forall\delta>0.}
$$

这与 $W_2(p_0,p_\delta)\to0$ 不矛盾。它正是 De Bortoli 的 manifold 路线必须改用 Wasserstein/early stopping 的原因之一。所谓“换一个 metric”不是美学选择，而是问题是否可解的边界。

## 17. 数值离散必须另开一个证明接口

设连续近似反向终点为 $q_\delta^\theta$，实际步长 $h$ 的 solver 产生 $q_{\delta,h}^\theta$。只有在同一 metric $d(\cdot,\cdot)$ 下，才能写

$$
d(p_\delta,q_{\delta,h}^\theta)
\le
d(p_\delta,q_\delta^\theta)
+d(q_\delta^\theta,q_{\delta,h}^\theta).
$$

第二项依赖：

- drift/score 的 Lipschitz 或高阶导数；
- 时间网格和 endpoint stiffness；
- Euler、exponential integrator、Runge--Kutta 或 multistep solver；
- stochastic 与 deterministic sampler；
- local error 如何累积为 global distribution error。

[D6](/blog/diffusion/d6-sampling-solvers/) 讨论算法；本章只强调：连续理论不能自动给实际 DPM-Solver、Heun 或自适应 solver 背书。

## 18. Finite data、函数类和 optimizer 的接口

令函数类为 $\mathcal F$，population 最优为

$$
s_{\mathcal F}^*\in\arg\min_{s\in\mathcal F}\mathcal L_{\mathrm{pop}}(s).
$$

实际参数为 $\widehat s$。用平方范数的基本不等式，可作概念分解

$$
\|\widehat s-s_t\|^2
\le
2\|\widehat s-s_{\mathcal F}^*\|^2
+2\|s_{\mathcal F}^*-s_t\|^2.
$$

右边第一项还包含 empirical-to-population generalization 与 optimization；第二项是 approximation/misspecification。即使训练集 risk 为零，第二项也可能非零；即使函数类足够大，有限样本和 optimizer 也可能没有找到 population score。

## 19. 如何读一条 convergence theorem

每条 theorem 至少填完下表才可比较：

| 字段            | 必问问题                                                         |
| ------------- | ------------------------------------------------------------ |
| target        | 任意概率测度、光滑密度、compact support、tails 还是 manifold？               |
| score access  | exact、population $L^2$、uniform error、导数控制还是 learned network？ |
| sampler       | reverse SDE、DDPM chain、PF ODE 或某个命名 discretization？          |
| metric        | KL、TV、$W_1/W_2$ 还是 weak convergence？                         |
| endpoint      | $t=0$ 还是 $\delta>0$ 的 smoothed target？                       |
| complexity    | dimension、accuracy、early-stop、time horizon 和 log factors？    |
| omitted layer | finite data、optimizer、architecture、runtime 中还缺什么？            |

只抄 abstract 中的 $\widetilde O(\cdot)$，通常会丢掉最重要的信息。

## 20. 2022--2023：从强正则假设走向一般数据

Lee--Lu--Tan（补充材料暂未公开）（ALT 2023）研究一般数据分布，在 $L^2$-accurate score estimates 下，分别对有界支撑/足够快尾部给 Wasserstein 保证，并在进一步正则条件下给 TV 保证。

这类结果解决的问题是：为什么不必假设数据 log-concave 或满足很强 functional inequality，score-to-sampling 仍可能是 polynomially controlled。

它不解决：

- 网络如何用有限图像样本达到所需 population $L^2$ error；
- 现实 sampler 是否就是 theorem 的 discretization；
- $t=0$ 奇异数据是否在 TV 中可恢复。

工作区另有 Chen et al. 的 Sampling is as easy as learning the score（补充材料暂未公开），当前按预印本承担 minimal-data-assumption 路线，不把 source status 隐去。

## 21. Manifold theory：early stopping 不是技术尘埃

De Bortoli（补充材料暂未公开）（TMLR）针对目标可能没有 Lebesgue density 的情形，包括低维 manifold 或 empirical distribution，给出 Wasserstein convergence 分析。

核心变化不是“把旧证明中的 smoothness 删掉”，而是：

1. $t>0$ 的 forward-noised marginal 有 density 和 score；
2. $t=0$ 目标可能没有普通 score；
3. 反向过程先逼近 $p_\delta$；
4. 再用 Wasserstein smoothing/coupling 控制 $p_\delta\to p_0$。

这条路线明确告诉我们，endpoint 和 metric 是 theorem 的组成部分。

## 22. 2024：近 $d$-linear reverse-SDE complexity

Benton et al.（补充材料暂未公开）（ICLR 2024）在仅有限二阶矩等弱条件下，用改进的反向 SDE discretization 分析和 stochastic-localization 思路，给出近维度线性的步数依赖。

论文摘要中的代表性表述是：对加了方差 $\delta$ Gaussian noise 的任意 $\mathbb R^d$ 分布，为达到 $\varepsilon^2$ KL 精度，步数至多形如

$$
\widetilde O\!\left(
\frac{d\log^2(1/\delta)}{\varepsilon^2}
\right).
$$

安全读法是：

- target 是 smoothed distribution；
- score 假设是 $L^2$-accurate；
- metric 是指定方向的 KL；
- complexity 是理论 scheme 的 step bound；
- log factors、early-stop $\delta$ 和 moment normalization 不能删除。

它不等于“实际 20-step sampler 已被证明最优”。

## 23. Minimax theory：把生成建模视为分布估计

Oko--Akiyama--Suzuki（补充材料暂未公开）（ICML 2023）问的不是只给一个 fixed-score sampler bound，而是：在给定 smoothness/function class 与有限样本下，Diffusion 作为 distribution estimator 能否达到 minimax rate。

这一视角补上 error matrix 中的 finite-data 接口，但仍需逐项读取：

- 数据分布所属函数类；
- score network approximation rate；
- training estimator 与 sample size；
- 输出用何种 distribution metric 衡量。

“minimax optimal”永远是相对于一个统计模型类、损失和 asymptotic regime，不是对所有自然数据的无条件最高评价。

## 24. Probability-flow ODE 需要自己的理论

反向 SDE 和 PF ODE 共享 marginals 的精确结论，不意味着学到近似 score 后二者误差自动相同。Huang--Huang--Lin（补充材料暂未公开）（IEEE Transactions on Information Theory, 2025）专门分析 deterministic PF ODE。

在其 compact support、time-integrated $L^2$ score error 与 score-derivative 条件下，连续层 TV bound 形如

$$
\operatorname{TV}(p_0,\widehat p_0)
\lesssim d^{3/4}\eta_s^{1/2},
$$

其中 $\eta_s$ 表示相应 score-matching error。对 $p$ 阶 Runge--Kutta、步长 $h$，离散层增加形如

$$
\operatorname{TV}(p_0,\widehat p_{0,h})
\lesssim
d^{3/4}\eta_s^{1/2}
+d(dh)^p.
$$

这里最重要的不是背常数，而是看到 deterministic transport 对 score derivatives 和 numerical integrator order 提出了额外责任。

## 25. 2025--2026 前沿：内在维度、离散率与局部记忆

截至 2026-07-16，以下结果保留为前沿来源而非统一定论：

- Azangulov--Deligiannidis--Rousseau（补充材料暂未公开） 研究高维 manifold hypothesis 下的 convergence 与 intrinsic structure，catalog 中仍标为 preprint；
- Li--Yan（补充材料暂未公开） v2 预印本声称在 minimal assumptions 下得到忽略 log factor 的 $O(d/T)$ DDPM convergence；
- Merger--Goldt（补充材料暂未公开） 2026 年 6 月 v1 预印本用 local coverage 解释同一模型中 memorized 与 novel samples 共存。

它们适合承担“研究正在向哪里走”，不适合承担“社区已经一致确认”。

## 26. 理论演进图：真正变化的是假设与结论对象

![2022--2026 Diffusion 理论演进](/images/diffusion/d12_theory_timeline.png)

图中每个节点都应按七字段协议阅读。正式论文与预印本的颜色和标签不同，是为了防止时间新近性被误当成证据强度。

## 27. 理论部分小结：一条可复用的 proof map

本章的代表性链条可以压缩成

$$
\text{population score error}
\xrightarrow{\text{Girsanov}}
\text{path KL}
\xrightarrow{\text{data processing}}
\text{marginal KL}
\xrightarrow{\text{Pinsker}}
\text{marginal TV}.
$$

旁边还必须并行保留

$$
p_0\xleftrightarrow[\text{early stop}]{W_2}p_\delta,
\qquad
q_\delta^\theta\xleftrightarrow[\text{solver error}]{}q_{\delta,h}^\theta,
$$

以及 empirical-to-population 的统计学习接口。少任何一条，就不能从训练 loss 走到现实系统。

## 28. Likelihood/ELBO 是评估，但不是唯一答案

若模型定义了 normalized likelihood，可以在 held-out data 上报告

$$
-\mathbb E_{x\sim p_{\mathrm{test}}}\log p_\theta(x).
$$

DDPM 常通过 VLB/ELBO 给上界或估计；PF ODE 可通过 instantaneous change of variables 估计 likelihood。[D2](/blog/diffusion/d2-ddpm-objective/) 和 [D4](/blog/diffusion/d4-continuous-time-sde/) 已推导这些接口。

边界包括：

- dequantization 与 data preprocessing 会改变 bits/dim；
- ELBO 不等于 exact likelihood；
- likelihood 高不保证感知质量或条件一致性；
- manifold data 的 density 依赖支配测度和噪声模型；
- stochastic sampler quality 与 ODE likelihood 是不同对象。

因此 likelihood 是 distributional evidence 的一维，不是生成质量的最终定义。

## 29. FID 的 population 公式

Heusel et al.（补充材料暂未公开）（NeurIPS 2017）将真实和生成图像映射到 Inception feature，分别用 Gaussian

$$
\mathcal N(\mu_r,\Sigma_r),
\qquad
\mathcal N(\mu_g,\Sigma_g)
$$

近似。两个 Gaussian 的 squared $W_2$ 给出 FID：

$$
\boxed{
\operatorname{FID}
=\|\mu_r-\mu_g\|_2^2
+\operatorname{tr}
\left(
\Sigma_r+\Sigma_g
-2(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}
\right)
}.
$$

它看见的是**所选 feature 的一、二阶矩**。不同非 Gaussian feature distributions 可以有完全相同的 FID。

## 30. Empirical FID 为什么即使同分布也有正 bias

取一维最简单反例。真实和生成 feature 都是标准 Gaussian：

$$
X_i,Y_i\stackrel{\mathrm{iid}}{\sim}\mathcal N(0,1),
\qquad i=1,\ldots,n.
$$

population FID 为 0。plug-in estimator 为

$$
\widehat{\operatorname{FID}}
=(\bar X-\bar Y)^2+(S_X-S_Y)^2,
$$

其中 $S_X,S_Y$ 为样本标准差。第一项期望已经是

$$
\mathbb E(\bar X-\bar Y)^2
=\operatorname{Var}(\bar X)+\operatorname{Var}(\bar Y)
=\frac2n.
$$

第二项非负且通常严格为正，所以

$$
\boxed{
\mathbb E\widehat{\operatorname{FID}}
>\frac2n>0
}
$$

即使两个分布完全相同。样本数变化会改变 bias，两个模型的 bias 还未必相同，因此跨样本数 FID 排名没有公平性保证。

## 31. CleanFID 修复什么，不修复什么

Parmar--Zhang--Zhu（补充材料暂未公开）（CVPR 2022）展示 resize kernel、antialiasing 和 JPEG compression 足以显著改变 FID。官方实现已固定在

$$
\texttt{GaParmar/clean-fid@e88c4d6...}
$$

并保存为 本地代码快照（补充材料暂未公开）。

CleanFID 解决的是：

- feature preprocessing 的统一；
- resize/compression implementation 的可复现；
- reference statistics 与工具版本的固定。

它不解决：

- FID 只看一、二阶矩；
- Inception feature 与 text-to-image 语义不匹配；
- plug-in FID 的统计 bias；
- 条件一致性、公平性、安全和成本缺失。

## 32. MMD 与 KID 的 population 定义

对正定核 $k$，squared maximum mean discrepancy 为

$$
\operatorname{MMD}^2(P,Q)
=\mathbb E k(X,X')
+\mathbb E k(Y,Y')
-2\mathbb E k(X,Y),
$$

其中

$$
X,X'\stackrel{\mathrm{iid}}{\sim}P,
\qquad
Y,Y'\stackrel{\mathrm{iid}}{\sim}Q.
$$

Binkowski et al.（补充材料暂未公开）（ICLR 2018）的 KID 在 Inception feature 上使用 polynomial kernel，将 MMD 作为生成评估。

## 33. 无偏 MMD U-statistic 的完整推导

有限样本 $x_1,\ldots,x_m$ 与 $y_1,\ldots,y_n$ 下，去掉 within-sample diagonal：

$$
\widehat{\operatorname{MMD}}_u^2
=\frac1{m(m-1)}\sum_{i\ne j}k(x_i,x_j)
+\frac1{n(n-1)}\sum_{i\ne j}k(y_i,y_j)
-\frac2{mn}\sum_{i,j}k(x_i,y_j).
$$

因为 $i\ne j$ 时 $X_i,X_j$ 独立同分布，

$$
\mathbb E k(X_i,X_j)
=\mathbb E_{X,X'\sim P}k(X,X').
$$

所以

$$
\mathbb E
\left[
\frac1{m(m-1)}\sum_{i\ne j}k(X_i,X_j)
\right]
=\mathbb E k(X,X').
$$

对 $Q$ 项同理；cross term 的所有 $mn$ 对天然独立。于是

$$
\boxed{
\mathbb E\widehat{\operatorname{MMD}}_u^2
=\operatorname{MMD}^2(P,Q)
}.
$$

无偏不等于单次估计无误差。U-statistic 在有限样本下甚至可以为负，variance 也可能很大。

## 34. CMMD：更换 feature 与 kernel，而不是发现终极指标

Jayasumana et al.（补充材料暂未公开）（CVPR 2024）指出现代 text-to-image 中，Inception representation、Gaussian assumption 和 sample complexity 都会使 FID 失真，并提出 CLIP embedding 上 Gaussian-RBF MMD 的 CMMD。

它改善了 text/image semantic representation 与统计形式，但仍依赖：

- CLIP 的训练数据与偏差；
- RBF bandwidth；
- prompt distribution；
- sample size 和 uncertainty；
- image quality 与 text alignment 在 embedding 中如何混合。

所以“unbiased MMD + CLIP”仍是 proxy，不是 human preference、原创性或安全真值。

## 35. Precision/recall 为什么要把质量与覆盖拆开

Sajjadi et al.（补充材料暂未公开）（NeurIPS 2018）从 distribution decomposition 角度提出 precision/recall；Kynkäänniemi et al.（补充材料暂未公开）（NeurIPS 2019）用 feature-space kNN balls 给经验估计。

对真实 feature $r_i$，令 $\rho_i$ 是第 $k$ 个真实近邻距离。生成点 $g$ 被 real manifold 接受，当

$$
\exists i,\qquad \|g-r_i\|_2\le\rho_i.
$$

于是

$$
\widehat{\operatorname{precision}}
=\frac1{N_g}\sum_{j=1}^{N_g}
\mathbf1\{g_j\text{ 被 real balls 覆盖}\}.
$$

交换 real/generated 角色得到 recall。高 precision、低 recall 对应“看起来像真的，但只覆盖少数模式”；高 recall、低 precision 对应“覆盖广，但产生大量 off-manifold 样本”。

## 36. 一个 mode-dropping 的几何例子

![Precision/recall 的匹配、mode drop 与过度离散例子](/images/diffusion/d12_precision_recall.png)

图中真实分布是圆环。只生成半个圆环时，生成点仍落在真实邻域内，因此 precision 保持 1；但一半真实模式没有被生成邻域覆盖，recall 降到约 0.55。

这也提醒我们：结果取决于 embedding、$k$、sample density 和 outlier。precision/recall 比单值信息更丰富，但不是无参数真值。

## 37. 评估统计的数值对照

![Empirical FID bias、MMD variance 与多维评估责任](/images/diffusion/d12_metric_bias.png)

固定种子的 synthetic check 得到：

- $n=64$ 同分布 empirical FID 平均约 $4.68\times10^{-2}$，大于 $2/n=3.125\times10^{-2}$；
- 同分布 MMD U-statistic 平均约 $3.1\times10^{-4}$，接近 0，但单次估计区间跨过负值；
- 单一 feature distance 几乎不覆盖 condition alignment、bias/safety 和 system cost。

这些数字只承担公式说明，不是现实模型 benchmark。

## 38. 条件模型必须评估 joint behavior

text-to-image 模型真正建模的是条件分布

$$
p_\theta(x\mid c),
$$

而无条件 FID 主要比较边缘

$$
p_\theta(x)=\int p_\theta(x\mid c)p_{\mathrm{eval}}(c)\,dc.
$$

模型可以忽略 condition，却在某个 prompt mixture 下获得不错的 marginal FID。因此至少同时报告：

| 维度                  | 例子                                                   |
| ------------------- | ---------------------------------------------------- |
| image quality       | artifact、aesthetics、realism                          |
| condition alignment | text/image、layout、source preservation                |
| coverage/diversity  | mode coverage、intra-prompt diversity                 |
| reasoning           | count、spatial relation、compositional prompt          |
| originality         | training-neighbor audit、template repetition          |
| risk                | bias、toxicity、unsafe capability、fairness             |
| robustness          | paraphrase、language、seed、guidance、adversarial prompt |
| efficiency          | NFE、latency、memory、energy                            |

HEIM（补充材料暂未公开）（NeurIPS 2023）用 12 个方面、62 类场景和 26 个模型展示：没有一个模型在所有轴上最好。这个结论是 benchmark scope 内的多维证据，不是对所有部署 prompt 的完整审计。

## 39. Human evaluation 也需要统计设计

“让人看”不会自动消除 metric bias。一个可信 human study 至少说明：

- absolute rating、pairwise preference 还是 ranking；
- 是否盲化模型身份和随机化顺序；
- prompt 与样本如何抽取；
- 评审者数量、背景、地区和语言；
- inter-rater disagreement；
- confidence interval 与 multiple comparison；
- 质量、alignment、安全是否拆开问；
- 是否泄露研究目的或只展示成功样本。

人评是测量协议，不是无噪声 oracle。

## 40. 跨模态指标：通用 feature distance 不够

[D11](/blog/diffusion/d11-representative-applications/) 已保存三类反例：

- video 需要 VBench 类 temporal consistency、motion 与 condition adherence；
- audio 需要 FAD、听感与语义/任务指标；
- molecule/docking 需要 PoseBusters 类 physical validity，而不是只看生成 feature。

科学生成中，distribution similarity 甚至不能代替守恒、几何、能量、实验可合成性和 downstream utility。领域 constraint 常常是 hard validity，而不是多加一个感知分数。

## 41. NFE、latency、memory、energy 和 carbon 分开报告

$$
\text{NFE}\ne\text{latency}\ne\text{energy}\ne\text{carbon}.
$$

| 量          | 定义/单位                   | 主要混杂因素                                   |
| ---------- | ----------------------- | ---------------------------------------- |
| NFE        | denoiser/score 调用次数     | 每次调用的架构、分辨率、condition                    |
| latency    | ms/sample 或延迟分布         | batch、warmup、同步、I/O、VAE/safety checker   |
| throughput | samples/s               | batch、parallelism、队列                     |
| memory     | peak allocated/reserved | precision、activation、framework allocator |
| energy     | J 或 kWh                 | power 随时间积分、idle baseline、系统边界           |
| carbon     | CO2e                    | energy 加地区/时间 carbon intensity           |

Power Hungry Processing（补充材料暂未公开）（ACM FAccT 2024）实测多类部署推理任务，说明生成式任务在所测设置中通常比专用判别任务昂贵。它不能被复制成所有模型/硬件的固定能耗。

## 42. 公平系统比较的最小协议

一个可复核的质量--效率点至少记录：

$$
\mathcal R=\{
\text{checkpoint, sampler, NFE, guidance, resolution, batch, precision, hardware, software}
\}.
$$

具体包括：

1. checkpoint 与代码 commit；
2. sampler、schedule、随机性和 NFE；
3. resolution/sequence length、batch 与样本数；
4. text encoder、VAE、postprocess 与 safety checker 是否计时；
5. hardware、driver、framework、compile/quantization；
6. warmup、device synchronization、median/p95 latency；
7. peak memory 的口径；
8. energy measurement boundary 与 idle baseline；
9. 与效率同一配置下的 quality metric 和置信区间。

只报 NFE，无法证明 wall-clock 更快；只报参数量，无法推断 energy。

## 43. 记忆、复制、提取和成员推断不是同义词

| 概念                   | 操作性问题            | 不能自动推出    |
| -------------------- | ---------------- | --------- |
| memorization         | 模型行为是否显著依赖某训练样本  | 黑盒一定可恢复它  |
| copying/replication  | 输出是否跨过某相似度阈值     | 法律侵权结论    |
| extraction           | 给定访问和预算，攻击能否恢复内容 | 所有训练样本均泄露 |
| membership inference | 能否判断候选是否参与训练     | 已获得其内容    |

一个模型可以 memorise 但当前攻击提取不到；一个输出可以与训练样本相似，但相似来自常见模式而非该样本的因果影响；membership attack 可以成功，却不产生可读训练图像。

## 44. 为什么最近邻是候选证据，不是自动判决

设查询点 $q$ 与 reference distribution 的单点距离 CDF 为

$$
F_q(r)=\Pr(\|X-q\|\le r).
$$

reference bank 有 $N$ 个 iid 点，最近距离

$$
R_{\min}=\min_{1\le i\le N}\|X_i-q\|.
$$

则

$$
\Pr(R_{\min}>r)
=\Pr(\|X_i-q\|>r,\forall i)
=\bigl(1-F_q(r)\bigr)^N.
$$

所以 $N$ 增大时，最近邻自然变近。固定阈值跨不同训练库规模不可比。

此外还要说明：

- 像素、LPIPS、CLIP、SSCD 看见不同相似性；
- crop、局部复制、模板和语义复述可能绕过全图距离；
- false positive 要用 hold-out/non-member 校准；
- retrieval bank 是否真覆盖完整训练集；
- threshold、query count、prompt/seed 与人工复核如何定义。

## 45. 2023 年 extraction 证据链做了什么

Somepalli et al.（补充材料暂未公开）（CVPR 2023）构建训练图像检索框架，研究 dataset size、重复和 text conditioning 与 replication 的关系。

Carlini et al.（补充材料暂未公开）（USENIX Security 2023）明确给出 extraction/memorization 的操作定义，并在 generate-and-filter/query budget 下恢复训练样本。论文同时强调其定义是保守、特定的，并不覆盖所有隐私伤害。

Somepalli et al.（补充材料暂未公开）（NeurIPS 2023）进一步分析 caption conditioning 对 copying 的作用，并讨论 caption augmentation 等缓解。

Webster（补充材料暂未公开） 复现 extraction 并提出 template verbatim；截至核验日保留为 preprint。

共同结论不是“Diffusion 只会复制”，而是：某些训练样本在给定数据重复、conditioning 与攻击预算下确实可以被近似恢复。

## 46. Membership inference 的对象

给定候选样本 $x$、模型访问和辅助信息，membership adversary 输出

$$
A(x,M)\in\{0,1\},
$$

试图区分

$$
H_1:x\in D_{\mathrm{train}},
\qquad
H_0:x\notin D_{\mathrm{train}}.
$$

Duan et al.（补充材料暂未公开）（ICML 2023）的 SecMI 使用 forward posterior estimation 的 step-wise error 差异构造 attack。

评价必须报告 base rate、balanced/unbalanced accuracy、ROC/AUC、TPR at low FPR、query access 和 shadow/auxiliary data。高 AUC 是特定 attack 下的泄露证据；低 AUC 只是该攻击未成功，不是 privacy proof。

## 47. Differential privacy 给出的是算法级 worst-case stability

随机训练算法 $M$ 满足 $(\varepsilon,\delta_{\mathrm{DP}})$-differential privacy，若对任意相邻数据集 $D,D'$ 与可测事件 $S$：

$$
\boxed{
\Pr[M(D)\in S]
\le
e^\varepsilon\Pr[M(D')\in S]
+\delta_{\mathrm{DP}}
}.
$$

相邻关系可以是 add/remove-one 或 replace-one，必须明确。DP-SGD 的典型机制是：

$$
g_i\leftarrow
\frac{g_i}{\max(1,\|g_i\|_2/C)},
$$

$$
\widetilde g
=\frac1B
\left(
\sum_{i=1}^B g_i
+\mathcal N(0,\sigma_{\mathrm{DP}}^2C^2I)
\right),
$$

然后按 sampling rate、steps、noise multiplier 和 accountant 组合 privacy loss。

## 48. DP 的 post-processing 为什么重要

若 $M$ 是 DP，$A$ 是不再访问原数据的任意随机后处理。令

$$
K(m,S)=\Pr(A(m)\in S).
$$

则

$$
\Pr(A(M(D))\in S)
=\int K(m,S)\,dP_{M(D)}(m).
$$

DP inequality 对 $[0,1]$-值可测函数扩展后给出

$$
\Pr(A(M(D))\in S)
\le
e^\varepsilon\Pr(A(M(D'))\in S)
+\delta_{\mathrm{DP}}.
$$

所以由 DP checkpoint 生成、筛选或发布样本仍是 post-processing。边界是：

- 训练 accountant 必须先正确；
- 只保护 adjacency definition 中的数据；
- public pretraining 加 private fine-tuning 不追溯保护 public pretraining set；
- utility 与 $(\varepsilon,\delta)$ 形成前沿，不能只报一个 FID。

Ghalebikesabi et al.（补充材料暂未公开） 提供 private fine-tuning 的 utility 证据，但当前仍按预印本引用。

## 49. 记忆与隐私图：攻击证据和定义保证分开

![最近邻规模、四类记忆问题与 DP 后处理](/images/diffusion/d12_memorization_privacy.png)

左图说明 reference bank 变大时最近距离下降；中图强制区分四类问题；右图用 randomized response 的精确小例子验证 post-processing 不增大 privacy probability ratio。图中没有运行真实 extraction attack。

## 50. 2025--2026 的记忆理论：从全局过拟合到局部覆盖

Achilli et al.（补充材料暂未公开）（JSTAT 2025）在 hidden-manifold/high-dimensional 模型中研究样本量、内在维度、吸引域与 memorization/generalization transition。

2026 年的 Local Coverage Governs Memorization（补充材料暂未公开） v1 进一步强调：同一模型可以在稠密区域插值/generalize，在低覆盖区域被 isolated samples 主导。

这种解释与 2023 年“重复和 conditioning 影响 copying”的实证线相互呼应，但仍要保留：

- stylized KDE/high-dimensional assumptions；
- local density 的表示依赖；
- 大模型训练动态、augmentation 和 conditioning 更复杂；
- 2026 工作仍是前沿预印本。

## 51. 数据 provenance：LAION-5B 不是统一许可证明

LAION-5B（补充材料暂未公开）（NeurIPS 2022 Datasets and Benchmarks）提供数十亿 image--text pair 的索引、过滤和若干 watermark/NSFW/toxicity score。

技术上必须区分：

$$
\text{URL/index provenance}\ne
\text{copyright permission}\ne
\text{privacy consent}\ne
\text{model-use license}.
$$

一个数据集可公开描述来源，不代表每项内容共享同一许可；一个模型可有 OpenRAIL 使用限制，也不自动解决训练内容权利。

## 52. 模型卡是版本化证据接口

本地保存的 Stable Diffusion v1.4 官方模型卡（补充材料暂未公开） 记录：

- 模型架构与训练数据来源；
- intended 与 out-of-scope use；
- language、人物、文本生成和社会偏差；
- safety checker 的存在与非完备性；
- CreativeML OpenRAIL-M 使用限制。

核验时，模型库没有在尝试路径暴露独立 root LICENSE 文件，因此资料库不虚构该 artifact；许可信息直接来自模型卡 front matter 及其官方 license link。

模型卡也有边界：它只代表特定 checkpoint/version，不自动代表第三方 fine-tune、merge、后续产品或修改过的 safety pipeline。

## 53. Stable Bias：偏差评估也是测量问题

Stable Bias（补充材料暂未公开）（NeurIPS 2023 Datasets and Benchmarks）通过职业、性别、族裔等 prompt slices 分析 text-to-image 的社会表征。

需要保留两层不确定性：

1. 模型输出分布可能体现训练数据、caption、filter 和 representation bias；
2. 用于给输出自动标注 demographic attribute 的 classifier 本身也有 bias 和 measurement error。

因此安全表述是“在给定 prompt 与测量协议中观察到系统性 representation pattern”，不是“自动 classifier 揭示了人物的真实属性”。

## 54. Unsafe Diffusion：filter 不是 capability removal

Unsafe Diffusion（补充材料暂未公开）（ACM CCS 2023）研究 text-to-image 生成 unsafe images 和 hateful memes 的能力，并展示 prompt/filter 防线可被绕过。

攻击成功率依赖：

- checkpoint 与服务版本；
- prompt corpus、语言和 mutation strategy；
- guidance、sampler、seed 与 query budget；
- unsafe detector 阈值；
- human adjudication。

所以一次 safety benchmark 是已测场景的 risk evidence，不是 universal safety certificate。

## 55. Concept erasure 的目标到底是什么

Erasing Concepts from Diffusion Models（补充材料暂未公开）（ICCV 2023）通过参数编辑抑制 nudity、style 或 object concept。经验目标可写为：对 prompt 集 $\mathcal C$，降低某 detector/semantic score

$$
\mathbb E_{c\sim\mathcal C,z}
R_{\mathrm{target}}\bigl(G_{\theta'}(z,c)\bigr),
$$

同时保持一般质量

$$
\mathbb E_{c\sim\mathcal C_{\mathrm{retain}},z}
d\bigl(G_{\theta'}(z,c),G_\theta(z,c)\bigr)
$$

不过大。

这不等于证明：

- 所有同义词、隐喻、翻译和组合 prompt 都不能恢复概念；
- 相关训练样本的信息已从参数中删除；
- 邻近概念没有 collateral damage；
- adaptive adversary 无法寻找新 prompt。

因此本章称它为 concept suppression/model editing under a protocol，而不是无条件 machine unlearning。

## 56. Ring-A-Bell：为什么缓解必须面对 adaptive stress test

Ring-A-Bell（补充材料暂未公开）（ICLR 2024）构造能够重新触发被 removal method 抑制概念的 prompt，用于测试方法可靠性。

它修正了“只在原 prompt 上比较 before/after”这一弱协议。一个更完整的 removal audit 至少包括：

- original prompts；
- paraphrase/synonym/translation；
- compositional and indirect prompts；
- model-specific white-box 与 transferable black-box attacks；
- 多 seed、guidance 与 sampler；
- target success、retain quality 与 collateral damage。

平均 suppression rate 不能替代 worst-case/adaptive robustness。

## 57. Stable Signature：水印是统计检测，不是真假判断

Stable Signature（补充材料暂未公开）（ICCV 2023）微调 latent diffusion decoder，使生成图像携带可恢复 bit signature，并通过统计检验判断来源。

一个 watermark report 至少给：

$$
\operatorname{FPR}(\tau)=\Pr_{H_0}(S\ge\tau),
\qquad
\operatorname{FNR}(\tau)=\Pr_{H_1}(S<\tau),
$$

以及 payload、quality impact、crop/compression/edit attack 和 key assumption。

水印能帮助 provenance，但不能自动：

- 判断内容是真实照片还是合成内容；
- 证明训练数据许可；
- 防止 unsafe generation；
- 抵抗任意重生成、模型替换或 key compromise；
- 代替签名 metadata、发布日志和平台治理。

## 58. 安全是分层证据问题

![数据、模型、接口、缓解、溯源和部署的安全证据层](/images/diffusion/d12_safety_evidence.png)

一项完整系统审计至少跨六层：

1. **data**：coverage、duplication、caption、filter、license；
2. **model**：memorization、bias、unsafe capability；
3. **interface**：prompt、guidance、seed、checker、jailbreak；
4. **mitigation**：filter、editing、DP 与 utility；
5. **provenance**：watermark、metadata、model card、code commit；
6. **deployment**：access control、monitoring、user context、incident response。

没有一个单值 filter score 能关闭全部六层。

## 59. Copyright 与技术证据的边界

技术教程可以可靠说明：

- training-neighbor similarity 如何测；
- extraction attack 的访问、阈值和样本；
- dataset/model provenance 如何记录；
- opt-out、deletion propagation 和版本控制是否存在；
- model-use license 写了什么。

技术教程不能仅凭距离阈值给出具体法域中的 infringement、fair use、authorship 或 damages 结论。法律概念与 memorization metric 不共享自动映射。

因此本章把 copyright 放在 governance boundary 中，而不伪装成一个可由 FID/SSCD 解出的分类任务。

## 60. Benchmark saturation、evaluation leakage 与生成数据污染

评估还面临三个越来越重要的问题。

**Benchmark saturation**：模型、prompt engineering 和 hyperparameter 反复针对固定 benchmark 优化，测试集逐渐成为训练信号。

**Evaluation leakage**：公开 prompt、reference image、human preference data 或 metric encoder 可能进入预训练/微调数据，使 held-out 假设失效。

**Generated-data contamination**：网页和后续训练集混入生成内容，导致 provenance、diversity 和 model-collapse 研究更困难。

可信报告应记录 benchmark version、prompt secrecy、dedup/contamination audit，并在可能时使用动态或新建测试集。

## 61. 证据等级：来源类型决定它能承担什么

| 来源                   | 可以承担                          | 不应单独承担               |
| -------------------- | ----------------------------- | -------------------- |
| 同行评审原论文              | theorem、method、指定实验结论         | 所有部署版本的性能            |
| benchmark/data paper | protocol、dataset 与已测覆盖        | 全现实世界安全性             |
| 官方模型卡                | checkpoint、用途、限制、license 声明   | 独立第三方验证              |
| fixed code commit    | implementation mapping 与可复现接口 | 论文 theorem 的正确性      |
| preprint             | 前沿问题和暂定结果                     | settled consensus    |
| tutorial/blog        | 教学导航和直觉                       | priority、正式发表状态、定理责任 |
| product claim        | 组织公开声称                        | 可复核 benchmark 证据     |

本章资料库明确标记 Azangulov、Li--Yan、Ghalebikesabi、Webster 与 Merger--Goldt 为 preprint；不会因为“更近”而提升证据等级。

## 62. 说明代码：它验证什么

运行：

```bash
# 本地验证脚本暂未公开
# 本地验证脚本暂未公开
```

核心数值检查包括：

```python
path_kl = initial_kl + 0.5 * weighted_score_mse
tv_bound = np.sqrt(0.5 * path_kl)

fid = (x.mean() - y.mean()) ** 2 + (x.std(ddof=1) - y.std(ddof=1)) ** 2

mmd_u = (
    off_diagonal_mean(k_xx)
    + off_diagonal_mean(k_yy)
    - 2.0 * k_xy.mean()
)
```

脚本还实现 kNN precision/recall、reference-bank nearest distance、randomized response DP ratio 和 early-stop Gaussian coupling。它不训练网络、不下载数据、不运行真实隐私或安全攻击。

## 63. 代码输出与独立交叉检查

说明代码固定种子结果为：

```text
path_kl: 9.000000000e-02
pinsker_tv: 2.121320344e-01
fid_null_mean: 4.684296973e-02
fid_mean_floor: 3.125000000e-02
mmd_null_mean: 3.103613163e-04
mode_drop_precision: 1.000000000e+00
mode_drop_recall: 5.500000000e-01
nn_monotone_fraction: 1.000000000e+00
dp_probability_ratio: 3.320116923e+00
early_stop_coupling: 2.117602783e-01
early_stop_bound: 2.116601049e-01
```

独立脚本 d12\_theory\_evaluation\_checks.py（补充材料暂未公开） 使用不同 trial count 和实现路径复核同一组 invariants，避免“绘图函数自证”。

## 64. 常见错误

1. 把 empirical denoising loss 当作已知 population score error；
2. 把 score error 当作已经包含 initialization、early stopping 和 discretization；
3. 不说明 path KL 的方向；
4. 忽略 Girsanov 的共同 diffusion coefficient 与绝对连续条件；
5. 把 terminal projection 的 data processing 反向使用；
6. 在不同 TV convention 下漏掉 $1/2$；
7. 对 manifold endpoint 声称 $p_\delta\to p_0$ 的 TV convergence；
8. 把理论 step complexity 直接称作实际 latency；
9. 比较两个大 O 时忽略 metric、endpoint 和 score access 不同；
10. 把 preprint 写成已正式发表；
11. 把 FID 当作真实/生成分布本身的距离；
12. 跨样本数、resize、compression 或 feature implementation 比 FID；
13. 认为 KID 无偏就没有 variance 或不能为负；
14. 用一个 precision/recall 数字而不报告 embedding、$k$ 和样本数；
15. 条件模型只报 marginal image quality；
16. 把 human evaluation 当作无偏 oracle；
17. 把 NFE、latency、energy 和 carbon 混用；
18. 不在同一 quality point 比较效率；
19. 把 memorization、copying、extraction 和 membership inference 互换；
20. 只因没发现近邻就声称没有记忆；
21. 忽略 reference bank 规模对 nearest distance 的影响；
22. 把某个 membership attack 失败称作 differential privacy；
23. 不报告 DP adjacency、sampling rate、steps、noise multiplier 与 accountant；
24. 把 concept suppression 称作可证明数据删除；
25. 用非 adaptive prompt 集评估 concept removal；
26. 把 watermark 当作内容真假或 license 证明；
27. 把模型卡当作对所有 derivative checkpoint 的担保；
28. 把技术相似度直接翻译为法律结论；
29. 把 safety benchmark 通过称作系统安全证明；
30. 用产品宣传替代可复核来源。

## 65. 章节小结

这一章建立了四个闭环。

**理论闭环**：population score error 经 Girsanov 控制 path KL，再经 data processing 与 Pinsker 控制 smoothed marginal；initialization、early stopping、discretization、finite data 与 approximation 仍需独立接口。

**评估闭环**：FID 是 feature Gaussian moment distance，empirical estimator 有偏；KID/MMD 可无偏但有方差；precision/recall 拆开 fidelity/coverage；条件、人评、模态有效性和系统成本必须组成矩阵。

**隐私闭环**：memorization、copying、extraction、membership inference 是四个问题；nearest-neighbor audit 要校准；只有 formal DP 定义给算法级 worst-case stability。

**安全闭环**：data、model、interface、mitigation、provenance 和 deployment 是六层证据；concept editing、水印、模型卡各有局部职责，不能互相代替。

最终判断不是“Diffusion 被证明正确/安全了吗”，而是：**给定一个具体 claim，证据链缺哪一环？**

## 66. 研究式思考题

1. 如果 theorem 给出 $\operatorname{KL}(p_\delta\|q_\delta)$ 很小，能否推出 $\operatorname{KL}(q_\delta\|p_\delta)$ 很小？构造反例需要什么 tail behavior？
2. 对 VP diffusion，推导 $W_2(p_0,p_\delta)$ 的 coupling bound，并分离 signal attenuation 与 Gaussian noise 两项。
3. Girsanov path KL 中若 score error 在模型 marginal 而非真实 $p_t$ 下测量，需要怎样改变 KL 方向或使用 change of measure？
4. 设计一个 adaptive time grid，使 discretization budget 与 $g(t)^2\mathbb E\|s_\theta-s_t\|^2$ 的局部分布匹配。什么条件下它真的改善全局 bound？
5. 比较 Lee--Lu--Tan、Benton 和 Huang PF-ODE 结果时，哪些假设差异比 dimension exponent 更重要？
6. 构造两个非 Gaussian feature distributions，使它们均值、协方差相同而 FID 为 0，但样本明显可分。
7. 无偏 MMD estimator 可以为负。若 leaderboard 强制截断为 0，会引入怎样的 bias？
8. kNN precision/recall 在高维 distance concentration 下会怎样退化？是否可用局部 intrinsic dimension 校正？
9. 条件生成的 joint metric 应比较 $p(c,x)$ 还是 $p(x\mid c)$？prompt distribution 改变时排名应怎样解释？
10. 如何把 human preference uncertainty 与 automatic metric uncertainty 放进同一个 decision rule？
11. 设计一个 nearest-neighbor copying audit，使 reference-bank size、成员/非成员 calibration 和局部 crop 都被显式处理。
12. membership inference 与 extraction 的最优攻击可能需要不同访问接口。怎样定义一个统一但不混淆的 threat-model lattice？
13. private fine-tuning 使用 public pretrained model 时，怎样在 model card 中精确写清被保护和未被保护的数据？
14. concept-removal benchmark 如何防止静态 prompt 集被方法反向优化？设计一个 held-out adaptive red-team protocol。
15. 若 watermark detector 的 FPR 为 $10^{-6}$，平台每天扫描 $10^9$ 张图，expected false alarms 是多少？base rate 如何影响 posterior precision？
16. 数据 opt-out 发生后，checkpoint、adapter、distillation student 与 synthetic dataset 的 deletion propagation 应如何记录？
17. 怎样在相同质量点比较 stochastic 50-NFE sampler、deterministic 20-NFE solver 和 one-step distilled model 的 latency、energy 与 memory？
18. 2026 local-coverage 预印本的结论若要扩展到大型 text-conditioned model，最需要验证哪些 assumptions？

## 67. 去向：从边界回到整篇技术路线

下一章 [D13 技术演进总结与论文阅读路线](/blog/diffusion/d13-evolution-reading-roadmap/) 不再引入新的核心 theorem，而会把 D1--D12 中的可训练性、质量、速度、控制、规模、模态与理论边界重组为一条问题链。

进入 D13 前，应保留本章的判断方式：任何“统一”“取代”“首次”“更快”“更安全”都必须回到对象、假设、metric、protocol 和 source status。
