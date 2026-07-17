---
title: 生成建模问题与 Diffusion 全局地图
description: 从生成建模问题出发，建立 DDPM、Score、SDE、采样、架构、Flow 与离散扩散的全局关系。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 0
slug: d0-generative-modeling-map
tags:
  - diffusion
  - generative-modeling
  - reading-map
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 定义生成任务、评价维度、核心问题链与全系列阅读路线，不在本章展开具体训练推导。
---
Diffusion 最反直觉的地方，也是理解整套方法最好的入口：我们的目标明明是从噪声生成数据，却先花大量精力研究怎样把数据变成噪声。

如果把它记成“不断加噪，再让神经网络去噪”，后面的 DDPM、score matching、SDE、ODE、solver、latent diffusion、DiT、flow matching 和 consistency 很快会变成一串互不相干的名字。真正贯穿它们的问题是：

> 如何把一个未知的高维分布，改写成可以从数据构造监督、可以数值求解、又可以从简单先验采样的生成过程？

本章只建立问题、对象、历史因果和阅读路线。所有关键公式都会出现，但不在这里重做 D1--D13 的证明。

***

## 1. 学完本章后应该能回答什么

1. 生成模型的 sampling、density、likelihood 和 representation 分别是什么任务；
2. 为什么“能生成样本”和“能计算密度”不是同一件事；
3. 为什么一次从噪声映射到数据很难，而已知 forward corruption 能制造局部监督；
4. DDPM、score model 与 score-SDE 改变的对象分别是什么；
5. latent diffusion、DiT、sampler、distillation 与 flow family 位于系统的哪一层；
6. 2015、2019、2020、2021 四个节点各自解决了什么，而不把 DDPM 写成 Diffusion 的起点；
7. 怎样按自己的目标选择 D1--D13 阅读路线。

## 2. 从有限数据到一个生成分布

设训练集为

$$
\mathcal D=\{x^{(1)},\ldots,x^{(n)}\},
\qquad x^{(i)}\sim q_{\mathrm{data}}.
$$

我们只看到有限样本，不知道真实分布 $q_{\mathrm{data}}$ 的解析式。生成建模希望学习一个带参数的分布 $p_\theta$，使它在关心的意义下接近数据分布。

“接近”不能省略。它可能指 log-likelihood 高、样本看起来真实、模式覆盖完整、条件一致、下游任务有效，或者满足物理与安全约束。不同含义并不自动一致。

## 3. Sampling：生成模型最直接的交付物

sampling 任务要求从模型得到新样本：

$$
X\sim p_\theta.
$$

典型实现是先从简单 prior 采样 $Z\sim p_Z$，再经过某种随机或确定过程得到 $X$。如果只有 sampling interface，我们可以生成图像，却未必能精确计算某张图像的概率。

这足以支持很多创作和模拟任务，但不足以回答 anomaly detection、compression、Bayesian comparison 等依赖密度或 likelihood 的问题。

## 4. Density estimation：能否评价任意输入

density estimation 希望对给定 $x$ 求

$$
p_\theta(x)
\quad\text{或至少}\quad
\log p_\theta(x).
$$

高维连续数据中，“某个点的概率”本身通常为零，真正求的是相对于参考测度的密度。密度还依赖数据表示：对离散像素、加 dequantization noise 的连续像素和 learned latent，数值不在同一个问题上。

## 5. Likelihood：训练目标、报告量与模型能力

对训练数据，maximum likelihood 使用

$$
\max_\theta
\mathbb E_{q_{\mathrm{data}}}
[\log p_\theta(X)].
$$

但实践中可能优化 exact likelihood、variational lower bound、score matching、adversarial loss 或其他 surrogate。于是必须分开：

- 模型分布是否定义 likelihood；
- 训练时是否直接优化它；
- 报告时是否能精确或近似计算它；
- likelihood 改善是否带来更好的感知样本。

DDPM 的历史恰好提醒我们：更好的 sample quality 不必来自更紧的 likelihood objective。

## 6. Representation learning：生成并非唯一目的

生成过程的中间变量可以用于压缩、表征、编辑、异常检测或下游预测。但“能生成”不自动推出“学到好的 representation”。

教程把 representation 视为独立接口：state 在 pixel、latent、token 或 graph 上定义，会改变计算成本、可逆性、decoder error 和任务语义。[D8](/blog/diffusion/d8-architecture-representation/) 专门处理这一层。

## 7. 条件生成把目标变成一族分布

有条件生成不是在无条件模型外面附加一个文本框，而是要学习

$$
p_\theta(x\mid c),
$$

其中 $c$ 可以是类别、文本、布局、参考图、测量值、动作状态或物理约束。此时评价还要问：样本是否服从条件、是否保持应保留的内容、guidance 是否扭曲覆盖。

这些问题进入 [D7](/blog/diffusion/d7-guidance-conditioning-editing/)，而不是由“去噪更准”自动解决。

## 8. 生成模型的共同抽象

许多生成模型都可写成：

$$
Z\sim p_Z,
\qquad
X\sim K_\theta(\cdot\mid Z),
$$

其中 $K_\theta$ 可能是 deterministic map、conditional decoder、Markov chain、SDE/ODE solution 或离散 transition process。

方法之间真正的差别不是“有没有噪声”，而是：

1. state 是什么；
2. path 或 map 如何定义；
3. 学习目标是什么；
4. 从模型采样需要解什么计算问题；
5. 哪些概率量可计算；
6. 误差与约束如何验证。

## 9. VAE、GAN 与 Flow 提供什么背景

D0 不展开其他生成模型，但需要知道 Diffusion 在回应哪些长期取舍。

| 路线               | 训练接口                                | 采样接口                      | 密度接口                     | 典型约束                                     |
| ---------------- | ----------------------------------- | ------------------------- | ------------------------ | ---------------------------------------- |
| VAE              | latent-variable ELBO                | prior + decoder           | 常用 variational bound     | posterior approximation 与 decoder family |
| GAN              | discriminator/adversarial objective | 一次 generator map          | 通常无 tractable density    | minimax optimization、coverage 与评价        |
| Normalizing Flow | change of variables                 | invertible map            | exact likelihood         | invertibility、Jacobian 与 architecture    |
| Diffusion/score  | denoising、score 或 path objective    | reverse chain/SDE/ODE/map | VLB、score 或 PF-ODE route | 多步求解、endpoint、score 与 solver error       |

这不是优劣榜。每条路线选择了不同的可计算对象，也留下不同的限制。

## 10. 为什么不直接学一个噪声到数据的映射

设 $Z\sim\mathcal N(0,I)$。当然可以直接训练

$$
X=G_\theta(Z).
$$

难点不是神经网络不能表示复杂函数，而是数据本身没有告诉我们某个 $z$ 应该对应哪个 $x$。从 prior 到 data 的 coupling 未知；仅给两边样本时，监督配对并不存在。

GAN 用 adversarial comparison 处理这个问题，flow 用可逆结构和 Jacobian 处理，VAE 引入 inference model。Diffusion 的选择是：**先人为规定 data-to-noise path，让这条路径反过来产生训练对。**

## 11. 一次彻底破坏为什么没有帮助

若直接令

$$
x_1=\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
$$

则 $x_1$ 与 $x_0$ 独立。要求模型从 $x_1$ 恢复 $x_0$，仍然等价于原始的无配对生成问题。

有用的 forward process 必须保留一系列难度连续的中间状态，使模型可以从“局部去噪”逐渐学习整个 data-to-noise path 的逆方向。

## 12. 已知 corruption 如何制造监督

最常见的 Gaussian channel 写成

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

训练时我们自己抽了 $x_0,t,\epsilon$，因此知道 clean data、corrupted state 和 injected noise。网络不再面对无配对的 $z\to x$，而是面对可无限在线合成的监督样本。

![二维数据沿已知 Gaussian channel 逐渐变成噪声](/images/diffusion/d0_forward_corruption.png)

图中的相同噪声只用于让视觉变化连续；真实 Markov path 的跨时刻噪声结构见 [D1](/blog/diffusion/d1-forward-diffusion/)。

## 13. “先破坏、再逆转”的三个收益

第一，terminal distribution 可设计为简单 prior，生成时有明确起点。

第二，任意 noise level 的训练输入可直接构造，不必先跑完整 forward chain。

第三，复杂全局生成被拆成 denoising、score 或 local velocity 等局部对象，可用同一个 time-conditioned network 学习。

这三个收益都不是免费午餐：path、schedule、target、network、solver 和 endpoint 的误差会重新耦合起来。

## 14. Forward process 不是模型学出来的答案

在经典 DDPM 主线中，forward process $q$ 是设计者固定的 corruption/inference process；learned generative model 是 reverse process $p_\theta$。

因此：

- forward noising 不等于生成；
- 加噪容易不代表反向容易；
- 知道 $q(x_t\mid x_0)$ 不等于知道无条件 $q(x_{t-1}\mid x_t)$；
- 训练 pair 可构造不等于 finite network 已恢复真实 reverse law。

这些对象在 D1 与 [D2](/blog/diffusion/d2-ddpm-objective/) 中严格区分。

## 15. 三种观察 Diffusion 的视角

同一系统至少有三种互补视角：latent-variable likelihood、transport/map 和 stochastic process。它们回答不同问题，不能把其中一套语言当成唯一“本质”。

## 16. 视角一：显式或隐式 likelihood

离散 diffusion 可以定义完整 latent path：

$$
p_\theta(x_{0:T})
=p(x_T)
\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).
$$

对中间 latent 积分得到 $p_\theta(x_0)$，训练可用 path ELBO。这个视角解释每个 reverse transition 如何贡献 likelihood bound，但实际常用的 reweighted denoising loss不必等于 exact ELBO。

## 17. 视角二：Score field

score 定义为

$$
s_t(x)=\nabla_x\log q_t(x).
$$

它不需要知道 density 的归一化常数。通过 denoising score matching，可用已知 conditional corruption score 监督网络。给定 time-dependent score 后，Langevin、reverse SDE 与 PF ODE 都能把它转为采样方向。

score estimation 的完整等价证明、边界条件和 manifold 问题见 [D3](/blog/diffusion/d3-score-matching/)。

## 18. 视角三：Stochastic process 与 transport

连续时间下，forward SDE 写成

$$
dX_t=f(X_t,t)dt+g(t)dW_t.
$$

reverse-time dynamics 由 forward drift、diffusion coefficient 和 marginal score 决定；另一个 probability-flow ODE 可在理想条件下共享 one-time marginals。

这给出从 stochastic path 到 deterministic transport 的桥梁，但“相同 marginal”不等于“逐样本同路径”。完整推导见 [D4](/blog/diffusion/d4-continuous-time-sde/)。

## 19. 2015：把渐进破坏与 learned reversal 组合成生成模型

Sohl-Dickstein et al. 2015 从“灵活性与可计算性冲突”出发：简单分布容易评价和采样，却不能拟合复杂数据；灵活非归一化模型又常需要昂贵 MCMC 或 partition function。

其代表性响应是：固定许多小步组成的 forward diffusion，学习 reverse Markov chain，并用 trajectory ratio/path lower bound 训练。这建立了现代 diffusion probabilistic model 的基本构型。

安全边界：它不是后来 DDPM 完整系统；小步极限的 reverse-family 论证也不能扩写成任意有限步真实逆条件都自动是简单 Gaussian。页码与版本见 [2015 原论文笔记](https://arxiv.org/abs/1503.03585 "官方论文页面")。

## 20. 更早支流：Score Matching 与 DSM

现代 Diffusion 的可训练性还来自另一条线。

Hyvärinen 2005 用 data/model score 的 Fisher divergence 绕开未知归一化常数。Vincent 2011 进一步证明：对 corruption conditional score 做回归，与匹配 smoothed marginal score 只差模型无关常数。

它们分别解决“density 未归一化”和“data score 不可见”两个问题，却还没有组成现代多噪声图像生成系统。详细证据见 前驱关系笔记（补充材料暂未公开）。

## 21. 2019：NCSN 让多噪声 Score 成为图像生成路线

Song--Ermon 2019 指出 naïve score + Langevin 有三类实际障碍：数据可能集中在低维 manifold，低密度区域缺少 score 监督，多模态分布的 Langevin mixing 困难。

其响应是：对数据加入多个强度的 Gaussian noise，用一个 noise-conditioned network 学习所有 perturbed scores，再从大噪声到小噪声运行 annealed Langevin dynamics。

这里的历史角色是“把已有 score/DSM/Langevin 对象组织为多噪声图像生成框架”，不是首次提出 score matching 或 Langevin dynamics。版本和原文页码见 [NCSN 笔记](https://arxiv.org/abs/1907.05600 "官方论文页面")。

## 22. 2020：DDPM 的关键转折不是“开始加噪”

Ho et al. 2020 延续 2015 的 Gaussian Markov chain，找到简单而有效的 epsilon-prediction parameterization，并在实践中使用重新加权的 uniform-t noise MSE 与 U-Net 系统，展示有竞争力的高质量图像生成。

DDPM 还明确连接 diffusion、multi-noise DSM 与 Langevin sampling。它的历史转折应写成“有效参数化与高质量系统”，而不是“首次提出 Diffusion”。

另一个重要边界是：常用 $L_{\mathrm{simple}}$ 丢弃了 exact VLB 中的时间权重。因此“从 ELBO 推导”不等于“训练目标与 ELBO 完全相同”。见 [DDPM 原论文笔记](https://arxiv.org/abs/2006.11239 "官方论文页面") 与 D2。

## 23. 2021：Score-SDE 提供连续时间统一接口

Song et al. 2021 观察到 SMLD/NCSN 与 DDPM 都沿 noise levels 学习 time-dependent denoising direction，却使用不同的 perturbation、parameterization 和 sampler 语言。

score-SDE 将前者表成 VE-style SDE，将后者表成 VP-style SDE，并用 time-dependent score 连接 reverse SDE、predictor--corrector、probability-flow ODE 和 likelihood computation。

它的“统一”有明确范围：不是说 VE 与 VP path law 相同，不是说所有 finite-step sampler 等价，也不是统一了后来所有 flow/consistency 方法。见 score-SDE 笔记（补充材料暂未公开）。

## 24. 这段历史不是四篇论文的孤立接力

更可靠的依赖图是：

$$
\text{score matching}
\to
\text{DSM}
\to
\text{multi-noise score generation},
$$

与

$$
\text{nonequilibrium path}
\to
\text{diffusion probabilistic model}
\to
\text{DDPM parameterization},
$$

再通过 stochastic time reversal 与 continuous dynamics 在 score-SDE 中汇合。

这是概念依赖，不是完整 priority 裁决。正式年份、公开版本、原文页码和禁止用语已经单独记录在 D0 历史事实账本（补充材料暂未公开）。

## 25. 2021--2022：采样速度成为独立问题

DDPM 的训练可并行，但生成需要顺序执行大量 reverse steps。后续 DDIM、learned variance、PNDM/DEIS、DPM-Solver、EDM-style design 把问题改写为 trajectory choice、numerical integration、time grid 和 discretization error。

这类方法通常保持 trained field，改变求解方式。它们属于 [D6](/blog/diffusion/d6-sampling-solvers/)，不能与重新训练 few-step student 混为一类。

## 26. 2021--2023：条件与编辑成为模型接口

classifier guidance、classifier-free guidance、cross-attention、ControlNet、adapter 和 inverse-problem guidance 解决的是“条件如何改变 target distribution 或 sampling field”。

guidance scale 提高条件一致性时可能牺牲 diversity、calibration 或图像自然度；编辑还要处理 source preservation 与 inversion error。[D7](/blog/diffusion/d7-guidance-conditioning-editing/) 按 target/score/posterior/interface 区分这些方法。

## 27. 2022--2024：Latent Diffusion 与 DiT 位于不同层

Latent Diffusion 把 state 从 pixel 移到 learned latent，主要响应高分辨率计算成本；它引入 encoder/decoder bottleneck 和 representation error。

DiT 等 transformer backbone 改变 denoiser architecture 和 scaling interface，不改变“Diffusion”这一 path/objective 的定义。一个系统可以同时使用 latent state、transformer backbone、flow-style path 和特定 sampler。

因此“LDM、DiT、DDPM 谁替代谁”不是正确问题。应问它们分别改变 state、backbone、target 还是 solver。见 [D8](/blog/diffusion/d8-architecture-representation/)。

## 28. 2022--2026：从数值积分到学习有限时间 Map

当多步 local field evaluation 仍太慢，另一条路线不再只改 solver，而是改变 learned object：progressive distillation 学 fewer-step student，consistency family 学 endpoint-consistent function，flow matching 学 velocity field，flow-map/MeanFlow 类方法进一步面向有限区间 map 或 average velocity，distribution distillation 则对齐 student distribution。

它们可能都实现 few-step generation，但监督对象、coupling、path 和 guarantee 不同。[D9](/blog/diffusion/d9-few-step-flow/) 的首要任务就是防止“都快，所以都是 sampler”这种分类错误。

## 29. 离散状态空间不是把 Gaussian 换成 token

文本、类别和图结构上没有可直接套用的 Euclidean log-density gradient。D3PM、continuous-time Markov chain、masked diffusion 和 discrete flow/generator matching 使用 transition matrix、reverse rate、density ratio 或 generator。

因此离散路线在 [D10](/blog/diffusion/d10-discrete-diffusion-language-models/) 中单独展开。语言系统还必须报告 length、decoding policy、cache、wall-clock 和 likelihood，不能仅凭“可并行更新 token”声称更快。

## 30. 应用会反过来改变方法

视频提出长时一致性和因果/世界动态，音频提出 waveform 与感知频带，3D 提出 geometry 与 score distillation，科学生成提出 symmetry 与 validity，robotics 提出 receding horizon，time series 提出 mask 和 missingness。

[D11](/blog/diffusion/d11-representative-applications/) 不把应用当模型清单，而是追踪 domain constraint 如何反馈到 state、path、condition、architecture 和 metric。

## 31. 理论、评估与安全是每条边的审计层

低训练 loss 是否推出 score 准确？score 准确是否推出 sampler distribution 接近？低 FID 是否同时表示 fidelity 与 coverage？训练数据是否被记忆？erasure 是否真的移除能力？

这些不是教程末尾的附加伦理话题，而是模型 claim 的 proof obligations。[D12](/blog/diffusion/d12-theory-evaluation-safety/) 分开处理 convergence、metric、privacy、safety、provenance 与 model-card evidence。

## 32. 七条耦合问题轴

![D0 开篇问题--方案--章节地图](/images/diffusion/d0_problem_solution_map.png)

Diffusion 的演进可以压缩成七条持续问题：trainability、quality、speed、control、scale、state space 和 guarantees。

一项方法经常同时作用于多轴。例如 latent representation 同时影响 scale、speed 与 quality；few-step map 同时改变 speed、training object 与 guarantee。图中的章节只是第一入口，不表示方法只能属于一行。

## 33. 九组件读法：遇到新模型先拆名字

对任何新方法，先填写：

| 组件                     | 要问的问题                                                                  |
| ---------------------- | ---------------------------------------------------------------------- |
| state                  | pixel、latent、token、graph 还是 mixed state？                               |
| forward/reference path | 如何从 data 走向 prior，或如何规定 interpolation？                                 |
| learned target         | score、noise、clean data、velocity、finite map 还是 distribution？            |
| objective              | ELBO、DSM、flow matching、consistency、distillation 还是 adversarial？        |
| parameterization       | epsilon、x0、v、preconditioned output 如何换算？                               |
| backbone               | U-Net、Transformer、equivariant network 或 domain model？                  |
| condition              | condition 改变 target、field、features 还是只改 sampling？                      |
| sampler                | ancestral、SDE、ODE、solver、distilled student 或 learned map？              |
| evidence               | likelihood、sample metric、task validity、latency、theorem 或 safety audit？ |

品牌名可以改变，九个槽位仍然有效。

## 34. 教程主干：D1--D6

第一次系统学习建议按以下顺序：

$$
\boxed{
D1\to D2\to D3\to D4\to D5\to D6
}
$$

- D1：forward corruption 和 noise coordinate；
- D2：reverse model、posterior、ELBO 与 noise objective；
- D3：score matching、DSM 与 denoising identity；
- D4：SDE、time reversal 与 PF ODE；
- D5：parameterization、weighting、schedule 与 training design；
- D6：discrete/continuous sampler 与 numerical error。

这条顺序是教学依赖，不是历史发表顺序。

## 35. 系统与控制路线：D7--D8

若目标是实现现代 image generation system，在完成 D1--D6 的核心对象后读：

$$
D7\to D8.
$$

D7 处理 condition/guidance/editing；D8 处理 representation/backbone/scaling。二者是横向系统接口，不是“比 SDE 更高级的下一代数学”。

## 36. 分支与研究回路：D9--D13

根据研究问题选择：

- few-step/flow：D4、D6 后进入 D9；
- discrete/language：D3、D4 后进入 D10；
- domain application：按需读 D7--D11；
- theorem/metric/privacy/safety：D3、D4、D6 后进入 D12；
- 选题与最新论文：最后用 D13 回看问题轴和证据状态。

D13 是研究级回程地图，不应替代第一次学习 D1--D6。

## 37. 三层阅读深度

每章都可以按三层读：

1. **直觉主线**：问题、对象、图和算法输入输出；
2. **完整推导**：条件分布、目标等价、SDE/ODE 和 solver；
3. **论文级边界**：assumption、error decomposition、publication status 与 open problem。

第一次阅读可以跳过部分证明，但不能跳过对象区分。把 $q_t$、$q(x_t\mid x_0)$、trained model 和 sampler output 混在一起，会使后面的公式全部失去语义。

## 38. 统一符号入口

本教程默认：

| 符号                  | 含义                                                    |
| ------------------- | ----------------------------------------------------- |
| $q_{\mathrm{data}}$ | 未知真实数据分布                                              |
| $x_0$               | data endpoint                                         |
| $x_t$               | noise level/time $t$ 的状态                              |
| $x_T$               | 接近简单 prior 的 terminal state                           |
| $q$                 | 设计的 forward/corruption law                            |
| $q_t$               | forward process 在 $t$ 的 aggregate marginal            |
| $p_\theta$          | learned generative/reverse model                      |
| $s_\theta(x,t)$     | learned score field                                   |
| $f,g$               | forward SDE drift 与 diffusion coefficient             |
| $T$                 | discrete final step 或 continuous terminal time，按上下文声明 |

论文 step $t=1,\ldots,T$ 与 Python array index `0,...,T-1` 相差 1；每章代码会单独标注。

## 39. 两个时间方向

forward/noising direction：

$$
x_0\longrightarrow x_T,
\qquad
\text{data}\longrightarrow\text{noise}.
$$

generation/reverse direction：

$$
x_T\longrightarrow x_0,
\qquad
\text{noise}\longrightarrow\text{data}.
$$

连续时间论文还可能用“保持 $t$ 标签但 $dt<0$”或“定义新反向时间 $\tau=T-t$”两套 convention。D4 会显式转换，不能只凭正负号比较公式。

## 40. 说明代码在做什么

d0\_global\_map.py（补充材料暂未公开） 不训练神经网络，只做两件事：

1. 从二维 Gaussian mixture 采样 $x_0$，用 closed-form channel 绘制多个 $\bar\alpha_t$ 的 marginal；
2. 生成七问题轴到章节的静态地图。

核心操作只有：

```python
def gaussian_corruption(x0, alpha_bar, noise):
    return np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * noise
```

它展示 forward sampling 的可计算性，不展示 reverse network 已经学会生成。

## 41. 常见类别错误

1. 把 DDPM 写成 Diffusion 的首次提出；
2. 把 NCSN 写成 score matching 或 DSM 的提出者；
3. 把 forward noising 当作 learned generative model；
4. 把一次从数据换成独立噪声当作有用的 diffusion path；
5. 把 conditional corruption score 当作未知 marginal score 本身；
6. 把 $L_{\mathrm{simple}}$ 与 exact ELBO 视为完全相同；
7. 把 PF ODE 与 reverse SDE 的 same marginal 写成 same path；
8. 把 sampler、distillation 和 learned finite map 全部叫作“加速采样器”；
9. 把 DiT 当作 Diffusion 的替代模型，而不是 backbone；
10. 把 latent representation 的收益归因于 solver；
11. 把 categorical/token state 直接套入 Gaussian score 公式；
12. 把低 FID 当成 likelihood、coverage、privacy 与 safety 的共同证明；
13. 把系统报告、项目页或预印本写成正式同行评审结论；
14. 把教学顺序当历史优先权；
15. 用“统一”“取代”“已经解决”替代对象级比较。

## 42. 本章小结

生成建模不是单一的“画出好看样本”，而是一组 sampling、density、likelihood、condition、representation 与 evaluation 任务。

Diffusion 的核心策略是先规定一个已知 data-to-noise path。这个 path 提供简单 terminal prior、任意噪声层级的训练输入和可计算 conditional target，从而把无配对的全局生成改写为一系列局部学习问题。

历史上，2015 diffusion probabilistic model、2005--2011 score/DSM 前驱、2019 NCSN、2020 DDPM 与 2021 score-SDE 分别补上 path、training target、multi-noise sampling、有效系统和连续接口。后续 latent、transformer、solver、few-step map、discrete state 和 application 并非一条单线，而是对七个耦合问题轴的不同响应。

## 43. 研究式思考题

1. 若一个模型只能采样但不能计算 density，它仍能在哪些任务中被称为成功的生成模型？哪些任务不够？
2. 构造一个一次 corruption，使 $x_1$ 与 $x_0$ 独立。为什么它没有给 reverse learning 增加监督信息？
3. forward path 同时承担“制造 target”和“规定 sampling route”是否必要？哪些现代方法试图拆开这两个职责？
4. VAE、GAN、flow 与 diffusion 分别用什么结构解决 prior/data 无配对问题？
5. 若两个方法使用同一 network，却改变 state representation 与 sampler，应该如何设计 component-matched comparison？
6. 为什么“same one-time marginals”不足以比较 SDE 与 ODE 的路径性质、随机性和 coupling？
7. 选择一篇最新论文，用九组件表拆解。其真正变化在哪个槽位，宣传名称又强调了什么？
8. 把一个低 FID claim 拆成 fidelity、coverage、finite-sample bias、preprocessing 和 data leakage 五项证据义务。
9. 什么证据足以把“代表性节点”升级为“首次提出”？为什么只读一篇综述不够？
10. Diffusion generation 与 Schrödinger Bridge 都使用 stochastic process。增加双端边缘约束和 reference path-space KL 后，优化问题发生了什么根本变化？

## 44. 本章来源与继续阅读

D0 的历史主张优先回到原论文与专门账本：

- D0 历史事实账本（补充材料暂未公开）：版本、页码、可用措辞与被拒绝的 priority claim；
- 五条前驱关系（补充材料暂未公开）：score matching、DSM、Langevin、non-equilibrium path 与 stochastic time reversal；
- [D0 survey/tutorial cross-check](https://doi.org/10.1145/3626235 "官方论文页面")：两份正式综述、CVPR tutorial 与 researcher blog 的教学责任和证据边界；
- 近期专著与基础博客交叉核验（补充材料暂未公开）：*The Principles of Diffusion Models* v2 的 D0--D2 阅读范围，以及 Dieleman 2022 的 score/noise 符号边界；
- 2023--2026 frontier status（补充材料暂未公开）：稳定基础、正式活跃分支、预印本、system/project evidence 与 critique。

两份综述的用途不同。Yang et al. 的 ACM Computing Surveys 文章适合检查采样、likelihood、结构化数据、其他生成模型和跨模态应用是否漏项；Croitoru et al. 的 TPAMI survey 适合检查 DDPM/NCSN/SDE 三框架与视觉任务分类。它们不替代 2015/2019/2020/2021 原论文，也不承担 2025--2026 方法的正式发表状态。

CVPR 2022 tutorial 用来比较教学顺序，Dieleman 2023 blog 用来比较解释角度。二者分别属于 conference tutorial 与 researcher blog，不作为 theorem 或 priority source。

## 45. 下一步：真正拆开 Forward Diffusion

现在我们只知道“已知 corruption 可以制造监督”，还没有证明：

- 多步 Gaussian chain 为什么能直接采样任意 $x_t$；
- $\beta_t,\alpha_t,\bar\alpha_t$ 分别控制什么；
- shared-noise visualization 与真实 Markov trajectory 有何差别；
- schedule、SNR 与 terminal mismatch 如何影响后续训练和采样。

这些问题从 [D1 前向扩散](/blog/diffusion/d1-forward-diffusion/) 开始正式推导。
