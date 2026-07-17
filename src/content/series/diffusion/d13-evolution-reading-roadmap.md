---
title: Diffusion 技术演进总结与论文阅读路线
description: 按问题—方案—限制重组 Diffusion 技术史、概念依赖和分层论文阅读路线，连接全系列章节。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 13
slug: d13-evolution-reading-roadmap
tags:
  - diffusion
  - history
  - reading-guide
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 总结 D0-D12 的问题链、依赖图和研究入口，区分概念演进、时间先后与可验证的历史归属。
---
## 1. 为什么最后一章不能只是“总结”

D1--D12 已经分别解释 forward process、ELBO、score、SDE、参数化、solver、guidance、architecture、few-step flow、discrete diffusion、应用和理论安全。如果最后只把这些名词再列一遍，读者仍然不知道：

- 哪些变化解决的是同一个问题；
- 哪些名字相似但优化对象不同；
- 哪些方法在概念上依赖前者，哪些只是发表得更晚；
- 开始一个研究题目时，究竟应该先读哪几篇；
- 2025--2026 的新结果是稳定基础、正式活跃分支、预印本还是系统报告。

本章因此承担三件事：

1. 把历史改写成**问题链**；
2. 把章节改写成**概念依赖图**；
3. 把参考文献改写成**研究路线**。

它不新增核心 theorem，也不假装已经解决所有历史优先权争议。

## 2. 三种顺序必须分开

同一组工作至少有三种合法排列。

**发表/历史顺序**回答：某个问题在什么时候被提出、哪些公开版本先后出现、正式发表状态是什么。

**概念依赖顺序**回答：理解公式 $B$ 前必须掌握哪些对象 $A$。例如先理解 DSM，再理解 score-SDE，不意味着 DSM 的所有论文都在 score-SDE 之前发表。

**研究阅读顺序**回答：为解决一个具体问题，怎样用最短路径达到工作前沿。它可以跳过某些历史细节，也可以先读一篇教学综述再回原论文。

三者混在一起会产生典型错误：“后发表”被写成“概念上更一般”，“教学上先讲”被写成“历史上最早”，“模型榜单顺序”被写成“理论替代关系”。

## 3. 本章的历史措辞纪律

D13 使用“代表性节点”“重新组织了某个问题”“提供了广泛采用的接口”。D0 历史事实账本（补充材料暂未公开） 已完成本教程所需的专项核验；但账本同时表明，原文已读不自动证明全领域优先权，因此本章仍不轻易写：

- 首次提出；
- 首次证明；
- 完全统一；
- 彻底取代；
- 已经解决。

原因不是回避判断，而是这些词需要比一般技术说明更强的证据：原始首版、版本时间、相关前驱、正式发表和至少一个独立交叉来源。

2025--2026 工作则必须附证据类型：formal paper、preprint、system report、project page 或 critique。

## 4. 七条耦合的问题轴

![Diffusion 的七条问题轴](/images/diffusion/d13_problem_solution_map.png)

Diffusion 的演进可以投影到七条轴：

| 轴            | 持续问题                                           | 本教程主章节       |
| ------------ | ---------------------------------------------- | ------------ |
| trainability | 如何把复杂反向过程变成稳定监督目标                              | D2--D5       |
| quality      | likelihood、fidelity、coverage 和 validity 如何协调   | D2、D5、D8、D12 |
| speed        | 如何从多步积分走到 few/one-step                         | D6、D9        |
| control      | class/text/layout/source/measurement 如何进入生成    | D7           |
| scale        | 分辨率、模型、数据与模态如何扩展                               | D8、D11       |
| state space  | latent、token、graph、trajectory、mixed state 怎么建模 | D8、D10、D11   |
| guarantees   | loss、metric、privacy/safety claim 到底支持什么        | D12          |

每条轴都不是独立排行榜。latent representation 同时影响 quality、speed 与 scale；few-step map 同时改变 speed、training object 与 guarantee；科学应用同时改变 state space、architecture 与 evaluation。

## 5. Trainability：从反向链到可监督 target

早期核心难题是：真实数据分布未知、normalizing constant 难算、反向转移也未知，如何训练？

[D1](/blog/diffusion/d1-forward-diffusion/) 的 forward corruption 先构造可采样的条件分布

$$
q(x_t\mid x_0).
$$

[D2](/blog/diffusion/d2-ddpm-objective/) 再用 conditioned posterior 与 variational bound 把 reverse learning 分解为 Gaussian KL，并映射到 noise prediction。

[D3](/blog/diffusion/d3-score-matching/) 说明 denoising target 的本质：条件 score 的 regression optimum 等于 marginal score，

$$
\mathbb E[\nabla_{x_t}\log q(x_t\mid x_0)\mid x_t]
=\nabla_{x_t}\log q_t(x_t).
$$

[D5](/blog/diffusion/d5-parameterization-training-design/) 则显示“可训练”不是一个二元属性。parameterization、time proposal、loss weighting、preconditioning 与 noise schedule 共同决定 optimization conditioning。

遗留边界由 [D12](/blog/diffusion/d12-theory-evaluation-safety/) 补上：低 empirical loss 不自动给 population score error，更不自动给终端生成分布误差。

## 6. Quality：目标、表示和评估共同决定“好”

“提高质量”至少有四种不同含义：

$$
\text{likelihood},\quad
\text{fidelity},\quad
\text{coverage},\quad
\text{task validity}.
$$

Improved variance、hybrid objective、schedule/weighting 改善的是训练和 likelihood/sample tradeoff；U-Net/ADM、latent diffusion 与 DiT 改善的是 representation/backbone capacity 与 scale；guidance 改变 conditional fidelity--diversity；D12 的 FID/KID/precision-recall 说明评估标量本身也有 bias。

所以“模型质量进步”必须写成：

$$
\text{method configuration}
+\text{sampling configuration}
+\text{evaluation protocol}.
$$

不注明 prompt、guidance、NFE、sample count、feature extractor 和 domain validity，质量比较是不完整的。

## 7. Speed：先分清在加速什么

采样速度有三种不同对象。

**Numerical solver** 固定学到的 local field，只改变积分：

$$
\dot x_t=v_\theta(x_t,t)
\quad\Longrightarrow\quad
x_{t_{i-1}}=\Phi_h(x_{t_i};v_\theta).
$$

**Learned finite map** 直接学习

$$
F_\theta(x_t,t,s)\approx x_s,
$$

或学习沿 interval 的 average velocity/consistency relation。

**Distribution distillation** 不要求逐 path 对齐，而是让 student distribution 或其 gradient 接近 teacher。

[D6](/blog/diffusion/d6-sampling-solvers/) 属于第一类；[D9](/blog/diffusion/d9-few-step-flow/) 主要处理后两类。把它们只按 NFE 排序，会掩盖 training cost、single-call architecture、stochasticity 和 guarantee 差异。

## 8. Control：条件改变的是 target，不只是输入通道

[D7](/blog/diffusion/d7-guidance-conditioning-editing/) 的主问题是：想采样的究竟是

$$
p(x),\quad p(x\mid c),\quad
p(x\mid y),\quad
p(x\mid x_{\mathrm{source}},m,c)
$$

中的哪一个？

Classifier guidance 通过外部 $\nabla_x\log p(c\mid x_t)$ 修改 score；CFG 用 conditional/unconditional predictor difference；cross-attention、ControlNet 和 adapter 改变 architecture interface；inverse problems 再加入 measurement likelihood。

它们都可表现为“加条件”，但 normalization、approximation 和 preservation 边界不同。高 guidance 改善 alignment 时，可能牺牲 diversity、calibration 与 image quality。

## 9. Scale：representation、backbone、objective 三件事

[D8](/blog/diffusion/d8-architecture-representation/) 强制区分：

1. **representation**：pixel、VAE latent 或其他 token；
2. **backbone**：U-Net、Transformer/DiT、U-ViT；
3. **path/objective**：diffusion、flow matching、其他 transport。

Latent diffusion 主要通过压缩空间减少计算；DiT 用 token transformer 承载 time-conditioned field；SiT 等工作可在 Transformer backbone 上换成 interpolation/flow objective。

因此“Transformer 取代 Diffusion”是类别错误。Transformer 是函数逼近器，Diffusion/Flow 描述训练路径与生成 dynamics。

## 10. State space：Gaussian 公式并不统治一切

连续图像 Diffusion 常用

$$
x_t=a_t x_0+b_t\epsilon.
$$

但 token/graph 使用 categorical transition matrix 或 CTMC generator。D10 的基本对象是

$$
q_t=q_0Q_{1:t},
\qquad
\frac{dp_t}{dt}=p_tQ_t,
$$

以及 reverse rate/density ratio，而不是 Euclidean gradient。

[D10](/blog/diffusion/d10-discrete-diffusion-language-models/) 进一步说明 language model 还要处理 sequence factorization、length、remasking、block structure 与 KV cache。离散步数少不自动等于 wall-clock 快。

## 11. Guarantees：批评工作也是技术演进

技术史不应只保留“模型更强”的论文。D12 中的以下工作同样改变了研究对象：

- CleanFID 说明 resize/compression 足以破坏可比性；
- CMMD 质疑 Inception/Gaussian assumptions；
- extraction/membership work 证明高质量不等于不记忆；
- Stable Bias/Unsafe Diffusion/Ring-A-Bell 说明默认安全和 concept removal 可被弱协议夸大；
- convergence work 说明低 score error 到 sampling 仍需 initialization、early stopping 与 discretization。

发现评价对象错误，本身就是一项方法学进步。

## 12. 七轴图应该怎样读

图的每一行都遵循

$$
\text{persistent problem}
\rightarrow
\text{representative response}
\rightarrow
\text{retained frontier}.
$$

这比“年份--模型名”多出两个信息：为什么会出现这条路线，以及它没有解决什么。

例如 solver/consistency/flow-map 都响应 speed，但 retained frontier 是“相同质量点上的 latency 与 guarantee”；LDM/DiT 响应 scale，但 retained frontier 是 data/compute/decoder bottleneck；privacy/safety audit 响应 guarantee，但仍缺 end-to-end system evidence。

## 13. 代表性时间线：节点是问题转折，不是冠军榜

| 公开时期       | 问题转折                                                   | 代表性来源职责                       |
| ---------- | ------------------------------------------------------ | ----------------------------- |
| 2015       | forward corruption + learned reversal 成为生成构造           | D1/D2 中 Sohl-Dickstein source |
| 2019--2020 | multi-noise DSM 与稳定 DDPM objective/system              | D2/D3                         |
| 2021       | score-SDE/PF-ODE 连续接口；DDIM 类采样视角                       | D4/D6                         |
| 2021--2023 | guidance/control 与高阶采样加速                               | D6/D7                         |
| 2022--2024 | latent/Transformer scale；distillation/flow/consistency | D8/D9                         |
| 2021--2026 | categorical/CTMC/masked language diffusion             | D10                           |
| 2021--2025 | 应用反馈出 time/symmetry/horizon/mask                       | D11                           |
| 2022--2026 | weak-assumption theory、metric、memory/safety audit      | D12                           |

这里故意使用“公开时期”和“代表性”。D0 已核验核心节点的原始首版、正式发表与 priority wording；时间线继续采用保守措辞，因为它表达问题转折，不承担穷尽所有前驱的优先权裁决。

## 14. 2015 节点：构造一个可控的破坏过程

2015 工作的重要性在本教程中不是“已经拥有后来 DDPM 的全部工程形式”，而是建立了一个可计算方向：

1. 设计逐步破坏数据的 forward Markov chain；
2. 让终点接近简单分布；
3. 学习反向 transition；
4. 用 variational objective 训练。

问题从“直接拟合复杂数据密度”变成“学习一连串局部逆过程”。代价是长链与训练/采样成本。

## 15. 2019--2020 节点：score supervision 与 DDPM 系统闭环

NCSN 路线通过多个 noise level 的 DSM 缓解 data manifold 与 low-density Langevin 问题；DDPM 把 Gaussian forward/posterior、epsilon prediction 和 U-Net system 组合成稳定图像生成流程。

两条路线在数学上通过

$$
s_t(x_t)
=-\frac{1}{\sigma_t}
\mathbb E[\epsilon\mid x_t]
$$

相接，但历史和实现关注点不同：score route 强调 density-gradient field/Langevin，DDPM route 强调 variational chain/noise prediction。

## 16. 2021 节点：连续时间接口与 path/marginal 边界

score-SDE 将 VP、VE、sub-VP、reverse SDE 与 PF ODE 放进共同连续语言：

$$
dX_t=f(X_t,t)dt+g(t)dW_t,
$$

$$
d\bar X_t=
\left[f(\bar X_t,t)-g(t)^2s_t(\bar X_t)\right]dt
+g(t)d\bar W_t.
$$

PF ODE 提供 same-marginal deterministic route。这里的“统一”只指特定 scalar-diffusion/marginal interface，不意味着 stochastic path 与 deterministic trajectory 相同，也不涵盖所有 flow/Bridge optimization。

## 17. 2021--2023 节点：采样与控制成为独立设计轴

DDIM、PNDM/DEIS、DPM-Solver 等路线把 sampler 从“照搬 ancestral chain”变成 numerical design；classifier guidance/CFG/cross-attention 把 conditional generation 从 class label 扩展到 text 与结构控制。

这两个方向相互作用：

- guidance 增大有效 field 的 stiffness；
- solver 的 low-NFE behavior 依赖 guidance scale；
- editing/inverse tasks 还需 preservation/likelihood gradient；
- quality--speed comparison 必须固定条件协议。

## 18. 2022 节点：latent、EDM design space 与高阶 solver

2022 前后，多条问题线同时成熟：

- latent diffusion 用 learned compression 降低高分辨率成本；
- EDM 拆开 noise distribution、preconditioning、network output 与 sampler；
- DPM-Solver/EDM sampler 把 semi-linear structure 与高阶 integration 引入主流实践；
- progressive distillation 开始直接学习少步 student。

它们共同响应 scale/speed，但优化对象不同。将它们写成一条“模型版本升级史”会失去真正的因果关系。

## 19. 2023--2024 节点：backbone、flow 与应用压力

DiT/U-ViT 说明 Transformer 可以承载 diffusion field；ControlNet/IP-Adapter 等将 spatial/reference control 模块化；Consistency/Flow Matching/Rectified Flow 让 few-step generation 转向 finite map 与 transport path；video、3D、science、robotics 则把 temporal consistency、SDS、equivariance 和 action horizon 推回通用方法。

这一时期最值得保留的不是模型名数量，而是四种新接口：

$$
\text{token backbone},\quad
\text{modular condition},\quad
\text{learned finite map},\quad
\text{domain constraint}.
$$

## 20. 2024--2026：正式活跃分支与前沿候选分开

到检索截止日，可以分为：

**稳定基础**：DDPM/DSM/SDE、主要参数化、基本 solver、CFG/LDM/DiT、FID/KID/PR 定义。

**有正式锚点的活跃分支**：consistency/flow matching、discrete diffusion LM、video/world/science/control、manifold/PF-ODE theory、holistic/safety audits。

**前沿预印本**：部分 MeanFlow/shortcut/flow-map、2025--2026 diffusion LM、Li--Yan rate、Azangulov manifold、Merger--Goldt local coverage 等。

**系统证据**：Sora、Movie Gen、Cosmos 等 project/system report，只承担组织公开的系统和 capability claim。

新近不等于稳定，closed system 不等于正式 theorem。

## 21. 概念依赖图：不要按发布时间连箭头

![Diffusion 跨章概念依赖](/images/diffusion/d13_concept_dependency.png)

图的中部是连续主干 D1--D6；D7 conditioning 与 D8 architecture 是横向接口；D9 改变 learned object；D10 分叉到 discrete mathematics；D11 将领域约束反馈给方法；D12 审计每条边；Bridge 再加入 endpoint marginals 与 reference path-space KL。

箭头表示理解/对象依赖，不表示论文引用关系或历史优先权。

## 22. 连续主干：六个对象不能跳步混用

| chapter | core object                      | common category mistake               |
| ------- | -------------------------------- | ------------------------------------- |
| D1      | $q(x_t\mid x_0)$、aggregate $q_t$ | shared-noise coupling 当真实 Markov path |
| D2      | posterior、reverse model、ELBO     | simple MSE 当 exact likelihood         |
| D3      | marginal score、DSM optimum       | conditional score 当 marginal score    |
| D4      | path law、reverse SDE、PF ODE      | same marginal 当 same path             |
| D5      | parameterization/weight/proposal | 换 parameterization 当换 target          |
| D6      | numerical trajectory/grid/NFE    | theoretical order 当 wall-clock        |

这六个对象是后续分支的公共语言。

## 23. 参数化、架构和路径不应绑成一个模型名

一个现代系统可以写成组件组

$$
\mathcal M=
(\text{state representation},
\text{probability path},
\text{prediction target},
\text{backbone},
\text{condition interface},
\text{sampler}).
$$

例如 latent + linear VP path + v-prediction + DiT + cross-attention + DPM-Solver 是一个组合，不是一条不可拆的理论。

组件化阅读能解释为什么论文常只替换一项，却在系统名称上看起来像全新模型。

## 24. Numerical solver 与 learned finite map

solver 路线的误差对象是

$$
\|\Phi_{t\to s}(x)-\widehat\Phi^{\mathrm{num}}_{t\to s}(x)\|.
$$

learned-map 路线的误差对象是

$$
\|\Phi_{t\to s}(x)-F_\theta(x,t,s)\|,
$$

还可能通过 consistency、teacher distillation 或 distribution objective 间接训练。

两者都能降低 NFE，但 learned map 引入新的 approximation/generalization error。把它称为“更高阶 solver”通常不准确。

## 25. PF ODE、Flow Matching、Rectified Flow 与 stochastic interpolants

这些路线共享 continuity equation 语言

$$
\partial_t p_t+\nabla\cdot(p_tv_t)=0,
$$

但 velocity 的来源不同：

- PF ODE：由 diffusion score 构造；
- Flow Matching：从选定 probability path 的 conditional velocity 回归 marginal velocity；
- Rectified Flow：强调 coupling、straightness 与 reflow；
- stochastic interpolants：从 interpolation/coupling 推出 velocity/score identity。

“都能得到 ODE”不等于训练目标、path、coupling 和 endpoint map 相同。

## 26. Consistency、flow map、MeanFlow 与 distribution distillation

Consistency family 关注沿 dynamics 的状态是否映到同一 endpoint；flow-map family 直接学习 $(s,t)$ 间映射与 composition；MeanFlow 类 average velocity identity 连接 interval displacement 与 local field；DMD/ADD/DMD2 则通过 distribution/adversarial signal 训练 few-step student。

研究者阅读时应填写：

| field          | question                                                                          |
| -------------- | --------------------------------------------------------------------------------- |
| learned object | local field、endpoint map、finite map、average velocity 还是 distribution student？     |
| supervision    | exact pair、teacher trajectory、self-consistency、adversarial/distribution gradient？ |
| coupling/path  | 谁决定中间状态？                                                                          |
| stochasticity  | map 是否保留随机 kernel？                                                                |
| composition    | finite map 是否满足 semigroup/consistency？                                            |
| guarantee      | pointwise、distribution、有限网格还是连续极限？                                                |

这比只问“一步还是两步”更有研究价值。

## 27. Discrete branch：从 gradient 转到 ratio/rate/generator

D10 不能作为连续主干的一个小附录。离散过程的核心对象包括：

$$
Q_{1:t}=Q_1Q_2\cdots Q_t,
$$

$$
\bar Q_t(i,j)\quad\text{和}\quad
q(x_{t-1}\mid x_t,x_0),
$$

连续时间下则是 generator $Q_t$ 与 reverse rate

$$
\bar q_t(y,x)
=q_t(x,y)\frac{p_t(x)}{p_t(y)}.
$$

SEDD/score entropy 学 density ratio；MDLM 利用 mask/substitution constraints；discrete flow/generator matching 学概率质量的 continuity/Kolmogorov dynamics。

因此 Gaussian epsilon-prediction 不是跨 state space 的通用定义。

## 28. Language Diffusion 的真正比较对象

Diffusion LM 常被概括为“并行生成”，但系统比较至少包含：

| object        | question                                             |
| ------------- | ---------------------------------------------------- |
| factorization | full masked、arbitrary order、block AR、hybrid？         |
| step          | 一步更新多少 token，是否 remask？                              |
| length        | 固定长度、EOS、length model？                               |
| likelihood    | exact、ELBO、bound、estimate？                           |
| cache         | 是否可复用 KV，block 内外如何复用？                               |
| latency       | token step、network call、wall-clock 哪一个？              |
| quality       | perplexity、generation benchmark、reasoning、diversity？ |

这也是为什么 2025--2026 DLM 论文必须和 efficiency critique 同行阅读。

## 29. 应用不是终点，而是方法压力测试

[D11](/blog/diffusion/d11-representative-applications/) 用统一五元组组织应用：

$$
\text{task--condition--state--output--metric}.
$$

应用反馈出的通用概念包括：

| domain      | method feedback                                        |
| ----------- | ------------------------------------------------------ |
| video/world | per-token noise、temporal latent、causal rollout         |
| audio       | semantic/acoustic multimodal representation            |
| 3D/SDS      | frozen prior gradient through renderer                 |
| science     | invariance/equivariance、mixed state、physical validity  |
| control     | trajectory reward tilt、action horizon、receding horizon |
| imputation  | arbitrary condition/target masks                       |

这比“Diffusion 可用于很多领域”更具体：领域约束改变了状态、条件、objective 和评估。

## 30. D12 不是最后的伦理附录，而是每条边的审计层

D12 回答每条依赖边是否成立：

- D2/D3：empirical denoising loss 是否变成 population score error；
- D4：reverse-time/Girsanov 假设是否成立；
- D6/D9：discretization、NFE 与 latency 是否分开；
- D8：latent/backbone approximation 是否隐藏；
- D10：离散 likelihood/factorization 是否可比；
- D11：domain metric 是否真正衡量 validity；
- deployment：memorization、privacy、bias、safety、provenance 是否有 threat model。

理论和安全不是完成模型后“再加的一章”，而是方法声明的 proof obligation。

## 31. Diffusion、Flow、Consistency 与 Bridge 的边界

可以共享的语言：

- probability path $p_t$；
- velocity/score/drift；
- continuity/Fokker--Planck equation；
- coupling 与 endpoint map；
- SDE/ODE simulation。

不可压扁的优化对象：

| family                    | primary object                            |
| ------------------------- | ----------------------------------------- |
| score/diffusion           | noising path 的 score 与 reverse dynamics   |
| flow matching             | 选定 path/coupling 的 marginal velocity      |
| consistency/flow map      | finite-time invariant/map                 |
| distribution distillation | teacher/student distribution discrepancy  |
| Schrödinger Bridge        | reference path law 上满足双端边缘的 KL projection |

Schrödinger Bridge 的典型问题是

$$
\min_{P:\,P_0=\mu_0,\;P_T=\mu_T}
\operatorname{KL}(P\|R).
$$

若一个方法没有 reference path law $R$、path-space KL 和 endpoint constraints，就不能只因使用 Brownian interpolation 而自动称为 SB。

## 32. 现代系统的组件审计表

读一个新模型时，先填组件而不是记品牌名：

| component    | candidate choices                                           |
| ------------ | ----------------------------------------------------------- |
| state        | pixel、latent、token、graph、trajectory、mixed                   |
| forward/path | VP/VE/sub-VP、interpolation、categorical kernel、CTMC          |
| target       | epsilon、x0、v、score、velocity、ratio、finite map                |
| backbone     | U-Net、DiT/U-ViT、sequence transformer、equivariant net        |
| condition    | classifier、CFG、attention、adapter、measurement、mask           |
| sampler      | ancestral、DDIM、ODE/SDE solver、tau-leap、learned map          |
| objective    | ELBO/DSM、flow matching、consistency、distillation、adversarial |
| evaluation   | likelihood、quality/coverage、alignment、validity、latency、risk |
| evidence     | formal、preprint、system report、code、model card、critique      |

这张表能快速发现论文到底替换了什么，以及哪些变化来自 baseline。

## 33. 阅读路线图

![按研究问题选择 Diffusion 阅读路线](/images/diffusion/d13_reading_routes.png)

路线图不是课程先修树，而是“为一个研究问题达到可工作状态”的最短路径。每条路线最后都经过 D12，是因为研究阅读不仅要知道方法，还要知道 claim 的 metric 和证据边界。

## 34. Route A：第一次系统学习

推荐顺序：

$$
\mathrm{D0}
\to\mathrm{D1}
\to\mathrm{D2}
\to\mathrm{D3}
\to\mathrm{D4}
\to\mathrm{D6}
\to\mathrm{D7/D8}
\to\mathrm{D12}.
$$

为什么暂时跳过 D5？第一次阅读先建立 forward--objective--score--SDE--sampler 闭环，再回 D5 理解 parameterization/weighting，会比在主干尚未成形时进入 design space 更清楚。

第一轮不要求掌握 D9--D11 所有分支；先能回答“模型学什么、怎样采样、怎样评估”。

## 35. Route B：训练、参数化与 schedule

最短路径：

$$
\mathrm{D1}\to\mathrm{D2}\to\mathrm{D5}\to\mathrm{D6}\to\mathrm{D12}.
$$

论文锚点按问题读：

- Improved DDPM：variance 与 hybrid objective；
- VDM：SNR/log-SNR measure；
- P2/Min-SNR：time weighting 与 multi-task conflict；
- EDM/EDM2：preconditioning、noise distribution、magnitude control；
- schedule critique：terminal SNR 与 train/inference consistency。

每篇记录 prediction target、weight、time proposal、schedule、variance、sampler 是否联动。不要只复制 loss 名称。

## 36. Route C：sampler 与 few-step generation

先走 solver track：

$$
\mathrm{D4}\to\mathrm{D5}\to\mathrm{D6}.
$$

阅读顺序可用：

DDIM -> PNDM/DEIS -> DPM-Solver/DPM-Solver++ -> UniPC/EDM -> grid/solver extensions。

再走 learned-map track：

$$
\mathrm{D9}\to\mathrm{D12}.
$$

Progressive Distillation -> Consistency Models/Training -> Flow Matching/Rectified Flow -> flow map/Shortcut/MeanFlow -> DMD/ADD/DMD2。

对每项记录 learned object、teacher/path/coupling、stochasticity、NFE、training cost、quality point 与 guarantee。

## 37. Route D：条件、编辑与 inverse problems

最短路径：

$$
\mathrm{D2}\to\mathrm{D7}\to\mathrm{D8}\to\mathrm{D11}\to\mathrm{D12}.
$$

从 classifier guidance/CFG 理解 score composition；再看 cross-attention、ControlNet/T2I-Adapter/IP-Adapter 的 architecture condition；再看 SDEdit、Prompt-to-Prompt、inversion 和 DPS/DDRM。

四个持续问题：

1. condition target 是否 normalized；
2. source preservation 如何定义；
3. measurement likelihood gradient 是否精确；
4. guidance/solver/seed 的 robustness 如何评估。

## 38. Route E：architecture 与 scaling

最短路径：

$$
\mathrm{D2}\to\mathrm{D5}\to\mathrm{D8}\to\mathrm{D11}\to\mathrm{D12}.
$$

建议按组件读：

U-Net/DDPM -> ADM -> LDM -> DiT/U-ViT -> SiT/representation alignment。

记录 representation bottleneck、patch/tokenization、time/condition injection、normalization、scaling variables、training compute 与 decoder error。

不要把更大的 backbone 与更好的 diffusion objective 混成一个原因。

## 39. Route F：discrete diffusion 与 language

最短路径：

$$
\mathrm{D3}\to\mathrm{D4}\to\mathrm{D10}\to\mathrm{D12}.
$$

建议路线：

Multinomial/D3PM -> ARDM -> tauLDR/CTMC -> SEDD -> MDLM -> DiffusionBERT/PLAiD/LLaDA/Block Diffusion -> discrete flow/generator matching。

每篇先写 corruption kernel/rate、reverse ratio、factorization、likelihood，再看 benchmark。若论文只给 token-level step 数而不报告 wall-clock/cache，应把 efficiency claim 降级。

## 40. Route G：应用研究

不要从“我想用 Diffusion 做领域 X”直接跳到一个最新模型。先填：

$$
(\text{task},\text{condition},\text{state},\text{output},\text{metric}).
$$

然后按 D11 的五类证据门槛收集：

1. foundation/turning point；
2. recent representative；
3. survey/tutorial；
4. public implementation；
5. limitation/evaluation。

领域路线必须回到 D12 的 validity 和 system-cost audit。

## 41. Route H：理论、评估与安全

理论最短路径：

$$
\mathrm{D3}\to\mathrm{D4}\to\mathrm{D6}\to\mathrm{D9}\to\mathrm{D12}.
$$

代表性 theorem reading：

De Bortoli manifold -> Lee--Lu--Tan general data -> Benton near-d-linear -> Oko minimax -> Huang PF-ODE -> clearly labeled frontier。

评估：

FID -> KID -> precision/recall -> CleanFID -> HEIM/CMMD/domain validity。

隐私安全：

replication -> extraction/membership -> DP -> bias/unsafe audit -> concept-removal stress test -> watermark/model card/provenance。

## 42. Route I：进入 Schrödinger Bridge

进入 Bridge 前先完成：

1. D4：path law、time reversal、Fokker--Planck；
2. D9：coupling、stochastic interpolants、flow boundary；
3. D12：path KL、evidence discipline；
4. Bridge B0--B4：Brownian/diffusion bridge、reciprocal process、Schrödinger problem；
5. Bridge B5 以后：entropic OT、control、algorithms、neural SB。

不要从含糊的“diffusion bridge”进入。先判断对象是 conditioned diffusion path、Brownian bridge，还是双端边缘约束下的 path-space entropy projection。

## 43. 一篇新论文的十字段阅读卡

读任何新 Diffusion 论文，先填：

| field      | question                                           |
| ---------- | -------------------------------------------------- |
| problem    | 它明确修复哪个 bottleneck？                                |
| object     | 学 score、velocity、ratio、map、policy 还是 distribution？ |
| path/state | 连续、离散、latent、mixed？                                |
| condition  | target distribution 怎样定义？                          |
| objective  | population target 与 empirical estimator 分别是什么？     |
| algorithm  | 训练与采样各需要什么？                                        |
| comparison | baseline 是否同 quality/compute/protocol？             |
| guarantee  | assumptions、metric、endpoint、omitted layer？         |
| evidence   | formal/preprint/system/code/model card/critique？   |
| limitation | 哪个失败将推动下一条路线？                                      |

十字段卡比“摘要--方法--实验”三段摘抄更能支持研究。

## 44. Annotated bibliography：基础主干

### Forward and objective

- Sohl-Dickstein et al. 2015：forward diffusion + learned reversal 的早期主节点；读 D1/D2 note，不把它写成后来 DDPM 完整系统。
- Ho et al. 2020：DDPM Gaussian chain、epsilon objective 与 image system；同时读 D2 中 simple loss/VLB 边界。
- Nichol--Dhariwal 2021：learned variance、hybrid objective 与 step reduction。
- Kingma et al. VDM：SNR/log-SNR 与 variational design。

### Score and continuous time

- Hyvärinen 2005、Vincent 2011：score matching/DSM 原理；
- Song--Ermon 2019/2020：multi-noise score generation；
- Song et al. 2021 score-SDE：reverse SDE/PF ODE 连续接口；
- Tang--Zhao technical tutorial：导航，不替代原 theorem。

## 45. Annotated bibliography：sampling、control 与 architecture

### Sampling

- DDIM：non-Markovian forward family 与 deterministic/stochastic update；
- PNDM/DEIS：multistep/exponential-integrator view；
- DPM-Solver/DPM-Solver++：semi-linear diffusion ODE solver；
- UniPC/EDM：predictor-corrector/design-space implementation；
- recent solver papers：只在版本和正式状态明确时进入 active/frontier。

### Control

- classifier guidance 与 CFG：score composition 和 condition/dropout；
- LDM：cross-attention 与 latent condition；
- ControlNet、T2I-Adapter、IP-Adapter：modular spatial/reference control；
- SDEdit/inversion/DPS：source/measurement constraints。

### Architecture

- DDPM/ADM U-Net：multi-resolution denoiser；
- LDM：representation compression；
- DiT/U-ViT：token backbone；
- SiT/REPA 等：backbone 与 objective/representation alignment 分开读。

## 46. Annotated bibliography：few-step、discrete 与 applications

### Few-step/flow

- Progressive Distillation：teacher trajectory 到 fewer-step student；
- Consistency Models/Training：endpoint consistency；
- Flow Matching/Rectified Flow/Stochastic Interpolants：path/velocity/coupling；
- Shortcut/MeanFlow/flow-map：finite interval map/average field frontier；
- DMD/ADD/DMD2：distribution/adversarial distillation 旁支。

### Discrete

- Multinomial diffusion/D3PM：categorical transition/posterior/ELBO；
- tauLDR：CTMC and tau-leaping；
- SEDD/MDLM：ratio/score entropy/masked objective；
- LLaDA/Block/Scaling DLLM：large language system and efficiency boundary；
- discrete flow/generator matching：generator/continuity branch。

### Applications

用 D11 的七条 route note，不在 D13 重复模型清单。每条路线保留 method feedback 与 limitation evidence。

## 47. Annotated bibliography：theory、metrics 与 governance

### Convergence

- De Bortoli：manifold target、early stopping、Wasserstein；
- Lee--Lu--Tan：general data distribution；
- Benton et al.：near-d-linear reverse-SDE bound；
- Oko et al.：minimax distribution estimation；
- Huang et al.：PF-ODE continuous/discrete theory；
- 2025--2026 preprints：只承担 frontier。

### Metrics

- FID/KID/precision-recall：定义与 finite-sample/coverage decomposition；
- CleanFID/CMMD：implementation 与 representation critique；
- HEIM/VBench/PoseBusters：conditional/temporal/domain validity。

### Governance

- replication/extraction/membership：memorization evidence；
- DP diffusion：formal privacy route；
- Stable Bias/Unsafe Diffusion/Ring-A-Bell：bias/safety/adaptive audit；
- Stable Signature/model card：provenance and versioned system evidence。

## 48. 当前最重要的开放问题

1. 如何从 finite empirical denoising loss 得到 end-to-end trained sampler guarantee？
2. 如何把 architecture approximation、optimizer、data generalization 与 discretization 放进同一非空泛 bound？
3. few-step learned map 的 pointwise、path 与 distribution guarantee 如何统一而不混淆？
4. quality--NFE--latency--energy 怎样形成跨架构公平 Pareto protocol？
5. discrete LM 的 length、factorization、cache 与 likelihood 怎样共同评估？
6. world model 的 temporal realism 如何转成 causal/control utility？
7. scientific generation 的 symmetry 保证如何连接 physical/experimental validity？
8. local data coverage、duplicates、conditioning 与 memorization 的因果关系是什么？
9. benchmark 如何动态更新以减少 saturation 和 training leakage？
10. concept removal 如何从 static prompt suppression 走向 adaptive、可证明的 unlearning？
11. watermark/model card/data card 如何组成可验证 provenance chain？
12. Diffusion、Flow、Consistency、Bridge 的统一语言如何保留优化对象差异？

## 49. 常见类别错误

1. 把发表时间线当概念依赖图；
2. 把教学顺序当历史优先权；
3. 把 epsilon、v、x0 parameterization 当不同生成分布；
4. 把 DiT/Transformer 当 Diffusion 的替代品；
5. 把 latent representation 改进归因于 sampler；
6. 把 PF ODE same marginal 当 same path；
7. 把 Flow Matching 当所有 flow/diffusion 方法的同义词；
8. 把 consistency model 当普通高阶 solver；
9. 把 distribution distillation 当逐 path imitation；
10. 把 stochastic interpolation 自动称作 Schrödinger Bridge；
11. 把 categorical diffusion 套进 Gaussian score 公式；
12. 把 token-level parallel 当 wall-clock acceleration；
13. 把 NFE 少当 energy 低；
14. 把 system report 当可复现 formal paper；
15. 把 preprint 新结论写成稳定基础；
16. 把 project page 用作 theorem source；
17. 只读 model paper，不读 metric/limitation critique；
18. 只看 benchmark mean，不看 protocol/uncertainty；
19. 把 application list 当 task-interface analysis；
20. 把 symmetry/equivariance 当 scientific validity；
21. 把 extraction case 当所有输出都是复制；
22. 把 membership attack 失败当 DP；
23. 把 concept suppression 当数据删除；
24. 把 watermark 当真假/版权判断；
25. 用“统一”“取代”“首次”代替对象级比较。

## 50. 章节小结

Diffusion 的技术史不是一条单线，而是七条耦合问题轴。

trainability 把复杂生成问题变成 forward corruption、posterior、DSM 与参数化；quality 把 objective、representation、guidance 和 metric 串在一起；speed 分成 numerical solver 与 learned finite map；control 改变 conditional target；scale 拆成 representation/backbone/path；state space 分叉到 discrete/CTMC 和 domain-specific mixed geometry；guarantees 则审计每一条边。

概念依赖也不是发表顺序。D1--D6 构成连续主干，D7/D8 是横向接口，D9 改变 learned object，D10 改变数学状态空间，D11 把领域压力反馈回来，D12 提供 theorem/metric/safety proof obligations，Bridge 再加入双端边缘与 reference path-space KL。

最有效的阅读方式不是“把论文全读完”，而是先明确研究问题，再选择最短 chapter route，并为每篇论文填写 object、path、objective、algorithm、metric、evidence 和 limitation。

## 51. 研究式思考题

1. 选择一个新方法，把它拆成 state、path、target、backbone、condition、sampler 九组件；哪些性能来自真正的新组件？
2. 构造一个例子，说明发表更晚的方法在概念上反而依赖更早的一个旁支，而不是主干后继。
3. 对同一 trained score，比较高阶 solver 与 consistency student 的误差账本；哪些项可共享，哪些新增？
4. Flow Matching 与 PF ODE 都给 velocity field。设计一张 assumption/objective/path/metric 表，避免“统一”过度。
5. 对 masked Diffusion LM，设计同时报告 likelihood、generation quality、length、KV cache 与 wall-clock 的协议。
6. 若一个 DiT 系统从 pixel diffusion 换成 latent flow matching，怎样做 component-matched ablation？
7. 把 Diffusion Policy 的 action horizon 放进七问题轴，它同时影响哪些轴？
8. 选择一个 scientific diffusion model，区分 symmetry guarantee、geometric validity、physical validity 和 experimental validity。
9. 设计一个 few-step leaderboard，使 solver、distillation 和 distribution student 在相同 quality point 上可比。
10. 为一篇 2026 preprint 写“可进入正文”和“只能进入前沿框”的判定规则。
11. 什么证据足以把“代表性节点”升级为“首次提出”？列出历史账本所需字段。
12. 画出 generic diffusion、stochastic interpolant 与 Schrödinger Bridge 的 optimization-variable/constraint/reference-law 对比。
13. 如何让动态 benchmark 同时减少 saturation，又保持跨年份可比性？
14. 将 local-coverage memorization 假说转成一个不依赖单一 embedding 的可证伪实验。
15. 设计个人研究阅读路线时，哪些章节必须回读，哪些可以按目标跳过？给出理由。

## 52. 与 D0 闭环：从开篇地图回到研究地图

D13 建立回看全篇的研究地图；[D0](/blog/diffusion/d0-generative-modeling-map/) 则已经完成与它互补的两件事：

1. 对 2015 diffusion model、2019 NCSN、2020 DDPM、2021 score-SDE 及其前驱作有边界的一手历史核验；
2. 把七轴全局地图压缩成第一次进入教程时可理解的开篇路线。

读者可以按 D0 的顺序进入 D1--D6，再按研究问题选 D7--D12，最后回到 D13。历史顺序、概念依赖和研究阅读顺序因此形成闭环，而不被压成一条“新论文取代旧论文”的直线。
