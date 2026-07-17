---
title: 代表性应用：新问题如何反过来改变 Diffusion
description: 比较视频、音频、三维、分子、控制和时间序列任务如何改变状态空间、条件与评价接口。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 11
slug: d11-representative-applications
tags:
  - diffusion
  - applications
  - multimodal
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 覆盖代表性跨模态应用及其证据边界，区分通用扩散机制与任务特定监督、对称性和数据贡献。
---
前十章主要从概率模型内部看 diffusion：forward corruption 是什么、score 怎样学习、reverse dynamics 怎样离散化、条件怎样进入、网络怎样表示、少步生成怎样改变函数接口。现在把视角反过来：当对象不再是一张固定大小的图像，而是一段视频、一段声音、一个三维场景、一个分子、一条机器人轨迹或带缺失值的时间序列时，原有定义中究竟哪一项必须改变？

应用不是理论完成后的“例子列表”。真正有影响力的应用会提出原方法无法回避的新问题：

- 视频迫使模型显式处理时间、压缩和长时一致性；
- text-to-3D 把冻结的二维 score 变成另一个生成器参数上的梯度；
- 分子与材料要求 score 尊重群对称性，并同时处理离散类型与连续坐标；
- 控制把“一个样本”改成可滚动执行的 action horizon；
- 插补把任意 observation mask 变成条件分布的一部分；
- 科学与系统应用迫使我们区分“看起来合理”“指标高”“物理有效”和“实验成立”。

所以本章的主角不是产品，而是这些应用反过来贡献给生成建模的**接口**。

***

## 1. 不从模型名开始：先写五元组

对任意应用，先固定

$$
\mathcal T=(\text{task},\text{condition},\text{state},\text{output},\text{metric}).
$$

这五项分别回答：

1. 要解决的决策或生成问题是什么；
2. 生成时已知哪些信息；
3. forward/reverse process 实际作用在哪个空间；
4. denoising 结束后交付什么对象；
5. 什么证据能说明这个对象完成了任务。

例如“用 diffusion 做机器人”仍然不完整。它可能表示：给定当前视觉观测，在 action sequence 上采样未来 16 步；也可能表示：在 state-action trajectory 上做目标条件规划。二者的条件、状态、输出和闭环执行方式都不同，因而不是换一个数据集这么简单。

![跨应用 task--condition--state--output--metric 矩阵](/images/diffusion/d11_task_interface_matrix.png)

这张矩阵有一个刻意安排：最后一列不是“代表模型”，而是应用带来的 method pressure。读者应优先问“它迫使方法新增了什么对象”，再问论文名。

***

## 2. 同一个 denoising 公式为何不能抹平领域差异

很多系统都可以写出形式相似的目标：

$$
\mathcal L(\theta)
=\mathbb E_{t,x_0,\epsilon,c}
\left[
w(t)\left\|\epsilon-\epsilon_\theta(x_t,t,c)\right\|^2
\right].
$$

但这个公式隐藏了最关键的建模决策：

- 图像里的 $x_t$ 是 pixel 还是 VAE latent；
- 视频里的一个“坐标”是否带 frame index；
- 分子坐标是否已经去除质心，以及 atom type 是否单独 corruption；
- 控制里的 $x_0$ 是一个 action 还是一段 action horizon；
- 插补里的 loss 是否只作用在 target mask；
- 条件 $c$ 是文本、受体口袋、传感器历史还是 observed values。

因此“都用 epsilon prediction”只说明局部训练接口相似，不能说明概率对象、inductive bias、采样成本或评价证据相同。

***

## 3. 技术史应写成问题压力，而不是发布日期

应用史中值得保留的节点，是那些改变接口的节点：TimeGrad/CSDI 把预测与任意 mask 条件化；latent diffusion 把高分辨率建模移到压缩空间；GeoDiff/EDM 把等变性写进 coordinate score；DreamFusion 把 score 接到 renderer Jacobian；Diffusion Policy 把 action chunk 作为样本；Diffusion Forcing 给不同 token 分配不同噪声级别。

![应用问题推动的接口演进](/images/diffusion/d11_application_timeline.png)

图中的时间是首次公开论文或系统报告的近似年份，不是性能排名。一个较新的系统可以规模更大，却未必提出新的数学接口；一个较早的方法也可能持续成为后续路线的基础。

***

## 4. 先固定证据等级，避免把演示写成结论

本章使用五类证据：

| 证据类型                    | 主要职责                    | 不能自动承担的职责              |
| ----------------------- | ----------------------- | ---------------------- |
| 正式论文                    | 方法、公式、受控实验和正式版本         | 跨协议的普遍优越性              |
| preprint                | 最新方法与可核验推导              | 假装已经同行评审               |
| system/technical report | 规模、工程、声明的能力边界           | 一般定理或公平的跨系统比较          |
| project page/model card | provenance、媒体示例、发布与用途边界 | 代替论文实验                 |
| evaluation critique     | 暴露 metric/protocol 盲区   | 从一个 benchmark 推出普遍不可能性 |

Movie Gen、Cosmos 和 Sora 属于这一问题的典型例子：它们对于理解现代视频系统很重要，但报告、页面和演示的职责不同。Sora 官方研究页在本地资料包中只保留已核验元数据，不承担公式来源。

***

## 5. 图像路线给后续应用留下了什么

图像生成本身已经在 D7--D8 展开。本章只保留三项会被其他领域复用的转折。

第一，Latent Diffusion 把 state 从像素 $x$ 改成编码器输出 $z=E(x)$：

$$
z_t=\alpha_tz_0+\sigma_t\epsilon,
\qquad
\hat x=D(z_0).
$$

这使高分辨率生成的主要计算发生在低空间分辨率 latent 上。视频、音频和 3D 随后都复用了“先选 representation，再做 diffusion”的设计。

第二，SDEdit、Palette 等图像编辑/translation 工作说明 condition 不只有文本。source image、mask、measurement 和 instruction 决定哪些坐标可改变、哪些信息必须保留。这个思想在 CSDI 的 observed mask 和 inverse problem 中会再次出现。

第三，图像评估暴露了 feature metric 的协议依赖。Stein 等人的评测研究说明 FID 类指标会受 feature extractor、样本数、resize 和模型族影响。因此后续领域不能只发明一个“Fréchet-X Distance”就认为评估问题结束。

***

## 6. 压缩不是免费午餐

若图像大小为 $H\times W\times C$，latent 大小为

$$
h\times w\times c,
\qquad
h=H/f,quad w=W/f,
$$

则空间 token 数约缩小 $f^2$ 倍。但 decoder 必须从 $z$ 重建像素，因而整个系统误差至少包含：

$$
\text{representation error}
+\text{generative error}
+\text{decoder artifact}.
$$

视频进一步压缩时间维；音频可能先变成 spectrogram 再进入 VAE；Shap-E 生成的是 implicit-function 参数的 latent。压缩提高可计算性，也可能丢掉运动细节、高频音色、细薄几何或科学约束。不能把“latent 更便宜”写成“latent 与原空间等价”。

***

## 7. 图像指标为何不能直接迁移到所有领域

FID 比较 feature distribution 的前两阶统计。即使估计完全准确，它回答的也是“选定 feature space 中两个分布的 Gaussian approximation 有多接近”，不是：

- prompt 是否满足；
- 编辑是否保留无关区域；
- 视频是否时间一致；
- 3D 背面是否正确；
- docking pose 是否没有原子碰撞；
- 机器人是否闭环成功。

因此一个应用至少需要同时报告 sample quality、condition adherence 和 domain validity。三者有时相关，但任何一个都不能替代另外两个。

***

## 8. 从图像到视频：state 多了时间轴

视频状态可以写成

$$
x_0\in\mathbb R^{F\times H\times W\times C},
$$

其中 $F$ 是帧数。最直接的 Gaussian corruption 仍是

$$
x_t=\alpha_tx_0+\sigma_t\epsilon,
$$

但 denoiser 现在必须区分三种结构：

1. 单帧内部的空间纹理；
2. 相邻帧的局部运动；
3. 长时间跨度的身份、场景和因果一致性。

独立逐帧生成只解决第一项，常导致 flicker、物体身份漂移和不连续运动。视频 diffusion 的关键不是“对更多像素加噪声”，而是让网络和条件接口看见时间。

***

## 9. Video Diffusion 为什么分解空间与时间

Video Diffusion Models（补充材料暂未公开） 把 2D U-Net 扩展成 space-time factorized 3D U-Net：空间卷积/attention 处理每帧，temporal attention 沿同一空间位置跨帧通信。

若把 $N=FHW$ 个 token 全部做 dense attention，复杂度约为

$$
O((FHW)^2).
$$

分解后，空间 attention 与时间 attention 的主项近似为

$$
O\bigl(F(HW)^2+HW F^2\bigr).
$$

这不是声称二者表达能力严格相同，而是用结构假设换取可计算性：先在帧内建模，再在同位置或局部轨迹上传播时间信息。

***

## 10. Latent video 解决计算，不自动解决长时一致性

Align Your Latents（补充材料暂未公开） 把视频 diffusion 放入图像 latent，并加入 temporal alignment。设 encoder 给出

$$
z_0=E(x_0)\in\mathbb R^{F\times h\times w\times c}.
$$

若只压缩空间，token 数仍随 $F$ 线性增长；若再压缩时间，则 decoder 还要恢复帧间细节。两种方案都没有从定义上保证：

- 长视频中的同一角色保持身份；
- 物体离开画面后仍被记住；
- 动作满足真实物理；
- 生成未来能用于控制。

“latent video”主要是 representation/compute 方案，不能与“world understanding”画等号。

***

## 11. 视频生成的三种时间条件

常见条件可按信息方向分成：

1. **全局条件**：文本、风格、类别，对所有帧共享；
2. **前缀条件**：给定最初若干帧，预测或扩展未来；
3. **局部/动作条件**：每个时间点带 control、camera 或 action。

对应目标分别接近

$$
p(x_{1:F}\mid y),
\qquad
p(x_{K+1:F}\mid x_{1:K},y),
\qquad
p(x_{1:F}\mid a_{1:F},x_{\mathrm{context}}).
$$

它们不能只用“text-to-video”概括。尤其第三种条件要求模型区分 correlation 与 intervention；仅生成视觉上合理的未来，不证明动作真的控制了未来。

***

## 12. Diffusion Forcing：每个 token 有自己的噪声级别

标准 full-sequence diffusion 常给整段序列同一个 $t$。autoregressive model 则把过去当 clean condition、未来逐 token 预测。Diffusion Forcing（补充材料暂未公开） 用向量噪声级别

$$
k_{1:F}=(k_1,ldots,k_F)
$$

连接两者：

$$
x_i^{k_i}=\alpha_{k_i}x_i+\sigma_{k_i}\epsilon_i,
$$

$$
\mathcal L_{\mathrm{DF}}
=\mathbb E_{k_{1:F},\epsilon}
\sum_{i=1}^F
\left\|\epsilon_i-\epsilon_\theta(x_{1:F}^{k_{1:F}},k_{1:F})_i\right\|^2.
$$

当过去 token 的 $k_i=0$、未来 token 较 noisy 时，它接近 causal prediction；当所有 $k_i$ 相同，它接近 full-sequence diffusion。新概念不是“又一个视频模型”，而是**token-wise noise assignment**。

***

## 13. World model 的证据门槛比视频质量更高

Movie Gen 是覆盖 video generation/editing/personalization/audio 的工业 system report；Cosmos 把 tokenizer、diffusion/autoregressive world model 和 physical-AI 平台组织在一起。它们说明系统规模和任务组合已经变化，但“world model”至少可能指三种不同要求：

1. 生成看起来像真实世界的视频；
2. 预测给定动作后的未来 observation distribution；
3. 提供足够准确的 rollout，使 policy/planner 获益。

第一项不能推出第二项，第二项也不能推出第三项。真正的控制证据需要 action-conditioned counterfactual、长时误差和 downstream decision utility，而不只是人类偏好的视频样例。

***

## 14. VBench 说明为什么视频不能只报一个分数

VBench（补充材料暂未公开） 把评测拆成 subject consistency、background consistency、motion smoothness、dynamic degree、aesthetic quality、imaging quality、condition consistency 等维度。

这种分解的重要性在于：

- 极少运动的视频可能 temporal consistency 很高，却不满足“动态”要求；
- 每帧都清晰不表示物体身份稳定；
- prompt 相似度高不表示物理合理；
- aggregate score 上升可能由某一维度主导。

所以本章将视频评测视为 vector-valued evidence，而不是一个可跨论文无条件排序的标量。

***

## 15. 音频 diffusion 的 state 可以位于三层

音频系统常在三种状态空间工作：

$$
\text{waveform }x\in\mathbb R^T,
$$

$$
\text{spectrogram }S\in\mathbb R^{F\times L},
$$

$$
\text{audio latent }z=E(S)\in\mathbb R^{f\times l\times c}.
$$

waveform 保留相位与时间细节，但序列很长；spectrogram 更接近二维局部结构，却需要 vocoder 或相位恢复；latent 更便宜，但再增加 representation/decoder error。选择哪一层 diffusion，是音频系统的首要建模决策。

***

## 16. DiffWave：从 vocoder 到通用 waveform generator

DiffWave（补充材料暂未公开） 证明一维 dilated convolution denoiser 可以直接生成 waveform，并可在 mel spectrogram 条件下作为 neural vocoder。

其条件形式仍可写成

$$
\mathcal L
=\mathbb E\left[
\left\|\epsilon-\epsilon_\theta(x_t,t,S)\right\|^2
\right],
$$

但这里 $S$ 不是“描述声音的文本”，而是与 waveform 时间结构强对齐的 acoustic condition。音频中的 condition alignment 可能是 frame-level、clip-level 或 semantic-level，三者不能混为一种 cross-attention。

***

## 17. AudioLDM：文本条件先变成共享音频语义空间

AudioLDM（补充材料暂未公开） 使用 CLAP 构造 audio/text 对齐 embedding，并在 mel-spectrogram VAE latent 上 diffusion。简化的数据流是

$$
x_{\mathrm{wave}}
\xrightarrow{\mathrm{STFT/mel}}S
\xrightarrow{E}z_0
\xrightarrow{\mathrm{diffusion}}\hat z_0
\xrightarrow{D}\hat S
\xrightarrow{\mathrm{vocoder}}\hat x_{\mathrm{wave}}.
$$

条件 $c$ 在训练时可来自 audio embedding，在生成时来自 text embedding：

$$
c_{\mathrm{train}}=f_{\mathrm{CLAP}}^{\mathrm{audio}}(x),
\qquad
c_{\mathrm{test}}=f_{\mathrm{CLAP}}^{\mathrm{text}}(y).
$$

这依赖一个前提：两种 encoder 输出在共享空间中足够对齐。CLAP 因而是 representation/condition source，不是 diffusion 本身。

***

## 18. AudioLDM 2 与“language of audio”

AudioLDM 2 试图用 self-supervised audio representation 统一 speech、music 和 sound effects。这里的“language”是可供生成模型条件化或预测的连续/离散音频语义表示，不等于自然语言 token。

统一 representation 的好处是共享数据与模型；风险是一个表示可能对 speech intelligibility 敏感，却压缩音乐音色，或反之。系统声称“holistic audio generation”时，应分别检查各子领域，而不是只看总平均。

音频专项综述（补充材料暂未公开） 在本章承担 taxonomy 职责；其 preprint 身份不被写成正式会议结论。

***

## 19. FAD 与听感、语义不是同一个问题

Fréchet Audio Distance 在某个 audio embedding 中比较生成与参考分布，思想类似 FID。它可以检测分布层面的变化，却不直接回答：

- 语音是否可懂；
- 文本描述是否被满足；
- 音乐结构是否长期连贯；
- 是否存在短促 click、phase artifact 或失真；
- 条件数据是否覆盖目标文化、语言和声学环境。

音频评估至少应组合 embedding metric、任务特定指标和 listening study，并说明 feature extractor、采样率、时长和样本数。

***

## 20. Text-to-3D 首先要区分三条数据路线

Eurographics 2024 的 Text-to-3D STAR（补充材料暂未公开） 将路线区分为：

1. 有 paired text--3D 数据，直接学习条件 3D generator；
2. 有 3D corpus 但文本未对齐，借助 shared embedding 或 synthetic caption；
3. 没有 3D training corpus，冻结 2D text-to-image prior，对每个 prompt 优化一个 3D representation。

DreamFusion/Magic3D 主要属于第三条；Point-E/Shap-E 更接近用 3D 数据训练 amortized generator。它们的训练成本、每 prompt 成本和 prior bias 完全不同。

***

## 21. 为什么冻结二维模型也能给三维参数提供方向

设场景参数为 $\phi$，相机为 $c$，可微 renderer 给出

$$
x_0=g(\phi,c).
$$

对 rendered image 加噪：

$$
x_t=\alpha_tg(\phi,c)+\sigma_t\epsilon.
$$

冻结的 text-to-image diffusion 提供 conditional score

$$
s_p(x_t\mid y,t)=\nabla_{x_t}\log p_t(x_t\mid y).
$$

renderer-induced corruption kernel 的 conditional score 是

$$
s_q(x_t\mid x_0,t)
=-\frac{x_t-\alpha_tx_0}{\sigma_t^2}
=-\frac{\epsilon}{\sigma_t}.
$$

如果模型使用 epsilon prediction，

$$
\epsilon_\theta(x_t;y,t)=-\sigma_t s_p(x_t\mid y,t).
$$

因此 score difference 与 noise residual 精确对应：

$$
-\sigma_t(s_p-s_q)=\epsilon_\theta-\epsilon.
$$

这一步是 SDS 的核心桥梁。

***

## 22. SDS 梯度的完整接口

DreamFusion（补充材料暂未公开） 使用估计器

$$
\widehat{\nabla_\phi\mathcal L}_{\mathrm{SDS}}
=w(t)
\bigl(\epsilon_\theta(x_t;y,t)-\epsilon\bigr)
\frac{\partial g(\phi,c)}{\partial\phi}.
$$

它包含三段：

1. frozen diffusion 给出 image-space direction；
2. noise residual 将 target score 与 renderer kernel score 相减；
3. renderer Jacobian 把 image-space direction 拉回 scene parameter space。

![SDS 从冻结 score 到参数空间更新](/images/diffusion/d11_sds_gradient.png)

这解释了为什么二维 prior 能优化三维对象：prior 不直接知道 $\phi$，但 renderer 建立了 $\phi\to x$ 的可微映射。

***

## 23. 为什么要 stop U-Net Jacobian

若直接对标量

$$
\frac12\|\epsilon_\theta(x_t)-\epsilon\|^2

$$

关于 $\phi$ 反向传播，会出现

$$
\frac{\partial\epsilon_\theta}{\partial x_t}
\frac{\partial x_t}{\partial x_0}
\frac{\partial x_0}{\partial\phi}.

$$

SDS 实践中把 diffusion output 当作冻结 vector field，停止 $\partial\epsilon_\theta/\partial x_t$。这不是普通 pixel MSE 的梯度。U-Net Jacobian 计算昂贵、数值条件差，而且训练得到的是 score/noise field，并没有要求其 Jacobian 适合作为 scene optimization Hessian。

因此教程中更严谨的写法是“使用 score-distillation gradient estimator”，而不是先虚构一个简单 scalar loss 再假装完全 backprop。

***

## 24. Gaussian audit：SDS 在什么情况下等于 KL 梯度

考虑一维检查：

$$
q_{\phi,t}=\mathcal N(\alpha_t\phi,\sigma_t^2),
\qquad
p_t=\mathcal N(\mu_t,v_t).
$$

直接求导得到

$$
\frac{\partial}{\partial\phi}
\mathrm{KL}(q_{\phi,t}\|p_t)
=\alpha_t\frac{\alpha_t\phi-\mu_t}{v_t}.
$$

另一方面，

$$
\mathbb E[\epsilon_\theta-\epsilon]
=\sigma_t\frac{\alpha_t\phi-\mu_t}{v_t}.
$$

所以选择

$$
w(t)=\frac{\alpha_t}{\sigma_t}

$$

时，期望 SDS estimator 恰好等于该 KL 梯度。说明代码用 Monte Carlo 和 finite difference 同时检查了这一点。

这个 audit 不是说所有 SDS 论文都在优化同一个 KL。实际权重、CFG、parameterization、camera distribution 和 stop-gradient 约定会改变目标。论文必须写清 convention。

***

## 25. SDS 的三类系统性失败

SDS 的问题并不只是“优化慢”。

第一，二维 prior 只约束随机视角的 render，不能直接观察完整三维一致性，因而可能出现 Janus/multi-face artifact。

第二，单个 $\phi$ 是 point estimate。对多模态 posterior 做 mode-seeking optimization 容易过饱和、过平滑或坍缩到某种高概率视觉解释。

第三，CFG 会放大 condition direction，也可能放大 prior bias 和不自然纹理。高 guidance 不是免费的 prompt fidelity。

Magic3D 用 coarse-to-fine representation 改善分辨率；ProlificDreamer（补充材料暂未公开） 用 Variational Score Distillation 把 point parameter 扩展成 parameter distribution。这些方案缓解特定问题，但不证明 geometry 已正确。

***

## 26. Point-E 与 Shap-E：从逐 prompt 优化到 amortized generation

Point-E 直接生成 point cloud；Shap-E 生成 implicit function 参数的 latent。它们都把大量成本移到离线训练，使测试时不必对每个 prompt 运行长时间 SDS optimization。

两种代表性 state 是

$$
x_0^{\mathrm{point}}\in\mathbb R^{N\times(3+d)},

$$

和

$$
z_0^{\mathrm{implicit}}=E(\text{3D asset}).
$$

amortization 的代价是需要 3D corpus，并受到 encoder/decoder representation 限制。Point cloud 不直接给 watertight surface；implicit latent 的 decoder 也可能限制细节和拓扑。

***

## 27. 三维结果为什么不能只看 turntable

一个物体的少数渲染视角看起来合理，仍可能：

- 背面为空或纹理错误；
- 存在浮动几何和 disconnected component；
- 法线、材质或照明被烘焙进 texture；
- 多视角身份不一致；
- mesh 不可编辑、不可打印或不可用于 simulation。

因此 text-to-3D 评价至少要区分 rendered-view quality、multi-view consistency、geometry/topology 与 downstream asset usability。CLIP similarity 只覆盖其中一部分。

***

## 28. 科学生成的 state 是异构的

图像像素通常共享同一种连续数值语义。科学对象却常同时包含：

$$
x=(a,b,r,L),
$$

其中 $a$ 是 atom/residue type，$b$ 是 bond 或 graph edge，$r\in\mathbb R^{N\times3}$ 是坐标，$L\in\mathbb R^{3\times3}$ 是 periodic lattice。

一种典型 forward factorization 是

$$
q_t(x_t\mid x_0)
=q_t^{\mathrm{type}}(a_t\mid a_0)
q_t^{\mathrm{edge}}(b_t\mid b_0)
q_t^{\mathrm{coord}}(r_t\mid r_0)
q_t^{\mathrm{lattice}}(L_t\mid L_0).
$$

这只是 corruption 的条件独立设计；learned denoiser 必须耦合这些分量，否则可能生成“坐标像分子、类型却不匹配”的对象。D10 的 categorical diffusion 与 D4 的 continuous score 在这里真正相遇。

***

## 29. 从 invariant density 推出 equivariant score

设正交群作用为 $x\mapsto\rho(g)x$，目标密度满足

$$
p(\rho(g)x)=p(x).
$$

对 $x$ 求梯度。链式法则给出

$$
\rho(g)^\top
\nabla\log p(\rho(g)x)
=\nabla\log p(x).
$$

因为 $\rho(g)^{-1}=\rho(g)^\top$，两边左乘 $\rho(g)$：

$$
s(\rho(g)x)=\rho(g)s(x).
$$

所以 invariant density 的 score 是 equivariant vector field。对三维旋转 $R\in SO(3)$，若所有 atom coordinate 同时旋转，则预测的 coordinate score 也应同步旋转。

![等变 score 与科学有效性层级](/images/diffusion/d11_symmetry_validity.png)

等变性减少了网络从数据中重新学习旋转规律的负担，但它只保证 transformation consistency，不保证化学、能量或功能正确。

***

## 30. 平移不是普通旋转：先处理 gauge

分子整体平移通常不改变物理对象。若直接对绝对坐标 diffusion，center-of-mass 会引入没有意义的自由度。常见做法是投影到 zero-center subspace：

$$
C(r)=r-\frac1N\mathbf1\mathbf1^\top r.
$$

对任意全局 translation $u\in\mathbb R^3$，

$$
C(r+\mathbf1u^\top)=C(r).
$$

对全局 rotation $R$，

$$
C(rR^\top)=C(r)R^\top.
$$

因此 centering 消去 translation gauge，并与 rotation commute。加入 Gaussian noise 后也应保持 zero center，例如对 noise 同样投影，而不是 corruption 后重新引入质心漂移。

***

## 31. GeoDiff 与 EDM：几何约束怎样进入 denoiser

GeoDiff（补充材料暂未公开） 面向给定 molecular graph 的 conformation generation；Equivariant Diffusion for Molecule Generation（补充材料暂未公开） 联合生成 atom features 和 coordinates。

几何网络常从 pairwise relative displacement 构造 message：

$$
d_{ij}=r_i-r_j,
\qquad
h_{ij}=\psi(h_i,h_j,\|d_{ij}\|^2,t),
$$

再用 scalar coefficient 乘 vector：

$$
\Delta r_i
=\sum_j d_{ij}\,\phi(h_{ij}).
$$

因为距离是 invariant，而 $d_{ij}$ 随 rotation 变换，$\Delta r_i$ 自然 equivariant。这里最重要的不是某个 GNN 名称，而是 scalar/invariant 与 vector/equivariant 通道的职责分离。

***

## 32. DiffDock：docking 是受体条件下的 pose distribution

给定 receptor $P$ 与 ligand graph $G$，docking 目标是

$$
p(\text{pose}\mid P,G).
$$

pose 不只是 $3N$ 个自由坐标；通常可分解为 translation、rotation 与 torsion。DiffDock 在这些几何自由度上构造 diffusion/score，使模型能表示多个可能 binding modes。

但 RMSD 接近 crystallographic pose 不是充分条件。PoseBusters（补充材料暂未公开） 发现多种 deep docking 方法会产生：

- 不合理 bond length/angle；
- stereochemistry 改变；
- ligand 内部冲突；
- protein--ligand clash；
- 对新 protein sequence 泛化不足。

所以 docking 评价至少需要 native-like pose 与 physical plausibility 两条轴。

***

## 33. RFdiffusion：生成 backbone 还要经过实验链

RFdiffusion（补充材料暂未公开） 将结构预测网络改造成 protein backbone denoiser，并支持 motif scaffolding、unconditional backbone generation 等任务。

抽象地，backbone state 可写成 residue frames 或 coordinates：

$$
x_0=(R_1,t_1,\ldots,R_N,t_N),
$$

条件可能固定部分 motif：

$$
c=(x_0^{\mathrm{motif}},m_{\mathrm{fixed}},\text{sequence/design constraints}).
$$

生成 backbone 只是 pipeline 第一段。随后还需要 sequence design、structure prediction/filtering、expression、folding 与 functional assay。RFdiffusion 的 Nature 论文之所以重要，不只是生成图片，而是部分设计通过了实验测试。

“模型生成了一个低 predicted error 的 backbone”不能被改写成“实验功能已验证”。

***

## 34. AlphaFold 3 中的 diffusion 处于更大的预测系统内

AlphaFold 3（补充材料暂未公开） 用 diffusion module 生成/精化原子坐标，面向 protein、nucleic acid、small molecule、ion 等复合物结构预测。

这里需要三个边界：

1. 它主要是 conditional structure prediction system，不是任意 de novo molecule generator；
2. diffusion module 接收上游 sequence/pair representation，不能脱离整个系统单独解释性能；
3. predicted structure 与 confidence 不等于 binding affinity、dynamics 或实验机制。

“系统里用了 diffusion”不表示整套方法可以归约成 DDPM；representation、pairformer、training data 和 confidence head 同样决定输出。

***

## 35. MatterGen：材料让 lattice 和 periodicity 成为一等公民

材料晶体状态可写成

$$
x=(a,r,L),
$$

其中坐标通常是 fractional coordinate，满足 periodic boundary。MatterGen 对 atom type 使用 categorical corruption，对 coordinate/lattice 使用连续 corruption，并通过 property adapter 与 classifier-free guidance 控制 composition、symmetry 或性质。

如果 property 为 $y$，conditional score 仍可形式化为

$$
s(x_t,t\mid y)
=s(x_t,t)+\nabla_{x_t}\log p(y\mid x_t),
$$

但工程上不一定显式训练 property classifier；adapter/CFG 是另一种条件实现。

预测 energy above hull 较低只是 computational stability signal。材料真正落地还需更高精度计算、动力学/缺陷分析、合成路径与实验表征。

***

## 36. 科学生成的四层证据不能跳级

本章将科学证据分为：

$$
\text{symmetry consistency}
\Rightarrow
\text{geometric/chemical validity}
\Rightarrow
\text{energetic/functional evidence}
\Rightarrow
\text{experimental validation}.
$$

箭头表示下层通常以部分上层为前提，不表示上层自动推出下层。

- equivariant network 仍可生成碰撞结构；
- chemically valid molecule 仍可能不稳定或不可合成；
- predicted stable material 仍可能没有可行 synthesis route；
- computational binding score 仍可能与真实 assay 不一致。

科学应用最危险的叙述错误，是把较便宜的 surrogate 当成较昂贵的真实证据。

***

## 37. 规划中的“样本”是一整条 trajectory

令

$$
\tau=(s_0,a_0,s_1,a_1,\ldots,s_H).
$$

trajectory diffusion 对 $\tau$ corruption，再条件生成满足起点、终点、return 或 constraint 的轨迹：

$$
p_\theta(\tau\mid c).
$$

这与单步 policy $p(a_t\mid s_t)$ 的差别是：模型一次表达跨时间的联合多模态结构。绕过障碍物的“左路”和“右路”可以作为两个 trajectory mode，而不必让每个时刻独立平均成不可执行动作。

代价是 state 维度随 horizon 增长，采样要多次评估整条 trajectory，并且离线生成的轨迹还要接受真实环境反馈。

***

## 38. Reward guidance 是对 trajectory density 做指数倾斜

给定 learned trajectory prior $p_\theta(\tau\mid c)$ 与 reward $R(\tau)$，定义

$$
p_\lambda(\tau\mid c)
=\frac{p_\theta(\tau\mid c)\exp\{\lambda R(\tau)\}}
{Z_\lambda(c)}.
$$

对 $\tau$ 求 score：

$$
\nabla_\tau\log p_\lambda(\tau\mid c)
=\nabla_\tau\log p_\theta(\tau\mid c)
+\lambda\nabla_\tau R(\tau),
$$

因为 $Z_\lambda(c)$ 不依赖 $\tau$。这就是 classifier/reward guidance 在 trajectory space 的基本形式。

若

$$
p_\theta=\mathcal N(\mu,\Sigma),
\qquad
R(\tau)=r^\top\tau,
$$

配方可精确化为

$$
p_\lambda
=\mathcal N(\mu+\lambda\Sigma r,\Sigma).
$$

说明代码检查了 tilted mean 与 score addition。形式正确不保证 reward 正确：learned value/reward 的 extrapolation error 会被 guidance 放大。

***

## 39. Diffuser 与 Decision Diffuser 解决的是不同条件接口

Diffuser（补充材料暂未公开） 在 state-action trajectory 上生成，并可用 reward gradient 或 constraints 引导。Decision Diffuser 更强调 conditional generative modeling：用 desired return、skill 或 constraint 作为 condition，常先生成 state trajectory，再用 inverse dynamics 得到 action。

两条路线分别暴露不同问题：

- gradient guidance 依赖 reward differentiability 与准确性；
- return conditioning 依赖训练数据中 condition coverage；
- state-only trajectory 依赖 inverse dynamics 是否唯一且可靠；
- 离线 trajectory realism 不等于在线 robustness。

所以“diffusion 能规划”必须附带 environment、dataset、conditioning 与 execution protocol。

***

## 40. Diffusion-QL：behavior support 与 value improvement 的拉扯

离线 RL 需要避免选择数据支持外的高估 action。Diffusion-QL 用 diffusion objective 表示多模态 behavior policy，同时加入 Q maximization。抽象写成

$$
\mathcal L(\theta)
=\mathcal L_{\mathrm{diffusion}}(\theta)
-\eta\,
\mathbb E_{s\sim\mathcal D,\,a\sim\pi_\theta(\cdot\mid s)}
[Q_\psi(s,a)].
$$

第一项把 action 拉向 dataset support，第二项推动高价值动作。系数 $\eta$ 不是普通超参数，而是 support--improvement tradeoff。若 Q 在 OOD action 上高估，增大 $\eta$ 会把 expressive policy 的优势变成风险。

***

## 41. Diffusion Policy：把 action horizon 作为条件样本

在控制时刻 $n$，给定 $K$ 步 observation history：

$$
o_{n-K+1:n},
$$

目标 action horizon 为

$$
a_{n:n+H-1}\in\mathbb R^{H\times d_a}.
$$

对整段 action corruption：

$$
a^{(k)}
=\sqrt{\bar\alpha_k}a
+\sqrt{1-\bar\alpha_k}\epsilon.
$$

Diffusion Policy（补充材料暂未公开） 的核心条件目标可写为

$$
\mathcal L_{\mathrm{act}}
=\mathbb E_{n,k,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(a^{(k)},k,o_{n-K+1:n})
\right\|^2
\right].
$$

它学习的是 observation-conditioned action-sequence distribution，而不是每步独立 Gaussian action。

***

## 42. 训练 horizon 与执行 prefix 必须分开

采样得到 $H$ 步 action 后，系统通常只执行前 $h\le H$ 步：

$$
\hat a_{n:n+H-1}
\longrightarrow
\text{execute }\hat a_{n:n+h-1}
\longrightarrow
\text{observe and replan}.
$$

![Action horizon 与 conditional imputation mask](/images/diffusion/d11_control_imputation.png)

$H$ 决定 joint temporal structure；$h$ 决定反馈频率与推理预算。增大 $H$ 可能提高动作一致性，也增加输出维度；减小 $h$ 提高 feedback responsiveness，却要求更频繁 denoising。

因此报告 NFE 时还必须报告 control frequency、hardware latency、executed prefix 和 observation pipeline。只说“采样 10 步”不足以判断能否实时控制。

***

## 43. DP3：三维 observation 是表示选择，不是自动泛化保证

DP3 使用 point cloud 与 proprioception 条件 action diffusion，希望减少 2D image 对 viewpoint、texture 和 background 的敏感性。条件可写成

$$
c_n=(E_{3D}(P_n),q_n),
$$

其中 $P_n$ 是 point cloud，$q_n$ 是 proprioceptive state。

三维表示可能更直接表达 geometry，但仍受 depth noise、occlusion、calibration、point density 和 encoder bias 影响。某组 manipulation benchmark 上更好，不等于所有机器人任务都应先重建 point cloud。

***

## 44. 轨迹 diffusion 不是自动的 Schrödinger Bridge

trajectory diffusion、reward-guided sampling 与 Schrödinger Bridge 都可能出现 stochastic path 和 control 语言，但优化对象不同。

Schrödinger Bridge 需要明确 reference path measure $R$，并在 endpoint marginals 约束下求

$$
P^*
=\arg\min_{P:\,P_0=\mu_0,\,P_T=\mu_T}
\mathrm{KL}(P\|R).
$$

普通 Diffuser/Diffusion Policy 通常是从离线 trajectory/action 数据学习 denoising distribution，再加 condition 或 reward。没有 reference path KL 与双端 marginal 约束时，不应因为“生成一条桥接轨迹”就称为 Schrödinger Bridge。详细理论留给 Bridge 教程目录（补充材料暂未公开）。

***

## 45. TimeGrad：forecasting 外层仍可 autoregressive

对 multivariate time series $x_{1:T}$，probabilistic forecasting 目标是

$$
p(x_{T+1:T+H}\mid x_{1:T},c).
$$

TimeGrad 用 RNN state 汇总历史，再用 conditional diffusion 表示下一步 distribution：

$$
h_t=f(h_{t-1},x_t,c_t),
\qquad
p(x_{t+1}\mid h_t).
$$

完整 joint 仍按 autoregressive 外层分解：

$$
p(x_{T+1:T+H}\mid x_{1:T})
=\prod_{j=1}^H
p(x_{T+j}\mid x_{1:T+j-1}).
$$

所以“内部用 diffusion”没有消除 horizon-wise error accumulation；它主要提高每步 conditional distribution 的表达力。

***

## 46. CSDI：observed mask 与 target mask 是统计问题的一部分

令

$$
m_{\mathrm{obs}},m_{\mathrm{tar}}\in\{0,1\}^{T\times D},
\qquad
m_{\mathrm{obs}}\odot m_{\mathrm{tar}}=0.
$$

条件包含 observed values、两个 mask 与 time/feature side information：

$$
c=
(m_{\mathrm{obs}}\odot x_0,
m_{\mathrm{obs}},m_{\mathrm{tar}},
\text{side info}).
$$

只对 target coordinates 加噪：

$$
x_t^{\mathrm{tar}}
=m_{\mathrm{tar}}\odot
(\alpha_tx_0+\sigma_t\epsilon).
$$

CSDI（补充材料暂未公开） 风格的目标是

$$
\mathcal L_{\mathrm{mask}}
=\mathbb E
\left[
\frac{
\left\|m_{\mathrm{tar}}\odot
(\epsilon-\epsilon_\theta(x_t^{\mathrm{tar}},t,c))
\right\|^2
}{\max(1,\|m_{\mathrm{tar}}\|_1)}
\right].
$$

observed coordinates 不应计入 target loss；采样时它们作为条件保留或 clamp。

***

## 47. 自监督 target mask 决定模型学到哪类 conditional

真实缺失值没有 ground truth，训练时通常从已观测数据中再选一部分作为 artificial target。mask proposal 可为：

- random point mask；
- contiguous block mask；
- historical missing-pattern mask；
- forecast mask，即未来连续区间全为 target。

训练最小化的是 mask distribution 下的风险：

$$
\mathbb E_{m_{\mathrm{tar}}\sim\pi_{\mathrm{train}}}
[\mathcal L(m_{\mathrm{tar}})].
$$

若部署 mask 来自不同的 $\pi_{\mathrm{test}}$，尤其是 MNAR missingness，随机 mask benchmark 的好成绩不能直接外推。mask 不是数据预处理细节，而是 conditional task distribution。

***

## 48. TimeDiff 与 Diffusion-TS 代表两种不同改动

TimeDiff 直接生成整个 future horizon，试图避免 TimeGrad 的 autoregressive outer loop：

$$
p(x_{T+1:T+H}\mid h_T).
$$

它把 horizon dependence 交给 joint denoiser，但 context $h_T$ 是否保留足够历史信息成为新瓶颈。当前资料中 TimeDiff 的正确来源是 arXiv `2306.05043v1` preprint，不写成正式会议论文。

Diffusion-TS 在 denoiser 中显式引入 trend/seasonal structure，强调 general time-series generation 与 interpretability。结构分解可以改善 inductive bias，但只有当分量可识别、稳定并对 downstream analysis 有用时，才能称为解释，而不只是 architecture visualization。

***

## 49. 时间序列评价必须区分 point、distribution 与 conditional correctness

不同指标回答不同问题：

- MAE/RMSE：point forecast 是否接近；
- CRPS/quantile loss：predictive distribution 是否校准且尖锐；
- coverage：prediction interval 是否达到声明概率；
- imputation error：给定特定 mask 后 conditional mean/sample 是否合理；
- discriminative/predictive score：synthetic sequence 是否保留某类 downstream signal。

即使 unconditional sample 的 marginal histogram 很像，也可能破坏 autocorrelation、cross-channel dependence 或 conditional dynamics。时间序列生成不能只看逐点分布。

***

## 50. 七个领域最终贡献了哪些通用接口

把模型名去掉，应用带来的方法概念可以压缩成七项：

| 应用压力               | 通用接口                                                |
| ------------------ | --------------------------------------------------- |
| 高维媒体计算             | learned latent / tokenizer                          |
| 视频与序列不同位置的已知程度不同   | token-wise noise level                              |
| 冻结 prior 要指导另一个生成器 | score difference + differentiable renderer Jacobian |
| 科学对象有群对称性          | invariant density / equivariant score               |
| state 同时含类别与坐标     | coupled discrete--continuous corruption             |
| 行为需要时间一致性和反馈       | action/trajectory horizon + receding execution      |
| 任意部分观测             | observed/target mask as condition                   |

这些接口会继续出现在未来论文中，即使具体 backbone、dataset 和产品名称变化。

***

## 51. 证据矩阵：什么来源可以证明什么

![不同证据类别的职责边界](/images/diffusion/d11_evidence_boundaries.png)

“primary”表示该类来源通常可以直接承担该 claim；“support”表示需要与更合适的来源交叉；“avoid”表示不应把它当作主要证据。

对每个定量比较，至少记录：

$$
(\text{task},\text{condition},\text{dataset/split},
\text{representation},\text{metric implementation},
\text{compute/NFE},\text{publication class}).
$$

缺少其中任何关键项，数字都可能无法复现或不可比较。D12 会进一步讨论 metric、memorization、privacy、copyright 和 safety；本章只建立应用侧的证据纪律。

***

## 52. 说明代码如何映射本章公式

说明代码（补充材料暂未公开） 不训练网络，而是把五个接口变成可执行检查：

```python
# SDS: frozen score/noise field -> renderer-parameter gradient
x_t, eps_hat, grad_samples = sds_gaussian_quantities(
    phi, alpha, sigma, target_mean, target_variance, epsilon
)

# conditional imputation: loss only on target coordinates
loss = masked_epsilon_loss(target_mask, noise, prediction)

# scientific symmetry: rotate coordinates and score together
error = np.max(
    np.abs(radial_score(x @ rotation.T) - radial_score(x) @ rotation.T)
)
```

运行：

```powershell
# 本地验证脚本暂未公开
# 本地验证脚本暂未公开
```

固定种子下，SDS Monte Carlo gradient error 约为 $5.54\times10^{-4}$，finite-difference error 约为 $2.16\times10^{-12}$；reward-tilt score、action-horizon round trip、mask loss 与 equivariance 检查均达到 floating-point 精度。

***

## 53. 常见错误

1. 把应用章节写成模型和产品的排行榜；
2. 看到 epsilon-MSE 相同，就认为不同领域的概率对象相同；
3. 把 latent compression 当成无损等价变换；
4. 把高分辨率或更长视频当成 world understanding 证明；
5. 用逐帧质量代替 temporal consistency；
6. 把 text-conditioned video 与 action-conditioned world model 混为一谈；
7. 用单个 VBench aggregate 掩盖维度间 tradeoff；
8. 把 CLAP 当成 audio diffusion model，而不是 conditioning representation；
9. 用 FAD 代替 listening、semantic alignment 和任务指标；
10. 把 SDS 写成普通 pixel MSE 的完整反向传播；
11. 忽略 SDS 中的 stop-gradient 与 weighting convention；
12. 用 turntable render 证明三维 geometry 正确；
13. 混淆逐 prompt optimization 与 amortized 3D generator；
14. 认为 equivariance 自动保证化学有效；
15. 对分子绝对坐标加噪，却不处理 translation gauge；
16. 只用 docking RMSD，不检查 clash、stereochemistry 和 energy；
17. 把 predicted stability/confidence 当成实验验证；
18. 把 AlphaFold 3 简化成一个独立 DDPM；
19. 把 trajectory diffusion 自动称为 Schrödinger Bridge；
20. 在离线 RL 中忽略 Q extrapolation error；
21. 报 Diffusion Policy NFE，却不报 control frequency 和 latency；
22. 把 action horizon $H$ 与执行 prefix $h$ 混为一谈；
23. 在 CSDI loss 中把 observed coordinates 也当 target；
24. 用 random missing mask 结果外推到 MNAR deployment；
25. 把 system report 或 project demo 写成同行评审结论；
26. 比较数字时省略数据、representation、metric 实现或 compute。

***

## 54. 章节小结

1. 应用比较应从 task--condition--state--output--metric 五元组开始；
2. 图像路线留下了 latent representation、structured conditioning 和 metric protocol 三个基础；
3. 视频把时间、temporal compression、token-wise noise 和 world-model evidence 推到前台；
4. 音频说明 waveform、spectrogram、latent 与 cross-modal embedding 是不同职责；
5. SDS 通过 score difference 与 renderer Jacobian，把冻结二维 prior 变成三维参数更新；
6. 科学生成要求 invariant density、equivariant score、mixed state 与分层 validity evidence；
7. 控制把 trajectory/action horizon 作为样本，并通过 receding horizon 接回反馈；
8. 时间序列插补把 observed/target mask 变成条件分布本身；
9. formal paper、preprint、system report、project page 与 critique 的证据职责不可互换；
10. 应用真正推动的不是模型名，而是可迁移的新概率接口。

***

## 55. 研究式思考题

1. 若视频 VAE 同时压缩空间和时间，怎样把 reconstruction error、temporal aliasing 与 diffusion error 分开测量？
2. token-wise noise assignment 能否统一 video infilling、forecasting 和 bidirectional editing？什么条件下训练分布覆盖测试 noise pattern？
3. action-conditioned video model 要通过哪些 intervention test 才能被称为 world model，而不只是 conditional video generator？
4. AudioLDM 的 CLAP condition 若存在语言或文化偏差，这种偏差会怎样经 CFG 放大？
5. SDS 的 camera distribution 如何改变优化的隐式 3D objective？哪些视角采样能减少 Janus artifact？
6. 能否构造不依赖 U-Net Jacobian、又比 point-estimate SDS 更稳定的可验证 objective？
7. 3D generator 的 geometry metric、render metric 和 usability metric 应如何组成 Pareto evaluation，而不是单一分数？
8. 对含 chirality 的分子，使用 $O(3)$ equivariance 与 $SE(3)$ equivariance 有什么差别？reflection 是否应该是 symmetry？
9. mixed discrete--continuous diffusion 中，atom type 与 coordinate schedule 应如何耦合，才能避免早期类型不确定性破坏 geometry？
10. 如何把 PoseBusters 类 hard validity check 融入 sampling，而不让模型只优化可微 surrogate？
11. 在 protein/material generation 中，计算筛选与实验验证之间怎样做 active-learning allocation？
12. reward tilt 的 $\lambda$ 是否可以根据 score uncertainty 或 Q uncertainty 自适应，而不是固定？
13. Diffusion Policy 中 $H$、执行 prefix $h$、NFE 与 control frequency 的最优关系是否能从闭环误差界推导？
14. CSDI 若面对 MNAR missingness，需要联合建模 missingness mechanism 还是只改变 mask proposal？
15. joint-horizon forecasting 与 autoregressive forecasting 的误差应如何在 calibration、latency 和 long-range dependence 上公平比较？
16. 当 industrial report 不开放训练数据和完整 compute 时，学术教程应怎样表述其 capability 而不越过证据边界？
17. 哪些应用接口可以用 Schrödinger Bridge 的 path-space KL 重新解释，哪些只是表面上都有“trajectory”？

本章完成了从通用 diffusion 到领域约束的映射。下一章将反过来审问这些系统的共同基础：训练误差是否控制生成分布，指标是否有效，模型是否记忆数据，以及规模化应用带来的安全与社会边界。
