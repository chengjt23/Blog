---
title: 一个现代 Diffusion 系统究竟由什么决定
description: >-
  把 prediction target、loss weighting、guidance、latent representation、U-Net/DiT 与
  sampler 组织成一个系统闭环。
publishedAt: '2026-07-17'
updatedAt: '2026-07-19'
draft: false
type: series-chapter
series: diffusion
order: 4
slug: d4-modern-diffusion-system
tags:
  - diffusion
  - guidance
  - architecture
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: true
includeInFeed: false
indexable: true
scope: 解释训练目标、条件向量场、表示空间、网络架构与数值采样如何共同决定质量和成本，而不罗列方法名录。
---
前三章建立的是 Diffusion 的理论闭环：前向过程把数据变成简单噪声，去噪目标学习 noisy marginal 的 score，score 又决定 reverse-time SDE 或 probability-flow ODE。可是，一条正确的连续动力学还不是一个能生成图像的系统。工程实现必须继续回答四个彼此独立的问题：

1. 网络究竟预测 noise、clean data，还是某种 velocity；不同噪声层的误差怎样加权？
2. 文本或类别条件怎样进入网络，又怎样在采样时增强？
3. 动力学运行在 pixel space 还是 latent space，由 U-Net 还是 Transformer 近似？
4. 已学到的向量场由哪条随机或确定性轨迹、哪种 solver、哪些离散点进行积分？

为了把这些接口放在同一条链上，设 $x$ 是最终图像，$E_\phi,D_\psi$ 是可选的 encoder 与 decoder；pixel model 中二者就是恒等映射。令 $z_0=E_\phi(x)$ 是干净表示，$z_t$ 是时刻 $t$ 的带噪表示，$T$ 是 terminal noise time，$c$ 是文本或类别条件。网络 $F_\theta(z_t,t,c)$ 产生 prediction，经 guidance 与 target conversion 变成采样所需的向量场；solver 把 $z_T$ 积分回 $z_0$，最后解码为图像。

因此，一个现代系统不是“DDPM 加上若干技巧”，而是下面这条计算链：

$$
x
\xrightarrow{E_\phi}
z_0
\xrightarrow{\text{forward corruption}}
z_t
\xrightarrow{F_\theta(\,\cdot\,,t,c)}
\text{prediction}
\xrightarrow{\text{guidance + conversion}}
\text{vector field}
\xrightarrow{\text{solver}}
\hat z_0
\xrightarrow{D_\psi}
\hat x.
$$

本章沿这条链讨论历史：DDPM 给出最小闭环；后续工作逐步拆开 target、weight 与 preconditioning，发展 guidance、latent representation、DiT 与更快 solver，并证明训练好的网络不绑定唯一采样算法。

## 1. 2020：DDPM 给出了最小闭环，但没有唯一规定现代系统

Ho、Jain 与 Abbeel 在 NeurIPS 2020 论文 [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239 "官方论文页面") 中组合了一套非常具体的系统：在 pixel space 建立固定的 Gaussian forward chain；用 time-conditioned U-Net 预测加入的 noise；以近似均匀的 timestep sampling 优化 simple MSE；生成时沿离散 reverse chain 做 ancestral sampling。

这套组合的重要性在于它把可扩展训练、稳定图像网络与概率采样闭合起来。但 DDPM 的成功容易造成一种误解：仿佛“预测 noise、使用 U-Net、逐步随机采样”都是 diffusion 理论强制要求的。前三章已经给出反例：同一个 noisy marginal score 可以写成 noise prediction，也可以写成 data prediction；同一组 marginals 可以由随机 SDE 或确定性 probability-flow ODE 实现。

2022 年，Karras、Aittala、Aila 与 Laine 在 NeurIPS 论文 [*Elucidating the Design Space of Diffusion-Based Generative Models*](https://arxiv.org/abs/2206.00364 "官方论文页面") 中明确拆开这些决定。他们以 Gaussian noise standard deviation $\sigma$ 为主要坐标，分别讨论 preconditioning、training distribution、loss weight、trajectory、solver 与 grid。EDM 的历史作用不只是给出更好的超参数，而是说明：**这些组件可以替换，但误差仍会在端到端系统中相互作用。**

本文把一个现代系统压缩成四层职责：

- representation 决定网络在哪个坐标空间中学习分布，也决定哪些细节提前交给 encoder/decoder；
- network 与 conditioning interface 决定怎样近似同一个时间相关函数；
- objective 决定有限模型把容量和梯度分配给哪些 noise levels；
- sampler 决定怎样消费已经学到的函数，而不改变训练数据本身。

判断一种改进时，应先问它改了哪一层：坐标换算属于 parameterization，重分配各 $t$ 的损失属于 training metric，组合 conditional 与 unconditional output 会改变采样向量场，只替换积分公式或离散点则属于 numerical approximation。

## 2. 2021—2023：可转换的 prediction targets，为什么训练起来仍不相同

先固定 representation 与 forward path。令 $x_0\in\mathbb R^d$ 表示当前建模空间中的干净向量：pixel model 中它是图像，latent model 中它是 encoder 输出。对时刻 $t$，variance-preserving path 可写为

$$
z_t
=
\alpha_t x_0
+
\sigma_t\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I),
\qquad
\alpha_t^2+\sigma_t^2=1.
$$

这里 $z_t\in\mathbb R^d$ 是网络输入，$\alpha_t\ge 0$ 与 $\sigma_t\ge 0$ 分别是 signal 和 noise amplitude，$\varepsilon$ 是与 $x_0$ 同维的标准 Gaussian noise。网络可以预测 $\varepsilon$，也可以预测 $x_0$，因为给定 $z_t$ 后二者满足线性换算。

少步生成把换算的数值问题暴露得更明显。Salimans 与 Ho 在 ICLR 2022 论文 [*Progressive Distillation for Fast Sampling of Diffusion Models*](https://arxiv.org/abs/2202.00512 "官方论文页面") 中指出：当 high-noise endpoint 满足 $\alpha_t\to0$ 时，由 noise prediction 恢复 $x_0$ 需要除以很小的 $\alpha_t$，误差会被放大。论文因此在 VP path 上引入

$$
\boxed{
v_t
=
\alpha_t\varepsilon
-
\sigma_t x_0.
}
$$

把 $z_t$ 与 $v_t$ 联立，可以反解

$$
\boxed{
x_0
=
\alpha_tz_t
-
\sigma_tv_t,
\qquad
\varepsilon
=
\sigma_tz_t
+
\alpha_tv_t.
}
$$

在 exact arithmetic 与 exact prediction 下，$x_0,\varepsilon,v$ 携带可互相恢复的信息。若令 $\alpha_t=\cos\varphi_t$、$\sigma_t=\sin\varphi_t$，则 $v_t=d z_t/d\varphi_t$：它是 VP 圆相对于 angle coordinate 的 velocity，不是任意 path 的通用速度。

信息可逆并不意味着优化问题相同。为了看清差别，定义 signal-to-noise ratio

$$
\operatorname{SNR}(t)
=
\frac{\alpha_t^2}{\sigma_t^2}.
$$

若不同 target 的 prediction 先转换到同一个 $\hat x_0$，平方误差满足

$$
\boxed{
\|\varepsilon-\hat\varepsilon\|^2
=
\operatorname{SNR}(t)
\|x_0-\hat x_0\|^2,
\qquad
\|v_t-\hat v_t\|^2
=
[1+\operatorname{SNR}(t)]
\|x_0-\hat x_0\|^2.
}
$$

因此 constant noise、data 与 velocity MSE 对同一个 reconstruction error 施加了不同的 time-dependent metric。下图左侧把 $(x_0,\varepsilon)$ 到 $(z_t,v_t)$ 画成旋转；中间比较三种 target MSE 隐含的 $x_0$-error 权重；右侧显示低 SNR 时，从 $\varepsilon$ 反解 $x_0$ 的误差放大远强于从 $v$ 反解。重点是：algebraic equivalence 不等于 conditioning equivalence。

![prediction target、隐式训练 metric 与低 SNR conditioning](/images/diffusion/d5_parameterization_geometry.png)

因此 objective 至少要同时写出 target、weight 与 timestep sampling。令 $y_t$ 表示所选的 $x_0,\varepsilon$ 或 $v_t$ target，$w_y(t)\ge0$ 是对应权重，则

$$
\mathcal L(\theta)
=
\mathbb E_{t,x_0,\varepsilon}
\left[
w_y(t)
\left\|
y_t-F_\theta(z_t,t)
\right\|^2
\right].
$$

对 $t$ 的期望还隐含 training proposal：提高某层出现频率与提高 $w_y(t)$ 都会增加其梯度贡献，但二者不是同一个设计量。

围绕“有限网络应把容量放在哪里”，历史上出现了不同回答。Choi 等人在 CVPR 2022 论文 [*Perception Prioritized Training of Diffusion Models*](https://arxiv.org/abs/2204.00227 "官方论文页面") 中认为 large-SNR 区更多承担细微 clean-up，于是在已有 baseline weight $w(t)$ 上乘

$$
w_{\mathrm{P2}}(t)
=
\frac{w(t)}
{[k+\operatorname{SNR}(t)]^{\gamma_{\mathrm{P2}}}},
$$

其中 $k>0$ 与 $\gamma_{\mathrm{P2}}\ge0$ 控制抑制形状。这是图像实验支持的 capacity-allocation 假说。Hang 等人在 ICCV 2023 论文 [*Efficient Diffusion Training via Min-SNR Weighting Strategy*](https://arxiv.org/abs/2303.09556 "官方论文页面") 中把 timesteps 视为共享参数的 tasks，并观察到远隔噪声层的 gradients 可能冲突。他们在共同的 $x_0$-error metric 中采用

$$
w_{x_0}(t)
=
\min[\operatorname{SNR}(t),\gamma_{\mathrm M}],
$$

其中 $\gamma_{\mathrm M}>0$ 是 clipping threshold，再按前面的 error identity 换算到 noise 或 velocity target。Min-SNR 是受 multi-task 观点启发的固定 heuristic，不是 theorem 唯一推出的 Pareto optimum，也不同于 P2 的平滑衰减。

Parameterization 之外还有 network preconditioning。Karras 等人 2022 将真正的 denoiser 写成

$$
D_\theta(z;\sigma)
=
c_{\mathrm{skip}}(\sigma)z
+
c_{\mathrm{out}}(\sigma)
F_\theta\!\left(
c_{\mathrm{in}}(\sigma)z;
c_{\mathrm{noise}}(\sigma)
\right).
$$

$F_\theta$ 是 raw network，四个 $c(\sigma)$ 分别控制 skip、output scale、input scale 与 noise conditioning。EDM 根据数据二阶矩规范 input 与 residual target，并另选 training distribution 与 loss weight。它改善的是数值条件和初始 loss scale，不是证明某个 target 普遍占优。

这一阶段最终留下的系统原则是：\*\*“网络预测什么”必须与“误差按什么 metric 衡量”“输入输出怎样缩放”“训练在哪些噪声层取样”一起说明。\*\*只报告 prediction target，无法确定真实 objective。

## 3. 2021—2022：guidance 不是额外条件，而是改写采样向量场

Conditional denoiser $F_\theta(z_t,t,c)$ 已能接收类别或文本，但普通 conditional training 不保证结果足够服从 $c$。Dhariwal 与 Nichol 在 NeurIPS 2021 论文 [*Diffusion Models Beat GANs on Image Synthesis*](https://arxiv.org/abs/2105.05233 "官方论文页面") 中提出 classifier guidance，在采样时把轨迹推向更符合条件的区域。

令 $p_t(z_t)$ 是时刻 $t$ 的 unconditional noisy marginal，$p_t(z_t\mid c)$ 是 conditional marginal，$p_t(c\mid z_t)$ 是 noisy classifier 给出的条件概率。Bayes rule 对 $z_t$ 求 log-gradient 后得到

$$
\boxed{
\nabla_{z_t}\log p_t(z_t\mid c)
=
\nabla_{z_t}\log p_t(z_t)
+
\nabla_{z_t}\log p_t(c\mid z_t).
}
$$

第一项是 unconditional score，第二项是 classifier gradient。Classifier 必须在每个 noise level 上提供梯度，因此要用 noisy inputs 与 timestep 训练。Scale 大于 1 会 temper likelihood，提高 fidelity 但改变目标分布并损失 diversity。

Classifier guidance 的问题是需要额外 classifier。Ho 与 Salimans 在 2022 年 arXiv 论文 [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598 "官方论文页面") 中用 conditional dropout 训练同一个 denoiser：一部分样本保留条件 $c$，另一部分替换为空条件。采样时，令

$$
\varepsilon_c
=
\varepsilon_\theta(z_t,t,c),
\qquad
\varepsilon_u
=
\varepsilon_\theta(z_t,t,\varnothing)
$$

分别表示 conditional 与 unconditional noise estimate，则常见 framework convention 下的 classifier-free guidance 为

$$
\boxed{
\tilde\varepsilon_\theta
=
\varepsilon_u
+
s(\varepsilon_c-\varepsilon_u).
}
$$

这里 $s\ge0$ 是 guidance scale：$s=0$ 给出 unconditional estimate，$s=1$ 给出普通 conditional estimate，$s>1$ 沿差值外推。原论文写作 $(1+w)\varepsilon_c-w\varepsilon_u$，对应 $s=1+w$；不注明 convention，同一个数字可能表示不同强度。

下图用本文构造的二维 mixture 解释这次改动。左上是四个 modes 的 unconditional score，右上是只保留右侧类别后的 conditional score；左下把两种 exact fields 以 $s=4$ 外推。高 scale 不只“删除错误类别”，还会使目标 modes 周围的向量场更陡，因此同时影响 fidelity--diversity trade-off 与 solver 的数值条件。

![unconditional、conditional 与高 scale guidance 的 score fields](/images/diffusion/d7_guidance_vector_fields.png)

在 exact、彼此兼容且 conservative 的 scores 下，conditional 与 unconditional score 之差可以解释为隐式 classifier gradient；有限 neural fields 却未必是某个 normalized density 的精确梯度，所以不能无条件把任意高 scale CFG 宣称为 exact posterior sampling。它还是一个有计算代价的系统选择：每个 noise level 通常需要 conditional 与 unconditional 两份 network evaluation。把二者沿 batch dimension 拼接可以减少 wall-clock call 数，却不会把总 FLOPs 变成一次普通 forward。

Guidance 与 architecture conditioning 也必须分开。Cross-attention、adaptive normalization 或 condition tokens 决定 $c$ 怎样产生 $\varepsilon_c$；CFG 决定怎样组合 $\varepsilon_c$ 与 $\varepsilon_u$。前者建立 conditional predictor，后者改写送入 solver 的向量场。

## 4. 2015—2023：representation 决定在哪里生成，architecture 决定怎样近似

早期图像 Diffusion 直接在 pixels 上运行。随着分辨率上升，网络必须在数十万甚至数百万个像素变量上反复预测，采样的每一次 function evaluation 都很昂贵。解决这一问题有两条互补路线：先改变 representation，减少需要建模的空间自由度；再改变 architecture，更有效地在该空间中混合局部与全局信息。

Representation 路线继承 two-stage generative modeling。Esser、Rombach 与 Ommer 在 CVPR 2021 的 *Taming Transformers for High-Resolution Image Synthesis* 中用 perceptual 与 adversarial objectives 训练 VQGAN；Rombach、Blattmann、Lorenz、Esser 与 Ommer 随后在 CVPR 2022 论文 [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752 "官方论文页面") 中把 diffusion 放进预训练 autoencoder 的 latent space。令

$$
z_0=E_\phi(x),
\qquad
\hat x=D_\psi(z_0),
$$

其中 $E_\phi$ 把图像 $x\in\mathbb R^{H\times W\times 3}$ 压缩为 latent $z_0$，$D_\psi$ 再解码。第二阶段学习 encoder 输出的 latent distribution；最终图像分布是生成 latent 经 decoder 得到的 pushforward。

这不是无损 change of variables。Encoder 丢失的信息不能由 latent denoiser 凭空恢复；强 decoder 又可能承担一部分局部细节合成。因此 reconstruction、latent generation 与最终 image quality 是三个不同指标。

Latent compression 还直接改变 Transformer 的 token budget。设 encoder 在高、宽方向各下采样 $f$ 倍，Transformer 再把 latent grid 切成边长为 $p$ 的 non-overlapping patches，则 image token 数为

$$
\boxed{
N
=
\frac{HW}{(fp)^2}.
}
$$

这里 $H,W$ 是原图高宽，$f$ 是 autoencoder spatial compression factor，$p$ 是 latent patch size。Dense self-attention 构造 $N\times N$ interaction matrix，主导项随 $O(N^2d)$ 增长，其中 $d$ 是 token width。增大 $f$ 或 $p$ 会减少计算，也会把更多内容压进单个 latent cell 或 token。

Architecture 路线更早开始。Ronneberger、Fischer 与 Brox 在 MICCAI 2015 论文 [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597 "官方论文页面") 中提出 contracting path、expanding path 与同分辨率 lateral skips；原始任务是医学图像分割，并不是生成。Ho 等人 2020 把这一结构改造成 time-conditioned denoiser：输入输出保持 image-like tensor 形状，在多尺度 residual blocks 中注入 timestep embedding，并在部分分辨率加入 attention。U-Net 的核心优势是用低分辨率层汇聚大范围 context，同时通过 skip connections 把浅层空间细节送回 decoder。

Peebles 与 Xie 在 2022 年发布、ICCV 2023 发表的论文 [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2209.12152 "官方论文页面") 中提出 DiT。它把 noisy latent patchify 成 $N$ 个 tokens，并通过 adaLN-Zero 将 time 与 class condition 转成 shift、scale 和 residual gate；zero-initialized gates 让 block 在初始化附近接近 identity。论文观察到受控模型族中 GFLOPs 与 sample quality 强相关，这是经验 scaling result，不是普适定理。

下图只保留理解系统所需的结构差异。左侧 U-Net 改变 spatial resolution，并用同尺度 skips 连接 encoder 与 decoder；右侧 DiT 主要保持 token grid，通过 patch embedding、self-attention 与 condition-gated residual blocks 处理信息。二者最终承担相同接口：接收 noisy representation、time 与 condition，返回同形状 prediction。它们都不自行决定 forward path、prediction target 或 sampler。

![Diffusion U-Net 与 DiT 的最小系统接口比较](/images/diffusion/d8_unet_dit_interfaces.png)

文本条件通常通过 cross-attention 进入这两类网络。令 $Q\in\mathbb R^{N_x\times d}$ 是 image/latent queries，$K,V\in\mathbb R^{N_c\times d}$ 是由 $N_c$ 个 condition tokens 投影得到的 keys 与 values，则

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt d}
\right)V.
$$

这个操作让 image tokens 聚合文本信息。它回答“条件怎样进入 denoiser”，并不等于 guidance：同一网络可以用 $s=1$ 普通条件采样，也可以进行 CFG 外推。

从 pixel U-Net 到 latent U-Net，再到 latent DiT，真正稳定的系统分工是：representation 决定被建模的信息与 token 数，architecture 决定函数近似和信息混合方式，objective 决定训练偏好。把某个端到端模型的提升全部归因于“换成 DiT”或“压缩到 latent”都缺少因果依据，因为数据、autoencoder、text encoder、loss 与 sampler 往往同时改变。

## 5. 2021—2022：sampler 消费已训练的向量场，而不是重新学习模型

训练结束后，network weights $\theta$ 已固定。Sampler 在有限次 model evaluations 内把 prior noise 沿 reverse dynamics 推回数据表示。它可以改变 stochasticity、formula 与 grid，却不能消除 learned score 本身的系统偏差。

Song、Meng 与 Ermon 在 ICLR 2021 论文 [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502 "官方论文页面") 中说明：DDPM objective 主要依赖 fixed-time marginals，并不把 denoiser 锁死在唯一 Markov reverse chain 上。DDIM 构造相同训练 marginals 的 non-Markovian family；$\eta=0$ 时轨迹是 deterministic，并可只访问 schedule 的 subsequence。速度来自跳过 noise levels，而不是单次 forward 更便宜；deterministic 也不表示 coarse discretization 精确可逆。

第三章的 probability-flow ODE 又把采样变成标准连续积分问题。可是真正有效的 diffusion solver 不必把整个 vector field 当作黑盒。Lu、Zhou、Bao、Chen、Li 与 Zhu 在 NeurIPS 2022 论文 [*DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*](https://arxiv.org/abs/2206.00927 "官方论文页面") 中利用 diffusion ODE 的 semilinear structure，对已知的 schedule-dependent linear term 精确积分，只近似 neural-network term。

令 $x_s$ 是较高噪声时刻 $s$ 的状态，$x_t$ 是较低噪声时刻 $t\le s$ 的状态；对 $r\in\{s,t\}$，$\alpha_r,\sigma_r$ 是相应的 signal 与 noise amplitude，并定义 half-log-SNR coordinate

$$
\lambda_r
=
\log\frac{\alpha_r}{\sigma_r}.
$$

若 $\hat\varepsilon_\theta(x,t)$ 是 noise predictor，DPM-Solver 的 exact solution representation 为

$$
\boxed{
x_t
=
\frac{\alpha_t}{\alpha_s}x_s
-
\alpha_t
\int_{\lambda_s}^{\lambda_t}
e^{-\lambda}
\hat\varepsilon_\theta(\hat x_\lambda,\lambda)
\,d\lambda.
}
$$

这里 $\hat x_\lambda$ 是 trajectory 在 coordinate $\lambda$ 处的状态。第一项精确传播 linear dynamics，困难只剩 network-output integral。DPM-Solver-1、2、3 用不同阶数近似该积分；在极少 evaluations、强 CFG 或存在模型误差时，高 formal order 不保证更好的 perceptual result。

Karras 等人 2022 的 EDM 从另一角度把 sampler 拆成 sigma coordinate、Euler/Heun formula、polynomial grid 与可选 stochastic churn。下图把常被一个“sampler 名称”捆绑的决定分成四列。阅读时应横向比较 model output、trajectory law、solver formula 与 grid，而不是只比较 DDIM、DPM-Solver 或 EDM 这些标签；同名实现若采用不同 prediction conversion、endpoint handling 或 grids，有限步结果也会不同。

![sampler 名称背后的 prediction、trajectory、solver 与 grid](/images/diffusion/d6_sampler_design_matrix.png)

计算成本应以 network function evaluations，简称 NFE，作为基本尺度。Euler 通常每个 interval 使用一次新 evaluation；Heun predictor--corrector 通常需要 endpoint 的第二次 evaluation；multistep solver 可以复用历史 outputs，但需要 startup；CFG 又常让每个 noise point 包含 conditional 与 unconditional 两份 prediction。因而“20 steps”“20 batched calls”与“20 sample-equivalent NFE”可能不是同一个成本。

Sampler 还与前面三层发生真实交互。Prediction conversion 在低 SNR 处可能放大误差；高 guidance scale 会增大 vector-field derivatives，使 coarse solver 更容易不稳定；latent compression 会降低一次 evaluation 的空间成本，却可能让 decoder error 成为最终瓶颈。因此不存在脱离 model parameterization、guidance 与 representation 的普适最佳 sampler。

现在可以回答本章标题：

> 一个现代 Diffusion 系统由四种分工共同决定：objective 决定网络学好哪些噪声层，guidance 决定实际采样的条件向量场，representation 与 architecture 决定一次函数近似的内容和成本，sampler 决定用多少次近似把该向量场积分成样本。

DDPM 在 2020 年给出了其中一套成功组合；后续工作的核心不是不断往同一模型上堆名字，而是逐步解除这些组件之间并非必要的绑定。评价一个系统时，也应分别报告它学了什么函数、在哪个空间学习、采样时怎样修改该函数，以及用了多少真实 network evaluations。

## 本章论文索引

| 时间        | 论文                                                                                          | 本章中的作用                                                                   |
| --------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 2015      | Ronneberger et al., *U-Net*, MICCAI                                                         | 建立多尺度 encoder--decoder 与 lateral skip 的结构祖先                              |
| 2020      | Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS                              | 给出 pixel U-Net、noise regression 与 ancestral sampling 的最小系统闭环             |
| 2021      | Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS                 | 用 noisy classifier gradient 改写 conditional reverse dynamics              |
| 2021      | Song et al., *Denoising Diffusion Implicit Models*, ICLR                                    | 说明同一 denoiser 不绑定唯一 reverse chain，并支持 deterministic subsequence sampling |
| 2021      | Esser et al., *Taming Transformers for High-Resolution Image Synthesis*, CVPR               | 建立感知压缩与 two-stage high-resolution generation 的背景                         |
| 2022      | Choi et al., *Perception Prioritized Training of Diffusion Models*, CVPR                    | 用 P2 weight 抑制 large-SNR clean-up region 的训练占比                           |
| 2022      | Ho & Salimans, *Classifier-Free Diffusion Guidance*, arXiv                                  | 用 conditional dropout 与双分支外推替代外部 classifier                              |
| 2022      | Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, NeurIPS | 拆分 preconditioning、training weight、solver 与 sampling grid                |
| 2022      | Lu et al., *DPM-Solver*, NeurIPS                                                            | 精确传播 diffusion ODE 的 linear term，并近似 neural integral                     |
| 2022      | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR        | 将 diffusion 移到 autoencoder latent space，并以 cross-attention 接入条件          |
| 2022      | Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models*, ICLR       | 引入 VP $v$-prediction，改善低 SNR 与少步生成的 target conditioning                  |
| 2022/2023 | Peebles & Xie, *Scalable Diffusion Models with Transformers*, arXiv / ICCV                  | 建立 patch-token、adaLN-Zero 与可扩展 DiT backbone                              |
| 2023      | Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*, ICCV            | 从 noise-level task conflict 出发设计统一 metric 下的 Min-SNR weight              |
