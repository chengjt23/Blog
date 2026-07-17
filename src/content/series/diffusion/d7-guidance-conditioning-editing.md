---
title: 条件生成、Guidance、反演与编辑
description: 区分 classifier guidance、CFG、条件接口、DDIM inversion、图像编辑与 posterior sampling 的职责。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 7
slug: d7-guidance-conditioning-editing
tags:
  - diffusion
  - guidance
  - image-editing
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 覆盖条件进入系统的不同层级、控制强度与正确性标准，不把所有条件方法统称为 guidance。
---
无条件 diffusion 回答的是“从数据分布中生成什么”；真实系统更常遇到的问题却是“生成哪一类、符合哪段文本、遵循哪张边缘图、保留哪幅输入图像、像哪个特定主体，或者满足什么物理观测”。这些请求都叫“条件”，但条件进入系统的位置完全不同。

本章的中心观点是：**不要把所有条件方法都叫 guidance。** 有的方法在采样时修改 score，有的方法把条件喂进网络，有的方法控制 latent trajectory，有的方法学习新的 embedding 或权重，还有的方法试图采样 Bayesian posterior。它们可以组合，却有不同的公式、计算成本和正确性标准。

![Bayes guidance、conditional score、CFG field 与隐式 classifier。](/images/diffusion/d7_guidance_vector_fields.png)

本章分三层阅读。第一层只需掌握“条件进入哪一层”的地图和 classifier-free guidance（CFG）的常见公式；第二层跟随 Bayes score、Gaussian mean shift、DDIM inversion 和 posterior score 的完整推导；研究层则需要分辨 pointwise score identity、time-dependent dynamics、finite solver、likelihood approximation 与 posterior correctness。

***

## 1. 先画地图：五种条件方法改变的不是同一个对象

设无条件 denoiser 为 $\epsilon_\theta(x_t,t)$，条件 denoiser 为 $\epsilon_\theta(x_t,t,c)$。同一个条件 $c$ 至少可以从五个位置进入：

| 类别                           | 被改变的对象                                 | 代表方法                                             | 主要评价问题                      |
| ---------------------------- | -------------------------------------- | ------------------------------------------------ | --------------------------- |
| score/vector-field guidance  | 采样时的 score 或 denoiser output           | classifier guidance、CFG、CFG++                    | 条件一致性、分布偏移、稳定性              |
| architecture conditioning    | 网络内部 feature、attention 或 residual path | cross-attention、ControlNet、T2I/IP-Adapter        | 条件表达力、可控粒度、训练成本             |
| trajectory editing/inversion | 初始状态、latent path 或 attention map       | SDEdit、DDIM inversion、Prompt-to-Prompt、Null-text | 重建、可编辑性、局部保真                |
| personalization              | embedding 或模型参数                        | Textual Inversion、DreamBooth、LoRA                | 主体保真、组合性、过拟合                |
| inverse-problem posterior    | likelihood 或 measurement consistency   | RePaint、DDRM、DPS                                 | 数据一致性、posterior correctness |

这张表还暗含五种不同的“正确”：

1. 文本或类别语义正确；
2. 空间结构条件正确；
3. 输入图像重建正确；
4. 观测残差小；
5. 输出分布真的是 $p(x_0\mid y)$。

前四项都不能自动推出第五项。例如一张修复图像可能与观测完全一致，却只落在 posterior 的某一个 mode；一个编辑器也可能很好地服从指令，但根本没有定义观测 likelihood。

![cross-attention、ControlNet 与 Adapter 的条件注入位置。](/images/diffusion/d7_conditioning_interfaces.png)

***

## 2. Bayes score identity：classifier guidance 的数学起点

先固定一个噪声时刻 $t$。设 $p_t(x)$ 是 noisy marginal，$c$ 是类别或一般条件。Bayes rule 给出

$$
p_t(x\mid c)
=\frac{p_t(x)p_t(c\mid x)}{p_t(c)}.
$$

对 $x$ 求 log-gradient，normalizer $p_t(c)$ 消失：

$$
\boxed{
\nabla_x\log p_t(x\mid c)
=\nabla_x\log p_t(x)
+\nabla_x\log p_t(c\mid x).
}
$$

这条式子是 exact identity。它说：如果已经知道无条件 score，再得到 noisy classifier $p_t(c\mid x)$ 的 input gradient，就能构造 conditional score。

这里的 classifier 不是普通 clean-image classifier。它必须接收与生成器相同噪声分布上的 $x_t$ 和时刻 $t$：

$$
p_\phi(c\mid x_t,t).
$$

若直接把 clean classifier 用在高噪声图像上，输入已经严重 out of distribution，Bayes identity 中需要的 likelihood gradient 与实际网络 gradient 没有相等保证。

### 2.1 Scale 大于 1 改变了目标分布

实践常把 classifier gradient 乘 $s>1$：

$$
\tilde s_t(x,c)
=\nabla_x\log p_t(x)
+s\nabla_x\log p_t(c\mid x).
$$

若两个 gradient 都 exact，则

$$
\tilde s_t(x,c)
=\nabla_x\log\left[p_t(x)p_t(c\mid x)^s\right].
$$

因此相应 normalized density 为

$$
\boxed{
\tilde p_{t,s}(x\mid c)
\propto p_t(x)p_t(c\mid x)^s.
}
$$

$s=1$ 才是原始 Bayes conditional。$s>1$ 将 likelihood temper，强化 classifier 偏好的区域，也通常减少 diversity。Dhariwal 与 Nichol 的 precision/recall 结果正体现了这项 fidelity--diversity trade-off；它不是免费的条件增强。

***

## 3. 从 score 到 reverse Gaussian：均值为什么平移

[Dhariwal--Nichol](https://arxiv.org/abs/2105.05233 "官方论文页面") 不只写了 score identity，还给出离散 reverse kernel 中 classifier gradient 的具体位置。

设未引导的一步 reverse kernel 为

$$
p_\theta(x_{t-1}\mid x_t)
=\mathcal N(x_{t-1};\mu_t,\Sigma_t).
$$

条件 reverse kernel 按 Bayes 关系近似为

$$
p_{\theta,\phi}(x_{t-1}\mid x_t,c)
\propto
p_\theta(x_{t-1}\mid x_t)
p_\phi(c\mid x_{t-1},t-1).
$$

在 $x_{t-1}=\mu_t$ 附近一阶展开 classifier log-likelihood：

$$
\log p_\phi(c\mid x_{t-1},t-1)
\approx C+g_t^\top(x_{t-1}-\mu_t),
$$

其中

$$
g_t
=\nabla_x\log p_\phi(c\mid x,t-1)\big|_{x=\mu_t}.
$$

令 $u=x_{t-1}-\mu_t$。条件 kernel 的 log-density 中与 $u$ 有关的项是

$$
-\frac12u^\top\Sigma_t^{-1}u+g_t^\top u.
$$

配方：

$$
\begin{aligned}
-\frac12u^\top\Sigma_t^{-1}u+g_t^\top u
&=-\frac12(u-\Sigma_tg_t)^\top
\Sigma_t^{-1}(u-\Sigma_tg_t)\\
&\quad+\frac12g_t^\top\Sigma_tg_t.
\end{aligned}
$$

最后一项与 $u$ 无关，吸收到 normalizer 中，于是

$$
\boxed{
p_{\theta,\phi}(x_{t-1}\mid x_t,c)
\approx
\mathcal N(\mu_t+\Sigma_tg_t,\Sigma_t).
}
$$

这一步的责任边界必须保留：Bayes score identity 是 exact，**finite Gaussian transition 的均值平移使用了 classifier log-likelihood 的局部线性近似**。非线性 classifier 在一步跨度较大时不一定只改变均值，也可能改变有效 covariance 与非 Gaussian 形状。

### 3.1 映射到 epsilon prediction

在线性 Gaussian marginal

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
$$

下，D3 已得到

$$
s_\theta(x_t,t)
\approx\nabla_{x_t}\log p_t(x_t)
=-\frac{\epsilon_\theta(x_t,t)}{\sigma_t}.
$$

将 classifier gradient 加到 score：

$$
-\frac{\epsilon_{guided}}{\sigma_t}
=-\frac{\epsilon_\theta}{\sigma_t}
+s\nabla_{x_t}\log p_\phi(c\mid x_t,t).
$$

两边乘 $-\sigma_t$：

$$
\boxed{
\epsilon_{guided}
=\epsilon_\theta
-s\sigma_t\nabla_{x_t}\log p_\phi(c\mid x_t,t).
}
$$

VP/DDPM 记号中 $\sigma_t=\sqrt{1-\bar\alpha_t}$。classifier gradient 需要反向传播到高维输入，所以一次采样通常包含 denoiser forward、classifier forward 和 classifier-to-input backward；这也是 CFG 试图去掉的额外模型与计算负担。

***

## 4. Classifier-free guidance：把 classifier 藏进两个 denoiser 分支

Classifier guidance 的关键缺点是需要单独训练 noisy classifier，而且文本、图像 prompt 等高维条件并不总适合压成一个 discriminative head。[Ho--Salimans](https://arxiv.org/abs/2207.12598 "官方论文页面") 的方案是让同一个 conditional denoiser 同时学会“有条件”和“空条件”。

### 4.1 Conditional dropout training

训练时以概率 $p_{uncond}$ 将条件 $c$ 替换为空条件 $\varnothing$：

$$
\tilde c=
\begin{cases}
\varnothing,&B=1,\\
c,&B=0,
\end{cases}
\qquad B\sim\operatorname{Bernoulli}(p_{uncond}).
$$

epsilon objective 为

$$
\mathcal L_{CFG}(\theta)
=\mathbb E_{x_0,c,t,\epsilon,B}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t,\tilde c)
\right\|_2^2
\right].
$$

在无限容量、population MSE optimum 下，两种输入分别逼近

$$
\epsilon_\theta(x_t,t,c)
\to\mathbb E[\epsilon\mid x_t,c],
$$

$$
\epsilon_\theta(x_t,t,\varnothing)
\to\mathbb E[\epsilon\mid x_t].
$$

记作

$$
\epsilon_c=\epsilon_\theta(x_t,t,c),
\qquad
\epsilon_u=\epsilon_\theta(x_t,t,\varnothing).
$$

“同一个网络”指共享参数并使用不同 condition input，不是一次 forward 同时免费得到两个值。

### 4.2 两套 scale convention

Ho--Salimans 原文写成

$$
\boxed{
\tilde\epsilon
=(1+w)\epsilon_c-w\epsilon_u.
}
$$

许多 framework 则写成

$$
\boxed{
\epsilon_{cfg}
=\epsilon_u+s(\epsilon_c-\epsilon_u).
}
$$

展开第二式：

$$
\epsilon_{cfg}
=s\epsilon_c+(1-s)\epsilon_u.
$$

逐项比较可得

$$
\boxed{s=1+w.}
$$

因此几个常见位置是：

| 语义                          | Ho--Salimans $w$ | framework $s$ |
| --------------------------- | ---------------: | ------------: |
| unconditional branch        |             $-1$ |           $0$ |
| ordinary conditional branch |              $0$ |           $1$ |
| extrapolated guidance       |             $>0$ |          $>1$ |

如果论文说 $w=0$“关闭 guidance”，而代码里 guidance\_scale=0，两者不是同一个设置。教程、代码和实验表格必须写清 convention。

### 4.3 计算成本：batched call 不等于一支普通 forward

标准 CFG 需要计算 $\epsilon_u$ 和 $\epsilon_c$。实现可以把两份输入沿 batch dimension 拼接，只调用一次 model API：

```
eps_u, eps_c = model(
    concat([x_t, x_t]),
    t,
    concat([empty_condition, condition]),
).chunk(2)
eps_cfg = eps_u + scale * (eps_c - eps_u)
```

若按 D6 的 API-call convention，这可记作 1 次 batched NFE；但 batch 约翻倍，FLOPs、activation memory 与 latency 不等于一支 ordinary forward。报告 sampling cost 时应同时说明 branch count、batching 和实际硬件时间。

***

## 5. CFG 的 density 解释：点态恒等式为什么还不够

将 epsilon 换回 score：

$$
s_u=-\frac{\epsilon_u}{\sigma_t},
\qquad
s_c=-\frac{\epsilon_c}{\sigma_t}.
$$

CFG score 为

$$
s_{cfg}
=s_u+s(s_c-s_u)
=(1-s)s_u+s s_c.
$$

若 $s_u=\nabla\log p_t(x)$、$s_c=\nabla\log p_t(x\mid c)$ 都是 exact scores，则

$$
\begin{aligned}
s_{cfg}
&=(1-s)\nabla\log p_t(x)
+s\nabla\log p_t(x\mid c)\\
&=\nabla\log\left[
p_t(x)^{1-s}p_t(x\mid c)^s
\right].
\end{aligned}
$$

再用 $p_t(x\mid c)\propto p_t(x)p_t(c\mid x)$：

$$
\boxed{
s_{cfg}
=\nabla\log\left[p_t(x)p_t(c\mid x)^s\right].
}
$$

这解释了 CFG 与 classifier guidance 在理想极限中的联系，也解释了为什么 $s>1$ 通常强化条件并牺牲 diversity。

但是这条推导只证明了**每个固定 $t$ 上的点态 vector-field identity**。它还没有证明实际 sampler 输出某个简单的 $p_0(x\mid c)^s p_0(x)^{1-s}$。至少还差三层。

### 5.1 Neural fields 未必 conservative

有限网络给出的 $s_u,s_c$ 可能不是任何全局 scalar log-density 的 gradient。若 Jacobian 不对称或 line integral 依赖路径，线性组合依然只是一个 vector field。[Ho--Salimans](https://arxiv.org/abs/2207.12598 "官方论文页面") 已明确提醒这一点；D3 也讨论过 non-conservative learned score。

### 5.2 每个时刻的 power density 未必组成同一 forward process

即使每个 $t$ 上都能定义

$$
q_t^{(s)}(x)
\propto p_t(x)^{1-s}p_t(x\mid c)^s,
$$

这些 $q_t^{(s)}$ 未必满足原 forward SDE 的 Fokker--Planck equation，也未必来自某个固定 $q_0^{(s)}$ 加同一 corruption kernel。一个 time-indexed density family 需要动态一致性，不能只靠逐时刻归一化拼起来。

[Bradley--Nakkiran](https://arxiv.org/abs/2408.09000 "官方论文页面") 用 Gaussian counterexamples 说明：CFG ODE/SDE 的实际 marginals 可以不同于简单的 gamma-powered 叙事。这不否定上面的点态代数，而是否定从点态代数到通用动态结论的跳步。

### 5.3 Solver、thresholding 和 time schedule 继续改变路径

实际采样只在有限网格上调用 learned field。D6 已说明 guidance 会增加 curvature、stiffness 与 multistep history contamination。若再加入 thresholding、CFG rescale 或 limited interval，最终 vector field 已不是原始线性组合本身。

所以最稳妥的表述是：

> CFG 的 power-density 公式给出理想 compatible scores 下的局部几何解释；实际输出分布还取决于 time consistency、network error、guidance intervention 与 finite solver。

***

## 6. Guidance 过强以后：rescale、threshold 与 limited interval 修复的不是同一个问题

高 CFG scale 往往改善 prompt alignment，却产生过饱和、对比度异常、细节坍缩或 solver instability。下面三类方法经常同时出现在代码中，但它们作用在不同位置。

![likelihood tempering、limited interval 与 dynamic thresholding 是三种不同干预。](/images/diffusion/d7_guidance_controls.png)

### 6.1 CFG rescale：匹配输出尺度

记 raw guided prediction 为

$$
\epsilon_g=\epsilon_u+s(\epsilon_c-\epsilon_u).
$$

一种常见 rescale 先按样本或通道统计量匹配 conditional branch 的标准差：

$$
\epsilon_{g,rescaled}
=\epsilon_g
\frac{\operatorname{Std}(\epsilon_c)}
{\operatorname{Std}(\epsilon_g)+\delta}.
$$

再以 $r\in[0,1]$ 插值：

$$
\boxed{
\epsilon_{out}
=(1-r)\epsilon_g+r\epsilon_{g,rescaled}.
}
$$

它主要修复 output scale inflation，不撤销 CFG 的方向变化，也不是 Bayes correction。标准差沿哪些 dimensions 计算、是否作用于 epsilon、$v$ 还是 $x_0$ prediction，都会改变结果。[zero-terminal-SNR 与 rescale note](https://arxiv.org/abs/2305.08891 "官方论文页面") 还提醒：schedule endpoint、prediction target 和 guidance rescale 必须联动检查。

### 6.2 Imagen dynamic thresholding：约束 $\hat x_0$ 范围

[Imagen](https://arxiv.org/abs/2205.11487 "官方论文页面") 观察到高 guidance 会让 predicted clean image $\hat x_0$ 超出训练数据的 $[-1,1]$ 范围。static threshold 直接 clip：

$$
\hat x_0^{static}
=\operatorname{clip}(\hat x_0,-1,1).
$$

dynamic threshold 对每个样本取 absolute values 的高 percentile：

$$
r=\operatorname{quantile}_q(|\hat x_0|),
\qquad m=\max(1,r),
$$

$$
\boxed{
\hat x_0^{dynamic}
=\frac{\operatorname{clip}(\hat x_0,-m,m)}{m}.
}
$$

若 $m>1$，所有未饱和值也会被压缩。它不是普通 clip，而是 sample-dependent nonlinear map，会改变 data prediction 与下一步 sampler trajectory。像素空间中 $[-1,1]$ 有明确训练范围；latent-space representation 未必有相同语义，因此不能机械照搬。

### 6.3 Limited-interval guidance：把常数 scale 改成函数

标准 CFG 默认 $s(t)=s_\star$ 全程不变。[Kynkäänniemi 等](https://arxiv.org/abs/2404.07724 "官方论文页面") 直接挑战这一假设：

$$
s(t)
=1+(s_\star-1)
\mathbf 1\{t\in[t_{lo},t_{hi}]\}.
$$

在 framework convention 中，区间外 $s=1$ 是 ordinary conditional branch，不是 unconditional branch。论文在所测模型上发现：只在部分噪声区间启用 guidance 可以改善 FID/precision/recall，并减少 conditional evaluations。

但 $t_{lo},t_{hi}$ 不是通用常数。它们依赖 model、schedule、parameterization、sampler 和 metric。可以把 limited interval 理解为新的 sampling grid/control policy，而不能写成“高噪声 guidance 永远无用”的定理。

***

## 7. CFG++ 与 predictor--corrector：近期理论到底修正了什么

### 7.1 将 DDIM update 拆成 denoised point 与 correction direction

deterministic DDIM step 可写成

$$
x_s
=\alpha_s\hat x_0(x_t,t)
+\sigma_s\hat\epsilon(x_t,t),
\qquad s<t.
$$

普通 CFG 同时用 guided output 构造 $\hat x_0^{cfg}$ 和 residual/noise direction。高 scale 下，第二项也带有很强的 conditional extrapolation，可能把轨迹推离 unconditional data geometry。

[CFG++](https://arxiv.org/abs/2406.08070 "官方论文页面") 的核心拆分是：guided branch 负责 denoised target，而 unconditional branch 负责 correction direction。抽象写成

$$
\boxed{
x_s^{CFG++}
=\alpha_s\hat x_0^{guided}(x_t,t,c)
+\sigma_s\epsilon_u(x_t,t).
}
$$

具体 DDIM/DDPM implementation 还包含各自 coefficients，但职责分离不变。它解释了为什么 CFG++ 可在较小 interpolation scale 下获得条件增强，并改善 inversion。

“manifold-constrained”应理解为论文给出的几何解释：unconditional correction 更贴近 data manifold 的 denoising direction。它不是显式求解一个已知 manifold projection，也不保证任意 learned denoiser 下严格留在某个数学流形上。

### 7.2 CFG as predictor--corrector

[Bradley--Nakkiran](https://arxiv.org/abs/2408.09000 "官方论文页面") 提出另一种动力学解释：

1. conditional denoising step 作为 predictor，将状态朝条件数据方向推进；
2. unconditional score 驱动的 Langevin step 作为 corrector，将状态重新拉回高密度 data geometry。

概念上可写为

$$
x'\leftarrow\operatorname{Predict}_c(x_t),
$$

$$
x''\leftarrow x'
+\eta\nabla_x\log p_t(x')
+\sqrt{2\eta}\,z,
\qquad z\sim\mathcal N(0,I).
$$

论文在特定 SDE limit、scale 和算法定义下给出等价关系，并据此构造 predictor-corrector guidance。这里的 corrector 与 D6 中“用第二个 slope 提升 ODE order”的 Heun corrector 不是一回事：前者是 distributional/Langevin correction，后者是 numerical quadrature correction。

CFG++ 与 predictor-corrector interpretation 共同说明：条件方向与 data-density correction 可以分开设计。但它们不是同一个算法，也不能把各自 theorem 跨出原 SDE、离散公式和 assumptions 使用。

***

## 8. Architecture conditioning：条件先进入网络，再谈 guidance

如果 denoiser 根本没有条件输入，就只能依赖外部 classifier gradient。现代 text-to-image systems 通常先训练 conditional denoiser，再按需叠加 CFG。

### 8.1 Cross-attention 的张量推导

设某层 spatial/image features 为

$$
X\in\mathbb R^{n_x\times d_x},
$$

condition encoder 输出 tokens

$$
C=\tau_\theta(c)
\in\mathbb R^{n_c\times d_c}.
$$

线性投影：

$$
Q=XW_Q\in\mathbb R^{n_x\times d},
$$

$$
K=CW_K\in\mathbb R^{n_c\times d},
\qquad
V=CW_V\in\mathbb R^{n_c\times d_v}.
$$

attention logits 的 shape 是 $n_x\times n_c$：

$$
L=\frac{QK^\top}{\sqrt d}.
$$

对 condition-token dimension 做 softmax：

$$
M_{ij}
=\frac{\exp L_{ij}}
{\sum_{k=1}^{n_c}\exp L_{ik}},
\qquad
\sum_jM_{ij}=1.
$$

输出为

$$
\boxed{
\operatorname{Attn}(X,C)
=MV
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V.
}
$$

[Latent Diffusion](https://arxiv.org/abs/2112.10752 "官方论文页面") 用这一接口把文本、layout 等条件注入 U-Net。GLIDE 与 Imagen 则展示了 large text encoder、conditional diffusion 与 CFG 组合成系统后的能力。

Cross-attention 是 $\epsilon_\theta(x_t,t,c)$ 的**计算架构**。它不等于

$$
\nabla_x\log p_t(c\mid x),
$$

attention weight 也不是 calibrated likelihood 或 causal explanation。

### 8.2 CFG 与 cross-attention 可以同时存在

conditional branch 计算

$$
\epsilon_c
=\epsilon_\theta(x_t,t,C),
$$

unconditional branch 计算

$$
\epsilon_u
=\epsilon_\theta(x_t,t,C_{empty}).
$$

两支内部都可以经过同一个 cross-attention module，只是 condition tokens 不同。CFG 在网络输出端组合两支；cross-attention 在每支内部决定条件如何进入 feature。把两者混成“attention guidance”会丢失这层结构。

***

## 9. 结构控制：ControlNet、T2I-Adapter 与 IP-Adapter

文本能表达“一个人抬手”，却很难精确指定每个关节、边缘或深度值。结构控制方法把 spatial condition 直接注入 intermediate features。

### 9.1 ControlNet：从零 residual 开始

设预训练 block 为

$$
y=F(x;\Theta).
$$

[ControlNet](https://arxiv.org/abs/2302.05543 "官方论文页面") 锁定这条 branch，复制一个 trainable block $F_c$，并使用 zero-convolution $Z_1,Z_2$ 接入条件：

$$
\boxed{
y_c
=F(x;\Theta)
+Z_2\left(
F_c(x+Z_1(c);\Theta_c)
\right).
}
$$

初始化时 $Z_1,Z_2$ 的 weights 与 bias 为零，所以

$$
Z_2(\cdot)=0,
\qquad
y_c=F(x;\Theta).
$$

这保证训练起点不突然破坏 pretrained function；它不保证训练后 residual 仍为零，也不保证任意 optimizer、normalization 和 mixed-precision 设置下函数永远无扰动。

### 9.2 T2I-Adapter：小模块注入多尺度 condition features

T2I-Adapter 冻结 text-to-image backbone，用较小的 condition-specific adapters 将 edge、depth、sketch 等转换为多尺度 feature maps，再注入对应 U-Net resolutions。与 ControlNet 相比，它不必复制完整 backbone，训练和存储成本较低；代价是 adapter capacity 与注入接口限制表达力。

ControlNet 与 T2I-Adapter 都属于 architecture conditioning，而不是在 sampler 外加一个 Bayes gradient。两者可以继续配 CFG，因为 text branch 仍有 conditional/unconditional outputs。

### 9.3 IP-Adapter：文本与图像 prompt 分开做 attention

若把 image tokens 与 text tokens 简单拼接，二者在同一个 softmax 中竞争。IP-Adapter 保留共享 query，却为图像 prompt 使用独立 key/value：

$$
Z_t
=\operatorname{Attn}(Q,K_t,V_t),
$$

$$
Z_i
=\operatorname{Attn}(Q,K_i,V_i),
$$

$$
\boxed{
Z_{new}=Z_t+\lambda_i Z_i.
}
$$

其中 $\lambda_i$ 控制 image prompt branch 的 feature contribution。它不是 likelihood temperature；改变 $\lambda_i$ 不等同于将某个 $p(c\mid x)$ 提幂。[IP-Adapter note](https://arxiv.org/abs/2308.06721 "官方论文页面") 与本章代码中的 ip\_adapter\_attention 对应这一解耦。

***

## 10. SDEdit：从输入图像的中间噪声层开始

编辑问题给定 guide image $x^{guide}$。若从纯噪声开始，输出可能真实但不保留 guide；若完全不加噪，模型几乎没有改动空间。[SDEdit](https://arxiv.org/abs/2108.01073 "官方论文页面") 在中间时刻 $t_0$ 建立折中。

VP 写法中先采样

$$
x_{t_0}
=\alpha_{t_0}x^{guide}
+\sigma_{t_0}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
$$

再从 $t_0$ 到 0 运行 pretrained reverse SDE/ODE 或离散 sampler。

$t_0$ 小时：

- guide 信息保留多；
- 输入 artifacts 也可能保留；
- 可发生的语义变化有限。

$t_0$ 大时：

- reverse model 有更大自由度；
- 输出更接近 learned data distribution；
- guide structure 更容易丢失。

这是一条 realism--faithfulness control curve。SDEdit 中的“guided”指输入图像引导 initialization，不是 classifier guidance。除非另行定义 observation model $p(y\mid x_0)$ 并证明 reverse dynamics 对应 posterior，否则“加噪再去噪”不自动是 Bayesian posterior sampling。

Palette 等 image-to-image diffusion 则直接在 paired data 上训练

$$
p_\theta(x_{target}\mid x_{source}),
$$

把 colorization、inpainting 或 restoration 作为 conditional generation。它与 SDEdit 的 pretrained prior workflow 不同，也仍不能单凭 conditional objective 声称 posterior calibration。

***

## 11. DDIM inversion：确定性为什么不等于精确可逆

D6 的 deterministic DDIM step 为

$$
\hat x_0(x_t,t)
=\frac{x_t-\sigma_t\epsilon_\theta(x_t,t,c)}{\alpha_t},
$$

$$
\boxed{
x_s
=\alpha_s\hat x_0(x_t,t)
+\sigma_s\epsilon_\theta(x_t,t,c),
\qquad s<t.
}
$$

给定 $x_t$ 与 model，这一步没有额外随机噪声。于是很容易产生一个错误推论：“deterministic，所以把时序倒过来就能精确求逆。”

真正从 $x_s$ 恢复 $x_t$ 时，右端需要的是未知 future point 上的

$$
\epsilon_\theta(x_t,t,c),
$$

而不是当前已知点上的

$$
\epsilon_\theta(x_s,s,c).
$$

实践 DDIM inversion 用后者作显式近似：

$$
\hat x_0^{inv}
=\frac{x_s-\sigma_s\epsilon_\theta(x_s,s,c)}{\alpha_s},
$$

$$
\boxed{
x_t^{inv}
\approx
\alpha_t\hat x_0^{inv}
+\sigma_t\epsilon_\theta(x_s,s,c).
}
$$

若 epsilon prediction 在这一步冻结且一致，forward/backward algebra 恰好互逆；本章代码验证了 roundoff error 约 $5.1\times10^{-15}$。真实 denoiser 随 $x,t,c$ 非线性变化，两边使用的 prediction 不同，round-trip error 就会累积。

![有限步 DDIM inversion 的误差随步数和 nonlinear guided field 改变。](/images/diffusion/d7_ddim_inversion_error.png)

说明代码中，同一 nonlinear field 在 guidance scale 7 时：

- 8 步 round-trip error 约 $3.57\times10^{-1}$；
- 128 步降到约 $1.41\times10^{-2}$；
- 冻结 epsilon 的所有网格只剩 floating-point roundoff。

这个 toy result 说明误差来源，不是对 Stable Diffusion reconstruction quality 的 benchmark。更细网格通常减小 smooth-field discretization error，但高 guidance、attention intervention、thresholding 和 endpoint mismatch 都可能破坏简单的收敛趋势。

***

## 12. Prompt-to-Prompt：编辑 cross-attention trajectory

[Prompt-to-Prompt](https://arxiv.org/abs/2208.01626 "官方论文页面") 的观察是：文本 token 与空间位置之间的 cross-attention maps 很大程度上决定布局。如果只改 prompt 并完全重新采样，非编辑区域也可能移动；若控制 attention trajectory，则可以把改变集中到特定 token 或时段。

记 source prompt 在时刻 $t$、层 $\ell$ 的 attention map 为

$$
M_{t,\ell}^{src}
=\operatorname{softmax}
\left(
\frac{Q_{t,\ell}K_{src,\ell}^\top}{\sqrt d}
\right),
$$

target prompt map 为 $M_{t,\ell}^{tgt}$。三类操作是：

### 12.1 Word swap

对 aligned tokens，在选定时段用 source map 替换 target map：

$$
\tilde M_{t,\ell}^{tgt}[:,j_{tgt}]
\leftarrow
M_{t,\ell}^{src}[:,j_{src}].
$$

这保留原 token 的空间布局，同时让 value/content 来自新 prompt。

### 12.2 Prompt refinement

保留词沿用 source attention，新增词保留 target attention。需要 token alignment map $A$：

$$
\tilde M^{tgt}
=M^{src}A+M^{tgt}(I-P_A),
$$

其中 $P_A$ 表示已对齐 token 的选择。实现会处理 tokenizer 把一个词拆成多个 subwords 的情况。

### 12.3 Attention re-weighting

对指定 token $j$ 乘权重 $\gamma_j$：

$$
\tilde M[:,j]=\gamma_jM[:,j],
$$

再按 implementation 需要归一化或交给后续 attention computation。这提供“更红”“更多条纹”等连续控制。

Attention control 是 internal trajectory intervention。attention map 不是精确 segmentation mask，也不是 causal attribution theorem。方法最初主要面向 synthesized images；真实图像还需要先找到可重建且可编辑的 latent trajectory。

***

## 13. Null-text inversion：不改 prompt，逐时刻优化 unconditional embedding

[Null-text inversion](https://arxiv.org/abs/2211.09794 "官方论文页面") 处理真实图像编辑。直接以高 CFG scale 做 DDIM inversion 会放大每步 prediction mismatch，重建变差；完全优化 model weights 又容易损害 editability。

方法分两阶段。

### 13.1 Pivot trajectory

先把输入图像编码为 $z_0^\star$，以 framework scale $s=1$ 做近似 DDIM inversion：

$$
z_0^\star\to z_1^\star\to\cdots\to z_T^\star.
$$

这条轨迹 reconstruction 不够精确，但保留了可编辑的 conditional model geometry，作为 optimization pivot。

### 13.2 Per-timestep null embedding optimization

记 prompt conditional embedding 为 $C$，空文本 embedding 为 $\phi_t$。在 reverse step $t\to t-1$，固定 model 与 $C$，只优化 $\phi_t$：

$$
\boxed{
\phi_t^\star
=\arg\min_{\phi_t}
\left\|
z_{t-1}^\star
-\operatorname{DDIMStep}
(z_t;C,\phi_t,s)
\right\|_2^2.
}
$$

从 $t=T$ 到 1 依次优化，并用上一时刻结果 warm start。最终得到

$$
\{\phi_T^\star,\ldots,\phi_1^\star\}.
$$

它们是**一张图像、一条轨迹、逐时刻特定**的 unconditional embeddings，不是一个通用 unconditional text representation。优化后可结合 Prompt-to-Prompt 改 conditional prompt，同时使用这些 $\phi_t^\star$ 保持 reconstruction。

Null-text inversion 提高了重建与可编辑性的兼容性，但仍不是 general exact inverse：pivot 是有限步近似，loss 只约束离散 latent points，VAE encoding/decoding 也有误差。

***

## 14. InstructPix2Pix：把编辑从逐图优化变成 learned conditional task

Prompt-to-Prompt 与 Null-text 需要 attention control 或 per-image optimization。[InstructPix2Pix](https://arxiv.org/abs/2211.09800 "官方论文页面") 直接学习

$$
p_\theta(x_{edited}\mid x_{input},c_{instruction}).
$$

训练数据由 instruction generation 与 Prompt-to-Prompt image pairs 合成。模型同时有 image condition $c_I$ 与 text condition $c_T$，训练时对二者进行层级 dropout，得到三支 output：

$$
\epsilon_{\varnothing,\varnothing},
\qquad
\epsilon_{I,\varnothing},
\qquad
\epsilon_{I,T}.
$$

采样时双 guidance：

$$
\boxed{
\begin{aligned}
\tilde\epsilon
&=\epsilon_{\varnothing,\varnothing}\\
&\quad+s_I
(\epsilon_{I,\varnothing}
-\epsilon_{\varnothing,\varnothing})\\
&\quad+s_T
(\epsilon_{I,T}
-\epsilon_{I,\varnothing}).
\end{aligned}
}
$$

第一条差分衡量 input image condition 的增量，第二条差分是在已知 input image 后 text instruction 的增量。$s_I$ 与 $s_T$ 分别控制 image faithfulness 和 instruction strength。

这不是任意两个条件都能直接套用的 inclusion--exclusion 公式；它依赖训练时的 dropout hierarchy 和三个 branch 的语义。synthetic pairs 也继承 language model 与 Prompt-to-Prompt 的偏差。

***

## 15. Personalization：学一个词、微调整个模型，还是只学低秩更新

编辑改变一张输入图像；personalization 让模型记住一个跨 prompt 重用的主体或风格。三种经典方法应按“优化什么”分类。

### 15.1 Textual Inversion：只学习 token embedding

引入 placeholder token $S_\star$，其 embedding 为 $v$。冻结 text encoder 的其他参数与 diffusion model，只优化

$$
\boxed{
v_\star
=\arg\min_v
\mathbb E_{x,t,\epsilon}
\left[
\left\|
\epsilon
-\epsilon_\theta
(\alpha_t x+\sigma_t\epsilon,t,c_v)
\right\|_2^2
\right].
}
$$

这里 $c_v$ 是包含 placeholder embedding 的 prompt representation。[Textual Inversion](https://arxiv.org/abs/2208.01618 "官方论文页面") 的优点是参数少、模型冻结、embedding 易分享；局限是一个或少数 embedding 的 capacity 可能不足以精确表达复杂主体。

不要与 Null-text inversion 混淆：

- Textual Inversion 是跨图像 personalization training；
- Null-text inversion 是单图逐时刻 trajectory optimization。

### 15.2 DreamBooth：模型 fine-tuning 与 prior preservation

DreamBooth 用稀有 identifier 与 class noun 组成 prompt，例如“a \[V] dog”，并 fine-tune model parameters。instance denoising loss 记为

$$
\mathcal L_{instance}
=\mathbb E
\left[
\|\epsilon-\epsilon_\theta(x_t,t,c_{instance})\|_2^2
\right].
$$

只训练少量 subject images 容易 language drift：模型把整个 class “dog” 收缩成这个特定主体。方法用原模型按 class prompt 生成 prior samples，加入

$$
\mathcal L_{prior}
=\mathbb E
\left[
\|\epsilon-\epsilon_\theta(x_t^{prior},t,c_{class})\|_2^2
\right],
$$

$$
\boxed{
\mathcal L_{DreamBooth}
=\mathcal L_{instance}
+\lambda\mathcal L_{prior}.
}
$$

prior preservation 是 regularizer，不保证 class distribution 完全不变。主体 fidelity、prompt editability、class diversity 与 memorization 需要分别评价。[DreamBooth note](https://arxiv.org/abs/2208.12242 "官方论文页面") 还记录了 official project 只有材料、没有原作者训练实现的 provenance 边界。

### 15.3 LoRA：限制参数更新的 rank

对线性层

$$
y=W_0x,
\qquad
W_0\in\mathbb R^{d_{out}\times d_{in}},
$$

冻结 $W_0$，令更新为

$$
\boxed{
\Delta W=BA,
}
$$

其中

$$
A\in\mathbb R^{r\times d_{in}},
\qquad
B\in\mathbb R^{d_{out}\times r}.
$$

于是

$$
\operatorname{rank}(\Delta W)\le r,
$$

可训练参数从 $d_{out}d_{in}$ 变为

$$
r(d_{in}+d_{out}).
$$

forward 为

$$
y=W_0x+\gamma BAx.
$$

LoRA 是 parameterization，不是 personalization objective；它可以承载 DreamBooth-style loss、风格 fine-tuning 或其他任务。[LoRA 原论文](https://arxiv.org/abs/2112.10741 "官方论文页面") 面向语言模型，后续 diffusion 使用是方法迁移，不能把它写成最早提出 diffusion personalization 的论文。

***

## 16. Inverse problems：exact posterior score 中真正难的是 likelihood integral

现在给条件一个明确的概率语义。clean signal $x_0\sim p_0$，观测模型

$$
y=A(x_0)+n,
$$

其中 Gaussian noise 情形

$$
n\sim\mathcal N(0,\sigma_y^2I),
\qquad
p(y\mid x_0)
\propto
\exp\left[
-\frac{\|y-A(x_0)\|_2^2}{2\sigma_y^2}
\right].
$$

目标是 posterior

$$
p_0(x_0\mid y)
\propto p_0(x_0)p(y\mid x_0).
$$

在 noisy time $t$，Bayes score 仍然 exact：

$$
\boxed{
\nabla_{x_t}\log p_t(x_t\mid y)
=\nabla_{x_t}\log p_t(x_t)
+\nabla_{x_t}\log p_t(y\mid x_t).
}
$$

但观测只直接依赖 $x_0$，所以

$$
\boxed{
p_t(y\mid x_t)
=\int
p(y\mid x_0)
p(x_0\mid x_t)
dx_0.
}
$$

这就是困难。即使 $p(y\mid x_0)$ 是简单 Gaussian，只要 data prior 非 Gaussian，积分后的 $p_t(y\mid x_t)$ 也通常没有闭式。

在 $t=0$ 且 $A(x)=Hx$ 时，clean likelihood score 的确是

$$
\nabla_{x_0}\log p(y\mid x_0)
=\frac1{\sigma_y^2}H^\top(y-Hx_0).
$$

不能把这条 clean-state 公式直接替换 $x_0$ 为 $x_t$ 并声称对所有噪声时刻 exact。

***

## 17. DPS：用 Tweedie posterior mean 替代一个分布积分

D3 的 VP Tweedie identity 给出

$$
\hat x_0(x_t)
=\mathbb E[x_0\mid x_t]
=\frac{x_t+\sigma_t^2\nabla_{x_t}\log p_t(x_t)}{\alpha_t}.
$$

用 learned score 替换 exact score，就得到 denoised estimate。

[Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687 "官方论文页面") 的关键近似是

$$
\boxed{
p_t(y\mid x_t)
=\mathbb E_{p(x_0\mid x_t)}[p(y\mid x_0)]
\approx
p(y\mid\hat x_0(x_t)).
}
$$

注意左右两边的操作顺序不同：

$$
\mathbb E[f(X)]
\quad\text{vs.}\quad
f(\mathbb E[X]).
$$

除非 $f$ 线性或 posterior 集中，二者一般不相等。

Gaussian measurement 下：

$$
\log p(y\mid\hat x_0(x_t))
=C-\frac1{2\sigma_y^2}
\|y-A(\hat x_0(x_t))\|_2^2.
$$

对 $x_t$ 求梯度：

$$
\boxed{
\nabla_{x_t}\log p_t(y\mid x_t)
\approx
-\frac1{2\sigma_y^2}
\nabla_{x_t}
\|y-A(\hat x_0(x_t))\|_2^2.
}
$$

链式法则展开为

$$
\nabla_{x_t}\log p(y\mid\hat x_0)
=J_{\hat x_0}(x_t)^\top
J_A(\hat x_0)^\top
\frac{y-A(\hat x_0)}{\sigma_y^2}.
$$

所以实现需要通过 denoiser 与 forward operator $A$ 反向传播。最终 surrogate posterior score 为

$$
\boxed{
s_{DPS}(x_t,t,y)
=s_\theta(x_t,t)
-\rho_t
\nabla_{x_t}
\|y-A(\hat x_0(x_t))\|_2^2.
}
$$

论文实现把系数和 residual normalization 吸收到 step size $\rho_t$ 或 $\zeta_t$ 中；它不必等于物理 likelihood 中的固定 $1/(2\sigma_y^2)$。

![解析 Gaussian-mixture posterior score 与 DPS plug-in surrogate 的差异。](/images/diffusion/d7_posterior_dps_comparison.png)

本章代码构造一个可解析 Gaussian-mixture prior、线性观测和中等噪声时刻。左图是 exact posterior score，中图是 DPS plug-in score，右图是 pointwise error；posterior-density-weighted RMS 约为 3.447。这个例子不是说 DPS 一定误差很大，而是证明：**Bayes decomposition exact，不会让 plug-in likelihood 自动 exact。**

DPS 的 Jensen-gap theorem 量化特定 surrogate 的差距，并解释某些 measurement-noise regime；它不等于整个离散 chain 收敛到 exact posterior 的 theorem。score error、Tweedie error、likelihood surrogate、step-size、solver 与 finite-time error仍需分别处理。

***

## 18. Palette、RePaint、DDRM 与 DPS 在逆问题地图中的位置

### 18.1 Palette：paired conditional generator

Palette 在 paired source/target data 上训练统一 image-to-image diffusion model。它适合有监督 colorization、inpainting 与 restoration；目标是 learned conditional distribution。若 paired training distribution 与真实 observation process 不一致，不能把它自动解释为给定物理 likelihood 的 posterior sampler。

### 18.2 RePaint：known region replacement 与 resampling

Inpainting mask 将 pixels 分为 known $K$ 与 unknown $U$。RePaint 在每个 reverse time：

1. 从 forward noising law 采样与观测一致的 known region；
2. 从 unconditional reverse model 更新 unknown region；
3. 合并两部分；
4. 通过 forward/backward time jumps 重复 resampling，改善边界协调。

抽象写成

$$
x_{t-1}
=m\odot x_{t-1}^{known}
+(1-m)\odot x_{t-1}^{model}.
$$

它强化 hard-mask consistency 和 harmonization，但 replacement/resampling 本身不构成一般 noisy-observation posterior correctness theorem。

### 18.3 DDRM：在线性算子的 SVD coordinates 中恢复

对

$$
y=Hx_0+n,
$$

作

$$
H=U\Sigma V^\top.
$$

旋转到 spectral coordinates：

$$
\bar y=U^\top y,
\qquad
\bar x=V^\top x.
$$

每个 singular direction 根据 singular value、measurement noise 与 diffusion noise 决定更新：可观测方向融合 measurement，null-space directions 依赖 generative prior。[DDRM](https://arxiv.org/abs/2201.11793 "官方论文页面") 的理论与 objective relation 依赖 linear/SVD structure 和具体参数条件；不能直接推广到任意 nonlinear $A$。

### 18.4 DPS：可微 nonlinear operator 的通用接口

DPS 不要求完整 SVD，只要能计算

$$
A(\hat x_0)
\quad\text{和}\quad
\nabla_{x_t}\|y-A(\hat x_0(x_t))\|^2.
$$

因此更容易接入 phase retrieval、nonlinear deblurring 等算子；代价是 likelihood plug-in 和 step-size heuristic。这里的“通用”指 implementation interface 广，不表示对所有 nonlinear inverse problems 都有统一 exactness guarantee。

| 方法      | 是否需 task-specific training | operator assumption              | posterior 边界                       |
| ------- | -------------------------- | -------------------------------- | ---------------------------------- |
| Palette | 是，paired data              | training pairs 定义                | learned conditional，不自动 calibrated |
| RePaint | 否                          | hard mask/known pixels           | consistency workflow               |
| DDRM    | 否，复用 prior                 | linear $H$、SVD 结构                | theorem tied to construction       |
| DPS     | 否，复用 prior                 | differentiable $A$、显式 likelihood | plug-in approximate                |

***

## 19. 一条统一算法接口

条件 sampling 可以抽象为四个可替换部件：

1. condition\_encoder：将文本、图像、结构或 observation 编码；
2. conditional\_model：在 architecture 内产生 conditional output；
3. guidance\_wrapper：执行 classifier/CFG/CFG++/interval control；
4. solver\_step：执行 D6 的有限步 sampler。

伪代码：

```
input: terminal x_T, condition c, grid {t_N > ... > t_0}
state = x_T
for i = N, ..., 1:
    cond = condition_encoder(c)
    eps_c = denoiser(state, t_i, cond)

    if guidance is CFG-like:
        eps_u = denoiser(state, t_i, empty_condition)
        eps = guidance_wrapper(eps_u, eps_c, t_i)
    elif guidance is classifier-based:
        grad = grad_x log classifier(c | state, t_i)
        eps = eps_c - sigma(t_i) * scale(t_i) * grad
    else:
        eps = eps_c

    if inverse_problem uses DPS:
        x0_hat = to_x0(state, eps, t_i)
        eps = add_likelihood_gradient(eps, state, x0_hat, y, A)

    state = solver_step(state, eps, t_i, t_{i-1})
return state
```

这段接口揭示几个工程事实：

- ControlNet/IP-Adapter 通常发生在 denoiser 内；
- CFG 在两个 denoiser outputs 之后；
- DPS 需要 gradient-through-denoiser，内存模式不同；
- solver theorem 只覆盖最后一层，除非把 wrapper 的 regularity 一起纳入；
- thresholding 可能作用于 to\_x0 后，必须再转换回 solver 需要的 parameterization。

***

## 20. 配套代码：解析分布而不是大型图像实验

d7\_guidance\_conditioning.py（补充材料暂未公开） 不下载 Stable Diffusion，也不训练 classifier。它用可解析对象检查本章最容易混淆的身份：

| 代码组件                             | 教学职责                                                 |
| -------------------------------- | ---------------------------------------------------- |
| IsotropicGaussianMixture         | exact density、responsibility 与 score                 |
| cfg\_score / power\_log\_density | CFG vector field 与 power-density identity            |
| dynamic\_threshold               | sample-dependent clipping/rescaling                  |
| cross\_attention                 | query/token shapes 与 row normalization               |
| ip\_adapter\_attention           | text/image K/V 分支相加                                  |
| ddim\_inversion\_round\_trip     | frozen-field exactness 与 nonlinear finite-step error |
| DiffusedMixturePosterior         | 线性观测下 exact $p(x_t\mid y)$                           |
| dps\_surrogate\_score            | Tweedie plug-in likelihood gradient                  |

内置 checks 结果：

- CFG $s=1+w$ convention error：$1.78\times10^{-15}$；
- power-density finite-difference score error：$2.07\times10^{-9}$；
- dynamic threshold bound excess：0；
- attention row-sum error：$2.22\times10^{-16}$；
- zero-convolution initial residual：0；
- LoRA factor rank：3；
- exact posterior analytic/numerical score error：$1.72\times10^{-10}$；
- frozen-epsilon DDIM inversion error：$4.58\times10^{-16}$。

运行：

```
python diffusion\code\d7_guidance_conditioning.py --no-figures
python diffusion\code\d7_guidance_conditioning.py
```

代码只证明 algebra、shape 与构造例的行为，不证明真实大模型的 FID、CLIP score、编辑质量或 posterior convergence。

***

## 21. 技术演进：每一步解决了什么，又留下什么

| 时间                       | 问题                                                 | 方案                                       | 新局限或下一步                                          |
| ------------------------ | -------------------------------------------------- | ---------------------------------------- | ------------------------------------------------ |
| 2021 classifier guidance | conditional model 仍缺可调 fidelity                    | noisy classifier gradient 修改 score       | 额外 classifier、backward cost、tempered target      |
| 2021--2022 GLIDE/CFG     | classifier 难扩展到开放文本                                | condition dropout 与两支 denoiser 外推        | 两支计算、过强 guidance、field compatibility             |
| 2022 SDEdit/Palette      | 怎样利用输入图像而非只给文本                                     | 中间噪声起点；paired conditional diffusion      | fidelity--realism trade-off；不自动 posterior        |
| 2022 LDM cross-attention | 条件如何进入高分辨率 latent model                            | token-to-spatial attention               | attention 不等于 likelihood；架构复杂度留 D8               |
| 2022 Imagen              | 大 CFG scale 产生 saturation                          | static/dynamic threshold                 | nonlinear sampler intervention                   |
| 2022 RePaint/DDRM        | pretrained prior 怎样用于 inverse problems             | mask resampling；linear SVD coordinates   | operator-specific assumptions                    |
| 2023 DPS                 | nonlinear/noisy operator 难做 spectral decomposition | Tweedie plug-in likelihood gradient      | approximate $p(y\mid x_t)$、step-size sensitivity |
| 2023 P2P/Null-text       | 怎样局部编辑并处理真实图像                                      | attention control；pivot + null embedding | token alignment；per-image optimization           |
| 2023 ControlNet/Adapters | 文本不能精确指定空间结构                                       | residual branch 与 lightweight adapters   | 额外训练；condition conflicts                         |
| 2023 personalization     | 怎样记住少量样本中的主体                                       | embedding、fine-tuning、low-rank update    | capacity、遗忘、memorization                         |
| 2024 limited interval    | guidance 是否需要全程开启                                  | time-dependent guidance schedule         | interval 依赖模型与 sampler                           |
| 2025 CFG++/PC view       | power-density 故事与实际 dynamics 不一致                   | 分离 denoise/correction；PC 解释              | theorem 与算法依赖具体 dynamics                         |

这条路线不是“旧方法被新方法完全替代”。Classifier guidance 仍适合已有可微 reward/classifier 的任务；CFG 是文本系统默认接口；ControlNet/Adapter 解决结构输入；DPS 面向显式观测；它们处在不同坐标轴上。

***

## 22. 常见错误

1. **“所有条件方法都是 guidance。”** Cross-attention、ControlNet 与 Adapter 首先是 architecture conditioning。
2. **“CFG 不需要 classifier，所以没有 unconditional model。”** CFG 明确需要空条件 branch，只是与 conditional branch 共享参数。
3. **“论文 $w=0$ 等于代码 guidance\_scale=0。”** Ho--Salimans 与 framework convention 差 1：$s=1+w$。
4. **“CFG scale 大只是在同一 $p(x\mid c)$ 上采得更准。”** 理想解释已将 likelihood 提幂，目标本身改变。
5. **“两个 exact scores 的线性组合证明实际 CFG 输出 power density。”** 还缺 time consistency、learned-field、solver 与 intervention 分析。
6. **“Batched CFG 只算一次，所以计算与普通 conditional forward 相同。”** API call 可以一次，batch/FLOPs/显存仍增加。
7. **“Dynamic threshold 只是 harmless clip。”** 它是 sample-dependent nonlinear map，并缩放未饱和值。
8. **“Limited interval 有一个跨模型通用区间。”** 最佳区间依赖 schedule、model、sampler 和 metric。
9. **“ControlNet zero conv 让模型训练后仍等于原模型。”** 只保证初始化 residual 为零。
10. **“Attention map 是像素级概率或 causal explanation。”** 它是内部 normalized weight，不具备这些自动语义。
11. **“DDIM deterministic，所以精确可逆。”** inversion 需要未知 future-state model output，有限步只能近似。
12. **“Textual Inversion 与 Null-text inversion 是一类。”** 前者学概念 token，后者优化单图逐时刻空条件 embedding。
13. **“LoRA 是一种 diffusion loss。”** LoRA 只限制 $\Delta W$ 的参数化；loss 由 personalization task 决定。
14. **“观测残差为零就说明 posterior sampling 正确。”** 数据一致性不保证 mode weights、uncertainty 或 sample distribution 正确。
15. **“DPS 由 Bayes rule 推出，所以 exact。”** Bayes identity exact，$p(y\mid x_t)\approx p(y\mid\hat x_0)$ 是核心近似。
16. **“SDEdit、Palette、RePaint 都是 posterior samplers。”** 它们可解决条件生成/编辑/修复，但 posterior claim 需要额外观测模型与理论。
17. **“DDRM 可直接处理任意 nonlinear operator。”** 其解析结构依赖 linear $H$ 与 SVD coordinates。
18. **“CFG++ 与 predictor-corrector guidance 是同一个算法。”** 二者共享职责分离直觉，但更新与理论不同。

***

## 23. 章节小结

- 条件可以进入 score、network architecture、trajectory、parameters 或 observation likelihood；分类位置比方法名字更重要。
- classifier guidance 来自 exact Bayes score identity；离散 Gaussian mean shift 还使用局部线性近似。
- CFG 用 condition dropout 学 conditional/unconditional 两支，framework scale $s$ 与原文 $w$ 满足 $s=1+w$。
- exact compatible scores 下，CFG 对应 $p_t(x)^{1-s}p_t(x\mid c)^s$ 的点态 score；这不自动给出一般 time-dependent sampler 的最终分布定理。
- CFG rescale、dynamic threshold 与 limited interval 分别修改输出尺度、predicted clean range 与 time schedule。
- CFG++ 与 predictor-corrector view 将 conditional denoising 和 data-density correction 分开，但 theorem scope 不能外推。
- Cross-attention、ControlNet、T2I-Adapter 与 IP-Adapter 规定 conditional denoiser 怎样计算，不等于 Bayes likelihood gradient。
- SDEdit 控制 initialization；Prompt-to-Prompt 控制 attention；Null-text optimization 修正真实图像 trajectory；DDIM inversion 在有限步下不是精确逆。
- Textual Inversion、DreamBooth、LoRA 分别优化 embedding、model objective 与 low-rank parameter update。
- inverse problems 的 exact posterior score 包含 $p_t(y\mid x_t)$；DPS 用 Tweedie mean plug-in 近似这个积分，因此 posterior correctness 必须单独讨论。

***

## 24. 研究式思考题

1. **Time consistency。** 给定 $q_t^{(s)}\propto p_t^{1-s}p_{t,c}^s$，将它代入原 forward Fokker--Planck equation。哪些额外条件能让 $\{q_t^{(s)}\}$ 真正来自一个固定 $q_0^{(s)}$？Gaussian family 中先做完整计算。
2. **Non-conservative CFG。** 在二维构造两个 neural vector fields，使各自 Jacobian 有非零 antisymmetric part。它们的 CFG 组合何时可能恰好 conservative？如何用 closed-loop line integral 检查？
3. **Mean-shift remainder。** 对 classifier log-likelihood 保留二阶 Taylor term，重新完成 Gaussian 配方。effective covariance 怎样改变？Hessian 在什么条件下会使近似 kernel 不可归一化？
4. **Scale convention audit。** 选择一个主流 library，追踪 guidance\_scale 从 pipeline 到 scheduler/model wrapper 的代码路径。它采用 $s$ 还是 $w$，是否还有 rescale 或 threshold？
5. **Cost accounting。** 对 batched CFG 建立 latency model，包含 kernel launch、batch scaling、memory bandwidth 和 classifier-free branch cache。什么时候“一次 batched NFE”接近一次、1.5 次或两次 ordinary forward？
6. **Optimal interval。** 把 $s(t)$ 看成 control function，在固定 conditional-evaluation budget 下写出一个最优控制问题。目标若从 FID 换成 prompt alignment，解为什么会改变？
7. **CFG++ decomposition。** 从 deterministic DDIM transfer 展开 standard CFG 与 CFG++ 的差异。若 conditional/unconditional outputs 完全相同，两者怎样退化？若只在切向分量不同会怎样？
8. **Attention conflict。** Text、ControlNet 和 IP-Adapter 同时存在时，定义三个 condition branches 的局部 Jacobian。怎样检测一个 condition 的增强是否系统削弱另一个？
9. **Inversion order。** 将 DDIM inversion 解释为 PF ODE 的显式积分。在哪些 smoothness、endpoint 和 stability 条件下 round-trip error 随最大步长收敛？高 CFG 为何可能造成 order reduction？
10. **Null embedding identifiability。** 若不同 $\phi_t$ 产生相同一步 latent，optimization 是否可辨识？这种 non-identifiability 对后续 prompt edit 有何影响？
11. **Personalization capacity。** 比较一个 $d$-dimensional token embedding、rank-$r$ LoRA 和 full fine-tuning 的局部 function-space tangent dimension。参数数目是否等于有效表达维度？
12. **DPS exact case。** 找出 $p(y\mid x_0)$ 和 $p(x_0\mid x_t)$ 的一组条件，使 $\mathbb E[p(y\mid x_0)] = p(y\mid\mathbb E[x_0])$ exact。除线性 $p(y\mid x_0)$ 或 degenerate posterior 外还有什么情形？
13. **Posterior calibration。** 对可解析 Gaussian-mixture inverse problem，同时比较 posterior mean、MAP、DDRM-like update 与 DPS samples。只看 PSNR 会隐藏哪些 distributional error？
14. **Hard consistency vs likelihood。** RePaint 的 hard replacement 可看成 $\sigma_y\to0$ 极限吗？在 mask boundary 和 learned prior 存在时，这个极限与完整 posterior conditional 是否交换？

***

## 25. 资料与实现入口

本章的来源职责、固定版本、公式页和 license 边界统一记录在 D7 chapter packet（补充材料暂未公开）。主要一手笔记：

- [classifier guidance](https://arxiv.org/abs/2105.05233 "官方论文页面") 与 [classifier-free guidance](https://arxiv.org/abs/2207.12598 "官方论文页面")；
- [Imagen dynamic thresholding](https://arxiv.org/abs/2205.11487 "官方论文页面")、[limited-interval guidance](https://arxiv.org/abs/2404.07724 "官方论文页面")、[CFG++](https://arxiv.org/abs/2406.08070 "官方论文页面") 与 [predictor-corrector view](https://arxiv.org/abs/2408.09000 "官方论文页面")；
- [Latent Diffusion cross-attention](https://arxiv.org/abs/2112.10752 "官方论文页面")、[ControlNet](https://arxiv.org/abs/2302.05543 "官方论文页面") 与 [IP-Adapter](https://arxiv.org/abs/2308.06721 "官方论文页面")；
- [SDEdit](https://arxiv.org/abs/2108.01073 "官方论文页面")、[Prompt-to-Prompt](https://arxiv.org/abs/2208.01626 "官方论文页面")、[Null-text inversion](https://arxiv.org/abs/2211.09794 "官方论文页面") 与 [InstructPix2Pix](https://arxiv.org/abs/2211.09800 "官方论文页面")；
- [Textual Inversion](https://arxiv.org/abs/2208.01618 "官方论文页面")、[DreamBooth](https://arxiv.org/abs/2208.12242 "官方论文页面") 与 [LoRA/T2I-Adapter/RePaint excerpts](https://arxiv.org/abs/2112.10741 "官方论文页面")；
- [DPS](https://arxiv.org/abs/2209.14687 "官方论文页面") 与 [DDRM](https://arxiv.org/abs/2201.11793 "官方论文页面")；
- 独立推导台账（补充材料暂未公开） 与 独立数值检查（补充材料暂未公开）。

官方实现均固定 commit；selected files 与许可边界见 Diffusion code provenance（补充材料暂未公开）。Sander Dieleman 的 guidance 教学笔记（补充材料暂未公开） 只承担几何直觉，不替代 primary papers 的历史或 theorem 职责。

上一章：[D6. 采样算法演进](/blog/diffusion/d6-sampling-solvers/)。下一章将进入 [D8. 架构与表征](/blog/diffusion/d8-architecture-representation/)，系统讨论 U-Net、latent bottleneck、DiT 与 scaling；本章只固定了 conditioning interface。
