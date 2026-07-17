---
title: 离散 Diffusion 与 Diffusion Language Models
description: 从有限状态转移矩阵和 CTMC 生成器进入 D3PM、SEDD、MDLM 与并行语言生成的概率接口。
publishedAt: '2026-07-17'
updatedAt: '2026-07-17'
draft: false
type: series-chapter
series: diffusion
order: 10
slug: d10-discrete-diffusion-language-models
tags:
  - diffusion
  - discrete-diffusion
  - language-models
authors:
  - chengjt23
license: all-rights-reserved
math: true
mermaid: false
toc: true
featured: false
includeInFeed: false
indexable: true
scope: 聚焦离散噪声、概率比 score、反向生成器和并行解码成本，不直接套用连续 Gaussian 直觉。
---
图像 diffusion 的直觉几乎都依赖连续空间：给像素加 Gaussian noise，学习 $\nabla_x\log p_t(x)$，再沿 reverse SDE 或 ODE 回到数据。但 token、类别标签、图的节点类型并没有天然的“加一个很小的高斯扰动”。把 `cat` 的编号 17 改成 17.01 也没有语言意义。

离散 diffusion 因而不是把连续公式中的积分换成求和。它需要重新回答三个问题：

1. **噪声是什么？** 由有限状态 Markov transition matrix 或 CTMC rate matrix 定义；
2. **score 是什么？** 不再天然是梯度，而是状态之间的 probability ratio；
3. **并行生成是什么？** 一次可以提出多个 token，但条件依赖、全序列计算和反复修订仍决定真实成本。

本章以 D3PM、continuous-time discrete diffusion、SEDD 和 MDLM 为理论主轴，再把 Diffusion-LM、DiffusionBERT、ARDM、LLaDA、Block Diffusion、Discrete Flow Matching 与 Generator Matching 放回同一张“问题—方案—局限”地图。重点不是罗列 diffusion LLM 产品，而是看清它们改变了哪个概率对象。

***

## 1. 先固定状态、向量和时间约定

设有限状态空间

$$
\mathcal X=\{1,2,\ldots,K\}.
$$

状态 $i$ 既可以写成整数，也可以写成 one-hot **行向量** $e_i\in\mathbb R^{1\times K}$。全文统一使用 row-vector convention：

$$
p_t=[p_t(1),\ldots,p_t(K)],
\qquad
p_{t+1}=p_tQ_{t+1}.
$$

因此：

- $Q_t(i,j)$ 表示从 $i$ 到 $j$ 的概率；
- 每一行求和为 1；
- 多步 kernel 按时间从左到右相乘；
- continuous-time generator 满足 $\partial_tp_t=p_tR_t$。

时间方向沿用 diffusion convention：

$$
x_0\sim p_{\mathrm{data}},
\qquad
x_T\approx p_{\mathrm{base}}.
$$

forward corruption 走 $0\to T$，生成走 $T\to0$。讨论 DFM 时会单独说明 data/base orientation，避免把符号方向偷换。

对长度 $L$ 的 token sequence，写作

$$
x_t=(x_t^1,\ldots,x_t^L)\in\mathcal X^L.
$$

forward process 常在给定 $x_0$ 后按位置独立 corruption；这不表示数据分布或 denoiser 按位置独立。Transformer 仍可以用整条 noisy sequence 预测每一个位置。

***

## 2. 一步 categorical corruption

给定 row-stochastic matrix $Q_t$，定义

$$
q(x_t\mid x_{t-1})
=\operatorname{Cat}(x_t;x_{t-1}Q_t).
$$

若 $x_{t-1}=e_i$，参数 $e_iQ_t$ 正是 $Q_t$ 的第 $i$ 行。把 categorical probability 展开：

$$
q(x_t=e_j\mid x_{t-1}=e_i)=Q_t(i,j).
$$

这一步已经揭示 continuous diffusion 与 discrete diffusion 的核心差别。连续 Gaussian kernel 主要由 mean/variance 决定；离散 kernel 的每一行都可以编码不同的替换结构。`noise schedule` 不再只是一个标量方差，它还包括**哪些状态允许互相跳转**。

有效的 $Q_t$ 必须满足

$$
Q_t(i,j)\ge0,
\qquad
\sum_{j=1}^KQ_t(i,j)=1.
$$

在 sequence 上最简单的 forward kernel 是

$$
q(x_t\mid x_{t-1})
=\prod_{\ell=1}^Lq(x_t^\ell\mid x_{t-1}^\ell),
$$

但 learned reverse model 可以写成

$$
p_\theta(x_{t-1}\mid x_t)
=\prod_{\ell=1}^L
p_\theta(x_{t-1}^\ell\mid x_t,t),
$$

其中每一项都读取完整 $x_t$。输出分布的 factorization 与网络表示的上下文范围是两件事。

***

## 3. 多步 marginal 为什么仍可精确计算

定义 cumulative transition

$$
\bar Q_t=Q_1Q_2\cdots Q_t,
\qquad
\bar Q_0=I.
$$

两步时，Chapman--Kolmogorov equation 给出

$$
\begin{aligned}
q(x_2\mid x_0)
&=\sum_{x_1}q(x_2\mid x_1)q(x_1\mid x_0)\\
&=\operatorname{Cat}(x_2;x_0Q_1Q_2).
\end{aligned}
$$

归纳得到

$$
\boxed{
q(x_t\mid x_0)
=\operatorname{Cat}(x_t;x_0\bar Q_t).
}
$$

这相当于 DDPM 的 closed-form $q(x_t\mid x_0)$：训练时无需逐步模拟 $x_1,\ldots,x_{t-1}$，可以直接从 $x_0\bar Q_t$ 采样 $x_t$。

注意矩阵顺序：

$$
Q_1Q_2\ne Q_2Q_1
$$

一般成立。只有 uniform 等特殊 commuting family 才能把顺序隐藏进单一 scalar schedule。

***

## 4. Uniform corruption：忘掉身份，但不标记哪里坏了

令

$$
U=\frac{\mathbf1\mathbf1^\top}{K},
\qquad U^2=U,
$$

并取

$$
Q_t=(1-\beta_t)I+\beta_tU
=\alpha_tI+(1-\alpha_t)U,
\qquad \alpha_t=1-\beta_t.
$$

直觉上，token 以 identity 分量保留，否则从 uniform distribution 抽样。因为 $U$ 是投影矩阵，两个 kernel 的乘积为

$$
[aI+(1-a)U][bI+(1-b)U]
=abI+(1-ab)U.
$$

所以

$$
\boxed{
\bar Q_t
=\bar\alpha_tI+(1-\bar\alpha_t)U,
\qquad
\bar\alpha_t=\prod_{s=1}^t\alpha_s.
}
$$

从 clean state $i$ 出发：

$$
q(x_t=j\mid x_0=i)
=\bar\alpha_t\mathbf1[j=i]
+\frac{1-\bar\alpha_t}{K}.
$$

这里有一个容易忽略的细节：uniform replacement 仍可能抽回原 token。因此 $(1-\bar\alpha_t)$ 是“进入 replacement 分量”的概率，不是“可见符号一定改变”的概率。

当 $\bar\alpha_T\to0$ 时，terminal law 趋近 uniform。它的优点是对称、closed form 简单；缺点是网络无法仅凭一个 noisy token 判断它是否被替换。

### 4.1 Multinomial Diffusion 与 D3PM 的关系

2021 年的两条代表性工作不能只保留 D3PM 这个名字。Hoogeboom et al. 的 *Argmax Flows and Multinomial Diffusion* 将 categorical data 上的 uniform transition、closed-form marginal 与 variational training 组织为 **Multinomial Diffusion**；Austin et al. 的 D3PM 则把离散 forward kernel 系统化为更一般的 transition-matrix family，并展开 structured/absorbing kernels、posterior parameterization、ELBO 与 auxiliary clean prediction。

因此，本节的 uniform corruption 可以看成 Multinomial Diffusion 的核心有限状态构造，也是 D3PM transition family 的一个特殊选择；后面的 absorbing 与 structure-aware kernels 展示 D3PM 更宽的设计空间。两篇工作都发表于 2021 年，本教程不在没有专项 priority 审计时声称其中一篇“首次定义了所有离散 diffusion”。来源职责和公式页见 [D10 离散 diffusion 笔记](https://arxiv.org/abs/2102.05379 "官方论文页面")。

***

## 5. Absorbing corruption：把未知位置显式变成 MASK

加入 absorbing state $m$，对 $i\ne m$ 定义

$$
Q_t(i,i)=\alpha_t,
\qquad
Q_t(i,m)=1-\alpha_t,
$$

并令

$$
Q_t(m,m)=1.
$$

一旦进入 mask 就不再离开。于是对 clean token $x_0\ne m$，

$$
\boxed{
q(x_t\mid x_0)
=\bar\alpha_t\delta_{x_0}
+(1-\bar\alpha_t)\delta_m.
}
$$

terminal law 是 $\delta_m$，不是 uniform。更重要的是：若观察到 $x_t\ne m$，就精确知道 $x_t=x_0$。不确定性只集中在 masked positions。

这解释了 masked diffusion 与 BERT-style denoising 的天然联系，也解释了它们的限制：mask 是一个人工信息标志。真实文本错误、替换和插入删除未必提供这样的 oracle 标志。

![Uniform、absorbing 与邻域 corruption 的 transition matrix 和 marginal path](/images/diffusion/d10_categorical_kernels.png)

图中三个 kernel 有相同的“逐步遗忘”目标，却保留完全不同的中间信息。forward process 不是无关紧要的预处理；它决定 reverse problem 的条件结构。

***

## 6. Structure-aware corruption：何时应该保留邻域

D3PM 还允许 discretized-Gaussian、nearest-neighbor 或 embedding-aware transition。一般可以写成

$$
Q_t=(1-\beta_t)I+\beta_tM,
$$

其中 $M$ 是某个 row-stochastic mixing matrix。

若类别本身有顺序，例如量化像素值，邻近状态的转移可能比 uniform replacement 更自然。若词表 embedding 提供语义邻域，也可以让 corruption 优先替换近邻 token。但这里必须区分：

1. **有限状态结构**：$M$ 的边和权重；
2. **网络表示结构**：Transformer embedding；
3. **数据语义结构**：哪些替换真正保持句法或语义。

前两者不自动等于第三者。embedding-neighbor kernel 还可能随预训练模型改变，矩阵乘积和归一化也可能昂贵。2026 年 uniform-state scaling 的结果说明：mask 的优势不是不可推翻的定理；但它也不证明任意 structure-aware kernel 都会更好。

***

## 7. D3PM posterior：离散 reverse target 的核心推导

训练 reverse model 需要

$$
q(x_{t-1}\mid x_t,x_0).
$$

Bayes rule 给出

$$
q(x_{t-1}\mid x_t,x_0)
=\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}
{q(x_t\mid x_0)}.
$$

把候选 $x_{t-1}$ 的概率同时写成向量：

- $x_tQ_t^\top$ 的第 $i$ 个分量是 $Q_t(i,x_t)$，即 candidate $i$ 生成 observed $x_t$ 的 likelihood；
- $x_0\bar Q_{t-1}$ 是 candidate $x_{t-1}$ 的 prior；
- 两者逐元素相乘得到 unnormalized posterior。

因此

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=\operatorname{Cat}\!\left(
x_{t-1};
\frac{(x_tQ_t^\top)\odot(x_0\bar Q_{t-1})}
{x_0\bar Q_tx_t^\top}
\right).
}
$$

分母确实是 normalization constant，因为

$$
\begin{aligned}
\sum_{x_{t-1}}
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)
&=x_0\bar Q_{t-1}Q_tx_t^\top\\
&=x_0\bar Q_tx_t^\top.
\end{aligned}
$$

这不是“把一个向量除以它的和”后碰巧得到的公式，而是同一 forward Markov chain 的 exact Bayes posterior。

***

## 8. 三个 reverse distribution 不能混为一谈

与 D2 一样，离散 diffusion 也有三层对象：

1. **conditioned teacher**

$$
q(x_{t-1}\mid x_t,x_0),
$$

训练时已知 clean $x_0$，所以可精确计算；

2. **true unconditional reverse**

$$
q(x_{t-1}\mid x_t)
=\sum_{x_0}q(x_{t-1}\mid x_t,x_0)q(x_0\mid x_t),
$$

需要未知 data posterior；

3. **learned reverse model**

$$
p_\theta(x_{t-1}\mid x_t),
$$

由网络逼近。

`posterior is tractable` 指第一层，不表示真实 unconditional reverse 已知。D3PM 的关键是：第一层给出一个可计算 teacher family，网络只需学习如何对未知 clean state 做正确混合。

***

## 9. Predict-$x_0$ parameterization：混合 exact posterior

令 denoiser 输出

$$
\tilde p_\theta(x_0\mid x_t,t).
$$

一种自然 reverse parameterization 是

$$
\boxed{
p_\theta(x_{t-1}\mid x_t)
=\sum_{\tilde x_0}
q(x_{t-1}\mid x_t,\tilde x_0)
\tilde p_\theta(\tilde x_0\mid x_t,t).
}
$$

它与 continuous diffusion 的 predict-$x_0$ 接口相似，但输出是 categorical clean distribution。网络不需要直接输出一个任意 reverse matrix；forward algebra 把 clean prediction 转成合法 reverse transition。

不要把上式实现成“先取 $\arg\max\tilde x_0$，再用一个 posterior”。那会丢失 clean uncertainty。也不要把 clean-token distribution 与 $x_{t-1}$ distribution 当作同一个对象：图中的两组柱状概率显然不同。

![D3PM exact posterior 与 clean-prediction reverse mixture](/images/diffusion/d10_d3pm_reverse.png)

***

## 10. D3PM path ELBO

forward path 和 generative path 分别是

$$
q(x_{1:T}\mid x_0)
=\prod_{t=1}^Tq(x_t\mid x_{t-1}),
$$

$$
p_\theta(x_{0:T})
=p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}\mid x_t).
$$

由 Jensen inequality，

$$
\begin{aligned}
-\log p_\theta(x_0)
&=-\log\sum_{x_{1:T}}p_\theta(x_{0:T})\\
&\le
\mathbb E_{q(x_{1:T}\mid x_0)}
\log\frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}.
\end{aligned}
$$

用 conditioned posterior 重排，得到

$$
\begin{aligned}
\mathcal L_{\mathrm{NELBO}}
= {}&D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))\\
&+\sum_{t=2}^T
\mathbb E_{q(x_t\mid x_0)}
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\|p_\theta(x_{t-1}\mid x_t)
\right)\\
&-\mathbb E_{q(x_1\mid x_0)}
\log p_\theta(x_0\mid x_1).
\end{aligned}
$$

三类项与 DDPM 完全平行：terminal prior matching、intermediate denoising KL、data reconstruction。区别在于所有 distribution 都是 categorical，KL 可以精确求和。

D3PM 还使用 auxiliary clean-token cross-entropy：

$$
\mathcal L_{\mathrm{aux}}
=-\mathbb E_{t,x_0,x_t}\log\tilde p_\theta(x_0\mid x_t,t).
$$

混合目标

$$
\mathcal L=\mathcal L_{\mathrm{NELBO}}+\lambda\mathcal L_{\mathrm{aux}}
$$

可以增强监督，却不再是原 ELBO 的纯代数改写。`likelihood objective` 与 `helpful denoising loss` 必须分开报告。

***

## 11. D3PM 训练与 ancestral sampling

训练骨架：

```text
sample clean sequence x0
sample t uniformly or from a chosen proposal
sample xt directly from Cat(x0 @ Q_bar[t])
predict clean-token distribution p_clean = denoiser(xt, t)
mix exact q(x_{t-1} | xt, candidate x0) using p_clean
evaluate ELBO term and optional clean-token cross-entropy
update theta
```

生成骨架：

```text
sample xT from the terminal/base distribution
for t = T, ..., 1:
    predict p_clean = denoiser(xt, t)
    construct p_theta(x_{t-1} | xt) by posterior mixing
    sample x_{t-1}
return x0
```

对 sequence，网络一次输出所有位置的 vocabulary logits。即使一次 reverse step 并行提出多个位置，整个 Markov chain 仍有 $T$ 个 sequential rounds。

***

## 12. 从 stochastic matrix 到 CTMC generator

若时间步长缩小，令

$$
Q_{t,h}=I+hR_t+o(h).
$$

rate matrix $R_t$ 满足

$$
R_t(x,y)\ge0\quad(x\ne y),
\qquad
R_t(x,x)=-\sum_{y\ne x}R_t(x,y).
$$

因此 row sum 为 0。对 row marginal：

$$
\begin{aligned}
p_{t+h}
&=p_tQ_{t,h}\\
&=p_t+hp_tR_t+o(h),
\end{aligned}
$$

从而

$$
\boxed{
\partial_tp_t=p_tR_t.
}
$$

若 $R$ 不随时间变化，finite transition 是

$$
P_{t\mid s}=e^{(t-s)R}.
$$

$I+hR$ 只是 Euler approximation；$h$ 太大时 diagonal 可能变负。generator 是 infinitesimal object，不是任意时间跨度的 transition matrix。

***

## 13. Reverse CTMC rate 的完整推导

在时刻 $t$，forward process 从 $y$ 跳到 $x$ 的 infinitesimal probability flux 是

$$
p_t(y)R_t(y,x)\,dt.
$$

time-reversed process 从 $x$ 跳回 $y$ 必须匹配同一 joint flux：

$$
p_t(x)\widetilde R_t(x,y)\,dt
=p_t(y)R_t(y,x)\,dt.
$$

因此对 $x\ne y$，

$$
\boxed{
\widetilde R_t(x,y)
=R_t(y,x)\frac{p_t(y)}{p_t(x)}.
}
$$

diagonal 由 zero-row-sum 决定：

$$
\widetilde R_t(x,x)
=-\sum_{y\ne x}\widetilde R_t(x,y).
$$

代回可验证

$$
p_t\widetilde R_t=-p_tR_t,
$$

这正是物理时间反向运行时的 marginal derivative。

严格边界：公式只在 $p_t(x)>0$ 的 conditioning state 上有意义。absorbing endpoint 或稀疏 support 附近必须使用 support-aware limit，不能直接除以零。

***

## 14. Probability ratio 如何由 clean denoiser 给出

设 finite transition

$$
P_t(z,x)=q(X_t=x\mid X_0=z).
$$

noisy marginal 为

$$
p_t(x)=\sum_zp_0(z)P_t(z,x).
$$

clean posterior 是

$$
p(z\mid X_t=x)
=\frac{p_0(z)P_t(z,x)}{p_t(x)}.
$$

于是

$$
\begin{aligned}
\mathbb E\!\left[
\frac{P_t(X_0,y)}{P_t(X_0,x)}
\middle|X_t=x
\right]
&=\sum_z
\frac{p_0(z)P_t(z,x)}{p_t(x)}
\frac{P_t(z,y)}{P_t(z,x)}\\
&=\frac{p_t(y)}{p_t(x)}.
\end{aligned}
$$

因此一个估计 $p(X_0\mid X_t=x)$ 的 denoiser 足以构造 reverse ratio。Campbell 等人的 continuous-time discrete framework 正是用这个 posterior expectation 参数化 reverse rate。

这与 D3PM predict-$x_0$ mixture 是同一思想的 continuous-time 表达：难点从“直接学所有 reverse transitions”转成“预测 clean conditional”。

***

## 15. Exact event simulation 与 tau-leaping

精确 CTMC simulation 在当前状态 $x$ 计算 total exit rate

$$
\lambda_t(x)=\sum_{y\ne x}R_t(x,y).
$$

若 rate 在局部近似常量，下一个等待时间服从 Exponential distribution，再按

$$
\Pr(y\mid x,\text{jump})
=\frac{R_t(x,y)}{\lambda_t(x)}
$$

选择 event type。一次 exact event 通常只改变一个 coordinate。

高维 sequence 若逐 event 模拟会很慢。tau-leaping 在区间 $[t-\tau,t]$ 内冻结 reverse rates，并对多个 event count 做近似采样，使多个位置可以一起改变。代价是：

- rate 在区间内其实会随时间和状态改变；
- 同一位置可能发生多次 event；
- 较大 $\tau$ 可能违反状态约束或产生明显 bias；
- step-size control 与 corrector 仍然重要。

![CTMC marginal、reverse rates 与 finite leap error](/images/diffusion/d10_ctmc_ratio.png)

图右展示固定 rate Euler steps 的一阶收敛。`一次网络调用更新很多位置` 是并行实现优势，不等于 exact CTMC 在一个大步内完成。

***

## 16. 离散空间的 score 为什么是 ratio

连续 score

$$
\nabla_x\log p_t(x)
$$

依赖 $x$ 的微分结构。有限 token set 没有 canonical gradient。若 forward graph 允许 $x\to y$，自然可识别的局部量是

$$
\boxed{
s_t(x,y)=\frac{p_t(y)}{p_t(x)}.
}
$$

它直接进入 reverse rate：

$$
\widetilde R_t(x,y)=R_t(y,x)s_t(x,y).
$$

这个 `concrete score` 有三个性质：

$$
s_t(x,y)>0,
\qquad
s_t(x,y)s_t(y,x)=1,
\qquad
s_t(x,z)=s_t(x,y)s_t(y,z),
$$

后两式在相关状态概率为正时成立。但神经网络的独立 ratio outputs 未必自动满足 cycle consistency。

不要把 $s_t(x,y)$ 写成“token ID 方向上的梯度”。若要得到 gradient limit，必须额外给出网格、邻域尺度和连续极限。

***

## 17. Score entropy 的 population optimum

对支持中的 directed edge $(x,y)$，令模型输出 $s_\theta(x,y)>0$。忽略与模型无关的常数，一个 edgewise score-entropy population risk 可写成

$$
\mathcal J(s)
=\sum_x\sum_{y\ne x}w_{xy}
\left[p(x)s(x,y)-p(y)\log s(x,y)\right],
$$

其中 $w_{xy}>0$。对单个 output 求导：

$$
\frac{\partial\mathcal J}{\partial s(x,y)}
=w_{xy}\left[p(x)-\frac{p(y)}{s(x,y)}\right].
$$

stationary point 为

$$
s^*(x,y)=\frac{p(y)}{p(x)}.
$$

二阶导

$$
\frac{\partial^2\mathcal J}{\partial s(x,y)^2}
=w_{xy}\frac{p(y)}{s(x,y)^2}>0
$$

保证正 support edge 上唯一最优。令 $r=p(y)/p(x)$，excess risk 为

$$
w_{xy}p(x)r
\left[
\frac{s}{r}-1-\log\frac{s}{r}
\right]\ge0.
$$

这就是 SEDD `score entropy` 的核心：构造一个 proper ratio objective，而不是对离散状态伪造微分。若 $w_{xy}=0$ 或数据 support 为零，该 edge 上 ratio 不可识别。

***

## 18. Denoising score entropy

训练时不知道 noisy marginal $p_t$，却知道 conditional transition $P_t(x_0,x)$。上一节的 denoiser identity 给出

$$
\frac{p_t(y)}{p_t(x)}
=\mathbb E\!\left[
\frac{P_t(X_0,y)}{P_t(X_0,x)}
\middle|X_t=x
\right].
$$

因此可以采样

$$
x_0\sim p_{\mathrm{data}},
\qquad
x_t\sim P_t(x_0,\cdot),
$$

再用 tractable conditional ratio 构造 denoising score-entropy estimator。它与 continuous DSM 的逻辑平行：conditional target 在 population projection 后恢复 marginal object；但目标是 probability ratio，不是 vector score。

SEDD 还把该 objective 与 continuous-time likelihood bound 相连。需要保留三个误差层：

1. ratio estimation error；
2. finite neighbor/output parameterization error；
3. reverse CTMC discretization/tau-leap error。

ratio objective 的 population consistency 不保证有限步 sampler 精确。

***

## 19. Masked diffusion 的 reverse posterior

回到 absorbing process。使用 continuous noise-time $t\in[0,1]$，令 $\alpha_t$ 从 1 单调下降到 0：

$$
q(z_t\mid x)
=\operatorname{Cat}\bigl(z_t;\alpha_tx+(1-\alpha_t)m\bigr).
$$

对 $0\le s<t\le1$：

- 若 $z_t=x\ne m$，则 $z_s=x$ 概率为 1；
- 若 $z_t=m$，则在 $s$ 时仍 mask 的 posterior probability 是

$$
q(z_s=m\mid z_t=m,x)
=\frac{1-\alpha_s}{1-\alpha_t},
$$

而已经恢复 clean token 的概率是

$$
q(z_s=x\mid z_t=m,x)
=\frac{\alpha_s-\alpha_t}{1-\alpha_t}.
$$

两项相加为 1。reverse process 只需决定 masked position 在何时 unmask，以及 unmask 成哪个 clean token。

***

## 20. SUBS：把 exact support 写进 parameterization

MDLM 的 substitution parameterization 使用两个 absorbing process 的 exact facts。

第一，clean token 不可能是 mask：

$$
[x_\theta(z_t,t)]_m=0.
$$

第二，已经 unmasked 的位置精确等于 clean token，因此直接 copy：

$$
x_\theta(z_t,t)=z_t
\qquad\text{when }z_t\ne m.
$$

只有 $z_t=m$ 时，网络才输出 vocabulary distribution。这两条约束合称 SUBS。它们有三项作用：

1. 排除 absorbing reverse process 不允许的输出；
2. 消除已知位置上的无意义 loss；
3. 简化 ELBO 中的 reverse transition。

但 SUBS 不假设 masked positions 在数据中独立。网络可以读取所有 visible/masked context；只是最终 categorical heads 通常 factorize。

***

## 21. 从 finite-step masked ELBO 到 continuous objective

考虑离散网格 $0=t_0<t_1<\cdots<t_T=1$。若位置在 $t_i$ 仍为 mask，向 $t_{i-1}$ 的 exact posterior 以

$$
\frac{\alpha_{t_{i-1}}-\alpha_{t_i}}
{1-\alpha_{t_i}}
$$

的概率恢复 clean token。SUBS 后，与模型有关的 finite-step term 化成 masked clean-token cross-entropy：

$$
-\frac{\alpha_{t_{i-1}}-\alpha_{t_i}}
{1-\alpha_{t_i}}
\log\langle x_\theta(z_{t_i},t_i),x\rangle.
$$

令网格变密，

$$
\alpha_{t_{i-1}}-\alpha_{t_i}
\approx-\alpha_t'\,dt.
$$

得到 MDLM continuous Rao--Blackwellized NELBO：

$$
\boxed{
\mathcal L_\infty
=\mathbb E_q\int_0^1
\frac{\alpha_t'}{1-\alpha_t}
\log\langle x_\theta(z_t,t),x\rangle\,dt.
}
$$

符号必须一起看：

$$
\alpha_t'\le0,
\qquad
\log\langle x_\theta,x\rangle\le0.
$$

所以 integrand 是非负 cost。也可定义

$$
w(t)=-\frac{\alpha_t'}{1-\alpha_t}\ge0
$$

并显式写成 weighted negative log probability。

***

## 22. Rao--Blackwellization 与 schedule change of variables

一个位置在时刻 $t$ 被 mask 的概率是

$$
1-\alpha_t.
$$

raw loss weight 却含

$$
w(t)=-\frac{\alpha_t'}{1-\alpha_t},
$$

在 data endpoint 附近可能发散。把 mask event 的 probability 一起取期望：

$$
(1-\alpha_t)w(t)=-\alpha_t'.
$$

令 $u=1-\alpha_t$，则

$$
w(t)dt=\frac{du}{u}.
$$

若 ideal conditional loss 不显式依赖具体 parameterization of time，

$$
\int_0^1(1-\alpha_t)w(t)\ell\,dt
=\ell\int_0^1-\alpha_t'dt
=\ell.
$$

这解释了 continuous objective 对 monotone schedule 的理想不变性。

Rao--Blackwellization 的含义是：把可解析的 mask/unmask transition 随机性积分掉，只留下 clean-token log probability。它降低 estimator variance，却不会让未知 data conditional 变得已知，也不保证 finite network optimization 更容易。

![Score entropy ratio optimum 与 MDLM schedule weighting](/images/diffusion/d10_score_entropy_mdlm.png)

图中 raw weight 的 endpoint singularity 被 mask probability 抵消。实际训练使用 finite timestep proposal、endpoint cutoff 和 time-conditioned network，因此仍可能有 schedule sensitivity。

***

## 23. Time-agnostic optimum 不等于时间输入永远无用

对 absorbing process，当前 noisy sequence 的 mask pattern 已包含剩余不确定性的关键信息。在无限 capacity、population optimum 下，clean conditional 可以写成

$$
p(x^i\mid z_t)
$$

而不必显式依赖连续 scalar $t$。2025 年 ICLR 工作进一步把 masked NELBO 与 order-agnostic objective 联系起来，并指出 categorical sampling/caching 是重要瓶颈。

但从 theorem 到实现还隔着四层：

- finite model 可能用 time embedding 补偿 capacity/optimization error；
- 训练 batch 的 global mask rate 可携带有用 calibration；
- sampling schedule 会改变 model query distribution；
- chain 近似和 remasking rule 会改变 observed pattern 的统计。

所以正确结论是“存在 time-agnostic population structure”，不是“删掉 time embedding 对所有训练和采样都无影响”。

***

## 24. 四种 sequence factorization

### 24.1 Left-to-right autoregression

$$
p(x^{1:L})
=\prod_{i=1}^Lp(x^i\mid x^{<i}).
$$

它固定顺序，训练可并行 teacher forcing，生成必须沿 sequence order 逐 token 前进。KV cache 复用历史 hidden states。

### 24.2 任意顺序 chain rule

对任意 permutation $\pi$：

$$
p(x^{1:L})
=\prod_{j=1}^L
p(x^{\pi_j}\mid x^{\pi_{<j}}).
$$

exact distribution 下所有 order 的乘积相同；approximate neural conditionals 下，不同 order 的误差、variance 与 sample quality 可以不同。

### 24.3 Masked diffusion

mask set $M_t$ 随时间缩小，模型学习

$$
p_\theta(x^{M_t}\mid x^{\bar M_t},M_t,t).
$$

一次可以提出多个 masked tokens，但它们通常在 output head 中 conditionally factorized；下一轮用更新后的 context 修正。

### 24.4 Block diffusion

把 sequence 划成 blocks $B_1,\ldots,B_M$：

$$
p(x)=\prod_{b=1}^M
p(x_{B_b}\mid x_{B_{<b}}).
$$

block 之间 autoregressive，block 内用 diffusion。block size 为 1 时回到 token AR；单个 full-length block 接近 full masked diffusion。

![AR、masked、block factorization，技术演进与系统成本轴](/images/diffusion/d10_factorization_history.png)

上图右侧是概念坐标，不是 benchmark。它强调 sequential rounds 与每条 sequence 的总工作量是两个轴。

***

## 25. ARDM：diffusion 与 autoregressive 的早期桥梁

Autoregressive Diffusion Models 使用逐步消失/恢复变量的 corruption，并对 generation order 取随机 permutation。其 objective 可整理成 order-agnostic autoregressive loss：

$$
\mathcal L_{\mathrm{OA}}
=\mathbb E_\pi
\sum_{j=1}^L
-\log p_\theta
(x^{\pi_j}\mid x^{\pi_{<j}}).
$$

这说明 absorbing diffusion 并不是与 autoregression 完全割裂的概率族。它可以看成对 reveal order 和中间 mask state 的重新组织。

ARDM 还研究 depth upscaling：不同变量在不同 stage 被恢复，以允许 coarse-to-fine 或结构化顺序。保留的局限是：任意顺序训练不自动给出最优并行 schedule，模型还要处理不同 observed sets 的 combinatorial variety。

***

## 26. Diffusion-LM：先在 embedding space 中连续扩散

Diffusion-LM 走的是另一条路线。令 token embedding 为

$$
e(x)\in\mathbb R^d,

$$

在 continuous embedding sequence 上做 Gaussian diffusion，再把 denoised vectors round 回 vocabulary。

它的优势是可在 continuous space 施加 gradient-based control，例如句法、情感或 semantic constraint。困难包括：

1. embedding vector 与合法 token 之间有 rounding gap；
2. Euclidean distance 未必对应语言替换代价；
3. long Gaussian reverse chain 仍有高 NFE；
4. surrogate denoising objective 与 discrete text likelihood 需要仔细区分。

因此 Diffusion-LM 是 diffusion language modeling 的重要历史节点，但不是 D3PM/MDLM 的 finite-state special case。两者的 state space 和 forward law 不同。

***

## 27. DiffusionBERT 与 PLAiD：预训练和 likelihood 两条修正

DiffusionBERT 观察到 BERT 本来就是 masked denoiser，于是用 pretrained bidirectional Transformer 初始化 absorbing discrete diffusion。它提出 spindle schedule，让 token 的 corruption 时间与 informativeness 联系起来。

这解决的是 representation/optimization 问题：利用已有 MLM denoising knowledge。它不意味着一次 BERT forward pass 就是 exact reverse diffusion。

Likelihood-Based Diffusion Language Models（PLAiD）则回到 likelihood-first 设计，系统加入：

- categorical output parameterization；
- learned embedding/conditional likelihood；
- output prior；
- self-conditioning；
- sequence-length handling。

它的教训是：当 text evaluation 依赖 likelihood/perplexity 时，rounding 和 surrogate loss 不能被 sample quality 掩盖。`continuous latent diffusion` 仍需要定义最终 discrete observation model。

***

## 28. MDLM：为什么简单 masked likelihood 成为主线

MDLM 将 absorbing kernel 的结构用到底：

- 只预测 masked positions；
- unmasked positions carry over；
- mask 不属于 clean vocabulary；
- 用 continuous Rao--Blackwellized NELBO 训练；
- Transformer 读取双向上下文。

相较一般 D3PM，矩阵 algebra 极度简化；相较 generic MLM，模型有明确的 generative Markov chain 和 likelihood bound。这种“理论简单 + 系统可扩展”的组合，是 masked diffusion 在 2024--2025 年成为语言主线的重要原因。

但每个 denoising round 通常仍要处理完整 sequence，且 confidence-based unmasking、remasking、temperature 和 step schedule 会显著影响实际样本。

***

## 29. LLaDA：大规模 system evidence 的正确位置

LLaDA 将随机 mask pretraining、bidirectional Transformer、iterative generation 和 instruction fine-tuning 扩展到大参数规模。其 generative loop 可以概括为：

```text
start from a fully masked response region
for each denoising round:
    predict token distributions at all currently masked positions
    select positions to reveal using a schedule/confidence rule
    sample or choose their tokens
    optionally revise/remask according to the sampler
return the completed sequence
```

LLaDA 证明了 masked factorization 可以承载 large language model training 和 instruction following 的系统可能性。但在本章的证据层级中，它仍是 influential preprint/system：

- 不承担 D3PM、SEDD 或 MDLM theorem 的来源；
- benchmark 不能证明所有 reasoning task 与 AR 等价；
- 可修订 token 不表示每轮计算免费；
- 真实 latency 依赖 generation length、rounds、full-sequence FLOPs 和 kernel implementation。

***

## 30. Block Diffusion：把 length、cache 和 parallelism 放进 factorization

Block Diffusion 令

$$
x=(x^{(1)},\ldots,x^{(B)}),

$$

并建模

$$
p(x)=\prod_{b=1}^B
p_\theta(x^{(b)}\mid x^{(<b)}).
$$

每个 block 内执行 masked diffusion。这样做不是单纯换 sampler，而是改变 joint distribution factorization：

| 小 block                          | 大 block                            |
| -------------------------------- | ---------------------------------- |
| 更多 sequential blocks             | 更多 within-block parallel proposals |
| 更接近 token AR likelihood          | 更接近 full diffusion                 |
| KV cache 更容易复用                   | bidirectional recomputation 更多     |
| continuation/arbitrary length 自然 | fixed block/padding 设计更重要          |
| block conditional 较简单            | 单次条件建模更难                           |

Block size 暴露出 AR 与 diffusion 并非二选一。任何“谁更快”的结论都必须说明 block size、cache policy 和生成长度。

***

## 31. Likelihood、perplexity 与 arbitrary length

语言模型论文中的 `likelihood` 至少有四种含义：

1. exact AR log likelihood；
2. latent diffusion ELBO/NELBO；
3. importance-sampled 或更紧的 likelihood estimator；
4. masked-token cross-entropy proxy。

perplexity 通常是

$$
\operatorname{PPL}
=\exp\left(
-\frac{1}{N_{\mathrm{token}}}
\log p(x)
\right),
$$

但若 $\log p(x)$ 被一个 loose bound 代替，数值就不再与 exact AR PPL 同义。比较时必须匹配：

- tokenizer 和 vocabulary；
- BOS/EOS 与 length model；
- context/window length；
- likelihood bound/estimator；
- token normalization；
- validation preprocessing。

AR 可以持续 append token，天然定义 arbitrary length。full-sequence diffusion 常先固定 output slots；需要 EOS、length predictor、padding 或 block factorization 才能扩展长度。`bidirectional` 本身不解决 length distribution。

***

## 32. 为什么 NFE 不是 latency

NFE 只计算网络被调用多少次。对语言系统，还要至少记录：

$$
\text{cost}
=f(
\text{NFE},L,\text{batch},\text{cache},
\text{architecture},\text{hardware},\text{sampler}
).
$$

AR 的一步通常只处理一个新 query token，并复用 KV cache；full masked diffusion 的一步可能重新处理整个 sequence。于是：

- diffusion sequential rounds 更少，却可能 FLOPs 更多；
- AR rounds 更多，却可能单步极便宜；
- large batch/hardware parallelism 可能反过来有利于 diffusion；
- confidence/remasking 会使 `每轮确定多少 token` 不稳定；
- block diffusion 和 cache-aware diffusion 位于两者之间。

公平报告至少要给出 latency、throughput、memory、NFE、total processed tokens/FLOPs、batch size、prompt/generated length、hardware、precision 和 matched quality point。

***

## 33. 2025 理论：能否证明 parallel acceleration

当前理论应分成三个不同问题。

### 33.1 模型表达与任务结构

2025 年理论预印本构造了 diffusion/bidirectional update 有利或受限的任务。结论依赖 stylized distribution、metric、network/execution assumptions，不能直接给现有 LLM 排名。

### 33.2 采样收敛与依赖强度

NeurIPS 2025 的工作在明确 prediction error、token dependence 和 sampling schedule 假设下，给出 masked diffusion sampling error 随 iterations 衰减的 bound，并讨论与 AR sequential bottleneck 的差异。

它支持的表述是：

> 在受限但明确的条件下，parallel diffusion sampling 可以获得可证明的加速。

它不支持：

> 任意自然语言分布、任意模型误差和任意硬件上，diffusion 总比 AR 快。

### 33.3 系统效率审计

2025 efficiency critique 指出大量比较没有同时控制 batch、length、NFE、cache、hardware 和 decoding protocol。理论 sequential depth 与实测 wall-clock 是互补证据，不能互相替代。

***

## 34. Scaling 与 2026 frontier

近期工作把一个早期经验判断重新打开：absorbing mask 是否真是唯一可扩展的 discrete corruption？

`Scaling Behavior of Discrete Diffusion Language Models` 比较 model/data/compute 与 noise family，提示 objective、noise type 和 optimal capacity 相互作用。`Scaling Beyond Masked Diffusion` 报告经过 objective/architecture/scaling redesign 后，uniform-state diffusion 仍可竞争。

同时出现两条 ICLR 2026 分支：

- **Soft-Masked DLM**：不只做 hard mask/unmask decision，而把 model uncertainty feedback 给后续步骤；
- **FS-DFM**：用 discrete flow 与 few-step solver 减少 sequential rounds。

这些结果应放在 stable D3PM--CTMC--SEDD--MDLM 理论之后。新的 benchmark gain 可以否定“mask 必然唯一”的强经验信念，却还不足以宣布旧理论失效或新分支已经成为长期标准。

***

## 35. Discrete Flow Matching：velocity 是 probability flux

沿用 D9 的生成方向：$\tau=0$ 是 base，$\tau=1$ 是 data。给定 target datum $z$，考虑 conditional path

$$
p_\tau(x\mid z)
=(1-\kappa_\tau)p_{\mathrm{base}}(x)
+\kappa_\tau\delta_z(x),
$$

其中 $\kappa_0=0,\kappa_1=1$。定义

$$
c_\tau=\frac{\dot\kappa_\tau}{1-\kappa_\tau}.
$$

一个实现该 path 的 conditional jump generator 是：对 $x\ne z$，

$$
R_\tau^z(x,z)=c_\tau,
\qquad
R_\tau^z(x,x)=-c_\tau,
$$

而 $z$ 无 outgoing jump。对 $x\ne z$：

$$
(p_\tau R_\tau^z)(x)
=-c_\tau(1-\kappa_\tau)p_{\mathrm{base}}(x)
=-\dot\kappa_\tau p_{\mathrm{base}}(x).
$$

在 $z$ 处由 conservation 得

$$
(p_\tau R_\tau^z)(z)
=\dot\kappa_\tau[1-p_{\mathrm{base}}(z)].
$$

所以

$$
\boxed{
\partial_\tau p_\tau
=p_\tau R_\tau^z
=\dot\kappa_\tau
(\delta_z-p_{\mathrm{base}}).
}
$$

这就是离散 probability velocity 的具体例子：它是满足 conservation 和 nonnegative off-diagonal constraints 的 jump flux，不是 token ID 的几何位移。

训练时从 tractable conditional generator 出发，再用 posterior

$$
p(z\mid X_\tau=x)
$$

做 conditional-to-marginal projection。普通 DFM 仍学习 local generator；few-step generation 还要处理 finite transition/discretization，不能只把 $\tau$ 步长放大。

***

## 36. Generator Matching：统一的是 KFE，不是 path law

对 Markov process generator $L_t$ 和 test function $f$，Kolmogorov forward equation 的 weak form是

$$
\boxed{
\frac{d}{dt}\langle p_t,f\rangle
=\langle p_t,L_tf\rangle.
}
$$

不同状态空间对应不同 generator：

| Markov class        | generator on $f$                                     |
| ------------------- | ---------------------------------------------------- |
| ODE flow            | $L_tf=v_t^\top\nabla f$                              |
| diffusion           | drift first derivative + diffusion second derivative |
| finite jump process | $L_tf(x)=\sum_yR_t(x,y)[f(y)-f(x)]$                  |

若 conditional path $p_t(\cdot\mid z)$ 由 $L_t^z$ 生成，定义

$$
L_tf(x)=\mathbb E[L_t^zf(x)\mid X_t=x].
$$

则

$$
\begin{aligned}
\langle p_t,L_tf\rangle
&=\mathbb E[L_t^Zf(X_t)]\\
&=\mathbb E_Z
\langle p_t(\cdot\mid Z),L_t^Zf\rangle\\
&=\frac{d}{dt}\langle p_t,f\rangle.
\end{aligned}
$$

这把 continuous Flow Matching、diffusion 与 discrete jumps 放进 conditional Generator Matching 原理。

但 KFE 只约束 one-time marginals。可以存在多个 generator 生成同一 $(p_t)$：

$$
\text{same marginal path}
\not\Rightarrow
\text{same transition kernel or path law}.
$$

Generator Matching 统一的是训练/投影接口，不是 likelihood estimator、finite solver、trajectory correlation 或硬件成本。

它也不自动成为 Schrödinger Bridge。离散 SB 还要给定 reference path law，以 endpoint marginals 为 hard constraints，并在 path space 上最小化 relative entropy。一个满足 KFE 的 jump generator 或 conditional projection 本身没有完成这项 variational problem；相关 path-space KL、Doob transform 与 reciprocal-process 理论留给 Bridge 篇。

***

## 37. 图与蛋白：离散应用额外增加了什么

### 37.1 DiGress

图包含 node types、edge types 和 permutation symmetry。DiGress 分别对节点和边使用 categorical transition，并用 D3PM posterior 构造 reverse model。

新的难点不是“把词表换成原子类型”这么简单：

- absent edge 本身也是类别；
- 同一图有多个 node permutations；
- 网络必须 permutation equivariant；
- valence、connectivity 等 global validity 不是独立 categorical noise 自动保证的；
- empirical marginal transition 可改善 frequency matching，却不保证 topology。

### 37.2 DPLM

protein sequence 也是有限 amino-acid vocabulary。DPLM 使用 masked discrete diffusion 做 generative pretraining，并把 bidirectional representation 用于 generation 和 understanding tasks。

它说明同一 probabilistic machinery 可以越过自然语言。但 sequence likelihood、predicted structure、functional activity 和 wet-lab validation 是不同证据层。更完整的科学应用评价见 [D11](/blog/diffusion/d11-representative-applications/)。

***

## 38. 说明代码与 official implementation map

d10\_discrete\_diffusion\_language.py（补充材料暂未公开） 不训练网络。它实现：

1. uniform、absorbing、ring-neighbor kernels 和 cumulative marginals；
2. exact D3PM posterior 与 predict-clean reverse mixture；
3. finite-state CTMC matrix exponential、reverse generator 和 denoiser-ratio identity；
4. score-entropy edge risk；
5. MDLM schedule/change-of-variable check；
6. exact enumerable AR、arbitrary-order 和 block factorizations；
7. 五张章节图。

运行：

```powershell
# 本地验证脚本暂未公开
# 本地验证脚本暂未公开
```

核心检查：

| identity                       |           error/value |
| ------------------------------ | --------------------: |
| uniform closed form            | $5.551\times10^{-17}$ |
| absorbing closed form          |                   $0$ |
| D3PM posterior Bayes identity  |                   $0$ |
| reverse marginal derivative    | $1.388\times10^{-17}$ |
| denoiser-ratio identity        | $2.220\times10^{-16}$ |
| score-entropy optimum gradient |                   $0$ |
| MDLM schedule integral         | $1.028\times10^{-11}$ |
| arbitrary-order factorization  | $5.551\times10^{-17}$ |

official code snapshot 的职责不是相同的：

| source family                    | code responsibility                                     |
| -------------------------------- | ------------------------------------------------------- |
| Google D3PM/ARDM                 | matrix posterior、objective、order-agnostic transitions   |
| tauLDR                           | CTMC rates 与 tau-leaping                                |
| SEDD                             | score-entropy loss 与 reverse sampler                    |
| MDLM                             | SUBS、noise schedule、continuous masked loss              |
| Diffusion-LM/DiffusionBERT/PLAiD | embedding/MLM/likelihood language routes                |
| LLaDA/Block/Scaling              | large-system sampling、block factorization、noise scaling |
| DiGress/DPLM                     | graph/protein domain mapping                            |

完整 fixed commits 和 license boundary 见 D10 Chapter Packet（补充材料暂未公开）。无 root license、CC BY-NC 和 Apple software license 快照只承担相应许可范围内的本地研究核对。

***

## 39. 理论、算法与系统误差分层

| 层                          | 代表误差                                                    |
| -------------------------- | ------------------------------------------------------- |
| forward design             | terminal mismatch、corruption geometry、support           |
| posterior/ratio estimation | clean conditional、concrete score、calibration            |
| objective                  | ELBO looseness、auxiliary weighting、Monte Carlo variance |
| CTMC simulation            | rate freezing、tau-leap、endpoint truncation              |
| factorization              | order/block approximation、length model                  |
| architecture               | bidirectional representation、capacity、time conditioning |
| system                     | full-sequence FLOPs、KV cache、batch/hardware utilization |
| evaluation                 | unmatched tokenizer、bound、NFE、latency 或 quality         |

一次更低的 PPL 或更快的 demo 可能同时改变多层。没有 matched ablation 时，不能把系统收益全部归因于“diffusion factorization”。

***

## 40. 常见错误

1. **把 token ID 当连续坐标。** 编号差没有语义。
2. **矩阵乘法顺序写反。** row convention 下是 $Q_1\cdots Q_t$。
3. **把 uniform replacement probability 当 visible-change probability。** replacement 可抽回原 token。
4. **认为 uniform 与 absorbing terminal 相同。** 一个趋近 uniform，一个趋近 mask point mass。
5. **把 structured kernel 的 embedding 邻域当真实语义。** 两者需独立验证。
6. **把 conditioned posterior 当 true reverse。** 前者知道 $x_0$，后者不知道。
7. **对 predict-clean distribution 先 argmax。** 会丢失 reverse mixture uncertainty。
8. **把 auxiliary CE 称为 exact ELBO。** 加权后 objective 已改变。
9. **把 generator 当 finite transition matrix。** finite transition 需要 exponential/time ordering 或 solver。
10. **reverse rate 忘记转置方向。** $R_t(y,x)$ 对应 forward incoming edge。
11. **在零概率状态直接计算 ratio。** support condition 不能省略。
12. **把 concrete score 叫 gradient。** 它是 edge ratio。
13. **只证明 score-entropy optimum 就宣称 sampler exact。** 还存在估计和 discretization error。
14. **把 tau-leaping 称为 exact parallel CTMC。** 它冻结 rates，是近似。
15. **让 SUBS 在 unmasked token 上继续随机预测。** exact posterior 要求 copy。
16. **忽略 MDLM objective 的双负号。** $\alpha_t'$ 和 log probability 都非正。
17. **看到 raw weight 发散就断言 loss 发散。** mask probability 会抵消；endpoint numerical handling 仍需检查。
18. **把 time-agnostic optimum 推广到所有 finite models。** capacity、schedule 和 query shift 会改变结论。
19. **把 order-agnostic chain-rule equality 当 neural models 等价。** approximation error 与 training variance 不同。
20. **把一次并行 token proposal 当独立正确生成。** token predictions 仍通过 context 相互依赖。
21. **用 NFE 代替 latency。** AR cache 与 full-sequence work 完全不同。
22. **比较 PPL 却不说明是 exact likelihood 还是 bound。** 数值不可直接对齐。
23. **认为 bidirectional model 自动支持 arbitrary length。** 仍需 EOS/length/block design。
24. **把 LLaDA system result 当基础 theorem。** 证据职责不同。
25. **把 restricted acceleration theorem 写成 universal speedup。** assumptions 必须保留。
26. **把 DFM probability velocity 当 Euclidean token velocity。** 它是 conservative jump flux。
27. **认为 Generator Matching 唯一确定 process。** KFE marginal solution 不唯一。
28. **把图 validity 或蛋白 function 归结为 likelihood。** domain validation 是额外证据层。

***

## 41. 问题—方案—局限—下一步

| 问题                                 | 方案                     | 保留局限                           | 引出的下一问                        |
| ---------------------------------- | ---------------------- | ------------------------------ | ----------------------------- |
| Gaussian noise 不适合类别               | categorical transition | kernel 如何选                     | 能否利用结构                        |
| 需要 tractable reverse teacher       | D3PM posterior         | true reverse 仍未知               | 学 clean conditional           |
| 多步离散链难分析                           | CTMC generator         | exact events 串行                | tau-leap/ratio                |
| marginal ratio 不可见                 | denoiser identity/SEDD | support 与采样误差                  | 更简单 masked objective          |
| 一般 D3PM 目标复杂                       | SUBS/MDLM              | full-sequence repeated compute | factorization 如何折中            |
| 固定 left-to-right 太串行               | ARDM/masked diffusion  | token interaction 仍需 rounds    | block/parallel theory         |
| continuous text diffusion rounding | Diffusion-LM/PLAiD     | embedding/likelihood gap       | 直接 finite-state scaling       |
| masked model 扩到 LLM                | LLaDA                  | protocol 与 system cost         | cache/length/block            |
| AR 与 diffusion 二选一                 | Block Diffusion        | block-size tradeoff            | 如何选 operating point           |
| reverse corruption 不是唯一路径          | DFM/Generator Matching | finite solver/path law         | 更一般 discrete transport/Bridge |

***

## 42. 本章小结

1. 离散 diffusion 的 forward noise 是 Markov transition，不是 token ID 上的 Gaussian perturbation。
2. row-vector convention 下 $q(x_t\mid x_0)=\operatorname{Cat}(x_0\bar Q_t)$。
3. Uniform、absorbing 和 structure-aware kernels 有不同 terminal law 与信息结构。
4. D3PM posterior 来自 exact Bayes rule，分子是 likelihood 与 prior 的 Hadamard product。
5. Conditioned posterior、true reverse 和 learned reverse 是三个对象。
6. Predict-$x_0$ reverse model 应混合 exact posteriors，而不是先做 argmax。
7. Categorical path ELBO 仍分 terminal、intermediate KL 和 reconstruction。
8. CTMC generator 满足 $\partial_tp_t=p_tR_t$，finite kernel 需积分/exponential。
9. Reverse rate 是 transposed forward rate 乘 marginal probability ratio。
10. Clean denoiser 的 posterior expectation 可以恢复该 ratio。
11. Exact CTMC 逐 event；tau-leaping 用 parallelism 换取 finite-step approximation。
12. Concrete score 是 $p_t(y)/p_t(x)$，不是 gradient。
13. Score entropy 在正 support/正 edge weight 下以真实 ratio 为唯一 population optimum。
14. Denoising score entropy 用 tractable conditional ratio 学 marginal ratio。
15. Absorbing posterior 使 masked reverse process 只需决定 unmask time 与 clean token。
16. SUBS 强制 zero-mask output 和 carry-over unmasked tokens。
17. MDLM continuous objective 是正权重 cross-entropy；Rao--Blackwellization 降低 transition variance。
18. 理想 continuous objective 可通过 $u=1-\alpha_t$ 消除 monotone schedule parameterization。
19. AR、order-agnostic、masked 和 block 是相关但不同的 sequence factorizations。
20. Diffusion-LM、DiffusionBERT、PLAiD、MDLM、LLaDA 解决的是 state space、pretraining、likelihood 和 scaling 的不同问题。
21. Block size 同时改变 sequential depth、cache、length 与 conditional difficulty。
22. NFE、FLOPs、latency、throughput 和 memory 必须分开。
23. Parallel acceleration 已有受限理论结果，但不是无条件 universal guarantee。
24. 2026 uniform/soft/few-step branches是 frontier evidence，不替代基础理论。
25. Discrete Flow Matching 的 velocity 是 probability flux/generator。
26. Generator Matching 用 KFE 统一 Markov classes；same marginals 不等于 same path law。
27. 图和蛋白应用还需要 symmetry、validity、structure 和 function evidence。

***

## 43. 研究式思考题

1. 给定 vocabulary graph，什么条件保证 local edge ratios 唯一恢复 global distribution？若 graph 不连通会发生什么？
2. Uniform kernel 中 replacement event 与 visible change event 的差异会如何影响 corruption-rate calibration？
3. 对 noncommuting $Q_t$，能否设计既保留 structure 又有 closed-form $\bar Q_t$ 的 matrix family？
4. D3PM predict-clean mixture 与直接 parameterize reverse logits 的 model-class capacity 是否相同？在哪些 support 下不同？
5. Auxiliary clean-token CE 如何改变 ELBO 各 timestep 的 effective weighting？能否写成 importance proposal 的改变？
6. 当 $p_t(x)$ 很小时，reverse ratio 会爆炸。怎样用 clipping、reparameterization 或 detailed balance 控制 variance 而不破坏 consistency？
7. Tau-leaping 同时更新多个 sequence positions 时，如何量化 shared Transformer context 导致的 rate-freezing error？
8. Score-entropy outputs 若不满足 reciprocal/cycle consistency，是否可以通过 graph potential parameterization 强制 $s(x,y)=e^{g(y)-g(x)}$？代价是什么？
9. Denoising score entropy 与 clean-posterior cross-entropy 在什么 forward kernels 下有相同 population minimizer？finite-sample variance 如何比较？
10. MDLM schedule invariance 在 finite timestep sampling 下如何破坏？能否构造 variance-optimal time proposal？
11. Time-agnostic optimum 与 mask-count conditioning 的关系是什么？sequence-level random mask rate 是否引入额外 latent variable？
12. Order-agnostic objective 对所有 permutations 平均，但 sampling 只走少数 orders。如何做 train--sample order matching？
13. Block size 是否可以按 context uncertainty 自适应，而不是预先固定？这会如何定义 normalized likelihood？
14. 比较 AR 与 diffusion latency 时，怎样建立同时控制 quality、batch、length、cache 和 hardware 的实验协议？
15. 受限 parallel-acceleration theorem 中最难在真实语言验证的 assumption 是什么？可以用哪些 empirical proxy 检查？
16. Uniform-state scaling 改善来自 corruption、objective、architecture 还是 compute allocation？怎样设计因果 ablation？
17. DFM 中同一 marginal path 可以由多个 jump generators 实现。哪一个最小化 event count、variance 或硬件成本？
18. Generator Matching 的 conditional-to-marginal projection 与 Schrödinger Bridge 的 path-space KL projection 有何本质差别？
19. 对 graph diffusion，如何把 permutation equivariance、valence constraints 和 exact likelihood 同时放进一个 generator？
20. 对 protein diffusion，sequence likelihood 与 structural/function validity 之间应建立怎样的多层评价和不确定性报告？

***

下一章进入[代表性应用](/blog/diffusion/d11-representative-applications/)：重点不再是枚举“diffusion 可以做什么”，而是分析 video、audio、3D、science、planning 和 time series 各自增加了什么状态空间、条件、约束与评价难题，以及这些难题如何反过来改变算法。
