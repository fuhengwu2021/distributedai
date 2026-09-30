# 第1章：现代分布式 AI 简介 {-}

*从单个GPU到分布式集群构建可扩展的AI系统*

> 分布式系统是指即使一个你根本不知道存在的计算机出故障，也能导致你自己的计算机无法使用的系统。
- Leslie Lamport, 1987

**Code Summary**

- `torch.distributed`: PyTorch 分布式计算的核心模块（涵盖分布式训练与高性能推理）
- `dist.init_process_group()`: 初始化分布式进程组，配置通信后端（NCCL、GLOO）
- `dist.all_reduce()`: 在所有 rank 间执行全局数学规约（求和、极值等），并将规约结果写回所有 rank
- `dist.all_gather()`: 从所有 rank 收集局部张量切片，沿指定维度拼接后广播回所有 rank
- `dist.broadcast()`: 将根节点（src rank）上的张量单向广播复制到所有其他 rank
- `dist.reduce()`: 将所有 rank 的张量规约汇总至指定的单个目标 rank（通常为 rank 0）
- `dist.gather()`: 从所有 rank 收集张量切片并按序汇总至单个目标 rank
- `dist.scatter()`: 将根节点的张量切片列表按 rank 索引定向分发给各个 rank
- `dist.reduce_scatter()`: 在所有 rank 间执行全局规约，并将规约结果切片分发给对应的各个 rank
- `dist.all_to_all()`: 全对全数据置换原语，每个 rank 向其他所有 rank 定向发送并接收专属的数据切片


## 概述

现代AI模型已经超出了单个GPU能处理的范围。大型语言模型现在拥有从数十亿到超过一万亿的参数。在单个GPU上训练具有数十亿参数的模型需要数月时间，即使模型能装入内存。大规模服务这些模型需要分布式架构。

本章介绍资源估计、在分布式训练、微调或推理之间进行选择的决策框架，以及帮助你入门的实践示例。

![全书技术路线图：分布式 AI 系统工程路线图](img/fig_chapter_roadmap_zh.png){#fig:chapter-roadmap width=95%}

## 为什么现代AI需要分布式

![模型参数与年份](img/model_comparison_table_zh.png){#fig:model-comparison .block width=100%}

几年前，你可以在单个GPU上训练大多数模型。ResNet-50在ImageNet上需要几天。今天，在单个GPU上训练70B参数的语言模型需要数月时间，即使模型能装入内存。模型变得更大了，数据集变得更大了，单GPU训练变得不切实际。

如@fig:model-comparison所示，近年来模型参数的指数级增长是显而易见的。查看@tbl:model-comparison中详细的最新模型，规模很清楚[^model_size_comp]。GPT-4拥有超过1万亿参数，前沿模型继续突破这个规模[^llm_param_lie]。训练它们需要数千个GPU一起工作[^gpt4_training]。即使是较小的模型如Llama 2（70B参数）也需要多个GPU才能装入内存，更不用说有效地训练了。

这不仅是一个训练问题——在生产工作负载中大规模服务这些模型需要能处理数千并发请求的分布式推理架构。

单机AI的时代已经结束；现代AI系统在本质上是为分布式而设计的。根据PyTorch分布式训练文档，分布式训练是将训练工作负载分散到多个工作节点上，这对于大型模型和深度学习中的计算密集型任务尤为有益。此外，行业报告表明，训练万亿参数模型需要数千万美元的基础设施投资[^training_costs]，这使得分布式计算不仅是技术上的必然选择，也是现代AI开发在经济上的必然要求。

::: {width=80%}

| 模型 | 参数 | 公司 | 年份 |
|--|----------------------|-|-|
| ViT-22B | 22B | Google | 2023 |
| Grok-1 | 314B | xAI | 2023 |
| Gemini-1 | 1.6T | Google | 2023 |
| LLaMA-2 | 70B | Meta | 2023 |
| PanGu-$\Sigma$ | 1.085T | 华为 | 2023 |
| DeepSeek-V1 | 6.7B | DeepSeek | 2023 |
| GPT-4V | ~1.8T | OpenAI | 2024 |
| DeepSeek-V2 | 236B | DeepSeek | 2024 |
| Qwen-Max | ~1.2T | 阿里 | 2025 |
| GPT-5 | ~2–5T | OpenAI | 2025 |
| DeepSeek-V3 | 671B | DeepSeek | 2025 |
| Gemini 3.1 Pro | ~2–3T | Google | 2026 |
| Grok 4.3 | ~3–6T | xAI | 2026 |
| Claude Opus 4.7 | ~1T+ | Anthropic | 2026 |
| GPT-5.5 | ~2–5T | OpenAI | 2026 |
| Kimi K2.6 | 1T | Moonshot AI | 2026 |
| DeepSeek-V4-Pro | 1.6T | DeepSeek | 2026 |
| Grok V9 Medium | 1.5T | xAI | 2026 |
| Claude Mythos 5 | ~10T | Anthropic | 2026 |

Table: 大型 AI 模型的比较 {#tbl:model-comparison}
:::

[^model_size_comp]: 波浪号（~）表示近似参数计数。许多大型模型是闭源的，所以确切的参数计数未公开。这些近似值基于模型架构、训练成本和行业估计的推断。

[^llm_param_lie]: Wu，"The LLM Parameter Lie，"《Summer in Charlotte》（日记），2026年6月7日。\url{https://wu-99.com/diary/20260607.html\#the-llm-parameter-lie}。讨论MoE总参数与活跃参数、不可靠的基于回归的探测、前沿闭源模型的行业估计（GPT-4~1.76T总计、Claude Opus 4.x~5T MoE、GPT-5/Gemini 3.1在~2T–5T范围内），以及Claude Mythos 5作为首次公开讨论的~10T级模型（~800B–1.2T每token活跃）。

[^gpt4_training]: SemiAnalysis，"GPT-4 Architecture, Infrastructure, Training Dataset, Costs, Vision, MoE，"2023；Epoch AI，"Compute Trends Across Three eras of Machine Learning，"2023。

[^training_costs]: Epoch AI，"Trends in GPU price-performance，"2024；SemiAnalysis，"The Cost of Training Large Language Models，"2023；OpenAI，"GPT-4 Technical Report，"2023。


### 规模挑战

以 70B 参数模型为例。在全精度（FP32）下，仅存放模型权重就需要 280 GB 显存。当今主流的数据中心 GPU 没有哪一款单卡能够直接装载——即便是配备 141 GB 的 H200 和 192 GB 的 B200 也远远不够（第~\ref{chap:gpu-hardware-networking-and-parallelism-strategies}章将系统讨论 GPU 架构演进与显存容量趋势）。仅为了将该模型加载进显存，你就必须引入多张 GPU 进行协同托管，遑论更复杂的训练过程。

训练前沿大模型动辄需要消耗数以万计的 GPU 卡时。若在单张 GPU 上进行完整预训练，往往需要耗费数月甚至数年；与此同时，训练数据集的规模也呈爆发式增长——动辄覆盖数万亿 tokens。想要高吞吐、低延迟地加载与预处理这些海量数据，必须构建高度优化的分布式数据处理流水线。

底层物理矛盾显而易见：**模型参数规模与计算复杂度呈指数级跃升，而单张 GPU 的显存容量与算力密度至多呈线性增长**。

![增长不匹配：指数级模型增长与线性GPU增长](img/growth_mismatch_zh.png)

### 估算模型资源需求

在正式启动训练或部署之前，精确核算显存与算力需求是第一要务。一旦预估失误，轻则频繁遭遇 OOM 显存溢出错误，重则因盲目过度配置集群算力而造成惊人的资金浪费。

内存占用取决于你存储的内容。仅对于模型权重，计算很直接。FP32中的每个参数占4字节，FP16/BF16占2字节，Int8占1字节，Int4占0.5字节。对于7B参数模型，这是FP32中的28GB、BF16（或FP16）中的14GB、Int8中的7GB、Int4中的3.5GB。

以下是常见精度格式的快速参考：

| 格式 | 字节 | 格式详情 | 主要用途 |
|------|--|----------------|---------------|
| FP32 | 4 | 32位浮点（1符号位，8指数位，23尾数位） | 训练、高精度推理 |
| BF16 | 2 | 16位bfloat（1符号位，8指数位，7尾数位） | 训练（首选）、推理 |
| FP16 | 2 | 16位浮点（1符号位，5指数位，10尾数位） | 推理 |
| FP8 E4M3 | 1 | 8位浮点（1符号位，4指数位，3尾数位） | 推理（激活值、权重） |
| FP8 E5M2 | 1 | 8位浮点（1符号位，5指数位，2尾数位） | 训练（梯度存储） |
| MXFP8 | ~1 | E4M3 + 32值块缩放 | Blackwell训练 |
| NVFP4 | ~0.5 | E2M1 + 16值块缩放 | Blackwell推理 |
| Int8 | 1 | 8位整数 | 量化推理 |
| Int4 | 0.5 | 4位整数 | 量化推理（极端压缩） |


注意FP8有两种格式：E4M3（更高精度，用于推理激活值和权重）和E5M2（更宽的动态范围，用于存储）。两者都使用每个参数1字节，但用于不同目的。在NVIDIA Blackwell上，**MXFP8**改进了Hopper风格的按张量FP8，具有更精细的块缩放；**NVFP4**将推理推至8位以下（以及某些训练堆栈）。两者都使用微缩放——有效内存略高于标称位值。有关硬件上下文，请参见第~\ref{chap:gpu-hardware-networking-and-parallelism-strategies}章，有关FP8训练，请参见第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章。

对于训练，BF16（bfloat16）比FP16（float16）更受欢迎。BF16与FP32有相同的指数范围（8位）但尾数减少（7位），使其与FP32有相同的动态范围。这使训练更稳定——不太可能溢出或下溢。FP16有更小的指数范围（5位），可能在训练期间导致数值问题。现代GPU（A100、H100）的Tensor Cores针对BF16优化。对于推理，FP16和BF16都可行，但BF16对训练一致性仍然更受欢迎。

但模型权重只是开始。在训练期间，你还需要为梯度、优化器状态和前向传播的激活值预留空间。对于推理，你需要注意力机制的KV缓存。KV缓存对LLM推理至关重要——它存储序列中先前token的键值对，允许模型避免为每个新token重新计算整个序列上的注意力。这大大加快了自回归生成，但它需要大量内存，随批大小和序列长度缩放。总内存要求可能是仅模型权重的好几倍。

#### 训练内存需求

训练需要远比推理更多的内存。你需要存储模型权重、梯度（每个参数一个）、优化器状态和前向传播的激活值。

__优化器状态__

优化器状态大小取决于你使用哪个优化器。以随机梯度下降（SGD）为例，模型权重根据这个公式更新：

$$
w_{t+1} = \boxed{w_t} - \eta  \boxed{g_t}
$$

其中：

$w_t$：迭代$t$时的模型参数  
$w_{t+1}$：更新的参数  
$g_t$：损失关于参数在迭代$t$时的梯度（$g_t = \nabla_w L(w_t)$）  
$\eta$：学习率（用于SGD和自适应矩估计（Adam））  

盒中的变量是我们需要保存在内存中的。SGD只需要学习率$\eta$（标量）来更新参数。查看公式，你在反向传播时计算$g_t$，然后从$w_t$中减去$\eta g_t$。优化器状态就是$\eta$——可忽略的内存。你需要存储$w_t$（模型权重）和$g_t$（梯度），但不需要额外的优化器张量。

自适应矩估计（Adam）的公式更复杂：

$$
w_{t+1} = \boxed{w_t} - \eta
\frac{\beta_1 \boxed{m_{t-1}} + (1-\beta_1) \boxed{g_t}}{\sqrt{\beta_2 \boxed{v_{t-1}} + (1-\beta_2) \boxed{g_t}^2} + \epsilon}
\cdot
\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
$$

变量定义如下：

$\beta_1$：第一矩（均值）的衰减率  
$\beta_2$：第二矩（无中心方差）的衰减率  
$m_{t-1}$：来自前一步的第一矩估计  
$v_{t-1}$：来自前一步的第二矩估计  
$\epsilon$：用于数值稳定性的小常数  
$t$：用于偏差修正的迭代索引

Adam需要更多。公式显示它维护两个每参数张量：$m_{t-1}$（第一矩估计）和$v_{t-1}$（第二矩估计）。$m_{t-1}$和$v_{t-1}$的形状与$w_t$相同——每个参数一个值。所以Adam存储$m_{t-1}$（1倍模型大小）和$v_{t-1}$（1倍模型大小），优化器状态总计2倍模型大小。

$\beta_1$和$\beta_2$是超参数——标量常数（通常$\beta_1=0.9$，$\beta_2=0.999$）。你存储一次作为配置，不是每个参数。$\eta$（学习率）和$\epsilon$也一样——它们是标量，可忽略的内存。只有形状与$w_t$相同的张量（每个参数一个值）需要大量内存：$w_t$、$g_t$、$m_{t-1}$和$v_{t-1}$。

这就是为什么Adam需要2倍的模型大小用于优化器状态，而不是SGD的接近零开销。AdamW（带解耦权重衰减的Adam）与Adam有相同的内存需求——两者都维护$m_{t-1}$和$v_{t-1}$，总计2倍模型大小。唯一的区别是如何应用权重衰减（AdamW直接应用于参数，而Adam将其添加到梯度）。

以下是常见优化器的优化器状态内存需求汇总：

| 优化器 | 优化器状态 | 内存|
|-------|------------------|-----|
| SGD | 学习率$\eta$（标量） | ~0× |
| SGD+动量 | $v_{t-1}$（速度/动量） | 1× |
| Nesterov | $v_{t-1}$（速度/动量） | 1× |
| Adagrad | 累积梯度平方 | 1× |
| RMSProp | $v_{t-1}$（第二矩） | 1× |
| Adam | $m_{t-1}$（第一矩）+ $v_{t-1}$（第二矩） | 2× |
| AdamW | $m_{t-1}$（第一矩）+ $v_{t-1}$（第二矩） | 2× |
| Adafactor | 行和列统计（分解） | ~0.5× |
| LAMB | $m_{t-1}$（第一矩）+ $v_{t-1}$（第二矩） | 2× |
| Lion | $m_{t-1}$（仅第一矩） | 1× |
| Nadam | $m_{t-1}$（第一矩）+ $v_{t-1}$（第二矩） | 2× |
| AMSGrad | $m_{t-1}$（第一矩）+ $v_{t-1}$（第二矩）+ $v_{\max}$ | 3× |
| SparseAdam | $m_{t-1}$（第一矩）+ $v_{t-1}$（第二矩，稀疏） | 2× |
| Shampoo | 左右预条件矩阵每参数 | >2×（变化） |
| AdaBelief | $m_{t-1}$（第一矩）+ $s_{t-1}$（信念项） | 2× |

该表仅显示优化器状态。无论使用哪个优化器，你仍然需要存储模型权重（1倍）和梯度（1倍）。

__激活值输出__

激活层（如ReLU、GELU、sigmoid）没有参数——它们只是逐元素应用的函数。但它们的输出（激活值输出，通常简称为"激活值"）需要在训练期间存储在内存中。

考虑一个简单的3层DNN `SimpleDNN`如下：

![](img/simplednn_zh.png)


数据流是$x \rightarrow z \rightarrow h \rightarrow \hat{y}$，带有Linear → Sigmoid → Linear层，其中输入$x$通过第一个线性层产生$z$，然后通过sigmoid激活产生$h$，最后通过第二个线性层产生预测$\hat{y}$。PyTorch中的代码如下：

```python
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1, bias=False):
        super(SimpleDNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=bias)  # x -> z
        self.activation = nn.Sigmoid()                              # z -> h
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=bias) # h -> y_hat

    def forward(self, x):
        z = self.linear1(x)      # z = W_1 * x
        h = self.activation(z)   # h = sigmoid(z)
        y_hat = self.linear2(h)  # y_hat = W_2 * h
        return y_hat
```

前向传播是：

$$
z = W_1 x, \quad h = \sigma(z), \quad \hat{y} = W_2 h, \quad L = \frac{1}{2}(y - \hat{y})^2
$$

为了使用反向传播计算梯度，我们应用链式法则。

对于$\frac{\partial L}{\partial W_2}$：

$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2} = (\hat{y} - y) \cdot h
$$

梯度取决于$h$——sigmoid层的激活值输出。你需要在内存中存储$h$来计算这个梯度。

对于$\frac{\partial L}{\partial W_1}$，链更长：

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial W_1}
$$

展开每一项：$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$（需要$\hat{y}$），$\frac{\partial \hat{y}}{\partial h} = W_2$（需要$W_2$），$\frac{\partial h}{\partial z} = \sigma'(z) = \sigma(z)(1-\sigma(z)) = h(1-h)$（需要$h$），以及$\frac{\partial z}{\partial W_1} = x$（需要输入$x$）。

所以：

$$
\frac{\partial L}{\partial W_1} = (\hat{y} - y) \cdot W_2 \cdot h(1-h) \cdot x
$$

这个梯度需要$h$（sigmoid层的激活值输出）和$x$（输入数据）。这就是为什么激活值输出和输入数据必须在前向传播期间保存在内存中，直到反向传播期间计算它们的梯度。

一个有趣的观察是$z$不需要。查看梯度计算，我们需要$\frac{\partial h}{\partial z} = \sigma'(z)$来计算$\frac{\partial L}{\partial W_1}$。对于sigmoid，导数是$\sigma'(z) = \sigma(z)(1-\sigma(z)) = h(1-h)$。由于我们已经从前向传播中存储了$h$，我们可以直接从$h$计算导数，而不需要原始$z$值。这是sigmoid和某些其他激活函数的属性——它们的导数可以用它们的输出表示，所以你不需要存储预激活值。

然而，这并非对所有激活函数都适用。某些激活函数的导数明确依赖于输入值$z$，而不仅仅是输出$h$。这意味着你无法仅从$h$计算导数——你需要存储原始$z$值。

为了理解哪些激活函数需要存储预激活值$z$，哪些可以仅从后激活值$h$计算导数，让我们检查常见激活函数的导数：

| 激活 | 公式 | 导数 |
|------------|---------------------------|-------------------------------------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2 \cdot z) - 1$ | $\tanh'(z) = 1 - \tanh^2(z) = \text{sech}^2(z)$ |
| **ReLU** | $\text{ReLU}(z) = \max(0, z)$ | $\text{ReLU}'(z) = 1$ if $z > 0$, $0$ if $z \leq 0$ |
| **Leaky ReLU** | $\text{LeakyReLU}(z) = \max(\alpha z, z)$ | $\text{LeakyReLU}'(z) = 1$ if $z > 0$, $\alpha$ if $z \leq 0$ |
| **ELU** | $\text{ELU}(z) = z$ if $z > 0$, $\alpha(e^z - 1)$ if $z \leq 0$ | $\text{ELU}'(z) = 1$ if $z > 0$, $\alpha e^z$ if $z \leq 0$ |
| **GELU** | $\text{GELU}(z) = z \cdot \Phi(z)$ | $\text{GELU}'(z) = \Phi(z) + z \cdot \phi(z)$ |
| **Swish** | $\text{Swish}(z) = z \cdot \sigma(z)$ | $\text{Swish}'(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z))$ |
| **Mish** | $\text{Mish}(z) = z \cdot \tanh(\text{Softplus}(z)) = z \cdot \tanh(\ln(1 + e^z))$ | $\text{Mish}'(z) = \frac{e^z (4(z+1) + 4e^{2z} + e^{3z} + e^z(4z+6))}{(1 + e^z)^2 (1 + e^{2z})}$ |
| **GEGLU** | $\text{GEGLU}(z) = z \odot \text{GELU}(z)$ | $\text{GEGLU}'(z) = \text{GELU}(z) + z \cdot \text{GELU}'(z) = \text{GELU}(z) + z(\Phi(z) + z \cdot \phi(z))$ |
| **ReGLU** | $\text{ReGLU}(z) = z \odot \text{ReLU}(z)$ | $\text{ReGLU}'(z) = 2z$ if $z > 0$, $0$ if $z \leq 0$ |
| **SwiGLU** | $\text{SwiGLU}(z) = z \odot \text{Swish}(z) = z^2 \cdot \sigma(z)$ | $\text{SwiGLU}'(z) = 2z \cdot \sigma(z) + z^2 \cdot \sigma(z)(1 - \sigma(z))$ |
| **Softplus** | $\text{Softplus}(z) = \ln(1 + e^z)$ | $\text{Softplus}'(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$ |
| **Softmax** | $\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$ | $\nabla_{\mathbf{z}} \text{Softmax}(\mathbf{z})_{ij} = \text{Softmax}(\mathbf{z})_i(1 - \text{Softmax}(\mathbf{z})_i)$ if $i = j$, $-\text{Softmax}(\mathbf{z})_i \cdot \text{Softmax}(\mathbf{z})_j$ if $i \neq j$ |


>NOTES: **激活表约定**

上面的所有激活都是逐元素的（每个输出仅取决于其对应的输入），除了Softmax是向量值的（将向量输入转换为概率分布，总和为1）。Softmax的导数是Jacobian矩阵，用$\nabla_{\mathbf{z}}$表示。Leaky ReLU和ELU中的参数$\alpha$是常数超参数（Leaky ReLU通常$\alpha = 0.01$，ELU$\alpha = 1.0$）。GLU（门控线性单元）族（GEGLU、ReGLU、SwiGLU）是门控激活，使用逐元素乘法（$\odot$）组合两个分支：一个分支保持不变（$z$），另一个分支应用激活函数。实际上，GLU变体通常用两个分支的单独线性投影实现，但这里简化的形式对两个分支使用相同输入$z$。

>NOTEE

查看导数，我们可以根据导数是否可以用输出$h$纯粹表示来分类激活函数：

- **仅用输出表示的导数**：Sigmoid、Tanh和Softplus属于这一类。正如我们对sigmoid看到的，$\sigma'(z) = h(1-h)$仅取决于$h$。类似地，$\tanh'(z) = 1 - h^2$和$\text{Softplus}'(z) = \sigma(z)$可以从输出计算。对于这些函数，你只需要在前向传播期间存储$h$。

- **需要输入$z$的导数**：ReLU、Leaky ReLU、ELU、GELU、Swish、Mish和GLU族（GEGLU、ReGLU、SwiGLU）需要存储$z$，因为它们的导数明确包含$z$或取决于$z$的符号。对于ReLU，你需要知道$z > 0$还是$z \leq 0$来计算导数。对于GELU、Swish、Mish和GLU变体，导数公式明确包含$z$，使得从$h$单独恢复$z$不可能。

让我们详细检查GELU和Swish以说明为什么它们需要存储$z$。对于GELU（高斯误差线性单元），精确形式使用累积分布函数：

$$
h = z \cdot \Phi(z)
$$

其中$\Phi$是标准正态分布的累积分布函数。导数是：

$$
\frac{\partial h}{\partial z} = \Phi(z) + z \cdot \phi(z)
$$

其中$\phi$是标准正态的概率密度函数（PDF）。这个导数明确在项$z \cdot \phi(z)$中包含$z$。

实际上，GELU通常使用tanh近似以提高计算效率：

$$
h = 0.5 \cdot z \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (z + c \cdot z^3)\right)\right)
$$

其中$c \approx 0.044715$是常数。这个近似的导数是：

$$
\begin{aligned}
\frac{\partial h}{\partial z} &= 0.5 \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (z + c \cdot z^3)\right)\right) \\
&\quad + 0.5 \cdot z \cdot \left(1 - \tanh^2\left(\sqrt{\frac{2}{\pi}} \cdot (z + c \cdot z^3)\right)\right) \cdot \sqrt{\frac{2}{\pi}} \cdot (1 + 3c \cdot z^2)
\end{aligned}
$$

即使使用tanh近似，导数仍然在多个术语中明确包含$z$，包括$z$本身、$z^2$和tanh参数中的$z^3$。由于$\tanh$不容易可逆，表达式在多个地方涉及$z$，你无法从$h$单独恢复$z$。因此，你需要存储原始$z$值以在反向传播期间计算导数，无论你使用确切的GELU形式还是tanh近似。

类似地，Swish（也称为SiLU、Sigmoid线性单元）有：

$$
h = z \cdot \sigma(z)
$$

导数为：

$$
\frac{\partial h}{\partial z} = \sigma(z) + z \cdot \sigma(z)(1-\sigma(z))
$$

再次，导数明确包含$z$，你无法用纯粹$h$来表示它。为了计算$\frac{\partial h}{\partial z}$，你需要$z$和$\sigma(z)$，这意味着存储原始$z$值。

实际上，PyTorch等框架存储计算反向传播期间梯度所需的任何中间值。对于导数可以用其输出表示的激活函数（如sigmoid），框架可能仅存储后激活值。对于导数需要输入的函数（如GELU或Swish），框架存储预激活值。自动求导系统根据计算图中的操作自动确定需要保存什么，确保可以正确计算梯度，同时最小化内存使用。

在前向传播期间，你计算并存储所有层激活输出（我们简称为"激活值"）。在反向传播期间，你从最后一层向后逐层计算梯度。对于每一层，你需要其激活输出来计算梯度。一旦计算了某一层的梯度并更新了其参数，你可以释放该层的激活输出——不再需要它。然而，在反向传播期间有重叠：当计算一层的梯度时，你仍有较早层的激活输出在内存中。峰值内存出现在你同时有激活值（来自前向传播）和梯度（来自反向传播）的时候。

激活值内存随批大小和序列长度扩展——更大的批或更长的序列意味着更多激活值输出要存储。梯度检查点等技术通过重新计算激活值而不是存储所有激活值来权衡计算与内存。

__训练阶段__

在训练期间，内存使用在训练循环的不同阶段变化。理解何时需要每个组件有助于估计峰值内存需求并识别优化机会。

```python
for epoch in range(num_epochs):
    model.train()                         # set to training mode
    for x_batch, y_batch in dataloader:   # iterate over batches
        optimizer.zero_grad()             # 1. clear gradients
        y_hat = model(x_batch)            # 2. forward pass
        loss = criterion(y_hat, y_batch)  # 3. compute loss
        loss.backward()                   # 4. backward pass
        optimizer.step()                  # 5. update parameters
```

让我们分解每个步骤中发生的事情以及内存中存储的内容：

步骤2（前向传播）：模型从输入`x_batch`计算预测`y_hat`。在这个过程中，它存储中间激活值（如我们之前示例中的$h$）将用于反向传播。此时，内存包含：模型权重$w_t$、激活值和来自先前迭代的优化器状态（如Adam的$m_{t-1}$和$v_{t-1}$）。

步骤4（反向传播）：`loss.backward()`计算所有参数的梯度$g_t = \nabla_w L(w_t)$。这需要前向传播期间存储的激活值。内存现在包含：$w_t$、激活值（仍需要）、梯度$g_t$和优化器状态。

步骤5（优化器步）：`optimizer.step()`使用梯度$g_t$和优化器的内部状态从$w_t$更新参数到$w_{t+1}$。对于Adam，这意味着使用$m_{t-1}$和$v_{t-1}$以及$g_t$来计算更新。在这一步之后，优化器状态被更新（例如$m_{t-1}$变为$m_t$，$v_{t-1}$变为$v_t$）并存储用于下一次迭代。激活值现在可以释放，因为不再需要它。

激活值、梯度和优化器状态不是同时存在于内存中的。在前向传播期间，仅激活值被主动使用——梯度还不存在，优化器状态从前一次迭代持续但不被访问。在反向传播期间，激活值和梯度重叠，因为你需要激活值来计算梯度。在优化器步中，梯度和优化器状态重叠，当优化器使用两者更新参数。激活值通常在反向传播完成后释放，所以它们不与优化器状态重叠。

峰值内存使用发生在反向传播期间，当激活值和梯度同时在内存中。反向传播后，激活值可以释放，所以优化器步只需要梯度和优化器状态。


![训练内存时间线](img/training_memory_timeline_zh.png){.wrap #fig:training-memory-timeline width=60% align=top-right}

如@fig:training-memory-timeline所示，时间线显示7B模型使用Adam的训练循环的内存使用，假设**整个过程使用BF16**——权重、梯度和优化器状态（$m$、$v$）都以每参数2字节存储。这是一个清晰的教学示例。许多生产设置使用**混合精度**不同的方式：前向和反向使用BF16（通过`torch.autocast`），但优化器状态——有时还有权重的主副本——以FP32，这将优化器内存推向56 GB（4字节×2状态×7B参数）而不是28 GB。

以下是每个阶段如何映射到代码。在第2行（`y_hat = model(x_batch)`），前向传播计算并存储激活值。内存使用为权重（14 GB）加优化器状态（28 GB）加激活值（12 GB），总计54 GB。梯度还不存在。

在第4行（`loss.backward()`），反向传播是峰值内存发生的地方。在反向传播期间，你需要激活值来计算梯度和正在计算的梯度。内存使用为权重（14 GB）加优化器状态（28 GB）加激活值（12 GB）加梯度（14 GB），总计68 GB。这是峰值，因为激活值和梯度在内存中重叠。

在第5行（`optimizer.step()`），反向传播完成后，激活值可以释放。优化器使用梯度和其内部状态来更新参数。内存使用为权重（14 GB）加优化器状态（28 GB）加梯度（14 GB），总计56 GB。激活值不再需要。

68 GB的峰值内存发生在`loss.backward()`（第4行）期间，当激活值和梯度同时在内存中时。这就是为什么减少批大小、使用**梯度累积**（运行几个更小的微批，仅在最后一个后调用`optimizer.step()`——相同的有效批大小，更低的峰值激活值内存），或使用梯度检查点有助于遇到内存不足错误时：前两个在反向传播期间缩小激活值内存；检查点通过较少激活值存储交换额外计算。


__内存细分：__

对于相同的7B Adam设置（整个过程BF16）：模型权重（14 GB）+梯度（14 GB）+优化器状态（28 GB）= **56 GB固定**在激活值之前。激活值根据批大小和序列长度添加8–16 GB——时间线使用12 GB，给出**68 GB峰值**（56 + 12）在反向传播期间。完整范围因此是**64–72 GB**每GPU（56 GB + 8–16 GB激活值），不是图形中的单独估计。使用SGD你可以节省28 GB优化器状态，但Adam的自适应学习率通常收敛更快，所以在大多数情况下交换值得。这就是为什么7B模型在这个占用量下使用Adam进行训练需要至少A100（80 GB）；FP32优化器状态或主权重推高要求。较小的GPU没有分片（FSDP、DeepSpeed）或本书后面介绍的其他技术是无法进行的。



#### 推理内存需求

推理内存需求与训练不同，主要包括模型权重和注意力计算使用的键值（KV）缓存。KV缓存存储序列中先前token的预计算键值对，通过避免在整个序列历史上冗余的注意力计算来启用高效的自回归生成。虽然这个优化显著加快推理，但KV缓存引入内存开销，与批大小、序列长度和模型深度线性扩展——第~\ref{chap:distributed-inference-fundamentals-and-vllm}章详细介绍其大小调整。

推理的内存占用随三个主要因素扩展：模型大小、批大小和序列长度。对于70B参数模型使用BF16精度，模型权重消耗约140 GB。对于批大小32和序列长度2048，KV缓存增加额外的20-40 GB，导致总内存需求160-180 GB。这超过了单个A100 GPU（80 GB）的容量，需要多GPU配置或推理工作负载的模型并行策略。

具有超长上下文窗口的模型的内存需求大幅增加。例如，`meta-llama/Llama-4-Scout-17B-16E-Instruct`模型支持1000万token上下文窗口。尽管是17B参数混合专家（MoE）模型，带16个专家——每token仅激活专家子集——KV缓存在完整上下文长度下可能超过1 TB内存用于单个序列[^vllm-blog-llama4-mem]。这种规模的内存需求使单节点推理架构不可行，使分布式推理系统与数十个GPU不仅有益，而是根本必需。超长上下文模型的计算和内存需求建立分布式系统作为唯一可行的部署架构。

[^vllm-blog-llama4-mem]: vLLM中的Llama 4 - https://blog.vllm.ai/2025/04/05/llama4.html

#### GPU需求估算

估算GPU需求需要考虑模型内存和运营开销。对于训练工作负载，计算从模型内存占用开始，必须添加10-20%安全余量以适应通信缓冲和框架开销。考虑13B参数模型使用BF16精度：基础内存需求约72 GB每GPU。应用安全余量，这增加到85 GB。由于A100 GPU提供80 GB内存，这个配置需要使用模型并行或完全分片数据并行（FSDP）策略的2个GPU。

推理需求遵循类似的计算方法，组合模型权重和KV缓存内存。70B参数模型在BF16中需要140 GB仅用于权重。考虑KV缓存开销，总内存需求达到160-180 GB。这需要最小2个A100 GPU，或者，Int8量化可以将权重内存减少到约70 GB，通过谨慎的KV缓存管理可能启用单GPU部署。

#### 实际考虑

实际部署在基础模型需求之外引入额外的内存开销。PyTorch框架通常消耗1-2 GB，而操作系统需要5-10 GB。分布式训练架构为每GPU分配额外2-5 GB的通信缓冲。检查点操作在容量规划中创建必须考虑的临时内存峰值。保守的方法向基础估计添加20-30%缓冲以适应这些开销和运营变化。

内存优化策略在实际部署中起关键作用。使用BF16的混合精度训练（或推理的FP16/BF16）与FP32相比可减少约50%内存消耗，对模型准确性影响最小。对于推理工作负载，Int8量化可进一步将内存需求减半，同时保持可接受的准确性降级。通过`nvidia-smi`等工具监控实际内存使用提供对理论估计的实证验证。重要的是认识到激活值内存随批大小线性扩展；遇到内存不足（OOM）错误时，减少批大小代表最直接的缓解策略。

快速参考表[^memory_estimates]：

| 模型大小 | FP32权重 | BF16权重 | 训练（BF16+Adam） | 推理（BF16） |
|------------|--------------|--------------|----------------------|------------------|
| 1B         | 4 GB         | 2 GB         | ~8 GB                | 2-4 GB           |
| 7B         | 28 GB        | 14 GB        | ~60-70 GB            | 14-20 GB         |
| 13B        | 52 GB        | 26 GB        | ~110-130 GB          | 26-35 GB         |
| 70B        | 280 GB       | 140 GB       | ~600-700 GB          | 140-180 GB       |

[^memory_estimates]: 训练估计假设Adam优化器和中等批大小。实际值根据架构、序列长度和批大小变化。

### 从经典ML到基础模型的演进

从经典机器学习到现代基础模型的进展代表了计算需求和架构范例的根本转变。经典机器学习模型被设计用于在单机上运行，传统算法——包括线性回归、逻辑回归、决策树、随机森林、支持向量机（SVM）和梯度提升方法（XGBoost、LightGBM）——通常包括数千到数百万参数并训练于能装入系统内存的数据集。

深度学习时代引入了具有数百万参数的模型，由ResNet和BERT等架构范例。虽然这些模型需要GPU加速，它们在单设备配置上保持可管理。当代基础模型时代从根本上改变了这个格局：模型现在跨越数十亿到数万亿参数（例如，GPT-4、Gemini、LLaMA），需要分布式系统作为架构先决条件而不是优化。

向分布式AI的这个转变启用了突破性能力，包括能理解和生成类似人类文本、代码和多模式内容的模型。这个转变也促进了企业采用，组织为生产工作负载大规模部署AI系统。此外，分布式架构通过多计算节点上的并行实验启用的更快迭代周期加速了研究进展。

## 现代AI模型生命周期

构建AI模型不是一次性过程。这是一个周期：你收集数据、训练模型、部署它、看它如何表现、然后返回并改进数据或模型。每个阶段都馈入下一个。

![现代AI模型生命周期](img/mdlc_zh.png){#fig:lifecycle .block width=75% align=top-right}

如@fig:lifecycle所示，生命周期从数据工程开始，其中收集、策划、转换、验证、清理和准备数TB数据用于训练。训练遵循，涉及前向传播、反向传播、梯度下降、超参数调整，甚至微调。一旦训练，模型通过量化、ONNX转换、算子融合和CUDA内核优化进行推理优化。在部署之前，综合基准评估通过精度和召回指标、工程性能分析、瓶颈分析和压力测试的模型性能，分布式评估加快了大数据集上的测试。生产部署需要自动扩展、调度、负载平衡、可观察性、API网关和监控基础设施以处理每秒数千个请求。生产反馈识别数据收集优先级和模型失败模式，通过通知后续数据工程努力和模型改进完成循环。


这本书专注于你为训练、推理、基准和部署需要的分布式技术。数据工程获得简要概述，但不是主要焦点。分布式数据处理是重要的，但这是一个公认的话题。Spark、Dask和Ray已存在多年。这本书的主要焦点是AI特定分布式挑战：训练大型模型、优化推理和大规模服务。

#### 训练：学习阶段

训练是关于从数据学习模型参数。过程遵循模式：通过模型的前向传播、损失计算、计算梯度的反向传播和调整参数的梯度更新。这在多个epoch上迭代进行，直到模型收敛。

训练需要在内存中存储激活值、梯度和优化器状态。计算密集且迭代。在分布式训练中，你需要跨设备频繁的梯度同步以保持所有模型副本一致。挑战包括梯度同步开销、大型模型的内存约束、可能跨越数天到数周的长训练时间，以及对容错和检查点的需求。

在1万亿token上训练7B参数模型通常需要8个A100 GPU（每个80GB）并约2周连续训练。你需要仔细的梯度同步以跨所有GPU维护训练稳定性。

#### 推理：预测/生成阶段

推理是关于从训练的模型生成预测。与训练不同，你只需要前向传播——没有梯度、没有反向传播、没有优化器状态。内存需求较低，因为你只需要模型权重和用于注意力机制的KV缓存。每个请求的计算较低，但你需要高吞吐量来同时服务许多请求。通信最少，主要仅用于分布式推理。

挑战包括延迟（交互式应用的亚秒级）、吞吐量（每秒数千个请求）、通过KV缓存管理的高效内存使用，以及有效批处理和调度。为聊天服务70B参数模型需要优化推理引擎如vLLM或SGLang、持续批处理以最大化GPU利用率，以及用于可变长度序列的谨慎KV缓存管理。

#### 服务：生产系统

服务是关于提供对模型的可靠、可扩展访问。这不仅仅是运行推理——它是构建带模型运行器、API网关、负载均衡器和监控的生产系统。需求包括高可用性、容错和可观察性。在规模上，你处理多模型、多租户系统。

挑战包括系统可靠性与高可用保障、多模型智能路由与负载均衡、通过提升 GPU 利用率和动态自动伸缩实现成本控制，以及全链路调试的可观测性。工业级 LLM 线上服务平台通常需要同时管理多个模型变体（不同参数规格、领域微调版本），并配套 A/B 测试实验底座、金丝雀发布流水线以及分布式链路追踪与指标监控。

以下是_训练与推理与服务_的表格：

| 方面 | 训练 | 推理 | 服务 |
|--------|----------|-----------|---------|
| **目标** | 学习参数 | 生成预测 | 提供访问 |
| **内存** | 高（激活值+梯度） | 中等（权重+KV缓存） | 可变 |
| **计算** | 迭代、密集 | 单个前向传播 | 请求驱动 |
| **通信** | 频繁（梯度） | 中等 | API级别 |
| **延迟** | 小时到天 | 毫秒到秒 | 毫秒 |
| **吞吐量** | 样本每秒 | token每秒 | 请求每秒 |

## 决策框架：何时需要分布式系统？

分布式系统增加复杂性、通信开销和成本。在你必须使用时使用它们，不是在你想要时。

### 决策树：快速参考

如@fig:decision-framework所示，决策框架总结在下面的决策树。通过识别你的用例开始：训练（或微调）与推理和服务。


### 理解决策树

当训练或微调时，通过检查你的模型是否装入内存开始。在BF16中计算模型大小（参数×2字节）。如果它超过单GPU容量，你需要像FSDP或张量并行的模型或参数并行。13B模型仅用于权重需要26GB。添加Adam优化器状态，你在80GB A100上被紧紧地看。

如果模型装入但训练在数周或数月中拖延，数据并行可以加快速度。在8 GPU上训练7B模型1T token需要约两周，与单GPU上的数月相比。

当模型大小和训练时间都是可管理的，你的微调方法很重要。像LoRA和QLoRA的参数高效方法改变了方程式。在48GB GPU上的QLoRA在70B模型上装入因为它仅训练适配器权重，而完全微调相同模型需要多个GPU。但如果基础模型不装入，你仍然需要模型并行无论你选择哪个训练方法。

大型数据集其中数据加载成为瓶颈受益于分布式数据加载。多TB数据集是数据并行的好候选。

对于推理或服务，逻辑类似。如果模型超过单GPU内存，使用模型并行。70B模型在BF16中需要140GB用于权重。使用KV缓存，你看160-180GB，需要至少2个A100 GPU。如果内存很好但你需要高吞吐量——每秒数千个请求——使用多个GPU用于分布式推理。需要亚秒级延迟在高吞吐量的实时服务通常需要张量并行或多个推理实例。当内存和吞吐量都在单GPU限制内装入时，坚持一个GPU并使用优化引擎如vLLM或SGLang以最大化效率。

![决策框架：何时需要分布式系统？](img/decision_tree_zh.png){#fig:decision-framework .block width=100%}


\fancydividerwithicon[center]{python.png}


## 实际操作：运行分布式训练

我们将开始一个简单的基线以建立性能参考点，然后移动到分布式训练以看到实际的加速。

### 环境设置

在这本书中，我们将使用PyTorch作为我们的主要框架，代码可以从本书的git仓库克隆。

```bash
git clone https://github.com/PacktPublishing/Distributed-AI-Systems
```

为了运行本书的配套实战代码，建议使用配备多卡 GPU（如 A10、A100、H100/H200 乃至 B200）的算力环境。如果你本地没有多卡硬件，Kaggle 提供了免费的双卡 GPU 实验环境：登录 [https://www.kaggle.com](https://www.kaggle.com)，点击“Create”并新建 Notebook。

![Kaggle Notebook 创建](img/1.5_zh.png){.block align=center}

创建好 Jupyter Notebook 后，进入右侧“Settings”面板，在“Accelerator”下拉菜单中选择“GPU T4 x 2”。

![Kaggle GPU设置](img/3_zh.png){.block width=60% align=top-left}

你现在应该有2个T4 GPU可用。为了验证你的GPU设置，运行`code/check_cuda.py`中的代码：

```python
#LINENUM
import torch
print(f"CUDA available: {torch.cuda.is_available()}") #HL
print(f"Number of GPUs: {torch.cuda.device_count()}") #HL
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i) #HL
    vram_gb = props.total_memory / (1024**3) #HL
    print(f"GPU {i}: {props.name} ({vram_gb:.1f} GB)")
```
CODE_EXPLAIN_START:

- 2: 检查当前系统上 CUDA 环境是否可用
- 3: 获取系统中可用的物理 GPU 总数
- 5: 获取指定 GPU 的硬件属性与规格
- 6: 将显存容量从字节转换为以 GB 为单位

CODE_EXPLAIN_END

![GPU设置 - 2个Tesla T4 GPU](img/2_zh.png){#fig:gpu-setup .wrap width=65% align=top-right lines=10}

如@fig:gpu-setup所示，运行这个应该显示你可用的GPU。


### 单GPU基线

我们将使用ResNet18，一个轻量级18层卷积神经网络为图像分类设计。它足够小以快速训练同时有效地演示分布式训练概念。对于数据集，我们将使用FashionMNIST - 70,000灰度图像（28×28像素）跨10个服装类别。它是MNIST的直接替代品，稍微更具挑战，使其理想用于快速实验。

运行基线训练脚本：

```bash
python code/single_gpu_baseline.py
```

3个epoch后，你应该看到类似这样的输出：

```
Epoch 1/3, Loss: 0.4164, Accuracy: 84.94%
Epoch 2/3, Loss: 0.2936, Accuracy: 89.17%
Epoch 3/3, Loss: 0.2531, Accuracy: 90.58%

Total training time: 8.78s
```

你的确切结果将根据你的硬件变化，但损失和准确性值应该类似。这个基线给我们一个参考点——我们将在分布式版本中使用相同的模型架构以进行公平比较。

>NOTES: **ResNet18** 和 **FashionMNIST**

torchvision中的ResNet18被设计用于3通道RGB图像，但FashionMNIST使用1通道灰度图像。为了使ResNet18适配FashionMNIST，我们修改两层：（1）替换第一个卷积层以接受1个输入通道而不是3：`model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)`。（2）替换最终完全连接层以输出FashionMNIST的10个类：`model.fc = nn.Linear(model.fc.in_features, 10)`。这些最小改变允许ResNet18处理FashionMNIST的灰度图像，同时保留架构的其余证明设计。

>NOTEE


### 多GPU分布式训练

现在让我们运行分布式版本。脚本使用相同的ResNet18模型在FashionMNIST上，但将工作分散到多GPU。

```bash
torchrun --nproc_per_node=2 code/multi_gpu_ddp.py
```

或使用启动脚本：

```bash
bash code/launch_torchrun.sh
```

使用2个GPU，你将看到类似的损失和准确性值，但训练在5.91秒而不是8.78秒完成——一个**1.48倍加速**。

我们测试了从1到8 GPU的扩展以看性能如何改进：

| GPU | 训练时间 | 加速倍数 |
|------|---------------|---------|
| 1    | 8.78s         | 1.00×   |
| 2    | 5.91s         | 1.48×   |
| 4    | 3.69s         | 2.38×   |
| 6    | 2.92s         | 3.01×   |
| 8    | 2.44s         | 3.60×   |

![FashionMNIST缩放性能](img/fashionmnist_scaling_performance_zh.png){.block width=70%}

训练时间从8.78秒降至2.44秒使用8个GPU，实现**3.6倍加速**。添加更多GPU显著减少训练时间并加快开发周期。如我们接下来将看的，加速在更大工作负载上变得更明显。

### 扩展训练：CIFAR-10，20个Epoch

对于更现实的工作负载，我们测试了ResNet18在CIFAR-10上使用20个epoch。CIFAR-10比FashionMNIST更大更复杂，拥有50,000个训练图像使用3通道RGB图像32×32像素。这更好地展示分布式训练的好处。

运行单GPU基线：

```bash
python code/single_gpu_extended.py --epochs 20
```

然后测试分布式训练：

```bash
torchrun --nproc_per_node=2 code/multi_gpu_ddp_extended.py --epochs 20
torchrun --nproc_per_node=4 code/multi_gpu_ddp_extended.py --epochs 20
torchrun --nproc_per_node=6 code/multi_gpu_ddp_extended.py --epochs 20
torchrun --nproc_per_node=8 code/multi_gpu_ddp_extended.py --epochs 20
```

结果显示甚至更好的缩放：

| GPU | 训练时间 | 加速倍数 |
|------|---------------|---------|
| 1    | 73.00s (1.22 分钟) | 1.00×   |
| 2    | 46.47s (0.77 分钟) | 1.57×   |
| 4    | 27.72s (0.46 分钟) | 2.63×   |
| 6    | 21.24s (0.35 分钟) | 3.44×   |
| 8    | 18.20s (0.30 分钟) | 4.01×   |

![CIFAR-10缩放性能](img/cifar10_scaling_performance_zh.png){.block width=70%}

在 8 张 GPU 上，训练耗时从 73 秒大幅缩短至 18.2 秒，取得了 **4.01 倍的加速比**——这一扩展效果显著优于前面的 FashionMNIST 实验。原因在于：每个 Epoch 的计算负载越大，梯度同步在整体训练时间中所占的比例就越小。在持续 20 个 Epoch 的训练中，集合通信的启动开销被充分摊销到更多的计算步骤中；每张 GPU 在两次通信同步之间都有充沛的计算任务，从而最大化释放了硬件并行效率。

这一实验清晰地揭示了为什么分布式训练是超大模型训练的必由之路：单步计算密度越高，通信开销的相对占比就越低，系统的扩展效率也就越高。对于百亿甚至千亿参数的超大规模模型，当计算密集度占据绝对主导地位时，分布式扩展的加速比甚至可以逼近理想的线性加速。这里演示的 DDP 方案基于经典的数据并行思想，即每张 GPU 均维护一份完整的模型副本，各自独立处理不同的数据批次。这是分布式训练中最直观、最易落地的模式，非常适用于单卡显存足以容纳整个模型的场景。而对于参数量远超单卡显存上限的超大规模模型，我们在后续章节中将深入剖析张量并行（Tensor Parallelism，跨卡切分层内权重）、流水线并行（Pipeline Parallelism，跨卡切分网络层阶段）以及融合多种并行维度的 3D 混合并行体系。

### 分布式推理：高吞吐扩展

分布式训练的核心诉求是缩短模型收敛所需的总耗时，而分布式推理的核心指标则是**系统吞吐量（Throughput）**——即每秒能够稳定承载的服务请求数（req/s）。分布式推理允许多张 GPU 并行处理不同的并发请求，从而成倍提升系统的服务吞吐能力。

我们以 FashionMNIST 上的 ResNet18 推理服务为例进行压测，评估吞吐量的扩展表现。与分布式训练不同，推理阶段无需进行反向传播与跨卡梯度同步，每张 GPU 仅需独立处理分发至本卡的请求即可。这使得分布式推理在横向扩容时具备极高的扩展效率。

在服务架构中，分发推理请求通常有两种经典模式：

1. **数据切分模式（Data Partitioning，见 `multi_gpu_inference.py`）**：借助 `DistributedSampler` 将全量请求预先切分为互不重叠的数据子集，每张 GPU 独立处理固定配额的数据。该模式实现简单、执行效率极高，非常适用于离线批处理（Batch Inference）场景。
2. **请求队列模式（Request Queue，见 `multi_gpu_inference_queue.py`）**：基于共享任务队列以轮询（Round-Robin）或抢占方式动态拉取请求，每张 GPU 充当独立的 Worker 进程，高度模拟真实微服务架构下的工作流。该模式具备极高的灵活性，更契合生产环境中异步并发请求的实时处理。

首先运行单 GPU 基准测试以获取 baseline 性能：

```bash
python code/single_gpu_inference.py --requests 1000
```

随后分别以 2 卡、4 卡和 8 卡运行数据切分模式的分布式推理基准：

```bash
torchrun --nproc_per_node=2 code/multi_gpu_inference.py --requests 1000
torchrun --nproc_per_node=4 code/multi_gpu_inference.py --requests 1000
torchrun --nproc_per_node=8 code/multi_gpu_inference.py --requests 1000
```

运行请求队列模式的分布式推理基准：

```bash
torchrun --nproc_per_node=2 code/multi_gpu_inference_queue.py \
  --requests 1000
```

实测基准性能对比如下：

::: {width=100%}

| GPU 数量 | 并发调度模式 | 总耗时 | 吞吐量 | 加速倍数 |
|---|---|---|----------------------------------|---|
| 1 | 单卡基线 | 1.85s | 541.00 req/s | 1.00× |
| 2 | 数据切分模式 | 0.98s | 1025.45 req/s | 1.89× |
| 2 | 请求队列模式 | 1.22s | 819.09 req/s | 1.51× |

:::

在 2 卡环境下，数据切分模式实现了 **1.89 倍的加速比**，推理吞吐量几乎翻倍。由于各 GPU 在执行过程中互不干扰、无需跨卡协调，数据切分模式展现出了近乎完美的线性扩展能力。而请求队列模式虽然因进程间轮询调度和队列轮询带来了微量协调开销（加速比为 1.51 倍），但在面向生产级异步突发流量时，却具备更强大的动态负载均衡能力。

这两种模式生动展现了分布式推理的价值：系统吞吐量能够随着 GPU 算力的堆叠实现近线性增长，这正是高并发生产级 AI 服务不可或缺的底层支柱。上述实验采用的是每张 GPU 均完整持有模型权重的数据并行推理模式。而当面对百亿乃至千亿参数、单卡显存无法容纳的超大语言模型时，我们则必须进一步采用张量并行、序列并行、流水线并行及混合专家（MoE）并行，并深度集成 vLLM 与 SGLang 等现代推理引擎，这些高级架构将在后续章节全面展开。

## 使用PyTorch的分布式AI基础

现在你已看到分布式训练和推理运行，让我们理解使其工作的基本概念和API。

### 分布式AI堆栈

分布式AI系统分层构建，从高级框架到物理硬件。理解这个堆栈帮助你调试问题、优化性能并做关于要使用哪个工具的明智决定。

虽然堆栈适用于所有分布式框架（PyTorch、JAX、TensorFlow），这本书使用PyTorch作为主要示例。概念转换到其他框架，但API和实现详情不同。我们整本书将专注于PyTorch分布式API。

![分布式AI堆栈：从框架到物理层](img/fig_distai_stack_zh.png){.wrap width=40% align=top-right}

在分布式 AI 软件栈的最顶层是**框架层（Framework Layer）**。这是开发者日常编写核心业务代码的界面——包括 PyTorch 的 `torch.distributed` 模块、`DDP`（DistributedDataParallel）、`FSDP`（FullyShardedDataParallel）等高级抽象。当你定义模型结构并调用 `loss.backward()` 时，底层框架会自动接管梯度的反向传播与同步调度。在这一层，开发者无需操心底层的网络数据包或硬件链路细节，只需聚焦于编写训练循环，由框架来编排底层的分布式通信逻辑。

在框架层之下，PyTorch 引入了**梯度分桶（Gradient Bucketing）机制**。为了避免为每个小张量单独发起网络通信而产生巨大的网络延迟开销，DDP 会将反向传播计算出的众多梯度张量聚合装入连续的内存分桶（Buckets）中。当一个分桶填满后，便立即触发该分桶的异步网络通信，从而实现计算与通信的高效重叠（Overlap）。理解这一机制至关重要，它能帮助你在遇到通信延迟反常或 GPU 空转时，准确定位是否由于分桶尺寸设置不当而导致等待。

**集合通信层（Collective Operations Layer）**定义了具体的通信拓扑与数学规约语义。例如，AllReduce 负责在所有进程（rank）间汇总规约梯度并把最终结果广播回每个进程；AllGather 负责从所有进程收集张量并沿指定维度拼接；Broadcast 则负责将单个根进程的数据广播至其他所有进程。这些集合通信原语构成了 DDP、FSDP 以及各种并行方案的基石。在这一层，系统明确了数据交换的逻辑语义（“做什么”），而具体的传输实现则下沉至更底层。

**数据传输层（Transport Layer）**是集合通信算法具体落地的底层执行引擎。对于基于 NVIDIA GPU 的分布式训练集群，**NCCL（NVIDIA Collective Communications Library，集合通信库）**是当之无愧的标准后端。NCCL 针对 NVIDIA GPU 微架构、NVLink 拓扑及高速网络进行了极致的硬件级定制优化，能够以极高的带宽利用率实现 Ring AllReduce、Tree AllReduce 等核心集合通信原语。对于 CPU 分布式或无 GPU 的本地调试场景，PyTorch 提供了基于 CPU 的 **GLOO** 后端；而在传统高性能超算环境中，**MPI** 也是可选后端之一。在调用 `dist.init_process_group(backend="nccl")` 时，你实质上就是在指定由 NCCL 来接管底层的硬件数据传输。

**网络拓扑层（Network Topology）**决定了集群内部各计算设备的互联架构。在环形拓扑（Ring Topology）中，各 GPU 逻辑上组成单向或双向环，数据依次接力传递；在胖树拓扑（Fat-Tree Topology）中，交换机层级间提供无收敛、无阻塞的多路径冗余连接，大幅缓解通信拥塞；而在 Torus 或 Dragonfly 拓扑中，则在布线复杂度和网络对分带宽之间做出了精细的工程权衡。拓扑结构直接决定了 NCCL 寻找通信路径的方式，深刻影响着通信带宽与时延。

**物理链路层（Physical Links）**承载实际的高速数据传输。在单个节点内部，**NVLink** 以极高的双向带宽直连 GPU（根据不同硬件架构代际，每卡双向带宽可达 600 GB/s 至 1.8 TB/s，详见第~\ref{chap:gpu-hardware-networking-and-parallelism-strategies}章）。当单机容纳多张 GPU 时，卡间主要依托 NVLink 互联。而在跨节点的分布式训练中，**InfiniBand（IB）网络配合 RDMA（远程直接内存访问）技术**则扮演着中流砥柱的角色，它允许网卡直接绕过 CPU 与系统内存读写对端 GPU 显存，极大地降低了端到端通信延迟。此外，PCIe 总线负责卡内 GPU 与宿主机 CPU/内存的桥接，而普通以太网（RoCE/Ethernet）虽成本更低、更为普及，但其延迟与吞吐往往不及专用 IB 组网。物理链路层的性能规格，直接决定了整个集群所能达到的理论带宽上限。

在软件栈的最底层则是**物理硬件层（Physical Hardware）**——包括用于通用密集矩阵运算的 GPU、面向专用张量加速的 TPU（见第~\ref{chap:gpu-hardware-networking-and-parallelism-strategies}章），以及负责作业控制与数据预处理的宿主机 CPU。这一层奠定了系统绝对的计算算力峰值与显存物理边界。上层再精巧的软件优化，也无法突破物理硬件所决定的客观上限。

当你在业务代码中调用 `dist.all_reduce()` 时，整个调用链将自顶向下穿透整套技术栈：PyTorch 运行时将梯度张量规整进连续内存桶，发起 AllReduce 集合通信请求；NCCL 驱动探测并匹配当前物理拓扑（如 NVLink 或 InfiniBand），以最高效的通信算法驱动数据在物理介质上高速流动，最终将求和规约后的张量写回各 GPU 显存中。深刻理解这套分层架构，对于诊断分布式训练中的疑难杂症大有裨益：当系统遭遇通信挂起、吞吐不及预期或显存暴涨时，你能够迅速排查——究竟是底层网卡驱动配置有误？是拓扑路径被降级回了低速 PCIe？还是集合通信调度的时机不当？有了清晰的技术栈全景，故障定位便能有的放矢。

厘清了技术栈的分层关系后，我们接下来探讨在 PyTorch 中落地分布式编程的核心基础概念：进程组（Process Group）、Rank 标识与底层通信原语。无论你是使用 DDP 或 FSDP 进行数据并行，还是亲手实现复杂的张量并行与流水线调度，抑或是搭建高性能分布式推理服务，这些概念都是你必须掌握的底层积木。

### 进程组与 Rank

在分布式深度学习中，系统的并行计算本质上是由多个并发进程协作完成的。每个进程通常绑定并独占一张独立的 GPU。PyTorch 借助**进程组（Process Group）**这一核心概念来统一组织和调度所有参与计算的进程。在进程组内部，每个进程都被分配了一个唯一的整数编号，称为 **Rank**（编号从 0 递增）。整个分布式集群中参与协作的进程总数，则被称为 **World Size**。

术语 “World” 指代整个分布式作业的全局范围。例如，若你在 2 个物理节点上开展训练，每个节点配备 4 张 GPU，那么你的全局 World Size 即为 8。

若集群共有 4 张 GPU，则通常拉起 4 个独立的 Python 工作进程。进程 0 绑定 GPU 0，进程 1 绑定 GPU 1，以此类推。Rank 的核心作用是指引每个进程明确自己在全局计算图中的职责：应该独占哪张 GPU，以及应该切分并加载训练数据集的哪一部分。

在多机多卡场景下，Rank 细分为两种语义：**全局 Rank（Global Rank）** 与 **本地 Rank（Local Rank）**。全局 Rank 在整个分布式集群（World）内全局唯一，取值范围为 `0` 到 `world_size - 1`。而本地 Rank 则仅在单个物理节点内部唯一，每个物理节点内部的进程均从 `0` 开始编排。例如，在一个包含 2 个节点、每节点 4 张 GPU 的集群中，节点 0 内部进程的本地 Rank 为 0–3（对应的全局 Rank 为 0–3）；节点 1 内部进程的本地 Rank 同样为 0–3（但对应的全局 Rank 则为 4–7）。在工程实践中，本地 Rank 通常严格对应当前节点上的物理 GPU 设备编号，这就是为什么在分布式初始化代码中，我们总是能看到 `torch.cuda.set_device(local_rank)` 这一标准配置。

### 初始化进程组

在执行任何跨卡集合通信操作之前，必须显式调用进程组初始化函数。这相当于告知 PyTorch 底层：各个进程该通过何种通信后端建立连接。在基于 NVIDIA GPU 的集群中，标准后端为 NCCL（NVIDIA 集合通信库）。

最经典的初始化代码范式如下：

```python
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # 使用 NCCL 进行 GPU 间高速集合通信
        rank=rank,       # 当前进程的全局 rank
        world_size=world_size  # 集群总进程数
    )
    torch.cuda.set_device(rank)  # 将当前进程绑定到对应的 GPU 物理设备
```

要验证你的分布式多卡环境是否配置正常，最轻量的测试脚本见 `code/distributed_basic_test.py`。该脚本专注于校验基础进程组的握手与点对点连通性，不依赖任何上层 DDP 封装，纯粹用于验证多进程能否顺畅通信。

在具备多卡 GPU 的机器上，可以使用 `torchrun` 启动双卡验证：

```bash
torchrun --nproc_per_node=2 code/distributed_basic_test.py
```

如果你当前手头只有单张 GPU，但希望在本地调试多进程分布式逻辑，可以通过 CUDA 设备掩码在单卡上模拟多进程环境：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/multi_gpu_simulation.py
```

无论哪种方式，控制台若能顺利打印出 "Rank 0 says hello" 与 "Rank 1 says hello"，便确认了底层分布式通信链路已成功贯通。

当你使用 `torchrun` 启动分布式任务时，终端可能会打印关于 `OMP_NUM_THREADS` 的环境警告。这是因为 PyTorch 试图对底层 OpenMP 的 CPU 线程并发数进行合理约束，以防止多进程并发时过度挤占 CPU 调度资源。你可以在执行 `torchrun` 命令前通过显式指定 `OMP_NUM_THREADS=4`（或匹配你 CPU 物理核数的合理数值）来消除该警告：

```bash
OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/distributed_basic_test.py
```

请务必注意：该环境变量必须在 Shell 环境中随启动命令一同传入——而在 Python 脚本内部通过 `os.environ` 动态设置是无效的，因为 `torchrun` 在派生子进程前就已经完成了对宿主环境变量的读取与校验。

### 集合通信操作

集合通信（Collective Communications）是分布式多卡协同计算的核心基石。它们规范了数据在不同计算进程之间高效流转与数学规约的交互模式。每种集合通信原语都有其不可替代的工程应用场景。深入领会这些操作的底层原理，不仅能让你在设计模型并行方案时做出最佳选型，更能让你在定位网络通信性能瓶颈时做到游刃有余。

PyTorch 提供了八种核心集合通信操作。接下来我们通过具体的代码示例与时序图逐一剖析。

#### AllReduce

![AllReduce操作：跨所有Rank的归约](img/all_reduce_zh.png){#fig:allreduce}

如@fig:allreduce所示，AllReduce 是分布式深度学习训练中应用最频繁、最重要的集合通信原语。它首先在所有 Rank 之间对输入张量执行指定的数学规约操作（如求和 SUM、求最大值 MAX、求最小值 MIN），随后将最终计算出的聚合结果广播回每个 Rank 的输出缓冲区中。这是一种全对全（All-to-All）式的通信拓扑：每个参与的 Rank 既是数据的提供方，也是最终聚合结果的接收方。

AllReduce 是经典数据并行（Data Parallelism）训练的命脉。在分布式数据并行（DDP）中，各个 Rank 在反向传播过程中仅计算本卡本地微批次（Mini-batch）数据所对应的梯度。为了维持全局模型参数的一致性，这些分散的本地梯度必须在所有 Rank 之间进行全局同步。DDP 正是在反向传播阶段隐式调用 AllReduce，对全集群的梯度进行跨卡求和规约，并除以总进程数得出平均梯度[^allreduce-note]。完成 AllReduce 后，每张 GPU 上都拥有了一模一样的全局梯度，从而使得各卡上的优化器能够严格步调一致地推进模型参数更新。倘若没有 AllReduce，每张卡将只能依据片面的局部梯度更新模型，模型参数势必迅速发散，导致训练彻底崩溃。

[^allreduce-note]: AllReduce 是分布式训练中最核心、最频繁调用的集合通信操作。

规约操作的操作符包括 SUM（求和，梯度同步的标准操作）、MAX（求最大值）、MIN（求最小值）以及 PRODUCT（连乘，极少在深度学习中使用）。在梯度同步中，使用 SUM 求和后通常直接除以 `world_size` 计算平均值。这一平均操作保证了全局有效批大小（Effective Batch Size）能够随着参与训练的 GPU 数量线性扩展——例如在 8 卡集群上训练，单步迭代所消费的数据吞吐量即为单卡基线的 8 倍。

在实现机制上，AllReduce 的执行效率远高于朴素的“先 Reduce 汇聚到主节点，再由主节点 Broadcast 广播”的两步方案。尽管数学结果完全一致，但 NCCL 针对 AllReduce 研发了专属的环形（Ring AllReduce）与树状（Tree AllReduce）拓扑算法，极大程度压缩了网络通信轮次与传输开销。例如在经典的 Ring AllReduce 算法中，仅需 $2 \times (\text{world\_size} - 1)$ 轮轻量级的点对点数据传输，即可完成全量聚合，其通信带宽利用率接近物理极致。

```python
# Each rank has different input
tensor = torch.tensor([rank + 1, rank + 2, rank + 3], device=device)
# After all_reduce with SUM, all ranks have the same result
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# Result: [sum(1..world_size), sum(2..world_size+1), sum(3..world_size+2)]
```

运行演示：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_allreduce.py
```

AllReduce 的通信开销与单卡传输的数据量以及 World Size 紧密相关。得益于 NCCL 的 Ring AllReduce 等算法优化，系统能实现接近线性的扩展效率。但在参数量达数十亿的超大模型中，AllReduce 依然极易沦为集群扩展的阿喀琉斯之踵，因此梯度压缩、分桶聚合以及计算与通信异步重叠等工程优化对于维持高吞吐至关重要。

#### AllGather {#sec:allgather}

![AllGather操作：从所有Rank收集数据](img/all_gather_zh.png){#fig:allgather}

如@fig:allgather所示，AllGather 负责从所有 Rank 收集局部数据，并将拼接后的完整全量数据广播回每个 Rank。假设每个 Rank 贡献长度为 $N$ 的张量，则每个 Rank 最终都将接收到长度为 $\text{world\_size} \times N$ 的拼接张量。输出张量严格按照 Rank 索引升序排列（Rank 0 的数据排在最前，紧随其后的是 Rank 1 的数据，以此类推）。这同样是一种全对全的通信模式：每个 Rank 既向外发送自身的数据碎片，同时也接收来自其他所有 Rank 的数据。

当各个计算卡需要获取全局完整视图时，AllGather 是必不可少的核心原语。与 AllReduce 对数据进行规约聚合（求和、极值）不同，AllGather 完整保留了每个 Rank 的独立张量切片并按序拼接。这使得 AllGather 非常适用于跨卡收集 Embedding 特征表示、获取中间层激活值以执行全局自注意力计算、跨卡批归一化（Cross-GPU SyncBatchNorm）以及分布式评测指标的汇聚。

在完全分片数据并行（FSDP）训练体系中，AllGather 扮演着核心枢纽的角色：在模型的前向传播与反向传播计算前，每个 Rank 仅持有模型权重的一个分片，系统在运行时通过 AllGather 动态拉取其他卡上的权重分片，在显存中即时拼装出完整的层权重；一旦该层的矩阵乘计算完成，拼装出的临时全量参数便被立即释放，仅保留本地分片。这种按需拼装、用完即释的机制，使得 FSDP 能够轻松训练远超单卡显存容量的超大模型。

此外，AllGather 常见的使用场景还包括：收集所有 Rank 的预测结果用于集成评估、在 Transformer 模型中跨 Rank 收集特征图以计算注意力、在多机训练中同步词表嵌入或 Token 编码，以及在流水线并行中汇总各个流水线阶段的最终输出。

```python
# 每个 rank 拥有不同的局部输入
input_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], device=device)
# 预先分配用于接收所有 rank 拼接结果的列表
output_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
dist.all_gather(output_list, input_tensor)
# 此时所有 rank 均持有完整列表: [rank0_data, rank1_data, ..., rankN_data]
```

运行演示代码：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_allgather.py
```

AllGather 的通信数据量会随着数据尺寸与 World Size 共同增长。每个 Rank 发送 $N$ 个元素并接收 $\text{world\_size} \times N$ 个元素，集群内部流转的总数据量与 $\text{world\_size}^2$ 呈正比。这种二次方增长的通信开销使得 AllGather 在超大集群规模下成本不菲。因此，FSDP 等先进分片架构会严谨地控制 AllGather 的触发范围，并与 ReduceScatter 紧密搭配，通过计算重叠来最大化掩盖通信延迟。

>NOTES: **AllReduce 的等价分解**
在数学与逻辑上，先执行一次 **ReduceScatter**（跨卡规约并切片分发），紧接着执行一次 **AllGather**（全局拼接收集），其运算结果与一次 **AllReduce** 完全等价。许多高阶显存优化框架（尤其是 FSDP 和 ZeRO）正是巧妙地拆解并利用了这一等价性，通过将规约与收集切分到前后两个独立的计算阶段，实现了通信与计算的深度重叠。

>NOTEE

#### Broadcast（广播）

如@fig:broadcast所示，Broadcast 操作负责将根节点（Root Rank）上的数据原封不动地复制分发给集群内的其他所有 Rank。在操作发起前，只有指定的根节点持有源数据；操作完成后，所有参与的 Rank 都将拥有与根节点一模一样的副本。这是一种典型的一对多（One-to-All）通信拓扑，也是分布式系统中最直观的数据分发机制。

Broadcast 在分布式训练的作业初始化阶段起着至关重要的作用。当训练任务拉起时，通常由 Rank 0 负责从持久化磁盘或云存储中单点加载预训练模型权重、优化器检查点或配置参数。随后，Rank 0 通过 Broadcast 将权重秒级分发至其他所有计算卡，以确保各卡以绝对同步的模型参数开启训练。倘若没有 Broadcast，每个 Rank 都将不得不并发读取磁盘上的同一份大权重文件，这会引发严重的存储 I/O 拥塞甚至打崩分布式文件系统。

![广播操作：从根发送数据到所有Rank](img/broadcast_zh.png){#fig:broadcast}

除了分发模型初始化权重外，Broadcast 还广泛应用于：广播学习率及全局超参数配置、同步全局随机数种子以确保实验可复现性、在多节点间下达全局控制信号与停止标记（如 Early Stopping），以及在特定自定义并行拓扑中同步流水线阶段的状态信息。

```python
root = 0
if rank == root:
    tensor = torch.tensor([10.0, 20.0, 30.0], device=device)
else:
    tensor = torch.zeros(3, device=device)
# 执行广播后，所有 rank 的 tensor 内容均变为 [10.0, 20.0, 30.0]
dist.broadcast(tensor, src=root)
```

运行演示代码：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_broadcast.py
```

Broadcast 发送端的数据传输量仅取决于待分发张量的大小，而与接收端的 Rank 数量无关（根节点发出的总有效数据量是固定的）。在现代树状广播（Tree Broadcast）等网络优化算法的加持下，广播能够以对数级时延将超大张量平摊分发至数百上千张 GPU。

#### Reduce（规约）

如@fig:reduce所示，Reduce 执行与 AllReduce 完全相同的数学规约运算（求和、求极值等），但唯一的区别在于：计算出的最终聚合结果仅保存在指定的根节点（Root Rank）上，其他非根节点的缓冲区保持原有数值不变。这是一种典型的多对一（All-to-One）通信模式。

当只有单个主控进程需要获取聚合结果时，Reduce 的通信开销显著低于 AllReduce。因为 AllReduce 在完成规约后还需要额外执行一步全员广播，而 Reduce 则直接省去了这部分全网带宽消耗。在仅需集中处理聚合结果的场景（如记录日志、写入监控指标、保存检查点或触发全局控制决策）下，选用 Reduce 能够大幅减轻网络拥塞。

![规约操作：向根Rank的归约](img/reduce_zh.png){#fig:reduce}

Reduce 最典型的应用场景包括：在训练循环中汇总各卡上的评测指标并汇总至 Rank 0。在每轮迭代中，每张卡在本地微批次上计算出局部 Loss、准确率或梯度范数；通过 Reduce 操作，这些局部指标被统一求和汇聚到 Rank 0，由 Rank 0 负责打印训练日志、上传至 TensorBoard/Wandb，或据此调整学习率调度器。

需要特别强调的是：Reduce 操作执行完毕后，只有根节点持有更新后的聚合值，其他非根卡的数据不会被修改。如果你需要确保所有卡上的张量都完成同步更新，请使用 AllReduce。

```python
root = 0
tensor = torch.tensor([rank + 1, rank + 2, rank + 3], device=device)
dist.reduce(tensor, dst=root, op=dist.ReduceOp.SUM)
# Only root has the sum; other ranks unchanged
```

运行演示：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_reduce.py
```

在数学与逻辑上，先执行一次 Reduce 汇聚到主卡，再由主卡执行一次 Broadcast，其效果等价于 AllReduce。但在物理实现中，NCCL 将 AllReduce 深度优化为一个一体化的硬件级流水线操作，其通信效率远高于这种显式的两步拆分。因此，只有在业务逻辑上确实仅需要单个主卡汇总结果（无需广播回各卡）时，才应选用 Reduce 操作。

#### Gather（收集）

如@fig:gather所示，Gather 操作负责将集群中所有 Rank 的数据汇集到指定的根节点（Root Rank）。假设每个 Rank 发送长度为 $N$ 的张量，根节点最终将接收到长度为 $\text{world\_size} \times N$ 的拼接张量，且各切片严格按照发送端的 Rank 编号顺序排列。这是一种典型的多对一（All-to-One）通信模式。

Gather 是 Scatter 的逆操作，常用于分布式训练中的集中式归总任务。与 AllGather 将收集到的全量数据广播给全集群不同，Gather 仅仅将拼接结果传输给根节点，因而在仅需主控节点处理全量数据的场景下极具带宽优势。这在集中式 I/O 落盘、监控日志汇总、集中式模型评估以及聚合决策中应用极为广泛。

![收集操作：从所有Rank到根收集数据](img/gather_zh.png){#fig:gather}

在调用 Gather 时，数据的拼接顺序是确定性的：Rank 0 的数据永远排在最前面，其后紧随 Rank 1、Rank 2 等。根节点必须在调用前预先分配好能容纳所有 Rank 数据切片的输出列表缓冲区（`output_list`），而非根节点在传入该参数时直接指定为 `None` 即可。

```python
root = 0
input_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], device=device)
if rank == root:
    output_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
    dist.gather(input_tensor, output_list, dst=root)
    # 此时根节点持有拼接后的全量数据列表
else:
    dist.gather(input_tensor, None, dst=root)
```

运行演示代码：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_gather.py
```

Gather 的通信开销与 World Size 呈线性正比关系（根节点需接收 $\text{world\_size} \times N$ 的数据量）。在超大规模集群中，若单次汇总的数据量过大，极易导致主节点网卡入向带宽发生拥塞。如果下游计算任务需要各卡均持有汇总数据，应直接改用 AllGather。

#### Scatter（分散）

![分散操作：从根分发数据到所有Rank](img/scatter_zh.png){#fig:scatter}

如@fig:scatter所示，Scatter 是 Gather 的精准逆操作。根节点将一份由多个切片组成的大张量分发至各个 Rank，每个 Rank 仅接收属于自己的那一部分切片。这是一种典型的一对多（One-to-All）差异化分发模式。

Scatter 与 Broadcast 的核心差异在于：Broadcast 是将同一份数据原封不动地复制给所有人；而 Scatter 则是将整体数据“切片切割”，每个 Rank 分得互不重叠的数据碎片。当主节点持有一批全局数据、需要按卡切分以执行数据并行或模型并行处理时，Scatter 便派上了用场。

在主节点（Root Rank）上，必须提前准备一个长度等于 `world_size` 且张量形状完全一致的列表 `scatter_list`。系统会根据 Rank 索引进行对应分发：Rank 0 接收 `scatter_list[0]`，Rank 1 接收 `scatter_list[1]`。非根节点在发起 Scatter 调用时，`scatter_list` 参数只需传入 `None`，同时各卡均需预先分配好用于承接切片的 `output_tensor`。

```python
root = 0
if rank == root:
    scatter_list = [torch.tensor([r * 10 + 1, r * 10 + 2], device=device) 
                   for r in range(world_size)]
output_tensor = torch.zeros(2, device=device)
if rank == root:
    dist.scatter(output_tensor, scatter_list=scatter_list, src=root)
else:
    dist.scatter(output_tensor, scatter_list=None, src=root)
# 此时每个 rank 均独立接收到了分配给本卡的局部切片
```

运行演示代码：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_scatter.py
```

虽然 Scatter 在语义上非常符合数据切分，但在现代大规模数据并行训练中，通常更推荐借助 `DistributedSampler` 让各卡在本地独立按索引分块加载数据，从而避免因将全量数据强行加载到主卡而引发的显存瓶颈与单点 I/O 风暴。

#### ReduceScatter {#sec:reducescatter}

![ReduceScatter操作：先归约后分散](img/reduce_scatter_zh.png){#fig:reducescatter}

如@fig:reducescatter所示，ReduceScatter 是 Reduce 与 Scatter 的高度工程化融合。它首先在所有 Rank 之间对输入张量执行跨卡数学规约（例如求和 SUM），紧接着将规约后的结果等分为若干切片，并分别分发至对应的各个 Rank。每张卡根据自己的 Rank 索引接收一个规约切片。这一原语是现代显存高效并行（如 ZeRO-2/3、FSDP）最关键的通信支柱。

在完全分片数据并行（FSDP）训练中，模型参数被切分并均匀分布在集群的各个 Rank 上。在反向传播计算梯度时，每张卡在本地仅持有一小部分模型参数的分片。为了完成全局参数更新，所有卡上计算出的对应梯度切片必须在跨卡求和后，精准投递回负责维护该参数分片的目标 GPU 上。ReduceScatter 以极高的硬件效率在单一通信流程中一气呵成完成了求和规约与切片分发，完全规避了显式拆解为两步所带来的额外通信时延与显存占用。

在每个 Rank 上，传入 ReduceScatter 的输入张量列表包含 `world_size` 个切片。规约完成后，每个 Rank 接收到属于本卡 Rank 编号的最终规约块。这种设计天然保证了规约梯度切片与参数分片的严格内存对齐。

```python
# 每个 rank 准备长度为 world_size * N 的切片输入
input_list = [torch.tensor([rank * 10 + i, rank * 10 + i + 1], device=device) 
              for i in range(world_size)]
input_tensor = torch.cat(input_list)
# 每个 rank 仅接收全局规约结果中属于本卡的一块切片
output_tensor = torch.zeros(2, device=device)
dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
```

运行演示代码：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_reducescatter.py
```

ReduceScatter 的底层通信传输模式与 Ring AllReduce 高度相似，但其最大优势在于通信输出是分片存储而非全局复制的。每个 Rank 最终仅在显存中保留自己的 $1/\text{world\_size}$ 切片，显存利用率极高。正如前文所述，在逻辑上 “ReduceScatter + AllGather = AllReduce”，FSDP 正是通过在反向传播使用 ReduceScatter、在前向传播使用 AllGather，实现了在超大模型训练中对计算与通信的无缝重叠隐藏。

#### AlltoAll（全对全通信）

![AlltoAll操作：全对全通信](img/all2all_zh.png){#fig:alltoall}

如@fig:alltoall所示，AlltoAll 是集合通信家族中最为通用、也最复杂的通信模式。在 AlltoAll 中，每个 Rank 均向其他所有 Rank 发送独一无二的专属数据块，同时也从其他所有 Rank 接收专属于自己的数据块。假设集群包含 `world_size` 个进程，那么每个 Rank 均需准备 `world_size` 块不同的数据，并最终接收来自其他卡所投递的 `world_size` 块专属数据。这构成了一张全连接的网状数据置换图。

这种强大的灵活性，使得 AlltoAll 能够支撑起其他集合通信原语根本无法表达的复杂数据流转。在张量并行（Tensor Parallelism）与序列并行（Sequence Parallelism）中，当矩阵乘法的切分维度发生轴变换（例如从列并行切换到行并行，或从时间步序列切分转换到注意力头维度切分）时，张量数据必须在卡间进行矩阵转置式的重排，这必须依托 AlltoAll 方能完成。

在混合专家模型（Mixture of Experts, MoE）的分布式训练与推理中，AlltoAll 更是当仁不让的核心枢纽：门控路由（Gating Router）为每个 Token 动态计算出最匹配的目标专家（Expert），不同专家分布在集群的不同 GPU 上，系统必须通过 AlltoAll 将各个 Token 秒级投递至对应的专家卡上进行 FFN 前向运算，计算完毕后再通过一次逆向 AlltoAll 将结果精准回传。

在编码实现上，每个 Rank 准备一个包含 `world_size` 个切片的列表 `input_list`，其中 `input_list[i]` 即为定向发送给 Rank $i$ 的专属张量。完成 `dist.all_to_all` 调用后，每个 Rank 拿到的 `output_list[j]` 即代表来自 Rank $j$ 的对应数据。

```python
# 每个 rank 为其他各个 rank 定制专属的发送切片
input_list = []
for dst_rank in range(world_size):
    chunk = torch.tensor([rank * 100 + dst_rank * 10 + 1, 
                         rank * 100 + dst_rank * 10 + 2], device=device)
    input_list.append(chunk)
# 每个 rank 均接收来自其他所有 rank 投递给本卡的专属切片
output_list = [torch.zeros(2, device=device) for _ in range(world_size)]
dist.all_to_all(output_list, input_list)
```

运行演示代码：

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 code/collective-operation/demo_alltoall.py
```

AlltoAll 是所有集合通信操作中网络吞吐压力最大的原语。在多节点集群中，全互连的流量极易打满核心交换机的背板对分带宽（Bisection Bandwidth），引发网络拥塞。因此，在超大模型系统设计中，工程团队通常会通过精巧的拓扑感知调度，尽量将 AlltoAll 限制在具备高带宽 NVLink 互联的单机内部，避免跨机泛洪。

>NOTES: **AlltoAll 需要 NCCL 后端支持**
在 GPU 分布式场景下，高效的 AlltoAll 必须依赖 NCCL 通信后端。PyTorch 的 GLOO（CPU 后端）对此操作的支持非常有限。因此在运行 MoE 或高阶序列并行代码时，必须确保环境具备可用的 GPU 硬件及适配的 NCCL 库。

>NOTEE

#### 如何选择恰当的集合通信原语？

面对纷繁复杂的集合通信原语，在实际工程研发中我们该如何抉择？

- **梯度全局同步**：标准首选是 **AllReduce**（见@fig:allreduce）。在最经典的数据并行（DDP）中，PyTorch 底层会在反向传播时自动调用该原语同步全网梯度，开发者无需手写通信代码。
- **全量特征/权重拼装**：若各卡仅持有局部分块，但后续计算必须依赖全局完整视图（例如 FSDP 在前向传播前即时拼装权重层、汇总分布式评测指标），首选 **AllGather**（见@fig:allgather）。
- **主节点单点分发**：当主卡完成预训练模型权重加载或需要广播统一的超参数配置时，选用 **Broadcast**（见@fig:broadcast）。
- **指标全局归一与日志落盘**：当仅需汇总各卡的 Loss 并在主控卡上集中打印训练日志时，选用 **Reduce**（见@fig:reduce）；若需将各卡的数据切片汇集到主节点执行集中式落盘，选用 **Gather**（见@fig:gather）。
- **显存高效分片同步**：在 FSDP 与 DeepSpeed ZeRO 等先进分片训练方案中，反向传播必须通过 **ReduceScatter**（见@fig:reducescatter）实现梯度的“规约并分片归宿”，并与 **AllGather** 形成完美闭环。
- **动态路由与张量重排**：在张量并行跨轴转置、序列并行切片交换以及 MoE 专家动态路由等高级并行拓扑中，**AlltoAll**（见@fig:alltoall）是不可替代的核心引擎。

在绝大多数日常工业级开发中，我们很少直接手写这些集合通信原语，成熟的框架早已为我们完成了精细的工程封装：DDP 在后台默默调度 AllReduce，而 FSDP 则在后台精密交替执行 ReduceScatter 与 AllGather。然而，深刻理解每种操作的物理本质与通信成本，是你诊断集群性能劣化、排查死锁挂起，以及量身定制高阶并行策略的必备内功。

### 分布式数据并行（DDP）核心机制

在五花八门的分布式训练策略中，分布式数据并行（DistributedDataParallel, DDP）是体系最成熟、使用最广泛、也最容易落地的架构。它通常只需对单机单卡训练代码做极小的侵入式修改，是所有算法工程师迈向分布式训练的第一站。本节简要剖析 DDP 的核心执行流，第3章将围绕其高阶优化、底层通信调优与实战最佳实践展开全方位深潜。

DDP 的核心设计理念是**数据分而治之，模型完全冗余**：集群中的每个工作进程各自独占一张 GPU，并在显存中维护一份一模一样的完整模型副本；输入训练集被切分成互不相交的子集，由各卡独立执行前向传播并计算局部梯度；在反向传播即将结束时，DDP 自动接管底层的梯度同步，通过异步重叠的 AllReduce 集合通信对全局梯度求平均，进而确保各卡优化器能够基于完全相同的参数更新模型。

只要模型参数与优化器状态能够被单张 GPU 显存完全承载，DDP 永远是第一优先级选型。而当模型参数量达到数十亿乃至上百亿、单卡发生 OOM 显存溢出时，我们则需要进阶采用完全分片数据并行（FSDP）、张量并行（TP）与流水线并行（PP）等方案。

在 PyTorch 中使用 DDP 包装模型的经典范式非常简洁：

```python
from torch.nn.parallel import DistributedDataParallel as DDP

# 首先将模型移动至当前进程独占的 GPU 物理设备
model = YourModel().cuda(rank)
# 使用 DDP 封装模型，指定当前设备 ID
model = DDP(model, device_ids=[rank])
```

包装完成后，后续的前向与反向计算与单卡编程体验完全无缝统一，DDP 会在后台全自动拦截并在 `loss.backward()` 执行期间异步完成梯度的重叠同步。

### 分布式采样器（DistributedSampler）

为了让各个工作进程能够分别处理不同的训练样本，必须在数据加载管道中引入 `DistributedSampler`。它的核心职责是对整个数据集进行分布式均匀切分，确保每张卡在每个 Epoch 都能按序抽取出互不重叠的数据切片。倘若遗漏了该采样器，所有 GPU 都将重复读取一模一样的数据，从而使分布式扩展彻底丧失意义。

```python
from torch.utils.data import DataLoader, DistributedSampler

# 为当前数据集绑定分布式采样器
sampler = DistributedSampler(
    dataset, 
    num_replicas=world_size,  # 集群总工作进程数
    rank=rank                  # 当前进程的全局 rank
)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

在训练循环中，必须在每个 Epoch 的最开始显式调用 `sampler.set_epoch(epoch)`，这是通知采样器根据 Epoch 编号更新随机置换种子的关键机制，能够保证样本混洗（Shuffle）在多卡间既随机可变又绝对互斥。

### 启动分布式训练任务

现代 PyTorch 统一推荐使用官方的 `torchrun` 工具来拉起分布式作业，它会自动接管进程派生、环境变量注入与故障重启：

```bash
torchrun --nproc_per_node=2 code/multi_gpu_ddp.py
```

该命令将在本地单机上拉起 2 个平行的 GPU 训练进程。若要扩展到跨节点的多机训练，只需额外传入 `--nnodes`（总节点数）、`--node_rank`（当前机器节点编号）以及 `--master_addr`（主节点通信 IP 地址）即可。

这一行简明的启动命令，将我们在本章所学的所有分布式核心原语有机串联：`init_process_group()` 建立集群通信拓扑，`barrier()` 实现多卡间数据下载的原子同步，`DistributedSampler` 优雅切分数据流，而底层的 `DDP` 则在 `loss.backward()` 期间无缝调用高效的 `AllReduce` 驱动全局梯度同步。

### 本章小结

本章系统梳理了算力资源预估、技术选型决策框架以及分布式 AI 系统的实战构建流程。其核心原则非常明确：**先精确评估计算与显存需求，再客观判断是否真正需要引入分布式系统**。切勿盲目假设必须进行分布式扩展——先精确核算显存占用与计算开销，确认单卡是否能够承载；在确实面临显存物理瓶颈或算力上限时，再审慎选择恰当的分布式并行方案。

在透彻理解了分布式系统的应用时机与核心软件架构之后，我们接下来必须深入探究其赖以运行的硬件基石。下一章我们将全景剖析现代 GPU 硬件微架构、高速网络互连拓扑（NVLink/InfiniBand），以及支撑超大模型运转的核心并行策略。融会贯通这些底层物理规律，是你在这场算力工程战役中保持清醒判断与决胜千里的大前提。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
