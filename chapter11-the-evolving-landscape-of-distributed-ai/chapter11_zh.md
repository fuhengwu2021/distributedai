# 第11章：分布式 AI 的演进图景 {-}

*探索分布式 AI 的新兴技术与未来方向*

> 预测未来的最好方式是发明它。
- Alan Kay，计算机科学家

**Code Summary**

- `torch.distributed.checkpoint.save`：用 DCP 进行异步分布式检查点
- `torch.distributed.checkpoint.load`：跨 rank 加载分片检查点
- `torch.distributed.elastic.multiprocessing`：带容错的弹性训练
- `torch.ao.quantization.quantize_dynamic`：用于模型压缩的动态量化
- `flwr.client.NumPyClient`：Flower 联邦学习客户端接口
- `flwr.server.strategy.FedAvg`：联邦平均聚合策略
- `megatron.core.transformer.moe.router.TopKRouter`：Megatron MoE top-k 路由
- `vllm.LLM`：带 PagedAttention 的 vLLM 推理引擎
- `sglang.Engine`：带 RadixAttention 的 SGLang 运行时



## 分布式 AI 的演进

贯穿本书，我们探索了分布式 AI 的基础：从 DDP 的梯度同步到 FSDP 的内存优化，从 vLLM 的 PagedAttention 到 SGLang 的 RadixAttention。这些技术使训练和服务几年前无法想象的模型成为可能。但该领域不会停滞——图景继续以令甚至经验丰富的从业者惊讶的方式变化。

### 我们今天所处的位置

数字讲述了一个非凡的故事。训练基础设施已扩展到单个集群超过 100,000 块 GPU，由像 NCCLX 和 Torchcomms 这样的新通信框架支持。SGLang 现在为全球超过 400,000 块 GPU 提供动力[^sglang-scale]，运行 xAI 的 Grok 3 和 Microsoft Azure 的生产工作负载。也许最引人注目的是经济性：DeepSeek-V3，一个 6710 亿参数的 MoE 模型，每 token 有 370 亿活跃参数，在 2,048 块 H800 GPU 上仅用 550 万美元训练——一个几年前看起来低得不可能的成本。

设备端 AI 已跨越一个关键门槛。像 Gemma 3n 这样的模型用像逐层嵌入这样的技术舒适地在 2-3GB RAM 中运行，使 LLM 在智能手机上实用。曾经看起来实验性的 4 位量化方法现在是标准实践。

但也许最显著的变化是优先级的变化。多年来，AI 行业痴迷地专注于训练：构建更大的集群、训练更大的模型、推动前沿。那个时代没有结束，但它不再是主导故事。推理工作负载现在消耗超过 55% 的 AI 基础设施支出，而那个份额继续增长。全球推理市场的预测在未来几年从数千亿到远超一万亿美元——这个跨度反映了分析师计算的是硅、云服务还是打包应用。

当你思考它时，这种转变有意义。训练发生一次；推理发生数百万次。随着模型成熟和部署扩大，经济性不可避免地青睐推理优化。基础设施成本在过去几年大幅下降，使在以前不切实际的环境中部署 AI 在经济上可行。

[^sglang-scale]: LMSYS/SGLang 项目更新。\url{https://github.com/sgl-project/sglang}，\url{https://lmsys.org/blog/}。

### 新的瓶颈

随着这种增长带来新的约束。有趣的是，主要瓶颈不再是硅的可用性——而是电力和制冷。数据中心正围绕热限制而非机架空间设计。组织现在将制冷改造纳入总拥有成本计算，并提前几个月协商云容量承诺。

剩余的技术挑战同样有趣。高效扩展超过 100K 块 GPU 需要新的通信模式和容错机制。在生产推理中平衡延迟和吞吐量仍然既是艺术也是科学。而跨混合云-边缘架构的成本优化正在成为一门独立的学科。

### 新兴趋势一览

在深入每个领域之前，这里是重塑分布式 AI 的路线图：

__高级 MoE 架构__：LatentMoE 为每 FLOP 最优准确性带来硬件-软件协同设计，被 Nvidia 的 Nemotron-3 采用。MoSE（可精简专家混合）实现可变宽度专家执行，用于连续的准确性-计算权衡。弹性 MoE 将推理时专家数量扩展到训练值的 2-3 倍。ReMoE 引入用 ReLU 而非 TopK+Softmax 的完全可微路由。

__设备端和边缘 AI__：亚十亿参数模型现在有效处理实际任务。Gemma 3n 的逐层嵌入减少 RAM 需求——5B/8B 模型以 2B/4B 占用运行。内存带宽而非 FLOPs 限制边缘推理；当草稿模型很好地跟踪目标时，推测解码有帮助。

__下一代通信__：Torchcomms，PyTorch 的新 API，为 100K+ GPU 规模设计，带异构硬件支持。NCCLX/RCCLX（Meta 的增强后端）在 AllReduce 操作上提供 10-50% 的加速。异步检查点现在用缓存计划和减少的 GIL 竞争快 6 倍。

__推理引擎演进__：SGLang 在 H100 上实现 16,215 tok/s，带用于自动前缀重用的 RadixAttention 和用于百万 token 上下文的流水线并行。vLLM 的 PagedAttention 将 KV 缓存浪费从 60-80% 减少到 4% 以下。两者现在都支持 AMD MI355/MI300、Intel Xeon、Google TPU 和昇腾 NPU。



## 混合专家：主导架构

我们在第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章（训练的专家并行，见图~\ref{fig:expert-parallelism}）和第~\ref{chap:distributed-inference-fundamentals-and-vllm}章（用 vLLM 的 MoE 推理）涵盖了 MoE 基础。这里我们专注于将 MoE 能力推得更远的研究前沿。

研究继续将 MoE 能力推向超过我们在前面章节涵盖的范围：

__LatentMoE__[^latentmoe] 使用硬件-软件协同设计来跨不同推理场景优化每 FLOP 准确性。被 Nvidia 的 Nemotron-3 模型采用，它证明正确的协同设计可以在不牺牲能力的情况下显著提高效率。

[^latentmoe]: LatentMoE: Toward Optimal Accuracy per FLOP. \url{https://arxiv.org/abs/2601.18089}

__MoSE（可精简专家混合）__[^mose] 允许可变宽度专家执行。MoSE 不是固定大小的专家，而是可以在推理时动态调整专家宽度，实现连续的准确性-计算权衡。这对延迟需求变化的部署场景特别有价值。

[^mose]: MoSE: Mixture of Slimmable Experts. \url{https://arxiv.org/abs/2602.06154}

__弹性 MoE__[^elastic-moe] 将活跃专家的数量扩展到超过训练时值。一个用 top-2 路由训练的模型可以在推理时用 top-4 或 top-6，实现训练时专家数量的 2-3 倍，同时改善性能。这将训练和推理配置解耦。

[^elastic-moe]: Elastic MoE: Inference-Time Expert Scaling. \url{https://arxiv.org/abs/2501.03140}

__ReMoE__[^remoe] 用完全可微的基于 ReLU 的路由替换不可微的 TopK+Softmax 路由。这实现更高效的动态计算分配并简化训练动态。

[^remoe]: ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing. \url{https://arxiv.org/abs/2412.14711}

`code/moe_layer.py` 中的代码提供扩展第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章概念的实现，包括用无辅助损失方法的负载均衡路由。



## 边缘-云连续体

分布式 AI 中最深刻的转变之一是云和边缘之间边界的模糊。多年来，假设很简单：在云中训练，也许在云中服务，而边缘设备用于消费。那个模型正在崩溃。

### 为什么边缘 AI 现在重要

边缘 AI 的理由很有说服力。当你消除网络往返时，延迟从数百毫秒降到个位数。当数据从不离开设备时，隐私大幅改善。当你不为云推理按 token 付费时，成本降低。而离线能力开辟了全新的用例。

什么改变使这实用？三件事汇聚。首先，模型架构变得更高效。亚十亿参数模型现在处理以前需要 7B+ 模型的实际任务。其次，量化技术成熟。从 16 位到 4 位将存储和内存流量都减少 4 倍，用像 GPTQ[^gptq] 和 AWQ[^awq] 这样的方法质量损失极小。第三，硬件改善。移动 NPU 现在足够强大，以可接受的速度运行这些量化模型。

[^gptq]: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. \url{https://arxiv.org/abs/2210.17323}
[^awq]: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. \url{https://arxiv.org/abs/2306.00978}

但这里有一个常被忽略的微妙之处。移动设备上的瓶颈不是计算——而是内存带宽。手机移动约 50-90 GB/s；H100 级数据中心 GPU 每秒移动 TB 级——在通常比较中 30-50 倍的差距，对 H200 更宽，对像 V100 这样较旧的卡更窄。量化重要，因为它削减每次前向传播跨越那个总线的字节，而不只是存储在磁盘上的权重。

### 实用的设备端模型

边缘能力模型的图景已快速成熟：

| 模型 | 参数 | 内存 | 用例 |
|-------|-----------|--------|-----------|
| Gemma 3n | 5B/8B | 2-3GB | 通用助手 |
| Llama 3.2 | 1B/3B | 1-2GB | 文本任务 |
| Phi-4 mini | 3.8B | 2GB | 推理 |
| Qwen2.5 | 0.5B-1.5B | <1GB | 轻量任务 |

Google 的 Gemma 3n 使用逐层嵌入以分别 2B 和 4B 模型的内存占用运行 5B 和 8B 参数模型——总共约 2-3GB。Meta 的 Llama 3.2 提供专门为边缘部署设计的 1B 和 3B 变体。Microsoft 的 Phi-4 mini 将强大的推理能力打包进 3.8B 参数。阿里巴巴的 Qwen2.5 范围从 0.5B 到 1.5B 用于最轻量的任务。

关键洞见是在小规模下架构和训练质量比原始大小更重要。一个训练良好的 1B 模型可以在许多任务上胜过一个训练差的 3B 模型。

### 边缘-云协调

最有趣的系统不将边缘和云视为独立的世界——它们找到使它们协同工作的方式。记得第~\ref{chap:cross-request-optimization-with-sglang}章的推测解码吗？我们用它在单个服务器上通过让小草稿模型提议 token（然后更大的模型验证）来加速推理。相同的想法在边缘-云边界上完美工作。

想象你的手机运行一个微小但快速的模型。它生成一系列候选 token——也许 "The cat sat on the"——并将它们运送到一个强大的云模型。云不生成任何东西；它只检查每个 token 是否匹配它会产生的。验证很便宜：云模型可以在单个前向传播中评估所有五个 token，而一个一个生成它们需要五次传播。当草稿大多正确时（对可预测的文本，它们常常是），你以边缘般的速度获得云质量的输出。收益取决于任务：像代码或 JSON 这样的结构化输出常常达到 3-4 倍的端到端加速，而开放式创意文本可能只改善 1.2-1.5 倍，因为拒绝频繁[^spec-decode]。

![边缘-云推测解码工作流](img/speculative_decoding_zh.png){#fig:speculative-decoding .block width=100% align=center}

图~\ref{fig:speculative-decoding} 展示这个舞蹈。边缘设备提议 token（黄色圆圈）、将它们发送到云，验证器将每个标记为接受（绿色）或拒绝（红色）。在这个例子中，"on" 被拒绝——也许云模型偏好不同的介词——所以边缘需要从那个点重新生成。但五个 token 中的四个顺利通过，节省显著的延迟。

[^spec-decode]: Fast Inference from Transformers via Speculative Decoding. \url{https://arxiv.org/abs/2211.17192}

智能路由将这种协调推得更远。不是每个请求实际都需要云。一个设计良好的系统估计每个请求多难、检查边缘模型感觉多自信、瞥一眼网络条件，并决定：本地处理，还是发送到云？简单查询——"东京现在几点？"——留在设备上。复杂推理任务去云。路由逻辑甚至可以从过去的决策学习，在预测哪些请求会在本地成功上变得更好。

`code/edge_cloud.py` 文件实现这些模式：用于草稿-验证循环的 `EdgeCloudSpeculativeDecoding`、用于基于复杂性路由的 `IntelligentOffloading`，以及随时间改善的 `AdaptiveRouter`。

## 大规模并行

我们前面涵盖的并行策略——数据并行（第~\ref{chap:distributed-training-with-pytorch-ddp}章的 DDP、第~\ref{chap:scaling-with-fully-sharded-data-parallel-fsdp}章的 FSDP）、张量并行和流水线并行（第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章）——仍然是基础。但在 100K+ GPU 规模，出现需要新解决方案的新挑战。

### 通信革命

PyTorch 引入 Torchcomms 标志着分布式训练基础设施的显著演进。之前的通信栈虽然功能性，但在大规模下显出老态。Torchcomms 从头为 100K+ GPU 部署设计，带几个关键创新。

首先，它将通信原语与 PyTorch 核心解耦，允许研究人员独立迭代新的集合操作和后端。其次，它使用急切初始化和模型特定提示来在大规模下优化通信器和资源分配。第三，它支持异构硬件——单个训练作业中跨多个供应商和 GPU 代的混合部署。第四，它内建以前是事后考虑的容错机制。

Meta 的 NCCLX 后端（与 Torchcomms 一起发布），以及 AMD 平台的 RCCLX，通过像直接数据访问（DDA）这样的技术在 AllReduce 操作上提供 10-50% 的加速。这些不是渐进改进——它们可以意味着一个经济上可行的训练运行和一个不可行的之间的区别。

### 分层并行

在规模上，你不选择一种并行策略——你组合它们。一个典型的大型训练运行可能跨节点用数据并行、节点内用张量并行，跨模型阶段用流水线并行。算术很直接：如果你有 `world_size` 块 GPU，那么 `dp_size × tp_size × pp_size = world_size`。

艺术在于选择正确的组合。张量并行在节点内有低通信开销（NVLink 快）但跨节点有高开销。流水线并行可以隐藏通信延迟但引入气泡开销。数据并行扩展良好但需要梯度同步。最优配置取决于你的模型架构、硬件拓扑和批大小约束。

`code/parallelism.py` 中的 `HierarchicalParallelism` 类管理所有三个维度的进程组创建，确保每块 GPU 知道它对每种类型的并行属于哪些组。

### 长上下文的序列并行

随着上下文长度增长，序列并行变得必不可少。想法是将序列维度跨 GPU 拆分。每块 GPU 为它的本地序列块计算注意力，但注意力需要看到完整的键和值张量。这意味着从所有序列并行 rank all-gather K 和 V。

SGLang 的流水线并行报告强大的长提示数字：在 DeepSeek-V3.1 上 3.31 倍预填充吞吐量、在百万 token 上下文上高达 81% 的 TTFT 减少，以及 82.8% 的扩缩效率。那些技术解决内存和延迟；检索质量是一个单独的问题——许多模型在长文档上远低于它们宣传的上下文限制就损失准确性。

### 环形注意力

我们在第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章将环形注意力作为上下文并行的一部分介绍（见图~\ref{fig:seq-ctx-parallel}）。这里我们在扩展到百万 token 序列的背景下重新审视它。

环形注意力[^ring-attention] 为标准序列并行提供内存高效的替代。环形注意力不是 all-gather 完整的 K 和 V 张量（每块 GPU 需要 O(sequence_length) 内存），而是在 GPU 环中传递 K 和 V 块，在每步计算部分注意力分数。每块 GPU 从它的本地 Q、K、V 块开始。在每一轮，GPU 计算它们的本地 Q 和当前 K、V 之间的注意力，然后将 K 和 V 传给下一块 GPU。在 N−1 轮之后，每个 rank 都关注了所有键和值。

[^ring-attention]: Ring Attention with Blockwise Transformers for Near-Infinite Context. \url{https://arxiv.org/abs/2310.01889}

这以通信轮数换内存效率——在你受内存约束但有通信带宽富余时有用。在百万 token 规模，这种权衡变得越来越有吸引力：避免完整 K、V all-gather 的内存节省可以是装下一个工作负载和内存不足之间的区别。`code/parallelism.py` 中的 `RingAttention` 类实现这个模式。



## 当事情出错时：容错

在每 GPU 99.9% 的每日可靠性下，一个 100,000-GPU 集群应该预期每 24 小时约 100 次故障。每个设备存活一天的概率是 $0.999^{100,000} ≈ 3.7×10^{-44}$——实际上为零。在这个规模，故障是正常的运营条件，而非例外。

### 检查点挑战

传统检查点是同步的：停止训练、将状态保存到磁盘、恢复。当检查点需要几秒而训练运行持续几小时时这是可接受的。在规模上，检查点可以需要几分钟，训练运行持续几周。开销变得显著。

PyTorch 的分布式检查点（DCP）引入几个实现大约 6 倍更快检查点处理的优化：

- __缓存保存计划__：跨检查点重用张量元数据——形状和 dtype 不变，所以为什么重新计算它们？
- __后台保存__：通过将实际 I/O 移到单独的线程减少 GIL 竞争
- __增量保存__：只写自上次检查点以来改变的张量

`code/fault_tolerance.py` 中的 `AsyncCheckpointer` 实现这些想法。它维护一个从队列处理保存请求的后台线程，允许训练在检查点被写时继续。它还管理检查点保留，自动清理旧检查点以节省磁盘空间。

### 检测和处理故障

故障检测听起来简单——只检查 worker 是否响应——但细节很重要。你在宣布 worker 死亡之前等多久？太短，你会有来自网络打嗝的误报。太长，你浪费时间等待一个真正死亡的 worker。

`FailureDetector` 类使用一个心跳机制。worker 发送周期性心跳；缺失心跳触发故障回调。系统然后可以决定如何响应：用剩余 worker 继续、重启失败的 worker，或检查点并中止。

关键设计决策包括：

- __心跳间隔__：通常 1-5 秒，平衡响应性和开销
- __超时阈值__：通常 3-5 次缺失心跳后宣布故障
- __故障回调__：采取什么行动——记录、告警、检查点或中止

### 弹性训练

最复杂的方法是弹性训练，系统在不停止的情况下适应 worker 变化。worker 可以在训练期间加入或离开。当一个 worker 失败时，系统检查点、在剩余 worker 间重新分配工作，并继续。当一个新 worker 加入时，它接收一份工作。

这需要仔细的协调：

- __梯度平均__ 必须考虑变化的 worker 数量
- __数据加载__ 必须动态重新分配分片
- __学习率调度__ 可能需要为不同的 worker 计数调整
- __检查点__ 必须处理转换期间的部分状态

`code/fault_tolerance.py` 中的 `ElasticTrainer` 类处理这些关注点，提供故障时自动检查点和优雅降级。



## 超越文本：多模态分布式训练

如今捕获最多注意力的模型不是仅文本的——它们是多模态的。能看和读的视觉-语言模型（VLM）、处理视频的模型、理解音频与文本的系统。高效分布这些模型需要新的模式。

### 跨模态挑战

一个 VLM 通常有一个产生图像特征的视觉编码器（常常基于 ViT），和一个在关注那些图像特征时处理文本的语言模型。跨模态注意力——文本 token 关注视觉特征——是模态相遇的地方。

这种跨模态注意力可以用张量并行并行化，跨 GPU 分片注意力头。但有一个微妙之处：视觉编码器和语言模型可能有不同的最优并行策略。视觉编码器处理固定大小的图像并受益于与处理变长文本的语言模型不同的批大小。

`code/multimodal.py` 文件提供这些组件的实现：

- `VisionEncoder`：带 patch 嵌入和 transformer 块的 ViT 风格图像编码器
- `CrossModalAttention`：带张量并行支持的文本到视觉注意力
- `VisionLanguageModel`：结合视觉和语言组件的完整 VLM

### 处理多个模态

不同模态有不同的特征：

| 模态 | 大小 | 特征 |
|----------|------|-----------------|
| 图像 | 预处理后固定 | 批处理友好、可预测内存 |
| 文本 | 可变长度 | 需要填充、动态内存 |
| 视频 | 多帧 | 高内存、时间结构 |
| 音频 | 可变时长 | 时间结构、流式 |

`MultimodalDataParallel` 处理跨 worker 散播批次同时保持模态对齐。这比听起来更棘手——你需要确保对应于文本提示的图像最终在同一个 worker 上，即使批大小跨模态不同。



## 联邦学习：另一种范式

贯穿本书，我们假设所有 GPU 可以访问一个共享数据集——或至少存储在共同数据中心的它的分片。但如果数据不能移动呢？如果隐私法规、竞争关注或纯粹的物流使集中化不可能呢？这是联邦学习登场的地方。

### 联邦范式

联邦学习颠覆了脚本：它不是将数据带到模型，而是将模型带到数据。每个参与者——无论是智能手机、医院还是银行——在它的本地数据上训练，只共享模型更新，从不共享原始数据。一个中央服务器聚合这些更新以产生一个全局模型。这个范式自 2016 年就存在，它已成熟为隐私敏感领域的实用解决方案。

图~\ref{fig:federated-learning} 展示这个架构。顶部的中央服务器协调训练并持有全局模型。底部的四个客户端——带患者记录的医院、带交易数据的银行、带用户行为的移动应用，以及带传感器读数的物联网设备——每个在它们的私有数据上本地训练。蓝色箭头显示服务器分发当前模型；绿色箭头显示客户端发回它们的更新。关键点，在底部突出：数据从不离开客户端。只有梯度和权重穿越网络。

![联邦学习：数据留在客户端](img/federated_learning_zh.png){#fig:federated-learning .block width=80% align=center}

规范算法是 FedAvg[^fedavg]。每一轮，服务器将当前模型发送到客户端的一个子集。客户端本地训练几个 epoch，然后发回更新的权重。服务器平均这些权重以产生一个新的全局模型。虽然听起来简单，这个协议已被证明非常有效——该领域在它之上构建了许多改进。

[^fedavg]: Communication-Efficient Learning of Deep Networks from Decentralized Data. \url{https://arxiv.org/abs/1602.05629}

### 联邦学习在哪里蓬勃发展

联邦学习在特定垂直领域找到强大的采用：

__医疗__：医院可以协作训练诊断模型而不共享患者记录。像 NVIDIA FLARE[^flare] 这样的项目实现多机构医学影像研究，同时维持 HIPAA 合规。一个跨 20 家医院训练的模型看到比任何单个机构能提供的更多样的病理。

[^flare]: NVIDIA FLARE: Federated Learning Application Runtime Environment. \url{https://github.com/NVIDIA/NVFlare}

__移动键盘__：Google 的 Gboard 使用联邦学习来改善下一词预测而不上传用户键入的内容。模型从数百万设备学习，同时将文本保持在设备上。这是联邦学习最早的大规模生产部署之一。

__金融服务__：银行可以在欺诈检测模型上协作而不共享交易数据。每个机构贡献来自它的客户群的模式；组合模型捕获没有单个银行的数据会揭示的欺诈。

__边缘物联网__：工业传感器、自动驾驶车辆和智能设备生成集中化昂贵或不切实际的数据。联邦学习实现模型改善而无需大规模数据传输。

### 异构性挑战

联邦学习面临传统分布式训练中不存在的挑战：

__非 IID 数据__：在数据中心，你可以打乱数据以确保每块 GPU 看到代表性样本。在联邦学习中，每个客户端的数据反映它的本地分布。在日本训练的键盘模型看到与在巴西训练的不同的文本。这种统计异构性可能造成模型发散和慢收敛。

__系统异构性__：客户端有巨大不同的计算能力。一部旗舰智能手机和一部三年前的预算手机不能以相同速度训练。一些客户端可能因电池约束或网络问题在轮中途退出。系统必须对落后者和部分参与稳健。

__通信约束__：与 NVLink 连接的 GPU 不同，联邦客户端通过移动网络或互联网通信。带宽有限且昂贵。像梯度压缩、量化和不频繁同步这样的技术变得必不可少——不是为了性能，而是为了可行性。

### 现代联邦技术

研究用越来越复杂的方法解决这些挑战：

__FedProx__[^fedprox] 向本地目标添加一个近端项，防止客户端漂移得离全局模型太远。这在数据高度非 IID 时稳定训练。

[^fedprox]: Federated Optimization in Heterogeneous Networks. \url{https://arxiv.org/abs/1812.06127}

__Scaffold__[^scaffold] 使用控制变量来纠正客户端漂移，在异构数据上比 FedAvg 实现更快的收敛。

[^scaffold]: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. \url{https://arxiv.org/abs/1910.06378}

__差分隐私__ 可以叠加在联邦学习之上以提供正式的隐私保证。通过向更新添加校准的噪声，即使聚合模型也只揭示关于任何个人数据的有限信息。这在医疗和金融中特别重要。

__安全聚合__[^secagg] 确保服务器只看到客户端更新的和，而非单个贡献。即使一个被入侵的服务器也不能提取任何单个客户端的模型更新。

[^secagg]: Practical Secure Aggregation for Privacy-Preserving Machine Learning. \url{https://eprint.iacr.org/2017/281}

### LLM 的联邦学习

将联邦学习应用于大语言模型呈现独特的挑战。一个 7B 参数模型不能装进智能手机，即使微调也需要显著的内存。近期工作探索几种方法：

__带 LoRA 的联邦微调__：客户端不更新所有参数，而是训练低秩适配器。这减少通信（只交换适配器权重）和内存（适配器很小）。FedIT[^fedit] 和类似方法使联邦 LLM 微调实用。

[^fedit]: Federated Instruction Tuning of LLMs with Domain Coverage Augmentation. \url{https://arxiv.org/abs/2409.12568}

__拆分学习__：模型在客户端和服务器之间拆分。客户端在它们的数据上运行早期层并将激活值（而非原始数据）发送到服务器，服务器完成前向传播。这实现训练大型模型而无需客户端持有完整模型。

__设备端个性化__：每个设备维护一个个性化版本，而非训练单个全局模型。全局模型提供起点；本地微调适应个别用户。这对应该反映用户偏好的助手特别相关。

### Flower 框架

Flower[^flower] 已成为领先的开源联邦学习框架。它提供：

- 定义联邦训练循环的简单 API
- 支持各种聚合策略（FedAvg、FedProx 等）
- 与 PyTorch、TensorFlow 和 JAX 集成
- 用于开发和研究的模拟能力
- 生产部署工具

[^flower]: Flower: A Friendly Federated Learning Framework. \url{https://flower.ai/}

框架抽象掉大部分复杂性，让研究人员专注于算法而非基础设施。一个基本的联邦训练设置只需要几十行代码。

### 何时考虑联邦学习

联邦学习不是传统分布式训练的替代——它是特定情况的工具：

| 何时考虑联邦学习 | 何时坚持传统训练 |
|----------------------------------|-------------------------------------|
| 数据不能离开其来源（隐私、法规） | 数据可以集中化 |
| 数据自然分布（移动、边缘） | 你控制基础设施 |
| 参与者互不信任 | 单个组织训练 |
| 通信昂贵 | 有高带宽互连可用 |

联邦学习的开销——慢收敛、通信约束、异构性挑战——意味着当你可以简单地集中数据时它不是正确的选择。但当隐私或物流使集中化不可能时，联邦学习实现否则不会发生的协作。



## 智能体 AI：下一个前沿

也许最令人兴奋的近期发展是智能体 AI 的兴起——不只响应提示而是规划、使用工具和采取行动的系统。这些系统将分布式推理推向新方向。

### 为什么智能体需要分布

考虑当一个智能体决定调用一个工具时会发生什么。工具可能是一个代码解释器、一个 web 浏览器、一个数据库查询或一个 API 调用。每个工具有不同的资源需求和延迟特征。一些工具是有状态的，需要在特定 worker 上运行。一些是极易并行的；其他是顺序瓶颈。

多智能体系统添加另一层。多个智能体可能在一个任务上协作，每个有不同的能力。它们需要通信、协调，有时不同意。这种通信可以在单个节点内或跨分布式系统发生。

长推理链——像 o1 和 DeepSeek-R1[^deepseek-r1] 这样的模型产生的那种——需要在许多步骤上持续推理。每步可能涉及工具调用、内存查找或与其他智能体的协调。推理不是单个前向传播；它是一个可能运行几分钟的扩展计算。

[^deepseek-r1]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. \url{https://arxiv.org/abs/2501.12948}

### 多智能体模式

你如何编排多个智能体协同工作？三种模式已成为多智能体系统的主力，每种适合不同的问题结构。

![三种常见的多智能体协调模式](img/multi_agent_patterns_zh.png){#fig:multi-agent-patterns .block width=95% align=center}

图~\ref{fig:multi-agent-patterns} 展示这些模式。在 __顺序处理__ 中，智能体形成一个流水线：一个提取信息，下一个对它推理，第三个格式化响应。每个智能体的输出成为下一个智能体的输入。这在任务自然分解成有清晰交接的阶段时工作良好。

__并行处理__ 采取不同的方法——多个智能体同时处理同一任务。也许你想要多样的视角：一个智能体可能找到一个创意解决方案，而另一个找到一个安全的。或你在竞赛策略，取先完成的。结果流向一个组合或在它们之间选择的聚合器。

__分层处理__ 镜像人类团队常常如何工作。一个协调器接收任务、将它分解成子任务，并委派给专门的 worker。worker 报告回来（图中的绿色虚线箭头），协调器将它们的结果综合成最终输出。这对自然分解的复杂任务大放异彩——协调器处理策略，而 worker 处理战术。

`code/agentic_inference.py` 文件实现所有三种模式，以及基于任务特征在它们之间选择的路由逻辑。

### 分布式工具执行

当一个智能体决定调用一个工具时，那个工具实际上在哪里运行？不是所有工具都平等。一个代码解释器需要带适当安全隔离的沙盒环境——你不能只在任何地方运行任意代码。一个 web 浏览器可能需要 GPU 加速用于渲染。另一方面，一个简单的计算器可以在任何有空闲周期的 worker 上运行。

`code/agentic_inference.py` 中的 `DistributedToolExecutor` 处理这个路由。工具用可选的 worker 约束注册自己："我需要沙盒"、"我需要 GPU"或"我是无状态的，在任何地方运行我"。当一个智能体请求一个工具时，执行器找到一个合适的 worker、分派调用，并返回结果。

这开辟了复杂的放置策略。频繁使用的工具跨 worker 复制用于负载均衡。有状态工具——像数据库连接或浏览器会话——固定到特定 worker，使状态在调用之间持久。资源密集型工具隔离到专用 worker，防止它们饿死其他操作。执行器跟踪执行统计，随时间学习哪些放置工作最好。

### 推理智能体

最复杂的智能体不只响应——它们思考。给定一个复杂问题，它们分解它、尝试方法、观察结果并调整。这是像 o1 和 DeepSeek-R1 这样的推理智能体的领域，其中推理不是单个前向传播而是一个可能跨越几十步的扩展审议。

`code/agentic_inference.py` 中的 `ReasoningAgent` 捕获这个模式。它维护一个推理轨迹——思考、工具调用和观察的运行记录。在每一步，智能体面临一个选择：更深入地思考当前状态、调用一个工具收集信息，或提交一个最终答案。当它调用一个工具时，结果成为一个新观察，馈入下一个思考。循环继续直到智能体足够自信回答，或达到步骤限制。

这是生产推理系统所做的简化版本，但它捕获了本质动态：思考和行动之间、内部审议和外部观察之间的相互作用，使智能体能解决对单次推理传播太复杂的问题。



## 规模经济学

随着分布式 AI 成熟，经济考量变得越来越重要。"只是投入更多 GPU"的日子正在结束；效率和成本优化现在是一等关注。

### 工作负载分布

企业工作负载分布显示没有单一主导类别：

| 工作负载类型 | 份额 |
|--------------|-------|
| 大规模推理 | 34.6% |
| 基础模型训练 | 24.9% |
| 领域特定训练 | 23.3% |
| 微调 | 17.2% |

这种多样性意味着基础设施必须灵活——为一种工作负载类型优化的不会服务全范围的需求。

组织正在采用混合策略：

- __公共云__：处理需求不可预测的弹性和可变工作负载
- __本地__：服务可预测成本重要的大批量生产推理
- __近边缘数据中心__：处理延迟敏感的、面向用户的 AI 服务

### 资源管理

当工作负载变化时，动态资源分配至关重要。`code/resource_management.py` 中的 `AdaptiveGPUAllocator` 基于以下估计 GPU 需求：

- 模型大小（内存需求）
- 批大小（吞吐量需求）
- 作业优先级（业务重要性）

它为不能立即调度的作业维护一个队列，并在资源可用时处理队列。

### 多租户 GPU 共享

`MultiTenantGPUScheduler` 实现时间切片 GPU 共享：

- 优先级加权轮询调度
- 跨租户的公平分配
- 待处理工作的队列管理
- 高优先级作业的抢占

这允许多个团队或服务公平地共享 GPU 资源，优先级反映业务重要性。

### 梯度压缩

在规模上，通信开销可以主导计算时间。`code/resource_management.py` 中的 `GradientCompression` 减少这个开销：

__Top-k 稀疏化__[^topk-sgd]：只保留 k 个最大的梯度值。用 k=1% 的梯度，你实现 100 倍压缩。误差反馈——为下一次迭代累积丢弃的梯度——保持收敛。

[^topk-sgd]: Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training. \url{https://arxiv.org/abs/1712.01887}

__量化__：将精度减少到 8 位。这实现 4 倍压缩，对训练动态影响最小。可以与稀疏化结合以获得更高的压缩。

关键洞见是梯度高度可压缩。大多数梯度值很小，对学习贡献很少。通过将通信聚焦于最大的值并累积其余，你可以在不牺牲收敛的情况下大幅减少带宽需求。



## 为未来做准备

### 要发展的技能

__MoE 和稀疏架构__：随着 MoE 成为默认架构，理解专家路由策略（TopK、基于 ReLU）、负载均衡机制和推理时弹性专家扩展越来越必不可少。

__边缘-云协调__：推测解码实现、移动的内存带宽优化，以及量化技术（GPTQ、AWQ）开辟了直到最近才实用的新部署选项。

__大规模通信__：Torchcomms API 和 NCCLX/RCCLX 后端、异步检查点策略，以及异构硬件部署实现更大的训练运行和更高效的基础设施利用。

__推理优化__：RadixAttention 和 PagedAttention 内部、长上下文的流水线并行，以及多租户 GPU 调度改善服务效率并降低成本。

__智能体系统__：多智能体编排模式、分布式工具执行和推理链优化代表分布式 AI 遇上复杂问题解决的下一个应用前沿。

### 保持最新

该领域快速移动——今天看起来前沿的技术可能在六个月内是标准实践。保持最新需要围绕几个关键信息源建立习惯。

对于日常学习，arXiv 不可或缺。cs.DC 和 cs.LG 中的新论文每天出现，最好的工作常常在会议发表前几个月出现在这里。为像"分布式训练"、"推理优化"和"混合专家"这样的关键词设置警报。开源项目同样重要：vLLM、SGLang、DeepSpeed 和 Megatron-LM 是理论遇上实践的地方。观察它们的发布说明和 GitHub 讨论——那是你会学到在规模上实际有效的东西。来自 xAI、Anthropic、Google 和 Meta 团队的行业博客定期发布揭示论文中从未找到的实用洞察的技术深潜。不要忽视社区：Hugging Face 论坛、PyTorch Discuss 和 SGLang Discord 是从业者分享战争故事并一起调试棘手问题的地方。

对于更深入的探索，会议仍然必不可少。像 NeurIPS、ICML 和 ICLR 这样的研究场所是新算法和架构首次亮相的地方——特别注意关于高效 ML 和大规模系统的研讨会。像 MLSys、OSDI 和 SOSP 这样的系统会议专注于基础设施、通信优化和生产部署；如果你关心让事情变快，这些是必读。像 GTC 和 PyTorch Conference 这样的行业活动是供应商宣布新硬件和框架的地方，给你未来 12-18 个月内到来的东西的预览。

本章末尾的参考文献部分为贯穿全文提及的关键论文和项目提供 URL。



## 小结

本章探索了分布式 AI 的当前状态和塑造其未来的趋势：

1. __向推理的大转变__：推理工作负载现在消耗超过 55% 的 AI 基础设施支出，训练成为总计算的更小部分。这重塑基础设施优先级和优化目标。

2. __MoE 主导__：像 DeepSeek-V3 这样的架构证明 671B 参数模型可以用 550 万美元训练，每 token 只有 37B 活跃参数。理解专家路由、负载均衡和并行至关重要。

3. __设备端 AI 是实用的__：亚十亿参数模型、4 位量化和像逐层嵌入这样的技术使 LLM 在移动设备上成为可能。内存带宽而非计算是瓶颈。

4. __100K+ GPU 规模__：新的通信 API（Torchcomms、NCCLX/RCCLX）和异步检查点实现前所未有的规模训练。异构硬件支持越来越重要。

5. __容错必不可少__：在规模上，故障是预期的——在 100K GPU 集群中大约每天 100 次。弹性训练和异步检查点是强制性的，而非可选的。

6. __多模态和智能体__：VLM 和多智能体系统需要跨模态注意力和工具执行的新分布式模式。这些代表下一个应用前沿。

7. __联邦学习作为替代范式__：当数据因隐私或监管约束不能集中化时，联邦学习提供一个成熟的替代。像 FedProx 和安全聚合这样的技术解决非 IID 数据和不受信任参与者的独特挑战。

8. __推理引擎演进__：SGLang 用 RadixAttention 实现 16,215 tok/s；vLLM 的 PagedAttention 将 KV 缓存浪费减少到 4% 以下。两者都支持多样的硬件平台。

本书涵盖的技术——DDP、FSDP、DeepSpeed、Megatron-LM、vLLM、SGLang——仍然是基础，但图景继续快速演进。电力和制冷约束，而非硅可用性，现在是主要瓶颈。

如果你从第 1 章一直读过来，你从大多数从业者所在的地方开始：一个超出一块 GPU 的模型，以及关于硬件、并行、训练框架、推理引擎和生产部署的一系列日益增长的问题。本章将那些部分置于接下来的东西中——大规模 MoE、设备端推理、容错，以及一个推理而非训练驱动大部分基础设施支出的世界。工具会不断变化；你一路建立的系统思维不会。

分布式 AI 的未来正在现在被书写，由推动边界的研究人员和大规模部署的从业者。完成本书后，你更有能力跟随那项工作——并帮助发明它。

## 有用的链接

__MoE 架构__

- DeepSeek-V3 技术报告：\url{https://arxiv.org/abs/2412.19437}
- LatentMoE: Toward Optimal Accuracy per FLOP：\url{https://arxiv.org/abs/2601.18089}
- MoSE: Mixture of Slimmable Experts：\url{https://arxiv.org/abs/2602.06154}
- Elastic MoE: Inference-Time Expert Scaling：\url{https://arxiv.org/abs/2501.03140}
- ReMoE: Fully Differentiable MoE with ReLU Routing：\url{https://arxiv.org/abs/2412.14711}

__通信与规模__

- NVIDIA NCCL：\url{https://developer.nvidia.com/nccl}
- Torchcomms API：\url{https://pytorch.org/blog/torchcomms/}
- RCCLX for AMD Platforms：\url{https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/}
- PyTorch Distributed Checkpoint：\url{https://pytorch.org/docs/stable/distributed.checkpoint.html}
- Async Checkpointing Improvements：\url{https://pytorch.org/blog/6x-faster-async-checkpointing/}
- Deep Gradient Compression：\url{https://arxiv.org/abs/1712.01887}

__推理引擎__

- SGLang (RadixAttention)：\url{https://arxiv.org/abs/2312.07104}
- SGLang Pipeline Parallelism：\url{https://lmsys.org/blog/2026-01-15-chunked-pipeline/}
- SGLang 文档：\url{https://sgl-project.github.io/}
- vLLM (PagedAttention)：\url{https://arxiv.org/abs/2309.06180}
- vLLM 文档：\url{https://docs.vllm.ai/}
- Speculative Decoding：\url{https://arxiv.org/abs/2211.17192}

__内存高效注意力__

- Multi-head Latent Attention (DeepSeek-V2)：\url{https://arxiv.org/abs/2405.04434}
- Ring Attention：\url{https://arxiv.org/abs/2310.01889}
- FlashAttention-2：\url{https://arxiv.org/abs/2307.08691}

__设备端 AI 和量化__

- GPTQ: Post-Training Quantization：\url{https://arxiv.org/abs/2210.17323}
- AWQ: Activation-aware Weight Quantization：\url{https://arxiv.org/abs/2306.00978}
- KV Cache Compression：\url{https://arxiv.org/abs/2405.12981}
- Google LiteRT：\url{https://ai.google.dev/edge/litert}
- Gemma 3n：\url{https://ai.google.dev/gemma/docs/gemma-3n}

__推理和智能体 AI__

- DeepSeek-R1：\url{https://arxiv.org/abs/2501.12948}
- ReAct: Reasoning and Acting in Language Models：\url{https://arxiv.org/abs/2210.03629}

__容错和弹性训练__

- PyTorch Elastic：\url{https://pytorch.org/docs/stable/elastic/run.html}
- TorchSnapshot：\url{https://pytorch.org/torchsnapshot/}

__联邦学习__

- FedAvg: Communication-Efficient Learning from Decentralized Data：\url{https://arxiv.org/abs/1602.05629}
- FedProx: Federated Optimization in Heterogeneous Networks：\url{https://arxiv.org/abs/1812.06127}
- SCAFFOLD: Stochastic Controlled Averaging：\url{https://arxiv.org/abs/1910.06378}
- Secure Aggregation for Privacy-Preserving ML：\url{https://eprint.iacr.org/2017/281}
- Flower Framework：\url{https://flower.ai/}
- NVIDIA FLARE：\url{https://github.com/NVIDIA/NVFlare}
- FedIT: Federated Instruction Tuning of LLMs：\url{https://arxiv.org/abs/2409.12568}

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
