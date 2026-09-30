# 附录X：前言、序言和作者简介 {-}

*分布式 AI 系统的背景与展望*

## 本书背景

分布式人工智能系统已经从一个学术研究课题演变为生产环境中的刚需。如今，几乎每一个参数量超过十亿的语言模型，都需要跨多个 GPU、多个节点、甚至多个数据中心进行分布式训练。从 GPT-3 到 LLaMA、Gemini 以及其他前沿模型，分布式并行都已成为大规模 AI 的基石。

本书旨在弥合学术文献与工程实践之间的鸿沟。学术论文提出了新的并行策略，却往往缺少落地实现的细节；产品文档聚焦于具体工具，却常常不解释背后的基本原理。本书试图在两者之间架起桥梁：既深入讲解这些技术为何有效、如何生效，又提供可以直接运行的代码示例。

全书从单 GPU 训练讲起（第 1 章），系统性地覆盖每一个扩展难题。硬件基础（第 2 章）帮助你理解为什么分布式训练是必要的。数据并行（第 3 章）与状态分片（第 4 章）应对相对简单的场景；当它们不再够用时，你就需要转向计算分片（第 5 章）。在推理方面，vLLM（第 6 章）与生产服务栈（第 9 章）则展示了真实系统的样貌。

## 核心概念回顾

贯穿全书的三个核心理念：

**1. 各种并行策略是正交的。** 数据并行、张量并行、流水线并行等是相互独立的维度，可以自由组合。你不是“二选一”，而是根据自身约束把它们搭配使用。一个 70B 模型可能在节点内使用张量并行（借助 NVLink 的高带宽），在节点间使用流水线并行（容忍更高的延迟），同时全程使用 FSDP 来提升显存效率。

**2. 通信与计算的权衡无处不在。** 数据并行降低显存占用，却增加了通信开销（梯度的 AllReduce）。状态分片进一步节省显存，但需要 all-gather 参数。张量并行分摊了计算负载，但每一层都要做 all-reduce。序列并行分摊了激活值显存，代价同样是通信。天下没有免费的午餐——每一项优化都伴随着取舍。

**3. 量化指标很重要，但只看单一指标会产生误导。** GPU 利用率可能很高，但如果大部分时间都花在通信上，你实际上是受通信瓶颈制约的。模型 FLOPS 利用率（MFU）才真正反映计算效率。AllReduce 实测吞吐量与理论峰值之间的差距，则揭示了通信瓶颈。务必在真实硬件上做性能剖析——凭直觉猜测的假设往往是错的。

## 后续实践路径

本书的代码示例涵盖了从最简单的单 GPU 训练到多节点混合并行的方方面面。我们推荐的学习路径如下：

1. **从单 GPU 起步。** 先确保你能在一块 GPU 上把模型训起来，这样既建立了性能基线，也验证了代码的正确性。

2. **加入数据并行。** 用 DDP 扩展到 2–4 块 GPU，对大多数场景来说这就足够了。

3. **显存溢出（OOM）时改用 FSDP。** FSDP2 与 DDP 的代码结构相同，但会对参数和优化器状态进行分片。

4. **单层成为瓶颈时引入张量并行。** 这需要更细致的配置，但对超大模型至关重要。

5. **持续剖析并迭代。** 用 `torch.profiler` 弄清时间都花在哪里。受通信制约？调整 bucket 大小或开启预取（prefetch）。受计算制约？优化算子或尝试 FP8 精度。受显存制约？开启激活重计算（activation checkpointing）或 CPU offload。

6. **验证多节点场景。** 多节点训练会带来额外的故障模式（网络问题、同步超时），要在进入生产之前尽早发现它们。

## 常见陷阱

以下是我们在大规模训练实践中反复踩过的坑。

**忘记为 DistributedSampler 设置 epoch。** 不调用 `sampler.set_epoch(epoch)`，每个 epoch 看到的数据顺序都一样，症状看起来很像过拟合或训练效果不佳。

**把 world size 写死。** 假设训练固定跑在某个数量的 GPU 上，是一个常见错误。请使用 `dist.get_world_size()`，而不要硬编码成 `8`。

**在小模型上误判 DDP 与 FSDP 的取舍。** FSDP 会引入额外通信开销。对于小模型，由于通信成本占比较高，用 DDP 反而可能更快。请先做小规模测试。

**混合精度下忘记做梯度缩放。** FP16 容易导致梯度下溢。请使用 `GradScaler`（配合 FP16），或者直接改用 BF16（不存在下溢风险）。

**多节点场景下忽视网络配置。** InfiniBand 被禁用，或选错了网络接口？训练可能会慢上 10 倍。请设置 `NCCL_SOCKET_IFNAME` 指向集群的高速网络接口。

**检查点体积失控。** 分布式训练的检查点可能非常庞大。请使用分片检查点（DCP API），让每个 rank 只写自己那一份分片。

## 生态与工具

分布式训练涉及众多工具与框架。并不存在“唯一正确的选择”——最合适的工具取决于你的具体约束。

**PyTorch DDP 与 FSDP** 是 PyTorch 原生方案，新项目推荐使用 FSDP2。纯 PyTorch 技术栈，与 `torch.compile` 集成良好。

**DeepSpeed** 是微软主导的社区项目，集成了 ZeRO（状态分片）、高级 offload、MoE 支持等大量优化。对于需要超出 PyTorch 原生能力的团队非常有价值。

**Megatron-Core** 是 NVIDIA 面向大规模训练的方案，原生支持张量并行、流水线并行和序列并行。Megatron-FSDP 则在 Megatron 之上叠加 FSDP，进一步提升显存效率。

**Colossal-AI** 是一个一体化框架，打包了混合并行（DP/TP/PP）、ZeRO 式分片以及异构内存管理（GPU/CPU/NVMe），适合复杂的并行配置。

**vLLM** 是分布式推理的事实标准，支持张量并行、面向 KV 缓存的 PagedAttention，吞吐量很高，是 LLM 服务的必备工具。

**Ray Train** 及 **Ray 上的分布式 PyTorch** 为集群上的弹性分布式训练提供了抽象层，简化了机器故障处理和工作流编排。

在实际部署中，生产服务栈（第 9 章）分为多个层次：推理引擎（vLLM、TensorRT-LLM）负责模型执行；API 网关与负载均衡器（如 vLLM serving、Ray Serve、KServe）负责路由请求；监控与日志系统（Prometheus、Grafana、ELK）负责跟踪系统健康状况。

## 未来方向

分布式 AI 正在飞速发展。当你读到本书时，可能已有更新的技术涌现。

**编译器优化** 正日益成为关键。XLA、TorchScript、`torch.compile` 等编译器能够自动优化通信模式、重排算子以提升缓存命中率，甚至跨设备融合计算。随着技术成熟，越来越多的模型会逐步采用编译器加持。

**自动并行** 仍在研究之中。给定一个模型和一套硬件集群，系统可以自动挑选最优的并行策略。这非常困难（搜索空间极其庞大），但早期结果令人鼓舞。

**长上下文推理** 正在推动推理并行的创新。Ring Attention、上下文并行（Context Parallelism）和 DeepSpeed-Ulysses 是处理长序列的不同思路，目前尚未出现统一的标准方案。

**MoE 扩展。** 随着模型越做越大，混合专家（Mixture-of-Experts）越发重要。训练 MoE 的系统已相当成熟，但推理效率仍是一个开放问题。

**跨模态协同** 正在兴起。训练对象不再局限于文本——视频、音频、图像如今都可能是模型的一部分。这既改变了并行策略（不同模态的计算特征各异），也改变了系统设计。

**隐私保护的分布式学习** 仍处于早期阶段。联邦学习、差分隐私、同态加密如何与大规模分布式训练结合？这对某些应用（医疗、金融）至关重要，但目前大多数生产系统并未考虑。

## 致谢

本书建立在过去十五年积累的研究与实践之上：从 Hinton 与 Alex Krizhevsky 的 AlexNet，到 Krizhevsky 等人的 ImageNet 图像识别、Srivastava 等人的 ResNet、Vaswani 等人的 Transformer、Radford 等人的 GPT 系列，再到近期的 LLaMA、Gemini 及其他模型。

感谢为 Megatron-LM、PyTorch DDP/FSDP、DeepSpeed、vLLM 以及众多其他项目作出贡献的开源社区，正是他们的工作让现代分布式 AI 得以普及。

## 延伸阅读

以下资源可供你更深入地探索各个专题。

**分布式训练基础**

- Distributed Data Parallel Training in PyTorch（PyTorch 官方 DDP 教程）
- NVIDIA 分布式训练实践指南
- DeepSpeed 文档与教程

**并行策略**

- Megatron-LM 论文与代码
- vLLM 论文与 GitHub 仓库
- 面向超长序列的 Ring Attention

**前沿应用**

- GPT-3、GPT-4 论文及相关系统研究
- LLaMA 与 LLaMA 2 的扩展定律（scaling laws）
- DeepSeek-V3 的 MoE 架构与并行

**生产系统**

- Ray Train 文档
- KServe 与 Seldon Core
- Prometheus 与 Grafana 监控

请保持学习。分布式 AI 是一个持续演进的领域，要密切关注新论文、新基准和新工具。多读源码——开源项目是最好的学习资料。最重要的是，动手实验：每一个训练脚本、每一个集群、每一种工作负载都各不相同。剖析你自己的系统，建立你自己的基准，再据此优化。

这正是构建分布式 AI 系统所需要做的工作。今天的前沿模型，明天就会被超越。但那些底层原理——并行策略的取舍、通信与计算的平衡、系统的度量与优化——将长久留存。掌握了这些原则，你就能为当下和未来构建高效的大规模 AI 系统。

## 中英术语对照表

| 中文 | English |
| --- | --- |
| 分布式 AI 系统 | Distributed AI Systems |
| 分布式训练 | Distributed Training |
| 分布式推理 | Distributed Inference |
| 数据并行 | Data Parallelism |
| 张量并行 | Tensor Parallelism |
| 流水线并行 | Pipeline Parallelism |
| 序列并行 | Sequence Parallelism |
| 上下文并行 | Context Parallelism |
| 专家并行 | Expert Parallelism |
| 混合并行 | Hybrid Parallelism |
| 状态分片 | State Sharding |
| 计算分片 | Computation Sharding |
| 参数分片 | Parameter Sharding |
| 完全分片数据并行 | Fully Sharded Data Parallel (FSDP) |
| 分布式数据并行 | Distributed Data Parallel (DDP) |
| 梯度同步 | Gradient Synchronization |
| 集合通信 | Collective Communication |
| 全归约 | AllReduce |
| 全收集 | AllGather |
| 归约散射 | ReduceScatter |
| 广播 | Broadcast |
| 进程组 | Process Group |
| 世界规模（进程总数） | World Size |
| 显存 / 内存 | Memory (VRAM / RAM) |
| 显存溢出 | Out of Memory (OOM) |
| 激活值 | Activations |
| 激活重计算 | Activation Checkpointing |
| 优化器状态 | Optimizer States |
| 梯度 | Gradient |
| 混合精度 | Mixed Precision |
| 梯度缩放 | Gradient Scaling |
| 梯度下溢 | Gradient Underflow |
| 梯度累积 | Gradient Accumulation |
| 检查点 | Checkpoint |
| 分片检查点 | Sharded Checkpoint |
| 通信瓶颈 | Communication Bottleneck |
| 通信与计算重叠 | Communication-Computation Overlap |
| 预取 | Prefetch |
| 卸载 | Offload |
| 性能剖析 | Profiling |
| 性能基线 | Baseline |
| 吞吐量 | Throughput |
| 延迟 | Latency |
| 模型 FLOPS 利用率 | Model FLOPS Utilization (MFU) |
| 扩展定律 | Scaling Laws |
| 混合专家 | Mixture-of-Experts (MoE) |
| 键值缓存 | KV Cache |
| 连续批处理 | Continuous Batching |
| 推理引擎 | Inference Engine |
| 负载均衡 | Load Balancing |
| 弹性训练 | Elastic Training |
| 容错 | Fault Tolerance |
| 互连 | Interconnect |
| 集群 | Cluster |
| 节点 | Node |
| 作业调度器 | Job Scheduler |
| 前沿模型 | Frontier Model |
| 大语言模型 | Large Language Model (LLM) |
| 算子 | Operator / Kernel |
| 编译器优化 | Compiler Optimization |
| 自动并行 | Auto-Parallelism |
| 联邦学习 | Federated Learning |
| 差分隐私 | Differential Privacy |
| 同态加密 | Homomorphic Encryption |



