# 第 2 章：GPU 硬件、网络拓扑与并行策略 {-}

*深入理解硬件拓扑结构与分布式 AI 的核心并行策略*

> 真正的创新不仅仅在于芯片本身，而在于贯穿整个技术栈的系统工程。  
> —— 黄仁勋（Jensen Huang）

**核心代码速查**

- `nvidia-smi`：监控 GPU 状态与利用率的命令行核心工具
- `torch.cuda.device_count()`：获取当前环境可用的 CUDA 设备数量
- `torch.cuda.get_device_name()`：获取指定 CUDA 设备的硬件型号名称
- `torch.cuda.is_available()`：检查当前系统是否支持 CUDA 加速
- `torch.distributed.get_world_size()`：获取分布式进程组中的总进程数
- `torch.distributed.get_rank()`：获取当前进程在分布式组中的 Rank 编号
- `torch.distributed.get_backend()`：获取当前通信后端名称（如 'nccl', 'gloo'）
- `nvidia-ml-py`：用于在 Python 中精确查询 NVIDIA GPU 底层硬件指标的库
- `ibstat`：查询 InfiniBand 高速网卡状态与连接速率的底层命令


## 算力体系：AI 集群与核心评估指标

在第 1 章中，我们阐明了现代 AI 必须走向分布式的根本原因——大模型规模已经彻底突破了单卡显存的物理容量，且算力需求与硬件能力之间的剪刀差正在持续拉大。现在，我们必须深入剖析支撑分布式训练的硬件物理底座。在展开具体的 GPU 架构之前，让我们先站在系统全局视角，理解我们真正在构建的对象：能够输出海量吞吐的 **AI 计算集群（AI Clusters）**。在分布式训练中，数字规模决定了一切——当你训练一个 70B 参数的模型时，你不是在调动几张 GPU，而是在协调成百上千张 GPU；而它们之间的物理互联拓扑，直接决定了你的训练任务究竟能在几天内收敛，还是需要漫长的数周。

### 什么是算力（Computational Power）？

算力，即系统的计算吞吐能力，衡量的是一个计算系统每秒钟能够执行的运算次数。在 AI 负载中，我们最关心的核心指标是**每秒浮点运算次数（FLOPS，Floating-Point Operations Per Second）**。算力的量级呈指数级跃升：单张现代 GPU（如 NVIDIA H200）在 FP16/BF16 精度下能提供约 1,000 TFLOPS（TeraFLOPS，$10^{12}$ 次运算/秒）的密集算力；一个由 1,000 张此类 GPU 组成的集群，理论总算力约为 1,000 PFLOPS（PetaFLOPS，$10^{15}$ 次运算/秒），即 1 EFLOPS（ExaFLOPS，$10^{18}$ 次运算/秒）：$1000 \text{ GPUs} \times 1000 \text{ TFLOPS} = 10^{6} \text{ TFLOPS} = 1000 \text{ PFLOPS} = 1 \text{ EFLOPS}$。

![GPU 显存容量 vs 模型显存需求增长趋势对比](img/computational_growth_gap.png){#fig:computational-growth-gap .wrap width=60% align=top-right lines=15}

但必须强调的是：单纯的峰值 FLOPS 数值并不能代表实际性能。峰值吞吐量严重依赖于**数值计算精度**——传统科学计算（HPC）依赖高精度的 FP64，深度学习基准测试常以 FP32 为基线，现代大模型训练主流采用 FP16/BF16，而推理与量化部署则进一步下探到 FP8 乃至 FP4（见第~\ref{chap:introduction-to-modern-distributed-ai} 章精度对比表）。当厂商宣称“该集群算力达 500 PFLOPS”时，务必先追问是在何种精度下的标称值：HPC 集群通常引用 FP64 数据，而 AI 集群则通常指 FP16/BF16。同样的硬件在不同数值精度下的标称算力可能会相差数倍乃至数十倍。

AI 领域对算力增长的渴求是极为惊人的。近年来大语言模型所需的计算资源每隔几年就会呈数量级爆炸式增长，而同期单卡硬件算力与显存的物理增长速度仅有约 3 倍左右。正是这一巨大的硬件剪刀差，使得分布式训练不再是一个可选的锦上添花项，而是**在合理工程周期内完成现代大模型训练的唯一可行途径**。

如 @fig:computational-growth-gap 所示，模型参数量的膨胀速度呈陡峭的指数级上升，而单 GPU 的物理显存容量增长则平缓得多。这一不断扩大的鸿沟使得分布式并行训练成为工业级大模型研发的绝对物理刚需[^computational-gap-data]。

[^computational-gap-data]: 蓝色实线（GPU 物理显存）：主流单卡按部署年份的主流 HBM 容量——A100 40GB (2020), A100 80GB (2021), H100 80GB (2022–2023), H200 141GB (2024), B200 192GB (2025–2026)。黄色虚线（开源权重模型）：各年份代表性开源大模型权重显存——GPT-2 1.5B (~3GB), GPT-J 6B (~12GB), BLOOM 176B (~352GB), Grok-1 314B (~628GB), LLaMA 3 405B (~810GB), DeepSeek-V3 671B (~1342GB), DeepSeek-V4-Pro 1.6T (~3200GB)。红色实线（前沿闭源/万亿模型）：@tbl:model-comparison 中的超大规模模型（含估算值）——GPT-3 175B, MT-NLG 530B, PaLM 540B, Gemini-1 1.6T (~3200GB), GPT-4V ~1.8T (~3600GB), GPT-5 ~2T (~4000GB), Claude Mythos 5 ~10T (~20000GB)。所有模型数值均按 BF16 权重（2 字节/参数）计算。此处仅展示权重本身，实际训练还需额外叠加梯度、优化器状态与激活值。

[^h100-te]: NVIDIA, "NVIDIA H100 Tensor Core GPU Architecture," whitepaper, 2022, https://resources.nvidia.com/en-us-hopper-architecture (引入带 FP8 的 Transformer Engine；在 Transformer 模型上相比 A100 可实现最高 6× 训练吞吐提升)。关于大模型独立评测审计数据，参见 NVIDIA, "Breaking MLPerf Training Records with NVIDIA H100 GPUs," Technical Blog, 2023, https://developer.nvidia.com/blog/breaking-mlperf-training-records-with-nvidia-h100-gpus/ (MLPerf Training 3.0；在 GPT-3 175B 和 BERT 上验证 Transformer Engine 与 FP8 表现)。

### 为什么必须使用集群？

单张 GPU（哪怕是当前最高端的硬件）在面对现代 AI 工作负载时也是杯水车薪。以 FP16 精度的 70B 模型为例，仅存放模型权重就需要约 140 GB 显存。若加上反向传播所需的梯度、优化器状态以及中间激活值，每个训练步的显存消耗将超过 500 GB。这已经彻底超出了任何单张 GPU 的物理容量极限。

**集群（Cluster）** 是指由高速网络互联的多台计算计算机（节点/Nodes）组成的协同计算系统。每个计算节点通常配备有多张 GPU、多颗 CPU、高容量内存和高速本地存储。通过跨节点编排与协同，AI 集群能够实现以下三大维度的线性扩展：

- **扩展显存（Scale Memory）**：将模型参数、梯度和优化器状态横向切分并分摊到成百上千张 GPU 的显存中。
- **扩展算力（Scale Compute）**：通过多节点并行计算承载超大 Batch Size，大幅缩短训练迭代周期。
- **扩展存储（Scale Storage）**：高效并行吞吐无法存放在单机上的数 TB 至 PB 级海量训练数据集。

![现代 AI 集群物理与网络架构示意图](img/ai_cluster_demo.png){#fig:ai-cluster .block width=100% align=top-right}

如 @fig:ai-cluster 所示，典型的现代 AI 集群由多个计算节点组成，每个节点通常包含 2 颗高性能 CPU 和 8 张 GPU。在**节点内部（Intra-node）**，8 张 GPU 通过 NVSwitch 芯片实现全互联拓扑，提供极高带宽的 NVLink 互联（Ampere/Hopper 架构单卡聚合双向带宽达 300–900 GB/s，Blackwell B200 进一步跃升至 1.8 TB/s）。在**节点之间（Inter-node）**，GPU 通过 InfiniBand 高速网卡（如 HDR 200 Gb/s 或 NDR 400 Gb/s）实现跨机直连，在整套集群上支持大规模分布式训练与推理。

集群架构并不是新概念，高性能计算（HPC）领域已经使用集群数十年。但 AI 负载与传统 HPC 有着本质不同的通信特征：传统 HPC 往往倾向于低频、大块的数据交换；而 AI 分布式训练则是**极高频、固定节奏的细粒度数据同步（每个 Step 均需同步梯度）**。这使得 AI 集群对网络互联的有效带宽与延迟抖动极其敏感。我们将在第~\ref{chap:running-distributed-training-with-slurm} 章深入讲解如何在 SLURM 管理的集群上高效编排分布式作业。

### AI 集群特性：专为训练与推理定制

**AI 集群**是针对深度学习负载特征进行软硬件深度定制优化的专用集群：

**针对分布式训练**，AI 集群的核心诉求包括：

- **极致高带宽的互联网络**：每个训练步都必须进行全局梯度同步。如果网络通信缓慢，算力昂贵的 GPU 将陷入漫长的空闲等待。节点内 NVLink（900 GB/s–1.8 TB/s）与跨节点 InfiniBand/RoCE（200–400 Gb/s/端口）是当今大模型训练的工业标准。
- **海量聚合显存池**：模型状态跨卡分片。即便使用 FSDP 等分片技术，一个 70B 模型也至少需要 8–16 张 GPU 才能完整装下（详见第~\ref{chap:scaling-with-fully-sharded-data-parallel-fsdp} 章）。
- **极速并行存储**：训练数据集动辄数 TB 到数百 TB。集群需要高吞吐的分布式并行文件系统或对象存储，并配合节点本地 NVMe SSD 构建高速缓存。

**针对分布式推理**，系统核心诉求则发生转向（详见第~\ref{chap:distributed-inference-fundamentals-and-vllm} 章与第~\ref{chap:production-llm-serving-stack} 章）：

- **超低网络延迟**：训练追求吞吐，而在线推理更苛求延迟。用户期望以毫秒（ms）级响应首 Token。
- **KV 缓存显存管理**：注意力机制的 KV 缓存会随着并发请求数和上下文长度暴增，系统必须在显存上限与并发吞吐间取得平衡。
- **高弹性负载均衡**：推理流量具备突发性与长尾特征，需要高度智能的请求分发网关与连续批处理调度机制。

### AI 集群核心量化评估指标

在对 AI 集群进行架构设计、基准测试与性能调优时，绝不能轻信单纯的硬件标称参数。以下是工业界衡量系统真实效率的关键硬核指标：

#### 1. 模型算力利用率（MFU, Model FLOPS Utilization）

**MFU 是衡量大模型训练硬件利用效率最权威、最重要的核心指标**。它计算的是模型实际消耗的理论计算量与硬件理论峰值算力的比值：

```Python
MFU = (单次迭代模型理论计算量 FLOPs / 实际单步迭代时间) / 硬件理论峰值 FLOPS
```

MFU 能直接反映系统究竟是处于纯算力受限（Compute-Bound）状态，还是被其他系统瓶颈所拖累。一个高度优化的千卡大模型集群，其 MFU 通常在 **40%–60%** 之间。如果 MFU 跌落到 20% 甚至更低，说明系统大概率遭遇了显存带宽瓶颈、跨卡网络通信拥塞或低效的 CUDA Kernel 启动开销。

以在 8 张 H100 GPU 上训练 70B Transformer 模型为例：
- **单步迭代理论计算量**：约 860 TFLOP（取决于 Batch Size 与序列长度）
- **实际单步迭代时间**：约 2.0 秒
- **实测每秒算力输出**：$860 \text{ TFLOP} / 2.0 \text{ s} \approx 430 \text{ TFLOPS}$
- **单卡 H100 BF16 理论峰值**：约 989 TFLOPS
- **实测 MFU**：$430 / 989 \approx 43.5\%$

对于标准密集型 Transformer 模型，每个 Token 在前向与反向传播中消耗的理论计算量约为 **$6N$ FLOPs**（$N$ 为非 Embedding 参数量）[^mfu-flops]。通过训练日志中记录的单步耗时，即可快速计算出真实 MFU。

[^mfu-flops]: Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," *Journal of Machine Learning Research* 24 (2023): 1–113, Appendix B (推导了每个 Token 消耗 6N FLOPs 的标准公式及 MFU 评估方法)。另见 Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361, 2020。

#### 2. 线性扩展效率（Linear Scaling Efficiency）

衡量集群随 GPU 节点数量增加时性能的扩展能力：

```Python
线性扩展比 = 多 GPU 集群总吞吐 / (单 GPU 吞吐 × GPU 卡数)
```

理想线性扩展比为 1.0 (100%)。在优秀的大规模集群中，该指标通常维持在 0.75–0.92 之间。如果扩展比跌破 0.6，说明网络通信开销已经占据主导，发生了严重的扩展瓶颈。

#### 3. GPU 利用率（GPU Utilization）

反映 GPU 处于活跃执行计算的时间占比。通过 `nvidia-smi` 即可快速轮询：

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
```

需要注意：95% 以上的高利用率仅代表计算核心没有闲置，并不意味着计算效率最优（例如可能在执行低效的非融合算子）。MFU 比单纯的利用率具有更高的工程洞察价值。

#### 4. 通信有效利用率（Communication Efficiency）

```Python
通信有效利用率 = 实测有效通信带宽 / 硬件理论物理带宽
```

例如在 200 Gb/s 的 InfiniBand HDR 网络中，若 AllReduce 实测吞吐达到 180 Gb/s，则通信有效利用率为 90%。若该数值偏低，通常意味着遭遇了网络拓扑错配、RoCE 丢包重传或不合理的报文分片。

#### 5. 训练吞吐量（Throughput）

- **集群总体吞吐（Tokens/s）**：
  $$\text{Throughput}_{\text{cluster}} = \frac{\text{Global Batch Size} \times \text{Sequence Length}}{\text{Total Step Time}}$$
- **单卡平均吞吐（Tokens/s/GPU）**：
  $$\text{Throughput}_{\text{per\_GPU}} = \frac{\text{Throughput}_{\text{cluster}}}{\text{GPU Count}}$$

#### 6. 耗时分解（Time Breakdown）

健康且高度优化的训练集群，其单步耗时分解通常呈现：
- **GPU 纯计算耗时**：70%–85%（密集的 GEMM 矩阵乘法）
- **跨卡网络通信耗时**：10%–20%（梯度 AllReduce 与分片重构，且大部分已与计算重叠）
- **数据搬运与空闲等待**：< 5%

#### 7. 集群可靠性与通信延迟指标

- **MTBF（平均无故障时间）**：在万卡集群上，由于硬件失效率叠加，平均每隔数小时就会出现单卡或节点故障。
- **通信延迟**：
  - 节点内 NVLink AllReduce 延迟：通常 **< 1 ms**
  - 跨节点 InfiniBand AllReduce 延迟：通常 **< 5 ms**
  - **P99 尾部延迟**：在分布式同步屏障（Barrier）下，最慢的一张“掉队卡”（Straggler）将拖慢整个集群，因此 P99/P99.9 尾部延迟比平均延迟更具决定性。

---

## 中央处理器（CPU）在分布式 AI 中的角色

![](img/cpu_hardware.png){#fig:cpu-icon .wrap width=15% align=top-right vspaces=40pt}

尽管 GPU 承担了 95% 以上的浮点计算，但 CPU 在分布式 AI 体系中扮演着不可替代的总调度师与协同枢纽角色。

### CPU 体系结构特征

CPU 基于经典的**冯·诺依曼体系架构**设计。与专注于高并发吞吐的 GPU 不同，CPU 的微架构将绝大部分晶体管面积投入到了**复杂的控制逻辑（乱序执行、分支预测器）与多级高速缓存（L1/L2/L3 Cache）**，旨在最小化单线程任务的端到端执行延迟。

在分布式训练中，CPU 核心负责：
- **数据加载与预处理流水线**：从分布式存储拉取数据、图像解码与数据增强、文本分词（Tokenization）与 Dynamic Batching 组装。
- **任务编排与 Kernel 发射**：解析 Python 代码计算图，向 GPU 异步发射 CUDA Kernel 指令流。
- **集群通信协调**：管理进程组生命周期、协调跨节点 TCP/IP 控制信令、驱动 NCCL 初始化握手。

如果 CPU 算力或数据加载成为瓶颈，GPU 将被迫陷入饥饿等待（Data Starvation）。

### CPU 与 GPU 的交互机制

![CPU 与 GPU 的协同交互机制](img/cpu_gpu_interaction.png){#fig:cpu-gpu-interaction .block width=60% align=top-right}

如 @fig:cpu-gpu-interaction 所示，典型的执行流程包括：
1. **CPU 异步发射指令**：CPU 上的 Python 进程调用 PyTorch API，生成底层 CUDA 指令并通过 PCIe 总线推入 GPU 的命令队列（CUDA Stream）。
2. **显存数据交互**：CPU 负责在系统内存（Host RAM）与 GPU 显存（Device VRAM）之间调度数据搬运。
3. **驱动跨节点通信**：在多机训练中，CPU 负责建立网络连接套接字并配合网卡驱动协同工作。

PCIe 总线带宽往往是 Host-Device 交互的瓶颈所在：PCIe Gen4 x16 单向带宽仅为 31.5 GB/s（双向约 63 GB/s），PCIe Gen5 x16 单向带宽约为 64 GB/s。这与 GPU 之间动辄 900 GB/s–1.8 TB/s 的 NVLink 互联相比有着数量级的差距。因此，**所有跨 GPU 的高频张量通信必须走 NVLink/NVSwitch 直连，严禁绕经 CPU 和 PCIe**。

### NUMA 架构与 CPU 亲和性绑定

现代双路或四路服务器均采用 **NUMA（非一致性内存访问）** 架构。每颗 CPU Socket 拥有独立的内存控制器和 PCIe 控制器。挂载在不同 Socket 下的 GPU 在访问跨 Socket 内存时需要穿过 CPU 互联总线（如 Intel UPI 或 AMD Infinity Fabric），从而带来显著的延迟与带宽损失。

通过以下命令可查看服务器的 NUMA 拓扑布局：

```bash
numactl --hardware
```

在启动多卡训练时，**强烈建议将每个 GPU 进程严格绑定至其挂载的同侧 NUMA 节点与 CPU 核心**，以避免跨 Socket 内存搬运拖慢 DataLoader 吞吐。

### 生产级 8 卡服务器的 CPU 配置基线

- **CPU 核心配比**：每张 GPU 应配备至少 2–4 个物理 CPU 核心。标准 8 卡服务器建议配置 32–64 物理核心。
- **Host 内存容量**：建议为 GPU 显存总量的 1.5–2 倍。对于 8×80GB GPU 服务器，建议配备 1 TB–1.5 TB 系统内存。
- **PCIe 通道数**：8 张 GPU 需满血占用 128 条 PCIe Lanes，通常需要双路 AMD EPYC 或 Intel Xeon 处理器支持。
- **本地 NVMe 缓存**：建议配置 2–4 块高速 NVMe SSD 组建 RAID-0（提供 10–20 GB/s 顺序读带宽），作为训练数据集的高速本地二级缓存。

---

## 图形处理器（GPU）：现代 AI 的算力中枢

![](img/gpu_hardware.png){#fig:gpu-icon .wrap width=15% align=right}

### GPU 显存与计算体系结构

GPU 专为**高并发吞吐（High-Throughput）**而生。它集成了数以万计的精简计算核心，并配备了极高带宽的显存子系统。

GPU 内部同样存在层次分明的显存金字塔结构：
- **寄存器文件（Registers）**：带宽最高（数十 TB/s）、延迟最低，但单 SM 容量极小（通常 64 KB/SM）。
- **共享内存/L1 缓存（Shared Memory / L1 Cache）**：低延迟片上内存，供同一个线程块（Thread Block）内部共享（通常 48–228 KB/SM）。
- **L2 缓存（L2 Cache）**：全局共享的片上大容量缓存（Hopper 拥有 50 MB，Blackwell 达到数十 MB）。
- **全局显存（Device VRAM / HBM）**：容量最大（80–192 GB），通过宽位宽的 HBM 堆叠颗粒提供数 TB/s 的读取带宽。平时触发的 Out of Memory (OOM) 均指此处的 HBM 耗尽。

![GPU 显存层次体系结构](img/gpu_mem.png){#fig:gpu-memory-hierarchy .block width=60% align=top-right lines=8}

如 @fig:gpu-memory-hierarchy 所示，显存设计遵循基本物理规律：越靠近计算核心的存储级别，带宽越高、延迟越低，但物理容量越小；反之容量越大的级别，访问延迟越高。

在很多大模型负载中，**显存读取带宽往往比纯算力更早触达性能天花板（Memory-Bound）**。如果算子处于显存受限状态，单纯堆叠浮点算力无法带来任何加速。

使用 `nvidia-smi` 可精确查询 GPU 硬件底层规格：

```bash
nvidia-smi \
  --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max \
  --format=csv
```

### 拓扑探测：`nvidia-smi topo -m`

执行以下命令可以输出当前主机的 GPU 间物理互联拓扑矩阵：

```bash
nvidia-smi topo -m
```

在拓扑矩阵中，重点关注以下连接标识：
- **`NV18` / `NV12` / `NV4`**：表示两卡之间通过 NVLink 物理直连（数字代表链路数量）。在搭载 NVSwitch 的 HGX/DGX 系统中，所有 GPU 间均显示为满血 NVLink 全互联，支持全速 All-to-All 通信。
- **`PIX` / `PXB`**：表示两卡之间仅通过内部 PCIe Switch 或同一 Host Bridge 连接，通信带宽受限为 PCIe 级别。
- **`NODE` / `SYS`**：表示跨越了 NUMA 节点或 CPU Socket，通信延迟最高。

### NVIDIA GPU 代际核心架构演进

1. **Volta 与 Ampere 架构（2017–2020）**：
   - **V100 (Volta)**：首次引入第一代 Tensor Core，开启硬件加速深度学习新纪元。
   - **A100 (Ampere)**：引入 TF32 与 BF16 原生支持，搭载 NVLink 3.0（单卡聚合 600 GB/s 带宽），首创 MIG 硬件切分技术，提供 40GB/80GB HBM2e。
2. **Hopper 架构（2022）——当前工业级主力**：
   - **H100**：引入第四代 Tensor Core 与 **Transformer Engine（支持 FP8 动态精度切换）**，配备 NVLink 4.0（单卡聚合 900 GB/s 带宽），80GB HBM3（3 TB/s 带宽），在 Transformer 模型上实现相比 A100 最高 6× 的训练吞吐提升[^h100-te]。
   - **H200**：算力核心与 H100 保持一致，但升级为 **141 GB HBM3e 显存（4.8 TB/s 带宽）**，专为打破超大参数模型与超长上下文的显存墙而生。
3. **Blackwell 架构（2024–2026）——新一代旗舰**：
   - **B200**：采用双 Chiplet 封装整合为一个庞大的计算实体，搭载第二代 Transformer Engine 与 **NVFP4 极致精度支持**；配备 NVLink 5.0（单卡聚合双向带宽达 **1.8 TB/s**），集成 **192 GB HBM3e（8 TB/s 带宽）**，提供高达 2.25 PFLOPS 的 FP16/BF16 密集算力。

### GPU 产品交付形态：HGX vs DGX vs SuperPOD

- **HGX**：面向 OEM 服务器厂商（如 Dell、Supermicro、浪潮等）的标准化 GPU 基础板卡（Baseboard）。例如 HGX H100 集成 8 张 GPU 与 4 颗 NVSwitch 芯片。
- **DGX**：NVIDIA 官方自研的整机服务器系统。预先集成了优化的 CPU、NVMe 存储、8 张 GPU、NVSwitch 以及多端口 ConnectX InfiniBand 网卡，开箱即用。
- **SuperPOD**：由数十至上百台 DGX 节点通过 InfiniBand 多层胖树网络互联构成的超级计算集群（如 GB200 NVL72 机柜系统，通过背板全液冷 NVLink 将 72 张 B200 连为一个超高带宽的统一算力域）。

---

## 专用加速芯片：TPU 与 NPU

除了占市场绝对统治地位的 NVIDIA GPU 之外，业界还存在针对神经网络定制的专用集成电路（ASIC）。

### 谷歌张量处理器（TPU）

![](img/tpu_hardware.png){#fig:tpu-icon .wrap width=15% align=right}

TPU 是 Google 专为神经网络负载从零定制的 ASIC 芯片。

#### 核心架构：脉动阵列（Systolic Array）

与 GPU 依赖数千个通用线程核心不同，TPU 的核心计算单元是 **MXU（矩阵乘法单元）**，采用经典的**脉动阵列**架构。在脉动阵列中，数据如同血液流经心脏般在二维网格的各个处理单元（PE）之间直接流动：每个 PE 完成乘累加后将结果直接传递给相邻 PE，极大地减少了对片上寄存器和显存的高频读写，使得矩阵乘法的能效比大幅提升。

#### 互联架构：2D/3D Torus 拓扑与光路交换（OCS）

TPU Pod 摒弃了传统以太网/InfiniBand 交换机架构，采用芯片间直连的 **3D 环形网格拓扑（3D Torus）**。在 TPU v4/v5 及后续代际中，Google 引入了 **OCS（光路交换机）**，利用微机电系统（MEMS）光镜动态重构光路拓扑，以极低的延迟实现数千芯片的高效互联。

#### TPU 软件栈：XLA 与 JAX

TPU 采用 **XLA（加速线性代数编译器）**。开发者编写 JAX 或 TensorFlow 代码，XLA 将计算图整体验证并编译为底层硬件指令。其优势在于全图融合优化能力强，但首次编译时间较长，且调试报错相对黑盒。

### 神经网络处理器（NPU）

![](img/npu_hardware.png){#fig:npu-icon .wrap width=15% align=right}

**NPU** 是针对深度学习算子进行领域专用设计（DSA）的芯片总称，代表产品包括华为昇腾（Ascend 910B/910C 系列配合 CANN/MindSpore 栈）、亚马逊 AWS Trainium 系列、寒武纪 MLU 等。

- **优势**：在特定受支持的算子和模型上具备极佳的性价比与能效比。
- **挑战**：软件生态相对封闭，算子库与主流开源社区（PyTorch 原生生态、FlashAttention 等）的适配往往需要额外的迁移和调试成本。

---

## 高速互联体系：分布式 AI 的网络命脉

在分布式 AI 中，芯片间的互联网络直接决定了梯度同步与模型切分通信的实际效率。

### 节点内部互联技术

- **PCIe**：单卡与 CPU 之间的通用通道。PCIe Gen4 x16 双向带宽约 63 GB/s，PCIe Gen5 x16 双向带宽约 128 GB/s。
- **NVLink**：GPU 之间点对点的高速双向总线。单卡聚合带宽达到 600 GB/s (A100)、900 GB/s (H100)、1.8 TB/s (B200)。
- **NVSwitch**：片上网络交换芯片。将单机内的 8 张 GPU 构建为无阻塞的 All-to-All 全互联网络，使任意两张 GPU 之间都能以满血 NVLink 速率通信。

### 跨节点集群互联技术

- **InfiniBand (IB)**：专为超算与大模型集群打造的高吞吐、超低延迟网络。当前主流标准为 **HDR (200 Gb/s)** 与 **NDR (400 Gb/s)**。其原生支持硬件级无损网络与信用流控（Credit-based Flow Control）。
- **RoCE (RDMA over Converged Ethernet)**：在标准以太网物理层上封装 RDMA 协议。成本相对较低，但在高负载下依赖交换机严格配置 PFC（基于优先级的流量控制）和 ECN（显式拥塞通知）以维持无损传输。
- **GPUDirect RDMA**：允许跨节点的两张 GPU 直接通过网卡进行远程显存数据直传，**完全绕过 Host CPU 内存与系统内核**，将跨机通信延迟压制到微秒级。

---

## 芯片编程范式：SPMD 与 CUDA

理解底层芯片的执行模型，是编写高性能分布式代码的重要理论基石。

### 编程模型 vs 执行模型

- **编程模型（Programming Model）**：面向开发者的软件抽象接口（如 CUDA 线程网格、PyTorch 张量计算）。
- **执行模型（Execution Model）**：底层硬件实际调度执行物理指令的机制（如 SIMD、SIMT）。

### SPMD：单程序多数据

**SPMD（Single Program, Multiple Data）** 是一种顶层编程模型：所有处理核心运行同一份程序源码，但根据自身的唯一标识（如 CUDA 线程 ID、或分布式环境下的进程 Rank）处理不同的数据子集。

### CUDA 的 SIMT 执行模型

NVIDIA GPU 采用 **SIMT（单指令多线程，Single Instruction, Multiple Thread）** 机制执行 SPMD 程序：
- **线程（Thread）**：最小逻辑执行单元，拥有私有寄存器。
- **线程束（Warp）**：**GPU 硬件调度的基本物理单元**，每个 Warp 固定包含 **32 个线程**。同一个 Warp 内的 32 个线程在同一时钟周期执行同一条指令。
- **分支分化（Branch Divergence）**：若 Warp 内的线程在 `if-else` 分支中走向不同路径，硬件将被迫串行依次执行两个分支并屏蔽无关线程，造成性能骤降。
- **细粒度多线程隐藏延迟（FGMT）**：当某个 Warp 处于等待全局显存读取的停顿状态时，Warp 调度器能以零周期开销瞬间切换到就绪的另一个 Warp 执行，从而高效掩盖显存访问延迟。

---

## 分布式通信：NCCL 运行机制与环境调优

在 PyTorch 分布式框架中，GPU 间的集合通信默认由 **NCCL（NVIDIA 集合通信库）** 驱动。

### NCCL 核心运行机制

当执行 `dist.all_reduce()` 时，NCCL 会自动探测硬件互联拓扑：
- 节点内 GPU 对优先走 NVLink/NVSwitch；
- 跨节点优先调用 GPUDirect RDMA 通过 InfiniBand/RoCE 网卡直传；
- 根据传输张量的大小，在底层自动选择 **Ring（环形通信）** 或 **Tree（树形归约）** 算法。

### 生产环境关键 NCCL 环境变量

- `NCCL_DEBUG=INFO`：输出详细的 NCCL 初始化日志、拓扑检测结果与选取的通信通道，是排查通信异常的最重要工具。
- `NCCL_IB_DISABLE=1`：强制关闭 InfiniBand，回退至 TCP 模式（用于网络故障排查与对比隔离）。
- `NCCL_SOCKET_IFNAME=eth0,ib0`：显式指定跨机通信绑定的物理网卡接口，防止 NCCL 误走低速管理网口。

---

## 核心并行策略分类学与技术全景

在深入具体算法之前，必须理清分布式并行的分类学界限。

### 核心分水岭：切分状态 vs 切分计算

判断一种并行技术属性的核心标准在于：**它是在切分模型状态（State），还是在切分单样本的前向/反向计算本身（Computation）？**

1. **若单样本的前向传播必须跨越多个 GPU 协同计算才能完成** $\rightarrow$ **模型并行（计算切分）**。
2. **若每个 GPU 均能独立完成单样本的前向计算，仅在更新时切分参数、梯度或优化器状态** $\rightarrow$ **数据并行 / 状态分片**。

### 大模型并行策略全景分类表

| 并行策略 | 核心类别 | 子类别 | 适用阶段 | 代表性实现框架 |
| ----------------------- | ---------------- | -------------------------------- | ------------- | -------------------- |
| 数据并行 (DDP) | 数据切分 | 完整模型副本复制，Batch 数据切分 | 训练 | PyTorch DDP (第~\ref{chap:distributed-training-with-pytorch-ddp} 章) |
| 全分片数据并行 (FSDP) | 状态分片 | 参数 + 梯度 + 优化器状态全面分片 | 训练 | PyTorch FSDP (第~\ref{chap:scaling-with-fully-sharded-data-parallel-fsdp} 章) |
| ZeRO-1 | 状态分片 | 仅优化器状态分片（$4\times$ 内存节省） | 训练 | DeepSpeed ZeRO (第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron} 章) |
| ZeRO-2 | 状态分片 | 优化器状态 + 梯度分片（$8\times$ 内存节省） | 训练 | DeepSpeed ZeRO (第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron} 章) |
| ZeRO-3 | 状态分片 | 优化器 + 梯度 + 参数全分片（与 FSDP 等价） | 训练 | DeepSpeed ZeRO (第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron} 章) |
| 张量并行 (TP) | 计算切分 | 层内切分（矩阵按行/列切分到多卡） | 训练 / 推理 | Megatron-LM (第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron} 章) |
| 序列并行 (SP) | 计算切分 | 沿序列长度维度切分 LayerNorm/Dropout | 训练 | Megatron-LM (第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron} 章) |
| 上下文并行 (CP) | 计算切分 | 超长上下文注意力与 KV 跨卡切分 | 推理 / 训练 | vLLM (第~\ref{chap:distributed-inference-fundamentals-and-vllm} 章) / SGLang |
| 流水线并行 (PP) | 计算切分 | 沿模型网络深度将不同层分配到不同卡 | 训练 / 推理 | Megatron-LM / DeepSpeed PP |
| 专家并行 (EP) | 计算切分 | MoE 稀疏门控专家网络跨卡切分 | 训练 / 推理 | DeepSpeed-MoE (第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron} 章) |

### 混合并行组合范式（Composition Patterns）

在训练千亿乃至万亿参数的大模型时，工业界通常将上述基础策略进行正交组合：

| 组合模式 | 构成原语 | 典型应用场景 | 代表性工业系统 |
| ------------------- | ---------------------- | ---------------- | ---------------------- |
| **DP + TP** | 数据并行 + 张量并行 | 单机内 NVLink 做 TP，跨机做 DP | Megatron-LM |
| **DP + PP** | 数据并行 + 流水线并行 | 显存受限但卡数充足的多节点集群 | GPipe + DDP |
| **3D 并行 (DP+TP+PP)** | 数据 + 张量 + 流水线 | 万卡集群训练 100B–1T 超大模型 | Megatron-DeepSpeed |
| **DP + EP** | 数据并行 + 专家并行 | 训练千亿规模稀疏 MoE 模型 | DeepSpeed-MoE / ST-MoE |
| **FSDP + TP** | 状态分片 + 张量并行 | 显存极度紧凑下的超大 Dense 模型 | PyTorch FSDP2 + DTensor |
| **TP + CP** | 张量并行 + 上下文并行 | 100K–1M 超长序列推理服务 | vLLM / SGLang |

---

## 策略决策：分布式训练与推理决策树

### 训练策略选型决策树

![分布式训练策略决策树](img/training_tree.png){#fig:training-strategy-tree}

如 @fig:training-strategy-tree 所示，训练选型的标准决策路径如下：

1. **完整模型能否放入单卡显存？**
   - **能** $\rightarrow$ 首选 **DDP 数据并行**（简单、高效、通信与计算完美重叠）。
   - **不能** $\rightarrow$ 进入状态分片。
2. **状态分片能否解决显存问题？**
   - 使用 **PyTorch FSDP** 或 **DeepSpeed ZeRO-3**。若显存满足且吞吐达标，优先使用 FSDP/ZeRO，避免引入更复杂的模型切分。
3. **若单卡连单层计算都装不下，或需要极高计算吞吐：**
   - 节点内（具备 NVLink）引入 **张量并行 (TP)**；
   - 跨节点引入 **流水线并行 (PP)** 或 **数据并行 (DP)**；
   - 若针对 MoE 模型，引入 **专家并行 (EP)**。
4. **系统级极限优化**：
   - 开启**激活值检查点（Activation Checkpointing）**；
   - 开启 **CPU/NVMe Offloading**（以带宽换容量）。

### 推理策略选型决策树

![分布式推理策略决策树](img/inference_tree.png){#fig:inference-strategy-tree}

如 @fig:inference-strategy-tree 所示，推理优化侧重于延迟与 KV 缓存管理：

1. **请求并发扩展**：
   - 优先通过部署**多模型副本（Replicas）**配合负载均衡提升吞吐。
2. **单请求模型切分**：
   - 若模型无法放入单卡显存，在节点内使用 **张量并行 (TP)**（如 vLLM 中的 TP=2, 4, 8）；
   - 在超长上下文场景下，使用 **上下文并行 (CP)** 切分注意力计算与 KV 缓存。
3. **显存与 KV 缓存核心优化**：
   - 采用 **PagedAttention** 彻底消除显存碎片；
   - 采用 **INT8 / FP8 / INT4 量化** 压缩权重显存。

---

\fancydividerwithicon[center]{python.png}

## 实战：硬件拓扑检测与显存/通信带宽压测

在设计分布式方案之前，必须亲自动手对硬件物理拓扑与实际链路带宽进行基准压测。

### 步骤 1：GPU 基础硬件信息探测

运行 `code/check_cuda.py`（耗时不足 1 秒）：

```bash
python code/check_cuda.py
```

源码逻辑：

```python
#LINENUM
import torch
print(f"CUDA available: {torch.cuda.is_available()}") #HL
print(f"Number of GPUs: {torch.cuda.device_count()}") #HL
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i) #HL
    vram_gb = props.total_memory / (1024**3) #HL
    print(f"GPU {i}: {props.name}")
    print(f"  Total memory: {vram_gb:.1f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Multiprocessors: {props.multi_processor_count}")
```
CODE_EXPLAIN_START:
- 2: 检查当前系统 CUDA 是否可用
- 3: 获取当前可用 GPU 总数
- 5: 获取各 GPU 详细属性
- 6: 将显存字节数转换为 GB 显示
CODE_EXPLAIN_END

### 步骤 2：硬件互联拓扑探测

执行以下命令查看 GPU 间的实际物理连接：

```bash
nvidia-smi topo -m
```

确认卡间是 `NV18`（NVLink 直连）还是 `PIX`（PCIe 连接），以及 GPU 绑定的 NUMA 节点编号。

### 步骤 3：单 GPU 本地显存带宽基准压测

运行 `code/bandwidth_test.py` 测量单卡 HBM 显存的真实读写带宽：

```bash
python code/bandwidth_test.py
```

核心压测代码：

```python
#LINENUM
import torch
import time
size_mb = 64
iterations = 200
nbytes = size_mb * 1024 * 1024
a = torch.randn(nbytes // 4, device='cuda')
b = torch.empty_like(a) #HL
# 热身 Warmup
for _ in range(10):
    b.copy_(a) #HL
torch.cuda.synchronize()
# 性能测试 Benchmark
t0 = time.time()
for _ in range(iterations):
    b.copy_(a)
torch.cuda.synchronize()
t1 = time.time()
bytes_moved = 2 * nbytes * iterations  # 每次复制包含读取 a 与写入 b
bandwidth_gb_per_s = bytes_moved / (1024**3) / (t1 - t0)
print(f"Effective bandwidth (read+write): {bandwidth_gb_per_s:.2f} GB/s")
```
CODE_EXPLAIN_START:
- 5: 在 GPU 上分配 Float32 张量（每元素 4 字节）
- 6: 分配等尺寸目标张量
- 10: 执行显存内部复制
- 18: 计算读写双向总数据搬运量
CODE_EXPLAIN_END

实测参考基准：
- **H100 / H200**：实测有效带宽通常达 **2.0–3.2 TB/s**
- **A100 (80GB)**：实测有效带宽约 **1.5–1.9 TB/s**

### 步骤 4：多 GPU 跨卡 AllReduce 集合通信带宽压测

使用 `torchrun` 启动多卡运行 `code/allreduce_microbench.py`：

```bash
torchrun --nproc_per_node=2 code/allreduce_microbench.py
torchrun --nproc_per_node=4 code/allreduce_microbench.py
```

算法带宽（Algorithm Bandwidth）与总线带宽（Bus Bandwidth）计算逻辑：

```python
#LINENUM
import torch.distributed as dist
size = size_mb * 1024 * 1024 // 4  # 每张 GPU 分配的 float32 元素数
tensor = torch.ones(size, device=f'cuda:{local_rank}') #HL
for _ in range(warmup):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM) #HL
torch.cuda.synchronize()
start = time.time()
for _ in range(iterations):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
elapsed = time.time() - start
size_bytes = size_mb * 1024 * 1024
n = world_size
alg_bytes = size_bytes * n * iterations
bus_bytes = size_bytes * 2 * (n - 1) * iterations  # Ring 算法下每卡传输 2*(n-1)/n * n
alg_bw_gb_s = alg_bytes / (1024**3) / elapsed
bus_bw_gb_s = bus_bytes / (1024**3) / elapsed
```
CODE_EXPLAIN_START:
- 3: 各卡分配测试张量
- 5: 执行跨卡 AllReduce 求和
- 15: Ring AllReduce 总线传输总量修正公式：$2 \times (n-1) \times \text{size}$
CODE_EXPLAIN_END

实测总线带宽参考：
- **NVLink 直连卡对**：实测总线带宽通常达到数十至数百 GB/s；
- **纯 PCIe 连接卡对**：实测总线带宽通常仅有 10–40 GB/s。

### 步骤 5：压测结果与并行策略映射

| 测量指标 | 预期健康基准 | 结果指导意义 |
|--------|----------------|-------|
| 单卡 HBM 读写带宽 | A100: 1.5–2 TB/s<br>H100: 2–3 TB/s | 低于预期说明显存访问出现严重竞争或非对齐 |
| 跨卡 AllReduce 总线带宽 (NVLink) | 数十至数百 GB/s | 带宽充裕，适合重度通信的张量并行 (TP) 与全分片 (FSDP) |
| 跨卡 AllReduce 总线带宽 (PCIe) | 仅 10–50 GB/s | 严禁使用张量并行，应优先选择 FSDP 或流水线并行 (PP) |

---

至此，我们已经完整建立了硬件拓扑、通信网络与核心并行策略的系统认知。在第 3 章中，我们将正式切入分布式训练的实战中枢——**PyTorch DDP（DistributedDataParallel）**，从环境搭建、源码剖析、故障排查到性能优化，全面掌握工业级分布式训练的核心技能。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
