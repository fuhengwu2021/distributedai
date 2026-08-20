# 第 4 章：基于全分片数据并行（FSDP）的大模型扩展 {-}

*通过参数分片突破单卡显存墙，训练超大参数量模型*

> 数据中心已成为全新的计算单元。  
> —— 黄仁勋（Jensen Huang，NVIDIA 创始人兼 CEO）

**核心代码速查**

- `torch.distributed.fsdp.fully_shard(module, mesh=..., mp_policy=...)`：FSDP2 核心函数，基于 DTensor 的按参数分片（Per-Parameter Sharding）就地修改模型
- `torch.distributed.fsdp.MixedPrecisionPolicy`：FSDP2 混合精度策略配置类
- `torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,))`：创建用于分布式分片的设备网格（Device Mesh）
- `torch.distributed.checkpoint`：分布式检查点（DCP）API，支持多进程并行保存与加载分片权重
- `torch.distributed.fsdp.FullyShardedDataParallel(module, ...)`：FSDP1 经典包装器类（基于 FlatParameter 扁平化拼接）
- `torch.distributed.fsdp.wrap()`：FSDP1 子模块分层包装工具
- `torch.distributed.fsdp.MixedPrecision`：FSDP1 混合精度配置类
- `torch.distributed.fsdp.set_state_dict_type()`：配置检查点权重字典格式（Full/Sharded）
- `torch.distributed.fsdp.StateDictConfig` / `OptimStateDictConfig`：状态字典导出配置


## 从 DDP 跨越到 FSDP

**全分片数据并行（FSDP, Fully Sharded Data Parallel）** 是一种将模型参数、梯度和优化器状态全面切分到集群所有可用 GPU 上的先进分布式训练范式。在第~\ref{chap:distributed-training-with-pytorch-ddp} 章中，我们学习了传统的 DDP：它在每张 GPU 上都完整复制一份模型副本，适用于模型能够完全装入单卡显存的场景。然而，当模型参数量突破数十亿乃至数百亿时，仅存放模型权重及其伴随的梯度与优化器状态就会彻底撑爆单卡物理显存，DDP 便不再可行。FSDP 通过消除所有冗余显存占用，使得数千亿参数规模的超大模型训练成为可能。

在 PyTorch 生态中，针对 GPU 训练提供了两套演进脉络的 FSDP API，同时针对 TPU 提供了基于 XLA 的 SPMD 实现：

- **FSDP1**（`FullyShardedDataParallel`）：2021 年发布的初代包装器类，采用**扁平化参数（FlatParameter）**拼接机制。
- **FSDP2**（`fully_shard()`）：2024 年推出的下一代全新架构，采用基于 DTensor 的**按参数独立分片（Per-Parameter Sharding）**范式，更加模块化、原生适配 `torch.compile`，是 PyTorch 官方强力推荐的未来标准。
- **SPMD FSDP**（`SpmdFullyShardedDataParallel`）：专用于 Google TPU/XLA 设备的 GSPMD 自动编译器切分实现。

在全书中，当我们泛指分片机制（All-Gather 参数重构、Reduce-Scatter 梯度归约）这一通用技术时统称为 **FSDP**；而在涉及具体 API 选型与底层实现差异时，则显式区分 **FSDP1** 与 **FSDP2**。

本章的核心焦点是基于 CUDA GPU 的 **FSDP2**。但值得注意的是，FSDP1 依然在许多前沿生产开源代码库中广泛使用（例如 Wan2.2 视频生成大模型结合 DeepSpeed Ulysses 开展多卡推理时即依赖 FSDP1 接口）[^wan22]。我们将在本章系统剖析其演进脉络，并以 FSDP2 为主线展开深度实战。

[^wan22]: Wan2.2 开源代码库：<https://github.com/Wan-Video/Wan2.2>
[^fsdp2-rfc]: PyTorch FSDP2 RFC 设计规范：<https://github.com/pytorch/pytorch/issues/114299>
[^t5-flan]: **T5**（Text-to-Text Transfer Transformer）是 Google 提出的经典 Encoder-Decoder 文本生成架构。**FLAN-T5** 是经过指令微调（Instruction-Tuned）的强化版本（涵盖 flan-t5-small 至 flan-t5-xxl），广泛用于长文本摘要与推理问答。

---

## 为什么 FSDP 能够突破单卡物理显存墙？

在传统 DDP 训练中，每张 GPU 必须完整承受以下四大显存开销：

1. **模型参数（Model Parameters）**：以 BF16 精度的 7B 模型为例，参数本身占用 $7\text{B} \times 2\text{ Bytes} = \mathbf{14\text{ GB}}$。
2. **参数梯度（Gradients）**：与参数尺寸完全一致，在 BF16 下同样占用 $\mathbf{14\text{ GB}}$。
3. **优化器状态（Optimizer States）**：以最常用的 Adam 优化器为例，必须以 FP32 高精度维护一阶动量与二阶方差（共 $2 \times 4\text{ Bytes}$），开销高达 $7\text{B} \times 8\text{ Bytes} = \mathbf{56\text{ GB}}$。
4. **前向激活值（Activations）**：取决于序列长度与 Batch Size，通常占用数十 GB。

仅计算前三项模型固有静态开销，DDP 在每张卡上就需要固化占用 $14 + 14 + 56 = \mathbf{84\text{ GB}}$ 显存（若加上 FP32 Master Weight 备份，实际静态占用直逼 **112 GB**）。这已经彻底超出了单张 80 GB H100 GPU 的物理极限。

而在 FSDP 体系下，这 84 GB 的静态开销被**均匀切分摊销到集群的 $N$ 张 GPU 上**。若在 8 卡集群上运行：
$$\text{每卡显存占用} = \frac{84\text{ GB}}{8} = \mathbf{10.5\text{ GB}}$$
显存占用瞬间从无法容纳的 84 GB 骤降至 10.5 GB，为前向激活值腾出了充裕的计算空间。

![单卡显存占用对比：DDP vs FSDP（以 7B 参数模型为例）](img/ddp_fsdp_mem.png){#fig:ddp-fsdp-mem .block width=100% align=center}

如 @fig:ddp-fsdp-mem 所示，DDP 在每张卡上均保留全量 84 GB 状态导致单卡直接 OOM；而 FSDP 使显存随卡数呈 $84 / N$ 线性下降，在 8 卡下仅需 10.5 GB/卡。

>NOTES: **核心区别：激活值并未被 FSDP 切分**
>
>FSDP 切分的仅仅是模型参数、梯度与优化器状态这三项“静态模型状态”，**每张 GPU 上的前向激活值（Activations）依然是本地独立持有的**。因此，激活值检查点（Activation Checkpointing / 激活值重计算）通常与 FSDP 深度搭配使用，以在反向传播时以少量算力重算换取宝贵的显存空间。
>NOTEE

### FSDP 的底层动态运行机制

FSDP 的理论源头是微软亚洲研究院与 Redmond 团队在 2019 年发表的重磅论文 **ZeRO（Zero Redundancy Optimizer，零冗余优化器）**[^zero-paper]。ZeRO 指出：数据并行中各卡保存相同模型副本是极大的显存浪费，通过动态按需通信，可以在不改变数据并行数学等价性的前提下消除所有冗余状态。

[^zero-paper]: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," SC 2020. <https://arxiv.org/abs/1910.02054>

FSDP 的精妙之处在于巧妙利用了神经网络**前向与反向逐层按序推进**的时间局部性特征，其单步训练闭环由两大核心集合通信原语驱动：

1. **前向传播（All-Gather）**：当计算流推进到第 $L$ 层时，FSDP 触发 **All-Gather** 操作，瞬间从其余各卡拉取参数分片，在本地临时拼出该层的完整权重并执行前向计算；**该层前向计算一结束，立即释放临时拼装的完整权重，显存重新回落到分片状态**。
2. **反向传播（Reduce-Scatter）**：反向传播推进到第 $L$ 层时，再次触发 All-Gather 临时重构权重以计算梯度；梯度计算完毕后，FSDP 触发 **Reduce-Scatter** 操作——在跨卡累加梯度的同时，直接将归约后的平均梯度分散写回各卡对应的分片中，本地仅保留 $1/N$ 的梯度分片。

![FSDP 核心通信流：前向 All-Gather 与反向 Reduce-Scatter](img/fsdp_allgather_reducescatter.png){#fig:fsdp-allgather-reducescatter .block width=100% align=center}

如 @fig:fsdp-allgather-reducescatter 所示，左侧前向过程中每卡原本仅持有 $1/N$ 分片，经 All-Gather 临时拼出全量权重；右侧反向过程中本地计算出全量梯度，经 Reduce-Scatter 规约后每卡仅保留 $1/N$ 的梯度分片。

### FSDP1 vs FSDP2：架构演进与技术对比

PyTorch 最初在 2021 年发布的初代 FSDP（FSDP1）借鉴了 FairScale 的 **`FlatParameter`（扁平化拼接）** 方案：将一个子模块内的所有多维权重张量（如 Linear 的 Weight 与 Bias）全部拉平成一维大张量并拼接在一起，然后再进行分片。这种设计虽然粗暴有效，但带来了诸多严重的技术包袱：同一个模块内的参数必须强制共享相同的数值精度、无法混合冻结参数与可训练参数、破坏了原生计算图结构，导致 `torch.compile` 与动态图捕获极易失效。

**FSDP2**（2024 年正式推出，RFC #114299[^fsdp2-rfc]）彻底摒弃了扁平化拼接，转而采用基于 **DTensor（分布式张量）的按参数独立切分（Per-Parameter Sharding，沿第 0 维切分 `Shard(0)`）**：
- 例如一个形状为 `(4096, 1024)` 的线性层权重在 4 卡环境下，被自然切分为 4 个形状为 `(1024, 1024)` 的本地张量；
- 不再有任何多参数强制拼接逻辑，代码量从 FSDP1 的 ~14,000 行大幅精简至 ~3,000 行；
- **支持混合精度细粒度混用**（如部分算子使用 FP8，其余使用 BF16）；
- **原生完美兼容 `torch.compile`**，允许编译器透视单个算子并进行深度的通信与计算融合优化；
- **原生无缝集成分布式检查点（DCP）**，保存与加载分片 Checkpoint 无需经过低效的单卡 All-Gather 聚合。

![FSDP1 与 FSDP2 参数布局对比](img/fsdp1_vs_fsdp2_layout.png){#fig:fsdp1-vs-fsdp2-layout .block width=100% align=center}

如 @fig:fsdp1-vs-fsdp2-layout 所示，FSDP1（左）将 W1、W2、W3 强制拉平拼接为单个连续张量再分片；FSDP2（右）对每个参数张量独立在第 0 维分片，保持了清晰独立的参数实体。

---

## FSDP2 实战：基于 `fully_shard` 的下一代 API

在 FSDP2 中，API 从传统的面向对象包装器类（Wrapper Class）转变为更加优雅的**函数式就地变换（In-place Modification）**：`fully_shard()`。

### 核心极简代码范式

```python
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

# 1. 初始化 1D 设备网格（DeviceMesh）
world_size = torch.distributed.get_world_size()
mesh = init_device_mesh("cuda", (world_size,))

# 2. 定义混合精度策略（参数存为 BF16，梯度归约保持 BF16 或 FP32）
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

# 3. 对模型或子模块直接应用分片（就地转换，无需包装为新对象）
fully_shard(
    model,
    mesh=mesh,
    mp_policy=mp_policy,
)
```

### 设备网格（DeviceMesh）：1D 与 2D 混合拓扑

`DeviceMesh` 是 PyTorch 统一的多维计算设备拓扑抽象。
- **1D 网格（标准 FSDP）**：将单机或集群所有 GPU 排布为单一维度 `init_device_mesh("cuda", (8,))`，所有卡均参与全分片。
- **2D 网格（HSDP 混合分片数据并行）**：针对数千卡超大规模集群，若跨节点执行 All-Gather 通信开销过大，可通过 2D 网格实现**机内分片 + 跨机复制**（例如 4 台节点，每台 8 卡）：
  ```python
  # 2D 网格：第 0 维跨机复制（4 节点），第 1 维机内 NVLink 分片（8 卡）
  mesh_2d = init_device_mesh("cuda", (4, 8))
  fully_shard(model, mesh=mesh_2d)
  ```

![DeviceMesh 拓扑结构：1D FSDP 与 2D HSDP 对比](img/device_mesh.png){#fig:device-mesh .block width=100% align=center}

如 @fig:device-mesh 所示，1D 网格下全局 4 张 GPU 构成统一分片组；而在 2D HSDP 网格下，节点 N0 和 N1 内部通过高速 NVLink 执行参数 All-Gather 分片（绿色箭头），跨机之间仅传递梯度副本更新（红色箭头），大幅降低了跨节点低速网络的流量压力。

### 核心调优参数：`reshard_after_forward`

`reshard_after_forward` 控制着系统在**显存占用与通信流量之间的极致权衡**：
- **`True`（默认值，ZeRO-3 等价）**：每层前向计算完毕后立即释放拼装出的全量参数，显存占用最低，但反向传播时必须再次触发 All-Gather；
- **`False`（ZeRO-2 等价）**：前向计算完毕后保留全量参数在显存中不释放。反向传播直接复用，完全省去了反向阶段的 All-Gather 通信，通信开销减半，但峰值显存占用更高。

![reshard_after_forward 为 True 与 False 时的通信时序对比](img/reshard_after_forward.png){#fig:reshard-after-forward .block width=100% align=center}

如 @fig:reshard-after-forward 所示，`reshard_after_forward=True` 时前向后立即释放参数（free），反向需再次 All-Gather；为 `False` 时前向后驻留参数（keep），反向直接计算。

### 分层分片（Hierarchical Sharding）机制

在构建 Transformer 架构大模型时，**绝对禁止仅在最外层根模型上调用一次 `fully_shard(model)`**（扁平切分会导致整个模型在前向开始前瞬间把全量参数 All-Gather 到显存中，引发瞬时显存暴涨 OOM）。

标准的最佳实践是**分层细粒度分片（Hierarchical Sharding）**：自底向上，先对每个独立的 Transformer Block 依次调用 `fully_shard()`，最后再对最外层 Root 模型调用 `fully_shard()`：

```python
# 1. 递归对每个 Transformer Layer 独立进行分片包装
for layer in model.transformer.layers:
    fully_shard(layer, mesh=mesh, mp_policy=mp_policy)

# 2. 对最外层根模型应用分片
fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

![分层分片（Hierarchical Sharding）与扁平切分架构对比](img/fsdp_hierarchical_sharding.png){#fig:fsdp-hierarchical-sharding .block width=100% align=center}

如 @fig:fsdp-hierarchical-sharding 所示，分层切分（左）将每个 Block 作为独立通信单元，任何时刻显存中仅存在单个 Block 的完整参数；而扁平切分（右）被迫一次性重构全局参数。

---

## 完整生产级实战案例：基于 FSDP 的 FLAN-T5 摘要训练

本章在 `code/FSDP/` 目录下提供了完整的工业级对比脚本，基于 HuggingFace **FLAN-T5 (3B / 11B)** 进行文本摘要训练基准对比：
- `T5_training_Single.py`：单 GPU 训练基线
- `T5_training_FSDP1.py`：FSDP1 初代包装器实现
- `T5_training_FSDP2.py`：FSDP2 现代化 API + DCP 存储实现

### 真实硬件评测性能对比（基于 NVIDIA H200 141GB）

在处理 **FLAN-T5-XL (3B 参数)** 模型时，单卡与双卡 FSDP1 对比如下：

| 运行模式 | GPU 卡数 | 单卡显存占用 | 峰值显存占用 | 训练吞吐量 | 单 Epoch 耗时 |
|:---|:---|:---|:---|:---|:---|
| 单卡 Single | 1 卡 | ~43 GB | ~58 GB | ~4.36 it/s | ~91 s |
| FSDP1 | 2 卡 | ~22 GB | ~33 GB | ~4.09 it/s | ~49 s |

Table: FLAN-T5-XL (3B 参数) 单卡 vs 2 卡 FSDP 性能指标对比 {#tab:fsdp-t5-xl-comparison}

当模型规模扩大至 **FLAN-T5-XXL (11B 参数)** 时，单卡 140 GB 显存直接发生 OOM 崩溃，而双卡 FSDP1 与 FSDP2 均能稳定高效完成训练：

| 运行模式 | GPU 卡数 | 单卡显存占用 | 峰值显存占用 | 训练吞吐量 | 单 Epoch 耗时 |
|:---|:---|:---|:---|:---|:---|
| 单卡 Single | 1 卡 | **OOM 溢出** | **OOM 溢出** | — | — |
| FSDP1 | 2 卡 | ~84 GB | ~105 GB | ~1.96 it/s | ~101 s |
| FSDP2 | 2 卡 | ~84 GB | ~105 GB | ~1.86 it/s | ~106 s |

Table: FLAN-T5-XXL (11B 参数) 在 H200 上的极限评测对比 {#tab:fsdp-t5-comparison}

如 @tab:fsdp-t5-comparison 所示，FSDP1 与 FSDP2 在吞吐与显存上表现相当，但 FSDP2 的核心优势在于更简短优雅的代码结构、对 `torch.compile` 的深度支持以及基于 DCP 的零通信高吞吐检查点存盘能力。

---

## 基于分布式检查点（DCP）的高性能并行存盘

在传统 DDP 或 FSDP1 中，保存 Checkpoint 通常需要将所有参数 All-Gather 汇聚到 Rank 0 形成一个完整的 `state_dict`，再由 Rank 0 写入磁盘。在千亿模型下，这不仅会导致 Rank 0 发生单点显存 OOM，还会让多路 I/O 沦为单盘写入瓶颈。

FSDP2 推荐全面转向 **Distributed Checkpoint (DCP, `torch.distributed.checkpoint`)**：**所有 Rank 并发将自身持有的本地分片并行写入磁盘对应切片中**，实现绝对的零集合通信与多节点并行 I/O 满血吞吐。

```python
import os
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
)

def save_checkpoint_dcp(model, optimizer, epoch, checkpoint_dir):
    """使用 DCP 并行保存分片检查点（无 Rank 0 汇总瓶颈）。"""
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    optim_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    dcp.save(
        {"model": model_state_dict, "optimizer": optim_state_dict, "epoch": epoch},
        checkpoint_id=checkpoint_path
    )

def load_checkpoint_dcp(model, optimizer, checkpoint_dir, epoch):
    """使用 DCP 加载分片检查点（支持在不同 GPU 卡数拓扑间动态重分片加载）。"""
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    # 构造当前模型与优化器的分片状态模板
    state_dict = {
        "model": get_model_state_dict(model, options=StateDictOptions(full_state_dict=False)),
        "optimizer": get_optimizer_state_dict(model, optimizers=optimizer, options=StateDictOptions(full_state_dict=False)),
        "epoch": 0
    }
    # DCP 根据当前拓扑就地填充张量数据
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    set_model_state_dict(model, state_dict["model"], options=StateDictOptions(full_state_dict=False))
    set_optimizer_state_dict(model, optimizers=optimizer, optim_state_dict=state_dict["optimizer"], options=StateDictOptions(full_state_dict=False))
    return state_dict["epoch"]
```

---

## 性能进阶调优：预取、激活值重计算与 CPU Offloading

### 1. 通信预取（Prefetching）隐藏 All-Gather 延迟

FSDP 支持**前向与反向通信预取**：在当前计算流执行第 $L$ 层算子的同时，利用后台独立的 CUDA 通信 Stream 提前向网络发起第 $L+1$ 层的 All-Gather 参数拉取请求。当第 $L$ 层计算结束时，第 $L+1$ 层的参数已经就绪，实现了通信延迟的完全隐藏。

![FSDP 开启与未开启通信预取的时序流对比](img/fsdp_prefetch_timeline.png){#fig:fsdp-prefetch-timeline .block width=100% align=center}

如 @fig:fsdp-prefetch-timeline 所示，未开启预取时（上），每层计算前必须串行等待 All-Gather 完成；开启预取后（下），$L_1$ 的 All-Gather 在 $L_0$ 计算期间并发执行，消除等待空隙。

### 2. 激活值检查点（Activation Checkpointing）

Transformer 模型的激活值显存占用粗略估算公式为：
$$\text{Activation Memory} \approx L \times B \times S \times H \times \text{BytesPerElement} \times k$$
对于 7B 模型（$L=32, H=4096$），在 Batch=8, SeqLen=2048 下激活值显存超过 **52 GB**。通过开启激活值检查点（仅保存每个 Block 输入，反向时重新计算内部激活），可削减 **60%–80%** 的激活值显存。

### 3. CPU Offloading（以带宽换容量的终极手段）

当显存极度紧缺时，可通过 `CPUOffloadPolicy` 将冻结参数或优化器状态卸载至 Host 内存中：

```python
from torch.distributed.fsdp import CPUOffloadPolicy

fully_shard(
    model,
    mesh=mesh,
    offload_policy=CPUOffloadPolicy(pin_memory=True),
)
```

注意：CPU Offloading 受限于 PCIe 总线带宽，通常会导致 20%–50% 的训练速度下降，应作为显存优化的最后兜底手段。

---

## DDP vs ZeRO vs FSDP2 选型全景指南

| 特性维度 | PyTorch DDP | DeepSpeed ZeRO-3 | PyTorch FSDP2 |
|:---|:---|:---|:---|
| **参数分片机制** | 无分片（全量复制） | 扁平拼接分片 | **按参数维度原生分片 (DTensor)** |
| **显存节约能力** | 基准（无显存节省） | 极高（消除所有状态冗余） | **极高（消除所有状态冗余）** |
| **框架依赖** | PyTorch 原生内置 | 需安装 DeepSpeed 第三方库 | **PyTorch 2.4+ 原生内置** |
| **`torch.compile` 支持** | 良好 | 较弱/需专用兼容补丁 | **原生深度融合与算子合并优化** |
| **检查点存盘机制** | Rank 0 串行单点写入 | 转换脚本或分片存储 | **DCP 多卡并行零通信秒级存盘** |
| **首选应用场景** | 模型可放入单卡显存 | 深度使用 DeepSpeed 进阶生态 | **现代大模型训练与全量微调首选** |

---

## 本章小结

FSDP2 代表了 PyTorch 在超大规模模型分布式训练领域的最高工程水准。本章系统阐述了：
- FSDP 从 DDP 跨越的核心原理：基于 All-Gather 与 Reduce-Scatter 消除显存冗余；
- FSDP1 与 FSDP2 的代际差异：从扁平拼接走向基于 DTensor 的按参数原生分片；
- 基于 `DeviceMesh`、分层切分（Hierarchical Sharding）与 `reshard_after_forward` 的参数调优；
- 基于 DCP 的零通信高并发分布式检查点读写系统；
- 预取流水线（Prefetching）、激活值重计算与 CPU Offload 的系统级协同。

FSDP 将显存瓶颈从“单卡物理上限”彻底转变为“集群总显存池上限”。然而，对于千亿参数级别的大模型或超长上下文推理场景，除了数据维度的状态切分外，我们还需要探索将单层矩阵运算直接横向拆解的**张量并行（Tensor Parallelism）**与**流水线并行（Pipeline Parallelism）**。在下一章中，我们将深入剖析 **DeepSpeed 与 Megatron-LM**，探索构建 3D 混合并行的终极武器库。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
