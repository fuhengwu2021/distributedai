# 第4章：使用完全分片数据并行（FSDP）进行扩展 {-}

*用参数分片训练比单 GPU 内存更大的模型*

> 数据中心是计算的新单元。
- 黄仁勋（Jensen Huang），NVIDIA CEO

**Code Summary**

- `fsdp.fully_shard()`：用每参数分片对模块进行分片（DTensor）
- `fsdp.MixedPrecisionPolicy`：混合精度配置
- `device_mesh.init_device_mesh()`：为分片创建设备网格
- `distributed.checkpoint`：用于保存/加载分片状态字典的 DCP API
- `FullyShardedDataParallel`：带扁平参数分片的包装类（v1）
- `fsdp.wrap()`：将子模块包装为单独的 FSDP 单元（v1）
- `fsdp.MixedPrecision`：混合精度配置（v1）
- `set_state_dict_type()`：为检查点配置状态字典类型（v1）
- `fsdp.StateDictConfig` / `OptimStateDictConfig`：状态字典配置（v1）
- `apply_activation_checkpointing`：在分片之前包装选定的子模块

## 从 DDP 到 FSDP

**完全分片数据并行（FSDP）** 是一种训练策略，它将模型参数、梯度和优化器状态分片到多个设备上，使每个设备只持有完整模型的一部分。在第~\ref{chap:distributed-training-with-pytorch-ddp}章，我们使用了 DDP，它在每块 GPU 上复制整个模型——当模型适合单块 GPU 的内存时有效。当模型（加上梯度和优化器状态）超过那个内存时，DDP 不再可行。FSDP 通过将模型及其训练状态分布到 GPU 上来解决这个问题，使你能训练比任何单个设备内存更大的模型。

PyTorch 为 GPU 训练提供了两个主要的 FSDP API，外加一个用于 TPU 的单独实现：

- **FSDP1**（`FullyShardedDataParallel`）：使用扁平参数方法的原始包装类。
- **FSDP2**（`fully_shard()`）：较新的每参数分片设计，通过 `fully_shard()` 访问。更简单、更灵活，也是 PyTorch 前进的方向。
- **通过 SPMD 的 FSDP**（`SpmdFullyShardedDataParallel`）：用于 XLA/TPU 设备，使用 GSPMD 进行自动并行化。

贯穿本章，当讨论适用于两个 API 的通用技术时——分片参数、all-gather、reduce-scatter——我们使用 **FSDP**（不带数字）。当区别重要时，我们明确说 **FSDP1** 或 **FSDP2**。

本章专注于 GPU 训练的 FSDP2——它是 CUDA 设备上新项目的推荐方法。FSDP1 仍然有效，并在生产代码库中继续使用；例如，Wan2.2 使用 PyTorch FSDP 配合 DeepSpeed Ulysses 进行多 GPU 推理。[^wan22] 我们在下面总结 FSDP1，然后专注于 FSDP2。对于 TPU 训练，见第~\ref{sec:fsdp-spmd}节。

[^wan22]: <https://github.com/Wan-Video/Wan2.2>
[^fsdp2-rfc]: <https://github.com/pytorch/pytorch/issues/114299>
[^t5-flan]: **T5**（Text-to-Text Transfer Transformer）是 Google 的一个编码器-解码器模型，它将 NLP 任务表述为文本到文本。**FLAN-T5** 是 T5 模型的指令微调系列（如 flan-t5-small、flan-t5-xl、flan-t5-xxl），用于摘要和问答等任务。

## 为什么 FSDP 能训练大于内存的模型

基本思想很直接：不是在每块 GPU 上保留完整模型，而是将它拆分。每块 GPU 持有参数的一个分片。在前向传播期间，你 __all-gather__ 你需要的参数。在反向期间，你在本地分片上计算梯度，然后 __reduce-scatter__ 以跨 GPU 聚合。

但让我们深入探讨为什么这很重要。用 DDP 训练大型模型时，每块 GPU 需要存储：

1. **模型参数**：权重本身。对于 BF16 的 7B 参数模型，那是 7B × 2 字节 = 14 GB。
2. **梯度**：与参数相同大小。BF16 的另 14 GB。
3. **优化器状态**：对于 Adam，动量和方差是参数大小的 2 倍（FP32）。那是 7B × 4 字节 × 2 = 56 GB。
4. **激活值**：取决于批大小和序列长度，但对大型模型容易达到几十 GB。

所以对于带 Adam 的 7B 模型，仅参数、梯度和优化器状态就是 14 + 14 + 56 = 84 GB 每 GPU——超过 80 GB H100 所能容纳的，激活值还没算。混合精度 Adam 还保留一个 FP32 的权重主副本（对 7B 多约 28 GB），所以现实的 DDP 占用更接近 112 GB。用 FSDP，这些组件跨 GPU 分片：每个设备持有每项的 1/N（N = GPU 数量）。用 8 块 GPU，上面三项就是 84 / 8 = 10.5 GB 每 GPU，为激活值留下余量。实践中，那就是用 FSDP 在 8 块 GPU 上装下同一个 7B 模型，与用 DDP 在单块 80 GB GPU 上装不下之间的区别。

![每 GPU 内存：7B 模型的 DDP vs FSDP。](img/ddp_fsdp_mem_zh.png){#fig:ddp-fsdp-mem .block width=100% align=center}

图~\ref{fig:ddp-fsdp-mem} 展示了对比：用 DDP，每块 GPU 持有完整的 84 GB（参数、梯度和优化器状态）并超过 80 GB 设备；用 FSDP，每 GPU 内存降为 84/N，在 8 块 GPU 时每 GPU 的 10.5 GB 为激活值留下空间。

>NOTES: **激活值不被分片**

FSDP 只分片参数、梯度和优化器状态——不分片激活值。每块 GPU 在前向和反向期间仍然存储它那份批次的激活值，所以激活内存仍然是每 GPU 的成本。激活重计算（在反向中重新计算激活值而非存储它们）等技术常与 FSDP 一起使用，为临时 all-gather 的参数腾出余量。

>NOTEE

### FSDP 如何工作

FSDP 背后的核心思想来自 Microsoft Research 的 ZeRO（零冗余优化器）论文（2019）。[^zero-paper] ZeRO 观察到，在数据并行训练中，每块 GPU 持有模型、梯度和优化器状态的完整副本——其中大部分是冗余的。通过跨 GPU 划分这些并仅在需要时收集它们，你可以训练大得多的模型，而无需改变底层的数据并行算法。我们在第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章详细介绍 ZeRO；这里我们专注于 PyTorch 对这些思想的原生实现。

[^zero-paper]: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," SC 2020. <https://arxiv.org/abs/1910.02054>

PyTorch 的 FSDP 使用两个集合操作实现这个思想。在前向传播期间，当一个层需要它的参数时，FSDP 从所有 GPU **all-gather** 它们——临时重构完整的参数张量（见第~\ref{chap:introduction-to-modern-distributed-ai}章的第~\ref{sec:allgather}节）。层完成后，收集的参数被释放。在反向传播期间，梯度在完整（临时收集）的参数上本地计算，然后跨 GPU **reduce-scatter**，使每块 GPU 最终得到聚合梯度的它那个分片（见第~\ref{chap:introduction-to-modern-distributed-ai}章的第~\ref{sec:reducescatter}节）。

![FSDP：前向中 All-Gather，反向中 Reduce-Scatter。](img/fsdp_allgather_reducescatter_zh.png){#fig:fsdp-allgather-reducescatter .block width=100% align=center}

图~\ref{fig:fsdp-allgather-reducescatter} 展示了这两个步骤。在左面板（前向），每个 rank 持有一个参数分片（$1/N$）；All-Gather 之后，每个 rank 临时拥有完整参数。在右面板（反向），每个 rank 拥有完整梯度；Reduce-Scatter 之后，每个 rank 只保留归约梯度的它那个分片（$1/N$）。

关键洞见是你不需要一次性拥有所有参数。神经网络顺序处理层——前向通过层 1，然后层 2，以此类推。FSDP 利用这一点，为当前层 all-gather 参数，使用它们，然后在移动到下一层之前释放它们。这就是为什么激活重计算与 FSDP 配合良好：它减少激活内存，为临时 all-gather 的参数留出空间。

### FSDP1 vs FSDP2：演进

![FSDP1 vs FSDP2 参数布局。](img/fsdp1_vs_fsdp2_layout_zh.png){#fig:fsdp1-vs-fsdp2-layout .block width=100% align=center}

PyTorch 的原始 FSDP（2021 年发布，常称为 FSDP1）使用了从 FairScale 实现借来的 **扁平参数（flat-parameter）** 设计。它将包装模块中的所有参数扁平化为单个连续的 `FlatParameter` 张量，然后将该张量跨 GPU 分片。这可以工作，但有局限：一个组中的所有参数必须共享相同的 dtype，冻结参数需要单独的组，且扁平化使编译器更难优化通信模式。

**FSDP2**（2024 年通过 PyTorch RFC #114299 引入[^fsdp2-rfc]）采取了不同的方法：使用带 `Shard(0)` 的 DTensor 进行 **每参数分片**。它不是扁平化，而是在维度 0 上单独分片每个参数张量。一个形状为 $(4096, 1024)$ 的线性层权重，在 4 块 GPU 上变成四个形状为 $(1024, 1024)$ 的分片——每个 rank 持有四分之一的行。没有扁平化，没有 `FlatParameter` 类。当维度 0 不能被 world size 整除时，FSDP2 填充张量；非常小的参数可能被复制而非分片。

图~\ref{fig:fsdp1-vs-fsdp2-layout} 展示了区别。FSDP1（左）在跨 rank 分片之前将参数 W1、W2、W3 拼接成单个扁平张量。FSDP2（右）在维度 0 上独立分片每个参数——每个 rank 持有每个参数的一片。

每参数设计更简单（约 3k 行代码，而 FSDP1 约 14k）且更灵活。你可以混合 dtype（一些参数用 fp8，其他用 bf16），将冻结和可训练参数保持在同一组中，无需收集就能保存分片检查点，并让编译器看见各个参数以进行更好的优化。集合操作相同——All-Gather 和 Reduce-Scatter——但参数布局根本不同。

## FSDP1：原始的包装 API

`torch.distributed.fsdp` 中的包装类 `FullyShardedDataParallel` 实现了上面的扁平参数设计。用法类似于 DDP：包装模型（或通过 `wrap()` 包装子模块），然后照常训练。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 style
    device_id=torch.cuda.current_device(),
)
```

你可以使用 `FSDP.set_state_dict_type()` 和 `StateDictConfig` / `OptimStateDictConfig` 进行检查点；混合精度通过 `MixedPrecision` 配置。

虽然 FSDP2 是新项目的推荐 API，FSDP1 仍出现在我们合作过的生产技术栈中——**Wan2.2**[^wan22]，一个开源视频生成模型，就是一个例子。该代码库中的三个模式解释了为什么团队保留 FSDP1 而非一夜之间将所有东西迁移到 FSDP2。

首先，Wan2.2 集成 DeepSpeed Ulysses 以对高分辨率视频帧进行序列并行。Ulysses 依赖 all-to-all 注意力模式，在 Wan2.2 中，它们是针对 FSDP1 的进程组钩子接线的。FSDP2 的 `DeviceMesh` 是多维并行的长期答案，但混合的 FSDP + Ulysses 技术栈常常保留 FSDP1，直到等效的 FSDP2 路径被端到端重新验证。

其次，混合专家布局（27B 总参数 / 14B 活跃参数）在专家跨时间步被交换或卸载时需要显式控制。FSDP1 的 `ModuleWrapPolicy` 使那些包装边界显而易见；我们发现对于同样的非均匀 MoE 模式，那比早期的 FSDP2 `fully_shard` 布局更容易调试。

第三，流水线包括一个冻结的 UMT5-XXL 文本编码器和可训练的视频块。本章的 FSDP1 训练脚本（`T5_training_FSDP1.py`）遵循相同的包装与策略风格；将 Wan2.2 规模的流水线迁移到 FSDP2 意味着重新验证编码器集成，而不只是换掉分片 API。

当你遇到或扩展这类项目时，理解 FSDP1 的包装类风格和扁平参数行为变得至关重要。

## FSDP2：每参数分片 API

对于新项目，FSDP2 提供了更简洁的设计。FSDP2 不是将模块包装在一个类中，而是使用 `fully_shard()` 作为一个就地修改模块的函数——更函数式且可组合。这与原始 FSDP 有显著不同，后者使用类似 DDP 的包装类。

API 看起来像这样。一个完整的可运行示例在 `code/train_fsdp2.py`：

```bash
torchrun --nproc_per_node=2 code/train_fsdp2.py
```

核心模式：

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

# Initialize device mesh
mesh = init_device_mesh("cuda", (world_size,))

# Apply FSDP to your model
fully_shard(
    model,
    mesh=mesh,
    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
)
```

与原始 FSDP 的关键区别是 `fully_shard()` 就地修改模型。它不返回一个包装后的模型——你的模型变成一个 FSDP 模型。这使它更容易与其他变换组合，也与 `torch.compile` 配合得更好。

关于带分层分片的最小 transformer 示例（每个块单独包装），见 `code/fsdp2_basic.py`：

```bash
torchrun --nproc_per_node=2 code/fsdp2_basic.py
```

### 设备网格：基础

`DeviceMesh` 是 PyTorch 中一个新的抽象，表示设备的逻辑排列。对于 FSDP，你通常使用 1D 网格（所有 GPU 在单一维度中）：

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh for standard FSDP (4 GPUs)
mesh = init_device_mesh("cuda", (4,))
```

这创建一个所有 GPU 排列在单一维度中的网格。对于 4 块 GPU，这是 `[0, 1, 2, 3]`。

对于非常大的集群，跨所有 GPU 的完全分片会产生过多的跨节点通信。混合分片数据并行（HSDP）通过仅在每个节点内分片参数、而跨节点复制来解决这个问题——以一些内存换取减少的节点间流量。我们在第~\ref{sec:hsdp}节详细介绍 HSDP；现在，这是如何设置 2D 网格：

```python
# 2D mesh for hybrid sharding
# 2 nodes × 4 GPUs per node = 8 GPUs total
mesh = init_device_mesh("cuda", (2, 4))
```

这将 GPU 排列在 2D 网格中，这对你想在节点内分片但跨节点复制的超大规模训练很有用。

![设备网格：1D（FSDP）vs 2D（HSDP）。](img/device_mesh_zh.png){#fig:device-mesh .block width=100% align=center}

图~\ref{fig:device-mesh} 显示了两种网格配置。在 1D 网格（左面板）中，标记为 R0–R3 的四块 GPU 形成单个分片组——每块 GPU 持有每个参数的不同分片。在 2D 网格（右面板）中，N0 和 N1 表示两个物理节点（如通过 InfiniBand 连接的两台服务器）。在每个节点内，GPU 沿维度 1 分片（绿色箭头），所以 N0 中的 R0–R3 各持有不同的参数分片。跨节点，相同位置的 GPU 沿维度 0 共享相同的分片（红色箭头）——N0 中的 R0 和 N1 中的 R4 持有相同的数据。这种混合方法将繁重的 all-gather 流量保持在快速的节点内互连（NVLink）中，而只在较慢的节点间网络上交换梯度。

### 关键参数

`fully_shard()` 函数接受几个控制分片行为的参数。`mesh` 参数指定分片所在的 `DeviceMesh`——通常是标准 FSDP 的 1D 网格，或混合分片（HSDP）的 2D 网格。

最重要的参数是 `reshard_after_forward`，它控制内存-通信权衡。当设置为 `True`（默认）时，参数在每层前向传播后立即重新分片，释放内存但在反向期间需要额外的 all-gather。这对应 ZeRO-3 行为。将它设为 `False` 使参数在前向后保留在内存中，这使用更多内存但消除了反向 all-gather——类似 ZeRO-2。你也可以传入一个整数以重新分片到中间大小；例如，`reshard_after_forward=2` 只跨 2 块 GPU 分片而非所有 GPU，模仿 ZeRO++ 的混合参数 zero（hpZ）。

对于大多数内存受限的场景，默认 `True` 是正确的选择。如果你有内存余量且通信是你的瓶颈，试试 `False`。

![reshard_after_forward：True vs False。](img/reshard_after_forward_zh.png){#fig:reshard-after-forward .block width=100% align=center}

图~\ref{fig:reshard-after-forward} 比较了两种模式。每行显示一个两层模型的前向（Fwd）和反向（Bwd）传播。彩色块表示：AG（All-Gather，紫色）用于收集分片参数，L1/L2（计算，黄色）用于层计算，RS（Reduce-Scatter，红色）用于分发梯度。用 `reshard_after_forward=True`（上），参数在每层前向传播后被释放（标记 "free"），并且必须在反向中再次 all-gather——这使内存保持低但将 all-gather 通信翻倍。用 `False`（下），参数在前向后保留在内存中（标记 "keep"），所以反向传播完全跳过 all-gather——更高的峰值内存但更少的通信。

混合精度通过 `mp_policy` 配置。你为参数存储指定 `param_dtype`（如 `torch.bfloat16`），为梯度归约指定 `reduce_dtype`（常为数值稳定性用 `torch.float32`），以及可选的为层输出指定 `output_dtype`：

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
```

因为 FSDP2 按参数分片而非扁平化为单个缓冲区，你可以自由混合 dtype——一些层用 fp8，其他用 bf16。原始 FSDP 要求一个组中的所有参数共享相同的 dtype。

最后，`offload_policy` 在 GPU 内存耗尽时启用 CPU 卸载：

```python
from torch.distributed.fsdp import CPUOffloadPolicy

fully_shard(
    model,
    mesh=mesh,
    offload_policy=CPUOffloadPolicy(pin_memory=True),
)
```

卸载引入性能成本，所以在耗尽其他内存优化后将它作为最后手段。减速幅度因 PCIe 代数、CPU 内存带宽和优化器状态大小差异很大——CPU 卸载的粗略经验范围是 20-50%。

### 分层分片

除了这些每次调用的参数，FSDP2 让你控制在模型层次结构中 *哪里* 放置分片边界。你可以在不同级别应用 `fully_shard()`——例如，包装各个 transformer 块，同时不分片嵌入层：

```python
# Shard each transformer layer individually
for layer in model.transformer.layers:
    fully_shard(layer, mesh=mesh)

# Don't shard the embedding layer (it's small)
# fully_shard(model.embedding, mesh=mesh)  # Skip this
```

这给你对什么被分片的细粒度控制。小层（如嵌入）可能不会从分片中受益，且会增加通信开销，所以你可以不分片它们。

图~\ref{fig:fsdp-hierarchical-sharding} 对比了两种方法。两个面板都显示一个带嵌入层（Embed）、四个 transformer 块（Block 0–3）和一个输出头（Head）的 transformer 模型。红色虚线框表示 FSDP 单元边界。用分层分片（左），每个 transformer 块通过在循环中调用 `fully_shard(block)` 被包装为单独的 FSDP 单元，而嵌入和头保持不分片。这意味着 all-gather 和 reduce-scatter 在块边界发生，实现预取（下一个块的参数可以在当前块计算时收集）和细粒度内存管理（一次只需要一个块的完整参数在内存中）。用扁平分片（右），单个 `fully_shard(model)` 调用将整个模型包装为一个 FSDP 单元。这更简单但需要一次性收集所有参数，导致更高的峰值内存。

![分层 vs 扁平分片。](img/fsdp_hierarchical_sharding_zh.png){#fig:fsdp-hierarchical-sharding .block width=100% align=center}

## 一个完整的可运行示例：用 FSDP 进行 T5 摘要

`code/FSDP/` 中一个完整的可运行示例用 FSDP1 和 FSDP2 训练 **T5（FLAN-T5）**[^t5-flan] 进行文本摘要，包括检查点、混合精度和示例训练日志。

该示例提供三个入口点，使你可以在相同的任务和模型上比较单 GPU 训练、FSDP1 和 FSDP2：

- **FSDP1**：`T5_training_FSDP1.py` 用 `FullyShardedDataParallel`、`ShardingStrategy.FULL_SHARD`、混合精度和 FSDP1 风格的状态字典处理包装 FLAN-T5。在处理仍依赖原始 FSDP API 的代码库时使用它。
- **FSDP2**：`T5_training_FSDP2.py` 使用 `fully_shard()` 和分布式检查点（DCP）API；这是新项目的推荐脚本。
- **单 GPU 基线**：`T5_training_Single.py` 在一块 GPU 上训练相同的模型（无 FSDP），对检查正确性以及比较内存和吞吐量有用。

所有脚本都在 `code/FSDP/` 中。运行训练脚本之前，从 `code/FSDP/` 目录运行以下命令下载 WikiHow 数据集：

```
bash download_dataset.sh
```

将 CSV 文件获取到 `data/`。那里的 README 描述了模型选择（flan-t5-small 到 flan-t5-xxl）、VRAM 需求和命令行选项。

**运行示例。** 单 GPU 基线：

```bash
python code/FSDP/T5_training_Single.py
```

在 2 块 GPU 上的 FSDP1：

```bash
torchrun --nnodes 1 --nproc_per_node 2 code/FSDP/T5_training_FSDP1.py
```

在 2 块 GPU 上的 FSDP2：

```bash
torchrun --nnodes 1 --nproc_per_node 2 code/FSDP/T5_training_FSDP2.py
```

更大的模型（如 `--model-name google/flan-t5-xl` 或 `google/flan-t5-xxl`）可能需要更小的批大小。

**比较。** 下面的数字来自 H200 GPU 上的示例运行。单 GPU 训练将完整模型保持在一个设备上；当模型适合时，它可以有每 GPU 最高的迭代吞吐量（it/s），但用 2 块 GPU，由于批次被分布（如 XL：49 s vs 91 s），epoch 以更少的挂钟时间完成。单 GPU 无法扩展到超过一块 GPU 内存的模型，如 FLAN-T5-XXL。FSDP1 和 FSDP2 将参数、梯度和优化器状态跨 GPU 分片，所以每 GPU 内存下降，可以训练更大的模型。

对于较小的 FLAN-T5-XL（3B）模型，单 GPU 训练适合一块 H200；用 2 块 GPU，FSDP1 将每 GPU 内存大致减半，并以更少的挂钟时间完成每个 epoch（49 s vs 91 s）。那个加速主要来自将批次分散到两块 GPU 上——每 GPU 迭代率（it/s）几乎不变——而非 FSDP 使每一步更快。表~\ref{tab:fsdp-t5-xl-comparison} 给出了数字。

| 模式   | GPU 数 | 内存/GPU | 峰值内存/GPU | 吞吐量 | 时间/epoch |
|--------|------|---------|----------------------|------------|------------|
| 单卡 | 1    | ~43 GB  | ~58 GB       | ~4.36 it/s | ~91 s      |
| FSDP1  | 2    | ~22 GB  | ~33 GB       | ~4.09 it/s | ~49 s      |

Table: FLAN-T5-XL (3B)：单 GPU vs FSDP1（2 块 GPU）。H200 GPU 上的示例运行。 {#tab:fsdp-t5-xl-comparison}

对于 FLAN-T5-XXL（11B 参数），单 GPU 训练即使在 H200（140 GB）上也会内存不足（OOM）。表~\ref{tab:fsdp-t5-comparison} 比较了 2 块 GPU 上的 FSDP1 和 FSDP2。

| 模式   | GPU 数 | 内存/GPU | 峰值内存/GPU | 吞吐量 | 时间/epoch |
|--------|------|---------|----------------------|------------|------------|
| 单卡 | 1    | OOM     | OOM          | —          | —          |
| FSDP1  | 2    | ~84 GB  | ~105 GB      | ~1.96 it/s | ~101 s     |
| FSDP2  | 2    | ~84 GB  | ~105 GB      | ~1.86 it/s | ~106 s     |

Table: 单 GPU、FSDP1 和 FSDP2 在 FLAN-T5-XXL（11B）上的比较。H200 GPU 上的示例运行。 {#tab:fsdp-t5-comparison}

该表显示在此设置下 FSDP1 和 FSDP2 的内存和吞吐量几乎相同。这里选择 FSDP2 的理由不是原始速度——而是更简单的 API、更紧密的 `torch.compile` 集成，以及通过 DCP 的分片检查点，随着作业和代码库增长它们更重要。

**代码分析：** 以下代码片段展示了 T5 示例如何实现加载、分层分片和混合精度。

*FSDP1：策略和包装器。* 在 FSDP1 中，**策略（policy）** 是你传入包装器的配置对象：**混合精度策略** 为参数和梯度指定 dtype（如 bfloat16）以节省内存并加速计算；**包装策略** 是一个可调用对象，告诉 FSDP 将哪些子模块包装为单独的 FSDP 单元（这里是每个 `T5Block`），使分片是分层的，all-gather/reduce-scatter 在块边界发生。脚本从 `get_policies` 获取两者，然后用 `FSDP` 包装类包装已加载的模型。下面的代码片段来自 `T5_training_FSDP1.py` 和 `policies/wrapping.py`。

```python
# get_policies (T5_training_FSDP1.py): mixed precision + wrap policy
def get_policies(cfg, rank):
    mixed_precision_policy = None
    if cfg.mixed_precision:
        # e.g. policies.bfSixteen for bfloat16
        mixed_precision_policy = policies.bfSixteen
    wrapping_policy = policies.get_t5_wrapper()  # targets T5Block
    return mixed_precision_policy, wrapping_policy

# Wrap policy (policies/wrapping.py): wrap each T5Block as an FSDP unit
def get_t5_wrapper():
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={T5Block},
    )

# Apply FSDP (T5_training_FSDP1.py): model already from setup_model()
mixed_precision_policy, t5_auto_wrap_policy = get_policies(train_config, rank)
model = FSDP(model,
    auto_wrap_policy=t5_auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=fsdp_config.sharding_strategy,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=fsdp_config.limit_all_gathers)
```

*FSDP2：混合精度策略。* FSDP2 使用 `MixedPrecisionPolicy`（而非 FSDP1 的 `MixedPrecision` 对象）。脚本调用 `get_policies(train_config, rank)` 从配置构建策略；该策略然后通过 `fsdp_kwargs["mp_policy"]` 传入每个 `fully_shard(...)` 调用。下面的代码片段来自 `T5_training_FSDP2.py`。

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def get_policies(cfg, rank):
    """Establish mixed precision policy for FSDP2 (no wrap policy; sharding is explicit)."""
    mp_policy = None
    if cfg.mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not cfg.use_fp16:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            )
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using MixedPrecisionPolicy")
        elif cfg.use_fp16:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
            )
    return mp_policy

# In fsdp_main: get policy once, then pass into every fully_shard call
mp_policy = get_policies(train_config, rank)
fsdp_kwargs = {}
if mp_policy is not None:
    fsdp_kwargs["mp_policy"] = mp_policy
# ... later: fully_shard(block, **fsdp_kwargs) and fully_shard(model, **fsdp_kwargs)
```

*FSDP2：加载模型，然后分片每个块和根。* 脚本用 `from_pretrained` 加载权重并将模型移到设备——这条路径只在检查点在分片前适合一块 GPU 时有效（例如 H200 上 11B 的 FLAN-T5-XXL）。比你最大设备更大的模型应在 meta 设备上构建，用 `fully_shard` 分片，然后按 rank 物化（第~\ref{sec:fsdp-initialization-best-practices}节）。没有自动包装策略：脚本遍历编码器和解码器块并对每个 `T5Block` 调用 `fully_shard`，然后对根调用。子模块必须在根之前分片。下面的代码片段来自 `T5_training_FSDP2.py`。

```python
from torch.distributed.fsdp import fully_shard

model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)

# fsdp_kwargs already holds mp_policy from get_policies()

# Shard encoder blocks (each T5Block becomes one FSDP unit)
if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
    for block in model.encoder.block:
        fully_shard(block, **fsdp_kwargs)
# Shard decoder blocks
if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
    for block in model.decoder.block:
        fully_shard(block, **fsdp_kwargs)
# Shard the entire model (root); children must already be sharded
fully_shard(model, **fsdp_kwargs)
```

这个示例省略了 `mesh` 参数——在单个进程组中，`fully_shard()` 从 world size 推断默认网格。对于 HSDP 或其他多维布局，像前面章节那样显式传入 `mesh`。

## 使用 FSDP2 进行检查点

一旦你的模型被分片并在训练，你会想保存检查点。长时间运行可能失败——硬件错误、抢占、bug——丢失数天的进度令人痛苦。用 FSDP2，检查点比以前更简单：因为分片状态字典匹配训练表示，每个 rank 只直接保存它的分片。无需收集到 rank 0，加载时无需重新分片。保存和加载在本地进行，这更快且使用更少的内存。

有两种方法：使用分布式检查点（DCP）API，或手动处理分片状态字典。

### 使用 DCP API（推荐）

DCP API 是保存和加载 FSDP2 检查点的推荐方式。它处理分片状态字典的所有复杂性。一个完整的可运行示例在 `code/fsdp2_checkpoint_dcp.py`：

```bash
torchrun --nproc_per_node=2 code/fsdp2_checkpoint_dcp.py
```

关键函数是 `save_checkpoint_dcp` 和 `load_checkpoint_dcp`：

```python
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
)
import torch.distributed.checkpoint as dcp

def save_checkpoint_dcp(model, optimizer, epoch, checkpoint_dir):
    """Save checkpoint using DCP API."""
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    optim_state_dict = get_optimizer_state_dict(
        model=model, optimizers=optimizer,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True),
    )
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    dcp.save({"model": model_state_dict, "optimizer": optim_state_dict, "epoch": epoch},
             checkpoint_id=checkpoint_path)

def load_checkpoint_dcp(model, optimizer, checkpoint_dir, epoch):
    """Load checkpoint using DCP API."""
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    opt = get_optimizer_state_dict(model, optimizers=optimizer, options=StateDictOptions(full_state_dict=False))
    state_dict = {"model": get_model_state_dict(model, options=StateDictOptions(full_state_dict=False)),
                  "optimizer": opt,
                  "epoch": 0}
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    set_model_state_dict(model, state_dict["model"], options=StateDictOptions(full_state_dict=False))
    set_optimizer_state_dict(model, optimizers=optimizer, optim_state_dict=state_dict["optimizer"],
                             options=StateDictOptions(full_state_dict=False))
    return state_dict["epoch"]
```

保存路径很直接：每个 rank 用 `dcp.save` 写它的分片。加载不那么明显——`dcp.load` **不** 返回一个新的状态字典。相反，你通过对此 rank 上已初始化、已分片的模型和优化器调用 `get_model_state_dict` 和 `get_optimizer_state_dict` 来构建一个 **模板**。将该字典传给 `dcp.load`；DCP 从检查点分片 **就地** 填充每个条目。模板告诉 DCP 期待什么键、形状和分片布局；跳过它——在模型于每个 rank 上构建和分片之前加载——是形状不匹配或未初始化参数错误的常见来源。最后的 `set_model_state_dict` 和 `set_optimizer_state_dict` 调用完成 PyTorch 记录的加载路径；它们对优化器状态尤其需要，即使模型参数已通过模板引用就地更新。无需收集到 rank 0，无需广播——每个 rank 只读它的分片。

### 手动分片检查点

如果你需要更多控制，你可以手动处理分片状态字典。以下是方法：

```python
def save_checkpoint_manual(model, optimizer, epoch, checkpoint_dir):
    """Manually save sharded checkpoint."""
    rank = torch.distributed.get_rank()
    # Get sharded state dict
    model_sd = model.state_dict()  # Already sharded
    # Get optimizer state dict (also sharded)
    optim_sd = optimizer.state_dict()
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    os.makedirs(checkpoint_path, exist_ok=True)
    # Each rank saves its shard
    model_path = os.path.join(checkpoint_path, f"model_rank_{rank}.pt")
    optim_path = os.path.join(checkpoint_path, f"optim_rank_{rank}.pt")
    torch.save(model_sd, model_path)
    torch.save(optim_sd, optim_path)
    # Save metadata on rank 0
    if rank == 0:
        metadata = {"epoch": epoch, "world_size": torch.distributed.get_world_size()}
        torch.save(metadata, os.path.join(checkpoint_path, "metadata.pt"))
        print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint_manual(model, optimizer, checkpoint_dir, epoch):
    """Manually load sharded checkpoint."""
    rank = torch.distributed.get_rank()
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    # Each rank loads its shard
    model_path = os.path.join(checkpoint_path, f"model_rank_{rank}.pt")
    optim_path = os.path.join(checkpoint_path, f"optim_rank_{rank}.pt")
    model_sd = torch.load(model_path, map_location="cpu")
    optim_sd = torch.load(optim_path, map_location="cpu")
    model.load_state_dict(model_sd)
    optimizer.load_state_dict(optim_sd)
    if rank == 0:
        metadata = torch.load(os.path.join(checkpoint_path, "metadata.pt"))
        print(f"Checkpoint loaded from {checkpoint_path}, epoch {metadata['epoch']}")
    return epoch
```

手动方法给你更多控制，但将加载绑定到元数据中保存的 `world_size`——当 GPU 数量可能在保存和恢复之间改变时，使用 DCP。对大多数训练作业，DCP 仍然是默认。

### 用于评估的完整状态字典

有时你需要一个完整（非分片）的状态字典，例如保存最终模型用于推理或与他人共享。你可以将所有分片收集到 rank 0：

```python
def save_full_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save full (unsharded) checkpoint on rank 0."""
    rank = torch.distributed.get_rank()
    # Get full state dict (gathers all shards to rank 0)
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,  # Gather all shards
            cpu_offload=True,
        ),
    )
    optim_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        ),
    )
    # Only rank 0 saves
    if rank == 0:
        checkpoint = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
            "epoch": epoch,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Full checkpoint saved to {checkpoint_path}")
    torch.distributed.barrier()
```

这将所有分片收集到 rank 0。即使用 `cpu_offload=True`，rank 0 在保存时也必须在主机内存中持有完整状态字典——一个 FP32 的 70B 模型约 280 GB，可能在典型节点上 OOM。对于训练重启，坚持用本节前面的 DCP 分片格式：DCP 可以在 world size 改变时于加载时重新分片，而无需在内存中构建完整模型。仅在 rank 0 的 RAM 能容纳模型或你确实需要一个非分片文件时使用完整状态收集。

## 预取：优化通信

FSDP 在每一步都增加通信——前向中的 all-gather，反向中的 reduce-scatter。快速互连有帮助，但通常的目标是将该延迟隐藏在单独 CUDA 流上的计算之后。**FSDP2 已经隐式地这样做**：每层的前向前钩子在当前层运行时发出下一层的 all-gather。将显式预取 API 视为一个可选的调优旋钮，而非先决条件——先剖析，只在通信仍然阻塞计算时才添加 `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch`。

![预取时间线：无 vs 有。](img/fsdp_prefetch_timeline_zh.png){#fig:fsdp-prefetch-timeline .block width=100% align=center}

图~\ref{fig:fsdp-prefetch-timeline} 展示了理想情况：无重叠（上），每层在计算前等待它的 all-gather（AG）；有重叠（下），当层 L₀ 计算时，L₁ 的 all-gather 并行运行。FSDP2 的隐式预取默认针对这个底部模式；显式预取推进更多——提前预取两层或更多可以重叠更多通信，但 **提高峰值内存**，因为预取层的非分片参数与当前层的完整参数一起占用 GPU 内存。在紧张的内存预算下，激进的预取即使在非预取运行能装下时也可能触发 OOM。

### 前向预取

显式前向预取告诉一个层比默认的隐式调度更早地为即将到来的层 all-gather 参数：

```python
def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """Explicit forward prefetch on top of FSDP2's default implicit prefetch."""
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)
```

单元素列表（仅下一层）大致匹配默认重叠，但从 CPU 更早发出 all-gather；长度为二或更多的列表预取更远并预留更多内存。

### 反向预取

显式反向预取覆盖默认的逆序预取，以在梯度计算期间进行更精细的控制：

```python
def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """Explicit backward prefetch on top of FSDP2's default implicit prefetch."""
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)
```

### 何时使用显式预取

当剖析显示尽管有隐式预取，all-gather 或 reduce-scatter 仍在阻塞计算时——常常是受 CPU 限制的启动开销或慢速互连上非常快的层——显式预取有帮助。试试 `num_to_forward_prefetch=2` 和 `num_to_backward_prefetch=2`，然后调整。当内存已经紧张时跳过它；额外预留的非分片参数可能比节省的通信时间成本更高。

## 激活重计算与卸载

激活重计算几乎总是与 FSDP 一起使用。不是在前向期间存储所有激活值，而是在反向期间重新计算它们。这可以将激活内存削减 50-80%，当你已经内存受限时这很关键。

### 为什么激活重计算对 FSDP 重要

用 FSDP，你已经在分片参数、梯度和优化器状态。激活值仍然可能是内存瓶颈，特别是大批大小或长序列。激活重计算以计算换内存：你在反向期间重新计算激活值而非存储它们。

激活内存随模型架构和批大小缩放。transformer 的粗略估算是：

$$\text{激活内存} \approx L \times B \times S \times H \times \text{每元素字节数} \times k$$

其中 $L$ 是层数，$B$ 是批大小，$S$ 是序列长度，$H$ 是隐藏维度，$k$ 是一个因子（通常 10–20），说明注意力和 MLP 块中的中间张量。这不是一个精确公式——实际内存取决于实现细节，如注意力分数是否物化、MLP 扩展比和框架开销。但它给你正确的数量级。对于一个 7B 模型（$L=32$，$H=4096$），序列长度 2048，批大小 8，fp16：

$$32 \times 8 \times 2048 \times 4096 \times 2 \times 12 \approx 52\text{ GB}$$

用激活重计算，你只存储每个检查点块的输入，而非所有中间张量，将内存减少 50–80%。权衡是在反向期间重新计算激活值（前向大约慢 30%，但反向类似，因为你无论如何都要计算梯度）。

### 使用激活重计算

在 `fully_shard` **之前** 应用激活重计算——FSDP 钩子包装模块树，检查点包装器必须先就位。优先用 `apply_activation_checkpointing`（在 TorchTitan 和 PyTorch 分布式示例中使用），而非在已分片的模型上临时调用 `checkpoint()`：

```python
import functools
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

def check_fn(submodule):
    return isinstance(submodule, TransformerBlock)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=check_fn,
)

fully_shard(model, mesh=mesh)  # after activation checkpointing
```

`CheckpointImpl.NO_REENTRANT` 匹配 `torch.utils.checkpoint.checkpoint` 中的 `use_reentrant=False`。默认的可重入模式已弃用，且可能与 `torch.compile` 或 FSDP 的 autograd 钩子静默失败——在生产 FSDP 作业中始终使用非可重入检查点。

对于块内的手动控制，显式传入 `use_reentrant=False`：

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlockWithCheckpoint(nn.Module):
    def __init__(self, args: ModelArgs, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p
        )

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(
                self._forward_impl, x, use_reentrant=False,
            )
        return self._forward_impl(x)

    def _forward_impl(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

对于手动的每块检查点而非 `apply_activation_checkpointing`，在每个块内切换检查点：

```python
# Checkpoint every other layer to balance memory and speed
for i, layer in enumerate(model.layers):
    layer.use_checkpoint = (i % 2 == 0)
```

### CPU 卸载

CPU 卸载将参数、梯度和优化器状态移到 CPU 内存，以更慢的训练为代价释放 GPU 内存。FSDP2 API 支持这个：

```python
from torch.distributed.fsdp import CPUOffloadPolicy

fully_shard(
    model,
    mesh=mesh,
    offload_policy=CPUOffloadPolicy(pin_memory=True),
)
```

分片参数在每次 all-gather 之前被复制到 GPU；梯度和优化器步骤在 CPU 上运行。这增加显著的开销，但对非常大的模型可能是必要的。实际减速取决于 PCIe 代数（3.0 vs 4.0 vs 5.0）、CPU 内存带宽、NUMA 拓扑和优化器状态大小——经验上 20-50% 常见，但你的情况会有所不同。如果 CPU RAM 也耗尽，NVMe 卸载是 DeepSpeed ZeRO-Infinity 的特性（第~\ref{chap:beyond-state-sharding-with-deepspeed-and-megatron}章），而非 FSDP2 原生提供的。

### 何时使用卸载

CPU 卸载应在你的优化序列中较晚出现。如果你已经启用完全分片和激活重计算、尽可能减小批大小和序列长度，而你仍然遇到 OOM——那么卸载才有意义。对大多数模型，完全分片加激活重计算就足够，无需触及卸载。

## 性能优化

一旦你让 FSDP 工作，你会想优化性能。主要瓶颈是通信（all-gather/reduce-scatter）和激活内存。让我们看看如何剖析和优化。

### 剖析 FSDP 训练

使用 PyTorch 的性能分析器理解时间花在哪里。一个完整的可运行示例在 `code/fsdp2_profile.py`：

```bash
torchrun --nproc_per_node=2 code/fsdp2_profile.py
```

关键模式是用 `profile()` 包装你的训练循环，并用 `record_function()` 标记不同阶段：

```python
from torch.profiler import profile, record_function, ProfilerActivity
rank = dist.get_rank()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for i in range(num_iterations):
        with record_function("forward"):
            output = model(data)
            loss = criterion(output, target)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
if rank == 0:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    prof.export_chrome_trace(f"fsdp_trace_rank{rank}.json")
```

只有 rank 0 打印并导出 trace。由于 FSDP 在所有 rank 上同步运行相同的操作，时序模式几乎相同——检查一个 rank 的 trace 通常就够了。如果你怀疑负载不均衡或落后者，你可以移除 `if rank == 0` 保护，从所有 rank 导出 trace 并比较它们。

trace 文件 `fsdp_trace_rank0.json` 是 Chrome trace 格式。关于如何打开和解释这些 trace，见第~\ref{chap:distributed-training-with-pytorch-ddp}章的第~\ref{sec:ddp-profiling}节。简短版本：在 Chrome 中打开 `chrome://tracing`，点击 "Load"，并选择 `.json` 文件。在时间线中，查找 all-gather 和 reduce-scatter 操作——理想情况下它们与计算重叠。如果它们尽管有 FSDP2 的隐式预取仍然阻塞，试试显式预取 API（带预取一节指出的内存权衡）。还要检查峰值内存使用（`profile_memory=True` 启用这个）以查看激活值是否吃掉了比预期更多的内存。

### 优化通信

如果性能分析器将通信显示为瓶颈，你有几个选项。预取（前面介绍）可以将通信与计算重叠。如果你有内存余量，设置 `reshard_after_forward=False` 避免反向中的 all-gather——参数在前向后保持非分片，所以反向不需要再次获取它们：

```python
fully_shard(model, mesh=mesh, reshard_after_forward=False)
```

这以内存换速度。仅在剖析后你有余量时使用它。

硬件也重要。节点内的 NVLink 和节点间的 InfiniBand 有很大区别。检查 NCCL 实际在使用它们：

```bash
export NCCL_IB_DISABLE=0 && export NCCL_DEBUG=INFO
```

调试输出会显示 NCCL 检测到哪些互连。

### 优化激活内存

如果激活内存是瓶颈，最简单的修复是减小批大小或序列长度——更小的输入意味着更少的激活值要存储。但如果你需要大的有效批大小以收敛，梯度累积让你在没有内存成本的情况下达到那里：

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (data, target) in enumerate(dataloader):
    loss = criterion(model(data), target) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

这给你 `batch_size * accumulation_steps` 的有效批大小，但只有单个 `batch_size` 的内存占用。你也可以尝试选择性检查点——只检查点某些层而非全部，并实验以找到内存和重计算开销之间的正确平衡。

### 内存剖析

要理解内存去哪了，在你的代码中散布一些打印语句：

```python
def print_memory_usage(step_name):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{step_name}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
print_memory_usage("Before model")
print_memory_usage("After FSDP")
print_memory_usage("After forward")
print_memory_usage("After backward")
```

这告诉你每个阶段使用了多少内存。如果 "After forward" 比 "After FSDP" 高得多，激活值是罪魁祸首；如果 "After backward" 保持高，梯度或优化器状态可能是。`nvidia-smi` 和 `torch.cuda.memory_summary()` 增加更多细节。

时间点打印常常错过 FSDP 短暂的 all-gather 尖峰。要看完整时间线，记录分配器历史并转储快照：

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
# ... run one or more training steps ...
torch.cuda.memory._dump_snapshot("fsdp_mem_snapshot.pickle")
```

将 pickle 上传到 [pytorch.org/memory_viz](https://pytorch.org/memory_viz) 以检查临时非分片参数何时出现以及哪些操作预留了峰值。

## 多节点 FSDP 训练

多节点 FSDP 的工作方式与多节点 DDP 相同——你需要进程组初始化和适当的网络。主要区别是检查点：用 FSDP2，分片状态字典很直接——每个 rank 写它的分片，你可以在不 all-gather 的情况下加载它们。

### 设置多节点 FSDP

设置类似于多节点 DDP。在每个节点上，你需要：

1. 为进程组初始化设置环境变量
2. 用 `torchrun` 启动训练
3. 确保节点之间的网络连接

在主节点（节点 0）上：

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=<master_ip> --master_port=29500 code/train_fsdp2.py
```

在工作节点（节点 1）上：

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 --master_addr=<master_ip> --master_port=29500 code/train_fsdp2.py
```

用主节点的实际 IP 地址替换 `<master_ip>`。你可以用以下命令找到它：

```bash
hostname -I
```

### 多节点的网络配置

对于多节点 FSDP，网络带宽和延迟至关重要。FSDP 比 DDP 做更多通信（all-gather 和 reduce-scatter），所以快速互连更重要。

**InfiniBand 比以太网更受青睐**，因为：

- 更高带宽：每链路 200-400 Gb/s，而以太网 10-100 Gb/s
- 更低延迟：亚微秒级 vs 微秒级
- RDMA 支持：GPU 到 GPU 直接内存访问

确保 NCCL 在使用 InfiniBand：

```bash
export NCCL_IB_DISABLE=0 && export NCCL_DEBUG=INFO
```

检查 NCCL 日志以验证它在使用 InfiniBand。你应该看到如下消息：

```
NCCL INFO NET/IB: Using [device] for node [rank]
```

### 多节点上的检查点

用 FSDP2，即使在多节点上检查点也很直接。每个 rank 保存它的分片，所以你需要从所有节点可访问的共享存储。

**选项 1：共享文件系统（NFS、Lustre 等）**

如果所有节点挂载相同的文件系统，每个 rank 可以直接写：

```python
checkpoint_dir = "/shared/checkpoints"  # Mounted on all nodes
save_checkpoint_dcp(model, optimizer, epoch, checkpoint_dir)
```

**选项 2：并行写入本地存储**

每个节点在本地写，然后你稍后同步：

```python
# Each node writes to local storage
local_checkpoint_dir = f"/local/checkpoints/node_{node_rank}"
save_checkpoint_dcp(model, optimizer, epoch, local_checkpoint_dir)
```

然后在训练后同步到共享存储（或使用分布式文件系统）。

**选项 3：对象存储（S3 等）**

使用像 `s3fs` 这样的库直接写到 S3：

```python
import s3fs

fs = s3fs.S3FileSystem()
checkpoint_path = f"s3://bucket/checkpoints/epoch_{epoch}"
# Save using DCP with S3 backend
```

分片方法有帮助，因为每个 rank 只写它的分片（更小的文件，更少的带宽）。

### SLURM 集成

大多数 HPC 集群使用 SLURM 进行作业调度。下面是多节点 FSDP 的最小示例：

```bash
#!/bin/bash
#SBATCH --job-name=fsdp_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
# Get node list
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_fsdp2.py
```

或者直接用 `torchrun` 配合 SLURM：

```bash
#!/bin/bash
#SBATCH --job-name=fsdp_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_fsdp2.py
```

我们在第~\ref{chap:running-distributed-training-with-slurm}章详细介绍 SLURM。

### 扩展考量

随着你扩展到更多节点，通信开销增长。确保你的模型足够大，使计算仍然占主导——否则你为大部分时间花在等待网络上的 GPU 付费。

检查点也变得更棘手。用许多节点，你最终会有许多分片文件。一个能很好处理小文件的分布式文件系统或对象存储（如 Lustre 或 S3）在这里有帮助。更多硬件带来更多故障——频繁保存检查点，如果你的集群支持则考虑弹性训练。

网络拓扑也重要。在同一网络段上带 InfiniBand 的节点会胜过分散在通过较慢链路连接的机架上的节点。

## 调试 FSDP 问题

FSDP 增加了复杂性，当事情出错时，错误消息并不总是有帮助的。以下是如何处理常见问题。

### 内存不足（OOM）

首次设置 FSDP 时 OOM 错误很常见。一个快速的理智检查是参数是否实际被分片。在 FSDP2 下它们是 DTensor——`param.shape` 是全局形状，所以用本地张量来查看你的分片：

```python
for name, param in model.named_parameters():
    local = param.to_local() if hasattr(param, "to_local") else param
    print(f"{name}: local_shape={local.shape}, device={param.device}")
```

如果参数看起来正确但你仍然 OOM，激活值很可能是罪魁祸首。使用前面的内存剖析方法确认，然后减小批大小、序列长度，或启用激活重计算。还要留意内存泄漏——张量在迭代之间累积，因为你忘记 detach 或删除它们。

### 挂起或死锁

当进程失去同步时，FSDP 会挂起。最常见的原因是只有部分 rank 执行的条件逻辑：

```python
# BAD: only rank 0 runs this collective-triggering code
if rank == 0:
    model.some_operation()

# GOOD: all ranks execute the same code
model.some_operation()
```

FSDP 还有一种额外的失败模式：**条件前向路径**，即不同 rank 使用不同的参数子集（例如 `if rank == 0: out = model.head_a(x) else: out = model.head_b(x)`，或跳过某些层的 `if/else` 分支）。FSDP 期望每个 rank 每一步都以相同的顺序对相同的模块做 all-gather；在部分 rank 上跳过某个子模块会打破这个调度，job 会挂起，等待永远不会到达的集合通信操作。应重构代码使所有 rank 运行相同的模块图，或者把可选分支隔离在 FSDP 包装单元之外。

其他原因：数据不均衡（某个 rank 比其他 rank 先耗尽批次）、部分 rank 上的检查点加载失败，或 NCCL 问题。对于 NCCL 问题：

```bash
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

`TORCH_NCCL_ASYNC_ERROR_HANDLING` 和 `TORCH_NCCL_BLOCKING_WAIT` 会暴露 NCCL 失败而不是静默挂起；`TORCH_DISTRIBUTED_DEBUG=DETAIL` 会在 rank 出现分歧时记录集合通信操作不匹配的情况。

### 训练缓慢

如果训练比预期慢，先做性能剖析——猜测只会浪费时间。检查通信是否与计算重叠；如果没有，在确认隐式预取已经生效之后，再尝试显式预取调优。对于多节点场景，验证是否用上了 InfiniBand（`NCCL_DEBUG=INFO` 会显示这一点）。激活重计算会带来约 30% 的额外开销，所以要确认显存节省值得这个代价。另外确认硬件支持的情况下混合精度确实已经启用。

### 结果不正确

结果不正确通常来自几个常见原因之一。权重初始化应该在**每个 rank 上使用相同的种子**，让分片起始状态一致；而 dropout 和数据顺序**不应**共享这个种子，否则每个 rank 会应用相同的掩码，看到相关联的噪声：

```python
base_seed = 42
torch.manual_seed(base_seed)          # same init weights on every rank
torch.cuda.manual_seed_all(base_seed + rank)  # distinct dropout noise per rank
```

其次，验证 `DistributedSampler` 是否配置正确，并且你在每个 epoch 都调用了 `sampler.set_epoch(epoch)`——忘记这一步意味着所有 epoch 看到的都是同一种打乱顺序。第三，通过打印各 rank 的梯度范数来检查梯度是否被正确同步。如果排查无果，在单块 GPU 上跑同一个模型建立基线，然后再扩大规模做对比。

### 调试工具

有几个工具有助于 FSDP 调试。对于 NCCL 问题，`export NCCL_DEBUG=INFO`（需要更多细节可用 `NCCL_DEBUG_SUBSYS=ALL`）会显示通信层正在发生的事情。前面介绍过的 PyTorch profiler 能揭示通信模式和瓶颈。对于显存问题，`torch.cuda.memory_summary()` 给出详细的显存分解。而对于分布式相关的问题，`torch.distributed.set_debug_level(torch.distributed.DebugLevel.DETAIL)` 可以启用分布式运行时的详细日志。

## FSDP2 与 ZeRO、DDP 的比较

在 DDP、ZeRO（DeepSpeed）和 FSDP2 之间做选择，取决于你的模型规模和生态偏好。

**DDP** 是最简单高效的选项。每块 GPU 都持有完整的模型、梯度和优化器状态副本。通信只需梯度同步（AllReduce），底层已被深度优化。当你的模型能够完全放入单块 GPU 显存时，应优先使用 DDP——在使用混合精度的现代 GPU 上，这足以覆盖绝大多数 7B 级别以内的模型。

**ZeRO**（来自 DeepSpeed）执行分阶段分片：ZeRO-1 分片优化器状态，ZeRO-2 在此基础上加上梯度，ZeRO-3 分片包括参数在内的一切。它经过生产环境验证，并与 DeepSpeed 的其他特性集成，如 ZeRO-Offload（CPU）和 ZeRO-Infinity（NVMe）。代价是要引入 DeepSpeed 这个依赖并学习它的 API。

**FSDP2** 是 PyTorch 原生的，像 ZeRO-3 一样执行完全分片。其每参数设计更简单（约 3000 行，相比 FSDP1 的约 14000 行），能很好地与 `torch.compile` 集成，且不需要外部依赖。它比 ZeRO 更新，因此经过实战检验的示例较少，但这是 PyTorch 正在前进的方向。

### 内存与性能

用具体数字说明：一个 70 亿参数的模型配 Adam 优化器，用 DDP 时每块 GPU 需要约 84 GB（参数 + 梯度 + 优化器状态，全部复制）。在 8 块 GPU 上用 FSDP2 或 ZeRO-3，这个数字降到每块 GPU 约 10.5 GB——所有内容都按 8 份分片。

在性能方面，当模型能够完全装入显存时 DDP 最快（通信更少）。对于单卡装不下的模型，FSDP2 和 ZeRO-3 表现相近——两者都执行 all-gather 和 reduce-scatter，差异更多来自实现细节和网络拓扑，而非根本性的设计不同。应根据生态契合度而非性能来选择。

### 该用哪个

如果你的模型能够完全放入单块 GPU 显存，优先使用 DDP——它架构更简单、执行更高效。如果装不下，先尝试优化（混合精度、激活重计算、梯度累积）。如果依然 OOM，再切换到 FSDP2；由于 API 相似，迁移很直接。如果你需要 DeepSpeed 特有的功能，如 ZeRO-Offload 或 ZeRO-Infinity，或者团队技术栈已经身处 DeepSpeed 生态，那就改用 ZeRO。

## 实践技巧

一些来自真实 FSDP 使用经验的教训值得强调。

用 FSDP2 保存检查点时，每个 rank 写自己的分片——前面介绍过的 DCP 流程是默认方式。将数据全部收集到 rank 0 是例外情况（见"用于评估的完整状态字典"一节）。

共享参数需要一些额外注意。如果同一个张量在模型中的多个地方出现（例如绑定的嵌入），这些用法必须位于同一个 FSDP 组内。FSDP 的参数交换不会跨组保留共享性，所以要么组织模型结构使共享参数保持在同一个模块层级中，要么干脆避免共享。

内存剖析常常会揭示一些意外情况。瓶颈并不总是出现在你预期的地方。用 `torch.profiler` 或 `nvidia-smi` 去排查。常见的元凶包括激活值（用重计算来应对）、在迭代之间累积的临时张量（记得 detach 或显式删除它们），以及在内存受限的系统上设置了 `pin_memory=True` 的 DataLoader。

最后，记住 `reshard_after_forward` 默认是 `True`，这通过在前向之后重新分片来节省内存，但代价是反向传播中需要额外一次 all-gather。如果你有内存余量而通信才是瓶颈，可以尝试将它设为 `False`，让参数在各次传播之间保持不分片。

### 初始化最佳实践 {#sec:fsdp-initialization-best-practices}

对于非常大的模型，先在 meta 设备上创建，应用 FSDP，然后再移动到实际设备并初始化：

```python
with torch.device("meta"):
    model = Transformer(args)
fully_shard(model, mesh=mesh)
model.to_empty(device=device)
model.reset_parameters()
```

这样可以避免在单个设备上完整实例化整个模型。所有 rank 应该为 `reset_parameters()` 共享同一个种子；而 dropout（如"调试"一节所述）和 `DistributedSampler` 的数据顺序应使用各 rank 独立的种子。

### 数据加载与梯度裁剪

使用 `DistributedSampler`，并在每个 epoch 调用 `sampler.set_epoch(epoch)`——忘记这一步意味着所有 epoch 看到的都是同一种打乱顺序。梯度裁剪与 FSDP 兼容；照常调用 `torch.nn.utils.clip_grad_norm_` 即可，FSDP 会自动处理取消分片/重新分片。

### 混合精度

对参数而言，BF16 通常优于 FP16（动态范围更宽，溢出风险更低）。为了数值稳定性，梯度归约要保持 FP32（`reduce_dtype=torch.float32`）。先让 FSDP 在不使用混合精度的情况下正常工作，再加上混合精度。

### 渐进式优化

从简单开始：带混合精度的 FSDP2。如果遇到 OOM，加上激活重计算。仍然 OOM？减小批大小或序列长度。CPU 卸载是最后的手段——它确实有效，但速度下降相当明显。不要过早优化，先让它跑起来。

## 进阶主题

### 混合分片（HSDP）{#sec:hsdp}

在非常大的规模下，你可能希望在节点内分片、跨节点复制——这减少了节点间通信，而节点间通信通常比节点内通信（NVLink 相对 InfiniBand）更慢。使用二维网格：

```python
mesh = init_device_mesh("cuda", (4, 8))  # 4 nodes × 8 GPUs per node
fully_shard(model, mesh=mesh)  # 2D mesh: replicate dim 0, shard dim 1 (HSDP)
```

这会在每个节点内把参数分片到 8 块 GPU 上，但跨 4 个节点复制。代价是：更高的内存占用（4 倍复制），但更少的跨节点流量。

### 编译器集成

当你先应用 FSDP 再编译时，FSDP2 可以与 `torch.compile` 集成：

```python
fully_shard(model, mesh=mesh)
model = torch.compile(model)
```

每参数设计在这里很有帮助——编译器看到的是独立的参数而不是一个被展平的缓冲区——但兼容性仍在演进中。某些组合（激活重计算、CPU 卸载、自定义通信钩子）会触发图断裂或静默的 eager 回退。先在小模型上验证，检查是否有意外的图断裂（`torch._dynamo.explain(model)`），再扩大规模。

### 其他集成

FSDP2 天然支持梯度累积和学习率调度——无需特殊处理。对于混合精度，使用 `MixedPrecisionPolicy` 而不是标准的 AMP 上下文管理器。自定义通信钩子可用于精细控制，但大多数用户不会需要它们。

## 面向 TPU/XLA 的 SPMD FSDP {#sec:fsdp-spmd}

本章聚焦于使用 CUDA 设备在 GPU 上训练的 FSDP2。不过，PyTorch 也为 TPU/XLA 设备提供了基于 SPMD 的 FSDP，它使用一种基于 GSPMD（广义单程序多数据）的不同方法来实现自动并行化。

**与 GPU FSDP2 的关键区别：**

- **使用 SPMD 模式**：XLA 编译器会根据分片标注自动划分计算，而不是显式的 all-gather/reduce-scatter 操作。
- **基于网格的分片**：使用 PyTorch/XLA 的 `Mesh` 抽象和带命名的维度（例如 `('fsdp', 'model')`）。
- **编译器驱动**：XLA 编译器负责通信优化，类似于 JAX 的 `pmap` 的工作方式。

完整示例在 `code/fsdp_spmd_tpu.py` 中。注意这需要 TPU 硬件——它不能在 GPU 上运行：

```bash
# On a TPU VM:
python code/fsdp_spmd_tpu.py
```

核心模式：

```python
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2
)

xr.use_spmd()  # Enable SPMD mode

# Create mesh with 'fsdp' axis
num_devices = xr.global_runtime_device_count()
mesh = xs.Mesh(np.array(range(num_devices)), (num_devices, 1), ('fsdp', 'model'))

# Shard inputs and wrap model
x = xs.mark_sharding(x, mesh, ('fsdp', None))
model = FSDPv2(model, mesh)
```

当你在 TPU 设备上训练，或者想要编译器优化的通信模式时，使用 SPMD FSDP。XLA 编译器会自动处理通信优化，类似于 JAX 的 `pmap`。对于 GPU 训练，使用本章介绍的 `fully_shard()` API——它让你能显式控制通信模式，且不需要 XLA。

更多细节请参见 PyTorch/XLA SPMD 文档。[^xla-spmd]

[^xla-spmd]: <https://docs.pytorch.org/xla/master/spmd.html>

## 小结

FSDP2 是 PyTorch 对"训练装不进单块 GPU 的模型"这一问题给出的答案。通过跨 GPU 分片参数、梯度和优化器状态，它让你能训练比单块 GPU 容量大 8 倍、16 倍甚至更多的模型。

值得退一步理解一下 FSDP 改变了什么——以及没有改变什么。FSDP 把内存上限从单 GPU 约束转变为集群级约束。一个需要 160 GB 内存（参数 + 梯度 + 优化器状态）的模型，可以在 8 块各 24 GB 的 GPU 上运行，因为每块 GPU 只持有总量的 1/8。这是一个根本性的转变：你不再受限于能买到的最大单块 GPU，而是受限于你能连接多少块 GPU。

然而，FSDP 本质上仍然是**数据并行**。每块 GPU 处理不同的数据批次，模型计算本身并没有跨设备拆分——每块 GPU 执行相同的操作，只是作用在需要时才被 all-gather 的不同分片上。这将 FSDP 与模型并行（张量并行、流水线并行）区分开来，在模型并行中，不同的 GPU 同时计算模型的不同部分。FSDP 扩展的是内存，而不是单个样本的计算量。要扩展计算量，你仍然依赖在更多 GPU 上使用更大的批大小，就像 DDP 一样。

在选择并行策略时，这个架构上的区别很重要。仅靠 FSDP 就能走得相当远——在大型集群上可以支持多达数千亿参数的模型。但对于最大的模型（万亿参数以上），或者当你需要降低单样本延迟时，你会把 FSDP 与张量并行或流水线并行结合起来使用。我们会在后面的章节中介绍这些组合方式。

实践建议很简单：如果 DDP 能用，就用 DDP——它更快也更简单。当你的模型超出单块 GPU 的容量时，先尝试优化（混合精度、激活重计算、梯度累积）。如果仍然 OOM，切换到 FSDP2。从完全分片和 DCP API 做检查点开始，用性能剖析找瓶颈，并在扩展到多节点之前先在 2-4 块 GPU 上测试。不要盲目优化——让性能剖析结果来指引你。

本章的代码示例是完整且可运行的。在你自己的硬件上试一试，看看分片实际是如何工作的：每个 rank 只持有模型的一部分，all-gather/reduce-scatter 操作会自动发生。

FSDP2 能很好地处理大多数大模型训练场景。但如果连完全分片都不够呢？如果你需要 CPU 或 NVMe 卸载来进一步突破内存限制，或者需要为跨多节点训练优化通信模式呢？这就是 DeepSpeed 的 ZeRO 发挥作用的地方。在下一章，我们将探索 ZeRO-Offload、ZeRO-Infinity 和 ZeRO++——这些特性扩展了 FSDP2 目前提供的能力——以及何时应该选择 DeepSpeed 而不是 PyTorch 原生方案。

## 有用的链接

__PyTorch FSDP 文档__

- PyTorch FSDP 文档：\url{https://pytorch.org/docs/stable/fsdp.html}
- PyTorch FSDP 教程：\url{https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html}
- 每参数分片 FSDP RFC：\url{https://github.com/pytorch/pytorch/issues/114299}
- TorchTitan FSDP 指南：\url{https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md}
- PyTorch XLA SPMD：\url{https://docs.pytorch.org/xla/master/spmd.html}

__教程与指南__

- UvA Deep Learning - Data Parallel FSDP（JAX）：\url{https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/data_parallel_fsdp.html}
- Hugging Face - FSDP and DeepSpeed：\url{https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed}
- Hugging Face - FSDP1 vs FSDP2：\url{https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2}
- Introduction to Parallelism：\url{https://ggrigorev.me/posts/introduction-to-parallelism/}

__研究__

- PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel (2023)：\url{https://arxiv.org/abs/2304.11277}
- Distributed Training Optimization (2024)：\url{https://arxiv.org/abs/2411.00284}

__项目__

- Wan2.2（FSDP + DeepSpeed Ulysses 用于多 GPU 推理）：\url{https://github.com/Wan-Video/Wan2.2}

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
