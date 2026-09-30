# 第5章：超越状态分片——DeepSpeed 与 Megatron {-}

*为超大型模型扩展内存容量并分片计算*

> 未来已经来临，只是分布不均。
- 威廉·吉布森（William Gibson），作家

**Code Summary**

- `deepspeed.initialize()`：用 ZeRO 配置初始化 DeepSpeed 引擎
- `deepspeed.DeepSpeedEngine`：用于模型训练的 DeepSpeed 引擎包装器
- `megatron.core.parallel_state`：Megatron 并行状态管理
- `megatron.core.tensor_parallel`：Megatron 张量并行工具
- `megatron.core.pipeline_parallel`：Megatron 流水线并行工具
- `deepspeed.zero.Init()`：DeepSpeed ZeRO 初始化上下文管理器
- `deepspeed.zero.OffloadOptimizerConfig`：ZeRO-Offload 的配置
- `deepspeed.zero.OffloadParamConfig`：ZeRO-Infinity 参数卸载的配置
- `megatron.model.parallel.layers.ColumnParallelLinear`：用于张量并行的列并行线性层
- `megatron.model.parallel.layers.RowParallelLinear`：用于张量并行的行并行线性层

## 超越状态分片

在上一章，我们探索了 FSDP——PyTorch 跨 GPU 分片参数、梯度和优化器状态的方法。FSDP2 的完全分片在功能上等价于 DeepSpeed 的 ZeRO 阶段 3[^zero-paper]：两者都通过确保每块 GPU 只持有训练状态的 1/N 来消除显存冗余。

[^zero-paper]: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020). https://arxiv.org/abs/1910.02054

状态分片从根本上解决了单卡显存容纳不下的难题，但它并未改变模型前向与反向计算的执行范式。每块 GPU 依然在完整的网络层架构上运转——只是每次喂入不同的数据微批次。然而面对千亿级（100B+）乃至更大的超前沿模型，单纯的状态分片便会捉襟见肘：要么单个线性层的矩阵规模过大而无法在单卡上高效完成张量运算，要么网络层数过深导致中间激活值即使使用梯度检查点也难以完全存放在有限的显存中。

这也正是 **Megatron** 登上历史舞台的关键原因[^megatron-paper]。Megatron 的张量并行将大矩阵运算拆分到 GPU 上，其流水线并行沿深度维度分片模型。这些技术分片 *计算本身*，而非只是训练状态。它们对训练前沿模型至关重要，并且仍然是 NVIDIA、Meta 等公司大规模训练基础设施的支柱。

[^megatron-paper]: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019). https://arxiv.org/abs/1909.08053

我们先介绍 DeepSpeed ZeRO——理解完整的 ZeRO 家族（阶段 1-3、卸载、ZeRO++）是值得的，因为许多代码库仍在使用它。但本章真正的焦点是 Megatron 风格的并行：张量并行、流水线并行，以及它们如何与状态分片结合成多维并行。

![ZeRO 各阶段比较：DDP vs ZeRO-1/2/3。](img/zero_stages_comparison_zh.png){#fig:zero-stages .block width=100% align=center}

图~\ref{fig:zero-stages} 展示了 DDP 和每个 ZeRO 阶段跨两个 rank（R0–R1）的内存布局。每行代表一块 GPU，三个彩色块显示那块 GPU 存储什么：P（参数，蓝色）、G（梯度，红色）和 O（优化器状态，绿色）。在 DDP 中，所有块都是全宽的，因为每块 GPU 持有所有东西的完整副本——这是我们想消除的内存冗余。ZeRO-1 保持参数和梯度复制，但分片优化器状态（注意更小的 O 块）。ZeRO-2 额外分片梯度，所以 G 和 O 块都缩小。ZeRO-3 分片全部三个组件——用 2 块 GPU，每个块变成原始大小的一半。从左到右的视觉递进显示了每 GPU 内存如何在每个阶段减少，代价是在需要时增加通信以重构完整张量。

## ZeRO 阶段 1：优化器状态分区

回想上一章的内存分解：对于使用 Adam 的模型，优化器状态主导内存使用——每个参数需要将动量和方差存储为两个 FP32 副本，每参数总共 8 字节。一个 7B 参数模型仅优化器状态就需要 56GB。混合精度训练还保留一个 FP32 的权重主副本（每参数另 4 字节），所以完整的优化器侧占用更接近 12 字节——我们在本章后面的 70B 示例中使用那个计算。在 DDP 中，每块 GPU 持有这些状态的完整副本，这是巨大的浪费。

ZeRO-1 的洞见很直接：既然每块 GPU 最终只更新它分配到的那部分参数，为什么要存储完整的优化器状态？用 2 块 GPU，每块只存储优化器状态的一半。前向和反向传播正常进行，梯度同步后，每块 GPU 只用它的本地优化器状态来更新对应的参数分片。56GB 的优化器状态被分布到 2 块 GPU 上，将每块 GPU 的负担减少到 28GB。

与需要极少设置的 PyTorch 原生 DDP 不同，DeepSpeed 使用一个配置字典（或 JSON 文件）来控制所有训练设置——优化器、精度、ZeRO 阶段等。我们的示例为可移植性启用 FP16；在 Hopper 及更新的 GPU 上，BF16 常常是更好的默认值（更宽的动态范围，无需损失缩放）。将配置传给 `deepspeed.initialize()`，它返回一个自动处理分布式训练的包装模型引擎：

```python
import deepspeed

ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 1}  # Enable ZeRO-1
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Training loop uses model_engine instead of model
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

与 DDP 的关键区别：DeepSpeed 从你的配置构建优化器，而非从你可能早先创建的单独的 `torch.optim.Adam(...)`——那个外部优化器被忽略。`model_engine` 包装你的模型并提供 `backward()` 和 `step()` 方法。

ZeRO-1 适合模型本身适合 GPU 内存、但加上优化器状态就超过限制的场景。它需要对训练循环极少的改动，且最容易调试，使它成为从 DDP 迁移到 ZeRO 时的自然第一步。

要体验 DeepSpeed API，运行这个最小示例：

```bash
pip install deepspeed
# Single GPU (for API familiarization)
deepspeed --num_gpus=1 code/zero_minimal.py --zero_stage 1
# Multiple GPUs (to see actual sharding benefits)
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 1
```

你应该看到如下输出：

```
ZeRO Stage 1 Demo
Model: 5,248,000 parameters
...
Step 1/10, Loss: 1.0342, Peak Memory: 1.06 GB
...
ZeRO Stage 1 training complete!
Final peak memory: 1.10 GB
```

用单块 GPU，分片效果有限（只有一个分区）。用 2 块 GPU，每块 GPU 只存储优化器状态的一半，你会观察到更低的每 GPU 内存使用。

## ZeRO 阶段 2：优化器状态 + 梯度分区

ZeRO-1 分片优化器状态，但梯度仍然完全复制。对于一个 7B 模型，那仍然是每块 GPU 上 14GB 的梯度（FP16）。ZeRO-2 迈出下一步：也分片梯度。

关键洞见是梯度，像优化器状态一样，只对每块 GPU 负责更新的参数才需要。在反向传播期间，ZeRO-2 不用 `all_reduce`（它给每块 GPU 完整的平均梯度），而用 `reduce_scatter`——每块 GPU 只接收它分配到的那片平均梯度。其余立即被丢弃，随着计算进行释放内存。

用 2 块 GPU，每块现在持有：完整参数 + 一半梯度 + 一半优化器状态。对于我们的 7B 模型，梯度内存从每块 GPU 14GB 降到 7GB。

配置增加了桶大小参数，控制梯度在通信前如何被批处理：

```python
ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    }
}
```

更大的桶通过摊销每个集合操作的开销来提高通信效率，但使用更多内存。500M 元素的默认值对大多数情况效果良好。

要比较 ZeRO-1 和 ZeRO-2：

```bash
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 1
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 2
```

当梯度内存成为瓶颈但你仍然想要参数复制以获得快速前向传播时，ZeRO-2 是一个好选择。

## ZeRO 阶段 3：完全分片（像 FSDP）

ZeRO-2 仍然在每块 GPU 上保持参数复制。对于一个 7B 模型，那是跨所有 rank 复制的 14GB（FP16）参数。ZeRO-3 通过也分片参数来消除这最后的冗余——现在每个组件（参数、梯度、优化器状态）都被分布。

这在功能上等价于 PyTorch FSDP。每块 GPU 只持有所有东西的 1/N。对于 2 块 GPU 上的 7B 模型：7GB 参数 + 7GB 梯度 + 28GB 优化器状态 = 每块 GPU 42GB，相比 DDP 的 14GB + 14GB + 56GB = 84GB。

权衡是通信。由于参数现在被分片，每层在前向计算之前需要一个 `all_gather` 来重构完整权重，然后收集的参数在使用后立即被释放。反向传播做同样的事，外加一个梯度的 `reduce_scatter`。这意味着每次迭代的通信是模型大小的 3 倍（1× 前向 all-gather，1× 反向 all-gather，1× 梯度 reduce-scatter）。

DeepSpeed 通过将通信与计算重叠来缓解这个开销——当一层计算时，下一层的参数在后台被收集。`overlap_comm` 和 `stage3_prefetch_bucket_size` 参数控制这个行为：

```python
ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_max_live_parameters": 1e9
    }
}
```

要看从 ZeRO-1 到 ZeRO-3 的完整递进：

```bash
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 1
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 2
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 3
```

当即使复制的参数也不适合 GPU 内存时，或当你想要最大的内存效率并能容忍一些通信开销时，ZeRO-3 是正确的选择。在带 NVLink 的单节点上，相较 ZeRO-2 的性能差距常常适中（10-20%）；跨许多节点时 all-gather 成本增长，差距扩大。

## ZeRO-Offload：CPU 内存扩展

即使用 ZeRO-3，你也可能耗尽 GPU 内存——特别是在像带 24GB VRAM 的 RTX 4090 这样的消费级硬件上。ZeRO-Offload 通过将优化器状态（以及可选的参数）移到 CPU 内存来解决这个问题，CPU 内存通常大得多且更便宜。

想法很简单：将前向和反向传播保持在它们快速的 GPU 上，但将内存饥渴的优化器状态卸载到 CPU RAM。梯度计算后，它们通过 PCIe 传输到 CPU，优化器步骤在 CPU 上运行，更新后的参数被送回 GPU。DeepSpeed 将这些传输与计算重叠——当 GPU 处理下一批次的前向传播时，CPU 同时运行前一批次的优化器步骤。

瓶颈是 PCIe 带宽（PCIe 4.0 约 32 GB/s，而 GPU HBM 约 2 TB/s）。预期相较仅 GPU 训练有 20-40% 的吞吐量降低。这不是性能优化——它是使训练本来装不下的模型成为可能的可行性解决方案。

```python
ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        }
    }
}
```

`pin_memory: True` 设置使用固定（页锁定）内存以加快 CPU-GPU 传输。要尝试 CPU 卸载：

```bash
deepspeed --num_gpus=1 code/zero_offload_example.py --offload_device cpu
```

你应该看到如下输出：

```
Model: 354.3M parameters (0.71 GB in FP16)
Offload device: cpu
...
Step 0, Loss: 11.0607, GPU Memory: 3.41 GB
Step 10, Loss: 11.0829, GPU Memory: 3.52 GB
...
Training complete with CPU offloading!
Peak GPU memory: 3.52 GB
```

注意 GPU 内存尽管模型大小仍保持低（3.5GB）——优化器状态存在 CPU 上。ZeRO-Offload 对于在 VRAM 有限但系统 RAM 充足的消费级 GPU 上训练是理想的。

## ZeRO-Infinity：用于大型模型的 NVMe 卸载

ZeRO-Offload 将优化器状态移到 CPU，但 CPU RAM 也有限制——工作站上通常 256-512GB。对于真正巨大的模型（数千亿参数），即使 CPU 内存也不够。ZeRO-Infinity 通过使用 NVMe SSD 作为额外的内存层，将卸载更进一步。

![ZeRO-Infinity 内存层次结构。](img/memory_hierarchy_zh.png){#fig:memory-hierarchy .block width=80% align=center}

图~\ref{fig:memory-hierarchy} 显示了三层层次结构：GPU HBM（最快，最小）、CPU RAM（中等）和 NVMe（最慢，最大）。Infinity 引擎管理跨层的数据移动，在参数被需要之前从 NVMe → CPU → GPU 预取它们，并将传输与计算重叠。

现代 NVMe SSD 提供 5-7 GB/s 的顺序读取速度（PCIe Gen4），这比 CPU 内存带宽慢，但以低成本提供 TB 级容量。一个典型设置可能将活跃层参数和激活值保持在 GPU 上，优化器状态和参数缓冲区在 CPU 上，冷参数在 NVMe 上。

配置增加了 NVMe 特定的设置：

```python
ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "buffer_count": 5,
            "buffer_size": 1e8
        }
    },
    "aio": {
        "block_size": 1048576,
        "queue_depth": 16,
        "thread_count": 2
    }
}
```

`aio` 部分为 NVMe 配置异步 I/O——`queue_depth` 和 `thread_count` 控制将读/写与计算重叠的并行性。

要检查你是否有 NVMe SSD 并找到它挂载在哪里：

```bash
# List NVMe devices and partitions with mount points
lsblk -o NAME,SIZE,MOUNTPOINT | grep nvme
# Example output:
# nvme0n1       1.9T
# └─nvme0n1p7   1.8T /home
```

`nvme_path` 必须是 **挂载文件系统上的目录**，而非原始设备路径（如 `/dev/nvme0n1p7`）。NVMe 卸载还需要 `libaio` 库用于异步 I/O：

```bash
# Install libaio (required for NVMe offloading)
sudo apt install libaio-dev  # Ubuntu/Debian
# or: sudo yum install libaio-devel  # CentOS/RHEL
# Increase open file limit (NVMe offloading opens many file handles)
ulimit -n 65535
# Use a directory path, NOT /dev/nvme*
deepspeed --num_gpus=1 code/zero_offload_example.py \
    --offload_device nvme --nvme_path /home/$USER/nvme_offload
```

用你的 NVMe 文件系统上的目录替换 `/home/$USER/nvme_offload`。示例默认训练一个 354M 参数模型——调整 `--hidden_size` 和 `--num_layers` 以试验更大的模型。确保你选择的路径有足够的空闲空间（大约模型大小的 2-4 倍，用于优化器状态和参数缓冲区）。

NVMe 卸载比 CPU 卸载慢（预期 30-50% 的吞吐量降低），而 CPU 卸载本身又比仅 GPU 训练慢。ZeRO-Infinity 的价值不是性能——是可行性。它让你训练那些否则根本装不下的模型。

## ZeRO++：通信优化的 ZeRO

ZeRO-3 消除了内存冗余，但它引入了显著的通信开销[^zero-pp]。

[^zero-pp]: Wang et al., "ZeRO++: Extremely Efficient Collective Communication for Giant Model Training" (2023). https://arxiv.org/abs/2306.10209 每次前向传播需要一个 all-gather 来重构参数；每次反向传播做同样的事，外加一个梯度的 reduce-scatter。对于多节点集群上的大型模型，这种通信会主导训练时间。

ZeRO++ 用三种互补技术解决这个问题（名字遵循论文的记法："q" 表示量化，"hp" 表示分层分区，"Z" 表示 ZeRO）。第一种，**量化权重（qwZ）**，通过以 INT8 而非 FP16 传输参数、然后在接收后反量化来减少 all-gather 流量——通信量减少 2 倍。INT8 只应用于 all-gather 期间传输中的参数；用于优化器步骤的主权重保持全精度，所以舍入误差不会在迭代之间传递。

第二种技术，**分层分区（hpZ）**，利用了节点内通信（NVLink，约 600 GB/s）比节点间（InfiniBand，约 400 GB/s）快得多的事实。hpZ 不是跨所有 GPU 均匀分片，而是在每个节点内复制参数，只跨节点分片。这意味着节点内 all-gather 使用快速的 NVLink，而节点间流量减少到每节点一个代表。

![hpZ 分层分区：ZeRO-3 vs hpZ。](img/hpz_hierarchical_zh.png){#fig:hpz .block width=100% align=center}

图~\ref{fig:hpz} 对比了 2 节点、每节点 2 GPU（共 4 GPU）设置的 ZeRO-3 和 hpZ。在 ZeRO-3（左）中，每块 GPU 持有一个唯一的分片（S0–S3），所以重构完整参数需要跨所有 4 块 GPU 的 all-gather。红色箭头显示每块 GPU 必须跨节点边界与其他每块 GPU 通信——节点 0 中的 S0 和 S1 各需要从节点 1 获取 S2 和 S3，反之亦然。这种跨节点流量使用较慢的 InfiniBand 互连。

在 hpZ（右）中，节点 0 内的两块 GPU 持有相同的分片（S0），节点 1 内的两块 GPU 持有分片 S1。单个绿色箭头表示简化的通信模式：只需要节点之间的一次交换来共享 S0 和 S1。在每个节点内，GPU 已经有相同的数据，所以复制部分不需要节点内通信。这大幅减少了慢速节点间流量的量。

第三种技术，**量化梯度（qgZ）**，在 reduce-scatter 期间对梯度应用相同的 INT8 量化。

配置有选择地启用这些优化：

```python
ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "zero_quantized_weights": True,      # qwZ
        "zero_hpz_partition_size": 2,        # hpZ (GPUs per node)
        "zero_quantized_gradients": True     # qgZ
    }
}
```

`zero_hpz_partition_size` 应匹配你集群中每节点的 GPU 数量。要试验各个优化：

```bash
deepspeed --num_gpus=2 code/zero_pp_example.py --enable_qwz
deepspeed --num_gpus=2 code/zero_pp_example.py --enable_hpz
```

要同时启用所有三个优化：

```bash
deepspeed --num_gpus=2 code/zero_pp_example.py --enable_qwz --enable_hpz --enable_qgz
```

你应该看到如下输出：

```
Model: 124.0M parameters
World size: 2
ZeRO++ features:
  qwZ (Quantized Weights): True
  hpZ (Hierarchical Partitioning): True
  qgZ (Quantized Gradients): True
...
Using quantizer for weights: CUDAQuantizer
...
Step 0, Loss: 10.9776
...
ZeRO++ training complete!
Peak GPU memory: 4.00 GB
```

`CUDAQuantizer` 消息确认 INT8 量化处于活跃状态。注意 ZeRO++ 需要 FP16 精度——量化特性反量化为 FP16，所以使用 BF16 会导致 dtype 不匹配。

ZeRO++ 对于节点间通信是瓶颈的大规模多节点训练最有价值。对于单节点训练或小型集群，好处适中，因为节点内通信已经很快。


## Megatron：作为第二个轴的计算并行

到目前为止，我们探讨的核心始终围绕 **状态分片（State Sharding）** 展开——即如何在多张 GPU 之间对参数、梯度和优化器状态进行切分分片，从而消除显存冗余。FSDP2 与 DeepSpeed ZeRO 等技术本质上解决的是 *显存容量瓶颈*：通过消除重复冗余的模型状态，使得超大模型能够被整个集群的聚合显存池所容纳。

然而，单纯依靠状态分片无法解决前沿超大模型的所有挑战。随着模型参数规模的持续膨胀，另一个彼此正交的物理限制浮出水面：**单层的张量计算量本身已经庞大到无法在单张 GPU 上高效执行**，即便全局显存被充分切分分片。这也正是 Megatron 登上历史舞台的关键时刻。

Megatron-LM 于 2019 年由 NVIDIA 应用深度学习研究团队提出，在具有里程碑意义的论文《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》中首次亮相[^megatron]。彼时，拥有 15 亿参数的 GPT-2 尚被视为庞然大物，学术与工业界刚刚开始探索如何突破单张 GPU 的计算承载边界。NVIDIA 团队敏锐地意识到，单纯为数据并行堆叠更多 GPU 无法解决根本瓶颈：部分网络层由于矩阵尺寸过大，根本无法在单卡上高效执行前向与反向运算。他们的开创性方案是将单个矩阵乘法直接沿维度切分到多张 GPU 上协同计算——这便是 **张量并行（Tensor Parallelism）**。最初的 Megatron 论文展示了在 83 亿参数模型上的训练成果，创下了当时的纪录。时至今日，Megatron 的并行思想已成为训练 GPT-3（175B）、Llama 系列（最高达 405B）以及几乎所有现代前沿大模型赖以运转的算力基石。

[^megatron]: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," arXiv:1909.08053, 2019. https://arxiv.org/abs/1909.08053

### 状态分片 vs 计算分片

要深刻理解 Megatron 的核心价值，有必要梳理 FSDP/ZeRO 与算力分片之间的本质差异。FSDP 与 ZeRO 解决的是 *存储分发* 问题：它们将模型的参数、梯度和优化器状态分摊至各卡，使得没有任何单台设备需要常驻所有权重。当需要计算某一特定层时，框架通过 AllGather 动态拉取该层的完整参数，执行前向与反向传播，随后立即释放并丢弃临时收集的完整张量。这一范式的核心假设是：一旦某一层所需参数就位，该层的张量计算本身能够轻松被单张 GPU 承载。

对于数十亿到上百亿参数的模型，这一假设完全成立。例如一个 7B 模型的最大线性层，其权重矩阵尺寸约为 4096×16384（约 2.7 亿参数，在 FP16 下约占 500 MB），这完全处于现代单张 GPU 矩阵乘法（GEMM）的高效吞吐甜点区。

但当模型参数迈入数千亿级别时，情况发生了质变。以一个典型的 175B 参数模型为例，其隐藏层维度高达 12,288，MLP 中间投影维度为 49,152。此时 MLP 中的单个权重矩阵尺寸便达到 12,288×49,152——仅单层矩阵参数量就超过 6 亿，单权重就需占用 1.2 GB 显存；与此同时，对应的中间激活值张量也呈爆炸式扩张。在这一规模下，即便状态分片做得再极致，单层的矩阵计算也已彻底超出单张 GPU 的高效执行能力，甚至仅该层的中间激活值就会直接撑爆单卡显存。

至此，状态分片触及了其物理天花板：你可以将存储状态切分得无比细碎，但如果单次计算本身装不下，再精巧的显存编排也无能为力。Megatron 正是通过迈出下一步来解决这种失效模式：它不再只局限于改变模型参数 *存放在哪里*，而是彻底重构了模型算子 *如何切分计算*。

### 张量并行：分片层

Megatron 的核心贡献是 **张量并行（TP）**——一种将单个矩阵运算拆分到多块 GPU 的技术。关键洞见很优雅：矩阵乘法沿某些维度本质上是可并行的，我们可以利用这一点来分布计算和内存占用。

考虑一个简单的线性层 $Y = XW$，其中 $X$ 是输入，$W$ 是权重矩阵。如果 $W$ 有形状 $d \times 4d$（MLP 第一个投影的典型形状），我们可以按列将它拆成两半：$W = [W_0 | W_1]$。现在 GPU 0 持有 $W_0$，GPU 1 持有 $W_1$。给定两块 GPU 上相同的输入 $X$，每块计算它的部分：$Y_0 = XW_0$ 和 $Y_1 = XW_1$。完整输出就是 $Y = [Y_0 | Y_1]$——无需通信，只是逻辑拼接。这称为 **列并行线性**。

但下一层呢？它期待完整输入，而非拆分的。这就是 **行并行线性** 的用武之地。如果第二层的权重矩阵按行拆分为 $W' = [W'_0; W'_1]$（垂直堆叠），那么每块 GPU 可以用它的本地输入部分计算一个部分结果：GPU 0 计算 $Y'_0 = Y_0 W'_0$，GPU 1 计算 $Y'_1 = Y_1 W'_1$。最终输出是 $Y' = Y'_0 + Y'_1$——一个求和部分结果的 all-reduce 操作。

图~\ref{fig:tensor-parallel} 展示了这个两步模式。列并行线性（上）按列拆分权重矩阵，所以每块 GPU 无需通信就计算输出的一片。行并行线性（下）按行拆分，一个 all-reduce 组合部分结果。通过配对这两个操作——列并行后接行并行——一个完整的 MLP 块只需要一个 all-reduce。这是 Megatron 效率的关键：通信被最小化到每层一个同步点，而非每个操作。

![张量并行：列并行和行并行线性。](img/tensor_parallelism_zh.png){#fig:tensor-parallel .block width=90% align=center}

相同的原理适用于自注意力。Q、K、V 投影矩阵按列跨 GPU 拆分，所以每块 GPU 为一个注意力头子集计算注意力。由于注意力头是独立的，注意力计算本身期间不需要通信。只有输出投影使用行并行线性，需要一个 all-reduce 来组合结果。

这里有一个重要的微妙之处：与状态分片相比，张量并行从根本上改变了通信模式。用 FSDP 或 ZeRO，通信发生在层 *之间*——你在计算一个层之前 all-gather 参数，然后继续。用张量并行，通信发生在层 *之内*——每个 MLP 和注意力块需要一个 all-reduce。这意味着张量并行对互连带宽敏感得多。实践中，你想让 TP 组在同一节点内，通过快速的 NVLink（约 600 GB/s）连接，而非跨节点通过较慢的 InfiniBand（约 400 GB/s）。

现代实现使用 Megatron 中的 `--tp-comm-overlap` 等技术将这种通信与计算重叠。当一层的 all-reduce 在进行时，下一层的计算可以开始，隐藏大部分延迟。

当启用张量并行时，**序列并行** 成为自然的延伸。序列并行不是跨所有 TP rank 复制激活值，而是沿序列维度拆分激活值。这将激活内存减少一个等于 TP 度的因子——对激活内存可能占主导的长上下文训练至关重要。

一个典型配置看起来像：

```bash
--tensor-model-parallel-size 2    # 2-way tensor parallelism
--sequence-parallel               # Enable sequence parallelism (recommended with TP)
--tp-comm-overlap                 # Overlap TP communication with computation
```

要在较低层次看到张量并行的实际运行，你可以运行代码示例中的纯 PyTorch 实现：

```bash
# Tensor parallelism demo with 2 GPUs
torchrun --nproc_per_node=2 code/tensor_parallel_mlp.py
```

这个示例从头实现列并行和行并行线性层，精确展示权重矩阵如何被拆分以及 all-reduce 如何组合部分结果。运行它有助于建立对 Megatron 底层做什么的直觉。示例日志位于 `code/tensor_parallel_mlp.log`。

### 流水线并行：分片深度

张量并行将单个层拆分到 GPU 上，但还有另一个维度我们可以利用：模型的深度。一个 32 层的 Transformer 不需要每块 GPU 上都有全部 32 层——我们可以将层 0–15 分配给一块 GPU，层 16–31 分配给另一块。这是 **流水线并行（PP）**。

流水线并行的核心理念非常直观：各层计算如同工业流水线一样分工衔接。GPU 0 负责前半部分网络层的前向计算，随后将中间激活值通过点对点通信（P2P）传递给 GPU 1，由后者接力完成剩余层的前向计算。然而，如果采用朴素的单批次串行调度，GPU 1 在 GPU 0 计算期间将完全处于空闲挂起状态，反向传播时亦然。这种因前后阶段等待而导致的算力闲置被称为“流水线气泡（Pipeline Bubble）”，在严重时可能浪费高达 50% 以上的集群算力。

解决方案是将一个全局批次切分为若干更小的 **微批次（micro-batches）**，并驱动它们在各阶段间重叠流动。当 GPU 1 正在处理微批次 1 的层 16–31 时，GPU 0 可以无缝开始处理微批次 2 的层 0–15。只要飞行中的微批次数量足够，就能在绝大多数时间内让所有 GPU 保持满载计算。这一经典调度范式被称为 **1F1B（One Forward One Backward）**：进入稳定阶段后，每块 GPU 在单次前向与单次反向之间严格交替执行，从而将流水线维持在全阶段高效运转的稳态。

图~\ref{fig:pipeline-parallelism} 对比了朴素串行流水线与 1F1B 交错调度的时序差异。在朴素流水线（上图）中，单个批次在 4 张 GPU 间顺序流转——GPU 0 执行前向计算（F），传输给 GPU 1，依次传递至 GPU 3 完成前向，随后反向传播（B）原路逐级回传，空白区域即为空闲等待的气泡。而在 1F1B 调度（下图）中，一个全局 Batch 被切分为 4 个微批次（F1–F4，B1–B4），各 GPU 紧凑交错执行不同微批次的前向与反向计算，从而将空闲气泡大幅压缩。

![流水线并行：朴素串行 vs 1F1B 调度。](img/pipeline_parallelism_zh.png){#fig:pipeline-parallelism .block width=100% align=center}

Megatron 支持几种流水线调度。上面显示的 1F1B 调度是最常见的。原始的 **GPipe** 调度[^gpipe] 先运行所有前向传播，然后所有反向传播——更简单但气泡更大。**交错流水线**[^interleaved]（也称为虚拟流水线并行）更进一步，为每块 GPU 分配多个非连续的层块，以更多通信为代价减少气泡大小。

[^interleaved]: Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," SC 2021. https://arxiv.org/abs/2104.04473

[^gpipe]: Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," NeurIPS 2019. https://arxiv.org/abs/1811.06965

虚拟流水线并行值得特别提及。我们可能不是将层 0–15 分配给 GPU 0、16–31 分配给 GPU 1，而是将层 0–7 和 16–23 分配给 GPU 0，层 8–15 和 24–31 分配给 GPU 1。每块 GPU 现在运行两个"虚拟阶段"。这种交错减少了流水线气泡，因为微批次更快地循环通过各阶段。权衡是各阶段之间额外的点对点通信。

一个典型配置看起来像：

```bash
--pipeline-model-parallel-size 4          # 4 pipeline stages
--num-layers-per-virtual-pipeline-stage 2 # VPP: 2 layers per virtual stage
```

流水线并行在一个特定场景中大放异彩：**多节点训练**。在一个节点内，GPU 通过快速的 NVLink（约 600 GB/s）连接。跨节点，你被限制在 InfiniBand（约 400 GB/s）或更差。张量并行需要在每层内频繁的 all-reduce 操作——在 NVLink 上没问题，在 InfiniBand 上痛苦。流水线并行只需要各阶段之间激活值的点对点通信，对较慢的互连宽容得多。

实用指南是：在节点内使用张量并行（NVLink 保持通信快速），跨节点使用流水线并行（通信模式更宽容）。这种组合——节点内 TP，跨节点 PP——是在 100B+ 规模训练模型的标准方法。

要用简化实现看到流水线并行的实际运行：

```bash
# Pipeline parallelism demo with 2 stages
torchrun --nproc_per_node=2 code/pipeline_parallel_simple.py
```

这个示例演示了模型分区、微批次调度以及跨流水线阶段的前向/反向协调。


### 序列并行与上下文并行：长上下文

到目前为止，我们讨论了解决模型大小的并行策略——分片参数、梯度、优化器状态和计算。但还有另一个变得越来越重要的维度：**序列长度**。现代模型用 8K、32K、甚至 128K token 的上下文窗口训练。在这些长度下，激活内存——前向传播期间为反向传播使用而存储的中间值——可以超过模型本身所需的内存。

考虑一个隐藏维度 4096、处理 32K token 序列的 Transformer。每层存储形状为（批次，32K，4096）的激活值，用 32 层，激活内存容易达到每块 GPU 几十 GB。这就是 **序列并行** 和 **上下文并行** 的用武之地。

**序列并行**[^seqpar] 是两者中较简单的。当启用张量并行时，某些操作如 LayerNorm 和 Dropout 不参与 TP 通信——它们在本地对完整隐藏维度操作。序列并行通过沿序列维度拆分激活值，将分片扩展到这些操作。如果你有 TP=4，序列并行意味着每块 GPU 只为这些操作存储序列激活值的 1/4。它通常与张量并行一起启用，开销极小。

[^seqpar]: Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," MLSys 2023. https://arxiv.org/abs/2205.05198

**上下文并行（CP）**[^ringatt] 采取更激进的方法。序列并行只分片 LayerNorm 和 Dropout 的激活值，而上下文并行沿序列维度分区 *所有东西*——输入、所有中间激活值和注意力计算本身。用 CP=2 在 8K 序列上，每块 GPU 在整个前向和反向传播中只处理 4K token。

[^ringatt]: Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context," ICLR 2024. https://arxiv.org/abs/2310.01889

挑战是注意力。在标准自注意力中，每个 token 的查询必须关注序列中所有的键和值。如果 GPU 0 持有 token 0–3999，GPU 1 持有 token 4000–7999，GPU 0 上的查询如何关注 GPU 1 上的键？上下文并行使用一种称为 **环形注意力（ring attention）** 的技术解决这个问题。想法很优雅：不是将所有 KV 对收集到每块 GPU（那会抵消内存节省），而是在环中传递 KV 块。GPU 0 为它的查询对其本地 KV 计算注意力，然后将它的 KV 发送给 GPU 1 并接收 GPU 1 的 KV。现在 GPU 0 对新的 KV 块计算注意力，累积结果。在环中完整旋转一圈后，每个查询都见过每个键值对，但没有 GPU 曾持有完整序列。

通信模式被仔细优化。现代实现将 KV 传输与注意力计算重叠——当对当前 KV 块计算注意力时，下一个块已经在传输。结合分组查询注意力（GQA），它跨多个查询头共享 KV 头，通信量大幅减少。

好处是显著的。没有 CP，在非常长的序列上训练常常需要激活重计算（在反向传播期间重新计算激活值），这增加约 30% 的开销。有 CP，你可以通过简单地跨更多 GPU 分布激活内存来完全消除这个重计算。权衡是通信，但对长序列，计算与通信的比率仍然有利。

图~\ref{fig:seq-ctx-parallel} 展示了两种技术。序列并行（左）沿序列维度拆分激活值——每块 GPU 只为 LayerNorm 和 Dropout 操作存储它那部分序列。上下文并行（右）使用环形注意力：每块 GPU 持有本地 Q、K、V 块，K、V 对在环中旋转，使每个查询可以关注所有键，而没有 GPU 持有完整序列。

![序列并行和上下文并行（环形注意力）。](img/sequence_context_parallelism_zh.png){#fig:seq-ctx-parallel .block width=100% align=center}

长上下文训练的典型配置：

```bash
--tensor-model-parallel-size 2
--context-parallel-size 4        # Split 32K sequence across 4 GPUs = 8K per GPU
--sequence-parallel              # Also enable sequence parallelism
```

经验法则：当序列长度超过 8K token 且激活内存是你的瓶颈时使用上下文并行。对于较短的序列，张量并行和序列并行通常足够。

配套的 `code/sp_demo.py` 展示了这些概念。`memory` 模式计算各种序列长度的激活内存——你会看到 32K 序列带 32 层可以超过 40GB，解释了为什么并行是必要的。`sequence_parallel` 模式展示每块 GPU 如何只持有序列的一部分并在本地应用 LayerNorm 而无需通信。`ring_attention` 模式演示核心的环形注意力模式：每块 GPU 从本地 Q、K、V 块开始，然后 K 和 V 在环中旋转。完整旋转一圈后，每个查询都关注了所有键，然而没有 GPU 曾持有完整序列。

```bash
# Show activation memory scaling (single GPU)
python code/sp_demo.py --mode memory
# Sequence parallelism: split activations along sequence dim (2 GPUs)
torchrun --nproc_per_node=2 code/sp_demo.py --mode sequence_parallel
# Context parallelism via ring attention (2 GPUs)
torchrun --nproc_per_node=2 code/sp_demo.py --mode ring_attention
```

这些运行的示例日志在 `code/sequence_parallel.log` 和 `code/ring_attention.log`。

### DeepSpeed-Ulysses：环形注意力的替代方案

环形注意力不是跨长序列并行化注意力的唯一方式。DeepSpeed 引入了 **DeepSpeed-Ulysses**[^ulysses]，一种以通信模式换简单性的不同方法。

[^ulysses]: Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models," arXiv 2023. https://arxiv.org/abs/2309.14509

Ulysses 背后的关键洞见很直接：不是在环中旋转 KV 块，为什么不在注意力之前收集完整序列、之后再散射回来？在环形注意力中，每块 GPU 对不同的 KV 块计算部分注意力分数并累积它们——这需要仔细记录跨块的 softmax 归一化。Ulysses 完全避开了这种复杂性。

它是这样工作的。在注意力计算之前，每块 GPU 持有形状为（批次，local_seq，num_heads，head_dim）的序列块。Ulysses 执行一个 **all-to-all** 通信来重组数据：不是每块 GPU 为序列的一部分持有所有头，而是每块 GPU 现在为头的一部分持有所有序列位置。这个转置之后，形状变成（批次，full_seq，local_heads，head_dim）。现在每块 GPU 可以对它的头子集计算标准自注意力——没有部分分数，没有累积，只是常规注意力。注意力之后，另一个 all-to-all 逆转变换，返回原始分区。

![DeepSpeed-Ulysses：通过 all-to-all 的 2D 转置。](img/ulysses_zh.png){#fig:ulysses .block width=85% align=center}

图~\ref{fig:ulysses} 展示了这个 2D 转置。在左侧（序列并行维度），每块 GPU 持有一个本地序列切片 $S_i$ 以及完整的注意力头集合 $H_{all}$——张量形状为 (local_seq, all_heads)。通过 All-to-All 集合通信原语执行分布式转置：在右侧（头并行维度），每块 GPU 持有完整序列 $S_{all}$，但仅负责局部的注意力头子集 $H_i$——张量形状相应转置为 (full_seq, local_heads)。此时每块 GPU 即可针对其专属的头子集独立执行标准的自注意力计算，无需引入跨卡 Partial Softmax 累加。自注意力计算完成后，再次执行一次逆向 All-to-All 通信，即可将数据布局无缝恢复至原始的序列分片格式。

权衡是通信量对通信模式。环形注意力在 P 块 GPU 的环中发送 KV 块 P 次，每次传输与计算重叠。Ulysses 执行两个 all-to-all 集合操作（注意力前后），它们同时涉及所有 GPU。对于小的并行度（P ≤ 8），Ulysses 常常胜出，因为像 NVLink 这样的现代互连上的 all-to-all 高度优化。对于较大的 P 或当互连带宽有限时，环形注意力的重叠通信可以更高效。

实践中，DeepSpeed-Ulysses 在你想要序列并行而不要环形注意力部分 softmax 处理复杂性的场景中大放异彩。它与 ZeRO 和其他 DeepSpeed 优化干净地集成。

配套的 `code/ulysses_demo.py` 从头实现核心 Ulysses 算法。它演示两个 all-to-all 操作：`all_to_all_seq_to_head` 从（批次，local_seq，num_heads，head_dim）转置到（批次，full_seq，local_heads，head_dim），`all_to_all_head_to_seq` 逆转这个变换。在这两个操作之间，每块 GPU 对它的头子集运行标准自注意力——不需要部分 softmax 累积。

```bash
torchrun --nproc_per_node=2 code/ulysses_demo.py
```

一个示例运行在 `code/ulysses_demo.log`。何时应该选择 Ulysses 而非环形注意力？如果你已经在 DeepSpeed 生态系统中并想要中等并行度的直接序列并行，Ulysses 是更容易的路径。如果你扩展到非常长的序列（100K+ token）且并行度大，环形注意力的通信重叠可能提供更好的效率。


### 专家并行：扩展 MoE 模型

混合专家（MoE）模型呈现了一个独特的扩展机会：不是让每一层更宽，而是添加多个"专家"子网络，并将每个 token 只路由到它们的一个子集。像 Mixtral 8x7B 这样的模型每个 MoE 层有 8 个专家，但每个 token 只激活其中 2 个。这意味着模型有一个大得多的网络的容量，同时保持每 token 计算可管理。但我们如何将这些专家分布到 GPU 上？

这就是 **专家并行（EP）** 的用武之地。想法很自然：如果我们有 8 个专家和 8 块 GPU，将一个专家放在每块 GPU 上。当一个 token 需要被专家 3 处理时，它被路由到 GPU 3，被处理，结果被送回。通信模式是 all-to-all：来自所有 GPU 的 token 可能需要去任何专家，结果流回它们的源头。图~\ref{fig:expert-parallelism} 展示了这种分布模式。

![专家并行将专家分布到多块 GPU 上](img/expert_parallelism_zh.png){#fig:expert-parallelism .block width=90% align=center}


挑战是负载均衡。如果路由器将 80% 的 token 发送给专家 0，只有 2% 给专家 7，GPU 0 就过载而 GPU 7 闲置。MoE 训练通常包括一个辅助损失，鼓励路由器更均匀地分布 token。Megatron 支持几种负载均衡策略：辅助损失（为不均衡路由添加惩罚）、Sinkhorn（迭代归一化以强制平衡），以及通过架构约束实现平衡的无辅助损失方法。

专家并行与其他并行维度自然结合。一个典型的大规模 MoE 训练可能对专家使用 EP=8，对流水线阶段使用 PP=4，对跨节点数据并行使用 DP。非专家层（注意力、LayerNorm）可以独立使用张量并行。这种灵活性对像 DeepSeek-V3 或 Qwen-MoE 这样有数百个专家的模型至关重要。

Mixtral 8x7B 训练的配置可能看起来像：

```bash
--num-experts 8
--expert-model-parallel-size 8   # One expert per GPU
--moe-router-topk 2              # Each token activates 2 experts
--moe-router-load-balancing-type aux_loss
--moe-grouped-gemm               # Batch expert computations
--pipeline-model-parallel-size 4
```

`--moe-grouped-gemm` 标志值得注意：当一块 GPU 托管多个专家（EP < num_experts）时，它将跨专家的计算批处理成单个分组矩阵乘法，显著提高 GPU 利用率。对于非常大的专家数量，像 DeepEP[^deepep] 这样的专门通信库用低延迟 GPU 内核和高效的跨节点传输优化 all-to-all token 调度。

[^deepep]: DeepEP 是 DeepSeek 的开源专家并行通信库。https://github.com/deepseek-ai/DeepEP

配套的 `code/expert_parallel_demo.py` 是一个从头实现，演示核心 EP 机制而无 Megatron 依赖。它展示路由器如何将 token 分配给专家、all-to-all 通信如何将 token 调度到它们的目标 GPU、每块 GPU 如何用它的本地专家处理 token，以及另一个 all-to-all 如何返回结果。演示打印 token 分布统计，使你可以看到路由决策如何影响负载平衡。

```bash
torchrun --nproc_per_node=2 code/expert_parallel_demo.py
```

一个示例运行在 `code/expert_parallel_demo.log`。

### 为什么 FSDP2 不能替代 Megatron

一个常见的误解是 FSDP2（或 ZeRO）和 Megatron 可互换——你根据偏好选择其一。这误解了每个系统实际做什么。

FSDP2 和 ZeRO 分片 *状态*：参数、梯度和优化器状态被分布到 GPU 上，然后在计算需要时收集。关键假设是每层的前向和反向传播适合单块 GPU。当 GPU 0 需要计算层 5 时，它从所有 GPU 收集层 5 的参数，在本地运行计算，然后释放内存。计算本身不被分布——只有存储被分布。

Megatron 分片 *计算*：单个层的矩阵乘法被拆分到多块 GPU 上，每块 GPU 计算结果的一部分。通信发生在层 *之内*，而非围绕它。这与在计算之前收集参数根本不同。

为什么这个区别重要？考虑一个模型，其中单个注意力层有一个 16K × 16K 权重矩阵。即使你用 FSDP2 将参数分片到 8 块 GPU，当到了计算时，一块 GPU 必须收集完整矩阵并执行乘法。如果那个矩阵不适合一块 GPU 的内存，或如果计算在一块 GPU 上太慢，FSDP2 无能为力——它只分片存储，而非计算。

这是 Megatron 变得必要的地方。用张量并行，那个 16K × 16K 矩阵被拆分到 8 块 GPU 上，每块持有一个 16K × 2K 片。计算并行发生，只有结果被通信。没有单个 GPU 曾需要持有或用完整矩阵计算。

实际含义：FSDP2 和 Megatron 是互补的，而非竞争的。你可能用 Megatron 的张量并行将大层拆分到节点内的 GPU 上，同时跨节点用 FSDP2 式分片以实现内存效率。选择不是"哪一个"而是"如何组合它们"。

### 混合并行：结合状态和计算分片

鉴于 FSDP2/ZeRO 和 Megatron 解决不同问题，自然的问题是：我们能同时用两者吗？答案是能，而这正是大规模训练系统所做的。

考虑在 8 个节点的 64 块 GPU 上训练一个 70B 参数模型。在每个节点内（8 块通过 NVLink 连接的 GPU），你用 Megatron 的张量并行 TP=8 来拆分大矩阵乘法。跨节点（通过较慢的 InfiniBand 连接），你用 ZeRO-3 或 FSDP2 来分片优化器状态和梯度——这在不需要张量并行要求的高带宽通信的情况下减少内存压力。如果模型深，你可能添加流水线并行以跨节点组分布层。

这种分层方法发挥每种技术的优势。张量并行需要高带宽（因此节点内 NVLink），但它使本来装不下单块 GPU 的计算成为可能。状态分片容忍更高的延迟（因此跨节点），但它大幅减少每 GPU 内存。流水线并行以相对适中的通信增加另一个扩展维度。

状态分片层在 FSDP2 和 ZeRO-3 之间的选择取决于你的生态系统。FSDP2 与 PyTorch 的编译器栈（torch.compile）紧密集成，是原生 PyTorch 解决方案。ZeRO-3，通过 DeepSpeed，为内存受限的设置提供 CPU 和 NVMe 卸载等额外特性，并有成熟的优化生态系统。两者都与 Megatron 风格的计算分片干净地结合——它们在正交的轴上运作。如果你已经用张量或流水线并行运行 Megatron Core，Megatron 的分布式优化器或 Megatron-FSDP 通常比将 DeepSpeed ZeRO-3 硬塞到同一作业上更容易接线，后者中进程组必须严格保持不相交。不喜欢手动组装 Megatron 和 DeepSpeed 的团队也可以评估像 **Colossal-AI**[^colossalai] 这样的集成框架，它打包了混合并行（DP + TP + PP）、ZeRO 式分片和 **Gemini**[^gemini] 异构内存管理（GPU、CPU 和 NVMe）。

[^colossalai]: Colossal-AI：\url{https://colossalai.org/}
[^gemini]: Colossal-AI Gemini 异构内存管理器（不是 Google 的 Gemini）：\url{https://colossalai.org/docs/advanced_tutorials/meet_gemini/}

### Megatron Core：生产就绪的库

贯穿本章，我们在概念上讨论了 Megatron 的并行策略。但你实际上如何在实践中使用它们？答案是 **Megatron Core**，一个从原始 Megatron-LM 研究代码库提取并为生产使用精炼的库。

Megatron Core 提供 GPU 优化的构建块：内建张量并行的注意力层、理解流水线边界的 MLP 块、处理词汇表并行的嵌入层。你不自己实现列并行和行并行模式——你使用库中的 `ColumnParallelLinear` 和 `RowParallelLinear`，通信被自动处理。

除了基本构建块，Megatron Core 包括大规模训练所需的基础设施：以计算换内存的激活重计算、高效保存和加载分片模型状态的分布式检查点，以及为 NVIDIA 最新 GPU（Hopper、Ada、Blackwell）优化的 FP8 精度支持。分布式优化器跨数据并行 rank 分片优化器状态，补充我们讨论的计算分片。

Megatron Core 需要 python 包 pybind11 和系统库 cuDNN 与 NCCL。先安装它们，然后：

```bash
pip install --no-build-isolation megatron-core[mlm,dev] pybind11

# Or use NVIDIA's container with cuDNN and NCCL pre-installed
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.04-py3
```

本章代码目录中的 `code/megatron_gpt_pretrain.sh` 脚本演示了一个生产配置：跨 GPU 的张量并行、用于内存效率的分布式优化器、用于速度的 flash attention，以及我们贯穿本章讨论的各种标志。

### Megatron-FSDP：优化的状态分片

我们已经确立状态分片（FSDP/ZeRO）和计算分片（Megatron）是互补的。但当你组合它们时，实现细节很重要。PyTorch 的 FSDP2 是一个通用解决方案；它不知道 Megatron 的张量并行或涉及的特定通信模式。这就是 **Megatron-FSDP** 的用武之地。

Megatron-FSDP 是 NVIDIA 对完全分片数据并行的实现，设计为与 Megatron 的其他并行维度无缝协作。在 NVIDIA 发布的基准中，它相较 PyTorch FSDP2 提供大约 15-25% 更高的吞吐量和约 23% 的内存节省。[^megatron-fsdp] 这些收益来自只有当 FSDP 实现理解周围上下文时才可能的优化——更好的参数分桶、更聪明的缓冲区管理，以及更激进的通信与计算重叠。

[^megatron-fsdp]: NVIDIA Megatron Core, "Megatron-FSDP." https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/megatron_fsdp.html

一个值得注意的技术细节：Megatron-FSDP 使用 NCCL 的 userbuffer 特性来减少通信期间的 GPU 流式多处理器（SM）消耗。在大规模训练中，花在通信上的 SM 是不可用于计算的 SM。这个优化保持更多 SM 空闲用于实际的矩阵乘法。

在你的训练脚本中启用 Megatron-FSDP 看起来像：

```bash
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params  # Equivalent to ZeRO-3
--overlap-grad-reduce
--overlap-param-gather
```

何时应该选择 Megatron-FSDP 而非 PyTorch FSDP2？如果你已经在用 Megatron 的张量并行、上下文并行或专家并行，Megatron-FSDP 自然集成并提供更好的性能。如果你需要用 Transformer Engine 进行 FP8 训练，Megatron-FSDP 有原生支持。另一方面，如果你想要带 torch.compile 支持且无外部依赖的纯 PyTorch 栈，PyTorch FSDP2 是更简洁的选择。

### 分布式优化器：内存高效的优化

即使张量并行和流水线并行处理计算，优化器状态仍然是显著的内存负担。例如，Adam 为每个参数存储两个额外的张量（动量和方差）——那是在参数和梯度本身之上每参数 8 字节。对于一个 70B 模型，仅优化器状态就可以消耗超过 500GB。

Megatron 的 **分布式优化器** 通过跨数据并行 rank 分片优化器状态来解决这个问题，概念上类似于 ZeRO 阶段 1。每块 GPU 只存储一部分参数的优化器状态。在优化器步骤期间，梯度被 reduce-scatter（每块 GPU 得到它分片的归约梯度），本地优化器更新它的分片，更新后的参数被 all-gather 回来。

内存节省随数据并行大小缩放。用 8 路数据并行、bf16 参数和 fp32 梯度，每 GPU 内存从每参数 18 字节降到大约 7.5 字节——2.4 倍的减少。确切公式取决于你的精度配置：

| 配置 | 无分布式 | 分布式（d 块 GPU） |
|--------------|-----------------|-------------------------------------|
| fp16 参数，fp16 梯度 | 20 字节/参数 | 4 + 16/d 字节/参数 |
| bf16 参数，fp32 梯度 | 18 字节/参数 | 6 + 12/d 字节/参数 |
| fp32 参数，fp32 梯度 | 16 字节/参数 | 8 + 8/d 字节/参数 |

实现包括几个超出基本分片的优化。梯度在计算时被复制到连续缓冲区中，实现高效的 reduce-scatter 操作。通信可以与反向计算（`--overlap-grad-reduce`）和下一次前向传播（`--overlap-param-gather`）重叠，隐藏大部分延迟。

```bash
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

### FP8 训练：下一代精度

从 FP32 到 FP16/BF16 的递进带来了显著的加速和内存节省。NVIDIA 最新的 GPU（Hopper、Ada、Blackwell）用原生 FP8 支持更进一步——8 位浮点，相比 FP16 将内存减半并将吞吐量翻倍。

FP8 训练不像改变一个 dtype 标志那么简单。8 位浮点的动态范围比 16 位窄得多，所以值必须仔细缩放以避免溢出和下溢。Megatron 通过 Transformer Engine 处理这个，它跟踪张量的最大绝对值（amax）并动态调整缩放因子。`--fp8-amax-history-len` 参数控制计算缩放时考虑多少个最近的 amax 值。

```bash
--fp8-format hybrid
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
--fp8-param-gather          # Gather parameters in FP8 to save communication
```

`hybrid` 格式为前向传播使用 E4M3（4 指数位，3 尾数位），为反向使用 E5M2（5 指数位，2 尾数位）——一种在实践中效果良好的范围与精度平衡。`--fp8-param-gather` 标志与分布式优化器特别有用：参数以 FP8 格式收集，将 all-gather 通信量减半。

FP8 需要硬件支持：NVIDIA H100、RTX 4090 或更新的 GPU，加上 Transformer Engine 1.1 或更高。如果你有硬件，加速是显著的——对大矩阵乘法常常比 BF16 快 1.5-2 倍。

### 你何时需要 Megatron？

在所有关于张量并行、流水线并行、上下文并行和专家并行的讨论之后，一个自然的问题是：你实际上何时需要这些？答案取决于你的模型和硬件。

如果单个 Transformer 层舒适地适合一块 GPU 并计算得足够快，你不需要 Megatron。状态分片（FSDP2 或 ZeRO）处理内存，数据并行处理扩展。现代 GPU 上大多数 10B 参数以下的模型落入这一类。

Megatron 在你撞上以下墙之一时变得必要。第一，层大小：如果你的隐藏维度是 16K 或更大，单个注意力层的权重矩阵可能不适合一块 GPU，或计算可能太慢。张量并行解决这个。第二，序列长度：如果你用 8K+ token 上下文训练，激活内存爆炸，上下文并行或序列并行变得必不可少。第三，MoE 模型：专家并行是分布数百个专家的自然方式。第四，规模：当你使用数百块 GPU 时，Megatron 优化的通信模式带来的效率收益显著累积。

如果这些都不适用——你的层装得下、你的序列适中、你不用 MoE、你在少数几块 GPU 上训练——仅状态分片就更简单且足够。

### 真实世界的训练配置

具体的配置有助于巩固理解。下面的示例基于实际的 Megatron 训练脚本。`pretrain_gpt.py` 脚本是 Megatron-LM 仓库的一部分：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM
```

仓库在 `docs/` 中包含文档，在 `examples/` 中包含按模型组织的示例脚本（如 `examples/llama/`、`examples/gpt3/`、`examples/mixtral/`）。

__带长上下文的 LLaMA-3 8B（8 × 80GB GPU）：__

这个配置在单个 8 GPU 节点（A100-80GB、H100、H200 等）上训练一个带 8K 上下文的 LLaMA-3 8B 模型。

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 8192 \
    --tensor-model-parallel-size 1 \
    --context-parallel-size 2 \
    --sequence-parallel \
    --fp8-format hybrid \
    --fp8-param-gather \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --max-position-embeddings 8192 \
    --mock-data \
    --bf16
```

模型架构标志定义 LLaMA-3 8B 结构。`--group-query-attention` 配合 `--num-query-groups 8` 启用分组查询注意力，其中 32 个查询头共享 8 个 KV 头——显著减少 KV 缓存内存。对于并行，我们跳过张量并行（`--tensor-model-parallel-size 1`），因为每层适合一块 GPU，但使用上下文并行（`--context-parallel-size 2`）将 8K 序列拆分到 2 块 GPU 上。总共 8 块 GPU，有效数据并行大小是 8 / (1 × 1 × 2) = 4——三个因子是张量、流水线（这里都是 1）和上下文并行。这使 `--global-batch-size 128` 算出每个数据并行副本 128 / 4 = 32 个样本，或 32 / `--micro-batch-size 1` = 每个梯度累积周期 32 个微批次。FP8 标志（`--fp8-format`、`--fp8-param-gather`）在 Hopper 及更新的 GPU 上提供加速；对 A100 移除它们。对于 40GB GPU，通过降低 `--num-layers`、`--hidden-size` 和 `--ffn-hidden-size` 减小模型大小。

__GPT-3 175B 规模（128 块 GPU）：__

这个配置扩展到跨 16 个节点（共 128 块 GPU）的 175B 参数，需要张量和流水线并行两者。

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 --nnodes=16 \
    pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 2048 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 16 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --use-distributed-optimizer \
    --mock-data \
    --bf16
```

在这个规模，一个隐藏维度 12288 的单层受益于跨节点内所有 8 块 GPU 的张量并行（`--tensor-model-parallel-size 8`）。96 层被分布在 16 个流水线阶段（`--pipeline-model-parallel-size 16`），每个阶段处理 6 层。有效数据并行是 128 / (8 × 16) = 1，意味着所有 GPU 都专用于模型并行。大的 `--global-batch-size 1536` 通过跨许多微批次的梯度累积实现。

__Mixtral 8x7B MoE（64 块 GPU）：__

这个配置训练一个 8 个专家分布在 GPU 上的混合专家模型。

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 --nnodes=8 \
    pretrain_gpt.py \
    --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-experts 8 \
    --expert-model-parallel-size 8 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --moe-router-topk 2 \
    --moe-grouped-gemm \
    --moe-permute-fusion \
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --mock-data \
    --bf16
```

MoE 特定的标志定义稀疏架构：`--num-experts 8` 每个 MoE 层创建 8 个专家网络，`--expert-model-parallel-size 8` 在专家并行组内每块 GPU 分布一个。`--moe-router-topk 2` 意味着每个 token 被路由到 2 个专家。优化标志 `--moe-grouped-gemm` 和 `--moe-permute-fusion` 批处理专家计算并融合 token 重排操作以提高效率。流水线并行（`--pipeline-model-parallel-size 4`）将 32 层分布在 4 个阶段。张量并行设为 1，因为专家并行已经分布了计算。

### 用 Megatron 的完整训练示例

为了把所有东西串起来，让我们看看如何实际运行一个 Megatron 训练作业。本章的 `code/train_megatron_mcore.py` 脚本演示了基本部分：初始化分布式状态、用 `TransformerConfig` 创建模型、用 Megatron 的 `DistributedDataParallel` 包装它，以及运行带适当梯度同步的训练循环。

在单节点 4 块 GPU 上运行：

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 \
code/train_megatron_mcore.py
```

对于多节点训练，指定集群拓扑。在节点 0 上：

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 --nnodes=2 \
    --node_rank=0 \
    --master_addr=node0 --master_port=29500 \
    code/train_megatron_mcore.py
```

在节点 1 上，用 `--node_rank=1` 运行相同的命令。脚本使用 Megatron Core 的 `GPTModel`，它内建张量并行——你不手动实现列并行和行并行模式。Megatron 的 DDP 包装器用优化的通信重叠处理梯度同步，分布式优化器自动分片优化器状态。

对于需要计算分片和激进状态分片两者的模型，通过添加这些标志将 Megatron 的张量并行与 Megatron-FSDP 结合：

```bash
--use-megatron-fsdp \
--data-parallel-sharding-strategy optim_grads_params \
--tensor-model-parallel-size 4 \
--overlap-grad-reduce \
--overlap-param-gather
```

这种组合给你两全其美：张量并行将大矩阵乘法拆分到 GPU 上，而 FSDP 跨数据并行维度分片参数、梯度和优化器状态。重叠标志确保通信尽可能与计算并发发生。

生产中值得启用的几个额外优化。`--tp-comm-overlap` 将张量并行的 all-reduce 与计算重叠。`--sequence-parallel` 通过为 LayerNorm 和 Dropout 沿序列维度分片来减少激活内存。`--calculate-per-token-loss` 为变长序列优化梯度缩放。

除了这些标志，Megatron 提供几个高级特性。虚拟流水线并行（交错调度）通过让每块 GPU 处理多个非连续阶段来减少流水线气泡。分布式检查点比天真的 PyTorch 检查点快多达 50 倍地保存和加载分片模型状态，支持重新分片——你可以保存一个 64 GPU 运行的检查点并在 128 块 GPU 上加载它。CUDA 图捕获整个训练迭代并以最小的 CPU 开销重放它们。激活重计算让你在需要时在反向传播期间选择性地重新计算激活值以计算换内存。

## 实践中的混合并行

我们现在已经涵盖了各个并行技术：ZeRO 的状态分片、Megatron 的张量和流水线并行、用于长序列的序列和上下文并行，以及用于 MoE 模型的专家并行。但真实的训练系统很少只用一个。一个 70B 模型可能在节点内用张量并行、跨节点用流水线并行、用 FSDP 分片优化器状态——所有这些同时进行。这些部分如何组合在一起？

### 双轴视角

关键洞见是状态分片和计算分片在正交的轴上运作。状态分片（FSDP、ZeRO）解决内存冗余：不是每块 GPU 存储所有参数和优化器状态，而是每块 GPU 存储一部分。计算分片（张量并行、流水线并行）解决计算负载：不是一块 GPU 计算整个层，而是多块 GPU 分担工作。

这些轴是独立的。你可以有无计算分片的状态分片（带 ZeRO-3 的 7B 模型）、无状态分片的计算分片（带 TP=8 和完整参数复制的 70B 模型），或两者（带 TP、PP 和 FSDP 的 405B 模型）。选择取决于你撞上哪个瓶颈。

### 构建混合配置

让我们走一遍你可能如何配置一个在 8 个节点的 64 块 GPU 上训练的 70B 模型。每个节点有 8 块通过 NVLink 连接的 GPU；节点通过 InfiniBand 连接。

从张量并行开始。模型的隐藏维度是 8192，每个注意力层有大权重矩阵。我们设 TP=4，将每层的计算拆分到节点内的 4 块 GPU 上。这需要高带宽通信（NVLink），所以我们将 TP 组保持在单个节点内。

接下来，考虑流水线并行。模型有 80 层，即使用 TP=4，存储所有层的激活值也很有挑战。我们设 PP=2，将模型拆分成每个 40 层的两个流水线阶段。流水线通信（在各阶段之间发送激活值）比 TP 通信不那么频繁，所以它可以容忍较慢的节点间 InfiniBand。

最后，数据并行。用 TP=4 和 PP=2，每个"模型副本"使用 8 块 GPU。我们总共有 64 块 GPU，所以 DP=8：八个副本并行处理不同的微批次。我们启用 FSDP 跨这 8 个副本分片优化器状态，减少每 GPU 内存。

算术：总 GPU = TP × PP × DP = 4 × 2 × 8 = 64。从单块 GPU 的视角，它存储 1/8 的优化器状态（FSDP）、计算每层的 1/4（TP），并处理模型深度的 1/2（PP）。

### 为什么这有效

这种分层方法成功，因为每种技术解决不同的约束。FSDP 消除冗余的优化器状态存储——对 Adam 的动量和方差张量至关重要。张量并行使本来装不下或在单块 GPU 上太慢的矩阵乘法成为可能。流水线并行通过限制同时活跃多少层来限定激活内存。每种技术都有成本（通信开销、流水线气泡），但当应用于正确的瓶颈时，好处大于成本。

### 选择你的配置

决策过程是渐进的。从简单开始，仅在需要时增加复杂性。

如果你的模型的层适合一块 GPU 并高效计算，仅用状态分片（FSDP 或 ZeRO-3）。这是最简单的设置，对现代 GPU 上大多数 10-15B 参数以下的模型有效。

如果层在一块 GPU 上太大或太慢，添加张量并行。将 TP 保持在节点内以利用 NVLink。TP=2、4 或 8 是取决于层大小的常见选择。

如果模型非常深或你需要跨许多节点扩展，添加流水线并行。PP 引入气泡，所以在必要时使用它而非默认。

如果序列长度是你的瓶颈（8K+ token），启用序列并行或上下文并行。这些按并行度成比例地减少激活内存。

如果你训练一个 MoE 模型，专家并行自然地分布专家。EP 常常替代专家层的 TP，因为专家已经是单独的计算。

### 运营现实

混合并行增加运营复杂性。张量并行组必须放在带快速互连的 GPU 上——将 TP 组跨节点会削弱性能。流水线并行需要调整微批次数量以最小化气泡开销。检查点变得更复杂：TP=4、PP=2 配置的检查点不能不重新分片就直接加载到 TP=8、PP=1 设置。

调试也变得更难。一个 bug 可能只在特定并行配置下表现，使重现困难。由于这些原因，从满足你需求的最简单配置开始，并渐进地添加并行维度。

### 性能优化最佳实践

一旦你有一个可工作的混合配置，有几个优化可以显著提高吞吐量。

最有影响力的是通信重叠。默认情况下，通信和计算顺序发生——GPU 计算，然后通信，然后再计算。启用重叠后，通信在后台发生，而下一次计算继续。Megatron 提供几个重叠标志：

```bash
--overlap-grad-reduce          # Overlap gradient all-reduce with backward
--overlap-param-gather         # Overlap parameter gather with forward
--tp-comm-overlap              # Overlap tensor parallel all-reduce
```

对于内存优化，序列并行通过为 LayerNorm 和 Dropout 沿序列维度分片来减少激活内存。分布式优化器跨数据并行 rank 分片优化器状态。激活重计算通过在反向期间重新计算激活值而非存储它们来以计算换内存。

```bash
--sequence-parallel
--use-distributed-optimizer
--recompute-activations        # When memory-constrained
```

基于通信特征的几个拓扑指南。张量并行和专家并行是通信密集型的——将它们保持在 NVLink 域内（同一节点）。流水线并行容忍更高的延迟——它可以跨节点。用于长序列的上下文并行在节点内工作最好，但如有必要可以跨节点扩展。

### 配置模式

从生产训练设置中浮现出几个模式。对于 10B 参数以下的稠密模型，张量并行常常没必要——层适合一块 GPU，所以状态分片（FSDP 或 ZeRO）处理内存，而数据并行处理扩展。对于更大的稠密模型（70B+），张量并行对大矩阵乘法变得必不可少，通常节点内 TP=4 或 TP=8。当你需要比 TP 组容纳更多的 GPU 时，流水线并行增加另一个扩展维度。

MoE 模型遵循不同的模式。专家并行自然地分布专家，常常完全替代专家层的张量并行。一个 Mixtral 风格的 8x7B 模型可能用 EP=8（每块 GPU 一个专家）而无张量并行，加上用于深度的流水线并行。

当序列长度驱动内存使用时出现上下文并行。对于 8K+ token 序列，CP=2 或 CP=4 可以在没有完整模型并行复杂性的情况下将激活内存减半或减到四分之一。

## 选择正确的策略

有这么多可用的并行技术，你如何决定用哪个？答案取决于你的具体约束：模型大小、层大小、序列长度、可用硬件，以及你训练的是稠密还是 MoE 模型。

### 一个决策框架

![并行策略决策树。](img/parallelism_decision_tree_zh.png){#fig:parallelism-decision-tree .block width=80% align=center}

图~\ref{fig:parallelism-decision-tree} 提供了一个可视化指南。决策过程从最简单的问题开始：你的模型用标准数据并行适合一块 GPU 吗？如果是，用 DDP——它最简单也最高效。如果不是，下一个问题是单个层是否适合一块 GPU。如果层适合但完整模型不适合，状态分片（FSDP2 或 ZeRO-3）是你的答案。如果单个层太大，你需要计算分片（张量并行）。从那里，流水线并行、上下文并行和专家并行等额外维度解决特定瓶颈。

### 理解权衡

每种技术在内存节省和通信开销之间做不同的权衡。DDP 复制所有东西——最大的通信效率但无内存节省。ZeRO-1 只分片优化器状态，以最小的开销将内存大致减半。ZeRO-2 添加梯度分片，ZeRO-3 分片所有东西，实现随 GPU 数量近线性的内存缩放，但需要在每层计算之前的 all-gather 操作。

张量并行分片计算而非只是存储。它按 TP 度成比例地减少每 GPU 内存和计算，但需要每层内的高带宽通信（all-reduce）。这就是为什么 TP 在 NVLink 提供带宽的节点内工作最好。

流水线并行按深度而非宽度分片。它引入流水线气泡（空闲时间）但容忍更高延迟的通信，使它适合跨节点扩展。

### 具体的内存示例

为了具体化，考虑一个带 Adam 优化器的 70B 参数模型。在 FP16 下，参数占 140GB，梯度另 140GB，Adam 的优化器状态（FP32 主权重、动量、方差）占 840GB——总共超过 1TB。

用 8 块 GPU 上的 DDP，每块 GPU 存储全部 1TB+。用 ZeRO-1，优化器状态被分片：每块 GPU 存储 140GB 参数 + 140GB 梯度 + 105GB 优化器 = 385GB。用 ZeRO-3，所有东西被分片：每块 GPU 总共存储大约 140GB（每个组件的 1/8）。内存随 GPU 数量线性缩放——添加更多 GPU，每 GPU 使用更少内存。

问题是通信。ZeRO-3 必须在每层之前 all-gather 参数并在之后 reduce-scatter 梯度。对于层相对于通信延迟较小的模型，这个开销可能显著。对于每层有大量计算的大型模型，开销被摊销，ZeRO-3 工作良好。

## 小结

本章涵盖了将模型训练扩展到超过单块 GPU 所能容纳的两种互补方法。

**状态分片**（ZeRO、FSDP2）通过跨 GPU 分布参数、梯度和优化器状态来消除内存冗余。ZeRO 的渐进阶段——从仅优化器分片（阶段 1）到完全参数分片（阶段 3）——让你以通信开销换内存节省。ZeRO-Offload 和 ZeRO-Infinity 等扩展在 GPU 内存耗尽时将内存推向 CPU 和 NVMe，尽管有吞吐量惩罚。ZeRO++ 通过量化和分层分区优化多节点通信。

**计算分片**（Megatron）解决一个不同的问题：当单个层对单块 GPU 太大或太慢时。张量并行将矩阵乘法拆分到层内的 GPU 上。流水线并行跨流水线阶段分布模型深度。上下文并行和序列并行通过分片激活值处理长序列。专家并行自然地分布 MoE 专家。

关键洞见是这些方法是正交的。状态分片减少内存冗余；计算分片减少每 GPU 计算负载。现代大规模训练结合两者：节点内张量并行（NVLink 提供带宽）、跨节点流水线并行（延迟可容忍），以及用于优化器内存效率的状态分片。

决策框架是渐进的。从可工作的最简单方法开始——对 30B 参数以下的模型常常只用状态分片。当层成为瓶颈时添加张量并行。为非常深的模型或多节点扩展添加流水线并行。为长序列添加上下文并行。为 MoE 模型添加专家并行。每个维度增加复杂性，所以仅在需要时添加它们。

到目前为止，我们专注于分布式训练。但训练只是故事的一半。一旦你训练了一个模型，你需要高效地服务它。本书的下一部分转向分布式推理：如何大规模运行大型模型、处理高吞吐量工作负载，以及在生产中服务模型。



## 有用的链接

__DeepSpeed 和 ZeRO__

- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2020)：\url{https://arxiv.org/abs/1910.02054}
- ZeRO-Offload: Democratizing Billion-Scale Model Training (2021)：\url{https://arxiv.org/abs/2101.06840}
- ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning (2021)：\url{https://arxiv.org/abs/2104.07857}
- ZeRO++: Extremely Efficient Collective Communication for Giant Model Training (2023)：\url{https://arxiv.org/abs/2306.10209}
- DeepSpeed 文档：\url{https://www.deepspeed.ai/}
- DeepSpeed GitHub：\url{https://github.com/microsoft/DeepSpeed}

__Megatron-LM__

- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (2019)：\url{https://arxiv.org/abs/1909.08053}
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (2021)：\url{https://arxiv.org/abs/2104.04473}
- Reducing Activation Recomputation in Large Transformer Models (2023)：\url{https://arxiv.org/abs/2205.05198}
- Megatron-LM GitHub：\url{https://github.com/NVIDIA/Megatron-LM}
- Megatron Core 文档：\url{https://docs.nvidia.com/megatron-core/}

__集成框架__

- Colossal-AI：\url{https://colossalai.org/}
- NVIDIA NeMo：\url{https://docs.nvidia.com/nemo-framework/}

__研究__

- Arctic Long Sequence Training: Scalable Training for Multi-Million Token Sequences (2025)：\url{https://arxiv.org/abs/2507.19845}
- SuperOffload: Large-Scale LLM Training on Superchips (2025)：\url{https://arxiv.org/abs/2502.19811}
- ZenFlow: Stall-Free Offloading Engine (2025)：\url{https://arxiv.org/abs/2502.07846}
- DeepCompile: Compiler Optimization for Distributed Training (2025)：\url{https://arxiv.org/abs/2505.11432}
- Universal Checkpointing for Large-Scale Training (2024)：\url{https://arxiv.org/abs/2503.15758}
- Ring Attention with Blockwise Transformers for Near-Infinite Context (2024)：\url{https://arxiv.org/abs/2310.01889}
- DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models (2023)：\url{https://arxiv.org/abs/2309.14509}

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
