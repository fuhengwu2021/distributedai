# 第 5 章：超越状态分片：DeepSpeed 与 Megatron-LM 进阶实战 {-}

*突破显存极限与算力切分：千亿大模型 3D 混合并行架构*

> 未来已经到来，只是尚未均匀分布。  
> —— 威廉·吉布森（William Gibson，科幻作家）

**核心代码速查**

- `deepspeed.initialize()`：初始化 DeepSpeed 运行时引擎并绑定 ZeRO 配置
- `deepspeed.DeepSpeedEngine`：DeepSpeed 分布式训练引擎封装器
- `megatron.core.parallel_state`：Megatron-LM 核心并行状态管理器（TP/PP/DP 进程组）
- `megatron.core.tensor_parallel`：Megatron 张量并行矩阵乘与通信算子库
- `megatron.core.pipeline_parallel`：Megatron 流水线并行 1F1B 调度与通信接口
- `deepspeed.zero.Init()`：DeepSpeed ZeRO 阶段参数分片延迟初始化上下文管理器
- `deepspeed.zero.OffloadOptimizerConfig`：ZeRO-Offload 优化器 CPU 卸载配置
- `deepspeed.zero.OffloadParamConfig`：ZeRO-Infinity 模型参数 NVMe 卸载配置
- `megatron.model.parallel.layers.ColumnParallelLinear`：Megatron 列并行线性层
- `megatron.model.parallel.layers.RowParallelLinear`：Megatron 行并行线性层


## 超越状态分片：从存储优化走向算力切分

在上一章中，我们深入探讨了 PyTorch FSDP。FSDP2 的全分片机制在数学逻辑上等价于 DeepSpeed 的 **ZeRO Stage 3**[^zero-paper]：二者均致力于消除静态显存冗余，使每张 GPU 仅需承担 $1/N$ 的模型状态（参数、梯度与优化器状态）。

[^zero-paper]: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020). https://arxiv.org/abs/1910.02054

然而，**状态分片（State Sharding）解决的仅仅是显存存储问题，它并没有改变单卡计算的物理执行方式**。在 FSDP/ZeRO 架构下，每张 GPU 依然需要独立完成整个网络层的前向与反向矩阵乘法。当模型参数量突破 1,000 亿（100B+）乃至万亿（1T+）时，单卡算力与显存将遭遇全新的物理瓶颈：**单层权重矩阵（例如 $16384 \times 65536$ 的超大 FFN 层）过大，单张 GPU 连执行一次单层 GEMM 都会导致显存溢出或吞吐严重下降；且超长上下文下的中间激活值即便开启重计算也无法完全容纳**。

这正是 **Megatron-LM** 诞生的历史契机[^megatron-paper]。Megatron 提出的**张量并行（Tensor Parallelism）**将单个网络层内部的矩阵乘法横向切分到多张 GPU 协同计算；其**流水线并行（Pipeline Parallelism）**则沿网络深度将不同层分配到不同 GPU 执行流水交错。这些技术**切分的是计算（Computation）本身，而不仅仅是存储状态（State）**。

[^megatron-paper]: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019). https://arxiv.org/abs/1909.08053

本章将首先系统剖析 DeepSpeed ZeRO 家族全貌（ZeRO-1/2/3、ZeRO-Offload、ZeRO-Infinity 与 ZeRO++），随后重点深入 Megatron 体系，全景呈现涵盖张量并行、流水线并行、序列/上下文并行、专家并行（MoE EP）以及 3D 混合并行的终极工业级架构。

![ZeRO 阶段显存布局演进对比：DDP vs ZeRO-1/2/3](img/zero_stages_comparison.png){#fig:zero-stages .block width=100% align=center}

如 @fig:zero-stages 所示，在 2 卡（R0–R1）环境下：
- **传统 DDP**：每卡完整复制参数 P（蓝色）、梯度 G（红色）与优化器状态 O（绿色）；
- **ZeRO-1**：仅对优化器状态 O 进行切分（每卡持有一半 O）；
- **ZeRO-2**：同时切分梯度 G 与优化器状态 O；
- **ZeRO-3**：参数 P、梯度 G 与优化器状态 O 全面均等切分，显存降至最低。

---

## DeepSpeed ZeRO 阶段演进剖析

### ZeRO Stage 1：优化器状态分片（$4\times$ 显存压缩）

在采用 Adam 优化器的大模型训练中，优化器状态占据了绝大部分显存。对于 7B 参数模型，Adam 维护的一阶动量与二阶方差（均为 FP32）需要 $7\text{B} \times 8\text{ Bytes} = \mathbf{56\text{ GB}}$；若加上混合精度训练维护的 FP32 Master Weight（$7\text{B} \times 4\text{ Bytes} = \mathbf{28\text{ GB}}$），优化器侧的静态开销高达 **84 GB**。

ZeRO-1 的核心思想是：**各 GPU 仅保留自身负责更新的参数子集所对应的优化器状态**。在 2 卡环境下，每张卡仅维护 28 GB 优化器状态；在 8 卡环境下，单卡优化器开销骤降至 7 GB。

DeepSpeed 通过 JSON 或字典配置即可无缝启用 ZeRO-1：

```python
import deepspeed

ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 1}  # 启用 ZeRO-1
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### ZeRO Stage 2：优化器状态 + 梯度分片（$8\times$ 显存压缩）

在 ZeRO-1 中，反向传播计算出的梯度在每张卡上依然是全量保存的（14 GB）。

ZeRO-2 进一步将梯度也纳入分片管理：反向传播过程中，不再调用全量广播的 `all_reduce`，而是直接调用 **`reduce_scatter`**——各 GPU 仅接收并保留自身负责参数分片所对应的平均梯度，其余不属于自己的梯度在归约后立即释放。

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

### ZeRO Stage 3：全状态分片（与 FSDP 等价）

ZeRO-3 将参数、梯度与优化器状态全面切分，单卡显存开销随卡数呈 $1/N$ 完美线性下降。其代价是在每层前向计算前需触发 `all_gather` 动态重构参数，反向计算前再次 `all_gather` 重构参数，并在求导后执行 `reduce_scatter`。

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

---

## 异构存储扩展：ZeRO-Offload 与 ZeRO-Infinity

当集群 GPU 显存即便经过 ZeRO-3 全分片后依然不足时，DeepSpeed 提供了利用 CPU RAM 乃至本地 NVMe SSD 的异构分层存储方案。

### ZeRO-Offload：CPU 内存卸载

ZeRO-Offload 将前向与反向计算保留在 GPU 上执行，但在反向传播结束后将计算出的梯度通过 PCIe 传输至 CPU RAM，由 CPU 核心执行 Adam 优化器更新，再将更新后的参数写回 GPU。

```python
ds_config = {
    "train_batch_size": 32,
    "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True  # 使用锁页内存加速 PCIe 搬运
        }
    }
}
```

### ZeRO-Infinity：NVMe SSD 极大规模模型卸载

![ZeRO-Infinity 异构三级存储金字塔：GPU HBM、CPU RAM 与 NVMe SSD](img/memory_hierarchy.png){#fig:memory-hierarchy .block width=80% align=center}

如 @fig:memory-hierarchy 所示，ZeRO-Infinity 将存储体系扩展为三级：
- **GPU HBM**：仅存放当前正在计算的网络层参数与激活值（极速、最小）；
- **CPU RAM**：存放优化器状态与参数通信预取缓冲区（中速、中等）；
- **NVMe SSD**：存放全量百亿/千亿静态参数文件（海量、通过异步 I/O 流水线预取）。

通过配合 Linux 原生异步 I/O 引擎（`libaio`），系统可在后台并发将下一层的参数从 NVMe 顺序预取至 CPU 和 GPU，突破单机显存物理天花板。

---

## ZeRO++：跨节点通信极致压缩

ZeRO-3 虽然消除了显存冗余，但在多节点（Multi-Node）训练中，跨节点的 All-Gather 与 Reduce-Scatter 流量会严重挤占网络带宽。**ZeRO++** 针对性提出了三大核心优化技术[^zero-pp]：

[^zero-pp]: Wang et al., "ZeRO++: Extremely Efficient Collective Communication for Giant Model Training" (2023). https://arxiv.org/abs/2306.10209

1. **量化权重通信（qwZ, Quantized Weights）**：在前向 All-Gather 传输过程中，将参数实时量化为 **INT8** 传输，到达目标卡后再反量化还原为 FP16，使通信数据量直接砍半；
2. **分层分片（hpZ, Hierarchical Partitioning）**：机内 GPU 通过极速 NVLink 共享完整参数副本，仅在跨机节点间切分参数，大幅减少跨节点低速网络的 All-Gather 流量；
3. **量化梯度归约（qgZ, Quantized Gradients）**：在反向 Reduce-Scatter 过程中采用块级量化（Block-based INT8）传输梯度。

![hpZ 分层分片拓扑通信流对比：ZeRO-3 vs ZeRO++ hpZ](img/hpz_hierarchical.png){#fig:hpz .block width=100% align=center}

如 @fig:hpz 所示，ZeRO-3（左）每个 GPU 各持有一份独立切片，跨节点产生密集的全互联通信网；而 hpZ（右）在机内复制切片，跨机仅需在对应代表卡间进行单次轻量交换，大幅减轻了 InfiniBand/以太网负担。

---

## Megatron：算力切分作为正交的第二维度

### 状态切分（FSDP/ZeRO） vs 算力切分（Megatron）

必须建立清晰的底层认知：**FSDP/ZeRO 与 Megatron 解决的是两个完全正交维度的系统瓶颈**。

- **FSDP / ZeRO（状态切分）**：解决**“存不下”**的问题。通过动态 All-Gather/Reduce-Scatter 消除静态冗余，但单层矩阵乘法的计算依然由单卡独立完成；
- **Megatron（算力切分）**：解决**“算不动 / 算太慢”**的问题。将单层矩阵乘法或超长序列直接纵横切开，多张卡在算子内部协同执行单次 Forward/Backward。

### 张量并行（Tensor Parallelism, TP）：层内算子级切分

Megatron-LM 的核心创新在于提出了**仅需极少通信开销的 Transformer 张量切分范式**。

#### 1. 列并行线性层（ColumnParallelLinear）

对于线性变换 $Y = XW$，设权重矩阵 $W$ 的形状为 $(d_{\text{in}}, d_{\text{out}})$。我们将 $W$ **按列切分为 $N$ 份**：$W = [W_0 \mid W_1]$。
- 输入 $X$ 完整广播给所有卡；
- 各卡独立计算自身局部列点积：$Y_0 = XW_0, Y_1 = XW_1$；
- **此时各卡输出即为目标特征的横向拼接，期间无需发生任何通信！**

#### 2. 行并行线性层（RowParallelLinear）

在紧随其后的第二个线性层中，权重矩阵 $W'$ 的形状为 $(d_{\text{out}}, d_{\text{in}})$。我们将 $W'$ **按行切分为 $N$ 份**：$W' = \begin{bmatrix} W'_0 \\ W'_1 \end{bmatrix}$。
- 各卡输入为其在上一层计算出的局部特征切片 $Y_0, Y_1$；
- 各卡计算局部矩阵乘积：$Z_0 = Y_0 W'_0, Z_1 = Y_1 W'_1$；
- **最终输出通过一次 `AllReduce(SUM)` 集合通信累加各卡局部结果**：$Z = Z_0 + Z_1 = XW W'$。

![Megatron 张量并行：列并行与行并行配对（仅需一次 AllReduce 同步）](img/tensor_parallelism.png){#fig:tensor-parallel .block width=90% align=center}

如 @fig:tensor-parallel 所示，通过将“列并行（MLP-1 / QKV 投影）”与“行并行（MLP-2 / 注意力输出投影）”完美配对，**一个完整的 MLP 块或 Multi-Head Attention 块在整个前向传播中仅需执行一次 AllReduce 集合通信**！

>NOTES: **TP 通信敏感性与 NVLink 域绑定**
>
>张量并行（TP）在每个 Transformer 层内部都要高频触发 AllReduce。因此，**TP 进程组必须严格限制在同一个物理机节点内部，通过高带宽的 NVLink（900 GB/s–1.8 TB/s）互联**；严禁跨越低速跨机网络执行 TP。
>NOTEE

---

### 流水线并行（Pipeline Parallelism, PP）：深度层间切分

流水线并行将一个包含数十层（如 80 层）的深层 Transformer 沿网络深度切分成若干个 Stage（例如 4 个 Stage，每个 Stage 承载 20 层），分配给不同的 GPU 节点。

#### 1F1B（One Forward One Backward）稳态调度

传统朴素流水线（如 GPipe）会产生巨大的“流水线气泡（Pipeline Bubble）”。Megatron 引入了高效的 **1F1B 调度算法**：

![流水线并行：朴素调度 vs 1F1B 交错调度对比](img/pipeline_parallelism.png){#fig:pipeline-parallelism .block width=100% align=center}

如 @fig:pipeline-parallelism 所示：
- 将全局 Batch 拆分为大量细粒度的 **Micro-batches**；
- 在启动阶段填充管道后，系统进入稳态：**每个 GPU 严格交替执行一次前向 Micro-batch（F）与一次反向 Micro-batch（B）**；
- 大幅降低了激活值显存积压，并将流水线气泡时间压缩到极低比例。

---

### 超长序列处理：序列并行（SP）与上下文并行（CP）

当训练与推理的上下文长度暴增至 32K、128K 甚至 1M tokens 时，前向激活值显存将超过模型权重本身。

1. **序列并行（Sequence Parallelism, SP）**[^seqpar]：在开启张量并行（TP）的基础上，将 LayerNorm 与 Dropout 等未参与矩阵切分的算子沿 **Sequence 维度**切分给各卡，消除非矩阵算子的激活值冗余。
2. **上下文并行（Context Parallelism, CP / Ring Attention）**[^ringatt]：将注意力计算本身的 Query/Key/Value 张量沿 Sequence 维度等分，通过 **Ring Attention（环形注意力）** 机制，各卡在环形拓扑中循环传递 KV 切片，使每张卡在不持有全量序列的前提下计算出完整的自注意力。

[^seqpar]: Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," MLSys 2023. https://arxiv.org/abs/2205.05198
[^ringatt]: Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context," ICLR 2024. https://arxiv.org/abs/2310.01889

![序列并行（SP）与上下文并行（Ring Attention）架构示意图](img/sequence_context_parallelism.png){#fig:seq-ctx-parallel .block width=100% align=center}

如 @fig:seq-ctx-parallel 所示，序列并行（左）按 Token 序列切分 LayerNorm；上下文并行（右）通过环形队列循环传递 K/V 块完成全局注意力计算。

#### DeepSpeed-Ulysses：基于 All-to-All 维度转置的替代方案

除 Ring Attention 外，微软提出的 **DeepSpeed-Ulysses** 提供了另一种优雅的序列并行思路[^ulysses]：在 Attention 计算前，通过一次 **All-to-All** 集合通信将张量从“序列切分、全量注意力头”瞬间转置为“全量序列、注意力头切分”；各卡独立计算各自持有的注意力头后，再通过一次反向 All-to-All 转置复原。

[^ulysses]: Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models," arXiv 2023. https://arxiv.org/abs/2309.14509

![DeepSpeed-Ulysses：通过 All-to-All 实现 2D 维度转置](img/ulysses.png){#fig:ulysses .block width=85% align=center}

---

### 专家并行（Expert Parallelism, EP）：MoE 稀疏扩展

针对 Mixtral 8x7B、DeepSeek-V3 等混合专家模型（MoE），**专家并行（EP）** 将稀疏门控网络中的各 Expert 实体分散部署在集群的不同 GPU 上。

![专家并行（Expert Parallelism）：基于门控路由与 All-to-All 的 Token 动态分发](img/expert_parallelism.png){#fig:expert-parallelism .block width=90% align=center}

如 @fig:expert-parallelism 所示：
1. Router 门控网络计算各 Token 的专家匹配概率；
2. 通过 **All-to-All Dispatch** 集合通信，将 Token 打包发送至其对应 Expert 所在的 GPU；
3. 各 GPU 上的 Expert 独立计算输出；
4. 通过 **All-to-All Combine** 集合通信将计算结果路由回原始 GPU 聚合。

---

## 3D/4D 混合并行架构实战（Hybrid Parallelism）

在训练数百亿至万亿参数规模的大模型时，工业界标准方案是将上述正交策略融会贯通，构建 **3D 混合并行（DP × TP × PP）** 甚至 **4D/5D 并行（DP + FSDP + TP + PP + CP/EP）**。

### 64 卡 70B 模型标准 3D 并行拓扑排布

以在 8 台服务器（每台 8 张 H100，共 64 卡）上训练 LLaMA-3 70B 为例：

1. **单机内部（Intra-Node）**：配置 **TP=4**（每 4 张 GPU 通过机内满血 NVLink 组建张量并行，切分超大 GEMM 矩阵）；
2. **跨机深度（Inter-Node）**：配置 **PP=2**（将 80 层 Transformer 切分为 2 个流水线阶段，通过 InfiniBand 传递 Stage 激活值）；
3. **数据并行与状态切分**：此时单个模型副本占用 $4 \times 2 = 8$ 张 GPU。集群总共拥有 $64 / 8 = \mathbf{8}$ 个独立模型副本，即 **DP=8**。在 8 个 DP 副本之间开启 **Megatron-FSDP / 分布式优化器**，全面切分优化器状态。
4. **总算力校验**：
   $$\text{总 GPU 卡数} = \text{TP} \times \text{PP} \times \text{DP} = 4 \times 2 \times 8 = \mathbf{64\text{ GPUs}}$$

---

## 并行策略决策树与全景选型指南

![大规模分布式并行策略选型决策树](img/parallelism_decision_tree.png){#fig:parallelism-decision-tree .block width=80% align=center}

对照 @fig:parallelism-decision-tree，选型核心原则如下：

1. **模型是否能放入单卡显存？**
   - **能** $\rightarrow$ 首选 **PyTorch DDP**（最高吞吐，零冗余通信）；
   - **不能** $\rightarrow$ 进入下一步。
2. **单层算子矩阵与激活值是否能放入单卡？**
   - **能** $\rightarrow$ 首选 **PyTorch FSDP2** 或 **DeepSpeed ZeRO-3**（无需改动模型代码，纯状态切分）；
   - **不能（单层超大，或追求极致吞吐）** $\rightarrow$ 引入 **Megatron 张量并行 (TP)**。
3. **模型层数极深且跨越大量多机节点？**
   - 引入 **流水线并行 (PP)** 降低跨机通信敏感度；
4. **序列长度突破 8K–128K？**
   - 引入 **上下文并行 (CP / Ring Attention)** 或 **DeepSpeed-Ulysses**；
5. **稀疏 MoE 架构？**
   - 引入 **专家并行 (EP)** 配合 All-to-All 路由。

---

## 本章小结

本章横跨了从“显存状态分片”到“算子级算力切分”的技术鸿沟：
- 剖析了 DeepSpeed ZeRO 1/2/3 演进、ZeRO-Offload/Infinity 异构存储扩展以及 ZeRO++ 通信压缩；
- 深入推导了 Megatron-LM 的核心算子切分机制：列并行与行并行配对、1F1B 流水线交错调度、Ring Attention 与 DeepSpeed-Ulysses 序列并行、MoE 专家并行；
- 阐明了 FSDP（状态切分）与 Megatron（算力切分）的正交协同关系，并完成了 3D/4D 混合并行体系的构建。

至此，全书的**分布式训练核心技术栈**已全部构建完毕。然而，训练只是大模型生命周期的前半程——训练好的千亿参数模型如何以毫秒级延迟、极高吞吐在生产集群中对外提供在线推理服务？在接下来的下半卷中，我们将正式开启**分布式推理与服务化中枢**，深入探索 **vLLM、PagedAttention、SGLang 与生产级推理集群编排**的全新天地。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
