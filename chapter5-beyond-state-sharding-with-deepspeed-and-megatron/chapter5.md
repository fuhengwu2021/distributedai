# Chapter 5: Beyond State Sharding with DeepSpeed and Megatron {-}

*Extending memory capacity and sharding computation for very large models*

> The future is already here. It's just unevenly distributed.
- William Gibson

**Code Summary**

- `deepspeed.initialize()`: Initialize DeepSpeed engine with ZeRO configuration
- `deepspeed.DeepSpeedEngine`: DeepSpeed engine wrapper for model training
- `megatron.core.parallel_state`: Megatron parallel state management
- `megatron.core.tensor_parallel`: Megatron tensor parallelism utilities
- `megatron.core.pipeline_parallel`: Megatron pipeline parallelism utilities
- `deepspeed.zero.Init()`: DeepSpeed ZeRO initialization context manager
- `deepspeed.zero.OffloadOptimizerConfig`: Configuration for ZeRO-Offload
- `deepspeed.zero.OffloadParamConfig`: Configuration for ZeRO-Infinity parameter offload
- `megatron.model.parallel.layers.ColumnParallelLinear`: Column-parallel linear layer for tensor parallelism
- `megatron.model.parallel.layers.RowParallelLinear`: Row-parallel linear layer for tensor parallelism

## Beyond State Sharding

In the previous chapter, we explored FSDP—PyTorch's approach to sharding parameters, gradients, and optimizer states across GPUs. FSDP2's full sharding is functionally equivalent to DeepSpeed's ZeRO Stage 3[^zero-paper]: both eliminate memory redundancy by ensuring each GPU holds only 1/N of the training state.

[^zero-paper]: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020). https://arxiv.org/abs/1910.02054

State sharding solves the memory problem, but it doesn't change *how* computation happens. Every GPU still executes the same operations on the same model architecture—just with different data batches. For the largest models (100B+ parameters), this becomes limiting: individual layers may be too large for efficient single-GPU execution, or the model may be too deep to fit activations in memory even with checkpointing.

This is where **Megatron** comes in[^megatron-paper]. Megatron's tensor parallelism splits large matrix operations across GPUs, and its pipeline parallelism shards the model along the depth dimension. These techniques shard *computation itself*, not just training state. They're essential for training frontier models and remain the backbone of large-scale training infrastructure at NVIDIA, Meta, and elsewhere.

[^megatron-paper]: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019). https://arxiv.org/abs/1909.08053

We'll cover DeepSpeed ZeRO first—it's worth understanding the full ZeRO family (stages 1-3, offloading, ZeRO++) since many codebases still use it. But the real focus of this chapter is Megatron-style parallelism: tensor parallelism, pipeline parallelism, and how they combine with state sharding for multi-dimensional parallelism.

![ZeRO stages comparison: DDP vs ZeRO-1/2/3.](img/zero_stages_comparison.png){#fig:zero-stages .block width=100% align=center}

Figure~\ref{fig:zero-stages} illustrates the memory layout across two ranks (R0–R1) for DDP and each ZeRO stage. Each row represents one GPU, and the three colored blocks show what that GPU stores: P (parameters, blue), G (gradients, red), and O (optimizer states, green). In DDP, all blocks are full-width because every GPU holds complete copies of everything—this is the memory redundancy we want to eliminate. ZeRO-1 keeps parameters and gradients replicated but shards optimizer states (notice the smaller O blocks). ZeRO-2 additionally shards gradients, so both G and O blocks shrink. ZeRO-3 shards all three components—every block becomes half the original size with 2 GPUs. The visual progression from left to right shows how memory per GPU decreases at each stage, with the trade-off being increased communication to reconstruct full tensors when needed.

## ZeRO Stage 1: Optimizer State Partitioning

Recall the memory breakdown from the previous chapter: for models using Adam, optimizer states dominate memory usage—each parameter requires storing momentum and variance as two FP32 copies, totaling 8 bytes per parameter. A 7B parameter model needs 56GB just for optimizer states. In DDP, every GPU holds a complete copy of these states, which is a massive waste.

ZeRO-1's insight is straightforward: since each GPU ultimately updates only its assigned portion of parameters, why store the full optimizer states? With 2 GPUs, each stores only half of the optimizer states. Forward and backward passes proceed normally, and after gradient synchronization, each GPU uses only its local optimizer states to update the corresponding parameter shard. The 56GB of optimizer states gets distributed across 2 GPUs, reducing each GPU's burden to 28GB.

Unlike PyTorch's native DDP which requires minimal setup, DeepSpeed uses a configuration dictionary (or JSON file) to control all training settings—optimizer, precision, ZeRO stage, and more. You pass this config to `deepspeed.initialize()`, which returns a wrapped model engine that handles distributed training automatically:

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

The key difference from DDP: DeepSpeed manages the optimizer internally based on your config, so you don't create it yourself. The `model_engine` wraps your model and provides `backward()` and `step()` methods.

ZeRO-1 fits scenarios where the model itself fits in GPU memory, but adding optimizer states pushes it over the limit. It requires minimal changes to the training loop and is the easiest to debug, making it a natural first step when migrating from DDP to ZeRO.

To experience the DeepSpeed API, run this minimal example:

```bash
pip install deepspeed
# Single GPU (for API familiarization)
deepspeed --num_gpus=1 code/zero_minimal.py --zero_stage 1
# Multiple GPUs (to see actual sharding benefits)
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 1
```

You should see output like:

```
ZeRO Stage 1 Demo
Model: 5,248,000 parameters
...
Step 1/10, Loss: 1.0342, Peak Memory: 1.06 GB
...
ZeRO Stage 1 training complete!
Final peak memory: 1.10 GB
```

With a single GPU, sharding has limited effect (there's only one partition). With 2 GPUs, each GPU stores only half of the optimizer states, and you'll observe lower per-GPU memory usage.

## ZeRO Stage 2: Optimizer State + Gradient Partitioning

ZeRO-1 shards optimizer states, but gradients remain fully replicated. For a 7B model, that's still 14GB of gradients (FP16) sitting on every GPU. ZeRO-2 takes the next step: shard gradients too.

The key insight is that gradients, like optimizer states, are only needed for the parameters each GPU is responsible for updating. During the backward pass, instead of using `all_reduce` (which gives every GPU the full averaged gradient), ZeRO-2 uses `reduce_scatter`—each GPU receives only its assigned slice of the averaged gradient. The rest is discarded immediately, freeing memory as computation proceeds.

With 2 GPUs, each now holds: full parameters + half the gradients + half the optimizer states. For our 7B model, gradient memory drops from 14GB to 7GB per GPU.

The configuration adds bucket size parameters to control how gradients are batched before communication:

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

Larger buckets improve communication efficiency by amortizing the overhead of each collective operation, but use more memory. The default of 500M elements works well for most cases.

To compare ZeRO-1 and ZeRO-2:

```bash
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 1
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 2
```

ZeRO-2 is a good choice when gradient memory becomes a bottleneck but you still want parameters replicated for fast forward passes.

## ZeRO Stage 3: Full Sharding (Like FSDP)

ZeRO-2 still keeps parameters replicated on every GPU. For a 7B model, that's 14GB (FP16) of parameters duplicated across all ranks. ZeRO-3 eliminates this last redundancy by sharding parameters too—now every component (parameters, gradients, optimizer states) is distributed.

This is functionally equivalent to PyTorch FSDP. Each GPU holds only 1/N of everything. For a 7B model on 2 GPUs: 7GB parameters + 7GB gradients + 28GB optimizer states = 42GB per GPU, compared to 14GB + 14GB + 56GB = 84GB with DDP.

The trade-off is communication. Since parameters are now sharded, each layer needs an `all_gather` before forward computation to reconstruct the full weights, then the gathered parameters are freed immediately after use. The backward pass does the same, plus a `reduce_scatter` for gradients. This means 3× the model size in communication per iteration (1× forward all-gather, 1× backward all-gather, 1× gradient reduce-scatter).

DeepSpeed mitigates this overhead by overlapping communication with computation—while one layer computes, the next layer's parameters are being gathered in the background. The `overlap_comm` and `stage3_prefetch_bucket_size` parameters control this behavior:

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

To see the full progression from ZeRO-1 to ZeRO-3:

```bash
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 1
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 2
deepspeed --num_gpus=2 code/zero_minimal.py --zero_stage 3
```

ZeRO-3 is the right choice when even replicated parameters don't fit in GPU memory, or when you want maximum memory efficiency and can tolerate some communication overhead. With fast interconnects like NVLink, the performance gap versus ZeRO-2 is often modest (10-20%).

## ZeRO-Offload: CPU Memory Extension

Even with ZeRO-3, you might run out of GPU memory—especially on consumer hardware like an RTX 4090 with 24GB VRAM. ZeRO-Offload addresses this by moving optimizer states (and optionally parameters) to CPU memory, which is typically much larger and cheaper.

The idea is simple: keep forward and backward passes on the GPU where they're fast, but offload the memory-hungry optimizer states to CPU RAM. After gradients are computed, they're transferred to CPU via PCIe, the optimizer step runs on CPU, and updated parameters are sent back to GPU. DeepSpeed overlaps these transfers with computation—while the GPU processes the next batch's forward pass, the CPU is simultaneously running the optimizer step from the previous batch.

The bottleneck is PCIe bandwidth (~32 GB/s for PCIe 4.0, versus ~2 TB/s for GPU HBM). Expect 20-40% throughput reduction compared to GPU-only training. This isn't a performance optimization—it's a feasibility solution that enables training models that wouldn't otherwise fit.

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

The `pin_memory: True` setting uses pinned (page-locked) memory for faster CPU-GPU transfers. To try CPU offloading:

```bash
deepspeed --num_gpus=1 code/zero_offload_example.py --offload_device cpu
```

You should see output like:

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

Notice how GPU memory stays low (3.5GB) despite the model size—optimizer states live on CPU. ZeRO-Offload is ideal for training on consumer GPUs with limited VRAM but plenty of system RAM.

## ZeRO-Infinity: NVMe Offload for Massive Models

ZeRO-Offload moves optimizer states to CPU, but CPU RAM has limits too—typically 256-512GB on a workstation. For truly massive models (hundreds of billions of parameters), even CPU memory isn't enough. ZeRO-Infinity takes offloading one step further by using NVMe SSDs as an additional memory tier.

![ZeRO-Infinity memory hierarchy.](img/memory_hierarchy.png){#fig:memory-hierarchy .block width=80% align=center}

Figure~\ref{fig:memory-hierarchy} shows the three-tier hierarchy: GPU HBM (fastest, smallest), CPU RAM (medium), and NVMe (slowest, largest). The Infinity Engine manages data movement across tiers, prefetching parameters from NVMe → CPU → GPU before they're needed and overlapping transfers with computation.

Modern NVMe SSDs offer 5-7 GB/s sequential read speeds (PCIe Gen4), which is slower than CPU memory bandwidth but provides terabytes of capacity at low cost. A typical setup might keep active layer parameters and activations on GPU, optimizer states and parameter buffers on CPU, and cold parameters on NVMe.

The configuration adds NVMe-specific settings:

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

The `aio` section configures async I/O for NVMe—`queue_depth` and `thread_count` control parallelism for overlapping reads/writes with computation.

To check if you have an NVMe SSD and find where it's mounted:

```bash
# List NVMe devices and partitions with mount points
lsblk -o NAME,SIZE,MOUNTPOINT | grep nvme
# Example output:
# nvme0n1       1.9T
# └─nvme0n1p7   1.8T /home
```

The `nvme_path` must be a **directory on a mounted filesystem**, not the raw device path (like `/dev/nvme0n1p7`). NVMe offloading also requires the `libaio` library for async I/O:

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

Replace `/home/$USER/nvme_offload` with a directory on your NVMe filesystem. The example trains a 354M parameter model by default—adjust `--hidden_size` and `--num_layers` to experiment with larger models. Make sure your chosen path has enough free space (roughly 2-4× the model size for optimizer states and parameter buffers).

NVMe offloading is slower than CPU offloading (expect 30-50% throughput reduction), which is itself slower than GPU-only training. The value of ZeRO-Infinity isn't performance—it's feasibility. It lets you train models that simply wouldn't fit otherwise.

## ZeRO++: Communication-Optimized ZeRO

ZeRO-3 eliminates memory redundancy, but it introduces significant communication overhead[^zero-pp].

[^zero-pp]: Wang et al., "ZeRO++: Extremely Efficient Collective Communication for Giant Model Training" (2023). https://arxiv.org/abs/2306.10209 Every forward pass requires an all-gather to reconstruct parameters; every backward pass does the same plus a reduce-scatter for gradients. For large models on multi-node clusters, this communication can dominate training time.

ZeRO++ addresses this with three complementary techniques (the names follow the paper's notation: "q" for quantized, "hp" for hierarchical partitioning, and "Z" for ZeRO). The first, **quantized weights (qwZ)**, reduces all-gather traffic by transmitting parameters in INT8 instead of FP16, then dequantizing after receipt—a 2× reduction in communication volume with minimal accuracy impact since quantization errors don't accumulate across iterations.

The second technique, **hierarchical partitioning (hpZ)**, exploits the fact that intra-node communication (NVLink, ~600 GB/s) is much faster than inter-node (InfiniBand, ~400 GB/s). Instead of sharding uniformly across all GPUs, hpZ replicates parameters within each node and shards only across nodes. This means intra-node all-gathers use fast NVLink, while inter-node traffic is reduced to one representative per node.

![hpZ hierarchical partitioning: ZeRO-3 vs hpZ.](img/hpz_hierarchical.png){#fig:hpz .block width=100% align=center}

Figure~\ref{fig:hpz} contrasts ZeRO-3 and hpZ for a setup with 2 nodes, 2 GPUs per node (4 GPUs total). In ZeRO-3 (left), each GPU holds a unique shard (S0–S3), so reconstructing full parameters requires all-gather across all 4 GPUs. The red arrows show that every GPU must communicate with every other GPU across the node boundary—S0 and S1 in Node 0 each need to fetch S2 and S3 from Node 1, and vice versa. This cross-node traffic uses the slower InfiniBand interconnect.

In hpZ (right), both GPUs within Node 0 hold the same shard (S0), and both GPUs within Node 1 hold shard S1. The single green arrow represents the simplified communication pattern: only one exchange between nodes is needed to share S0 and S1. Within each node, GPUs already have identical data, so no intra-node communication is required for the replicated portion. This dramatically reduces the amount of slow inter-node traffic.

The third technique, **quantized gradients (qgZ)**, applies the same INT8 quantization to gradients during reduce-scatter.

The configuration enables these optimizations selectively:

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

The `zero_hpz_partition_size` should match the number of GPUs per node in your cluster (2 in our example). To experiment with these optimizations:

```bash
deepspeed --num_gpus=2 code/zero_pp_example.py --enable_qwz
deepspeed --num_gpus=2 code/zero_pp_example.py --enable_hpz
deepspeed --num_gpus=2 code/zero_pp_example.py --enable_qwz --enable_hpz --enable_qgz
```

ZeRO++ is most valuable for large-scale multi-node training where inter-node communication is the bottleneck. For single-node training or small clusters, the benefits are modest since intra-node communication is already fast.


## Megatron: Computation Parallelism as the Second Axis

So far, we have focused on **state sharding**—how to distribute parameters, gradients, and optimizer states across GPUs to reduce memory footprint. Techniques such as FSDP2 and DeepSpeed ZeRO fundamentally address a *memory redundancy* problem: eliminating replicated model state so that larger models can fit within the aggregate GPU memory budget.

However, state sharding alone is not sufficient for the largest models. As model sizes continue to grow, a second, orthogonal limitation emerges: **computation itself becomes too large to execute efficiently on a single GPU**, even when memory is fully sharded. This is where Megatron enters the picture.

Megatron-LM emerged from NVIDIA's Applied Deep Learning Research team in 2019, introduced in the paper "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism."[^megatron] At the time, GPT-2 with 1.5B parameters was considered large, and the research community was just beginning to explore how to scale beyond what a single GPU could handle. The NVIDIA team recognized that simply adding more GPUs for data parallelism wouldn't solve the fundamental problem: some layers were simply too large to compute on one device. Their solution was to split individual matrix operations across GPUs—what they called *tensor parallelism*. The original Megatron paper demonstrated training of an 8.3B parameter model, unprecedented at the time. Since then, Megatron's techniques have become foundational infrastructure for training models like GPT-3 (175B), Llama (up to 405B), and virtually every frontier model today.

[^megatron]: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," arXiv:1909.08053, 2019. https://arxiv.org/abs/1909.08053

### State Sharding vs. Computation Sharding

To understand what Megatron brings to the table, it helps to step back and think about what FSDP and ZeRO actually do. These techniques solve a *storage* problem: they distribute the model's parameters, gradients, and optimizer states across GPUs so that no single device needs to hold everything. When you need to compute a layer, you all-gather its parameters from the other GPUs, perform the forward and backward passes, and then discard the gathered parameters. The key assumption here is that once you have the parameters in hand, the computation itself fits comfortably on one GPU.

This assumption holds remarkably well for models up to tens of billions of parameters. A 7B model's largest layer might have a weight matrix of shape 4096×16384—about 270 million parameters, or roughly 500MB in FP16. That's well within what a modern GPU can handle in a single matrix multiplication.

But what happens when models grow to hundreds of billions of parameters? Consider a hypothetical 175B model where the hidden dimension is 12,288 and the MLP intermediate dimension is 49,152. A single weight matrix in the MLP could be 12,288×49,152—over 600 million parameters, requiring 1.2GB just for that one matrix. The activations scale similarly. At some point, even with perfect state sharding, the computation of a single layer becomes too large: the matrix multiplication itself exceeds what one GPU can efficiently execute, or the intermediate activations overflow GPU memory.

This is where state sharding reaches its limit. You can shard the storage as finely as you want, but if the computation itself doesn't fit, no amount of clever memory management will help. Megatron addresses precisely this failure mode by taking the next logical step: instead of just sharding where the model *lives*, it shards how the model *computes*.

### Tensor Parallelism: Sharding the Layer

Megatron's core contribution is **Tensor Parallelism (TP)**—a technique that splits individual matrix operations across multiple GPUs. The key insight is elegant: matrix multiplication is inherently parallelizable along certain dimensions, and we can exploit this to distribute both the computation and the memory footprint.

Consider a simple linear layer $Y = XW$ where $X$ is the input and $W$ is a weight matrix. If $W$ has shape $d \times 4d$ (typical for an MLP's first projection), we can split it column-wise into two halves: $W = [W_0 | W_1]$. Now GPU 0 holds $W_0$ and GPU 1 holds $W_1$. Given the same input $X$ on both GPUs, each computes its portion: $Y_0 = XW_0$ and $Y_1 = XW_1$. The full output is simply $Y = [Y_0 | Y_1]$—no communication needed, just a logical concatenation. This is called **column-parallel linear**.

But what about the next layer? It expects a full input, not a split one. Here's where **row-parallel linear** comes in. If the weight matrix of the second layer is split row-wise as $W' = [W'_0; W'_1]$ (stacked vertically), then each GPU can compute a partial result using its local portion of the input: GPU 0 computes $Y'_0 = Y_0 W'_0$ and GPU 1 computes $Y'_1 = Y_1 W'_1$. The final output is $Y' = Y'_0 + Y'_1$—an all-reduce operation that sums the partial results.

![Tensor parallelism: column-parallel and row-parallel linear.](img/tensor_parallelism.png){#fig:tensor-parallel .block width=90% align=center}

Figure~\ref{fig:tensor-parallel} illustrates this two-step pattern. The column-parallel linear (top) splits the weight matrix by columns, so each GPU computes a slice of the output with no communication. The row-parallel linear (bottom) splits by rows, and an all-reduce combines the partial results. By pairing these two operations—column-parallel followed by row-parallel—a complete MLP block requires only one all-reduce. This is the key to Megatron's efficiency: communication is minimized to a single synchronization point per layer, rather than at every operation.

The same principle applies to self-attention. The Q, K, V projection matrices are split column-wise across GPUs, so each GPU computes attention for a subset of attention heads. Since attention heads are independent, no communication is needed during the attention computation itself. Only the output projection uses row-parallel linear, requiring one all-reduce to combine results.

There's an important subtlety here: tensor parallelism fundamentally changes the communication pattern compared to state sharding. With FSDP or ZeRO, communication happens *between* layers—you all-gather parameters before computing a layer, then move on. With tensor parallelism, communication happens *within* layers—every MLP and attention block requires an all-reduce. This means tensor parallelism is much more sensitive to interconnect bandwidth. In practice, you want TP groups to be within the same node, connected by fast NVLink (~600 GB/s), rather than across nodes over slower InfiniBand (~400 GB/s).

Modern implementations overlap this communication with computation using techniques like `--tp-comm-overlap` in Megatron. While one layer's all-reduce is in flight, the next layer's computation can begin, hiding much of the latency.

When tensor parallelism is enabled, **sequence parallelism** becomes a natural extension. Instead of replicating activations across all TP ranks, sequence parallelism splits activations along the sequence dimension. This reduces activation memory by a factor equal to the TP degree—essential for training with long contexts where activation memory can dominate.

A typical configuration looks like:

```bash
--tensor-model-parallel-size 2    # 2-way tensor parallelism
--sequence-parallel               # Enable sequence parallelism (recommended with TP)
--tp-comm-overlap                 # Overlap TP communication with computation
```

To see tensor parallelism in action at a lower level, you can run the pure PyTorch implementation in the code examples:

```bash
# Tensor parallelism demo with 2 GPUs
torchrun --nproc_per_node=2 code/tensor_parallel_mlp.py
```

This example implements column-parallel and row-parallel linear layers from scratch, showing exactly how weight matrices are split and how the all-reduce combines partial results. Running it helps build intuition for what Megatron does under the hood.

### Pipeline Parallelism: Sharding the Depth

In addition to tensor parallelism, Megatron provides **Pipeline Parallelism (PP)**, which shards the model along the layer (depth) dimension.

Pipeline parallelism:

* Assigns contiguous blocks of layers to different GPUs or nodes
* Executes micro-batches in a pipeline fashion to keep all stages busy
* Reduces per-device memory footprint by limiting the number of active layers

**Pipeline Schedules:**

Megatron supports multiple pipeline schedules:

1. **1F1B (One Forward One Backward)**: Standard pipeline schedule
2. **Interleaved Pipeline**: Virtual pipeline parallelism that interleaves micro-batches across stages to reduce pipeline bubbles
3. **Gpipe**: Original pipeline parallelism with forward-only then backward-only phases

**Virtual Pipeline Parallelism (VPP):**

Virtual pipeline parallelism reduces pipeline bubbles by splitting each pipeline stage into multiple virtual stages:

* Each physical GPU runs multiple virtual stages
* Micro-batches are interleaved across virtual stages
* Reduces idle time and improves GPU utilization
* Particularly effective when `PP_size >= 2`

**Configuration:**

```bash
--pipeline-model-parallel-size 8
--num-layers-per-virtual-pipeline-stage 4  # VPP configuration
```

**When to Use Pipeline Parallelism:**

* The model depth is very large (many layers)
* Inter-node scaling is required
* Tensor parallelism alone does not provide sufficient scalability
* You need to scale across multiple nodes with slower inter-node interconnects

**Best Practices:**

* Keep TP and EP within NVLink domain (intra-node)
* Use PP for inter-node scaling
* Enable virtual pipeline parallelism when PP >= 2
* Tune micro-batch count to maintain pipeline utilization

In practice, pipeline parallelism is almost always combined with tensor parallelism, forming a **2D parallelism scheme**.

To see pipeline parallelism in action with a simplified implementation:

```bash
# Pipeline parallelism demo with 2 stages
torchrun --nproc_per_node=2 code/pipeline_parallel_simple.py
```

This example demonstrates model partitioning, micro-batch scheduling, and forward/backward coordination across pipeline stages.

### Sequence Parallelism and Long Contexts

Megatron also introduces **sequence parallelism**, which addresses another emerging bottleneck: extremely long sequence lengths.

Instead of replicating activations across GPUs, sequence parallelism:

* Splits activations along the sequence dimension
* Reduces activation memory and communication overhead
* Improves scalability for long-context training

This is increasingly important for models trained with long context windows, where activation memory can dominate total memory usage.

### Context Parallelism: Advanced Long-Context Training

**Context Parallelism (CP)** is Megatron's advanced solution for extremely long sequences. Unlike sequence parallelism which only splits Dropout and LayerNorm activations, CP partitions all network inputs and activations along the sequence dimension.

**How Context Parallelism Works:**

* Each GPU processes only a chunk of the sequence (e.g., 8K sequence split across 2 GPUs = 4K tokens per GPU)
* For attention computation, each token's Q (query) needs to compute with KV (key and value) of all tokens
* CP uses all-gather across GPUs to collect full KV sequences, then reduce-scatter for gradients
* Communication is optimized using point-to-point ring topology under the hood
* Leverages MQA/GQA (Multi-Query/Grouped-Query Attention) to reduce communication volume

**Benefits:**

* **Eliminates OOM**: Activation memory per GPU is reduced by CP times
* **No recompute overhead**: Avoids the ~30% overhead of full activation recomputation
* **Better than TP scaling**: Unlike increasing TP which can make compute too short to overlap communication, CP reduces both computation and communication proportionally
* **Optimal performance**: TP+CP combinations achieve optimal performance by eliminating recompute overheads

**When to Use Context Parallelism:**

* Sequence length >= 8K tokens
* Activation memory dominates total memory usage
* Training with very long context windows (32K, 128K+)
* When full recompute causes significant overhead

**Example Configuration:**

```bash
# Enable context parallelism with TP
--tensor-model-parallel-size 4
--context-parallel-size 2        # Split 8K sequence across 2 GPUs
--sequence-parallel              # Also enable sequence parallelism
```

### Expert Parallelism: Scaling MoE Models

**Expert Parallelism (EP)** is Megatron's specialized parallelism for Mixture-of-Experts (MoE) models. In MoE architectures, different experts handle different tokens, making expert parallelism a natural fit.

**How Expert Parallelism Works:**

* Experts are partitioned across multiple GPUs
* Each GPU processes one or more experts for each MoE layer
* Tokens are routed to appropriate experts via all-to-all communication
* Combines seamlessly with TP, PP, CP, and DP

**Key Features:**

* **Token Routing**: Efficient all-to-all communication to dispatch tokens to experts
* **Load Balancing**: Multiple strategies (auxiliary loss, Sinkhorn, aux-loss-free)
* **GroupedGEMM**: Optimized computation when multiple experts per GPU
* **DeepEP/HybridEP**: High-performance token dispatching backends for large-scale training

**MoE Training Configuration Example:**

```bash
# Mixtral 8x7B training with expert parallelism
--num-experts 8
--expert-model-parallel-size 8   # 8-way expert parallelism
--moe-router-topk 2              # Top-2 routing
--moe-router-load-balancing-type aux_loss
--moe-grouped-gemm               # Optimize expert computation
--moe-permute-fusion             # Fuse token rearrangement
--tensor-model-parallel-size 1   # No TP for MoE layer
--pipeline-model-parallel-size 4 # 4 pipeline stages
--sequence-parallel               # Required when EP + TP
```

**Performance Highlights:**

* Megatron-Core MoE achieves **468 TFLOPS** for Mixtral 8X7B bf16 training
* Supports state-of-the-art MoE architectures: DeepSeek-V3, Qwen-MoE, Mixtral
* Distributed checkpointing with full resharding support across TP/CP/EP/PP

### Why FSDP2 Cannot Replace Megatron

It is tempting to view Megatron as an alternative to FSDP2 or ZeRO. This is incorrect. They operate on **different axes**.

FSDP2:

* Shards *state*
* Assumes per-layer computation fits on one GPU
* Uses all-gather and reduce-scatter around layer boundaries

Megatron:

* Shards *computation*
* Assumes per-layer computation must be distributed
* Introduces communication inside each layer

Once a single Transformer layer becomes too large or too slow for a single GPU, **Megatron-style computation sharding becomes necessary**, regardless of how aggressively the model state is sharded.

### Hybrid Parallelism: Combining State and Computation Sharding

A common pattern for training very large models is **hybrid parallelism**, which combines state sharding with computation sharding:

* **FSDP2 or ZeRO-3** handles state sharding across all GPUs
* **Megatron tensor parallelism** handles large per-layer computation
* **Pipeline parallelism** enables scaling across nodes
* Optional **sequence parallelism** reduces activation pressure

This combination allows:

* Memory-efficient storage of model state
* Efficient execution of massive matrix operations
* Scaling to hundreds or thousands of GPUs

Both FSDP2 + Megatron and ZeRO-3 + Megatron are viable approaches. FSDP2 offers tighter PyTorch integration and compiler support, while ZeRO-3 provides additional features like CPU/NVMe offloading and is well-integrated with the DeepSpeed ecosystem.

### Megatron Core: Production-Ready Library

**Megatron Core** is the production-ready library extracted from Megatron-LM, providing GPU-optimized building blocks for custom training frameworks. It offers:

**Key Components:**

* **Composable Transformer Blocks**: Attention mechanisms, MLP layers, embeddings
* **Advanced Parallelism**: TP, PP, CP, EP with seamless composition
* **Memory Management**: Activation recomputation, distributed checkpointing
* **FP8 Precision**: Optimized for NVIDIA Hopper, Ada, and Blackwell GPUs
* **Distributed Optimizer**: Shards optimizer states across data-parallel ranks
* **High-Performance Data Loaders**: Optimized dataset utilities

**Installation:**

```bash
# Install Megatron Core
pip install --no-build-isolation megatron-core[mlm,dev]

# Or use Docker (recommended)
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.04-py3
```

For a complete Megatron-LM pretraining example, see `code/megatron_gpt_pretrain.sh`. This script demonstrates a production-ready configuration with tensor parallelism, distributed optimizer, and flash attention.

### Megatron-FSDP: Optimized State Sharding

**Megatron-FSDP** is NVIDIA's high-performance implementation of Fully Sharded Data Parallelism, providing **15-25% speedup and 23% memory savings** compared to PyTorch FSDP2.

**Key Advantages:**

* **Better Performance**: Optimized bucketing, buffer management, and communication overlap
* **SM Usage Reduction**: Uses NCCL userbuffer to reduce Streaming Multiprocessor consumption
* **FP8 Support**: Native FP8 mixed precision with Transformer Engine
* **Compatibility**: Works with TP, CP, EP, and native PyTorch DTensor

**Usage:**

```bash
# Enable Megatron-FSDP
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params  # ZeRO-3 equivalent
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

**When to Use Megatron-FSDP vs PyTorch FSDP2:**

* **Use Megatron-FSDP** when: You need maximum performance, are using Megatron TP/CP/EP, or require FP8 training
* **Use PyTorch FSDP2** when: You want pure PyTorch without external dependencies, or need torch.compile support

### Distributed Optimizer: Memory-Efficient Optimization

Megatron's **distributed optimizer** shards optimizer states across data-parallel ranks, similar to ZeRO-1 but with additional optimizations.

**Memory Savings:**

| Configuration | Non-distributed | Distributed |
|--------------|-----------------|-------------|
| fp16 params, fp16 grads | 20 bytes/param | 4 + 16/d bytes/param |
| bf16 params, fp32 grads | 18 bytes/param | 6 + 12/d bytes/param |
| fp32 params, fp32 grads | 16 bytes/param | 8 + 8/d bytes/param |

Where `d` is the data-parallel size.

**Key Features:**

* Contiguous buffers for parameters and main gradients
* Immediate gradient copying to main gradients as they're computed
* Efficient reduce-scatter for gradient synchronization
* All-gather for parameter updates

**Usage:**

```bash
--use-distributed-optimizer
--overlap-grad-reduce      # Overlap gradient reduction with computation
--overlap-param-gather     # Overlap parameter gathering
```

### FP8 Training: Next-Generation Precision

Megatron supports **FP8 mixed precision training**, optimized for NVIDIA Hopper, Ada, and Blackwell GPUs.

**Benefits:**

* **Faster Training**: FP8 kernels provide significant speedups
* **Memory Savings**: Reduced memory footprint for weights and activations
* **Better Scaling**: Enables training of even larger models

**Configuration:**

```bash
# FP8 training configuration
--fp8-format hybrid
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
--fp8-param-gather          # Gather parameters in FP8
```

**Requirements:**

* NVIDIA Hopper (H100), Ada (RTX 4090), or Blackwell GPUs
* Transformer Engine >= 1.1
* Megatron Core >= 0.5.0

### When Do You Need Megatron?

Megatron becomes necessary when one or more of the following conditions hold:

* Transformer layers with extremely large hidden dimensions (e.g., 16K+)
* Large MoE expert layers requiring expert parallelism
* FP8 or other low-precision regimes with massive GEMMs
* Scaling to hundreds of GPUs where per-layer computation dominates
* Long-context training (>=8K tokens) requiring context parallelism
* Models where individual layers exceed single-GPU computation capacity
* Production training requiring maximum performance and scalability

If none of these apply, state sharding alone is usually sufficient.

### Real-World Training Configurations

Here are production-ready configurations based on actual Megatron training scripts:

**LLaMA-3 8B with FP8 Training (8 GPUs):**

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
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
    --bf16
```

**GPT-3 175B Scale (128 GPUs):**

```bash
torchrun --nproc_per_node=8 --nnodes=16 pretrain_gpt.py \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 2048 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 16 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --use-distributed-optimizer \
    --fp16
```

**Mixtral 8x7B MoE (64 GPUs):**

```bash
torchrun --nproc_per_node=8 --nnodes=8 pretrain_gpt.py \
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
    --bf16
```

### Complete Training Example with Megatron

The code examples include a complete Megatron Core training script (`code/train_megatron_mcore.py`) that demonstrates how to set up tensor parallelism, create a GPT model, and run a training loop. The script handles distributed initialization, model creation with `TransformerConfig`, and gradient synchronization with Megatron's `DistributedDataParallel`.

**Running the Megatron training script:**

```bash
# Single node, 4 GPUs with tensor parallelism
torchrun --nproc_per_node=4 code/train_megatron_mcore.py

# Multi-node (2 nodes, 4 GPUs each, TP=4 per node)
torchrun --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=node0 \
  --master_port=29500 \
  code/train_megatron_mcore.py
```

**Key points in this example:**

1. **Megatron Core Models**: Uses `GPTModel` from Megatron Core with built-in tensor parallelism
2. **DistributedDataParallel**: Megatron's DDP wrapper with optimized communication overlap
3. **Distributed Optimizer**: Shards optimizer states across data-parallel ranks
4. **Pipeline Schedule**: Uses Megatron's forward-backward function for efficient pipeline execution
5. **Memory Efficiency**: Each GPU only stores a fraction of each layer's parameters and optimizer states

**Using Megatron-FSDP for State Sharding:**

For even larger models, combine Megatron TP with Megatron-FSDP:

```bash
# Enable Megatron-FSDP with tensor parallelism
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--tensor-model-parallel-size 4
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

This gives you:
* **Computation sharding**: Megatron TP for large per-layer computation
* **State sharding**: Megatron-FSDP for parameters, gradients, and optimizer states
* **Performance**: 15-25% faster than PyTorch FSDP2 + Megatron TP
* **Memory**: 23% memory savings compared to PyTorch FSDP2

**Performance Optimizations:**

```bash
# Enable all performance optimizations
--overlap-grad-reduce              # Overlap gradient reduction
--overlap-param-gather             # Overlap parameter gathering
--tp-comm-overlap                  # Overlap TP communication
--sequence-parallel                # Reduce activation memory
--use-distributed-optimizer        # Shard optimizer states
--calculate-per-token-loss        # Optimize gradient scaling
```

**Advanced Features:**

* **Virtual Pipeline Parallelism**: Reduces pipeline bubbles by interleaving micro-batches
* **Distributed Checkpointing**: Up to 50x faster than native PyTorch, supports resharding
* **CUDA Graphs**: Capture and replay training iterations for reduced overhead
* **Activation Recomputation**: Selective recompute for memory-constrained scenarios

## Hybrid Parallelism in Practice

Large-scale model training rarely relies on a single parallelism strategy. In practice, modern systems combine multiple forms of parallelism to address different bottlenecks simultaneously. This section describes how **state sharding** and **computation sharding** are composed in real training systems, and provides guidance on common hybrid configurations.

### The Two-Axis View of Parallelism

Hybrid parallelism can be understood as operating along two orthogonal axes:

* **State axis**: how model parameters, gradients, and optimizer states are distributed across devices
  (e.g., FSDP or ZeRO)

* **Computation axis**: how the computation of a single forward and backward pass is distributed
  (e.g., Megatron tensor, pipeline, and sequence parallelism)

A key insight is that these axes are independent. State sharding reduces memory redundancy, while computation sharding reduces per-device computational load. Effective large-scale training requires both.

### A Canonical Hybrid Configuration

A widely used hybrid setup combines:

* **FSDP (or ZeRO-3)** for state sharding across all data-parallel ranks
* **Megatron Tensor Parallelism (TP)** within each data-parallel group
* **Megatron Pipeline Parallelism (PP)** across groups of layers
* Optional **Sequence Parallelism (SP)** for long sequences

Conceptually, the system is organized hierarchically:

1. **Tensor-parallel groups** cooperate to compute individual layers
2. **Pipeline stages** split the model depth across groups of GPUs
3. **Data-parallel / FSDP groups** replicate computation across batches while sharding state

Each layer of parallelism addresses a different scaling limit.

### Example: Training a Large Transformer Model

Consider training a large Transformer model whose individual layers are too large to compute efficiently on a single GPU.

A typical configuration might look like:

* Tensor Parallelism: TP = 4
* Pipeline Parallelism: PP = 2
* Data Parallelism with FSDP: DP = 8

This yields a total of:

$$
\text{Total GPUs} = \text{TP} \times \text{PP} \times \text{DP} = 4 \times 2 \times 8 = 64
$$

In this setup:

* Each layer's matrix multiplications are split across 4 GPUs (TP)
* The model is divided into 2 pipeline stages (PP)
* 8 replicas process different micro-batches, with parameters sharded across them (FSDP)

From the perspective of a single GPU, it:

* Stores only a shard of the model state
* Computes only a fraction of each layer
* Participates in pipeline execution for a subset of layers

### Why This Composition Works

This hybrid design works because it aligns each technique with the bottleneck it is best suited to address:

* **FSDP / ZeRO** minimizes memory usage by eliminating redundant state
* **Tensor Parallelism** reduces per-GPU compute and enables larger hidden dimensions
* **Pipeline Parallelism** limits activation memory and enables scaling across nodes
* **Sequence Parallelism** reduces activation replication for long-context models

No single technique can address all of these constraints alone.

### Choosing a Hybrid Strategy

In practice, the choice of hybrid configuration depends on a small number of structural questions:

* Can a single Transformer layer be computed efficiently on one GPU?
* Is the model too deep to fit activation memory comfortably?
* Is sequence length a dominant factor in memory usage?
* How many GPUs are available per node, and how fast is inter-node communication?

A useful rule of thumb is:

* If layers fit on one GPU, start with state sharding alone.
* If layers do not fit or are inefficient, add tensor parallelism.
* If depth or node count becomes limiting, add pipeline parallelism.
* If long sequences dominate memory, enable sequence parallelism.

Hybrid parallelism is typically introduced incrementally, as each additional dimension increases system complexity.

### Operational Considerations

Hybrid parallelism introduces new operational challenges:

* **Communication topology awareness**: Tensor parallel groups benefit from fast intra-node interconnects, while pipeline stages often span nodes.
* **Micro-batch sizing**: Pipeline parallelism requires careful tuning of micro-batch count to maintain utilization.
* **Checkpointing**: State-sharded checkpoints must be coordinated with tensor- and pipeline-parallel layouts.
* **Debugging complexity**: Errors may surface only under specific parallel configurations.

For this reason, hybrid setups are typically adopted only after simpler configurations have reached their limits.

### Summary

Hybrid parallelism combines state sharding and computation sharding to overcome both memory and compute limits. By composing FSDP or ZeRO with Megatron's tensor, pipeline, and sequence parallelism, training systems can scale far beyond what any single technique enables on its own. Understanding how these strategies interact is important for building robust large-scale training systems.

Large-scale training is no longer about choosing a single parallelism strategy, but about composing multiple strategies along orthogonal axes.

### Performance Optimization Best Practices

**Communication Overlap:**

Enable all available communication overlap options:

```bash
--overlap-grad-reduce          # Overlap gradient reduction (DP/FSDP)
--overlap-param-gather        # Overlap parameter gathering (FSDP)
--tp-comm-overlap             # Overlap tensor parallel communication
```

**Memory Optimizations:**

```bash
--sequence-parallel            # Reduce activation memory (required with TP+EP)
--use-distributed-optimizer   # Shard optimizer states
--calculate-per-token-loss   # Optimize gradient scaling
--recompute-activations       # Activation checkpointing when needed
```

**Parallelism Topology Guidelines:**

1. **Keep TP and EP within NVLink domain**: Both are communication-intensive
2. **Use PP for inter-node scaling**: Pipeline stages can span nodes
3. **CP for long sequences**: Enable when sequence length >= 8K
4. **Minimize model parallelism**: Prefer DP with distributed optimizer when possible

**Reference Configurations:**

Based on NVIDIA NeMo production configurations:

| Model | Size | GPUs | TP | PP | CP | EP | Notes |
|-------|------|------|----|----|----|----|-------|
| LLaMA-3 | 8B | 8 | 1 | 1 | 2 | 1 | CP for long seqlen (8K) |
| LLaMA-3 | 70B | 64 | 4 | 4 | 2 | 1 | TP+PP for large model |
| LLaMA-3.1 | 405B | 1024 | 8 | 8 | 2 | 1 | 3D parallelism |
| GPT-3 | 175B | 128-512 | 4-8 | 8-16 | 1 | 1 | Large model config |
| Mixtral | 8x7B | 64 | 1 | 4 | 1 | 8 | EP for MoE |
| Mixtral | 8x22B | 256 | 4 | 4 | 8 | 8 | Combined TP+EP |
| DeepSeek-V3 | 671B | 1024 | 2 | 16 | 1 | 64 | Large MoE config |

**Performance Benchmarks:**

Megatron Core achieves:
* **Up to 47% Model FLOP Utilization (MFU)** on H100 clusters
* **468 TFLOPS** for Mixtral 8X7B bf16 training
* **15-25% speedup** with Megatron-FSDP vs PyTorch FSDP2
* **50x faster checkpointing** with distributed checkpointing vs native PyTorch

## Choosing the Right Strategy: ZeRO, FSDP, and Megatron

### Decision Tree

```
Model Size < 10B?
+- Yes -> Use DDP or ZeRO-1
|         (simplest, fastest, easiest to debug)
+- No
    |
    Model Size < 50B?
    +- Yes -> Use ZeRO-2 or FSDP2
    |         (good balance, shards gradients too)
    +- No
        |
        Can single layer fit and compute efficiently on one GPU?
        +- Yes -> Use ZeRO-3 or FSDP2
        |         (state sharding is sufficient)
        |         Models: 7B-30B, standard architectures
        +- No -> Use FSDP2 + Megatron TP (or ZeRO-3 + Megatron TP)
                (need computation sharding for large layers)
                |
                Sequence Length >= 8K?
                +- Yes -> Add Context Parallelism (CP)
                |         FSDP2 + Megatron TP + CP
                |         (reduces activation memory for long sequences)
                +- No
                    |
                    Model Size < 200B?
                    +- Yes -> FSDP2 + Megatron TP
                    |         (or ZeRO-3 + Megatron TP)
                    |         Models: 50B-200B, large hidden dims
                    +- No
                        |
                        Multiple Nodes?
                        +- Yes -> Add Pipeline Parallelism (PP)
                        |         FSDP2 + Megatron TP + PP
                        |         (hierarchical parallelism for inter-node scaling)
                        |         Optional: ZeRO++ for communication optimization
                        +- No
                            |
                            Model Size < 500B?
                            +- Yes -> FSDP2 + Megatron TP
                            +- No -> ZeRO-Infinity + Megatron TP
                                    (offload to NVMe for extreme scale)
                                    |
                                    MoE Model?
                                    +- Yes -> Add Expert Parallelism (EP)
                                    |         FSDP2 + Megatron TP + EP
                                    |         (or ZeRO-3 + Megatron TP + EP)
                                    |         Models: Mixtral, DeepSeek-V3, Qwen-MoE
                                    +- No -> Continue with TP + PP
```

### Comparison Table

| Stage | Params | Grads | Opt States | Memory/GPU | Comm Overhead | Best For |
|-------|--------|-------|------------|------------|---------------|----------|
| **DDP** | Full | Full | Full | N× | All-reduce | <10B params |
| **ZeRO-1** | Full | Full | Shard | 0.5× | All-reduce | 10-30B params |
| **ZeRO-2** | Full | Shard | Shard | 0.33× | Reduce-scatter | 30-50B params |
| **ZeRO-3** | Shard | Shard | Shard | 1/N× | All-gather + RS | 50-200B params |
| **ZeRO-Offload** | Full | Shard | CPU | GPU: 0.25× | CPU-GPU transfer | Limited GPU mem |
| **ZeRO-Infinity** | NVMe | Shard | CPU/NVMe | GPU: minimal | Multi-tier transfer | >500B params |
| **ZeRO++** | Shard | Shard | Shard | 2/N× | Reduced by 4-6× | Multi-node large models |
| **FSDP2** | Shard | Shard | Shard | 1/N× | All-gather + RS | 7B-200B params (when layers fit on one GPU) |
| **Megatron TP** | Shard | Shard | Shard | 1/TP× | All-gather per layer | Large layers, 50B+ models |
| **Megatron TP + CP** | Shard | Shard | Shard | 1/(TP×CP)× | TP + CP comm | Long sequences (>=8K), activation memory reduction |
| **Megatron TP + PP** | Shard | Shard | Shard | 1/(TP×PP)× | TP + PP comm | Very large models, inter-node scaling |
| **Megatron EP (MoE)** | Shard | Shard | Shard | 1/EP× (MoE layer) | All-to-all | MoE models (Mixtral, DeepSeek-V3) |
| **FSDP2 + Megatron TP** | Shard | Shard | Shard | 1/(N×TP)× | Both patterns | 50B-200B+ models with large layers |
| **Megatron-FSDP + TP** | Shard | Shard | Shard | 1/(N×TP)× | Optimized overlap | Maximum performance, 15-25% faster than FSDP2+TP |
| **Full Hybrid (TP+PP+CP+EP)** | Shard | Shard | Shard | 1/(TP×PP×CP×EP)× | All patterns | Extreme scale (200B+), MoE, long context |

### Memory Savings Example

For a **175B parameter model with Adam** on **4 GPUs**:

| Configuration | Params/GPU | Grads/GPU | Opt/GPU | Total/GPU | Savings |
|---------------|------------|-----------|---------|-----------|---------|
| DDP | 350 GB | 350 GB | 1,400 GB | **2,100 GB** | 1× |
| ZeRO-1 | 350 GB | 350 GB | 350 GB | **1,050 GB** | 2× |
| ZeRO-2 | 350 GB | 88 GB | 350 GB | **788 GB** | 2.7× |
| ZeRO-3 | 88 GB | 88 GB | 350 GB | **526 GB** | 4× |
| ZeRO-3 (8 GPUs) | 44 GB | 44 GB | 175 GB | **263 GB** | 8× |

## Practical Tips and Best Practices

### Start Simple, Scale Up

```python
# Phase 1: Get it working
- Start with ZeRO-1 or ZeRO-2
- Verify convergence matches DDP baseline
- Profile memory usage

# Phase 2: Optimize for scale  
- Move to ZeRO-3 if needed
- Add activation checkpointing
- Tune batch size and gradient accumulation

# Phase 3: Production optimization
- Add ZeRO++ for multi-node
- Tune communication overlap
- Profile and eliminate bottlenecks
```

### Common Pitfalls

**1. Wrong stage for model size**
```python
# Bad: Using ZeRO-3 for 7B model
# - Unnecessary communication overhead
# - Slower than ZeRO-2

# Good: Match stage to model size (see decision tree)
```

**2. Checkpoint incompatibility**
```python
# Problem: ZeRO-3 checkpoints are sharded by default
# Can't load on different GPU count or for inference

# Solution: Gather full weights when saving
{
  "zero_optimization": {
    "stage": 3,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

**3. OOM despite using ZeRO**
```python
# Common causes:
# - Activations still too large → Use activation checkpointing
# - Batch size too large → Reduce or use gradient accumulation
# - Sequence length too long → Use sequence parallelism

# Check what's using memory:
torch.cuda.memory_summary()
```

**4. Slow multi-node training**
```python
# Symptoms: Good single-node, poor multi-node scaling
# Cause: Inter-node communication bottleneck

# Solutions:
# 1. Use ZeRO++ (hpZ for hierarchical partitioning)
# 2. Verify InfiniBand is working (not falling back to Ethernet)
# 3. Check network topology (should be non-blocking switch fabric)
```

### Hyperparameter Tuning

**Gradient accumulation with ZeRO:**
```json
{
  "gradient_accumulation_steps": 8,
  "zero_optimization": {
    "stage": 2
  }
}
```

**Key point**: With ZeRO-2/3, gradient accumulation is even more important because it amortizes communication overhead.

**Bucket sizes:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "reduce_bucket_size": 5e8,        // Larger = less overhead, more memory
    "stage3_prefetch_bucket_size": 5e8,  // Tune for overlap
    "stage3_param_persistence_threshold": 1e5  // Keep small params unsharded
  }
}
```

### Debugging ZeRO

**Enable verbose logging:**
```json
{
  "steps_per_print": 10,
  "wall_clock_breakdown": true
}
```

**Profile memory:**
```python
import deepspeed

# Add to training loop
if step % 100 == 0:
    deepspeed.runtime.utils.memory_status(
        "Memory Status", 
        reset_max=True
    )
```

**Check communication:**
```bash
# Monitor network traffic
nvidia-smi dmon -i 0 -s u
iftop -i ib0  # InfiniBand interface
```

## Complete Training Examples: ZeRO and Megatron

This section provides complete, production-ready training examples for both ZeRO and Megatron, based on real-world configurations used in large-scale model training. The full code is available in the `code/` directory.

### Example 1: Training with DeepSpeed ZeRO-3

The `code/train_zero3.py` script demonstrates training a large Transformer model (~7B parameters) using DeepSpeed ZeRO-3 with CPU offloading. The accompanying `code/ds_config_zero3.json` configuration file enables ZeRO Stage 3 with optimizer and parameter offloading to CPU, gradient accumulation, FP16 training, and activation checkpointing.

```bash
# Single node, 2 GPUs
deepspeed --num_gpus=2 code/train_zero3.py \
    --deepspeed --deepspeed_config code/ds_config_zero3.json

# Multi-node (2 nodes, 8 GPUs each)
deepspeed --num_gpus=8 --num_nodes=2 \
    --master_addr=node0 --master_port=29500 \
    code/train_zero3.py \
    --deepspeed --deepspeed_config code/ds_config_zero3.json
```

### Example 2: Training with Megatron Core (LLaMA-3 8B)

The `code/train_megatron_llama3_8b.sh` script provides a production-ready configuration for training LLaMA-3 8B with Megatron Core. Key features include context parallelism (CP=2) for 8K sequences, FP8 mixed precision for Hopper/Ada GPUs, distributed optimizer, and communication overlap.

```bash
# Single node training with mock data
./code/train_megatron_llama3_8b.sh

# With real data
./code/train_megatron_llama3_8b.sh \
    checkpoints/llama3_8b \
    tensorboard_logs/llama3_8b \
    /path/to/tokenizer.model \
    /path/to/data_prefix

# Multi-node (on each node, set NODE_RANK appropriately)
MASTER_ADDR=node0 NODE_RANK=0 ./code/train_megatron_llama3_8b.sh ...
```

### Example 3: Megatron MoE Training (Mixtral 8x7B)

The `code/train_megatron_mixtral.sh` script demonstrates training a Mixture-of-Experts model with Megatron's expert parallelism. This configuration requires 64 GPUs (8 nodes × 8 GPUs) and features 8-way expert parallelism, 4-stage pipeline parallelism, all-to-all token routing, and auxiliary loss for load balancing.

```bash
# Requires 64 GPUs across 8 nodes
# On each node (set NODE_RANK=0..7):
MASTER_ADDR=node0 NODE_RANK=0 ./code/train_megatron_mixtral.sh \
    checkpoints/mixtral \
    /path/to/tokenizer.model \
    /path/to/data_prefix
```

## ZeRO vs FSDP vs Megatron: When to Use Which?

Both ZeRO-3 and PyTorch FSDP do parameter sharding (state sharding). They solve the same problem—distributing model parameters, gradients, and optimizer states across GPUs to reduce memory usage. The choice between them depends on your specific requirements, existing infrastructure, and constraints.

### PyTorch FSDP2:

- **Native PyTorch integration**: Pure PyTorch, no external dependencies
- **torch.compile support**: FSDP2 works well with PyTorch's compiler
- **Simpler codebase**: FSDP2 is lightweight and straightforward to debug
- **Large models** (7B-200B): FSDP2 is mature and well-optimized for this range
- **GPU-only sharding is sufficient**: When your model fits with full parameter sharding across available GPUs

### DeepSpeed ZeRO:

- **GPU memory is insufficient even with full sharding**: When aggregate GPU memory across all devices is still not enough
- **CPU or NVMe offloading**: ZeRO-Offload (CPU) and ZeRO-Infinity (NVMe) extend memory beyond GPUs, though with throughput tradeoffs
- **Multi-node training optimizations**: ZeRO++ provides communication optimizations that can help in large, heterogeneous network environments
- **DeepSpeed ecosystem integration**: If you're using other DeepSpeed features (MoE, compression, etc.)

**Important caveat**: CPU and NVMe offloading come with substantial throughput penalties. These are "feasibility" solutions—they enable training that wouldn't otherwise be possible, but at the cost of slower training speeds. Use them only when GPU-only approaches are truly insufficient.

### Use Megatron (Computation Sharding):

Megatron becomes necessary when:

- **Individual layers exceed single-GPU computation capacity**: When a single Transformer layer's matrix operations are too large or too slow for one GPU
- **Very large hidden dimensions**: Models with hidden_size >= 16K require tensor parallelism
- **MoE models**: Expert parallelism is the standard approach for Mixture-of-Experts architectures
- **Long-context training**: Context parallelism (CP) is the most efficient solution for sequences >= 8K tokens
- **Multi-node scaling**: Pipeline parallelism enables efficient scaling across nodes
- **Maximum performance**: Megatron-FSDP provides 15-25% speedup over PyTorch FSDP2

**Megatron Parallelism Strategies:**

* **Tensor Parallelism (TP)**: Use when individual layers are too large for single GPU
* **Pipeline Parallelism (PP)**: Use for inter-node scaling and very deep models
* **Context Parallelism (CP)**: Use for long sequences (>=8K tokens) to reduce activation memory
* **Expert Parallelism (EP)**: Use for MoE models to distribute experts across GPUs
* **Sequence Parallelism**: Typically enabled with TP to reduce activation memory

**When to Combine with State Sharding:**

* **FSDP2 + Megatron TP**: Common pattern for 50B-200B models
* **Megatron-FSDP + Megatron TP**: High-performance option (15-25% faster than FSDP2+TP in some configurations)
* **ZeRO-3 + Megatron TP**: Widely used in existing codebases, especially when DeepSpeed ecosystem is already in place

### Hybrid Approach

The modern standard is to combine state sharding with computation sharding:

```bash
# FSDP2 + Megatron TP (common pattern)
--use-torch-fsdp2
--tensor-model-parallel-size 4
--sequence-parallel

# Megatron-FSDP + Megatron TP (maximum performance)
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--tensor-model-parallel-size 4
--sequence-parallel

# Full hybrid for extreme scale
--use-megatron-fsdp
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 8
--context-parallel-size 2
--expert-model-parallel-size 8  # For MoE
```


## Summary and Key Takeaways

**Practical guidance**: When training large models, practitioners typically begin with state sharding techniques (FSDP2 or ZeRO-3) and add Megatron-style computation parallelism when per-layer computation becomes the bottleneck. DeepSpeed ZeRO offers additional capabilities for CPU and NVMe offloading, which can be valuable when GPU memory is constrained. Common patterns include **FSDP2 + Megatron Tensor Parallelism** and **ZeRO-3 + Megatron Tensor Parallelism**, with the choice depending on your infrastructure and requirements.

**State sharding (FSDP2 / ZeRO):**

- **FSDP2**: PyTorch-native, well-integrated, suitable for many large-model training scenarios
- **ZeRO stages**: Progressive sharding from optimizer states (ZeRO-1) to full parameter sharding (ZeRO-3)
- **DeepSpeed extensions**: ZeRO-Offload (CPU), ZeRO-Infinity (NVMe), ZeRO++ (multi-node communication) for scenarios where GPU-only sharding is insufficient
- **Key insight**: These techniques eliminate memory redundancy but assume each layer can be computed on a single GPU

**Computation sharding (Megatron):**

- **Tensor Parallelism (TP)**: Splits large matrix operations across GPUs when individual layers exceed single-GPU limits
- **Pipeline Parallelism (PP)**: Shards model depth across GPUs/nodes for very deep models, with virtual pipeline support
- **Context Parallelism (CP)**: Advanced long-context solution that partitions sequences, eliminating recompute overhead
- **Expert Parallelism (EP)**: Specialized parallelism for MoE models, achieving 468 TFLOPS for Mixtral 8X7B
- **Sequence Parallelism**: Splits activations along sequence dimension, essential when TP is enabled
- **Key insight**: Megatron addresses a fundamentally different problem—computation itself, not just memory

**Hybrid parallelism:**

- **FSDP2 + Megatron TP**: Common pattern for 50B-200B+ models
  - FSDP2 handles state sharding across all GPUs
  - Megatron TP handles computation sharding for large layers
- **ZeRO-3 + Megatron TP**: Alternative pattern, especially when DeepSpeed ecosystem is in use
- **Megatron-FSDP + TP**: High-performance option (15-25% faster than FSDP2+TP in some configurations)
- **Full hybrid**: FSDP2/Megatron-FSDP/ZeRO-3 + Megatron TP + PP + CP for 200B+ models
  - TP for large layers (within nodes)
  - PP for inter-node scaling
  - CP for long sequences (>=8K tokens)
  - EP for MoE models
- **Why it works**: State sharding and computation sharding operate on orthogonal axes and address different bottlenecks
- **Performance**: Up to 47% MFU on H100 clusters, 468 TFLOPS for MoE training

**Decision framework:**
1. **Can a single layer fit and compute efficiently on one GPU?**
   - Yes → State sharding alone may be sufficient (e.g., FSDP2 or ZeRO-3 for 7B-30B models)
   - No → Consider adding computation sharding (e.g., FSDP2/ZeRO-3 + Megatron TP for 50B+ models)

2. **Is GPU-only state sharding insufficient?**
   - Yes → Consider DeepSpeed ZeRO-Offload/Infinity for CPU/NVMe offloading (with throughput tradeoffs)
   - No → FSDP2, Megatron-FSDP, or ZeRO-3 may be sufficient depending on your infrastructure

3. **Do you need multi-node scaling?**
   - Yes → Add Megatron Pipeline Parallelism (PP) for inter-node scaling
   - No → Tensor parallelism within nodes may be sufficient

4. **Is sequence length >= 8K tokens?**
   - Yes → Add Megatron Context Parallelism (CP) to reduce activation memory
   - No → Sequence parallelism with TP may be sufficient

5. **Is this an MoE model?**
   - Yes → Add Megatron Expert Parallelism (EP) for efficient expert routing
   - No → Standard TP/PP may be sufficient

6. **What is your infrastructure and ecosystem?**
   - PyTorch-focused → FSDP2 or Megatron-FSDP may integrate better
   - DeepSpeed ecosystem → ZeRO-3 + Megatron may be more natural
   - Performance-critical → Consider Megatron-FSDP (15-25% faster than FSDP2+TP in some configurations)

**The fundamental principle**: Large-scale training is no longer about choosing a single parallelism strategy, but about composing multiple strategies along orthogonal axes. State sharding and computation sharding are complementary, not competing.

So far, we've focused on distributed training—how to train large models across multiple GPUs. But training is only half the story. Once you've trained a model, you need to serve it efficiently at scale. The next part of this book shifts focus to distributed inference: how to run inference on large models efficiently, handle high-throughput workloads, and serve models in production. We'll start with vLLM, a high-performance inference engine that uses techniques like PagedAttention and continuous batching to maximize throughput and minimize latency.



## References

### DeepSpeed and ZeRO

- [ZeRO Paper (2020)](https://arxiv.org/abs/1910.02054): Original ZeRO optimization
- [ZeRO-Offload Paper (2021)](https://arxiv.org/abs/2101.06840): CPU offloading techniques  
- [ZeRO-Infinity Paper (2021)](https://arxiv.org/abs/2104.07857): NVMe offloading
- [ZeRO++ Paper (2023)](https://arxiv.org/abs/2306.10209): Communication-optimized ZeRO
- [DeepSpeed Documentation](https://www.deepspeed.ai/): Official docs and tutorials
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed): Source code and examples
- [DeepSpeed Megatron Tutorial](https://www.deepspeed.ai/tutorials/megatron/): Training with DeepSpeed and Megatron

### Megatron-LM

- [Megatron-LM Paper (2019)](https://arxiv.org/abs/1909.08053): Tensor parallelism for large language models
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM): Source code and examples
- [Megatron Core Documentation](https://docs.nvidia.com/megatron-core/): Official API documentation
- [ROCm AI Developer Hub - Megatron Setup](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html): AMD GPU setup guide
- [ROCm Megatron-LM Benchmark](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html?model=pyt_megatron_lm_train_llama-3.3-70b): ROCm training guide
- [AWS Neuron Megatron Training](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.9.1/frameworks/torch/torch-neuronx/tutorials/training/megatron_lm_gpt.html): AWS Inferentia training

### Recent Research Papers

- [Arctic Long Sequence Training (2025)](https://arxiv.org/html/2507.19845v1): Scalable training for multi-million token sequences
- [SuperOffload (2025)](https://arxiv.org/html/2502.19811v3): Large-scale LLM training on superchips
- [ZenFlow (2025)](https://arxiv.org/html/2502.07846): Stall-free offloading engine
- [DeepCompile (2025)](https://arxiv.org/html/2505.11432): Compiler optimization for distributed training
- [Universal Checkpointing (2024)](https://arxiv.org/html/2503.15758): Efficient checkpointing for large-scale training
- [Megatron MoE Performance (2024)](https://arxiv.org/html/2411.05288): MoE training optimizations
- [Context Parallelism (2024)](https://arxiv.org/html/2412.14711): Long-context training techniques




