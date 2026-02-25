# Chapter 5: Beyond State Sharding with DeepSpeed and Megatron {-}

*Extending memory capacity and sharding computation for very large models*

> The future is already here. It's just unevenly distributed.
- William Gibson, Writer

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

Figure~\ref{fig:tensor-parallel} illustrates this two-step pattern. The column-parallel linear (top) splits the weight matrix by columns, so each GPU computes a slice of the output with no communication. The row-parallel linear (bottom) splits by rows, and an all-reduce combines the partial results. By pairing these two operations—column-parallel followed by row-parallel—a complete MLP block requires only one all-reduce. This is the key to Megatron's efficiency: communication is minimized to a single synchronization point per layer, rather than at every operation.

![Tensor parallelism: column-parallel and row-parallel linear.](img/tensor_parallelism.png){#fig:tensor-parallel .block width=90% align=center}

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

Tensor parallelism splits individual layers across GPUs, but there's another dimension we can exploit: the model's depth. A 32-layer Transformer doesn't need all 32 layers on every GPU—we can assign layers 0–15 to one GPU and layers 16–31 to another. This is **Pipeline Parallelism (PP)**.

The idea is intuitive: data flows through the model like water through a pipe. GPU 0 processes the first half of the layers, then passes the intermediate activations to GPU 1, which processes the second half. But here's the catch—if we naively process one batch at a time, GPU 1 sits idle while GPU 0 is working, and vice versa. This "pipeline bubble" can waste up to 50% of compute.

The solution is to split the batch into smaller **micro-batches** and pipeline them. While GPU 1 is processing micro-batch 1 through layers 16–31, GPU 0 can start processing micro-batch 2 through layers 0–15. With enough micro-batches in flight, we keep all GPUs busy most of the time. This approach is called **1F1B (One Forward One Backward)**: each GPU alternates between forward passes and backward passes, maintaining a steady state where all stages are active.

![Pipeline parallelism: naive vs 1F1B schedule.](img/pipeline_parallelism.png){#fig:pipeline-parallelism .block width=100% align=center}

Figure~\ref{fig:pipeline-parallelism} contrasts the naive approach with the 1F1B schedule. In the naive pipeline (top), a single batch flows through all 4 GPUs sequentially—GPU 0 runs forward (F), passes to GPU 1, and so on until GPU 3 completes forward, then backward (B) propagates back. The white space represents idle time (bubbles). In the 1F1B schedule (bottom), we split the batch into 4 micro-batches (F1–F4, B1–B4). Each GPU processes multiple micro-batches in an interleaved fashion, dramatically reducing idle time.

Megatron supports several pipeline schedules. The 1F1B schedule shown above is the most common. The original **GPipe** schedule[^gpipe] runs all forward passes first, then all backward passes—simpler but with larger bubbles. **Interleaved pipelines**[^interleaved] (also called Virtual Pipeline Parallelism) go further by assigning multiple non-contiguous chunks of layers to each GPU, reducing bubble size at the cost of more communication.

[^interleaved]: Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," SC 2021. https://arxiv.org/abs/2104.04473

[^gpipe]: Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," NeurIPS 2019. https://arxiv.org/abs/1811.06965

Virtual Pipeline Parallelism deserves special mention. Instead of assigning layers 0–15 to GPU 0 and 16–31 to GPU 1, we might assign layers 0–7 and 16–23 to GPU 0, and layers 8–15 and 24–31 to GPU 1. Each GPU now runs two "virtual stages." This interleaving reduces the pipeline bubble because micro-batches cycle through stages faster. The trade-off is additional point-to-point communication between stages.

A typical configuration looks like:

```bash
--pipeline-model-parallel-size 4          # 4 pipeline stages
--num-layers-per-virtual-pipeline-stage 2 # VPP: 2 layers per virtual stage
```

Pipeline parallelism shines in a specific scenario: **multi-node training**. Within a node, GPUs are connected by fast NVLink (~600 GB/s). Across nodes, you're limited to InfiniBand (~400 GB/s) or worse. Tensor parallelism requires frequent all-reduce operations within each layer—fine over NVLink, painful over InfiniBand. Pipeline parallelism only requires point-to-point communication of activations between stages, which is much more tolerant of slower interconnects.

The practical guideline is: use tensor parallelism within a node (where NVLink keeps communication fast), and pipeline parallelism across nodes (where the communication pattern is more forgiving). This combination—TP within nodes, PP across nodes—is the standard approach for training models at the 100B+ scale.

To see pipeline parallelism in action with a simplified implementation:

```bash
# Pipeline parallelism demo with 2 stages
torchrun --nproc_per_node=2 code/pipeline_parallel_simple.py
```

This example demonstrates model partitioning, micro-batch scheduling, and forward/backward coordination across pipeline stages.


### Sequence Parallelism and Context Parallelism: Long Contexts

So far we've discussed parallelism strategies that address model size—sharding parameters, gradients, optimizer states, and computation. But there's another dimension that's becoming increasingly important: **sequence length**. Modern models are trained with context windows of 8K, 32K, even 128K tokens. At these lengths, activation memory—the intermediate values stored during forward pass for use in backward pass—can exceed the memory needed for the model itself.

Consider a Transformer with hidden dimension 4096 processing a 32K token sequence. Each layer stores activations of shape (batch, 32K, 4096), and with 32 layers, the activation memory can easily reach tens of gigabytes per GPU. This is where **sequence parallelism** and **context parallelism** come in.

**Sequence parallelism**[^seqpar] is the simpler of the two. When tensor parallelism is enabled, certain operations like LayerNorm and Dropout don't participate in the TP communication—they operate on the full hidden dimension locally. Sequence parallelism extends the sharding to these operations by splitting activations along the sequence dimension. If you have TP=4, sequence parallelism means each GPU only stores 1/4 of the sequence's activations for these operations. It's typically enabled alongside tensor parallelism with minimal overhead.

[^seqpar]: Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models," MLSys 2023. https://arxiv.org/abs/2205.05198

**Context parallelism (CP)**[^ringatt] takes a more aggressive approach. While sequence parallelism only shards the activations of LayerNorm and Dropout, context parallelism partitions *everything* along the sequence dimension—inputs, all intermediate activations, and attention computation itself. With CP=2 on an 8K sequence, each GPU processes only 4K tokens throughout the entire forward and backward pass.

[^ringatt]: Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context," ICLR 2024. https://arxiv.org/abs/2310.01889

The challenge is attention. In standard self-attention, each token's query must attend to all keys and values in the sequence. If GPU 0 holds tokens 0–3999 and GPU 1 holds tokens 4000–7999, how does a query on GPU 0 attend to keys on GPU 1? Context parallelism solves this using a technique called **ring attention**. The idea is elegant: instead of gathering all KV pairs to every GPU (which would defeat the memory savings), we pass KV chunks around in a ring. GPU 0 computes attention for its queries against its local KV, then sends its KV to GPU 1 and receives GPU 1's KV. Now GPU 0 computes attention against the new KV chunk, accumulating the results. After one full rotation around the ring, every query has seen every key-value pair, but no GPU ever held the full sequence.

The communication pattern is carefully optimized. Modern implementations overlap the KV transfer with attention computation—while computing attention against the current KV chunk, the next chunk is already being transferred. Combined with Grouped-Query Attention (GQA), which shares KV heads across multiple query heads, the communication volume is significantly reduced.

The benefit is substantial. Without CP, training on very long sequences often requires activation checkpointing (recomputing activations during backward pass), which adds ~30% overhead. With CP, you can eliminate this recompute entirely by simply distributing the activation memory across more GPUs. The trade-off is communication, but for long sequences the compute-to-communication ratio remains favorable.

![Sequence parallelism and context parallelism (ring attention).](img/sequence_context_parallelism.png){#fig:seq-ctx-parallel .block width=100% align=center}

Figure~\ref{fig:seq-ctx-parallel} illustrates both techniques. Sequence parallelism (left) splits activations along the sequence dimension—each GPU stores only its portion of the sequence for LayerNorm and Dropout operations. Context parallelism (right) uses ring attention: each GPU holds local Q, K, V chunks, and K, V pairs rotate around a ring so every query can attend to all keys without any GPU holding the full sequence.

A typical configuration for long-context training:

```bash
--tensor-model-parallel-size 2
--context-parallel-size 4        # Split 32K sequence across 4 GPUs = 8K per GPU
--sequence-parallel              # Also enable sequence parallelism
```

The rule of thumb: use context parallelism when sequence length exceeds 8K tokens and activation memory is your bottleneck. For shorter sequences, tensor parallelism and sequence parallelism are usually sufficient.

The accompanying `code/sp_demo.py` illustrates these concepts. The `memory` mode calculates activation memory for various sequence lengths—you'll see that a 32K sequence with 32 layers can exceed 40GB, explaining why parallelism is necessary. The `sequence_parallel` mode shows how each GPU holds only a portion of the sequence and applies LayerNorm locally without communication. The `ring_attention` mode demonstrates the core ring attention pattern: each GPU starts with local Q, K, V chunks, then K and V rotate around the ring. After one full rotation, every query has attended to all keys, yet no GPU ever held the full sequence.

```bash
# Show activation memory scaling (single GPU)
python code/sp_demo.py --mode memory
# Sequence parallelism: split activations along sequence dim (2 GPUs)
torchrun --nproc_per_node=2 code/sp_demo.py --mode sequence_parallel
# Context parallelism via ring attention (2 GPUs)
torchrun --nproc_per_node=2 code/sp_demo.py --mode ring_attention
```

### DeepSpeed-Ulysses: An Alternative to Ring Attention

Ring attention is not the only way to parallelize attention across long sequences. DeepSpeed introduced **DeepSpeed-Ulysses**[^ulysses], a different approach that trades communication pattern for simplicity.

[^ulysses]: Jacobs et al., "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models," arXiv 2023. https://arxiv.org/abs/2309.14509

The key insight behind Ulysses is straightforward: instead of rotating KV chunks around a ring, why not just gather the full sequence before attention and scatter it back afterward? In ring attention, each GPU computes partial attention scores against different KV chunks and accumulates them—this requires careful bookkeeping of softmax normalization across chunks. Ulysses sidesteps this complexity entirely.

Here's how it works. Before the attention computation, each GPU holds a chunk of the sequence with shape (batch, local_seq, num_heads, head_dim). Ulysses performs an **all-to-all** communication that reorganizes the data: instead of each GPU holding all heads for a portion of the sequence, each GPU now holds all sequence positions for a portion of the heads. After this transpose, the shape becomes (batch, full_seq, local_heads, head_dim). Now each GPU can compute standard self-attention on its subset of heads—no partial scores, no accumulation, just regular attention. After attention, another all-to-all reverses the transformation, returning to the original partitioning.

![DeepSpeed-Ulysses: 2D transpose via all-to-all.](img/ulysses.png){#fig:ulysses .block width=85% align=center}

Figure~\ref{fig:ulysses} illustrates this 2D transpose. On the left (sequence-parallel), each GPU holds a local sequence chunk $S_i$ with all heads $H_{all}$—the tensor shape is (local_seq, all_heads). The all-to-all collectively transposes the data: on the right (head-parallel), each GPU holds the full sequence $S_{all}$ but only its local heads $H_i$—the shape becomes (full_seq, local_heads). Now each GPU can compute standard self-attention independently on its subset of heads. After attention, another all-to-all reverses the transformation.

The trade-off is communication volume versus communication pattern. Ring attention sends KV chunks P times around a ring of P GPUs, with each transfer overlapped with computation. Ulysses performs two all-to-all collectives (before and after attention), which involve all GPUs simultaneously. For small parallelism degrees (P ≤ 8), Ulysses often wins because all-to-all on modern interconnects like NVLink is highly optimized. For larger P or when interconnect bandwidth is limited, ring attention's overlapped communication can be more efficient.

In practice, DeepSpeed-Ulysses shines in scenarios where you want sequence parallelism without the complexity of ring attention's partial softmax handling. It integrates cleanly with ZeRO and other DeepSpeed optimizations.

The accompanying `code/ulysses_demo.py` implements the core Ulysses algorithm from scratch. It demonstrates the two all-to-all operations: `all_to_all_seq_to_head` transposes from (batch, local_seq, num_heads, head_dim) to (batch, full_seq, local_heads, head_dim), and `all_to_all_head_to_seq` reverses this transformation. Between these two operations, each GPU runs standard self-attention on its subset of heads—no partial softmax accumulation required.

```bash
# Run Ulysses demo with 2 GPUs
torchrun --nproc_per_node=2 code/ulysses_demo.py
# Run with 4 GPUs (heads must be divisible by world_size)
torchrun --nproc_per_node=4 code/ulysses_demo.py
```

When should you choose Ulysses over ring attention? If you're already in the DeepSpeed ecosystem and want straightforward sequence parallelism with moderate parallelism degrees, Ulysses is the easier path. If you're scaling to very long sequences (100K+ tokens) with large parallelism degrees, ring attention's communication overlap may provide better efficiency.


### Expert Parallelism: Scaling MoE Models

Mixture-of-Experts (MoE) models present a unique scaling opportunity: instead of making every layer wider, we add multiple "expert" sub-networks and route each token to only a subset of them. A model like Mixtral 8x7B has 8 experts per MoE layer, but each token only activates 2 of them. This means the model has the capacity of a much larger network while keeping per-token computation manageable. But how do we distribute these experts across GPUs?

This is where **Expert Parallelism (EP)** comes in. The idea is natural: if we have 8 experts and 8 GPUs, put one expert on each GPU. When a token needs to be processed by expert 3, it gets routed to GPU 3, processed, and the result is sent back. The communication pattern is all-to-all: tokens from all GPUs may need to go to any expert, and results flow back to their origin.

The challenge is load balancing. If the router sends 80% of tokens to expert 0 and only 2% to expert 7, GPU 0 is overloaded while GPU 7 sits idle. MoE training typically includes an auxiliary loss that encourages the router to distribute tokens more evenly. Megatron supports several load balancing strategies: auxiliary loss (adds a penalty for imbalanced routing), Sinkhorn (iterative normalization to enforce balance), and aux-loss-free methods that achieve balance through architectural constraints.

Expert parallelism combines naturally with other parallelism dimensions. A typical large-scale MoE training might use EP=8 for the experts, PP=4 for pipeline stages, and DP for data parallelism across nodes. The non-expert layers (attention, LayerNorm) can use tensor parallelism independently. This flexibility is essential for models like DeepSeek-V3 or Qwen-MoE that have hundreds of experts.

A configuration for Mixtral 8x7B training might look like:

```bash
--num-experts 8
--expert-model-parallel-size 8   # One expert per GPU
--moe-router-topk 2              # Each token activates 2 experts
--moe-router-load-balancing-type aux_loss
--moe-grouped-gemm               # Batch expert computations
--pipeline-model-parallel-size 4
```

The `--moe-grouped-gemm` flag is worth noting: when a GPU hosts multiple experts (EP < num_experts), it batches the computations across experts into a single grouped matrix multiplication, significantly improving GPU utilization. For very large expert counts, specialized communication libraries like DeepEP[^deepep] optimize the all-to-all token dispatching with low-latency GPU kernels and efficient cross-node transfers.

[^deepep]: DeepEP is DeepSeek's open-source expert-parallel communication library. https://github.com/deepseek-ai/DeepEP

The accompanying `code/expert_parallel_demo.py` is a from-scratch implementation that demonstrates the core EP mechanics without Megatron dependencies. It shows how a router assigns tokens to experts, how all-to-all communication dispatches tokens to their destination GPUs, how each GPU processes tokens with its local expert, and how another all-to-all returns results. The demo prints token distribution statistics so you can see how routing decisions affect load balance.

```bash
# Run with 2 experts (2 GPUs)
torchrun --nproc_per_node=2 code/expert_parallel_demo.py
# Run with 4 experts (4 GPUs)
torchrun --nproc_per_node=4 code/expert_parallel_demo.py
```

### Why FSDP2 Cannot Replace Megatron

A common misconception is that FSDP2 (or ZeRO) and Megatron are interchangeable—that you pick one or the other based on preference. This misunderstands what each system actually does.

FSDP2 and ZeRO shard *state*: parameters, gradients, and optimizer states are distributed across GPUs, then gathered when needed for computation. The key assumption is that each layer's forward and backward pass fits on a single GPU. When GPU 0 needs to compute layer 5, it gathers layer 5's parameters from all GPUs, runs the computation locally, then releases the memory. The computation itself is not distributed—only the storage is.

Megatron shards *computation*: a single layer's matrix multiplication is split across multiple GPUs, with each GPU computing a portion of the result. The communication happens *inside* the layer, not around it. This is fundamentally different from gathering parameters before computation.

Why does this distinction matter? Consider a model where a single attention layer has a 16K × 16K weight matrix. Even if you shard the parameters across 8 GPUs with FSDP2, when it's time to compute, one GPU must gather the full matrix and perform the multiplication. If that matrix doesn't fit in one GPU's memory, or if the computation is too slow on one GPU, FSDP2 cannot help—it only shards storage, not compute.

This is where Megatron becomes necessary. With tensor parallelism, that 16K × 16K matrix is split across 8 GPUs, each holding a 16K × 2K slice. The computation happens in parallel, and only the results are communicated. No single GPU ever needs to hold or compute with the full matrix.

The practical implication: FSDP2 and Megatron are complementary, not competing. You might use Megatron's tensor parallelism to split large layers across GPUs within a node, while using FSDP2-style sharding across nodes for memory efficiency. The choice isn't "which one" but "how to combine them."

### Hybrid Parallelism: Combining State and Computation Sharding

Given that FSDP2/ZeRO and Megatron solve different problems, the natural question is: can we use both? The answer is yes, and this is exactly what large-scale training systems do.

Consider training a 70B parameter model on 64 GPUs across 8 nodes. Within each node (8 GPUs connected by NVLink), you use Megatron's tensor parallelism with TP=8 to split the large matrix multiplications. Across nodes (connected by slower InfiniBand), you use ZeRO-3 or FSDP2 to shard the optimizer states and gradients—this reduces memory pressure without requiring the high-bandwidth communication that tensor parallelism demands. If the model is deep, you might add pipeline parallelism to distribute layers across node groups.

This layered approach plays to each technique's strengths. Tensor parallelism needs high bandwidth (hence NVLink within a node), but it enables computation that wouldn't fit on a single GPU. State sharding tolerates higher latency (hence cross-node), but it dramatically reduces per-GPU memory. Pipeline parallelism adds another dimension of scaling with relatively modest communication.

The choice between FSDP2 and ZeRO-3 for the state sharding layer depends on your ecosystem. FSDP2 integrates tightly with PyTorch's compiler stack (torch.compile) and is the native PyTorch solution. ZeRO-3, through DeepSpeed, offers additional features like CPU and NVMe offloading for memory-constrained setups, and has a mature ecosystem of optimizations. Both work well with Megatron-style computation sharding—the key is understanding that they operate on orthogonal axes.

### Megatron Core: Production-Ready Library

Throughout this chapter, we've discussed Megatron's parallelism strategies conceptually. But how do you actually use them in practice? The answer is **Megatron Core**, a library extracted from the original Megatron-LM research codebase and refined for production use.

Megatron Core provides GPU-optimized building blocks: attention layers with tensor parallelism built in, MLP blocks that understand pipeline boundaries, embedding layers that handle vocabulary parallelism. You don't implement the column-parallel and row-parallel patterns yourself—you use `ColumnParallelLinear` and `RowParallelLinear` from the library, and the communication is handled automatically.

Beyond the basic building blocks, Megatron Core includes the infrastructure that large-scale training requires: activation recomputation to trade compute for memory, distributed checkpointing that saves and loads sharded model states efficiently, and FP8 precision support optimized for NVIDIA's latest GPUs (Hopper, Ada, Blackwell). The distributed optimizer shards optimizer states across data-parallel ranks, complementing the computation sharding we've discussed.

Getting started is straightforward:

```bash
pip install --no-build-isolation megatron-core[mlm,dev]

# Or use NVIDIA's container with everything pre-installed
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.04-py3
```

The `code/megatron_gpt_pretrain.sh` script in this chapter's code directory demonstrates a production configuration: tensor parallelism across GPUs, distributed optimizer for memory efficiency, flash attention for speed, and the various flags we've discussed throughout this chapter.

### Megatron-FSDP: Optimized State Sharding

We've established that state sharding (FSDP/ZeRO) and computation sharding (Megatron) are complementary. But when you combine them, the implementation details matter. PyTorch's FSDP2 is a general-purpose solution; it doesn't know about Megatron's tensor parallelism or the specific communication patterns involved. This is where **Megatron-FSDP** comes in.

Megatron-FSDP is NVIDIA's implementation of fully sharded data parallelism, designed to work seamlessly with Megatron's other parallelism dimensions. The performance difference is meaningful: benchmarks show 15-25% speedup and 23% memory savings compared to PyTorch FSDP2. These gains come from optimizations that are only possible when the FSDP implementation understands the surrounding context—better bucketing of parameters, smarter buffer management, and more aggressive overlap of communication with computation.

One technical detail worth noting: Megatron-FSDP uses NCCL's userbuffer feature to reduce GPU Streaming Multiprocessor (SM) consumption during communication. In large-scale training, SMs spent on communication are SMs not available for computation. This optimization keeps more SMs free for the actual matrix multiplications.

Enabling Megatron-FSDP in your training script looks like:

```bash
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params  # Equivalent to ZeRO-3
--overlap-grad-reduce
--overlap-param-gather
```

When should you choose Megatron-FSDP over PyTorch FSDP2? If you're already using Megatron's tensor parallelism, context parallelism, or expert parallelism, Megatron-FSDP integrates naturally and delivers better performance. If you need FP8 training with Transformer Engine, Megatron-FSDP has native support. On the other hand, if you want a pure PyTorch stack with torch.compile support and no external dependencies, PyTorch FSDP2 is the cleaner choice.

### Distributed Optimizer: Memory-Efficient Optimization

Even with tensor parallelism and pipeline parallelism handling the computation, optimizer states remain a significant memory burden. Adam, for example, stores two additional tensors (momentum and variance) for every parameter—that's 8 bytes per parameter on top of the parameters and gradients themselves. For a 70B model, optimizer states alone can consume over 500GB.

Megatron's **distributed optimizer** addresses this by sharding optimizer states across data-parallel ranks, conceptually similar to ZeRO Stage 1. Each GPU only stores the optimizer states for a fraction of the parameters. During the optimizer step, gradients are reduce-scattered (each GPU gets the reduced gradient for its shard), the local optimizer updates its shard, and the updated parameters are all-gathered back.

The memory savings scale with data-parallel size. With 8-way data parallelism using bf16 parameters and fp32 gradients, per-GPU memory drops from 18 bytes per parameter to roughly 7.5 bytes—a 2.4x reduction. The exact formula depends on your precision configuration:

| Config | Without distributed | Distributed (d GPUs) |
|--------------|-----------------|-------------------------------------|
| fp16 params, fp16 grads | 20 bytes/param | 4 + 16/d bytes/param |
| bf16 params, fp32 grads | 18 bytes/param | 6 + 12/d bytes/param |
| fp32 params, fp32 grads | 16 bytes/param | 8 + 8/d bytes/param |

The implementation includes several optimizations beyond basic sharding. Gradients are copied into contiguous buffers as they're computed, enabling efficient reduce-scatter operations. The communication can be overlapped with backward computation (`--overlap-grad-reduce`) and with the next forward pass (`--overlap-param-gather`), hiding much of the latency.

```bash
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

### FP8 Training: Next-Generation Precision

The progression from FP32 to FP16/BF16 brought significant speedups and memory savings. NVIDIA's latest GPUs (Hopper, Ada, Blackwell) take this further with native FP8 support—8-bit floating point that halves memory and doubles throughput compared to FP16.

FP8 training isn't as simple as changing a dtype flag. The dynamic range of 8-bit floats is much narrower than 16-bit, so values must be carefully scaled to avoid overflow and underflow. Megatron handles this through Transformer Engine, which tracks the maximum absolute values (amax) of tensors and adjusts scaling factors dynamically. The `--fp8-amax-history-len` parameter controls how many recent amax values to consider when computing scales.

```bash
--fp8-format hybrid
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
--fp8-param-gather          # Gather parameters in FP8 to save communication
```

The `hybrid` format uses E4M3 (4 exponent bits, 3 mantissa bits) for forward pass and E5M2 (5 exponent bits, 2 mantissa bits) for backward—a balance between range and precision that works well in practice. The `--fp8-param-gather` flag is particularly useful with distributed optimizer: parameters are gathered in FP8 format, reducing all-gather communication volume by half.

FP8 requires hardware support: NVIDIA H100, RTX 4090, or newer GPUs, plus Transformer Engine 1.1 or later. If you have the hardware, the speedup is substantial—often 1.5-2x over BF16 for large matrix multiplications.

### When Do You Need Megatron?

After all this discussion of tensor parallelism, pipeline parallelism, context parallelism, and expert parallelism, a natural question is: when do you actually need any of this? The answer depends on your model and hardware.

If a single Transformer layer fits comfortably on one GPU and computes fast enough, you don't need Megatron. State sharding (FSDP2 or ZeRO) handles memory, and data parallelism handles scaling. Most models under 10B parameters fall into this category on modern GPUs.

Megatron becomes necessary when you hit one of these walls. First, layer size: if your hidden dimension is 16K or larger, a single attention layer's weight matrices may not fit on one GPU, or the computation may be too slow. Tensor parallelism solves this. Second, sequence length: if you're training with 8K+ token contexts, activation memory explodes, and context parallelism or sequence parallelism becomes essential. Third, MoE models: expert parallelism is the natural way to distribute hundreds of experts. Fourth, scale: when you're using hundreds of GPUs, the efficiency gains from Megatron's optimized communication patterns compound significantly.

If none of these apply—your layers fit, your sequences are moderate, you're not using MoE, and you're training on a handful of GPUs—state sharding alone is simpler and sufficient.

### Real-World Training Configurations

Theory is useful, but seeing real configurations helps solidify understanding. Here are production-ready examples based on actual Megatron training scripts. Note that `pretrain_gpt.py` is part of the Megatron-LM repository—clone it from https://github.com/NVIDIA/Megatron-LM and run these commands from within that repository.

__LLaMA-3 8B with FP8 Training (8 GPUs):__

This configuration trains a LLaMA-3 8B model on a single 8-GPU node with long context (8K tokens) and FP8 precision.

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

The model architecture flags (`--num-layers`, `--hidden-size`, `--ffn-hidden-size`, `--num-attention-heads`) define the LLaMA-3 8B structure. `--group-query-attention` with `--num-query-groups 8` enables Grouped-Query Attention, where 32 query heads share 8 KV heads—this reduces KV cache memory significantly. For parallelism, we skip tensor parallelism (`--tensor-model-parallel-size 1`) since each layer fits on one GPU, but use context parallelism (`--context-parallel-size 2`) to handle the 8K sequence by splitting it across 2 GPUs. FP8 training (`--fp8-format hybrid`, `--fp8-param-gather`) provides speedup on H100 GPUs. The distributed optimizer with overlap flags maximizes memory efficiency and hides communication latency.

__GPT-3 175B Scale (128 GPUs):__

This configuration scales to 175B parameters across 16 nodes (128 GPUs total), requiring both tensor and pipeline parallelism.

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

At this scale, a single layer with hidden dimension 12288 benefits from tensor parallelism across all 8 GPUs within a node (`--tensor-model-parallel-size 8`). The 96 layers are distributed across 16 pipeline stages (`--pipeline-model-parallel-size 16`), with each stage handling 6 layers. The effective data parallelism is 128 / (8 × 16) = 1, meaning all GPUs are dedicated to model parallelism. The large `--global-batch-size 1536` is achieved through gradient accumulation across many micro-batches.

__Mixtral 8x7B MoE (64 GPUs):__

This configuration trains a Mixture-of-Experts model with 8 experts distributed across GPUs.

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

The MoE-specific flags define the sparse architecture: `--num-experts 8` creates 8 expert networks per MoE layer, and `--expert-model-parallel-size 8` distributes them one per GPU within the expert-parallel group. `--moe-router-topk 2` means each token is routed to 2 experts. The optimization flags `--moe-grouped-gemm` and `--moe-permute-fusion` batch expert computations and fuse token rearrangement operations for efficiency. Pipeline parallelism (`--pipeline-model-parallel-size 4`) distributes the 32 layers across 4 stages, while tensor parallelism is disabled for the MoE layers since expert parallelism handles the distribution.

### Complete Training Example with Megatron

To tie everything together, let's look at how to actually run a Megatron training job. The `code/train_megatron_mcore.py` script in this chapter demonstrates the essential pieces: initializing distributed state, creating a model with `TransformerConfig`, wrapping it with Megatron's `DistributedDataParallel`, and running a training loop with proper gradient synchronization.

Running on a single node with 4 GPUs:

```bash
torchrun --nproc_per_node=4 code/train_megatron_mcore.py
```

For multi-node training, you need to specify the cluster topology. On node 0:

```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=node0 --master_port=29500 \
  code/train_megatron_mcore.py
```

And on node 1, the same command with `--node_rank=1`. The script uses Megatron Core's `GPTModel`, which has tensor parallelism built in—you don't manually implement the column-parallel and row-parallel patterns. Megatron's DDP wrapper handles gradient synchronization with optimized communication overlap, and the distributed optimizer shards optimizer states automatically.

For models that need both computation sharding and aggressive state sharding, you can combine Megatron's tensor parallelism with Megatron-FSDP:

```bash
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--tensor-model-parallel-size 4
--overlap-grad-reduce
--overlap-param-gather
```

This configuration gives you the best of both worlds: tensor parallelism splits large matrix multiplications across GPUs, while FSDP shards parameters, gradients, and optimizer states across the data-parallel dimension. The overlap flags ensure that communication happens concurrently with computation whenever possible.

A few additional optimizations worth enabling in production. `--tp-comm-overlap` overlaps tensor parallelism's all-reduce with computation. `--sequence-parallel` reduces activation memory by sharding along the sequence dimension for LayerNorm and Dropout. `--calculate-per-token-loss` optimizes gradient scaling for variable-length sequences.

Beyond these flags, Megatron offers several advanced features. Virtual pipeline parallelism (interleaved scheduling) reduces pipeline bubbles by having each GPU handle multiple non-contiguous stages. Distributed checkpointing saves and loads sharded model states up to 50x faster than naive PyTorch checkpointing, with support for resharding—you can save a checkpoint from a 64-GPU run and load it on 128 GPUs. CUDA graphs capture entire training iterations and replay them with minimal CPU overhead. Activation recomputation lets you selectively recompute activations during backward pass to trade compute for memory when needed.

## Hybrid Parallelism in Practice

We've now covered the individual parallelism techniques: ZeRO's state sharding, Megatron's tensor and pipeline parallelism, sequence and context parallelism for long sequences, and expert parallelism for MoE models. But real training systems rarely use just one. A 70B model might use tensor parallelism within nodes, pipeline parallelism across nodes, and FSDP for optimizer state sharding—all simultaneously. How do these pieces fit together?

### The Two-Axis View

The key insight is that state sharding and computation sharding operate on orthogonal axes. State sharding (FSDP, ZeRO) addresses memory redundancy: instead of every GPU storing all parameters and optimizer states, each GPU stores a fraction. Computation sharding (tensor parallelism, pipeline parallelism) addresses computational load: instead of one GPU computing an entire layer, multiple GPUs share the work.

These axes are independent. You can have state sharding without computation sharding (a 7B model with ZeRO-3), computation sharding without state sharding (a 70B model with TP=8 and full parameter replication), or both (a 405B model with TP, PP, and FSDP). The choice depends on which bottleneck you're hitting.

### Building a Hybrid Configuration

Let's walk through how you might configure a 70B model training on 64 GPUs across 8 nodes. Each node has 8 GPUs connected by NVLink; nodes are connected by InfiniBand.

Start with tensor parallelism. The model's hidden dimension is 8192, and each attention layer has large weight matrices. We set TP=4, splitting each layer's computation across 4 GPUs within a node. This requires high-bandwidth communication (NVLink), so we keep the TP group within a single node.

Next, consider pipeline parallelism. The model has 80 layers, and even with TP=4, storing all layers' activations is challenging. We set PP=2, splitting the model into two pipeline stages of 40 layers each. Pipeline communication (sending activations between stages) is less frequent than TP communication, so it can tolerate the slower inter-node InfiniBand.

Finally, data parallelism. With TP=4 and PP=2, each "model replica" uses 8 GPUs. We have 64 GPUs total, so DP=8: eight replicas process different micro-batches in parallel. We enable FSDP to shard optimizer states across these 8 replicas, reducing per-GPU memory.

The math: Total GPUs = TP × PP × DP = 4 × 2 × 8 = 64. From a single GPU's perspective, it stores 1/8 of the optimizer states (FSDP), computes 1/4 of each layer (TP), and handles 1/2 of the model's depth (PP).

### Why This Works

This layered approach succeeds because each technique addresses a different constraint. FSDP eliminates redundant optimizer state storage—critical for Adam's momentum and variance tensors. Tensor parallelism enables matrix multiplications that wouldn't fit or would be too slow on a single GPU. Pipeline parallelism bounds activation memory by limiting how many layers are active simultaneously. Each technique has costs (communication overhead, pipeline bubbles), but when applied to the right bottleneck, the benefits outweigh the costs.

### Choosing Your Configuration

The decision process is incremental. Start simple and add complexity only when needed.

If your model's layers fit on one GPU and compute efficiently, use state sharding alone (FSDP or ZeRO-3). This is the simplest setup and works for most models under 10-15B parameters on modern GPUs.

If layers are too large or too slow on one GPU, add tensor parallelism. Keep TP within a node to leverage NVLink. TP=2, 4, or 8 are common choices depending on layer size.

If the model is very deep or you need to scale across many nodes, add pipeline parallelism. PP introduces bubbles, so use it when necessary rather than by default.

If sequence length is your bottleneck (8K+ tokens), enable sequence parallelism or context parallelism. These reduce activation memory proportionally to the parallelism degree.

If you're training an MoE model, expert parallelism distributes experts naturally. EP often replaces TP for the expert layers since the experts are already separate computations.

### Operational Realities

Hybrid parallelism adds operational complexity. Tensor parallel groups must be placed on GPUs with fast interconnects—putting a TP group across nodes will cripple performance. Pipeline parallelism requires tuning the number of micro-batches to minimize bubble overhead. Checkpointing becomes more complex: a checkpoint from a TP=4, PP=2 configuration can't be directly loaded into a TP=8, PP=1 setup without resharding.

Debugging also becomes harder. A bug might only manifest with specific parallelism configurations, making reproduction difficult. For these reasons, start with the simplest configuration that meets your needs and add parallelism dimensions incrementally.

### Performance Optimization Best Practices

Once you have a working hybrid configuration, there are several optimizations that can significantly improve throughput.

The most impactful is communication overlap. By default, communication and computation happen sequentially—the GPU computes, then communicates, then computes again. With overlap enabled, communication happens in the background while the next computation proceeds. Megatron provides several overlap flags:

```bash
--overlap-grad-reduce          # Overlap gradient all-reduce with backward
--overlap-param-gather         # Overlap parameter gather with forward
--tp-comm-overlap              # Overlap tensor parallel all-reduce
```

For memory optimization, sequence parallelism reduces activation memory by sharding along the sequence dimension for LayerNorm and Dropout. The distributed optimizer shards optimizer states across data-parallel ranks. Activation recomputation trades compute for memory by recomputing activations during backward instead of storing them.

```bash
--sequence-parallel
--use-distributed-optimizer
--recompute-activations        # When memory-constrained
```

A few topology guidelines based on communication characteristics. Tensor parallelism and expert parallelism are communication-intensive—keep them within the NVLink domain (same node). Pipeline parallelism tolerates higher latency—it can span nodes. Context parallelism for long sequences works best within a node but can extend across nodes if necessary.

### Configuration Patterns

A few patterns emerge from production training setups. For dense models under 10B parameters, tensor parallelism is often unnecessary—layers fit on one GPU, so state sharding (FSDP or ZeRO) handles memory while data parallelism handles scaling. For larger dense models (70B+), tensor parallelism becomes essential for the large matrix multiplications, typically TP=4 or TP=8 within a node. Pipeline parallelism adds another scaling dimension when you need more GPUs than fit in a TP group.

MoE models follow a different pattern. Expert parallelism naturally distributes the experts, often replacing tensor parallelism for the expert layers entirely. A Mixtral-style 8x7B model might use EP=8 (one expert per GPU) with no tensor parallelism, plus pipeline parallelism for depth.

Context parallelism appears when sequence length drives memory usage. For 8K+ token sequences, CP=2 or CP=4 can halve or quarter activation memory without the complexity of full model parallelism.

## Choosing the Right Strategy

With so many parallelism techniques available, how do you decide which to use? The answer depends on your specific constraints: model size, layer size, sequence length, available hardware, and whether you're training a dense or MoE model.

### A Decision Framework

![Parallelism strategy decision tree.](img/parallelism_decision_tree.png){#fig:parallelism-decision-tree .block width=80% align=center}

Figure~\ref{fig:parallelism-decision-tree} provides a visual guide. The decision process starts with the simplest question: does your model fit on one GPU with standard data parallelism? If yes, use DDP—it's the simplest and most efficient. If not, the next question is whether a single layer fits on one GPU. If layers fit but the full model doesn't, state sharding (FSDP2 or ZeRO-3) is your answer. If individual layers are too large, you need computation sharding (tensor parallelism). From there, additional dimensions like pipeline parallelism, context parallelism, and expert parallelism address specific bottlenecks.

### Understanding the Trade-offs

Each technique makes a different trade-off between memory savings and communication overhead. DDP replicates everything—maximum communication efficiency but no memory savings. ZeRO-1 shards only optimizer states, cutting memory roughly in half with minimal overhead. ZeRO-2 adds gradient sharding, and ZeRO-3 shards everything, achieving near-linear memory scaling with GPU count but requiring all-gather operations before each layer's computation.

Tensor parallelism shards computation rather than just storage. It reduces per-GPU memory and compute proportionally to the TP degree, but requires high-bandwidth communication (all-reduce) within each layer. This is why TP works best within a node where NVLink provides the bandwidth.

Pipeline parallelism shards by depth rather than width. It introduces pipeline bubbles (idle time) but tolerates higher-latency communication, making it suitable for cross-node scaling.

### Concrete Memory Example

To make this concrete, consider a 70B parameter model with Adam optimizer. In FP16, parameters take 140GB, gradients another 140GB, and Adam's optimizer states (FP32 master weights, momentum, variance) take 840GB—over 1TB total.

With DDP on 8 GPUs, each GPU stores all 1TB+. With ZeRO-1, optimizer states are sharded: each GPU stores 140GB params + 140GB grads + 105GB optimizer = 385GB. With ZeRO-3, everything is sharded: each GPU stores roughly 140GB total (1/8 of each component). The memory scales linearly with GPU count—add more GPUs, use less memory per GPU.

The catch is communication. ZeRO-3 must all-gather parameters before each layer and reduce-scatter gradients after. For models where layers are small relative to communication latency, this overhead can be significant. For large models with substantial per-layer computation, the overhead is amortized and ZeRO-3 works well.

## Summary

This chapter covered two complementary approaches to scaling model training beyond what fits on a single GPU.

**State sharding** (ZeRO, FSDP2) eliminates memory redundancy by distributing parameters, gradients, and optimizer states across GPUs. ZeRO's progressive stages—from optimizer-only sharding (Stage 1) to full parameter sharding (Stage 3)—let you trade communication overhead for memory savings. Extensions like ZeRO-Offload and ZeRO-Infinity push memory to CPU and NVMe when GPU memory is exhausted, though with throughput penalties. ZeRO++ optimizes multi-node communication through quantization and hierarchical partitioning.

**Computation sharding** (Megatron) addresses a different problem: when individual layers are too large or too slow for a single GPU. Tensor parallelism splits matrix multiplications across GPUs within a layer. Pipeline parallelism distributes model depth across pipeline stages. Context parallelism and sequence parallelism handle long sequences by sharding activations. Expert parallelism distributes MoE experts naturally.

The key insight is that these approaches are orthogonal. State sharding reduces memory redundancy; computation sharding reduces per-GPU computational load. Modern large-scale training combines both: tensor parallelism within nodes (where NVLink provides bandwidth), pipeline parallelism across nodes (where latency is tolerable), and state sharding for optimizer memory efficiency.

The decision framework is incremental. Start with the simplest approach that works—often just state sharding for models under 30B parameters. Add tensor parallelism when layers become the bottleneck. Add pipeline parallelism for very deep models or multi-node scaling. Add context parallelism for long sequences. Add expert parallelism for MoE models. Each dimension adds complexity, so add them only when needed.

So far, we've focused on distributed training. But training is only half the story. Once you've trained a model, you need to serve it efficiently. The next part of this book shifts to distributed inference: how to run large models at scale, handle high-throughput workloads, and serve models in production.



## References

__DeepSpeed and ZeRO__

- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2020). https://arxiv.org/abs/1910.02054
- ZeRO-Offload: Democratizing Billion-Scale Model Training (2021). https://arxiv.org/abs/2101.06840
- ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning (2021). https://arxiv.org/abs/2104.07857
- ZeRO++: Extremely Efficient Collective Communication for Giant Model Training (2023). https://arxiv.org/abs/2306.10209
- DeepSpeed Documentation: https://www.deepspeed.ai/
- DeepSpeed GitHub: https://github.com/microsoft/DeepSpeed

__Megatron-LM__

- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (2019). https://arxiv.org/abs/1909.08053
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (2021). https://arxiv.org/abs/2104.04473
- Reducing Activation Recomputation in Large Transformer Models (2023). https://arxiv.org/abs/2205.05198
- Megatron-LM GitHub: https://github.com/NVIDIA/Megatron-LM
- Megatron Core Documentation: https://docs.nvidia.com/megatron-core/

__Research__

- Arctic Long Sequence Training: Scalable Training for Multi-Million Token Sequences (2025). https://arxiv.org/abs/2507.19845
- SuperOffload: Large-Scale LLM Training on Superchips (2025). https://arxiv.org/abs/2502.19811
- ZenFlow: Stall-Free Offloading Engine (2025). https://arxiv.org/abs/2502.07846
- DeepCompile: Compiler Optimization for Distributed Training (2025). https://arxiv.org/abs/2505.11432
- Universal Checkpointing for Large-Scale Training (2024). https://arxiv.org/abs/2503.15758
- Ring Attention with Blockwise Transformers for Near-Infinite Context (2024). https://arxiv.org/abs/2310.01889
- DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models (2023). https://arxiv.org/abs/2309.14509




