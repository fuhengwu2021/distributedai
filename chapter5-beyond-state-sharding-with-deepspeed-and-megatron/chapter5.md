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

Here are production-ready configurations based on actual Megatron training scripts. Note that `pretrain_gpt.py` is part of the Megatron-LM repository—you need to clone it from https://github.com/NVIDIA/Megatron-LM and run these commands from within that repository:

__LLaMA-3 8B with FP8 Training (8 GPUs):__

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

__GPT-3 175B Scale (128 GPUs):__

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

__Mixtral 8x7B MoE (64 GPUs):__

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

__Running the Megatron training script:__

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

__Key points in this example:__

1. **Megatron Core Models**: Uses `GPTModel` from Megatron Core with built-in tensor parallelism
2. **DistributedDataParallel**: Megatron's DDP wrapper with optimized communication overlap
3. **Distributed Optimizer**: Shards optimizer states across data-parallel ranks
4. **Pipeline Schedule**: Uses Megatron's forward-backward function for efficient pipeline execution
5. **Memory Efficiency**: Each GPU only stores a fraction of each layer's parameters and optimizer states

__Using Megatron-FSDP for State Sharding:__

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

__Performance Optimizations:__

```bash
# Enable all performance optimizations
--overlap-grad-reduce              # Overlap gradient reduction
--overlap-param-gather             # Overlap parameter gathering
--tp-comm-overlap                  # Overlap TP communication
--sequence-parallel                # Reduce activation memory
--use-distributed-optimizer        # Shard optimizer states
--calculate-per-token-loss        # Optimize gradient scaling
```

__Advanced Features:__

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

__Communication Overlap:__

Enable all available communication overlap options:

```bash
--overlap-grad-reduce          # Overlap gradient reduction (DP/FSDP)
--overlap-param-gather        # Overlap parameter gathering (FSDP)
--tp-comm-overlap             # Overlap tensor parallel communication
```

__Memory Optimizations:__

```bash
--sequence-parallel            # Reduce activation memory (required with TP+EP)
--use-distributed-optimizer   # Shard optimizer states
--calculate-per-token-loss   # Optimize gradient scaling
--recompute-activations       # Activation checkpointing when needed
```

__Parallelism Topology Guidelines:__

1. **Keep TP and EP within NVLink domain**: Both are communication-intensive
2. **Use PP for inter-node scaling**: Pipeline stages can span nodes
3. **CP for long sequences**: Enable when sequence length >= 8K
4. **Minimize model parallelism**: Prefer DP with distributed optimizer when possible

__Reference Configurations:__

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

__Performance Benchmarks:__

Megatron Core achieves:
* **Up to 47% Model FLOP Utilization (MFU)** on H100 clusters
* **468 TFLOPS** for Mixtral 8X7B bf16 training
* **15-25% speedup** with Megatron-FSDP vs PyTorch FSDP2
* **50x faster checkpointing** with distributed checkpointing vs native PyTorch

## Choosing the Right Strategy: ZeRO, FSDP, and Megatron

### Decision Tree

![Parallelism strategy decision tree.](img/parallelism_decision_tree.png){#fig:parallelism-decision-tree .block width=80% align=center}

Figure~\ref{fig:parallelism-decision-tree} provides a decision tree for choosing the right parallelism strategy. The key questions to ask are: How large is your model? Does a single layer fit on one GPU? How long are your sequences? Are you training across multiple nodes? Is it a Mixture-of-Experts model? Each path leads to a recommended combination of techniques—from simple DDP for small models to complex combinations of FSDP2, tensor parallelism, pipeline parallelism, context parallelism, and expert parallelism for the largest models.

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

__1. Wrong stage for model size__
```python
# Bad: Using ZeRO-3 for 7B model
# - Unnecessary communication overhead
# - Slower than ZeRO-2

# Good: Match stage to model size (see decision tree)
```

__2. Checkpoint incompatibility__
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

__3. OOM despite using ZeRO__
```python
# Common causes:
# - Activations still too large → Use activation checkpointing
# - Batch size too large → Reduce or use gradient accumulation
# - Sequence length too long → Use sequence parallelism

# Check what's using memory:
torch.cuda.memory_summary()
```

__4. Slow multi-node training__
```python
# Symptoms: Good single-node, poor multi-node scaling
# Cause: Inter-node communication bottleneck

# Solutions:
# 1. Use ZeRO++ (hpZ for hierarchical partitioning)
# 2. Verify InfiniBand is working (not falling back to Ethernet)
# 3. Check network topology (should be non-blocking switch fabric)
```

### Hyperparameter Tuning

__Gradient accumulation with ZeRO:__
```json
{
  "gradient_accumulation_steps": 8,
  "zero_optimization": {
    "stage": 2
  }
}
```

**Key point**: With ZeRO-2/3, gradient accumulation is even more important because it amortizes communication overhead.

__Bucket sizes:__
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

__Enable verbose logging:__
```json
{
  "steps_per_print": 10,
  "wall_clock_breakdown": true
}
```

__Profile memory:__
```python
import deepspeed

# Add to training loop
if step % 100 == 0:
    deepspeed.runtime.utils.memory_status(
        "Memory Status", 
        reset_max=True
    )
```

__Check communication:__
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

__Megatron Parallelism Strategies:__

* **Tensor Parallelism (TP)**: Use when individual layers are too large for single GPU
* **Pipeline Parallelism (PP)**: Use for inter-node scaling and very deep models
* **Context Parallelism (CP)**: Use for long sequences (>=8K tokens) to reduce activation memory
* **Expert Parallelism (EP)**: Use for MoE models to distribute experts across GPUs
* **Sequence Parallelism**: Typically enabled with TP to reduce activation memory

__When to Combine with State Sharding:__

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

__State sharding (FSDP2 / ZeRO):__

- **FSDP2**: PyTorch-native, well-integrated, suitable for many large-model training scenarios
- **ZeRO stages**: Progressive sharding from optimizer states (ZeRO-1) to full parameter sharding (ZeRO-3)
- **DeepSpeed extensions**: ZeRO-Offload (CPU), ZeRO-Infinity (NVMe), ZeRO++ (multi-node communication) for scenarios where GPU-only sharding is insufficient
- **Key insight**: These techniques eliminate memory redundancy but assume each layer can be computed on a single GPU

__Computation sharding (Megatron):__

- **Tensor Parallelism (TP)**: Splits large matrix operations across GPUs when individual layers exceed single-GPU limits
- **Pipeline Parallelism (PP)**: Shards model depth across GPUs/nodes for very deep models, with virtual pipeline support
- **Context Parallelism (CP)**: Advanced long-context solution that partitions sequences, eliminating recompute overhead
- **Expert Parallelism (EP)**: Specialized parallelism for MoE models, achieving 468 TFLOPS for Mixtral 8X7B
- **Sequence Parallelism**: Splits activations along sequence dimension, essential when TP is enabled
- **Key insight**: Megatron addresses a fundamentally different problem—computation itself, not just memory

__Hybrid parallelism:__

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

__Decision framework:__
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




