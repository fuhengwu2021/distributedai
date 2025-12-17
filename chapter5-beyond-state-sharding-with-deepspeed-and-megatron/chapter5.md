# Chapter 5 — Beyond State Sharding with DeepSpeed and Megatron

In the previous chapter, we covered PyTorch FSDP2, which shards parameters, gradients, and optimizer states across GPUs to enable training of models larger than what fits on a single GPU. As a PyTorch-native solution, FSDP2 integrates well with the PyTorch ecosystem and is sufficient for the majority of large-scale training workloads.

However, there are scenarios where GPU-only state sharding is still not enough. A model may exceed the aggregate GPU memory budget even with full sharding, or the available GPU count may be limited. In other cases, practitioners may want to leverage CPU memory or NVMe storage to extend the effective memory capacity, accepting reduced throughput in exchange for feasibility.

DeepSpeed's ZeRO (Zero Redundancy Optimizer) addresses these memory-constrained scenarios. Like FSDP2, ZeRO-3 shards parameters, gradients, and optimizer states across GPUs. In addition, DeepSpeed provides ZeRO-Offload to CPU memory, ZeRO-Infinity to NVMe storage, and ZeRO++ for communication and scheduling optimizations in large, multi-node environments. These features extend the memory hierarchy beyond GPUs and offer practical solutions when GPU-only approaches are insufficient.

Yet memory is not the only bottleneck. As model sizes continue to grow, a different limitation emerges: individual layers themselves may become too large or too expensive to compute efficiently on a single GPU, even if their parameters are fully sharded. This is where computation parallelism becomes necessary.

Megatron-LM addresses this class of problems by introducing tensor parallelism and pipeline parallelism, which shard the computation of individual layers and the model depth itself across multiple GPUs. Rather than focusing on reducing memory redundancy, Megatron directly partitions large matrix multiplications and attention operations, enabling training of models whose per-layer computation would otherwise exceed single-GPU limits. This approach is important for very large Transformer-based language models.

This chapter covers two complementary families of techniques that address different bottlenecks. We first examine when state sharding alone is insufficient, then introduce DeepSpeed ZeRO as a memory extension toolbox that can leverage CPU and NVMe resources. The core of this chapter focuses on Megatron's computation parallelism strategies—tensor parallelism, pipeline parallelism, and sequence parallelism—which address a fundamentally different problem: sharding computation itself when individual layers exceed single-GPU limits. We then explore how these techniques are combined in practice through hybrid parallelism, which is commonly used for training very large models. Finally, we provide practical guidance on choosing the right combination of techniques.

**Practical guidance**: When training large models, practitioners often begin with state sharding techniques like FSDP2 or ZeRO-3, then add Megatron-style computation parallelism when per-layer computation becomes the bottleneck. DeepSpeed ZeRO offers additional capabilities for CPU and NVMe offloading, which can be valuable when GPU memory is constrained. Common patterns include FSDP2 + Megatron Tensor Parallelism, as well as ZeRO-3 + Megatron for scenarios where DeepSpeed's ecosystem is already in use.

## Understanding the Memory Problem

Before diving into ZeRO stages, let's understand what consumes memory during training.

### Memory Breakdown

Training a model with the Adam optimizer requires storing:

1. **Model Parameters**: The weights themselves (2 bytes per param in fp16)
2. **Gradients**: Same size as parameters (2 bytes per param in fp16)
3. **Optimizer States**: For Adam, two states per parameter:
   - Momentum (first moment): 4 bytes per param (fp32)
   - Variance (second moment): 4 bytes per param (fp32)
4. **Activations**: Forward pass activations saved for backward pass

For a 175B parameter model (GPT-3 scale) trained with Adam in mixed precision:

```
Parameters:        175B × 2 bytes  = 350 GB
Gradients:         175B × 2 bytes  = 350 GB
Optimizer States:  175B × 8 bytes  = 1,400 GB
-----------------------------------------------------------------------------
Total (per GPU):                     2,100 GB
```

An 80GB A100 GPU can't hold this. Even with 8 GPUs using traditional data parallel (DDP), each GPU still needs the full 2,100 GB because DDP replicates everything.

### The Data Parallel Problem

Traditional DDP (DistributedDataParallel) replicates the entire model on every GPU:

```
GPU 0: [params_full] [grads_full] [optimizer_states_full]  2,100 GB
GPU 1: [params_full] [grads_full] [optimizer_states_full]  2,100 GB
GPU 2: [params_full] [grads_full] [optimizer_states_full]  2,100 GB
GPU 3: [params_full] [grads_full] [optimizer_states_full]  2,100 GB
```

This is redundant. If we have N GPUs, we're storing N copies of everything. ZeRO eliminates this redundancy.

## ZeRO Stage 1: Optimizer State Partitioning

ZeRO-1 shards optimizer states across GPUs. This is the low-hanging fruit because optimizer states dominate memory usage.

### How It Works

Each GPU stores only 1/N of the optimizer states:

```
Before (DDP):
GPU 0: optimizer_states[all 175B params] = 1,400 GB
GPU 1: optimizer_states[all 175B params] = 1,400 GB
GPU 2: optimizer_states[all 175B params] = 1,400 GB
GPU 3: optimizer_states[all 175B params] = 1,400 GB

After (ZeRO-1):
GPU 0: optimizer_states[0:44B]   = 350 GB
GPU 1: optimizer_states[44B:88B] = 350 GB
GPU 2: optimizer_states[88B:132B]= 350 GB
GPU 3: optimizer_states[132B:175B]= 350 GB
```

Each GPU still holds full parameters and gradients, but optimizer states are partitioned.

### Training Flow

1. **Forward/Backward**: Normal DDP behavior - all GPUs have full parameters
2. **Gradient All-Reduce**: Standard DDP all-reduce to synchronize gradients
3. **Optimizer Step**: 
   - Each GPU updates only its partition of parameters
   - No extra communication needed
   - Parameters are implicitly partitioned during update

### Memory Savings

For a model with Adam optimizer:

- Parameters: No change (still replicated)
- Gradients: No change (still replicated)  
- Optimizer States: **Reduced by N×**

Total memory per GPU:
```
350 GB (params) + 350 GB (grads) + 350 GB (opt/4 GPUs) = 1,050 GB
Savings: 2,100 GB → 1,050 GB (2× reduction)
```

### When to Use ZeRO-1

- Model fits in GPU memory but optimizer states don't
- Want minimal changes to DDP training loop
- Training models up to ~10B parameters
- Debugging is easier (closest to standard DDP)

### DeepSpeed Config

```json
{
  "zero_optimization": {
    "stage": 1
  }
}
```

The configuration is straightforward—simply set `stage: 1` to enable optimizer state partitioning. Other optimizer and training settings (learning rate, precision, etc.) are configured separately.

## ZeRO Stage 2: Optimizer State + Gradient Partitioning

ZeRO-2 extends ZeRO-1 by also sharding gradients. This targets models that are larger but still manageable.

### How It Works

During backward pass, gradients are reduced and partitioned on-the-fly:

```
GPU 0: params[all] + grads[0:44B]   + opt_states[0:44B]
GPU 1: params[all] + grads[44B:88B] + opt_states[44B:88B]
GPU 2: params[all] + grads[88B:132B]+ opt_states[88B:132B]
GPU 3: params[all] + grads[132B:175B]+ opt_states[132B:175B]
```

### Training Flow

1. **Forward**: All GPUs have full parameters
2. **Backward**: 
   - Compute gradients layer-by-layer
   - As each layer's gradient is computed, immediately `reduce_scatter` it
   - Each GPU keeps only its assigned partition
   - Free the temporary full gradient
3. **Optimizer Step**: Update only the local partition (same as ZeRO-1)

### Reduce-Scatter Operation

Instead of `all_reduce` (which gives everyone the full averaged gradient), we use `reduce_scatter`:

```python
# All-Reduce (DDP): Everyone gets everything
# Input:  GPU0: [g0], GPU1: [g1], GPU2: [g2], GPU3: [g3]
# Output: GPU0: [avg(g)], GPU1: [avg(g)], GPU2: [avg(g)], GPU3: [avg(g)]

# Reduce-Scatter (ZeRO-2): Everyone gets their partition
# Input:  GPU0: [g0], GPU1: [g1], GPU2: [g2], GPU3: [g3]
# Output: GPU0: [avg(g)[0:N/4]], GPU1: [avg(g)[N/4:N/2]], ...
```

This is more efficient: same communication volume as all-reduce, but each GPU stores less.

### Gradient Bucketing

To amortize communication overhead, gradients are bucketed:

```python
# Bad: Reduce-scatter after every layer
for layer in reversed(layers):
    grad = compute_grad(layer)
    reduce_scatter(grad)  # Many small communications

# Good: Accumulate in bucket, then reduce-scatter
bucket = []
for layer in reversed(layers):
    grad = compute_grad(layer)
    bucket.append(grad)
    if len(bucket) >= BUCKET_SIZE:
        reduce_scatter(concat(bucket))
        bucket = []
```

Default bucket size is 5e8 elements (500M parameters worth).

### Memory Savings

```
350 GB (params) + 88 GB (grads/4) + 350 GB (opt/4) = 788 GB
Savings: 2,100 GB → 788 GB (2.66× reduction)
```

### When to Use ZeRO-2

- Models in the 10B-50B parameter range
- Gradient memory is a bottleneck
- Acceptable communication overhead (reduce-scatter is cheap)
- Good balance between memory and complexity

### DeepSpeed Config

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8
  }
}
```

The key parameters are `allgather_bucket_size` and `reduce_bucket_size`, which control communication batching for gradients. Larger buckets improve communication efficiency but use more memory.

## ZeRO Stage 3: Full Sharding (Like FSDP)

ZeRO-3 shards everything: parameters, gradients, and optimizer states. This is equivalent to PyTorch FSDP.

### How It Works

Each GPU stores only 1/N of all states:

```
GPU 0: params[0:44B]   + grads[0:44B]   + opt[0:44B]     = 262 GB
GPU 1: params[44B:88B] + grads[44B:88B] + opt[44B:88B]   = 262 GB
GPU 2: params[88B:132B]+ grads[88B:132B]+ opt[88B:132B]  = 262 GB
GPU 3: params[132B:175B]+grads[132B:175B]+opt[132B:175B] = 262 GB
```

### Training Flow

**Forward Pass:**
1. For each layer:
   - `all_gather` the layer's parameters from all GPUs
   - Compute forward with full parameters
   - Free the full parameters (keep only local shard)

**Backward Pass:**
1. For each layer (in reverse):
   - `all_gather` the layer's parameters again
   - Compute gradients
   - `reduce_scatter` gradients (each GPU keeps its shard)
   - Free the full parameters

**Optimizer Step:**

- Each GPU updates only its parameter shard (local operation)

### Communication Pattern

```
Forward:
  Layer N:   AllGather(params_N) → Compute → Free(params_N)
  Layer N-1: AllGather(params_N-1) → Compute → Free(params_N-1)
  ...

Backward:
  Layer 1:   AllGather(params_1) → Compute grads → ReduceScatter(grads_1) → Free(params_1)
  Layer 2:   AllGather(params_2) → Compute grads → ReduceScatter(grads_2) → Free(params_2)
  ...
```

### Memory Savings

```
88 GB (params/4) + 88 GB (grads/4) + 350 GB (opt/4) = 526 GB
Savings: 2,100 GB → 526 GB (4× reduction)
```

With more GPUs, memory scales linearly: 8 GPUs → 263 GB per GPU, 16 GPUs → 131 GB per GPU.

### Communication Overhead

ZeRO-3 has higher communication than ZeRO-1/2:

- **Communication volume**: 3× the model size per iteration
  - Forward: 1× (all-gather params)
  - Backward: 2× (all-gather params + reduce-scatter grads)
- **Latency-sensitive**: Many small all-gathers can hurt performance

**Optimization: Communication/Computation Overlap**

DeepSpeed overlaps communication with computation:

```python
# While computing layer N, prefetch parameters for layer N+1
with ComputeStream():
    compute_layer_N()

with CommunicationStream():
    prefetch_layer_N_plus_1_params()  # Overlapped!
```

### When to Use ZeRO-3

- Models 50B+ parameters that don't fit in GPU memory
- Have many GPUs (8+) to amortize communication
- Fast interconnect (NVLink, InfiniBand)
- Can tolerate 10-20% slowdown vs ZeRO-2

### DeepSpeed Config

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_max_live_parameters": 1e9
  }
}
```

**Key parameters:**

- `overlap_comm`: Overlap communication with computation for better throughput
- `stage3_prefetch_bucket_size`: Controls prefetching for parameter all-gather (larger = more overlap, more memory)
- `stage3_max_live_parameters`: Maximum parameters kept unsharded at once (controls memory peak)

For most use cases, the default values work well. Tune these only when memory or communication becomes a bottleneck.

## ZeRO-Offload: CPU Memory Extension

ZeRO-Offload extends ZeRO-2 by offloading optimizer states to CPU memory. This is useful when GPU memory is limited but CPU memory is abundant.

### Architecture

```
GPU:  [Parameters] [Gradients] 
       ↓ (gradient updates via PCIe)
CPU:  [Optimizer States] [Optimizer Computation]
```

### How It Works

1. **Forward/Backward**: On GPU (fast)
2. **Gradient Computation**: On GPU (fast)
3. **Gradient Transfer**: GPU → CPU via PCIe (~32 GB/s)
4. **Optimizer Step**: On CPU (slower, but frees GPU memory)
5. **Parameter Update**: CPU → GPU via PCIe

### Performance Considerations

- **PCIe Bandwidth**: Bottleneck is ~32 GB/s (vs ~1.5 TB/s for GPU HBM)
- **CPU Compute**: Optimizer step is slower on CPU than GPU
- **Overlap**: Transfer gradients while computing next layer to hide latency

### Speedup Tricks

DeepSpeed overlaps CPU optimizer computation with GPU forward/backward:

```python
Step N:
  GPU: Forward/Backward → Produce gradients
  CPU: (simultaneously) Running optimizer step from step N-1

Step N+1:
  GPU: Forward/Backward → Produce gradients  
  CPU: Running optimizer step from step N
```

### When to Use ZeRO-Offload

- Training on consumer GPUs (e.g., RTX 3090, 4090) with limited VRAM
- Have plenty of CPU RAM (256GB+)
- Model size 10B-30B where GPU-only sharding isn't sufficient
- **Important**: Expect 20-40% throughput reduction compared to GPU-only training due to PCIe bandwidth and CPU compute limitations. This is a "feasibility" solution—it enables training that wouldn't otherwise be possible, but at the cost of slower training speeds.
- Can tolerate 20-30% slowdown vs full GPU training

### DeepSpeed Config

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4
    }
  }
}
```

## ZeRO-Infinity: NVMe Offload for Massive Models

ZeRO-Infinity extends ZeRO-3 with CPU and NVMe offloading, enabling training of models with **trillions** of parameters.

### Memory Hierarchy

```
GPU VRAM:     Fast (1.5 TB/s), Expensive, Small (80 GB)
     <->
 PCIe Gen4: ~32 GB/s
CPU RAM:      Medium (100 GB/s), Cheaper, Medium (512 GB)
     <->
 NVMe: ~7 GB/s
NVMe SSD:     Slow (7 GB/s), Cheap, Large (4 TB+)
```

### What Gets Offloaded

**Typical configuration for 1T+ parameter model:**

1. **GPU**: Active layer parameters + gradients + activations
2. **CPU**: Optimizer states + inactive parameters
3. **NVMe**: Cold parameters + checkpoints

### Infinity Engine

DeepSpeed's Infinity Engine manages data movement across the hierarchy:

**Key features:**

- **Prefetching**: Loads parameters from NVMe → CPU → GPU before they're needed
- **Overlap**: Data movement overlaps with computation
- **Smart caching**: Keeps frequently-used parameters in faster memory

### Example: 1 Trillion Parameter Model

With 16× A100 80GB GPUs:

```
Traditional ZeRO-3:
  1T params × 2 bytes = 2 TB / 16 GPUs = 125 GB per GPU
  + optimizer states = 250 GB per GPU → Doesn't fit!

ZeRO-Infinity:
  GPU:  20 GB (active params + activations)
  CPU:  400 GB (optimizer states + param buffer)
  NVMe: 2 TB (cold parameters)
  Total: Works!
```

### Performance Trade-offs

- **Throughput**: 30-50% slower than pure GPU (due to PCIe/NVMe bandwidth)
- **Memory**: Scales to trillions of parameters
- **Cost**: Much cheaper than buying 10× more GPUs

### When to Use ZeRO-Infinity

- Models >500B parameters
- Limited GPU budget
- Have fast NVMe (PCIe Gen4, 7+ GB/s)
- Prototyping huge architectures
- Training is throughput-bound, not latency-critical

### DeepSpeed Config

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 1e8,
      "max_in_cpu": 1e9
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "stage3_max_live_parameters": 1e9,
    "stage3_prefetch_bucket_size": 5e8
  },
  "aio": {
    "block_size": 1048576,
    "queue_depth": 16,
    "thread_count": 2,
    "single_submit": false,
    "overlap_events": true
  }
}
```

**Important parameters:**

- `nvme_path`: Path to NVMe mount point
- `buffer_count` × `buffer_size`: Total CPU buffer for NVMe staging
- `aio` section: Async I/O tuning for NVMe performance

## ZeRO++: Communication-Optimized ZeRO

ZeRO++ reduces ZeRO-3's communication overhead through three techniques: quantized weights (qwZ), hierarchical partitioning (hpZ), and quantized gradients (qgZ).

### The Communication Problem in ZeRO-3

For a 175B parameter model with 4 GPUs, each iteration:

```
Forward:  AllGather 175B params → 175 GB × 2 bytes = 350 GB total traffic
Backward: AllGather 175B params → 350 GB
          ReduceScatter 175B grads → 350 GB
-------------------------------------------------------------------------------
Total: 1,050 GB per iteration

At 400 GB/s InfiniBand → 2.6 seconds just for communication!
```

### qwZ: Quantized Weight Communication

**Idea**: All-gather parameters in low precision (int8 or fp8), convert to fp16/bf16 after receiving.

```python
# Without qwZ
GPU 0: send params[shard_0] as fp16 → 88 GB
GPU 1: send params[shard_1] as fp16 → 88 GB
...

# With qwZ  
GPU 0: quantize to int8 → send → dequantize to fp16 → 22 GB (4× reduction)
GPU 1: quantize to int8 → send → dequantize to fp16 → 22 GB
...
```

**Communication savings**: 4× for all-gather (fp16 → int8 reduces by half per direction)

**Accuracy**: Surprisingly minimal impact! Quantization error is small and doesn't accumulate (parameters are re-quantized each time).

### hpZ: Hierarchical Partitioning (HSDP)

**Problem**: Inter-node communication (InfiniBand) is much slower than intra-node (NVLink).

```
NVLink (intra-node):   600 GB/s
InfiniBand (inter-node): 400 GB/s  

Multi-node ZeRO-3: Treats all GPUs equally → lots of slow inter-node traffic
```

**Solution**: Hybrid sharding
- **Intra-node**: Replicate (everyone in the node has same shard)
- **Inter-node**: Shard (different nodes have different shards)

```
Example: 2 nodes, 4 GPUs per node, 175B param model

Traditional ZeRO-3:
  Node 0: GPU0[0:22B], GPU1[22:44B], GPU2[44:66B], GPU3[66:88B]
  Node 1: GPU4[88:110B], GPU5[110:132B], GPU6[132:154B], GPU7[154:175B]
  
  Forward: Each GPU all-gathers from all 8 GPUs (lots of inter-node!)

hpZ (HSDP):
  Node 0: GPU0,1,2,3 all have [0:88B]  (replicated within node)
  Node 1: GPU4,5,6,7 all have [88:175B]
  
  Forward: 
    - Intra-node: AllGather [0:88B] via NVLink (fast!)
    - Inter-node: Only 1 GPU per node exchanges via InfiniBand
    - Much less inter-node traffic!
```

**Communication savings**: 
- Intra-node: Uses fast NVLink (no change)
- Inter-node: Reduces from N×GPUs to N×nodes (typically 4-8× reduction)

**Trade-off**: Uses more memory (2× vs full sharding) but much faster for multi-node.

### qgZ: Quantized Gradient Communication

Similar to qwZ but for gradients during reduce-scatter:

```python
# Reduce-scatter with quantization
gradients_fp16 → quantize to int8 → reduce_scatter → dequantize → fp16
```

**Key difference from qwZ**: 
- Quantization happens *before* reduction
- Needs careful handling with gradient clipping

### ZeRO++ Performance

For 175B model on 64 GPUs (8 nodes × 8 GPUs):

```
              Communication    Throughput
ZeRO-3:       1,050 GB/iter   100%
ZeRO++ (qwZ): 525 GB/iter     140%  (quantized weights)
ZeRO++ (hpZ): 350 GB/iter     180%  (hierarchical partition)
ZeRO++ (all): 175 GB/iter     220%  (qwZ + hpZ + qgZ)
```

### When to Use ZeRO++

- **Large-scale multi-node training** (8+ nodes) where inter-node communication becomes a bottleneck
- **Heterogeneous network environments** where intra-node (NVLink) is much faster than inter-node (InfiniBand)
- Large models (50B+) where communication overhead dominates training time
- Have sufficient memory for 2× replication within nodes (hpZ trade-off)
- **Note**: Modern NCCL and PyTorch DDP/FSDP already provide significant communication optimizations. ZeRO++ is most valuable in extreme-scale, multi-node scenarios where these optimizations are insufficient.

### DeepSpeed Config

```json
{
  "zero_optimization": {
    "stage": 3,
    "zero_quantized_weights": true,
    "zero_hpz_partition_size": 8,
    "zero_quantized_gradients": true
  },
  "communication_data_type": "fp16",
  "fp16": {
    "enabled": true
  }
}
```

**Parameters:**

- `zero_quantized_weights`: Enable qwZ
- `zero_hpz_partition_size`: GPUs per replica group (= GPUs per node for HSDP)
- `zero_quantized_gradients`: Enable qgZ

## Megatron: Computation Parallelism as the Second Axis

So far, we have focused on **state sharding**—how to distribute parameters, gradients, and optimizer states across GPUs to reduce memory footprint. Techniques such as FSDP2 and DeepSpeed ZeRO fundamentally address a *memory redundancy* problem: eliminating replicated model state so that larger models can fit within the aggregate GPU memory budget.

However, state sharding alone is not sufficient for the largest models. As model sizes continue to grow, a second, orthogonal limitation emerges: **computation itself becomes too large to execute efficiently on a single GPU**, even when memory is fully sharded. This is where Megatron enters the picture.

### State Sharding vs. Computation Sharding

The key distinction can be summarized as follows:

* **State sharding (FSDP2 / ZeRO)** answers the question:
  *"How do we store the model state across GPUs?"*

* **Computation sharding (Megatron)** answers a different question:
  *"How do we execute a single layer when its computation no longer fits on one GPU?"*

FSDP2 assumes that each Transformer layer—its attention and MLP blocks—can be computed independently on a single GPU once the parameters for that layer are all-gathered. This assumption holds for many models up to tens of billions of parameters. When it breaks, no amount of additional state sharding can help.

Megatron addresses precisely this failure mode by **sharding the computation itself**.

### Tensor Parallelism: Sharding the Layer

Megatron's core contribution is **Tensor Parallelism (TP)**. Instead of replicating a layer's computation on every GPU, tensor parallelism splits large matrix operations across multiple GPUs.

Consider a Transformer MLP layer with a weight matrix $W \in \mathbb{R}^{d \times 4d}$. For modern large language models, $d$ may be 16,384 or larger. The resulting matrix multiplication is both memory-intensive and compute-heavy.

**How Tensor Parallelism Works:**

1. **Column Parallel Linear**: Splits weight matrix column-wise
   * Input: Full input tensor (replicated)
   * Weight: Each GPU holds 1/TP of columns
   * Output: Partial output (split along last dimension)
   * Communication: None (output stays split)

2. **Row Parallel Linear**: Splits weight matrix row-wise
   * Input: Already split (from column parallel)
   * Weight: Each GPU holds 1/TP of rows
   * Output: Partial output that needs gathering
   * Communication: All-reduce to gather full output

**Attention with Tensor Parallelism:**

For self-attention, Megatron splits Q, K, V projections:
* Q, K, V are computed in parallel across TP ranks
* Attention scores are computed locally
* Output projection uses row-parallel linear
* All-reduce gathers final attention output

**Sequence Parallelism with TP:**

When TP is enabled, sequence parallelism further reduces activation memory:
* Splits activations along sequence dimension
* Reduces activation memory by TP times
* Essential for long-context training

**Configuration:**

```bash
--tensor-model-parallel-size 4    # 4-way tensor parallelism
--sequence-parallel                # Enable sequence parallelism (recommended)
--tp-comm-overlap                 # Overlap TP communication with computation
```

**Benefits:**

* Larger hidden dimensions (16K, 32K+)
* Better utilization of GPU compute resources
* Scaling beyond what single-GPU kernels can efficiently handle
* Reduced activation memory with sequence parallelism

**Trade-offs:**

* **Increased communication**: Communication happens at every layer (all-reduce/all-gather)
* **Communication overhead**: Must overlap communication with computation for efficiency
* **Topology sensitivity**: Best performance when TP groups are within NVLink domain

Importantly, **tensor parallelism increases communication frequency**—communication now happens at every layer. This is fundamentally different from state sharding, where communication is amortized across layers. However, with proper overlap (`--tp-comm-overlap`), this overhead can be largely hidden.

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

Here's a complete example training a large model with Megatron Tensor Parallelism using Megatron Core:

```python
# code/train_megatron_mcore.py
import os
import torch
from torch.optim import Adam
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads

def initialize_distributed(tensor_model_parallel_size=4, pipeline_model_parallel_size=1):
    """Initialize torch.distributed and Megatron-Core model parallel groups."""
    parallel_state.destroy_model_parallel()
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    # Initialize Megatron model parallelism
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )

def model_provider():
    """Build and return a GPT model using Megatron Core."""
    transformer_config = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
    )
    
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=50257,
        max_sequence_length=2048,
    )
    
    return gpt_model

def forward_step_func(data_iterator, model):
    """Forward step function for training."""
    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm loss": loss}
    
    data = next(data_iterator)
    tokens = data["tokens"].cuda()
    attention_mask = data["attention_mask"].cuda()
    position_ids = data["position_ids"].cuda()
    labels = data["labels"].cuda()
    loss_mask = data["loss_mask"].cuda()
    
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    
    return output_tensor, lambda: loss_func(loss_mask, output_tensor)

if __name__ == "__main__":
    # Initialize distributed training
    initialize_distributed(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    
    # Create model
    gpt_model = model_provider()
    gpt_model.cuda()
    
    # Wrap with DistributedDataParallel
    config = gpt_model.config
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
    )
    gpt_model = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=gpt_model,
    )
    
    # Optimizer
    optim = Adam(gpt_model.parameters(), lr=1e-4)
    
    # Get forward/backward function
    forward_backward_func = get_forward_backward_func()
    
    # Training loop
    for iteration in range(100):
        optim.zero_grad()
        
        # Forward and backward pass
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=2048,
            micro_batch_size=8,
            decoder_seq_length=2048,
            forward_only=False,
        )
        
        # Finalize gradients
        finalize_model_grads([gpt_model])
        
        optim.step()
        
        if iteration % 10 == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
            print(f"Iteration {iteration}: Losses: {losses_reduced}")
```

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

This section provides complete, production-ready training examples for both ZeRO and Megatron, based on real-world configurations used in large-scale model training.

### Example 1: Training with DeepSpeed ZeRO-3

Here's a complete example training a large model with ZeRO-3:

### Model Code

```python
# code/train_zero3.py
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset, DataLoader

class LargeTransformer(nn.Module):
    def __init__(self, vocab_size=50257, dim=4096, n_layers=32, n_heads=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=dim*4)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.head(x)

class DummyDataset(Dataset):
    def __init__(self, size=10000, seq_len=512, vocab_size=50257):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x, x  # Use same for input and target in this demo

def train():
    # Initialize DeepSpeed
    import argparse
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # Model
    model = LargeTransformer(
        vocab_size=50257,
        dim=4096,
        n_layers=32,
        n_heads=32
    )
    
    # Dataset
    dataset = DummyDataset(size=10000, seq_len=512)
    
    # Initialize DeepSpeed engine
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset
    )
    
    # Training loop
    for epoch in range(3):
        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(model_engine.device)
            targets = targets.to(model_engine.device)
            
            # Forward
            outputs = model_engine(inputs)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
            
            # Backward
            model_engine.backward(loss)
            
            # Optimizer step
            model_engine.step()
            
            if step % 10 == 0 and model_engine.local_rank == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        # Save checkpoint
        if model_engine.local_rank == 0:
            model_engine.save_checkpoint(f"./checkpoints", tag=f"epoch_{epoch}")

if __name__ == "__main__":
    train()
```

### DeepSpeed Config

```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 16,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 1000,
      "total_num_steps": 100000
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 4
  },
  "wall_clock_breakdown": true
}
```

### Launch Script

```bash
#!/bin/bash
# Single node (8 GPUs)
deepspeed --num_gpus=8 code/train_zero3.py \
  --deepspeed \
  --deepspeed_config code/ds_config_zero3.json

# Multi-node (2 nodes, 8 GPUs each)
deepspeed --num_gpus=8 \
  --num_nodes=2 \
  --master_addr=node0 \
  --master_port=29500 \
  code/train_zero3.py \
  --deepspeed \
  --deepspeed_config code/ds_config_zero3.json
```

### Example 2: Training with Megatron Core (Production Configuration)

Here's a production-ready example using Megatron Core to train a large model with tensor parallelism, based on actual Megatron-LM training scripts:

**Training Script (`code/train_megatron_llama3_8b.sh`):**

```bash
#!/bin/bash

# LLaMA-3 8B training with Megatron Core
# Configuration: 8 GPUs, FP8 precision, context parallelism for long sequences

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

CHECKPOINT_PATH=${1:-"checkpoints/llama3_8b_fp8"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/llama3_8b_fp8"}
TOKENIZER_ARG=${3:-"MOCK"}
DATA_ARG=${4:-"MOCK"}

# Model parallelism configuration
TP_SIZE=1      # Tensor parallelism (1 = no TP, use CP instead)
CP_SIZE=2      # Context parallelism for 8K sequence
PP_SIZE=1      # Pipeline parallelism
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128

# Model architecture (LLaMA-3 8B)
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --seq-length 8192
    --max-position-embeddings 8192
    --position-embedding-type rope
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-backend fused
)

# Training hyperparameters
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr 0.00015
    --min-lr 0.00001
    --lr-decay-style cosine
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
)

# FP8 configuration (for Hopper/Ada/Blackwell GPUs)
FP8_ARGS=(
    --fp8-format hybrid
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max
    --fp8-param-gather
)

# Parallelism configuration
PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# Data configuration
if [[ "$TOKENIZER_ARG" == "MOCK" ]]; then
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 128256
    )
else
    DATA_ARGS=(
        --data-path $DATA_ARG
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model $TOKENIZER_ARG
        --vocab-size 128256
    )
fi

# Logging and checkpointing
LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1000
    --eval-interval 100
    --eval-iters 32
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --ckpt-format torch_dist
)

# Run training
torchrun --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${FP8_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```

**Key Features of This Configuration:**

1. **Context Parallelism**: Uses CP=2 to handle 8K sequence length efficiently
2. **FP8 Training**: Enables FP8 mixed precision for Hopper/Ada/Blackwell GPUs
3. **Distributed Optimizer**: Shards optimizer states across data-parallel ranks
4. **Communication Overlap**: Enables gradient reduction and parameter gathering overlap
5. **Production-Ready**: Based on actual Megatron-LM training scripts

**Running the Script:**

```bash
# Single node training
./code/train_megatron_llama3_8b.sh \
    checkpoints/llama3_8b \
    tensorboard_logs/llama3_8b \
    /path/to/tokenizer.model \
    /path/to/data_prefix

# Multi-node training (2 nodes, 8 GPUs each)
# On node 0:
MASTER_ADDR=node0 NODE_RANK=0 ./code/train_megatron_llama3_8b.sh ...

# On node 1:
MASTER_ADDR=node0 NODE_RANK=1 ./code/train_megatron_llama3_8b.sh ...
```

### Example 3: Megatron MoE Training (Mixtral 8x7B)

Here's a complete example for training a Mixture-of-Experts model using Megatron:

**Training Script (`code/train_megatron_mixtral.sh`):**

```bash
#!/bin/bash

# Mixtral 8x7B MoE training with Megatron
# Configuration: 64 GPUs (8 nodes × 8 GPUs), Expert Parallelism

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
NNODES=8
MASTER_ADDR=${MASTER_ADDR:-"node0"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NODE_RANK=${NODE_RANK:-"0"}

CHECKPOINT_PATH=$1
TOKENIZER_MODEL=$2
DATA_PATH=$3

# Model architecture (Mixtral 8x7B)
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 32768
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --group-query-attention
    --num-query-groups 8
)

# MoE configuration
MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
)

# Parallelism configuration
PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

# Training configuration
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

# Data configuration
DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
)

# Logging
LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --ckpt-format torch_dist
)

torchrun --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${LOGGING_ARGS[@]}
```

**Key Features:**

* **Expert Parallelism**: 8-way EP distributes 8 experts across GPUs
* **Pipeline Parallelism**: 4 pipeline stages for inter-node scaling
* **Token Routing**: All-to-all communication for efficient expert routing
* **Load Balancing**: Auxiliary loss for balanced token distribution
* **Performance**: Achieves 468 TFLOPS for Mixtral 8X7B training

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




