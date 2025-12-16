# Chapter 5 — Beyond State Sharding with DeepSpeed and Megatron

In the previous chapter, we covered PyTorch FSDP2, which shards parameters, gradients, and optimizer states across GPUs to enable training of models larger than what fits on a single GPU. As a PyTorch-native solution, FSDP2 integrates well with the PyTorch ecosystem and is sufficient for the majority of large-scale training workloads.

However, there are scenarios where GPU-only state sharding is still not enough. A model may exceed the aggregate GPU memory budget even with full sharding, or the available GPU count may be limited. In other cases, practitioners may want to leverage CPU memory or NVMe storage to extend the effective memory capacity, accepting reduced throughput in exchange for feasibility.

DeepSpeed's ZeRO (Zero Redundancy Optimizer) addresses these memory-centric edge cases. Like FSDP2, ZeRO-3 shards parameters, gradients, and optimizer states across GPUs. In addition, DeepSpeed provides ZeRO-Offload to CPU memory, ZeRO-Infinity to NVMe storage, and ZeRO++ for communication and scheduling optimizations in large, multi-node environments. These features extend the memory hierarchy beyond GPUs and offer practical solutions when GPU-only approaches are insufficient.

Yet memory is not the only bottleneck. As model sizes continue to grow, a different limitation emerges: individual layers themselves may become too large or too expensive to compute efficiently on a single GPU, even if their parameters are fully sharded. This is where computation parallelism becomes necessary.

Megatron-LM addresses this class of problems by introducing tensor parallelism and pipeline parallelism, which shard the computation of individual layers and the model depth itself across multiple GPUs. Rather than focusing on reducing memory redundancy, Megatron directly partitions large matrix multiplications and attention operations, enabling training of models whose per-layer computation would otherwise exceed single-GPU limits. This approach is particularly critical for very large Transformer-based language models.

This chapter therefore covers two complementary families of techniques. We first examine the evolution of DeepSpeed ZeRO from ZeRO-1 through ZeRO-3, along with advanced variants such as ZeRO-Infinity and ZeRO++. We then introduce Megatron's computation parallelism strategies, including tensor and pipeline parallelism, and discuss how these approaches are often combined with ZeRO or FSDP-style state sharding in practice. Finally, we provide practical guidance on when to choose FSDP2 alone, when DeepSpeed is the right tool, and when Megatron-style parallelism becomes unavoidable.

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
─────────────────────────────────────────────
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
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-4
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

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
  },
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16
  }
}
```

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
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

**Key parameters:**
- `stage3_prefetch_bucket_size`: How much to prefetch (larger = more overlap, more memory)
- `stage3_max_live_parameters`: Max params kept unsharded (controls memory peak)
- `stage3_gather_16bit_weights_on_model_save`: Whether to gather full model for checkpointing

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
     ↕ PCIe Gen4: ~32 GB/s
CPU RAM:      Medium (100 GB/s), Cheaper, Medium (512 GB)
     ↕ NVMe: ~7 GB/s
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

ZeRO++ (2023) reduces ZeRO-3's communication overhead through three techniques: quantized weights (qwZ), hierarchical partitioning (hpZ), and quantized gradients (qgZ).

### The Communication Problem in ZeRO-3

For a 175B parameter model with 4 GPUs, each iteration:

```
Forward:  AllGather 175B params → 175 GB × 2 bytes = 350 GB total traffic
Backward: AllGather 175B params → 350 GB
          ReduceScatter 175B grads → 350 GB
──────────────────────────────────────────────────
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

## Choosing the Right ZeRO Stage

### Decision Tree

```
Model Size < 10B?
├─ Yes → Use DDP or ZeRO-1
│         (simplest, fastest, easiest to debug)
└─ No
    │
    Model Size < 50B?
    ├─ Yes → Use ZeRO-2 or FSDP2
    │         (good balance, shards gradients too)
    └─ No
        │
        Can single layer fit on one GPU?
        ├─ Yes → Use ZeRO-3 or FSDP2
        │         (state sharding is sufficient)
        └─ No → Use FSDP2 + Megatron TP (or ZeRO-3 + Megatron TP)
                (need computation sharding for large layers)
                │
                Model Size < 200B?
                ├─ Yes → FSDP2 + Megatron TP
                │         (or ZeRO-3 + Megatron TP)
                └─ No
                    │
                    Multiple Nodes?
                    ├─ Yes → FSDP2 + Megatron TP + PP + ZeRO++
                    │         (hierarchical parallelism)
                    └─ No
                        │
                        Model Size < 500B?
                        ├─ Yes → FSDP2 + Megatron TP
                        └─ No → ZeRO-Infinity + Megatron TP
                                (offload to NVMe)
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
| **FSDP2 + Megatron TP** | Shard | Shard | Shard | 1/(N×TP)× | Both patterns | 50B-200B+ models with large layers |

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

## Complete Training Example

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

## ZeRO vs FSDP: When to Use Which?

Both ZeRO-3 and PyTorch FSDP do parameter sharding (state sharding). They solve the same problem—distributing model parameters, gradients, and optimizer states across GPUs to reduce memory usage. In practice, **FSDP2 should be considered the default solution** for large-model training, while DeepSpeed is most valuable in memory-constrained or extreme-scale scenarios where GPU-only approaches are no longer sufficient.

### Use PyTorch FSDP2 (Default Choice):

- **Native PyTorch integration**: Pure PyTorch, no external dependencies
- **torch.compile support**: FSDP2 works better with PyTorch's compiler
- **Simpler codebase**: FSDP2 is more lightweight and easier to debug
- **Most large models** (7B-200B): FSDP2 is mature and well-optimized for this range
- **GPU-only sharding is sufficient**: When your model fits with full parameter sharding across available GPUs

### Use DeepSpeed ZeRO (Exception Cases):

- **GPU memory is insufficient even with full sharding**: When aggregate GPU memory across all devices is still not enough
- **Need CPU or NVMe offloading**: ZeRO-Offload (CPU) and ZeRO-Infinity (NVMe) extend memory beyond GPUs, though with significant throughput tradeoffs
- **Extreme-scale multi-node training**: ZeRO++ provides communication optimizations that can help in very large, heterogeneous network environments
- **Existing DeepSpeed ecosystem**: If you're already using other DeepSpeed features (MoE, compression, etc.)

**Important caveat**: CPU and NVMe offloading come with substantial throughput penalties. These are "feasibility" solutions—they enable training that wouldn't otherwise be possible, but at the cost of slower training speeds. Use them only when GPU-only approaches are truly insufficient.

### Hybrid Approach

You can even combine them:
```python
# Use FSDP for model sharding, DeepSpeed for other features
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import deepspeed

# This is possible but requires careful setup
```

## FSDP/ZeRO vs Megatron: Understanding the Fundamental Difference

The discussion so far has focused on FSDP and ZeRO, which both solve the **state sharding** problem—how to distribute model parameters, gradients, and optimizer states across GPUs to reduce memory usage. But there's another critical dimension: **computation sharding**. This is where Megatron comes in.

### The Core Distinction: State Sharding vs Computation Sharding

**FSDP2 and ZeRO solve:** "The model state is too large to fit in GPU memory"
- They shard parameters, gradients, and optimizer states
- Each GPU still computes entire layers independently
- Communication happens via all-gather (to get full parameters) and reduce-scatter (to aggregate gradients)

**Megatron solves:** "A single layer's computation is too large or too slow for one GPU"
- It shards the computation itself—splits weight matrices and attention across GPUs
- A single forward/backward pass requires multiple GPUs working together
- Communication happens every layer (all-gather/reduce for tensor parallelism)

This is a **fundamental difference**. FSDP2 assumes each layer can be computed on a single GPU—it just all-gathers parameters when needed, computes, then frees them. Megatron assumes a single layer needs multiple GPUs to compute efficiently.

### When You Need Megatron

Megatron's Tensor Parallelism (TP) becomes necessary when:

1. **Very large hidden dimensions**: Models with hidden_size=16384, 24576, or larger. A single linear layer's weight matrix might be 16384×65536, which is too large for efficient computation on one GPU.

2. **MoE expert layers**: In Mixture-of-Experts models, individual expert networks can be very large. Tensor parallelism distributes expert computation across GPUs.

3. **FP8 and extreme precision**: When using FP8 quantization with very large GEMM operations, single-GPU compute or SRAM might be insufficient.

4. **Ultra-scale training**: When scaling to 256, 512, or 1024 GPUs, you need hierarchical parallelism—tensor parallelism within nodes, pipeline parallelism across nodes.

5. **Sequence parallelism**: For very long sequences, Megatron's sequence parallelism splits attention computation along the sequence dimension, which FSDP/ZeRO don't address.

### The Reality: FSDP2 + Megatron is the Mainstream

In practice, the most common setup for large models (50B-200B parameters) is:

**FSDP2 + Megatron Tensor Parallelism**

- **FSDP2**: Handles state sharding (parameters, gradients, optimizer states) across all GPUs
- **Megatron TP**: Handles computation sharding for large layers within each FSDP group

This combination gives you:
- Memory efficiency from FSDP2 (state sharding)
- Computational efficiency from Megatron (computation sharding for large layers)
- The ability to train models where individual layers are too large for single GPUs

### Example: 70B Parameter Model Training

For a 70B parameter model with hidden_size=8192:

**Option 1: FSDP2-only**
- Works if each transformer layer fits on a single GPU
- FSDP2 shards parameters across GPUs, all-gathers when needed
- Simple to use, good for models up to ~30-50B

**Option 2: FSDP2 + Megatron TP**
- Use Megatron TP=2 or TP=4 within each FSDP group
- Each TP group handles large attention/MLP layers
- FSDP2 handles state sharding across TP groups
- Required for models 50B+ or when layers are too large

**Option 3: DeepSpeed ZeRO-3 + Megatron TP**
- Similar to Option 2, but using ZeRO-3 instead of FSDP2
- Still common in legacy codebases or when you need DeepSpeed-specific features

### Decision Framework

```
Can a single transformer layer fit and compute efficiently on one GPU?
├─ Yes → Use FSDP2-only (or ZeRO-3)
│         Models: 7B-30B, standard architectures
│
└─ No → Use FSDP2 + Megatron TP (or ZeRO-3 + Megatron TP)
        Models: 50B-200B+, very large hidden dims, MoE
        Megatron TP handles computation sharding
        FSDP/ZeRO handles state sharding
```

### Why Megatron Remains Essential

Even with FSDP2, Megatron is not obsolete because:

1. **Different problem domain**: FSDP2 solves state sharding; Megatron solves computation sharding. They address orthogonal concerns.

2. **Computation sharding is unavoidable**: When a single layer's matrix multiplication is too large for one GPU, you must split the computation. FSDP2 doesn't do this—it only shards state.

3. **Industry standard**: Most production training of 100B+ models uses Megatron TP. It's battle-tested, well-optimized, and integrates with FSDP/ZeRO.

4. **Sequence parallelism**: Megatron provides sequence parallelism for long-context training, which FSDP/ZeRO don't address.

### Practical Recommendation (2025 Perspective)

**For new projects:**

- **7B-30B models**: Use FSDP2-only. Simple, PyTorch-native, well-supported.
- **50B-200B models**: Use FSDP2 + Megatron TP. FSDP2 for state sharding, Megatron for computation sharding of large layers.
- **200B+ models**: Use FSDP2 + Megatron TP + Pipeline Parallelism. Full hierarchical parallelism.

**For existing projects:**

- If already using DeepSpeed ZeRO-3 + Megatron: Continue using it. It works well.
- If starting fresh: Prefer FSDP2 + Megatron over ZeRO-3 + Megatron for better PyTorch integration.

**Bottom line**: FSDP2 replaces ZeRO for state sharding, but Megatron remains essential for computation sharding. They solve different problems and are complementary, not competing.

## Summary and Key Takeaways

**ZeRO stages are progressive:**
- ZeRO-1: Shard optimizer states (easiest, 2× memory savings)
- ZeRO-2: + Shard gradients (3× memory savings)
- ZeRO-3: + Shard parameters (N× memory savings, linear scaling)

**Advanced variants address specific bottlenecks:**
- ZeRO-Offload: Limited GPU memory → Use CPU RAM
- ZeRO-Infinity: Extreme scale → Use NVMe
- ZeRO++: Multi-node → Optimize communication

**Practical advice:**
1. **Default choice**: Start with FSDP2 for most large-model training scenarios
2. **When GPU-only sharding isn't enough**: Consider DeepSpeed ZeRO-3 for equivalent functionality, or ZeRO-Offload/Infinity for CPU/NVMe offloading (with throughput tradeoffs)
3. **For extreme-scale multi-node**: ZeRO++ can provide communication optimizations in specific large-scale, heterogeneous environments
4. **When individual layers are too large**: Megatron Tensor Parallelism becomes necessary
5. **For very large models (50B+)**: Combine FSDP2 + Megatron TP for both state and computation sharding

**Understanding FSDP/ZeRO vs Megatron:**
- **FSDP2/ZeRO**: Solve state sharding (parameters, gradients, optimizer states) - these are equivalent approaches
- **Megatron**: Solves computation sharding (splitting layer computation across GPUs) - addresses a fundamentally different problem
- **They are complementary**: Use FSDP2 + Megatron TP for 50B+ models where both memory and computation need to be sharded
- **Key insight**: FSDP2 assumes each layer fits on one GPU; Megatron handles layers that don't

**Choosing the right approach:**
- **FSDP2 is the default**: Use it when GPU-only state sharding is sufficient (most cases)
- **DeepSpeed is for edge cases**: Use when you need CPU/NVMe offloading or are already in the DeepSpeed ecosystem
- **Megatron is for computation limits**: Use when individual layers exceed single-GPU computation capacity
- **Hybrid is common**: FSDP2 + Megatron TP is the mainstream approach for very large models (50B-200B+)

So far, we've focused on distributed training—how to train large models across multiple GPUs. But training is only half the story. Once you've trained a model, you need to serve it efficiently at scale. The next part of this book shifts focus to distributed inference: how to run inference on large models efficiently, handle high-throughput workloads, and serve models in production. We'll start with vLLM, a high-performance inference engine that uses techniques like PagedAttention and continuous batching to maximize throughput and minimize latency.



## References

- [ZeRO Paper (2020)](https://arxiv.org/abs/1910.02054): Original ZeRO optimization
- [ZeRO-Offload Paper (2021)](https://arxiv.org/abs/2101.06840): CPU offloading techniques  
- [ZeRO-Infinity Paper (2021)](https://arxiv.org/abs/2104.07857): NVMe offloading
- [ZeRO++ Paper (2023)](https://arxiv.org/abs/2306.10209): Communication-optimized ZeRO
- [DeepSpeed Documentation](https://www.deepspeed.ai/): Official docs and tutorials
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed): Source code and examples
- [Megatron-LM Paper (2019)](https://arxiv.org/abs/1909.08053): Tensor parallelism for large language models
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM): Source code and examples
