---
title: "DeepSpeed and ZeRO Optimization"
---

# Chapter 5 — DeepSpeed and ZeRO Optimization

When your model grows beyond what fits in GPU memory, you need to split it across devices. DeepSpeed's ZeRO (Zero Redundancy Optimizer) is a family of memory optimization techniques that progressively shard optimizer states, gradients, and parameters across GPUs. This chapter explains the evolution from ZeRO-1 through ZeRO-3, advanced variants like ZeRO-Infinity and ZeRO++, and practical guidance on which to use when.

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

- Training on consumer GPUs (e.g., RTX 3090, 4090)
- Have plenty of CPU RAM (256GB+)
- Model size 10B-30B
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

- **Multi-node training** (hpZ is critical here)
- Large models (50B+) where communication dominates
- Have enough memory for 2× replication within nodes
- Want maximum throughput

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
    ├─ Yes → Use ZeRO-2
    │         (good balance, shards gradients too)
    └─ No
        │
        Model Size < 200B?
        ├─ Yes → Use ZeRO-3 or FSDP
        │         (necessary for parameter sharding)
        └─ No
            │
            Multiple Nodes?
            ├─ Yes → Use ZeRO-3 + ZeRO++
            │         (optimize inter-node comm)
            └─ No
                │
                Model Size < 500B?
                ├─ Yes → Use ZeRO-3
                └─ No → Use ZeRO-Infinity
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
# code/chapter5/train_zero3.py
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
deepspeed --num_gpus=8 code/chapter5/train_zero3.py \
  --deepspeed \
  --deepspeed_config code/chapter5/ds_config_zero3.json

# Multi-node (2 nodes, 8 GPUs each)
deepspeed --num_gpus=8 \
  --num_nodes=2 \
  --master_addr=node0 \
  --master_port=29500 \
  code/chapter5/train_zero3.py \
  --deepspeed \
  --deepspeed_config code/chapter5/ds_config_zero3.json
```

## ZeRO vs FSDP: When to Use Which?

Both ZeRO-3 and PyTorch FSDP do parameter sharding. Here's when to use each:

### Use PyTorch FSDP When:

- **Native PyTorch integration**: Want pure PyTorch, no external dependencies
- **torch.compile support**: FSDP2 works better with compiler
- **Simpler codebase**: FSDP2 is ~3k lines vs ZeRO's larger codebase
- **Model < 100B**: FSDP is mature and well-optimized for this range

### Use DeepSpeed ZeRO When:

- **Very large models** (>100B): ZeRO has been battle-tested at scale
- **Need offloading**: ZeRO-Infinity's NVMe offload is more mature
- **Multi-node optimization**: ZeRO++ provides significant speedups
- **MoE models**: DeepSpeed has integrated MoE support
- **Existing DeepSpeed ecosystem**: Already using other DeepSpeed features

### Hybrid Approach

You can even combine them:
```python
# Use FSDP for model sharding, DeepSpeed for other features
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import deepspeed

# This is possible but requires careful setup
```

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
1. Start with ZeRO-1 or ZeRO-2 for debugging
2. Move to ZeRO-3 when model doesn't fit in memory
3. Add ZeRO++ for multi-node training
4. Use ZeRO-Infinity only for >500B parameter models
5. Combine with activation checkpointing for maximum memory savings

**The future:**
- ZeRO and FSDP are converging (both do parameter sharding)
- PyTorch FSDP2 is catching up in features
- For new projects: Use FSDP2 unless you need DeepSpeed-specific features
- For very large models (>100B): DeepSpeed still has an edge

---

## References

- [ZeRO Paper (2020)](https://arxiv.org/abs/1910.02054): Original ZeRO optimization
- [ZeRO-Offload Paper (2021)](https://arxiv.org/abs/2101.06840): CPU offloading techniques  
- [ZeRO-Infinity Paper (2021)](https://arxiv.org/abs/2104.07857): NVMe offloading
- [ZeRO++ Paper (2023)](https://arxiv.org/abs/2306.10209): Communication-optimized ZeRO
- [DeepSpeed Documentation](https://www.deepspeed.ai/): Official docs and tutorials
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed): Source code and examples
