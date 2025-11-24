---
title: "Scaling with Fully Sharded Data Parallel (FSDP)"
---

# Chapter 4 — Scaling with Fully Sharded Data Parallel (FSDP)

FSDP enables training models larger than single-GPU memory by sharding parameters, gradients, and optimizer states across ranks. This chapter explains sharding modes, integration with activation checkpointing and offloading, and practical multi-node deployments.

## 1. Why FSDP Enables Larger-Than-Memory Models

FSDP shards model parameters across processes so each rank holds only a subset of parameters in memory during forward/backward. By moving parameter and optimizer states off-device (CPU/NVMe) or using mixed sharding modes, you can trade CPU/IO for GPU memory to scale to models with tens of billions of parameters.

## 2. Understanding FSDP Sharding Strategies

- Full-shard: every parameter is sharded across ranks; requires careful state handling but minimizes GPU memory.
- Shard-with-overlap: keeps small parameters unsharded to reduce communication.
- Mixed modes: combine local replication for tiny layers and sharding for large layers.

Key concepts: flat parameter buffers, state dict differences (need `FSDP.state_dict()` helpers), the role of CPU offload.

## 3. Activation Checkpointing and Offloading

Activation checkpointing reduces activation memory by recomputing forward during backward. Combine checkpointing with FSDP to push both parameter and activation memory down. Offloading optimizer states to CPU or NVMe removes optimizer memory from GPU but adds IO overhead—tune batch size and offload frequency accordingly.

Example (schematic):

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.checkpoint import checkpoint

wrapped = FSDP(model)
def forward_step(x):
    return wrapped(x)

# Use checkpointing on large blocks to save activation memory
out = checkpoint(forward_step, inp)
```

## 4. Multi-Node FSDP Training

Multi-node FSDP requires the same process-group and rendezvous steps as DDP. Pay attention to:

- Synchronizing state dicts and checkpoints across ranks.  
- Using efficient filesystem for checkpoints when offloading (shared storage or parallel writes).  
- Tuning CPU/GPU overlap and IO to mask offload cost.

## 5. Comparing FSDP with ZeRO and DDP

- DDP: easiest to reason about; replicates all parameters and optimizer states per device.
- ZeRO (DeepSpeed): staged optimizer/gradient/parameter sharding with mature offload tooling.
- FSDP: PyTorch-native sharding with tight integration to autograd; more flexible for mixed-shard strategies.

Decision tips: prefer FSDP when you want PyTorch-first integration and control over sharding; prefer ZeRO when you need DeepSpeed optimizations like ZeRO-Offload and advanced pipeline features.

## Hands-on Recipes

1. FSDP config templates for single-node multi-GPU and multi-node clusters.  
2. Scripts for benchmarking memory saving vs throughput with/without checkpointing and offload.

## Best Practices

- Profile memory usage per layer to decide which layers to shard.  
- Start with partial sharding then move to full-shard as you stabilize training.  
- Use mixed precision and checkpointing together to maximize memory savings.

---

References: PyTorch FSDP docs, FairScale examples, community memory-profiling scripts.
---
title: "Scaling with Fully Sharded Data Parallel (FSDP)"
---

# Chapter 4 — Scaling with Fully Sharded Data Parallel (FSDP)

Status: TODO — draft placeholder

Chapter headings:
1. Why FSDP Enables Larger-Than-Memory Models
2. Understanding FSDP Sharding Strategies
3. Activation Checkpointing and Offloading
4. Multi-Node FSDP Training
5. Comparing FSDP with ZeRO and DDP

TODO: Add scripts, examples, and config templates.
