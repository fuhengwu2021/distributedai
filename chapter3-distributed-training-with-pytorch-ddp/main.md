---
title: "Distributed Training with PyTorch DDP"
---

# Chapter 3 — Distributed Training with PyTorch DDP

This chapter is a hands-on guide to using PyTorch DistributedDataParallel (DDP) for multi-GPU and multi-node training. It explains DDP internals, how to initialize process groups, common failure modes and debugging techniques, and practical optimization strategies (bucketization, overlap, mixed precision). The chapter includes runnable examples and launcher recipes (`torchrun`, SLURM).

## 1. How DDP Works Internally

DDP replicates the full model on each process (usually one process per GPU). During backward, gradients are synchronized across processes using an AllReduce (NCCL backend on GPUs). Key mechanisms:

- Gradient bucketing: DDP groups small gradient tensors into buckets to reduce AllReduce overhead.
- Overlap: DDP can perform AllReduce for buckets while subsequent backward computations continue, enabling overlap of communication and computation.
- Gradient scaling and mixed precision: use AMP to reduce memory and bandwidth, but be careful with loss-scaling and gradient synchronization timing.

Understanding these internals helps you tune DDP performance and avoid common pitfalls like hangs caused by mismatched collective calls.

## 2. Setting Up Single-Node and Multi-Node DDP

Typical steps:

1. Create a process per GPU (use `torchrun --nproc_per_node` or spawn).  
2. Initialize process group: `dist.init_process_group(backend='nccl', init_method='env://')`.  
3. Wrap model with `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])`.  
4. Use `DistributedSampler` for dataset sharding.

Minimal `torchrun` example:

```bash
# Single-node, 4 GPUs
torchrun --nproc_per_node=4 train.py --epochs 10 --batch-size 32
```

Inside `train.py`, read `LOCAL_RANK` and set device:

```python
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
model.to(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

For multi-node, set `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK` (or use a rendezvous/launch tool provided by your cluster manager).

## 3. Debugging and Troubleshooting Common DDP Failures

Common failures and fixes:

- Hangs: usually mismatched collective calls, firewall/port issues, or NCCL network problems. Verify `WORLD_SIZE` and that every process reaches `init_process_group`.
- Wrong results/inconsistent gradients: ensure all processes use the same model and data sharding; check for nondeterministic operations or incorrect use of random seeds.
- CUDA out-of-memory: reduce per-GPU batch size or enable gradient accumulation and/or mixed precision.

Debugging tips:

- Run with a single process to test basic correctness.
- Add logging around `init_process_group` and before/after collective calls.
- Use `NCCL_DEBUG=INFO` and `NCCL_SOCKET_IFNAME` to narrow network issues.

## 4. Optimizing DDP with Buckets and Overlap

Performance levers:

- Bucket size: adjust by setting `bucket_cap_mb` (in MB) when creating DDP to influence communication granularity.
- Gradient accumulation: accumulate gradients locally for `k` steps to reduce synchronization frequency (useful when small batch sizes limit GPU utilization).
- Mixed precision (AMP): reduces memory and bandwidth, increasing throughput. Combine with careful loss-scaling to preserve numerical stability.
- Overlap communications: ensure your model's backward has work to overlap with (use asynchronous kernels and avoid blocking ops inside hooks).

Example: enabling FP16 with AMP + DDP:

```python
scaler = torch.cuda.amp.GradScaler()
for data, target in loader:
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 5. Checkpointing and Resuming Distributed Jobs

Checkpointing in DDP requires saving and restoring:

- Model state dict (use `module.state_dict()` if using DDP wrapper).  
- Optimizer state dict and scaler state (if using AMP).  
- RNG state and current epoch/iteration for deterministic resumes.

Use a single rank (e.g., `rank==0`) to write checkpoint files and have all ranks load the same file before resuming training.

## Hands-on Examples

1. `train.py` with `torchrun` launcher: includes dataset `DistributedSampler`, `DistributedDataParallel` wrapping, AMP, and checkpointing.  
2. SLURM submit script showing `srun` or `torchrun` multi-node launch with environment variables.

## Best Practices

- Use `torchrun` for most cases; rely on cluster-provided launchers for SLURM/hpc when necessary.  
- Always validate on a single-GPU process before scaling to many GPUs.  
- Profile and instrument networking (NCCL) and compute (Nsight / nvprof) to identify bottlenecks.  
- Keep checkpoints atomic and rotated to avoid corrupted resumes.

---

References: PyTorch DDP docs, NCCL tuning guide, cluster launcher examples.
---
title: "Distributed Training with PyTorch DDP"
---

# Chapter 3 — Distributed Training with PyTorch DDP

Status: TODO — draft placeholder

Chapter headings:
1. How DDP Works Internally
2. Setting Up Single-Node and Multi-Node DDP
3. Debugging and Troubleshooting Common DDP Failures
4. Optimizing DDP with Buckets and Overlap
5. Checkpointing and Resuming Distributed Jobs

TODO: Write full chapter text and examples (DDP scripts, torchrun/SLURM instructions).
