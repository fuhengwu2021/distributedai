---
title: "GPU Hardware, Networking, and Parallelism Strategies"
---

# Chapter 2 — GPU Hardware, Networking, and Parallelism Strategies

This chapter covers GPU architecture and memory hierarchy, interconnect technologies (PCIe, NVLink, NVSwitch), distributed communication patterns, and major parallelism strategies (Data Parallelism, Tensor Parallelism, Pipeline Parallelism, Sharded Parallelism, FSDP, ZeRO). It includes hands-on examples, microbenchmarks, and decision guides.

## 1. Understanding GPU Memory and Compute Architecture

GPUs are designed for throughput: many ALUs, vectorized FP units, and high-bandwidth on-package memory. Key concepts:

- Memory hierarchy: registers, shared memory / L1, L2, device/global DRAM.
- Memory bandwidth vs compute throughput: how to spot memory-bound kernels.
- Streaming multiprocessors (SMs), CUDA cores, tensor cores.

Practical tip: use `nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv` to inspect GPU model and PCIe capabilities.

```
 👉 $nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
name, memory.total [MiB], pcie.link.gen.max, pcie.link.width.max
NVIDIA H200, 143771 MiB, 5, 16
NVIDIA H200, 143771 MiB, 5, 16
NVIDIA H200, 143771 MiB, 5, 16
NVIDIA H200, 143771 MiB, 5, 16
NVIDIA H200, 143771 MiB, 5, 16
NVIDIA H200, 143771 MiB, 5, 16
NVIDIA H200, 143771 MiB, 5, 16
NVIDIA H200, 143771 MiB, 5, 16
```

## 2. High-Speed Interconnects: PCIe, NVLink, NVSwitch

Interconnects matter for multi-GPU scaling:

- PCIe: ubiquitous, lower bandwidth and higher latency than NVLink.
- NVLink: GPU-to-GPU high-bandwidth links (peer-to-peer) within a server.
- NVSwitch: provides non-blocking all-to-all across many GPUs in DGX-style systems.

Bandwidth test example (quick): use NVIDIA NCCL tests or `nccl-tests`/`all_reduce_perf`. Simple Python-based bandwidth probe (uses `torch`):

```python
import torch
import time

def bandwidth_test(size_mb=64, iterations=100):
    nbytes = size_mb * 1024 * 1024
    a = torch.randn(nbytes // 4, device='cuda')
    b = torch.empty_like(a)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iterations):
        b.copy_(a)
    torch.cuda.synchronize()
    t1 = time.time()
    gb_transferred = (nbytes * iterations) / (1024**3)
    print(f"Bandwidth: {gb_transferred / (t1 - t0):.2f} GB/s")

if __name__ == '__main__':
    bandwidth_test(64, 200)
```

This test measures device-to-device memcpy bandwidth on a single GPU. For multi-GPU interconnects, run NCCL tests (see `nccl-tests` repo) or `torch.distributed` collective bandwidth microbenchmarks.

## 3. Distributed Communication Patterns (AllReduce, Broadcast)

- Broadcast: send parameters from rank 0 to all workers (model init).
- AllReduce: aggregate gradients across workers (synchronous SGD).
- ReduceScatter / AllGather: used in sharded/ tensor parallel flows.

When measuring communication, always consider: startup latency, bandwidth, and effective overlap with computation.

Example: measuring AllReduce with torch.distributed (pseudo):

```python
import torch
import torch.distributed as dist

def allreduce_microbench(tensor_size=1024*1024, iters=100):
    t = torch.randn(tensor_size, device='cuda')
    for _ in range(iters):
        dist.all_reduce(t)

# Requires initialized process group (NCCL backend)
```

## 4. Parallelism Strategies (DP, TP, PP, SP, FSDP, ZeRO, EP)

High-level summary and when to pick each:

- Data Parallelism (DP): replicate model, split batch across GPUs. Simple to implement; memory duplicates model per device. Best when model fits single GPU and batch parallelism suffices.
- Tensor Parallelism (TP): split tensors (layers) across devices (e.g., split weight matrices along hidden dimension). Good for very wide layers when single-GPU memory insufficient.
- Pipeline Parallelism (PP): split model layers into stages, pipeline microbatches. Useful for very deep models. Watch out for bubble/latency and pipeline scheduling.
- Sharded Parallelism (SP) / FSDP: shard optimizer and parameter states across ranks to reduce memory footprint.
- ZeRO (DeepSpeed): staged sharding for optimizer states, gradients, and parameters — scales to very large models with lower memory overhead.
- Expert Parallelism (EP): distribute different experts in Mixture-of-Experts (MoE) models across devices. Each device holds a subset of experts, and tokens are routed to the appropriate expert. Used for very large MoE models (e.g., models with 64+ experts). Requires efficient routing and load balancing across experts.

Decision guide:

- If model fits a single GPU and batch size is limiting throughput: DP.
- If a single layer is too big to fit: TP + (possibly) ZeRO/FSDP.
- If model depth is huge and pipeline parallelism can help: PP with microbatches.
- For minimal memory footprint and largest possible model: ZeRO Stage 3 or FSDP full-shard.
- For MoE models with many experts: EP combined with DP (data parallelism across expert groups) or EP + PP for very large MoE models.

### Small example: manual DP vs TP sketch

Manual DP (concept): replicate model on each device, use DDP to sync gradients.

Manual TP (concept): split large Linear weight W into W1,W2 across two devices and coordinate matmul parts and AllGather.

Implementing TP correctly usually relies on libraries (Megatron-LM style or transformer parallel helpers) or careful tensor slicing + communication.

## 5. Choosing the Right Strategy for Real Workloads

Checklist for picking parallelism:

1. Does the model fit on a single GPU? If yes, start with DP.
2. Which layers are the largest memory consumers? If a small subset dominates, consider TP for those layers.
3. Is throughput (samples/sec) or memory footprint the hard limit? Optimize accordingly.
4. What is the networking topology? Avoid cross-socket communication where possible; prefer NVLink-connected peers.

### Best practices

- Profile early: identify memory hotspots and communication hot paths.
- Prefer NCCL-backed collectives for GPUs (low-latency, optimized).
- Avoid frequent cross-socket transfers in single-node multi-GPU servers.
- Use mixed precision to reduce memory and bandwidth (AMP / FP16 / BF16).

## Hands-on Examples & Exercises

1. GPU topology detection script (Python): checks peer-to-peer accessibility and NVLink presence.
2. Run `nccl-tests` (all_reduce_perf) across GPUs and record bandwidth vs message size.
3. Implement a tiny tensor-parallel layer by splitting a Linear layer weight and measuring forward/backward time vs single-GPU baseline.

## Skills learned

1. Benchmark GPU interconnect and memory behavior.
2. Analyze and avoid communication bottlenecks.
3. Implement simple distributed communication patterns.
4. Compare parallelism strategies for different workloads.
5. Select the correct scaling approach for a given model.

## References and Tools

- `nccl-tests` for collective benchmarks
- NVIDIA `nvidia-smi`, `nvprof`, and Nsight tools
- Megatron-LM, DeepSpeed, FairScale, and PyTorch FSDP docs

---

Feedback requested: This is the full draft for review. Tell me which sections to expand or any additional examples you'd like.
