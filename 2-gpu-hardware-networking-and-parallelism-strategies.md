---
title: "GPU Hardware, Networking, and Parallelism Strategies"
---

# Chapter 2 — GPU Hardware, Networking, and Parallelism Strategies

This chapter covers GPU architecture and memory hierarchy, interconnect technologies (PCIe, NVLink, NVSwitch), distributed communication patterns, and major parallelism strategies (Data Parallelism, Tensor Parallelism, Pipeline Parallelism, Sharded Parallelism, FSDP, ZeRO, EP). It includes illustrative code snippets, basic microbenchmark examples, and high-level decision guidance for selecting parallelism strategies.

## 1. Understanding GPU Memory and Compute Architecture

GPUs are designed for throughput: many ALUs, vectorized FP units, and high-bandwidth on-package memory. Key concepts:

- Memory hierarchy: registers, shared memory / L1, L2, device/global DRAM.
- Memory bandwidth vs compute throughput: how to spot memory-bound kernels.
- Streaming multiprocessors (SMs), CUDA cores, tensor cores.

Practical tip: use `nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv` to inspect GPU model and PCIe capabilities.

**PCIe Information:**

NVIDIA H200:
```
$ nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
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
*Interpretation: H200 supports PCIe Gen 5 with 16 lanes (up to ~64 GB/s per direction).*

RTX 4090:
```
$ nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
name, memory.total [MiB], pcie.link.gen.max, pcie.link.width.max
NVIDIA GeForce RTX 4090, 24564 MiB, 4, 16
```
*Interpretation: RTX 4090 supports PCIe Gen 4 with 16 lanes (up to ~32 GB/s per direction).*

**NVLink Information:**

To check NVLink connectivity (GPU-to-GPU direct links), use:
```bash
$ nvidia-smi topo -m
```

Example output for a system with NVLink (visual representation shown below):

![NVLink Topology Matrix](code/chapter2/nvidia-smi-topo-example.png)

Text output:
```
 🎉 $nvidia-smi topo -m
	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	NIC6	NIC7	NIC8	NIC9	NIC10	NIC11	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NV18	PXB	NODE	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	0-55,112-167	0		N/A
GPU1	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NODE	NODE	NODE	PXB	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	0-55,112-167	0		N/A
GPU2	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NODE	NODE	NODE	NODE	PXB	NODE	SYS	SYS	SYS	SYS	SYS	SYS	0-55,112-167	0		N/A
GPU3	NV18	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NODE	NODE	NODE	NODE	NODE	PXB	SYS	SYS	SYS	SYS	SYS	SYS	0-55,112-167	0		N/A
GPU4	NV18	NV18	NV18	NV18	 X 	NV18	NV18	NV18	SYS	SYS	SYS	SYS	SYS	SYS	PXB	NODE	NODE	NODE	NODE	NODE	56-111,168-223	1		N/A
GPU5	NV18	NV18	NV18	NV18	NV18	 X 	NV18	NV18	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	NODE	PXB	NODE	NODE	56-111,168-223	1		N/A
GPU6	NV18	NV18	NV18	NV18	NV18	NV18	 X 	NV18	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	PXB	NODE	56-111,168-223	1		N/A
GPU7	NV18	NV18	NV18	NV18	NV18	NV18	NV18	 X 	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	NODE	PXB	56-111,168-223	1		N/A
NIC0	PXB	NODE	NODE	NODE	SYS	SYS	SYS	SYS	 X 	NODE	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS
NIC1	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	NODE	 X 	PIX	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS
NIC2	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	NODE	PIX	 X 	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS
NIC3	NODE	PXB	NODE	NODE	SYS	SYS	SYS	SYS	NODE	NODE	NODE	 X 	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS
NIC4	NODE	NODE	PXB	NODE	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	 X 	NODE	SYS	SYS	SYS	SYS	SYS	SYS
NIC5	NODE	NODE	NODE	PXB	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	NODE	 X 	SYS	SYS	SYS	SYS	SYS	SYS
NIC6	SYS	SYS	SYS	SYS	PXB	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	 X 	NODE	NODE	NODE	NODE	NODE
NIC7	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	NODE	 X 	PIX	NODE	NODE	NODE
NIC8	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	NODE	PIX	 X 	NODE	NODE	NODE
NIC9	SYS	SYS	SYS	SYS	NODE	PXB	NODE	NODE	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	NODE	 X 	NODE	NODE
NIC10	SYS	SYS	SYS	SYS	NODE	NODE	PXB	NODE	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	 X 	NODE
NIC11	SYS	SYS	SYS	SYS	NODE	NODE	NODE	PXB	SYS	SYS	SYS	SYS	SYS	SYS	NODE	NODE	NODE	NODE	NODE	 X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
  NIC8: mlx5_8
  NIC9: mlx5_9
  NIC10: mlx5_10
  NIC11: mlx5_11
```

*Legend (from nvidia-smi output):*
- `NV#` = NVLink connection (e.g., `NV18` = NVLink 1.8, `NV12` = NVLink 1.2). The number indicates the NVLink generation/bonding.
- `PIX` = Connection traversing at most a single PCIe bridge
- `PXB` = Connection traversing multiple PCIe bridges
- `NODE` = Connection traversing PCIe and interconnect between PCIe Host Bridges within a NUMA node
- `SYS` = Connection traversing PCIe and SMP interconnect between NUMA nodes
- `X` = Self (same device)

**Understanding Your Interconnect:**

- **PCIe only**: If `nvidia-smi topo -m` shows `PIX` or `PXB` between GPUs, they communicate via PCIe through the CPU/PCIe switch (higher latency, lower bandwidth, typically 16-64 GB/s depending on PCIe generation).
- **NVLink present**: If you see `NV#` (e.g., `NV18` = NVLink 1.8, `NV12` = NVLink 1.2, `NV4` = NVLink 4.0), GPUs have direct high-bandwidth links (lower latency, much higher bandwidth, typically 300-900 GB/s depending on NVLink generation and bonding). In the example above, all 8 GPUs are connected via `NV18`, indicating NVLink 1.8 with full all-to-all connectivity.
- **Mixed topology**: Some systems have NVLink between some GPU pairs and PCIe for others (common in multi-socket servers). You may also see `NODE` or `SYS` connections indicating cross-NUMA-node communication.
- **NUMA awareness**: Notice in the example that GPUs 0-3 are on NUMA node 0 (CPU affinity 0-55,112-167) while GPUs 4-7 are on NUMA node 1 (CPU affinity 56-111,168-223). This affects performance—prefer keeping communication within the same NUMA node when possible.

**Interpreting the example output above:**

The example shows an H200 system in a DGX/HGX configuration with NVSwitch:
- All 8 GPUs (GPU0-GPU7) have `NV18` connections to all other GPUs, indicating full all-to-all NVLink connectivity via NVSwitch.
- This is optimal for distributed training as all GPU pairs can communicate at high bandwidth (NVLink 1.8 speeds).

In contrast, a standard server configuration would show:
- `PIX`, `PXB`, `NODE`, or `SYS` connections between some GPU pairs, indicating PCIe-only or partial NVLink connectivity.
- Some GPUs may only connect via PCIe, creating communication bottlenecks.

## 2. High-Speed Interconnects: PCIe, NVLink, NVSwitch

Interconnects matter for multi-GPU scaling:

- **PCIe**: ubiquitous, lower bandwidth and higher latency than NVLink. Standard connection between CPU and GPU, also used for GPU-to-GPU communication when NVLink is not available.
- **NVLink**: GPU-to-GPU high-bandwidth links (peer-to-peer) within a server. Direct connections between GPUs, typically 300-900 GB/s depending on generation.
- **NVSwitch**: provides non-blocking all-to-all connectivity across many GPUs. Enables every GPU to communicate with every other GPU at full NVLink bandwidth simultaneously.

**DGX vs HGX Systems:**

- **DGX (Data Center GPU)**: Pre-integrated systems from NVIDIA with GPUs, CPUs, networking, storage, and optimized software stack. Examples: DGX H100, DGX A100. Typically feature NVSwitch for all-to-all GPU connectivity.
- **HGX (Hyperscale GPU eXpansion)**: Modular platform providing GPU baseboards and interconnects for OEMs/system integrators to build custom servers. More flexible but requires integration work. Can also include NVSwitch depending on configuration.

Both DGX and HGX systems often include NVSwitch for optimal multi-GPU communication, which is why the topology example above shows all-to-all NVLink connectivity.

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
