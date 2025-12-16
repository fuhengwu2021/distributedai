# Chapter 2 — GPU Hardware, Networking, and Parallelism Strategies

Before you can effectively distribute training or inference across multiple GPUs, you need to understand what you're working with. The hardware topology of your system—how GPUs connect to each other and to the CPU—directly impacts which parallelism strategies will work best and what performance you can expect.

This chapter walks through GPU architecture basics, interconnect technologies, and the major parallelism strategies you'll encounter. We'll also cover how to inspect your system and make informed decisions about which approach fits your workload.

## Understanding GPU Memory and Compute Architecture

GPUs are built for throughput, not latency. Unlike CPUs that optimize for fast single-threaded execution, GPUs pack thousands of simple cores and prioritize high-bandwidth memory access. When you're training large models, this design pays off—but it also means you need to think differently about memory and compute.

The memory hierarchy matters. Registers are fastest but tiny. Shared memory (L1 cache) is fast but limited. L2 cache sits between shared memory and device DRAM, which is your main GPU memory. When you see "out of memory" errors, it's usually the device DRAM that's full, not the caches.

One thing that trips people up: memory bandwidth often becomes the bottleneck before compute does. If your kernels are memory-bound, adding more compute won't help. You can spot this by profiling—if your GPU utilization is low but memory bandwidth is maxed out, you're memory-bound.

To see what you're working with, check your GPU specs:

```bash
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
```

Here's what you might see on different systems. An H200 system with 8 GPUs:

```
$ nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv
name, memory.total [MiB], pcie.link.gen.max, pcie.link.width.max
NVIDIA H200, 143771 MiB, 5, 16
NVIDIA H200, 143771 MiB, 5, 16
...
```

That's PCIe Gen 5 with 16 lanes—roughly 64 GB/s per direction. Compare that to a consumer RTX 4090:

```
NVIDIA GeForce RTX 4090, 24564 MiB, 4, 16
```

PCIe Gen 4, same 16 lanes, but only about 32 GB/s. The PCIe connection is what your GPU uses to talk to the CPU, but for multi-GPU communication, you want something faster.

But PCIe is just one part of the story. For multi-GPU communication, you want NVLink—direct GPU-to-GPU links that bypass the CPU entirely. To see what your system actually has, run:

```bash
nvidia-smi topo -m
```

This shows the topology matrix. The output can be dense, but here's what to look for:

If you see `NV18`, `NV12`, or `NV4` between GPUs, you have NVLink. That's good—those links give you 300-900 GB/s depending on the generation, way faster than PCIe. In a well-configured system like a DGX or HGX box, you'll see all GPUs connected via NVLink through an NVSwitch, meaning every GPU can talk to every other GPU at full speed.

If you see `PIX` or `PXB` between GPUs, they're only connected via PCIe. That works, but you'll hit bandwidth limits faster. You might also see `NODE` or `SYS`, which means the connection crosses NUMA boundaries—another thing that can slow things down.

Here's a real example from an H200 system with NVSwitch. All 8 GPUs show `NV18` connections to each other:

```
GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18
...
```

That's the ideal setup. Every GPU can communicate with every other GPU at NVLink speeds. In a standard server, you might see some GPUs only connected via `PIX` or `PXB`, which means they're going through PCIe. That still works, but you'll want to be more careful about which GPUs you pair together for communication-heavy operations.

One more thing: notice the CPU affinity column. GPUs 0-3 might be on NUMA node 0, while GPUs 4-7 are on NUMA node 1. If you're doing multi-node training, try to keep processes on the same NUMA node when possible—cross-NUMA communication adds latency.

## High-Speed Interconnects: PCIe, NVLink, NVSwitch, InfiniBand, and Ethernet

There are several ways GPUs connect, and which one matters depends on whether you're talking about communication within a single server or across multiple servers.

**Within a single server:**

**PCIe** is what you get by default. Every GPU connects to the CPU via PCIe, and if there's no NVLink, GPUs talk to each other through the CPU too. It works, but it's the slowest option—typically 16-64 GB/s depending on PCIe generation. The latency is also higher since everything goes through the CPU.

**NVLink** is NVIDIA's direct GPU-to-GPU interconnect. When two GPUs have NVLink between them, they can talk directly without involving the CPU. Bandwidth is much higher—300-900 GB/s depending on the generation. The catch is that not all systems have it, and even when they do, not all GPU pairs might be connected.

**NVSwitch** is what you see in high-end systems like DGX or HGX boxes. It's essentially a switch that connects all GPUs via NVLink, giving you all-to-all connectivity. Every GPU can talk to every other GPU at full NVLink speed simultaneously. This is what you want for large-scale distributed training within a single node.

**Across multiple servers:**

**InfiniBand** is the standard for multi-node GPU clusters. When you're running distributed training across multiple servers, GPUs on different nodes need to communicate, and that's where InfiniBand comes in. It provides high-bandwidth, low-latency networking—typically 200-400 Gb/s (25-50 GB/s) per port, with sub-microsecond latency. Modern systems use InfiniBand HDR (200 Gb/s) or NDR (400 Gb/s).

The key feature that makes InfiniBand fast is **RDMA (Remote Direct Memory Access)**. RDMA allows network adapters to read and write memory directly without involving the CPU or kernel. InfiniBand was designed from the ground up to support RDMA natively—it's built into the protocol. When you use InfiniBand, you get RDMA by default.

You'll see InfiniBand NICs in the `nvidia-smi topo -m` output—those are the network interface cards that connect your server to the cluster network. When NCCL does multi-node communication, it uses **GPUDirect RDMA**, which is NVIDIA's implementation that extends RDMA to GPU memory. This allows data to transfer directly from GPU memory on one node to GPU memory on another node, completely bypassing the CPU and system RAM. That's why it's so fast.

**Ethernet** is the other option for multi-node networking. There are two flavors:

Standard Ethernet (TCP/IP) works, but it's slower. You're looking at 10-100 Gb/s per port, and latency is higher because everything goes through the kernel network stack. For small clusters or when cost is a concern, it can work, but you'll see performance degradation as you scale up.

**RoCE (RDMA over Converged Ethernet)** is the interesting one. As the name suggests, it's RDMA over Ethernet instead of InfiniBand. So RDMA isn't exclusive to InfiniBand—it's a capability that can be implemented over different network technologies. RoCE v2 gives you the same RDMA benefits (GPU-to-GPU direct memory access, bypassing the CPU) but over standard Ethernet infrastructure. Bandwidth is comparable—100-400 Gb/s depending on the NIC—but latency is typically higher than InfiniBand, and you need proper switch configuration (DCB/PFC) to avoid packet loss under load.

The practical difference: InfiniBand is purpose-built for HPC workloads and tends to be more reliable at scale. RoCE works well in cloud environments where you're already using Ethernet infrastructure, but you need to tune it carefully. Many cloud providers offer both options—AWS has EFA (Elastic Fabric Adapter) which supports both InfiniBand and RoCE, and Google Cloud has similar offerings.

For most on-premise clusters, InfiniBand is still the default choice. But if you're in a cloud environment or have existing Ethernet infrastructure, RoCE is a viable alternative. The bandwidth and latency characteristics matter a lot when you're synchronizing gradients across hundreds of GPUs, so test both if you have the option.

If you're buying hardware, DGX systems are pre-integrated—NVIDIA ships you a complete system with GPUs, CPUs, networking (including InfiniBand), and software stack. HGX is more modular—it's a baseboard design that OEMs use to build custom servers. Both can include NVSwitch for intra-node communication and InfiniBand for inter-node.

To actually measure your interconnect bandwidth, you can use NCCL tests or write a simple benchmark. The `code/bandwidth_test.py` script gives you a basic single-GPU test. For multi-GPU within a node, you'll want to use `nccl-tests`. For multi-node, NCCL tests will show you the InfiniBand bandwidth between nodes.

## Distributed Communication Patterns

When you're running distributed training, GPUs need to communicate. The main patterns you'll see are:

**Broadcast** sends data from one GPU (usually rank 0) to all others. You use this during initialization to get the same model weights on every GPU.

**AllReduce** aggregates data from all GPUs and distributes the result back. This is what DDP uses for gradient synchronization—each GPU computes gradients on its local data, then AllReduce averages them across all GPUs.

**ReduceScatter** and **AllGather** show up in sharded parallelism. ReduceScatter splits the result across GPUs, while AllGather collects data from all GPUs into each GPU.

The thing to watch with communication is not just raw bandwidth, but also startup latency and whether you can overlap it with computation. A fast interconnect helps, but if your communication pattern has high latency, you'll still wait. DDP tries to overlap communication with computation by bucketing gradients, which we'll cover in the next chapter.

You can benchmark these operations yourself. The `code/allreduce_microbench.py` script shows a basic example, though you'll need to initialize the process group first (we'll cover that in Chapter 3).

## Parallelism Strategies

There are several ways to split work across GPUs. Each has tradeoffs, and you'll often combine them. Here's the quick rundown:

**Data Parallelism (DP)** is the simplest. You replicate the entire model on each GPU and split the batch across GPUs. Each GPU processes different data, then you synchronize gradients. It's easy to implement and works great when your model fits on a single GPU. The downside is that you're storing the full model on every GPU, so memory usage scales with the number of GPUs.

**Tensor Parallelism (TP)** splits individual layers across GPUs. Instead of replicating a layer, you split the weight matrix. For example, if you have a linear layer with a 4096x4096 weight matrix, you might split it into two 4096x2048 matrices on two GPUs. During forward pass, each GPU computes part of the output, then you AllGather to combine results. This lets you fit larger layers, but communication happens every layer, which can be expensive.

**Pipeline Parallelism (PP)** splits the model depth-wise. GPU 0 handles layers 0-10, GPU 1 handles layers 11-20, and so on. You pipeline microbatches through the stages to keep all GPUs busy. The challenge is pipeline bubbles—when one stage finishes before the next is ready, GPUs sit idle. Getting the scheduling right matters.

**FSDP (Fully Sharded Data Parallel)** and **ZeRO** are memory optimization strategies. Instead of replicating optimizer states and gradients on every GPU, you shard them. FSDP shards parameters, gradients, and optimizer states. ZeRO does similar things but with different stages—Stage 1 shards optimizer states, Stage 2 adds gradients, Stage 3 adds parameters. Both let you train much larger models with the same number of GPUs.

**Expert Parallelism (EP)** is for MoE models. You distribute different experts across GPUs, and tokens get routed to the right expert. If you have 64 experts and 8 GPUs, each GPU might hold 8 experts. The tricky part is load balancing—some experts get more traffic than others, so you need good routing.

In practice, you'll combine these. A common setup for a 70B model might be: FSDP for memory efficiency, plus some tensor parallelism for the largest layers, plus pipeline parallelism if you have enough GPUs. For MoE models, you might do expert parallelism plus data parallelism across expert groups.

The decision tree is roughly: Does your model fit on one GPU? If yes, start with DP. If not, does a single layer not fit? Use TP. Is the model just too deep? Consider PP. Need maximum memory efficiency? FSDP or ZeRO Stage 3. MoE model? EP.

Most people don't implement these from scratch—you'll use PyTorch's DDP/FSDP, DeepSpeed's ZeRO, or libraries like Megatron-LM that handle the tensor parallelism details. But understanding what's happening under the hood helps when things go wrong.

## Choosing the Right Strategy

There's no one-size-fits-all answer. Here's how I think about it:

First, does your model fit on a single GPU? If yes, data parallelism is usually the right starting point. It's simple, and you'll get good speedup as long as communication doesn't dominate.

If the model doesn't fit, figure out why. Is it one or two layers that are huge? Tensor parallelism might help. Is it the depth? Pipeline parallelism. Is it everything—weights, gradients, optimizer states? That's where FSDP or ZeRO shine.

Also consider your constraints. If you're memory-limited (model doesn't fit), prioritize FSDP or ZeRO. If your goal is to maximize throughput (samples per second) and you have fast interconnects (good NVLink), tensor parallelism can work well—it splits layers across GPUs to process them faster, but requires communication every layer, so fast interconnects are essential. If you have slow interconnects (PCIe only), avoid strategies that require frequent communication like tensor parallelism.

One thing that trips people up: the networking topology matters. If you're on a system where some GPU pairs are connected via NVLink and others via PCIe, try to keep communication-heavy operations on the NVLink-connected pairs. PyTorch and most frameworks don't do this automatically, so you might need to set process groups or device placement manually.

A few practical tips:

- Profile before you optimize. Use `nvidia-smi` to watch memory usage, and use PyTorch's profiler to see where time is spent. You might think communication is the bottleneck, but it could be data loading or something else.

- Use NCCL for GPU collectives. It's optimized and handles NVLink automatically. The alternative backends (GLOO, MPI) are slower.

- Mixed precision helps. FP16 or BF16 cuts memory and bandwidth in half. Most models train fine with it, and the speedup is significant.

- Watch out for NUMA. If you're on a multi-socket system, try to keep processes on the same NUMA node. Cross-NUMA communication adds latency.

The code examples in `code/` show basic topology detection and bandwidth testing. For real workloads, you'll use the higher-level APIs in PyTorch or DeepSpeed, but understanding what's happening underneath helps when things don't work as expected.

In the next chapter, we'll dive into PyTorch DDP, which is the most common way to do data parallelism. We'll cover the setup, common pitfalls, and how to optimize it.
