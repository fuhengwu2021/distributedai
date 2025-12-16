# Chapter 2 — GPU Hardware, Networking, and Parallelism Strategies


Before you can effectively distribute training or inference across multiple GPUs, you need to understand what you're working with. The hardware topology of your system—how GPUs connect to each other and to the CPU—directly impacts which parallelism strategies will work best and what performance you can expect.

This chapter walks through GPU architecture basics, interconnect technologies, and the major parallelism strategies you'll encounter. We'll also cover how to inspect your system and make informed decisions about which approach fits your workload.

![](img/h100.jpg)

## Computational Power and AI Clusters

Before diving into GPU specifics, let's step back and understand what we're really building: AI clusters that deliver massive computational power. The numbers matter here—when you're training a 70B parameter model, you're not just using a few GPUs. You're orchestrating hundreds or thousands of them, and the way they're connected determines whether your training job finishes in days or weeks.

### What is Computational Power?

Computational power, or compute capacity, measures how many operations a system can perform per second. For AI workloads, we care about floating-point operations per second (FLOPS). The scale is exponential: a single modern GPU like the H200 delivers around 1,000 TFLOPS (teraFLOPS, or 10^12 operations per second) for FP16 operations. A cluster with 1000 such GPUs gives you roughly 1 PFLOPS (petaFLOPS, 10^15 operations per second).

But here's the thing: raw FLOPS numbers don't tell the whole story. In practice, you'll see different precision formats used for different purposes:

- **FP64 (double precision)**: 64 bits, used in traditional HPC for scientific computing where precision matters
- **FP32 (single precision)**: 32 bits, common baseline for training
- **FP16/BF16 (half precision)**: 16 bits, standard for modern AI training—cuts memory and bandwidth in half
- **FP8/FP4**: 8 or 4 bits, used for inference and quantization

When someone says "this cluster delivers 500 PFLOPS," you need to ask: at what precision? An HPC cluster might quote FP64 numbers, while an AI cluster quotes FP16 or BF16. The same hardware can show very different numbers depending on which precision you're measuring.

The growth in computational demand for AI has been staggering. Large language models require computational resources that grow roughly 750x every two years, while hardware capabilities grow only about 3x in the same period. This gap is why distributed training isn't optional—it's the only way to train modern models in reasonable time.

### Why Clusters?

A single GPU, even a high-end one, isn't enough for modern AI workloads. A 70B parameter model with FP16 weights takes about 140 GB just to store the model. Add gradients, optimizer states, and activations, and you're looking at 500+ GB per training step. That's beyond what any single GPU can hold.

A **cluster** is a group of computers (nodes) connected by high-speed networks, working together as a single system. Each node typically has multiple GPUs, CPUs, memory, and storage. The key insight is that by coordinating work across many nodes, you can:

- **Scale memory**: Distribute model parameters, gradients, and optimizer states across GPUs
- **Scale compute**: Process larger batches or train faster by parallelizing work
- **Scale storage**: Handle datasets that don't fit on a single machine

Clusters aren't new—they've been used in high-performance computing (HPC) for decades. What's different for AI is the communication patterns. HPC workloads often do large, infrequent data exchanges. AI training does frequent, smaller exchanges (gradient synchronization every step), which makes network bandwidth and latency critical.

### AI Clusters: Built for Training and Inference

An **AI cluster** is a cluster specifically designed for AI workloads. Unlike general-purpose cloud data centers that handle diverse workloads, AI clusters are optimized for the unique characteristics of deep learning:

**For training**, AI clusters need:
- **High-bandwidth interconnects**: Gradient synchronization happens every training step. If communication is slow, GPUs sit idle waiting for gradients. NVLink (300-900 GB/s) within nodes and InfiniBand (200-400 Gb/s) between nodes are standard.
- **Large aggregate memory**: Model parameters, gradients, and optimizer states are sharded across GPUs. A 70B model might need 8-16 GPUs just to fit in memory, even with techniques like FSDP.
- **Fast storage**: Training datasets are large (ImageNet is 150 GB, text datasets can be terabytes). You need fast parallel filesystems or object storage to keep data pipelines fed.

**For inference**, the requirements shift:
- **Lower latency networking**: While training cares about bandwidth, inference cares about latency. Users expect responses in milliseconds, not seconds.
- **Efficient memory usage**: KV caches for attention mechanisms can consume significant memory. You need to balance cache size (for longer context) against memory limits.
- **Load balancing**: Inference workloads are bursty. You need to route requests efficiently across GPUs and handle traffic spikes.

The hardware topology—how GPUs connect within a node and how nodes connect to each other—directly impacts what parallelism strategies work. A cluster where all GPUs are connected via NVSwitch (all-to-all connectivity) can use tensor parallelism effectively. A cluster where GPUs are only connected via PCIe will struggle with communication-heavy strategies.

### Key Metrics for AI Clusters

When you're evaluating or optimizing an AI cluster, you need concrete metrics. Raw FLOPS numbers are marketing—what matters is how efficiently you use the hardware. Here are the metrics that actually matter:

**Model FLOPS Utilization (MFU)** is the most important metric for training efficiency. It measures what percentage of peak hardware FLOPS you're actually using:

```
MFU = (Model FLOPs per iteration / Iteration time) / Peak FLOPS
```

MFU tells you if you're compute-bound or limited by something else. A well-optimized cluster might achieve 40-60% MFU for large models. If MFU is low (say, 20%), you're likely hitting memory bandwidth limits, communication bottlenecks, or inefficient kernel launches.

For a 70B parameter transformer model training on H100 GPUs, you might see:
- **Theoretical FLOPs per iteration**: ~80 TFLOPS (depends on batch size, sequence length)
- **Actual FLOPS per second**: ~300 TFLOPS (if iteration takes 0.27 seconds)
- **Peak H100 FLOPS**: ~1000 TFLOPS (FP16)
- **MFU**: 300/1000 = 30%

That 30% MFU means 70% of your hardware is idle. Common causes: memory bandwidth saturation, communication overhead, or small batch sizes that don't keep GPUs busy.

**Linear scaling** measures how well performance scales with cluster size. The formula is:

```
Linear scaling = (Multi-GPU throughput) / (Single-GPU throughput × GPU count)
```

Perfect scaling gives you 1.0 (100%). In practice, you'll see 0.7-0.9 for well-optimized clusters. If scaling drops below 0.5, you have a communication bottleneck.

**GPU utilization** is simpler—it's the percentage of time GPUs spend computing vs waiting. You can check it with:

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
```

High utilization (90%+) is good, but it doesn't tell you if you're using the right kernels or if communication is blocking computation. MFU is more informative.

**Communication efficiency** measures how well you're using network bandwidth:

```
Communication efficiency = Actual bandwidth / Theoretical bandwidth
```

For InfiniBand HDR (200 Gb/s), you might achieve 180 Gb/s actual bandwidth, giving 90% efficiency. If efficiency is low, you might have topology issues, packet loss, or suboptimal communication patterns.

**Throughput** (samples per second or tokens per second) is what you care about for training speed:

```
Throughput = (Global batch size × Sequence length) / (Total training time × GPU count)
```

This tells you how fast you're processing data. Higher is better, but you need to balance it with convergence—larger batches might train faster per iteration but need more iterations to converge.

**Resource utilization** breaks down where time is spent:
- **GPU compute time**: Actual matrix multiplications
- **Memory transfer time**: Moving data between CPU and GPU, or between GPUs
- **Communication time**: Gradient synchronization, AllReduce operations
- **Idle time**: Waiting for data, synchronization, or other bottlenecks

A well-optimized cluster spends 70-80% of time in compute, 10-20% in communication, and minimal time idle.

**Energy efficiency** matters for large clusters. **FLOPS per Watt** measures compute efficiency:

```
FLOPS/Watt = Total FLOPS / Total power consumption
```

An H100 delivers about 1.4 TFLOPS/Watt at FP16. Higher is better—it means you're getting more compute for the same power bill.

**PUE (Power Usage Effectiveness)** measures datacenter efficiency:

```
PUE = Total facility power / IT equipment power
```

A PUE of 1.0 means all power goes to IT equipment (impossible in practice). Real datacenters achieve 1.2-1.5. Lower is better—it means less power wasted on cooling and overhead.

**Reliability metrics** matter when you're running week-long training jobs:
- **MTBF (Mean Time Between Failures)**: Average time between system failures. For a 10,000 GPU cluster, you might see failures every few hours.
- **Availability**: Percentage of time the system is operational. Target: 99%+ for production clusters.
- **MTTR (Mean Time To Recovery)**: Average time to recover from a failure. Good clusters recover in minutes, not hours.

**Communication latency** is critical for distributed training. AllReduce latency should be:
- **Within a node (NVLink)**: < 1 ms for typical gradient sizes
- **Between nodes (InfiniBand)**: < 5 ms for cross-node communication
- **P99 latency**: The 99th percentile latency matters more than average—one slow node can stall the entire training job

When you're benchmarking a cluster, measure these metrics at different scales: 8 GPUs, 64 GPUs, 512 GPUs, 2048 GPUs. The metrics that degrade with scale (like linear scaling or communication efficiency) tell you where your bottlenecks are.

In the rest of this chapter, we'll cover the hardware details that make clusters work: GPU memory hierarchies, interconnect technologies, and how to choose parallelism strategies based on your cluster's topology.

## CPU: The Orchestrator

While GPUs do the heavy lifting in distributed training, CPUs play a crucial supporting role. Understanding CPU architecture helps you optimize data loading, manage GPU coordination, and debug performance bottlenecks.

### CPU Architecture Basics

CPUs are built around the **von Neumann architecture**: a central processing unit with arithmetic logic unit (ALU), control unit (CU), and memory unit (registers). Unlike GPUs optimized for throughput, CPUs optimize for latency—fast single-threaded execution with complex control logic.

The key difference: CPUs spend most of their silicon on control logic and cache, not compute units. A modern CPU might have 8-64 cores, each with complex out-of-order execution, branch prediction, and multi-level caches. GPUs have thousands of simple cores optimized for parallel workloads.

For distributed training, CPUs handle:
- **Data loading and preprocessing**: Reading from disk, decoding images, tokenizing text
- **Orchestration**: Launching GPU kernels, managing process groups, handling communication
- **System management**: Memory allocation, process scheduling, network stack

If your CPU is the bottleneck, GPUs sit idle waiting for data. This is why data loading pipelines matter—you need enough CPU cores and fast storage to keep GPUs fed.

### CPU-GPU Interaction

When you run distributed training, here's what happens:

1. **CPU launches GPU kernels**: Your Python code (running on CPU) calls PyTorch, which generates CUDA kernels. The CPU sends these to the GPU via PCIe.

2. **CPU manages memory**: CPU allocates GPU memory, transfers data from CPU RAM to GPU memory, and coordinates multi-GPU communication.

3. **CPU handles communication**: For multi-node training, CPU processes handle network communication (InfiniBand, Ethernet) and coordinate with NCCL for GPU collectives.

The PCIe connection between CPU and GPU is often a bottleneck. PCIe Gen 4 x16 gives you about 32 GB/s bidirectional, while NVLink between GPUs gives 300-1800 GB/s. This is why you want GPUs to communicate directly via NVLink, not through the CPU.

### NUMA and CPU Affinity

Modern servers have multiple CPU sockets (NUMA nodes). Each socket has its own memory controllers and PCIe lanes. GPUs connected to different sockets have different memory access patterns.

You can check NUMA topology:

```bash
numactl --hardware
```

For distributed training, try to keep processes on the same NUMA node as their GPUs. This reduces memory access latency. PyTorch doesn't do this automatically—you may need to set CPU affinity manually or use `numactl` when launching jobs.

### CPU Requirements for Distributed Training

For a typical 8-GPU server:
- **CPU cores**: You want at least 2-4 CPU cores per GPU for data loading and orchestration. An 8-GPU system should have 16-32 CPU cores minimum.
- **Memory**: CPU RAM should be 1.5-2x GPU memory for data staging. With 8×80GB GPUs, you want at least 1 TB CPU RAM.
- **PCIe lanes**: Each GPU needs PCIe x16. An 8-GPU system needs 128 PCIe lanes, which typically means dual-socket CPUs (AMD EPYC or Intel Xeon).

The CPU doesn't need to be the latest generation—it's not doing the compute. But it needs enough cores and PCIe bandwidth to keep GPUs busy.

## NVIDIA GPU: Architecture Evolution and Hardware

When you're building distributed training systems, the GPU architecture matters. NVIDIA has been iterating on GPU designs since 2010, and each generation brings changes that affect how you design your training pipeline. Here's what you need to know about the GPUs you're likely to encounter.

### Understanding GPU Memory and Compute Architecture

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

### Key Architecture Milestones

**Fermi (2010)** introduced the first complete GPU computing architecture with CUDA cores and ECC memory support. **Pascal (2016)** was the breakthrough for AI—it introduced NVLink (160 GB/s bidirectional), enabling multi-GPU systems that could actually communicate fast enough for distributed training.

**Volta (2017)** added Tensor Cores, specialized units for matrix multiplication that accelerated deep learning by 10-100x. This is when GPUs stopped being just graphics cards and became AI accelerators. **Ampere (2020)** with the A100 brought Tensor Core 3.0, supporting TF32 and BF16, plus NVLink 3.0 (600 GB/s) and NVSwitch for 8-GPU all-to-all connectivity.

**Hopper (2022)** with the H100 pushed things further: FP8 precision, Transformer Engine for dynamic precision switching, and NVLink 4.0 (900 GB/s). The H200 added more HBM3 memory (141 GB vs H100's 80 GB), which matters when you're training large models.

**Blackwell (2024)** is the current generation. The B200 doubles NVLink bandwidth to 1.8 TB/s and introduces dual-die design—each B200 chip is actually two dies connected internally. This means a single B200 has roughly the compute of two H100s, but with better memory bandwidth (8 TB/s vs 3 TB/s).

### What These Numbers Mean for Training

When you're choosing GPUs for distributed training, you care about three things:

**Memory capacity and bandwidth**: A 70B parameter model with FP16 weights needs about 140 GB just for the model. Add gradients, optimizer states (Adam uses 2x model size), and activations, and you're looking at 500+ GB per training step. The H100 has 80 GB HBM3, the H200 has 141 GB, and the B200 has 192 GB. More memory means larger batch sizes or fewer GPUs needed.

HBM bandwidth matters too. The A100 has 2 TB/s, H100 has 3 TB/s, and B200 has 8 TB/s. When you're doing gradient synchronization, higher bandwidth means less time waiting for data transfers.

**Compute throughput**: This is where Tensor Cores shine. The H100 delivers about 1 PFLOP for FP16, while the B200 hits 2.25 PFLOP. But raw FLOPS don't tell the whole story—you need to look at what precision you're actually using. FP8 training can be 2x faster than FP16, but not all models train well at FP8. The Transformer Engine in Hopper and Blackwell architectures automatically switches between FP8 and FP16 during training, which is why you see claims of "6x faster" for certain workloads.

**Interconnect bandwidth**: NVLink bandwidth determines how fast GPUs can synchronize gradients. A100 has 600 GB/s, H100 has 900 GB/s, and B200 has 1.8 TB/s. When you're doing data parallelism, you're doing AllReduce operations every step. Faster NVLink means less communication overhead.

### GPU Product Families

NVIDIA ships GPUs in different form factors depending on your needs:

**HGX (Hyperscale GPU eXchange)** is a baseboard module that OEMs integrate into servers. An HGX H100 has 8 H100 GPUs connected via NVSwitch, giving you 640 GB total HBM and 7.2 TB/s aggregate NVLink bandwidth. You buy this from server vendors like Dell, Supermicro, or Inspur, who add CPUs, storage, and networking.

**DGX (Deep GPU Xceleration)** is NVIDIA's complete system. A DGX H100 is a pre-integrated server with 8 H100s, AMD EPYC CPUs, NVMe storage, and InfiniBand networking. It's more expensive but comes with optimized software stack and support. DGX systems are what most AI companies use for training—they're tested, documented, and just work.

**SuperPOD** scales beyond a single server. A DGX SuperPOD connects multiple DGX systems (typically 32-64 nodes) via InfiniBand, creating a cluster with thousands of GPUs. The GB200 SuperPOD connects 8 GB200 NVL72 units (each with 72 GPUs) for a total of 576 GPUs with 1 PB/s NVLink bandwidth.

The GB200 NVL72 is interesting—it's a liquid-cooled rack unit with 36 GB200 superchips (72 GPUs total) connected via NVLink. NVIDIA markets it as a "single massive GPU" because the NVLink topology makes all 72 GPUs appear as one unified memory space. This is what you'd use for training trillion-parameter models.

### Choosing the Right GPU

For most distributed training, you'll be choosing between H100, H200, or B200. Here's the decision tree:

- **H100**: Still the workhorse. 80 GB memory, 3 TB/s bandwidth, 900 GB/s NVLink. Good balance of performance and availability. Use this if you're building a cluster now and need proven hardware.

- **H200**: Same compute as H100 but 141 GB memory. Use this if you're memory-bound—larger models, longer sequences, or when you want bigger batch sizes. The extra memory costs more but can reduce the number of GPUs you need.

- **B200**: Latest generation, 192 GB memory, 8 TB/s bandwidth, 1.8 TB/s NVLink. Use this for new deployments where you want maximum performance. The dual-die design means you get roughly 2x the compute of H100, but you'll pay for it.

One thing to watch: GPU availability. As of 2024, H100s are still hard to get, and B200s are even scarcer. If you're building a cluster, factor in lead times—it can take 6-12 months from order to delivery.

For inference, the calculus changes. B200's FP4 performance (20 PFLOP) makes it attractive for high-throughput inference, but the cost per request matters more than peak FLOPS. Many inference deployments still use A100 or even consumer GPUs because they're cheaper and good enough.

### Architecture Features That Matter

**Tensor Cores** are the secret sauce. They're specialized units that do matrix multiplication 10-100x faster than CUDA cores. Every modern training framework (PyTorch, TensorFlow, JAX) uses them automatically through cuBLAS and cuDNN. You don't need to write special code—just make sure you're using FP16/BF16/FP8 precision.

**Transformer Engine** (Hopper and Blackwell) automatically switches between FP8 and FP16 during training. It monitors activation statistics and uses FP8 when safe, FP16 when needed for accuracy. This is why you see claims of "6x faster training" for transformer models—the hardware is doing precision optimization automatically.

**MIG (Multi-Instance GPU)** on A100 and H100 lets you partition a single GPU into multiple virtual GPUs. Each partition gets dedicated memory and compute. This is useful for cloud providers who want to rent GPU time to multiple customers, but for training large models, you'll want the full GPU.

**NVLink-C2C** in Grace Hopper systems connects the CPU and GPU with 900 GB/s bandwidth. This lets the GPU access CPU memory directly, useful for models that don't fit in GPU memory. The Grace CPU has 512 GB LPDDR5X, so a GH200 system gives you 608 GB total addressable memory (96 GB GPU + 512 GB CPU).

When you're designing distributed systems, these architectural details determine your parallelism strategy. High NVLink bandwidth means tensor parallelism is viable. Large memory means you can fit bigger models or use fewer GPUs. Fast HBM means you can process larger batches without hitting memory bandwidth limits.

## Google TPU: An Alternative Architecture

While NVIDIA GPUs dominate the distributed training landscape, Google's Tensor Processing Unit (TPU) offers a different approach. TPUs are application-specific integrated circuits (ASICs) designed from the ground up for neural network workloads. If you're working at Google or using Google Cloud, you'll encounter TPUs. Understanding how they differ from GPUs helps when choosing hardware or porting code between platforms.

### Why TPU Exists

Google started designing TPUs in 2013 when they realized that running neural networks on CPUs was too expensive. Their prediction: if people used 3 minutes of voice search per day with neural network-based speech recognition, they'd need to double their datacenter capacity. CPUs couldn't scale cost-effectively, and GPUs at the time (2013) weren't optimized for neural networks.

TPU v1 shipped in 2016—just 15 months from design to deployment. That's fast for a chip. The first silicon worked without any mask changes, which is rare. The key insight: neural networks don't need the flexibility of CPUs or GPUs. They're mostly matrix multiplication, so you can build a chip that does one thing extremely well.

### TPU Architecture: Systolic Arrays

The core of a TPU is the **Matrix Multiply Unit (MXU)**, which uses a **systolic array** architecture. Unlike GPUs that use thousands of CUDA cores with registers and caches, a systolic array is a grid of processing elements (PEs) where data flows through the array like a heartbeat—hence "systolic."

Here's how it works: instead of storing intermediate results in registers and fetching them later, data flows directly from one PE to the next. Each PE multiplies two values and passes results to neighbors. This eliminates most memory accesses because data is reused as it flows through the array. For matrix multiplication, this is incredibly efficient—you read each input value once and reuse it many times.

TPU v1 had a 256×256 systolic array (65,536 PEs) running at 700 MHz, giving about 92 TOPS for INT8 operations. TPU v2 and later use 128×128 arrays but have multiple MXUs per chip. The systolic design means TPUs excel at dense matrix multiplication but aren't as flexible as GPUs for other operations.

### TPU Generations

**TPU v1 (2016)** was inference-only. It used INT8 quantization (weights and activations converted from FP32 to 8-bit integers), 8 GB DDR3 memory, and connected via PCIe 3.0. It was fast for inference but couldn't train models because INT8 isn't stable for gradient computation.

**TPU v2 (2017)** added training support. Key changes: HBM memory (16 GB, 600 GB/s bandwidth), BF16 support for training (bfloat16 keeps FP32's exponent range but reduces mantissa bits), and chip-to-chip interconnects. Four TPU v2 chips form a module, and 64 modules (256 chips total) form a TPU v2 Pod with 11.5 PFLOPS peak performance.

**TPU v3 (2018)** doubled MXU count (4 per chip vs 2 in v2), increased clock speed (940 MHz vs 700 MHz), and doubled HBM capacity (32 GB). The v3 Pod scales to 1,024 chips with 100+ PFLOPS. It also switched to liquid cooling, which allowed higher power (450W vs 280W) and better performance.

**TPU v4 (2021)** moved to 7nm process, doubled MXUs again (8 per chip), and added **Sparse Core** units for embedding layers. The v4 Pod uses 3D torus topology (vs 2D in v2/v3) to connect 4,096 chips, delivering 1.1 exaflops of BF16 compute. It also introduced optical circuit switching (OCS) for chip-to-chip communication, reducing latency and power compared to electrical switches.

### TPU Pod Architecture

A **TPU Pod** is Google's term for a TPU cluster. Unlike GPU clusters that use InfiniBand switches, TPU Pods use custom interconnects:

- **2D Torus** (v2/v3): Chips arranged in a 2D grid where each chip connects to four neighbors. Data can wrap around edges, forming a torus. This gives high bandwidth between adjacent chips but longer paths for distant communication.

- **3D Torus** (v4): Chips arranged in a 3D cube. Each chip connects to six neighbors (up/down, left/right, forward/back). This reduces network diameter compared to 2D, meaning fewer hops for distant communication. A 4×4×4 cube (64 chips) is the basic unit, and 64 cubes form a 4,096-chip Pod.

- **Optical Circuit Switching (OCS)**: TPU v4 uses MEMS-based optical switches (Palomar) to route light signals between chips. This avoids electrical-to-optical conversion, reducing latency and power. The OCS can reconfigure routes dynamically, which helps with fault tolerance—if a chip fails, routes can be reconfigured around it.

The torus topology is different from GPU clusters' Clos/fat-tree networks. Torus is cheaper (fewer switches, simpler wiring) and has lower latency for local communication, but it's less flexible for scaling and load balancing. Clos networks are non-blocking (any input can talk to any output at full bandwidth simultaneously), while torus networks can have congestion.

### TPU vs GPU: When to Use Which

**Use TPUs if:**
- You're at Google or using Google Cloud Platform
- Your workload is mostly dense matrix multiplication (transformers, CNNs)
- You want maximum performance for specific models (Google optimizes TPU software stack for their models)
- You're training at Google scale (thousands of chips)

**Use GPUs if:**
- You need flexibility (different model architectures, research)
- You're using PyTorch (TPU support exists but GPU is first-class)
- You need to run on-premise or multi-cloud
- Your workload has sparse operations or irregular patterns

**Performance characteristics:**
- TPUs excel at dense matrix ops. A TPU v4 chip delivers about 275 TOPS for BF16, comparable to an H100's FP16 performance.
- GPUs are more general-purpose. They handle sparse operations, custom kernels, and mixed workloads better.
- TPU software stack (XLA compiler) is highly optimized but less flexible. You compile your model to XLA, and the compiler generates optimized code. This can be faster than GPU for supported operations but harder to debug.

**Cost and availability:**
- TPUs are only available on Google Cloud. You can't buy them.
- GPU pricing varies by cloud provider and availability. H100s are expensive but available from multiple vendors.
- TPU pricing is per-hour on GCP. For large-scale training, TPUs can be cost-effective if your workload fits.

### TPU Programming Model

TPUs use **XLA (Accelerated Linear Algebra)** compiler. You write code in TensorFlow or JAX, and XLA compiles it to TPU instructions. This is different from GPUs where you write CUDA kernels or use libraries like cuDNN.

The compilation step means TPUs have higher startup latency—your first run compiles the graph, which can take minutes. Subsequent runs are fast. GPUs have lower startup latency but may have more runtime overhead.

For distributed training on TPUs, you use TensorFlow's distribution strategies or JAX's `pmap`/`pjit`. The torus topology means communication patterns matter—you want to keep communication local when possible. XLA's compiler optimizes for this automatically, but understanding the topology helps when debugging performance.

### Sparse Core: TPU v4's Secret Weapon

TPU v4 introduced **Sparse Core** units specifically for embedding layers. Embedding layers are common in recommendation systems and NLP—they map discrete IDs (user IDs, word IDs) to dense vectors. The computation is sparse (most entries are zero) and doesn't map well to matrix multiplication units.

Sparse Core has 16 tiles, each with its own HBM channel and a programmable vector processing unit. It can fetch sparse data, process it, and flush results back efficiently. Google claims Sparse Core accelerates embedding-heavy models by 5-7x while using only 5% of die area and power.

This is an example of algorithm-hardware co-design: Google identified that embeddings are a bottleneck, so they built dedicated hardware. GPUs handle embeddings in software, which works but isn't as efficient.

If you're training recommendation models or models with large embedding tables, TPU v4's Sparse Core is a significant advantage. For transformer-only models, it doesn't matter as much.

## NPU: Domain-Specific AI Accelerators

While GPUs and TPUs dominate large-scale training, **Neural Processing Units (NPUs)** represent a different approach: domain-specific architecture (DSA) chips optimized for AI workloads. NPUs are ASICs (Application-Specific Integrated Circuits) designed from the ground up for neural network operations, trading general-purpose flexibility for efficiency.

### What Makes NPUs Different

NPUs are built around **AI Cores**—specialized units optimized for matrix multiplication, convolution, and other neural network primitives. Unlike GPUs that evolved from graphics hardware, NPUs are designed specifically for AI from day one.

The architecture tradeoff: CPUs are general-purpose (good at everything, great at nothing), GPUs are parallel processors (great at parallel workloads, less flexible), and NPUs are domain-specific (excellent at AI, limited elsewhere). NPUs allocate most of their silicon to AI Cores and memory bandwidth, with minimal control logic.

Major NPU vendors include:
- **Huawei Ascend**: Used in Huawei's Atlas servers and Cloud services. The Ascend 910 is designed for training, while Ascend 310 targets inference.
- **Cambricon MLU**: Chinese company focusing on edge and cloud AI acceleration.
- **Tesla Dojo**: Custom NPU for Tesla's autonomous driving training.
- **Google Edge TPU**: Smaller TPU variant for edge devices.

### NPU Architecture: AI Cores and Memory

NPU architecture centers on **AI Cores**—dedicated compute units for neural network operations. Each AI Core typically includes:
- **Matrix multiplication units**: Optimized for GEMM (General Matrix Multiply) operations
- **Vector processing units**: For element-wise operations, activations, normalization
- **Specialized units**: For operations like pooling, convolution, attention

Memory hierarchy is critical. NPUs use high-bandwidth memory (HBM) similar to GPUs, but the memory subsystem is often simpler—fewer cache levels, more direct paths to compute units. This reduces latency but requires careful memory management.

Huawei's Ascend 910, for example, has 32 GB HBM2 with 1.6 TB/s bandwidth and delivers about 256 TFLOPS for FP16. The architecture uses a "DaVinci" core design with multiple AI Cores per chip.

### Training vs Inference NPUs

Like GPUs, NPUs come in training and inference variants:

**Training NPUs** (like Ascend 910) need:
- High precision support (FP32, BF16, FP16) for stable gradient computation
- Large memory capacity for model parameters, gradients, and optimizer states
- High-bandwidth interconnects for multi-chip training
- Flexibility to support various model architectures

**Inference NPUs** (like Ascend 310, Edge TPU) prioritize:
- Lower precision (INT8, INT4) for efficiency
- Lower power consumption for edge deployment
- Lower latency for real-time applications
- Cost efficiency for mass deployment

The same chip rarely excels at both. Training requires flexibility and precision; inference requires efficiency and low cost.

### NPU Software Stack

NPU software stacks are typically more proprietary than GPU ecosystems. Huawei's Ascend uses **MindSpore** framework and **CANN** (Compute Architecture for Neural Networks) runtime. Unlike CUDA which works across NVIDIA GPUs, NPU software is often vendor-specific.

This creates a lock-in risk: code written for Ascend NPUs won't run on Cambricon MLUs or other NPUs without significant porting. The ecosystem is fragmented compared to CUDA's dominance in the GPU space.

However, some frameworks are trying to abstract this. PyTorch has experimental support for some NPU backends, and ONNX Runtime can target multiple NPU vendors. But the experience isn't as seamless as GPU development.

### NPU vs GPU: When to Choose Which

**Choose NPUs if:**
- You're in China and need domestic alternatives to NVIDIA GPUs (export restrictions)
- You have specific workloads that NPUs optimize for (e.g., Ascend's optimizations for certain model types)
- You're building edge devices where power efficiency matters more than flexibility
- You're working with vendors who provide NPU-optimized solutions

**Choose GPUs if:**
- You need flexibility (research, different model architectures)
- You want the largest ecosystem (PyTorch, TensorFlow, JAX all have first-class GPU support)
- You need to run on multiple clouds or on-premise
- You're doing general-purpose ML work, not just specific NPU-optimized workloads

**Performance comparison**: NPUs can match or exceed GPUs for specific workloads they're optimized for. Huawei claims Ascend 910 training performance is comparable to A100 for certain models. But GPUs have broader model support and better software ecosystem.

**Cost**: NPU pricing varies by vendor and region. In China, Ascend NPUs can be more cost-effective than imported GPUs due to trade restrictions. Elsewhere, GPU ecosystem maturity often makes GPUs the better choice.

### NPU Interconnects and Scaling

Like GPUs, NPUs need high-bandwidth interconnects for distributed training. Huawei uses **HCCS** (Huawei Collective Communication Service) and custom interconnects. Ascend clusters can scale to thousands of chips, similar to GPU clusters.

The interconnect topology matters. Ascend uses a hierarchical architecture with chip-to-chip, node-to-node, and cluster-level interconnects. Understanding this topology helps when designing distributed training strategies.

One challenge: NPU interconnects are often proprietary. Unlike InfiniBand which is an open standard, NPU interconnects are vendor-specific. This can make multi-vendor clusters difficult.

### The NPU Landscape

The NPU market is fragmented. Unlike GPUs where NVIDIA dominates, NPUs have multiple vendors with different architectures:

- **Huawei Ascend**: Strong in China, used in Huawei Cloud and Atlas servers
- **Cambricon**: Focuses on edge and cloud inference
- **Graphcore IPU**: UK company with a different architecture (not strictly an NPU but similar)
- **Tesla Dojo**: Custom solution for Tesla's specific needs

This fragmentation means less software support, fewer frameworks, and more vendor lock-in. But it also means innovation and competition, which can drive better performance for specific use cases.

For distributed training, NPUs are viable but require more vendor-specific knowledge than GPUs. If you're building a new cluster and have access to both, GPUs are usually the safer choice due to ecosystem maturity. But NPUs can be compelling for specific regions, workloads, or cost constraints.

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

## Chip Programming Systems: SPMD and CUDA

Understanding how chips are programmed helps you write efficient distributed training code. The programming model (how you write code) and execution model (how hardware runs it) are different layers, and knowing both helps when debugging performance or porting code between platforms.

### Programming Models vs Execution Models

**Programming models** are abstractions for developers. They define how you structure code, what concepts you use (threads, blocks, kernels), and how you express parallelism. You write code using the programming model.

**Execution models** describe how hardware actually runs your code. The hardware might execute SIMD instructions, but you program it using threads. The compiler bridges this gap.

For distributed training, you're usually working at the programming model level (PyTorch, TensorFlow, JAX), but understanding the execution model helps when things go wrong or when you need to optimize.

### SPMD: Single Program, Multiple Data

**SPMD (Single Program, Multiple Data)** is the programming model that CUDA uses. The idea: you write one program (kernel), and it runs on multiple threads, each processing different data.

Here's a simple CUDA kernel that adds two vectors:

```c
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

You launch this with:

```c
vectorAdd<<<numBlocks, threadsPerBlock>>>(A, B, C, N);
```

Each thread executes the same code (`vectorAdd`), but `threadIdx.x` gives each thread a unique ID, so they process different array elements. This is SPMD: one program, multiple data elements.

SPMD is different from SIMD (Single Instruction, Multiple Data). SIMD is an execution model—hardware executes one instruction on multiple data elements simultaneously. SPMD is a programming model—you write code as if each thread is independent, even though hardware might execute them in SIMD fashion.

### CUDA's Execution Model: SIMT

NVIDIA GPUs execute SPMD programs using **SIMT (Single Instruction, Multiple Thread)**. Here's how it works:

**Thread hierarchy**: CUDA organizes threads into a hierarchy:
- **Thread**: The smallest unit. Each thread has its own registers and can execute independently.
- **Warp**: A group of 32 threads that execute together. This is the hardware scheduling unit.
- **Block**: A group of threads (typically 128-1024) that can share memory and synchronize.
- **Grid**: A collection of blocks that execute the same kernel.

**Warp execution**: When you launch a kernel, the GPU groups threads into warps. All 32 threads in a warp execute the same instruction simultaneously (SIMD-style), but each thread operates on different data. If threads in a warp take different branches (divergence), the warp executes both paths sequentially, which hurts performance.

**Fine-grained multithreading (FGMT)**: GPUs use FGMT to hide memory latency. When one warp is waiting for memory, the scheduler switches to another warp that's ready to execute. This keeps the execution units busy even when individual warps are stalled.

This is why GPU utilization matters. If you have enough warps, the GPU can hide latency by switching between them. If you don't have enough parallelism, warps sit idle waiting for memory, and utilization drops.

### Why SIMT Over SIMD?

SIMT (what GPUs use) is more flexible than traditional SIMD (what CPUs use for vectorization):

**Data alignment**: SIMD requires data to be aligned and contiguous. SIMT doesn't—each thread can access different memory locations independently. This makes irregular memory patterns easier to handle.

**Branch divergence**: In SIMD, if one element takes a different branch, you execute both paths and mask results. SIMT handles this more gracefully—threads can diverge, though it still costs performance.

**Programming model**: SIMT lets you write scalar code (one thread, one element) that gets compiled to SIMD execution. You don't need to manually vectorize or think about vector widths.

**Dynamic grouping**: SIMT hardware dynamically groups threads into warps. You don't need to know the warp size when writing code—the hardware handles it.

### CUDA Thread Indexing

Understanding thread indexing is crucial for writing correct CUDA kernels. Each thread has identifiers:

- `threadIdx.x/y/z`: Thread's position within its block (0 to blockDim.x-1)
- `blockIdx.x/y/z`: Block's position within the grid
- `blockDim.x/y/z`: Number of threads per block (set at launch)
- `gridDim.x/y/z`: Number of blocks in the grid

To compute a global thread ID for a 1D grid:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

For a 2D grid (common for image processing):

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

The key insight: you use these indices to determine which data each thread processes. If you have N elements and launch M threads, thread i processes element i (with bounds checking).

### Memory Hierarchy in CUDA

CUDA exposes a memory hierarchy that maps to hardware:

- **Registers**: Fastest, private to each thread. Limited quantity (typically 64KB per SM).
- **Shared memory**: Fast, shared within a block. Used for communication and caching. Typically 48KB or 96KB per SM.
- **Global memory**: Slow but large. All threads can access it. This is GPU DRAM (HBM).
- **Constant memory**: Read-only, cached. Good for values that don't change.
- **Texture memory**: Cached, optimized for 2D access patterns.

For distributed training, you're mostly using global memory (model weights, activations, gradients). But understanding shared memory helps when writing custom kernels or optimizing data loading.

### How Frameworks Use CUDA

When you write PyTorch code like:

```python
output = torch.matmul(input, weight)
```

PyTorch doesn't generate CUDA kernels on the fly. Instead, it calls pre-compiled kernels from libraries like cuBLAS (for matrix multiplication) or cuDNN (for convolutions). These libraries are highly optimized and use techniques like:

- **Kernel fusion**: Combining multiple operations into one kernel to reduce memory traffic
- **Tile-based algorithms**: Breaking large matrices into tiles that fit in shared memory
- **Tensor Core usage**: Automatically using Tensor Cores when available

You rarely write CUDA kernels directly for distributed training. But understanding how CUDA works helps when:
- Debugging performance issues (why is my GPU utilization low?)
- Writing custom operations (maybe you need a fused kernel)
- Understanding framework limitations (why can't PyTorch do X?)

### SPMD in Distributed Training

SPMD extends naturally to distributed training. Each GPU runs the same program (your training script), but processes different data:

- **Data parallelism**: Each GPU gets a different batch. Same model, different data.
- **Model parallelism**: Each GPU gets different model layers. Same data, different model parts.

The communication primitives (AllReduce, AllGather, etc.) coordinate between GPUs, but each GPU still executes the same program structure.

This is why distributed training frameworks (DDP, FSDP) feel similar to single-GPU training—you're still writing SPMD code, just with communication added.

### AMD and Other Alternatives

AMD GPUs use a different execution model. AMD's CDNA architecture (MI300X) uses **SIMD execution units** rather than SIMT. Each Compute Unit (CU) has 4 SIMD units, and the scheduler picks which SIMD unit to use.

ROCm (AMD's CUDA alternative) provides a CUDA-like programming interface, but the hardware execution is different. This can lead to performance differences—code optimized for NVIDIA GPUs might not run as well on AMD GPUs.

For distributed training, stick with NVIDIA if possible. The ecosystem (CUDA, cuDNN, NCCL) is mature and well-optimized. AMD is catching up, but NVIDIA still has the advantage in software support.


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
