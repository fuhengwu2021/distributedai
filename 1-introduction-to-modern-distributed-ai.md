# Chapter 1: Introduction to Modern Distributed AI

**Chapter Length:** 28 pages

## Overview

Modern AI models have grown beyond what single GPUs can handle. Large language models now range from several billion to over a trillion parameters. Training models with tens of billions of parameters on a single GPU would take months, if they even fit in memory. Serving these models at scale requires distributed architectures.

This chapter covers resource estimation, decision frameworks for choosing between distributed training, fine-tuning, or inference, and practical examples to get started. The focus is on making informed decisions and running distributed workloads effectively.

---

## 1. Why Modern AI Requires Distribution

A few years ago, you could train most models on a single GPU. ResNet-50 on ImageNet took a couple of days. Today, training a 70B parameter language model on a single GPU would take months, if it even fits in memory. The models got bigger, the datasets got bigger, and single-GPU training became impractical.

Looking at recent models, the scale is clear. GPT-4 has over 1 trillion parameters. Training it requires thousands of GPUs working together. Even smaller models like Llama 2 (70B parameters) need multiple GPUs just to fit in memory, let alone train efficiently.

| Model Name | Parameters | Company | Year |
|------------|------------|---------|------|
| ViT-22B | 22B | Google | 2023 |
| Sora | 30B | OpenAI | 2023 |
| Grok-1 | 314B | xAI | 2023 |
| Gemini-1 | 1.6T | Google | 2023 |
| LLaMA-2 | 700B | Meta | 2023 |
| PanGu-Σ | 1.085T | Huawei | 2023 |
| DeepSeek-1 | 6.7B | DeepSeek | 2023 |
| Claude-3 | undisclosed (~>1T MoE est.) | Anthropic | 2024 |
| GPT-4V | 1.8T | OpenAI | 2024 |
| DeepSeek-V2 | 236B MoE (16 experts, 2 active) | DeepSeek | 2024 |
| DeepSeek-Coder-V2 | 236B MoE | DeepSeek | 2024 |
| Grok-4 | ~1.7T (MoE) | xAI | 2025 |
| Qwen-Max | >1T | Alibaba | 2025 |
| GPT-4.5 | undisclosed | OpenAI | 2025 |
| GPT-5 | ~2–5T | OpenAI | 2025 |
| DeepSeek-V3 | 671B MoE (64 experts, 8 active) | DeepSeek | 2025 |
| DeepSeek-R1 | 671B MoE (reasoning-optimized) | DeepSeek | 2025 |

### The Scale Challenge

Take a 70B parameter model as an example. In full precision (FP32), the model weights alone need 280GB of memory. An A100 GPU has 80GB. You can't even load the model, let alone train it.

Training these models takes thousands of GPU-hours. A single GPU training run would take months. The datasets are massive too—trillions of tokens. Loading and preprocessing this data efficiently requires distributed pipelines.

The mismatch is clear: model size and compute requirements have grown exponentially, while single-GPU memory and compute have grown linearly at best.

### Estimating Model Resource Requirements

Before you start training or deploying, you need to know how much memory and compute you'll need. Get this wrong, and you'll hit out-of-memory errors or waste money on over-provisioned infrastructure.

The memory footprint depends on what you're storing. For model weights alone, the calculation is straightforward. Each parameter in FP32 takes 4 bytes, FP16/BF16 takes 2 bytes, Int8 takes 1 byte, and Int4 takes 0.5 bytes. For a 7B parameter model, that's 28 GB in FP32, 14 GB in FP16, 7 GB in Int8, and 3.5 GB in Int4.

But model weights are just the start. During training, you also need space for gradients, optimizer states, and activations. For inference, you need KV cache for attention mechanisms. The total memory requirement can be several times larger than just the model weights.

#### Training Memory Requirements

Training needs way more memory than inference. You're storing model weights, gradients (one per parameter), optimizer states, and activations from the forward pass. With Adam optimizer, you need 2× the model size for optimizer states (momentum and variance). SGD is lighter—just the learning rate. Activations depend on batch size and sequence length, but typically add 8-16 GB for a 7B model.

Training a 7B model with FP16 requires about 14 GB for model weights, another 14 GB for gradients, 28 GB for optimizer states with Adam, and 8-16 GB for activations. That's 64-72 GB total per GPU. That's why a 7B model needs at least an A100 (80GB) for training, even with mixed precision. Smaller GPUs won't cut it.

#### Inference Memory Requirements

Inference is simpler. You just need the model weights and KV cache for attention. The KV cache size depends on your batch size, sequence length, and model architecture. For a 70B model with FP16, you're looking at 140 GB for model weights and another 20-40 GB for KV cache with a batch size of 32 and sequence length of 2048. That's 160-180 GB total. That's why inference for large models needs multiple GPUs or model parallelism. A single A100 won't hold it.

#### GPU Requirements Estimation

For training, calculate your memory needs, add 10-20% safety margin for communication buffers and framework overhead, then see if it fits. A 13B model with FP16 needs about 72 GB per GPU. With safety margin, that's 85 GB. An A100 has 80 GB, so you'll need 2 GPUs with model parallelism or FSDP.

For inference, it's similar. Calculate model size plus KV cache. A 70B model in FP16 needs 140 GB just for weights. With KV cache, you're looking at 160-180 GB total. You'll need 2+ A100 GPUs, or use Int8 quantization to get it down to 70 GB for weights, which might fit on one GPU with careful KV cache management.

#### Real-World Considerations

Don't forget the overhead. PyTorch adds 1-2 GB. The OS needs 5-10 GB. Distributed training needs 2-5 GB per GPU for communication buffers. Checkpointing causes temporary spikes. Start conservative and add 20-30% buffer to your estimates. Use mixed precision (FP16/BF16) to cut memory in half compared to FP32. Monitor with `nvidia-smi` to see actual usage. For inference, Int8 quantization can halve memory with minimal accuracy loss. And remember, activations scale linearly with batch size—if you hit OOM, reduce batch size first.

**Quick Reference Table:**

| Model Size | FP32 Weights | FP16 Weights | Training (FP16+Adam) | Inference (FP16) |
|------------|--------------|--------------|----------------------|------------------|
| 1B         | 4 GB         | 2 GB         | ~8 GB                | 2-4 GB           |
| 7B         | 28 GB        | 14 GB        | ~60-70 GB            | 14-20 GB         |
| 13B        | 52 GB        | 26 GB        | ~110-130 GB          | 26-35 GB         |
| 70B        | 280 GB       | 140 GB       | ~600-700 GB          | 140-180 GB       |

*Note: Training estimates assume Adam optimizer and moderate batch size. Actual values vary based on architecture, sequence length, and batch size.*

### The Evolution from Classic ML to Foundation Models

Classic machine learning models were designed to fit on a single machine. Traditional ML models had millions of parameters and were trained on datasets that fit in memory. The deep learning era brought models with hundreds of millions of parameters, requiring GPUs but still manageable on single devices. Today's foundation model era has models with billions to trillions of parameters, requiring distributed systems from day one.

The shift to distributed AI has enabled breakthrough capabilities—models that can understand and generate human-like text, code, and multimodal content. It has also driven enterprise adoption, with companies deploying AI at scale for production workloads, and accelerated research through faster iteration cycles enabled by parallel experimentation.

---

## 2. Training vs Inference vs Serving

Understanding the fundamental differences between training, inference, and serving is crucial for designing effective distributed systems. Each has distinct requirements, bottlenecks, and optimization strategies.

### Training: The Learning Phase

Training is about learning model parameters from data. The process follows a pattern: forward pass through the model, loss computation, backward pass to compute gradients, and gradient update to adjust parameters. This happens iteratively over multiple epochs until the model converges.

Training requires storing activations, gradients, and optimizer states in memory. The compute is intensive and iterative. In distributed training, you need frequent gradient synchronization across devices to keep all model copies consistent. The main challenges are gradient synchronization overhead, memory constraints for large models, long training times that can span days to weeks, and the need for fault tolerance and checkpointing.

Training a 7B parameter model on 1 trillion tokens typically requires 8 A100 GPUs (80GB each) and about 2 weeks of continuous training. Careful gradient synchronization is needed to maintain training stability across all GPUs.

### Inference: The Prediction Phase

Inference is about generating predictions from a trained model. Unlike training, inference only needs a forward pass—no gradients, no backward pass, no optimizer states. Memory requirements are lower: just model weights and KV cache for attention mechanisms. The compute intensity per request is lower, but you need high throughput to serve many requests simultaneously. Communication is minimal, mostly only for distributed inference.

The key challenges are low latency requirements (sub-second for interactive applications), high throughput (thousands of requests per second), efficient memory usage through KV cache management, and effective batching and scheduling strategies. Serving a 70B parameter model for chat applications requires optimized inference engines like vLLM or SGLang, continuous batching to maximize GPU utilization, and careful KV cache management to handle variable-length sequences.

### Serving: The Production System

Serving is about providing reliable, scalable access to models. It's not just running inference—it's building a production system with a model runner, API gateway, load balancer, and monitoring. The requirements are high availability, fault tolerance, and observability. At scale, you're dealing with multi-model, multi-tenant systems.

The key challenges are system reliability and uptime, multi-model routing and load balancing, cost optimization through GPU utilization and autoscaling, and observability for debugging. A production LLM serving platform might include multiple model variants (different sizes, fine-tuned versions), A/B testing infrastructure, canary deployment pipelines, and distributed tracing and monitoring.

### Comparison Table

| Aspect | Training | Inference | Serving |
|--------|----------|-----------|---------|
| **Primary Goal** | Learn parameters | Generate predictions | Provide access |
| **Memory Usage** | High (activations + gradients) | Medium (weights + KV cache) | Variable |
| **Compute Pattern** | Iterative, intensive | Single forward pass | Request-driven |
| **Communication** | Frequent (gradients) | Minimal | API-level |
| **Latency Requirement** | Hours to days | Milliseconds to seconds | Milliseconds |
| **Throughput Focus** | Samples per second | Tokens per second | Requests per second |

---

## 3. Decision Framework: When Do You Need Distributed Systems?

The question isn't whether distributed systems are cool—it's whether you actually need them. Distributed training adds complexity, communication overhead, and cost. Use it when you have to, not when you want to.

### When Do You Need Distributed Training?

Three scenarios force you into distributed training:

First, the model doesn't fit. Calculate model size in FP16 (parameters × 2 bytes). If that's more than 80% of your GPU memory, you need model parallelism or FSDP. A 13B model needs 26GB just for weights. With Adam optimizer, you're looking at 72GB total. An A100 has 80GB, so you're cutting it close. You'll need 2+ GPUs.

Second, training takes too long. If single-GPU training would take weeks or months, and you need faster iteration, go distributed. Training a 7B model on 1T tokens takes about 2 weeks on 8 GPUs. On 1 GPU, it's months.

Third, the dataset is too large. If data loading becomes the bottleneck, or you can't fit the dataset on a single node, use data parallelism. Multi-terabyte datasets benefit from distributed data loading.

When is single-GPU enough? If the model fits comfortably with room for activations, training completes in hours to days, the dataset fits locally, and budget favors single-GPU solutions, stick with one GPU.

### When Do You Need Distributed Fine-tuning?

Fine-tuning is usually lighter than full training, but you might still need distribution.

If the base model is too large for a single GPU, you need model parallelism. Fine-tuning a 70B model requires splitting it across multiple GPUs, just like training.

Large fine-tuning datasets benefit from distributed data loading. If you're fine-tuning on a domain-specific corpus with millions of examples, multiple GPUs help.

Parameter-efficient methods like LoRA and QLoRA change the equation. QLoRA on a 70B model can fit in a 48GB GPU because it only trains adapter weights. Full fine-tuning of the same model needs multiple GPUs. If you can use LoRA/QLoRA, you can often stick with a single GPU.

### When Do You Need Distributed Inference?

Inference has different constraints than training. You need distribution when:

The model doesn't fit. A 70B model in FP16 needs 140GB just for weights. With KV cache, you're at 160-180GB. That's 2+ A100 GPUs minimum.

Throughput exceeds single GPU capacity. If you need to serve thousands of requests per second, a single GPU won't cut it. A chat application with 10,000 concurrent users needs multiple GPUs or nodes.

You need low latency at high throughput. Real-time services need sub-second latency while handling many requests. This often requires tensor parallelism or multiple inference instances.

When is single-GPU inference enough? If the model fits with room for KV cache, throughput is within single GPU capacity, latency requirements are met, and cost favors single-GPU deployment, stick with one GPU. Use optimized engines like vLLM or SGLang to maximize utilization.

### Decision Tree: Quick Reference

```
Start: What is your use case?
│
├─ Training from scratch?
│  ├─ Model > 60GB? → Distributed Training (FSDP/Model Parallel)
│  ├─ Training time too long? → Distributed Training (Data Parallel)
│  └─ Both OK? → Single GPU
│
├─ Fine-tuning?
│  ├─ Base model > GPU memory? → Distributed Fine-tuning
│  ├─ Using LoRA/QLoRA? → Usually single GPU OK
│  └─ Large dataset? → Consider distributed
│
└─ Inference/Serving?
   ├─ Model > GPU memory? → Distributed Inference (Model Parallel)
   ├─ High throughput needed? → Distributed Inference (Multiple GPUs)
   └─ Both OK? → Single GPU with optimization (vLLM/SGLang)
```

### Real-World Examples

Consider a startup training a 7B model. The model size is 14GB in FP16, and they have one A100 with 80GB. The model fits with room for training overhead, so a single GPU is sufficient. They use mixed precision and gradient checkpointing if needed.

An enterprise fine-tuning a 70B model faces a different situation. The model size is 140GB in FP16, exceeding a single GPU's capacity. With 4 A100 GPUs available, they need distributed fine-tuning. They use FSDP or model parallelism, and might consider QLoRA to reduce memory requirements.

For production inference with a 13B model, the requirements are 1000 requests per second with less than 500ms latency. The model size is 26GB in FP16. With 2 A100 GPUs, distributed inference is needed to meet the throughput requirement. They use vLLM with tensor parallelism or multiple instances.

A research lab training a 1B model has a simpler setup. The model is only 2GB in FP16, the dataset is 100GB, and they have one RTX 4090 with 24GB. The model and data fit comfortably, so a single GPU is sufficient. They use a standard training pipeline.

---

## 4. PyTorch Distributed Fundamentals

Before diving into distributed training code, you need to understand the basic concepts and APIs that PyTorch provides. This section covers the essential building blocks that all distributed code relies on.

### Process Groups and Ranks

In distributed training, multiple processes work together. Each process runs on a different GPU or node. PyTorch organizes these processes into a process group. Within a process group, each process has a unique rank—an integer identifier starting from 0. The total number of processes is called the world size.

Think of it like this: if you have 4 GPUs, you'll have 4 processes. Process 0 runs on GPU 0, process 1 on GPU 1, and so on. The rank tells each process which GPU it should use and which part of the data it should process.

To check your GPU setup, you can use the code in `code/chapter1/ch01_check_cuda.py`:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

### Initializing the Process Group

Before any distributed operations, you must initialize the process group. This tells PyTorch how processes should communicate. The most common backend for GPU training is NCCL (NVIDIA Collective Communications Library).

Here's the basic initialization pattern:

```python
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPU communication
        rank=rank,       # This process's rank
        world_size=world_size  # Total number of processes
    )
    torch.cuda.set_device(rank)  # Set which GPU this process uses
```

The simplest test to verify your distributed setup works is in `code/chapter1/ch01_distributed_basic_test.py`. This is a basic distributed test that verifies process group initialization and communication. It doesn't use DDP—it just tests that multiple processes can communicate.

For multi-GPU testing:
```bash
torchrun --nproc_per_node=2 code/chapter1/ch01_distributed_basic_test.py
```

For single-GPU simulation (useful for testing without multiple GPUs), use `code/chapter1/ch01_multi_gpu_simulation.py`:
```bash
# Simulate 2 processes on GPU 0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_simulation.py
```

When you run either script, you should see "Rank 0 says hello" and "Rank 1 says hello" printed from different processes. The single-GPU simulation mode is helpful for testing distributed code logic before running on actual multi-GPU setups.

### DistributedDataParallel (DDP)

DDP is PyTorch's way of wrapping a model for distributed training. When you wrap a model with DDP, PyTorch automatically handles gradient synchronization across all processes. Each process computes gradients on its local data, then DDP averages these gradients across all processes before updating the model.

The key insight is that DDP assumes each process has a complete copy of the model. The model itself isn't split—only the data is partitioned. Each process trains on a different subset of the data, but all processes maintain identical model parameters after each training step.

Here's how you wrap a model:

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = YourModel().cuda(rank)
model = DDP(model, device_ids=[rank])
```

After wrapping, you use the model exactly as you would in single-GPU training. DDP handles the synchronization behind the scenes during `loss.backward()`.

### DistributedSampler

Since each process should train on different data, you need a DistributedSampler. It splits the dataset so each process gets a unique subset. Without it, all processes would see the same data, defeating the purpose of distributed training.

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset, 
    num_replicas=world_size,  # Total number of processes
    rank=rank  # This process's rank
)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

Important: you must call `sampler.set_epoch(epoch)` at the start of each epoch to ensure data shuffling works correctly across epochs.

### Launching Distributed Jobs

You can launch distributed training in two ways. The modern approach uses `torchrun`, which handles process spawning automatically:

```bash
torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_ddp.py
```

This launches 2 processes on the current machine. For multi-node training, you'd specify `--nnodes`, `--node_rank`, and `--master_addr` as well.

The alternative is using `torch.multiprocessing.spawn()` directly in your code, which is what `ch01_multi_gpu_ddp.py` does internally.

### Common Pitfalls

There are several mistakes that trip up beginners. The code in `code/chapter1/ch01_ddp_pitfalls.py` shows the wrong and right ways:

First, all processes must use the same `MASTER_PORT`. If each process uses a different port, they can't communicate. Set it once before initialization, not per-process.

Second, always use DistributedSampler with your DataLoader. Without it, each process sees all the data, which means you're not actually doing distributed training—just running the same training multiple times.

Third, call `sampler.set_epoch(epoch)` in your training loop. This ensures data shuffling works correctly. Without it, each epoch uses the same data order.

---

## 5. Quick Start: Your First Distributed Workloads

Now that you understand the basics, let's run some actual distributed training code. All examples in this section correspond to files in `code/chapter1/`.

### Prerequisites Check

First, verify your environment using `code/chapter1/ch01_check_cuda.py`:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

**Requirements:**
- PyTorch with CUDA support
- 2+ GPUs (for distributed examples)
- NCCL (usually included with PyTorch)

### Quick Start 1: Single-GPU Baseline

**File:** `code/chapter1/ch01_single_gpu_baseline.py`

Before comparing distributed training, establish a single-GPU baseline. This script trains a simple model on one GPU and measures training time and memory usage. Run it to get baseline metrics:

```bash
python code/chapter1/ch01_single_gpu_baseline.py
```

This gives you a reference point for comparing distributed training performance.

### Quick Start 2: Multi-GPU Distributed Training

**File:** `code/chapter1/ch01_multi_gpu_ddp.py`

This is a complete distributed training example using DDP. It includes proper setup, DistributedSampler usage, and cleanup. The code shows the full pattern you'll use in real training jobs.

**Run it:**
```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_ddp.py

# Or use the launch script
bash code/chapter1/ch01_launch_torchrun.sh
```

The launch script contains the torchrun command with proper arguments, making it easier to run distributed training.
```

Compare the training time with your single-GPU baseline. You should see a speedup, though not perfectly linear due to communication overhead.

### Quick Start 3: Profiling and Performance Analysis

**File:** `code/chapter1/ch01_profiling.py`

When you need to understand where time and memory are spent, use profiling. The `ch01_profiling.py` script demonstrates how to use PyTorch's profiler to measure CUDA operations and memory usage. Run it to see detailed timing and memory breakdowns:

```bash
python code/chapter1/ch01_profiling.py
```

Compare the single-GPU baseline (`ch01_single_gpu_baseline.py`) with multi-GPU training (`ch01_multi_gpu_ddp.py`) to see where distributed training helps and where communication overhead appears. Typical findings show data loading takes 20-40% of time, forward pass 30-50%, backward pass 40-60%, and communication adds 10-30% overhead in multi-GPU setups.

### Additional Helper Scripts

The `code/chapter1/` directory also contains utility scripts:

- `ch01_gpu_friendly_config.py`: Shows recommended GPU-friendly configuration settings for batch size, precision, and gradient checkpointing.
- `ch01_measure_components.py`: Helper function to measure time spent in data loading, computation, and communication separately.
- `ch01_ddp_pitfalls.py`: Examples of common mistakes and their correct implementations.

### Launching Distributed Jobs

**Using torchrun (recommended):**

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 train.py

# Multiple nodes (node 0)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
    --master_addr="192.168.1.1" --master_port=29500 train.py

# Multiple nodes (node 1)
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
    --master_addr="192.168.1.1" --master_port=29500 train.py
```

### Quick Validation

**Check if distributed training is working:**

```python
import time

# Measure training speed
start = time.time()
# ... training step ...
elapsed = time.time() - start

if rank == 0:
    print(f"Time per step: {elapsed:.3f}s")
    print(f"Using {world_size} GPUs")
```

**Common issues:**
- **Hanging:** Check NCCL communication, firewall settings
- **OOM:** Reduce batch size or use gradient accumulation
- **Slow:** Check data loading, use high-speed interconnects

---

## Key Takeaways

Before moving to the next chapter, remember:

1. **Estimate resources first:** Use the formulas in this chapter to calculate memory and compute requirements before starting
2. **Make informed decisions:** Use the decision framework to determine if you need distributed systems for your use case
3. **Start simple:** Begin with single-GPU solutions, then scale when necessary
4. **Profile before optimizing:** Measure actual bottlenecks before making changes
5. **Use the right approach:** Training, fine-tuning, and inference have different requirements and optimization strategies

---

## Summary

This chapter has helped you understand when and how to use distributed AI systems. We've covered:

1. **Resource estimation:** Formulas and methods to calculate memory and compute requirements for models
2. **Decision framework:** Clear guidelines for determining when you need distributed training, fine-tuning, or inference
3. **Quick start examples:** Practical, runnable code for distributed training, fine-tuning, and inference
4. **Understanding workloads:** The fundamental differences between training, inference, and serving

The most important lesson is to **estimate first, decide second, then implement**. Don't assume you need distributed systems—calculate your requirements and make an informed decision.

In the next chapter, we'll dive deeper into GPU hardware, networking, and parallelism strategies, building on the foundation established here.

---

## Exercises

1. **Profile a simple model:** Create a small transformer model and profile its memory and compute usage. Identify the largest memory consumers and slowest operations.

2. **Compare single vs multi-GPU:** Run the same training job on 1 GPU and 2 GPUs. Measure the speedup and identify any bottlenecks.

3. **Analyze communication overhead:** Add timing measurements to a DDP training script to measure how much time is spent on gradient synchronization.

4. **Design a distributed system:** For a given model size and dataset, design a distributed training setup including number of GPUs, parallelism strategy, and expected training time.

---

## Further Reading

- PyTorch Distributed Training: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- NVIDIA NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- GPU Memory Management: https://pytorch.org/docs/stable/notes/cuda.html
- Profiling PyTorch Models: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
