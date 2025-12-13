# Modern Distributed AI Systems

This repository contains the code examples and materials for the book "Modern Distributed AI Systems".

## Getting Started

Clone the repository with submodules:

```
git clone --recurse-submodules <repo-url>
```

or

```
git clone <repo-url>
git submodule update --init --recursive
```

## Repository Structure

Each chapter is organized in its own directory:

```
chapterX-slug(title)/
├── main.md          # Chapter content
├── code/            # Code examples for this chapter
└── paper/           # Related papers (if any)
```

## Code Examples

This repository contains runnable code examples extracted from the book chapters. The goal is to provide small, focused scripts you can run locally (single-node) or on a multi-GPU host for hands-on experiments.

### Quick Overview

- **Chapter 1** (`chapter1-introduction-to-modern-distributed-ai/code/`) — single-GPU baseline, profiling, and first multi-GPU DDP example.
- **Chapter 2** (`chapter2-gpu-hardware-networking-and-parallelism-strategies/code/`) — bandwidth and AllReduce microbenchmarks.
- **Chapter 3** (`chapter3-distributed-training-with-pytorch-ddp/code/`) — torchrun launcher snippet and DDP/AMP helper examples.
- **Chapter 4** (`chapter4-scaling-with-fully-sharded-data-parallel-fsdp/code/`) — FSDP checkpointing schematic (conceptual).
- **Chapter 5** (`chapter5-deepspeed-and-zero-optimization/code/`) — DeepSpeed ZeRO Stage 3 config example (JSON).
- **Chapter 6** (`chapter6-distributed-inference-fundamentals-and-vllm/code/`) — continuous batching scheduler pseudo-code.
- **Chapter 7** (`chapter7-sglang-and-advanced-inference-architectures/code/`) — SGLang examples and advanced inference architectures.
- **Chapter 8** (`chapter8-distributed-ai-training-in-action/code/`) — Distributed training examples.
- **Chapter 9** (`chapter9-production-llm-serving-stack/code/`) — production LLM serving stack (tokenizer, model runner, API gateway, canary deployment, tracing).
- **Chapter 10** (`chapter10-distributed-benchmarking-and-performance-optimization/code/`) — distributed benchmarking tools (genai-bench, scaling efficiency, network diagnostics).
- **Chapter 11** (`chapter11-trends-and-future-of-distributed-ai/code/`) — trends and future directions in distributed AI.

## Prerequisites

- Linux host with NVIDIA GPUs (for GPU examples).  
- Python 3.9+ (3.10/3.11 recommended).  
- PyTorch with CUDA support installed for your CUDA toolkit version. Install from https://pytorch.org according to your CUDA version.  
- (Optional) DeepSpeed for the DeepSpeed config examples: `pip install deepspeed` if you plan to run DeepSpeed examples.  

To create a virtual environment and install basic deps (CPU-only fallback):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
# Install minimal tooling; install torch separately following https://pytorch.org
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu118  # change per your CUDA
```

Note: Installing PyTorch with the correct CUDA build is important for GPU examples (`torch.cuda.is_available()`). Use the wheel instructions from PyTorch website matching your CUDA version.

## Running Examples

Each example is small and self-contained. Below are example commands.

### Chapter 1 (single-node experiments)

- Check GPUs:

```bash
python chapter1-introduction-to-modern-distributed-ai/code/ch01_check_cuda.py
```

- Run single-GPU baseline:

```bash
python chapter1-introduction-to-modern-distributed-ai/code/ch01_single_gpu_baseline.py
```

- Profile a single step with PyTorch profiler:

```bash
python chapter1-introduction-to-modern-distributed-ai/code/ch01_profiling.py
```

- Run multi-GPU DDP example (single node with 2 GPUs):

```bash
torchrun --nproc_per_node=2 chapter1-introduction-to-modern-distributed-ai/code/ch01_multi_gpu_ddp.py
```

### Chapter 2 (microbenchmarks)

- Device-to-device bandwidth test (single GPU):

```bash
python chapter2-gpu-hardware-networking-and-parallelism-strategies/code/bandwidth_test.py
```

- AllReduce microbenchmark (requires initialized NCCL process group):

```bash
# Example: run with torchrun to initialize NCCL
torchrun --nproc_per_node=2 python chapter2-gpu-hardware-networking-and-parallelism-strategies/code/allreduce_microbench.py
```

### Chapter 3 (DDP/AMP helpers)

- Example DDP init helper and AMP training step are in `chapter3-distributed-training-with-pytorch-ddp/code/` — integrate into your training script.

### Chapter 4 (FSDP)

- `chapter4-scaling-with-fully-sharded-data-parallel-fsdp/code/fsdp_checkpointing.py` shows a schematic of wrapping a model with FSDP and activation checkpointing. It's a conceptual helper — adapt it into a real training loop and initialize PyTorch process groups before use.

### Chapter 5 (DeepSpeed)

- `chapter5-deepspeed-and-zero-optimization/code/deepspeed_stage3_config.json` is a sample DeepSpeed config demonstrating ZeRO Stage 3 with offload. Use it with a DeepSpeed-enabled training script:

```bash
deepspeed --num_gpus=4 --deepspeed_config chapter5-deepspeed-and-zero-optimization/code/deepspeed_stage3_config.json train.py
```

### Chapter 6 (inference scheduler)

- `chapter6-distributed-inference-fundamentals-and-vllm/code/continuous_batch_scheduler.py` contains the simple scheduler loop. It is intentionally minimal; wire it to your request queue and `run_inference` implementation.

### Chapter 9 (production serving stack)

- `chapter9-production-llm-serving-stack/code/tokenizer_service.py` — FastAPI tokenizer service
- `chapter9-production-llm-serving-stack/code/model_runner.py` — vLLM-based model runner
- `chapter9-production-llm-serving-stack/code/ch09_api_gateway.py` — API gateway with routing and rate limiting
- `chapter9-production-llm-serving-stack/code/ch09_canary_deployment.py` — Canary deployment with rollback
- `chapter9-production-llm-serving-stack/code/ch09_tracing.py` — OpenTelemetry tracing example

### Chapter 10 (benchmarking)

- `chapter10-distributed-benchmarking-and-performance-optimization/code/ch10_genai_bench.py` — Inference benchmarking with genai-bench
- `chapter10-distributed-benchmarking-and-performance-optimization/code/ch10_scaling_efficiency.py` — Scaling efficiency measurement
- `chapter10-distributed-benchmarking-and-performance-optimization/code/ch10_network_diagnostics.py` — Network bandwidth testing

### Chapter 11 (trends and future)

- `chapter11-trends-and-future-of-distributed-ai/code/` — Examples related to emerging trends and future directions in distributed AI.

## Notes about Stubs and Non-Runnable Snippets

- Some files are intentionally minimal (stubs) to show the pattern in the book. Examples: `amp_ddp_example.py`, `fsdp_checkpointing.py`, and the scheduler script. If you want, I can convert any stub into a fully runnable example (generate synthetic data, full train loop, and tests).
