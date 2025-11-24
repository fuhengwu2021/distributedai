# Code snippets for the book

This folder contains runnable code examples extracted from the book chapters. The goal is to provide small, focused scripts you can run locally (single-node) or on a multi-GPU host for hands-on experiments.

Location: `code/`

## Quick overview

- `chapter1/` — single-GPU baseline, profiling, and first multi-GPU DDP example.
- `chapter2/` — bandwidth and AllReduce microbenchmarks.
- `chapter3/` — torchrun launcher snippet and DDP/AMP helper examples.
- `chapter4/` — FSDP checkpointing schematic (conceptual).
- `chapter5/` — DeepSpeed ZeRO Stage 3 config example (JSON).
- `chapter6/` — continuous batching scheduler pseudo-code.
- `chapter9/` — production LLM serving stack (tokenizer, model runner, API gateway, canary deployment, tracing).
- `chapter10/` — distributed benchmarking tools (genai-bench, scaling efficiency, network diagnostics).
- `chapter11/` — federated learning (Flower, FedAvg) and edge deployment.

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

## Running examples (selectively)

Each example is small and self-contained. Below are example commands.

Chapter 1 (single-node experiments)

- Check GPUs:

```bash
python code/chapter1/check_cuda.py
```

- Run single-GPU baseline:

```bash
python code/chapter1/ch01_single_gpu_baseline.py
```

- Profile a single step with PyTorch profiler:

```bash
python code/chapter1/ch01_profiling.py
```

- Run multi-GPU DDP example (single node with 2 GPUs):

```bash
torchrun --nproc_per_node=2 code/chapter1/ch01_multi_gpu_ddp.py
# or use the provided launcher script
bash code/chapter1/launch_torchrun_ch01.sh
```

Chapter 2 (microbenchmarks)

- Device-to-device bandwidth test (single GPU):

```bash
python code/chapter2/bandwidth_test.py
```

- AllReduce microbenchmark (requires initialized NCCL process group):

```bash
# Example: run with torchrun to initialize NCCL
torchrun --nproc_per_node=2 python code/chapter2/allreduce_microbench.py
```

Chapter 3 (DDP/AMP helpers)

- The `train_torchrun.sh` is a launcher snippet; adapt `train.py` to your project and run it with `torchrun`.
- Example DDP init helper and AMP training step are in `code/chapter3/train_init.py` and `code/chapter3/amp_ddp_example.py` — integrate into your training script.

Chapter 4 (FSDP)

- `code/chapter4/fsdp_checkpointing.py` shows a schematic of wrapping a model with FSDP and activation checkpointing. It's a conceptual helper — adapt it into a real training loop and initialize PyTorch process groups before use.

Chapter 5 (DeepSpeed)

- `code/chapter5/deepspeed_stage3_config.json` is a sample DeepSpeed config demonstrating ZeRO Stage 3 with offload. Use it with a DeepSpeed-enabled training script:

```bash
deepspeed --num_gpus=4 --deepspeed_config code/chapter5/deepspeed_stage3_config.json train.py
```

Chapter 6 (inference scheduler)

- `code/chapter6/continuous_batch_scheduler.py` contains the simple scheduler loop. It is intentionally minimal; wire it to your request queue and `run_inference` implementation.

Chapter 9 (production serving stack)

- `code/chapter9/tokenizer_service.py` — FastAPI tokenizer service
- `code/chapter9/model_runner.py` — vLLM-based model runner
- `code/chapter9/ch09_api_gateway.py` — API gateway with routing and rate limiting
- `code/chapter9/ch09_canary_deployment.py` — Canary deployment with rollback
- `code/chapter9/ch09_tracing.py` — OpenTelemetry tracing example

Chapter 10 (benchmarking)

- `code/chapter10/ch10_genai_bench.py` — Inference benchmarking with genai-bench
- `code/chapter10/ch10_scaling_efficiency.py` — Scaling efficiency measurement
- `code/chapter10/ch10_network_diagnostics.py` — Network bandwidth testing

Chapter 11 (federated learning and edge)

- `code/chapter11/ch11_flower_federated.py` — Federated learning with Flower
- `code/chapter11/ch11_fedavg.py` — FedAvg implementation
- `code/chapter11/ch11_edge_deployment.py` — Edge deployment pipeline

## Notes about stubs and non-runnable snippets

- Some files are intentionally minimal (stubs) to show the pattern in the book. Examples: `amp_ddp_example.py`, `fsdp_checkpointing.py`, and the scheduler script. If you want, I can convert any stub into a fully runnable example (generate synthetic data, full train loop, and tests).

## Suggested next steps (pick one)

1. Convert one stub into a fully runnable demo (I can implement `train.py` and a `requirements.txt`). Tell me which chapter example to expand.  
2. Add CI/test harness (GitHub Actions) to run the CPU-only parts automatically.  
3. Reorganize files by grouping training/inference/benchmarks directories.

If you want me to proceed with any of these, reply with the option number and the chapter to prioritize.
