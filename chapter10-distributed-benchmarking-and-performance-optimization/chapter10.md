# Chapter 10: Distributed Benchmarking and Performance Optimization {-}

*Measuring and optimizing performance of distributed AI systems*

> If you can't measure it, you can't improve it.
- Peter Drucker, Management Consultant and Author

**Code Summary**

- `torch.profiler`: PyTorch profiler for performance analysis
- `torch.utils.benchmark`: PyTorch benchmarking utilities
- `nvidia-ml-py`: Python library for GPU monitoring
- `psutil`: System and process utilities for resource monitoring
- `mlperf_logging`: MLPerf logging utilities for standardized benchmarks
- `tensorboard`: TensorBoard for visualization of training metrics
- `wandb`: Weights & Biases for experiment tracking
- `py-spy`: Sampling profiler for Python applications
- `nsys`: NVIDIA Nsight Systems for system-level profiling
- `ncu`: NVIDIA Nsight Compute for kernel-level profiling


## Overview

You've built the distributed AI system. DDP synchronizes gradients across your 8-GPU cluster. FSDP shards your 70B parameter model across nodes. vLLM serves inference requests with continuous batching. The production stack from Chapter 9 routes traffic to the right model instances. Everything *works*—but is it working *well*?

This is the question that separates functional systems from optimized ones. A training run that completes is not the same as a training run that efficiently utilizes your \$100,000 worth of GPUs. An inference endpoint that returns responses is not the same as one that meets your 200ms P99 latency SLA. The difference between "working" and "working well" can mean days of wasted training time, violated service agreements, and unnecessary cloud costs.

Consider this scenario: your team trains a model on 64 GPUs across 8 nodes. The training completes, the model performs well on benchmarks, and everyone celebrates. But hidden in the logs is a troubling pattern—GPU utilization hovers around 45%, and scaling efficiency from 8 to 64 GPUs is only 52%. You're paying for 64 GPUs but getting the effective compute of 33. Over a two-week training run, that's \$50,000 in wasted compute.

This chapter teaches you to find and fix these hidden inefficiencies. We'll cover the complete benchmarking lifecycle: the metrics that actually matter for distributed systems, profiling tools that reveal where time goes, accuracy evaluation to ensure optimizations don't degrade model quality, network diagnostics for communication bottlenecks, and scaling analysis to understand your system's limits. By the end, you'll have the skills to systematically identify performance bottlenecks, make data-driven optimization decisions, and validate that your distributed systems are running at peak efficiency.


## Why Benchmarking Matters

Before diving into tools and techniques, let's understand why benchmarking distributed AI systems is fundamentally different from benchmarking single-device workloads—and why getting it wrong can be costly.

A single-GPU training loop is relatively simple to measure: time the forward pass, backward pass, and optimizer step. But distributed systems introduce complexity at every layer. Your 8-GPU DDP training isn't just 8 copies of single-GPU training—it's 8 GPUs that must synchronize gradients, coordinate memory, and communicate over networks with varying bandwidths. A 10% inefficiency in gradient synchronization might be invisible in a single training step, but compounds to hours of wasted GPU time over a multi-day training run.

**The consequences of poor benchmarking are real and expensive:**

- **Wrong technology choices:** A team selects vLLM over SGLang based on benchmarks that used fixed-length prompts. In production, with variable-length requests, SGLang's radix attention provides 40% better throughput. The benchmark didn't reflect reality.

- **Over-provisioning:** Without accurate scaling measurements, capacity planning becomes guesswork. Teams often provision 2x the hardware they need "just to be safe," doubling cloud costs.

- **Missed optimizations:** A training job runs for two weeks. Post-hoc analysis reveals that data loading was the bottleneck—not compute. A \$50 SSD upgrade would have saved \$10,000 in GPU time.

- **SLA violations:** An inference service passes load testing with synthetic prompts. In production, real user queries trigger different code paths, and P99 latency spikes to 500ms—2.5x the promised SLA.

The techniques in this chapter help you avoid these pitfalls by establishing systematic, reproducible measurement practices. We'll start with the metrics that matter, then build up to comprehensive profiling workflows.


## Core Metrics for Distributed Systems

Not all metrics are created equal. Understanding which numbers to track—and which to ignore—is the foundation of effective benchmarking. Let's examine the metrics that actually matter for distributed AI systems.

### Throughput Metrics

Throughput measures how much work your system completes per unit time. For distributed systems, the key insight is that raw throughput must be normalized to be meaningful:

- **Samples per Second (Training):** Number of training samples processed per second *across all GPUs*. This is your primary training efficiency metric.

- **Tokens per Second (Inference):** Number of tokens generated per second. For LLM serving, this determines how many concurrent users you can support.

- **Requests per Second (Serving):** Number of API requests handled per second at the system level. Unlike tokens/second, this captures the full request lifecycle including queuing and routing.

- **GPU Utilization:** Percentage of time GPUs are actively computing vs waiting. Low utilization often indicates bottlenecks elsewhere (data loading, communication, CPU preprocessing).

### Latency Metrics

Latency measures how long individual operations take. For production systems, percentile latencies matter more than averages:

- **P50 (Median):** The latency experienced by a typical request. Useful for understanding normal behavior.

- **P95:** The latency experienced by 95% of requests. This is often the SLA target for production systems.

- **P99:** The worst-case latency for 99% of requests. Critical for user experience—even 1% of users experiencing 10x latency creates significant frustration.

- **Time to First Token (TTFT):** For streaming inference, how long until the first token appears. Users perceive this as "response time."

- **Time per Output Token (TPOT):** Average time between consecutive output tokens. Determines the perceived "typing speed" of streaming responses.

![Latency Distribution: Left shows histogram with P50/P95/P99 percentile lines and color-coded regions. Right shows CDF with percentile markers demonstrating tail latency.](img/latency_distribution.png)

>NOTES: **Why P99 Matters More Than Average**

Average latency can be misleading. Consider two systems: System A has 100ms average latency with all requests between 90-110ms. System B has 80ms average latency, but 5% of requests take 500ms. System B has better average latency but worse user experience—those 5% of slow requests create frustrated users and potential timeouts. Always report P95 and P99 alongside averages.

>NOTEE

### Efficiency Metrics

Efficiency metrics help you understand how well you're utilizing resources:

- **Scaling Efficiency:** How well performance scales with additional devices. If 8 GPUs provide 6.5x the throughput of 1 GPU, scaling efficiency is 81.25%.

- **Memory Efficiency:** Memory utilization vs available memory. High utilization is good, but watch for fragmentation that prevents larger batch sizes.

- **Communication Overhead:** Time spent on synchronization vs computation. In distributed training, this often becomes the bottleneck at scale.

- **Cost per Token/Request:** The economic efficiency metric. Combines performance with actual cloud/hardware costs.

### The Benchmarking Methodology

Rigorous methodology separates meaningful benchmarks from noise. Here's the four-phase approach that ensures reproducible, accurate results:

**Phase 1: Setup**
- Fix random seeds for reproducibility
- Document hardware configuration (GPU model, memory, interconnect)
- Version control benchmark scripts
- Record system state (driver versions, CUDA version, framework versions)

**Phase 2: Warmup**
```python
# Always include warmup iterations
def benchmark_with_warmup(model, dataloader, num_warmup=10, num_iterations=100):
    # Warmup: discard initial iterations
    for i in range(num_warmup):
        _ = model(next(iter(dataloader)))
    
    # Synchronize before measurement
    torch.cuda.synchronize()
    
    # Actual measurement
    timings = []
    for i in range(num_iterations):
        start = time.time()
        _ = model(next(iter(dataloader)))
        torch.cuda.synchronize()
        timings.append(time.time() - start)
    
    return timings
```

**Phase 3: Measurement**
- Run multiple independent iterations (minimum 100)
- Ensure CUDA synchronization before timing
- Measure over sufficient duration to capture variance

**Phase 4: Analysis**
- Report mean, standard deviation, and percentiles
- Use statistical significance tests when comparing systems
- Document any anomalies or outliers

![Benchmarking Methodology Flow: Shows the four-phase workflow (Setup, Warmup, Measurement, Analysis) with common pitfalls and best practices for rigorous benchmarking.](img/benchmarking_methodology.png)

### Common Benchmarking Pitfalls

Even experienced engineers fall into these traps. Here's how to avoid them:

**Pitfall 1: Insufficient Warmup**

The first iteration of any CUDA operation is slow—kernels must be JIT compiled, memory must be allocated, and caches must be populated. Measuring cold-start performance as typical performance is a common mistake.

```python
# Wrong: No warmup, first iteration is slow
start = time.time()
result = model(inputs)  # ❌ Cold start
time_taken = time.time() - start

# Correct: Warmup before measurement
for _ in range(10):
    _ = model(inputs)  # Warmup
torch.cuda.synchronize()
start = time.time()
result = model(inputs)  # ✅ Warm measurement
time_taken = time.time() - start
```

**Pitfall 2: Ignoring Variance**

A single measurement tells you almost nothing. Performance varies due to thermal throttling, background processes, network congestion, and dozens of other factors.

```python
# Wrong: Single measurement
time_taken = measure_once()

# Correct: Multiple measurements with statistics
times = [measure() for _ in range(10)]
mean_time = np.mean(times)
std_time = np.std(times)
print(f"Time: {mean_time:.3f} ± {std_time:.3f} seconds")
```

**Pitfall 3: Measuring the Wrong Thing**

It's easy to accidentally include setup time, data loading, or other operations in your measurements.

```python
# Wrong: Including data loading in inference time
for data in dataloader:  # Data loading included
    start = time.time()
    result = model(data)
    time_taken = time.time() - start  # ❌

# Correct: Pre-load data, measure only inference
data_batch = next(iter(dataloader))
torch.cuda.synchronize()
start = time.time()
result = model(data_batch)
torch.cuda.synchronize()
time_taken = time.time() - start  # ✅
```

With these foundational concepts in place, let's dive into the specific tools and techniques for benchmarking training and inference workloads.


## Training Benchmarking

Training a distributed model involves a complex pipeline: data loading, forward pass, backward pass, gradient synchronization, and optimizer updates. Each component contributes to overall training time, and bottlenecks can hide in any of them. Effective training benchmarking requires measuring each phase separately to identify where optimization efforts should focus.

![Training Iteration Breakdown: Left shows stacked bar chart of time per iteration phase across GPU configurations. Right shows percentage distribution highlighting communication overhead growth with scale.](img/training_breakdown.png)

The figure above illustrates a common pattern: as you scale from 1 GPU to 16 GPUs, communication overhead grows from 0% to over 50% of iteration time. This is why scaling efficiency decreases—you're spending more time synchronizing and less time computing. Understanding this breakdown is the first step toward optimization.

### PyTorch Profiler

PyTorch's built-in profiler is your first tool for understanding training performance. It captures CPU and CUDA operations, memory allocations, and can export traces for visualization.

**Basic Usage:**
```python
from torch.profiler import profile, record_function, ProfilerActivity
import torch

def profile_training_step(model, inputs, targets):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("forward"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        with record_function("backward"):
            loss.backward()
        
        with record_function("optimizer"):
            optimizer.step()
    
    # Print results
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    # Export for visualization
    prof.export_chrome_trace("trace.json")
```

The `record_function` context manager lets you label regions of code, making it easy to identify which phase consumes the most time. The exported Chrome trace can be viewed in `chrome://tracing` for detailed timeline analysis.

**Advanced Profiling with Schedule:**

For multi-iteration analysis, use a profiling schedule that handles warmup automatically:

```python
# Profile with schedule for multi-iteration analysis
with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=1,      # Skip first iteration
        warmup=1,    # Warmup for 1 iteration
        active=3,    # Profile 3 iterations
        repeat=2     # Repeat schedule 2 times
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, data in enumerate(dataloader):
        prof.step()
        # Training code
```

This schedule skips the first iteration, warms up for one iteration, then profiles three iterations—repeating this pattern twice. The results integrate directly with TensorBoard for visualization.

### NVIDIA Nsight Systems

For deeper analysis—especially of CUDA kernels and NCCL communication—NVIDIA Nsight Systems provides system-level profiling that PyTorch's profiler can't match.

**Command Line Usage:**
```bash
# Profile training script
nsys profile --trace=cuda,nvtx,osrt \
    --output=training_profile.nsys-rep \
    python train.py

# Generate report
nsys stats --report gputrace training_profile.nsys-rep
```

Nsight captures GPU kernel execution time, memory transfer time (H2D, D2H), CUDA API calls, synchronization points, and NCCL communication operations. This level of detail is essential for diagnosing subtle performance issues like kernel launch overhead or suboptimal memory access patterns.

>NOTES: **When to Use Nsight vs PyTorch Profiler**

Start with PyTorch Profiler for high-level analysis—it's easier to use and integrates with TensorBoard. Move to Nsight Systems when you need to understand CUDA kernel behavior, NCCL communication patterns, or system-level interactions. Nsight is also essential when profiling custom CUDA kernels or debugging performance issues that don't appear in PyTorch's view.

>NOTEE

### Custom Training Benchmark

For systematic benchmarking across configurations, a custom benchmark class provides more control than ad-hoc profiling:

```python
import time
import torch
import torch.distributed as dist
from collections import defaultdict

class TrainingBenchmark:
    def __init__(self, model, dataloader, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = defaultdict(list)
    
    def benchmark_iteration(self, warmup=True):
        """Benchmark a single training iteration with phase breakdown"""
        if warmup:
            data, target = next(iter(self.dataloader))
            _ = self._training_step(data, target)
            torch.cuda.synchronize()
        
        data, target = next(iter(self.dataloader))
        
        # Data loading time
        data_start = time.time()
        data, target = data.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - data_start
        
        # Forward pass
        forward_start = time.time()
        output = self.model(data)
        loss = self.criterion(output, target)
        torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        optimizer_start = time.time()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        optimizer_time = time.time() - optimizer_start
        
        return {
            'data_loading': data_time,
            'forward': forward_time,
            'backward': backward_time,
            'optimizer': optimizer_time,
            'total': forward_time + backward_time + optimizer_time
        }
    
    def benchmark(self, num_warmup=10, num_iterations=100):
        """Run full benchmark with statistics"""
        for _ in range(num_warmup):
            self.benchmark_iteration(warmup=False)
        
        for _ in range(num_iterations):
            metrics = self.benchmark_iteration(warmup=False)
            for key, value in metrics.items():
                self.metrics[key].append(value)
        
        stats = {}
        for key, values in self.metrics.items():
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        return stats
```

This class measures each phase separately, enabling you to identify exactly where time is spent. If `data_loading` dominates, you need more DataLoader workers. If `backward` is slow relative to `forward`, you may have inefficient gradient computation.

### Measuring Scaling Efficiency

Scaling efficiency quantifies how well your system utilizes additional resources. Perfect linear scaling (100% efficiency) means doubling GPUs doubles throughput—but communication overhead makes this impossible in practice.

```python
def calculate_scaling_efficiency(throughput_1gpu, throughput_ngpu, n):
    """
    Calculate scaling efficiency.
    
    Args:
        throughput_1gpu: Throughput with 1 GPU
        throughput_ngpu: Throughput with N GPUs
        n: Number of GPUs
    
    Returns:
        Efficiency percentage (100% = perfect linear scaling)
    """
    ideal_throughput = throughput_1gpu * n
    actual_throughput = throughput_ngpu
    efficiency = (actual_throughput / ideal_throughput) * 100
    return efficiency

# Example
throughput_1 = 100  # samples/sec with 1 GPU
throughput_8 = 650  # samples/sec with 8 GPUs
efficiency = calculate_scaling_efficiency(throughput_1, throughput_8, 8)
print(f"Scaling efficiency: {efficiency:.1f}%")  # 81.25%
```

![Scaling Efficiency: Left shows throughput scaling (ideal vs actual) with communication overhead gap. Right shows efficiency percentages by GPU count with color-coded thresholds.](img/scaling_efficiency.png)

**Interpreting Scaling Efficiency:**

- **>90%:** Excellent scaling—your system is well-optimized
- **70-90%:** Good scaling—typical for well-tuned distributed training
- **50-70%:** Moderate scaling—communication overhead is significant, investigate network
- **<50%:** Poor scaling—major bottleneck present, likely communication or data loading

Understanding your scaling efficiency helps with capacity planning. If you have 81% efficiency at 8 GPUs, you can predict that 16 GPUs will provide roughly 13x throughput (not 16x), helping you make informed hardware decisions.


## Inference Benchmarking

Inference benchmarking presents different challenges than training. Request patterns are variable, caching effects matter, and tail latency requirements are strict. A training job that's 10% slower is annoying; an inference endpoint that violates P99 SLA loses customers.

>NOTES: **Performance vs Accuracy Benchmarking**

This section covers **performance benchmarking** using tools like [genai-bench](https://github.com/sgl-project/genai-bench), which measures engineering metrics (throughput, latency, scaling). This is distinct from **accuracy benchmarking** tools like [GenAI-Bench for text-to-visual evaluation](https://linzhiqiu.github.io/papers/genai_bench/) that measure model output quality. We cover accuracy benchmarking in the next section.

>NOTEE

### genai-bench Overview

genai-bench is a CLI-based benchmarking tool designed for realistic inference workload testing. Unlike simple load generators, it supports configurable traffic patterns that mirror production workloads.

**Key Features:**

- Realistic prompt distributions via traffic scenarios
- Configurable load patterns (concurrency, request rates)
- Support for multiple inference engines (vLLM, SGLang, OpenAI API, cloud providers)
- Comprehensive latency percentile tracking (TTFT, E2E, TPOT)
- Automatic Excel reports and plot generation

**Installation:**
```bash
pip install genai-bench
```

### Running genai-bench

**Basic Command Line Usage:**
```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8000" \
    --api-key "your-api-key" \
    --api-model-name "llama-2-7b-chat" \
    --model-tokenizer "/path/to/tokenizer" \
    --task text-to-text \
    --max-time-per-run 15 \
    --max-requests-per-run 1000 \
    --num-concurrency 100 \
    --traffic-scenario "D(100,100)" \
    --server-engine "vLLM" \
    --server-gpu-type "H100" \
    --server-gpu-count 4
```

**Understanding Traffic Scenarios:**

Traffic scenarios define the distribution of input and output token lengths:

- `D(100,100)`: Deterministic—all requests have exactly 100 input and 100 output tokens
- `D(512,512)`: Longer context scenario
- `I(input_tokens, output_tokens)`: Image-text input with fixed tokens
- `E(input_tokens)`: Embedding requests

For realistic benchmarking, test multiple scenarios that reflect your production traffic:

```bash
genai-bench benchmark \
    --api-backend openai \
    --api-base "http://localhost:8000" \
    --api-key "your-api-key" \
    --api-model-name "llama-2-7b-chat" \
    --model-tokenizer "/path/to/tokenizer" \
    --task text-to-text \
    --max-time-per-run 15 \
    --max-requests-per-run 300 \
    --num-concurrency 1 \
    --num-concurrency 8 \
    --num-concurrency 16 \
    --traffic-scenario "D(100,100)" \
    --traffic-scenario "D(512,512)" \
    --traffic-scenario "D(2048,2048)" \
    --server-engine "vLLM" \
    --server-gpu-type "H100"
```

This runs a matrix of tests: 3 concurrency levels × 3 traffic scenarios = 9 benchmark configurations. The results reveal how your system behaves under different load patterns.

**Analyzing Results:**

After benchmarking, generate reports:

```bash
# Generate Excel report
genai-bench excel \
    --experiment-folder ./experiments/your_experiment \
    --excel-name benchmark_results \
    --metric-percentile mean \
    --metrics-time-unit s

# Generate plots
genai-bench plot \
    --experiments-folder ./experiments \
    --group-key traffic_scenario \
    --preset 2x4_default
```

### Custom Inference Benchmark

For scenarios where genai-bench doesn't fit, here's a custom benchmark class:

```python
import asyncio
import time
import numpy as np
from collections import defaultdict

class InferenceBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = defaultdict(list)
    
    def benchmark_single_request(self, prompt, max_tokens=512):
        """Benchmark a single inference request"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Warmup on first request
        if not hasattr(self, '_warmed_up'):
            _ = self.model.generate(**inputs, max_new_tokens=1)
            torch.cuda.synchronize()
            self._warmed_up = True
        
        # Measure generation
        torch.cuda.synchronize()
        start = time.time()
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        total_time = time.time() - start
        
        num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        
        return {
            'total_time': total_time,
            'tokens_per_second': num_tokens / total_time,
            'num_tokens': num_tokens
        }
    
    def get_statistics(self):
        """Get statistical summary"""
        stats = {}
        for key, values in self.metrics.items():
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        return stats
```

### Measuring Cold vs Warm Performance

Cold start latency—the time for the first request after model loading—can be 10-100x slower than warm requests. Understanding this gap is critical for autoscaling decisions:

```python
def benchmark_cold_start(model, prompts):
    """Measure cold start performance"""
    torch.cuda.empty_cache()  # Clear cache to simulate cold start
    
    # First request (cold)
    cold_start = time.time()
    _ = model.generate(prompts[0], max_new_tokens=100)
    torch.cuda.synchronize()
    cold_time = time.time() - cold_start
    
    # Subsequent requests (warm)
    warm_times = []
    for prompt in prompts[1:]:
        start = time.time()
        _ = model.generate(prompt, max_new_tokens=100)
        torch.cuda.synchronize()
        warm_times.append(time.time() - start)
    
    return {
        'cold_time': cold_time,
        'warm_mean': np.mean(warm_times),
        'cold_overhead': cold_time - np.mean(warm_times)
    }
```

### Measuring Reasoning & Multi-step Models

Reasoning models—chain-of-thought, tool-augmented LLMs, multi-step agents—require different benchmarking approaches. You need to measure both per-step latency and end-to-end session latency, separating local generation time from external calls.

**Key measurement points:**

- **Per-step latency:** Measure TTFT/TPOT for each reasoning step to identify slow steps
- **End-to-end session latency:** Total time for the complete reasoning session
- **External-call breakdown:** Time waiting for retrievals, tool calls, or APIs vs local generation
- **Cache effects:** Cold vs warm runs when KV cache or retrieval caches are populated

```python
import time
import numpy as np

def run_reasoning_step(model, step_input, max_new_tokens=64, do_tool_call=None):
    """Measure a single reasoning step with optional tool call"""
    gen_start = time.time()
    out = model.generate(step_input, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    gen_time = time.time() - gen_start

    tool_time = 0.0
    if do_tool_call:
        t0 = time.time()
        tool_result = do_tool_call()
        tool_time = time.time() - t0

    return gen_time, tool_time, out

def measure_reasoning_session(model, session_steps, do_tool_call_fn=None):
    """Measure complete reasoning session with per-step breakdown"""
    per_step = []
    total = 0.0
    
    for step_input in session_steps:
        gen_t, tool_t, out = run_reasoning_step(
            model, step_input, 
            do_tool_call=(do_tool_call_fn if do_tool_call_fn else None)
        )
        per_step.append({
            'gen_time': gen_t, 
            'tool_time': tool_t, 
            'step_total': gen_t + tool_t
        })
        total += gen_t + tool_t

    times = [s['step_total'] for s in per_step]
    return {
        'per_step': per_step,
        'total': total,
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
```

With training and inference benchmarking covered, let's turn to a different but equally important dimension: ensuring that performance optimizations don't degrade model quality.


## Accuracy and Quality Benchmarking

Performance benchmarking measures speed; accuracy benchmarking measures correctness. Both are essential. A model that generates garbage at 1000 tokens/second is worse than one that generates quality output at 100 tokens/second. More subtly, optimizations like quantization, distributed training, or different serving engines can introduce accuracy regressions that aren't obvious without systematic evaluation.

### Why Accuracy Benchmarking Matters

Consider these scenarios where accuracy benchmarking prevented disasters:

- **Quantization regression:** INT8 quantization improved inference throughput by 2x, but accuracy on math problems dropped 15%. Without accuracy benchmarking, this would have reached production.

- **Distributed training divergence:** A bug in gradient synchronization caused distributed training to converge to a different (worse) solution than single-GPU training. The loss curves looked similar, but downstream task accuracy was 8% lower.

- **Serving engine differences:** Two inference engines produced different outputs for the same prompt due to different sampling implementations. One was correct; one had a bug.

### Accuracy Metrics for LLMs

**Text Generation Quality Metrics:**

- **BLEU Score:** Measures n-gram overlap with reference text (common for translation)
- **ROUGE Score:** Measures overlap of n-grams, longest common subsequence (summarization)
- **BERTScore:** Semantic similarity using BERT embeddings
- **Human Evaluation:** Gold standard but expensive (Likert scales, pairwise comparisons)

**Task-Specific Metrics:**

- **Classification Accuracy:** For classification tasks
- **F1 Score:** For tasks with precision/recall trade-offs
- **Exact Match (EM):** For question answering
- **Pass@k:** For code generation (run code and check if it passes tests)

### Standard LLM Benchmarks

**GLUE/SuperGLUE:** General language understanding tasks
```python
from evaluate import load

glue = load("glue", "sst2")  # Sentiment classification
results = glue.compute(predictions=predictions, references=references)
print(f"Accuracy: {results['accuracy']:.3f}")
```

**MMLU:** Knowledge across 57 tasks (STEM, humanities, social sciences)

**HumanEval:** Code generation with execution-based evaluation

**HELM:** Holistic evaluation including accuracy, robustness, fairness, and efficiency

### Evaluating Distributed Training Accuracy

A critical check: does your distributed training produce the same model quality as single-GPU training?

```python
def compare_centralized_vs_distributed_accuracy(
    model_centralized,
    model_distributed,
    test_dataset
):
    """Compare accuracy between centralized and distributed training"""
    acc_centralized = evaluate_model(model_centralized, test_dataset)
    acc_distributed = evaluate_model(model_distributed, test_dataset)
    
    accuracy_drop = acc_centralized - acc_distributed
    
    print(f"Centralized accuracy: {acc_centralized:.4f}")
    print(f"Distributed accuracy: {acc_distributed:.4f}")
    print(f"Accuracy drop: {accuracy_drop:.4f}")
    
    if accuracy_drop > 0.01:  # More than 1% drop
        print("⚠️ Warning: Significant accuracy drop detected!")
    
    return {
        'centralized': acc_centralized,
        'distributed': acc_distributed,
        'drop': accuracy_drop
    }
```

### Evaluating Quantization Impact

Quantization trades precision for speed. Measure the accuracy cost:

```python
def evaluate_quantization_impact(model_fp32, model_int8, test_dataset):
    """Compare accuracy between FP32 and INT8 quantized models"""
    acc_fp32 = evaluate_model(model_fp32, test_dataset)
    acc_int8 = evaluate_model(model_int8, test_dataset)
    
    accuracy_drop = acc_fp32 - acc_int8
    
    return {
        'fp32_accuracy': acc_fp32,
        'int8_accuracy': acc_int8,
        'drop': accuracy_drop,
        'relative_drop': accuracy_drop / acc_fp32 * 100
    }
```

>NOTES: **Statistical Significance in Accuracy Comparisons**

Small accuracy differences may be noise, not signal. Use statistical tests to determine if differences are meaningful:

```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
if p_value < 0.05:
    print("Statistically significant difference")
```

A 0.5% accuracy drop with p=0.3 is probably noise. A 0.5% drop with p=0.001 is real.

>NOTEE


## Network and Communication Profiling

In distributed systems, the network is often the bottleneck. A single slow link between nodes can limit the entire system's performance. Understanding communication patterns and diagnosing network issues is essential for scaling efficiently.

![Network Topology: Left shows Ring AllReduce communication pattern. Right illustrates multi-node topology with NVLink (fast, intra-node) vs InfiniBand (slower, inter-node) bandwidth hierarchy.](img/network_topology.png)

The bandwidth hierarchy matters enormously: NVLink between GPUs on the same node provides ~600 GB/s, InfiniBand between nodes provides ~200 GB/s, and Ethernet provides only ~12.5 GB/s (100 Gbps). Communication patterns that cross these boundaries pay significant latency penalties.

### Network Monitoring Tools

**iftop:** Real-time network traffic monitoring
```bash
sudo iftop -i eth0
sudo iftop -i eth0 -f "host 192.168.1.10"  # Filter by host
```

**nload:** Bandwidth monitoring
```bash
nload eth0
```

**iperf3:** Bandwidth testing between nodes
```bash
# Server side
iperf3 -s

# Client side
iperf3 -c server_ip -t 60 -i 1
```

### NCCL Communication Tests

Test AllReduce bandwidth—the operation that dominates distributed training communication:

```python
import torch
import torch.distributed as dist
import time

def test_allreduce_bandwidth(rank, world_size):
    """Test AllReduce bandwidth across different message sizes"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    sizes = [1, 10, 100, 1000, 10000]  # MB
    
    for size_mb in sizes:
        size = size_mb * 1024 * 1024 // 4  # float32 elements
        tensor = torch.randn(size, device=f'cuda:{rank}')
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        num_iterations = 10
        for _ in range(num_iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate bandwidth (2x for ring allreduce)
        data_transferred = size_mb * 2 * num_iterations
        bandwidth = data_transferred / elapsed
        
        if rank == 0:
            print(f"Size: {size_mb}MB, Bandwidth: {bandwidth:.2f} MB/s")
    
    dist.destroy_process_group()
```

### Communication Overhead Analysis

Measure how much time is spent communicating vs computing:

```python
def analyze_communication_overhead(model, dataloader, num_iterations=100):
    """Analyze communication vs computation time"""
    comm_times = []
    compute_times = []
    
    for i, (data, target) in enumerate(dataloader):
        if i >= num_iterations:
            break
        
        # Computation (forward + backward)
        compute_start = time.time()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.cuda.synchronize()
        compute_time = time.time() - compute_start
        
        compute_times.append(compute_time)
        optimizer.step()
        optimizer.zero_grad()
    
    total_compute = sum(compute_times)
    # In DDP, communication happens during backward
    # Use profiler to separate these
    
    print(f"Compute time: {total_compute:.2f}s")
```

With network profiling complete, let's examine how to analyze and improve scaling efficiency.


## Scaling Efficiency Analysis

Scaling efficiency measures how well your system utilizes additional resources. Understanding scaling behavior helps with capacity planning, cost optimization, and identifying bottlenecks.

### Amdahl's Law

Amdahl's Law provides the theoretical limit on speedup from parallelization:

```python
def amdahl_speedup(serial_fraction, n):
    """
    Calculate maximum speedup using Amdahl's Law.
    
    Speedup = 1 / (S + P/N)
    where S = serial fraction, P = parallel fraction, N = processors
    """
    parallel_fraction = 1 - serial_fraction
    speedup = 1 / (serial_fraction + parallel_fraction / n)
    return speedup

# Example: If 10% of work is serial
serial_fraction = 0.10
for n in [2, 4, 8, 16, 32]:
    speedup = amdahl_speedup(serial_fraction, n)
    efficiency = speedup / n * 100
    print(f"{n} GPUs: {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")
```

With 10% serial work, even infinite GPUs can only provide 10x speedup. This is why identifying and reducing serial bottlenecks is critical.

### Identifying Scaling Bottlenecks

```python
def analyze_scaling_bottlenecks(metrics_1gpu, metrics_ngpu, n):
    """Analyze what's limiting scaling"""
    bottlenecks = []
    
    # Check data loading
    if metrics_ngpu['data_loading'] > metrics_1gpu['data_loading'] * 1.5:
        bottlenecks.append("Data loading not scaling well")
    
    # Check communication
    comm_overhead = metrics_ngpu['communication'] / metrics_ngpu['total']
    if comm_overhead > 0.3:
        bottlenecks.append(f"High communication overhead: {comm_overhead*100:.1f}%")
    
    # Check compute utilization
    gpu_util = metrics_ngpu['gpu_utilization']
    if gpu_util < 0.8:
        bottlenecks.append(f"Low GPU utilization: {gpu_util*100:.1f}%")
    
    return bottlenecks
```

### Optimization Strategies

**1. Overlap Communication and Computation:**
```python
# Use gradient bucketing in DDP
model = DDP(
    model,
    device_ids=[rank],
    bucket_cap_mb=25,  # Tune bucket size
    find_unused_parameters=False
)
```

**2. Optimize Data Loading:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,        # Parallel data loading
    pin_memory=True,      # Faster H2D transfer
    prefetch_factor=2     # Prefetch batches
)
```

**3. Gradient Accumulation:**
```python
accumulation_steps = 4
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```


\fancydividerwithicon[center]{python.png}

## Hands-On Examples

The following examples provide complete, runnable code for common benchmarking scenarios. Each example is self-contained and can be adapted to your specific use case.

### Example 1: genai-bench Inference Benchmarking

**File:** `examples/ch10_genai_bench.py`

```python
"""
Comprehensive inference benchmarking using genai-bench CLI.
Demonstrates how to programmatically run benchmarks and analyze results.
"""
import subprocess
import json
import os
from pathlib import Path

def run_genai_benchmark(
    api_base: str,
    api_key: str,
    model_name: str,
    tokenizer_path: str,
    max_requests: int = 1000,
    max_time_minutes: int = 15,
    concurrency: int = 100,
    traffic_scenario: str = "D(100,100)",
    server_engine: str = "vLLM",
    server_gpu_type: str = "H100"
):
    """Run genai-bench benchmark via CLI"""
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "openai",
        "--api-base", api_base,
        "--api-key", api_key,
        "--api-model-name", model_name,
        "--model-tokenizer", tokenizer_path,
        "--task", "text-to-text",
        "--max-time-per-run", str(max_time_minutes),
        "--max-requests-per-run", str(max_requests),
        "--num-concurrency", str(concurrency),
        "--traffic-scenario", traffic_scenario,
        "--server-engine", server_engine,
        "--server-gpu-type", server_gpu_type
    ]
    
    print("Running genai-bench benchmark...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    print("Benchmark completed successfully!")
    return result

def analyze_experiment_results(experiment_folder: str):
    """Generate Excel report and plots from experiment results"""
    subprocess.run([
        "genai-bench", "excel",
        "--experiment-folder", experiment_folder,
        "--excel-name", "benchmark_results",
        "--metric-percentile", "mean"
    ])
    
    subprocess.run([
        "genai-bench", "plot",
        "--experiments-folder", experiment_folder,
        "--group-key", "traffic_scenario",
        "--preset", "2x4_default"
    ])

if __name__ == "__main__":
    result = run_genai_benchmark(
        api_base="http://localhost:8000",
        api_key="your-api-key",
        model_name="llama-2-7b-chat",
        tokenizer_path="/path/to/tokenizer",
        max_requests=1000,
        concurrency=100,
        traffic_scenario="D(100,100)"
    )
```

### Example 2: Scaling Efficiency Measurement

**File:** `examples/ch10_scaling_efficiency.py`

```python
"""
Measure scaling efficiency across different GPU counts.
Run with: torchrun --nproc_per_node=N examples/ch10_scaling_efficiency.py
"""
import torch
import torch.distributed as dist
import time
import numpy as np

def measure_throughput(model, dataloader, num_iterations=100):
    """Measure throughput for current configuration"""
    # Warmup
    for i, (data, target) in enumerate(dataloader):
        if i >= 10:
            break
        _ = model(data)
    
    # Measurement
    torch.cuda.synchronize()
    start = time.time()
    
    for i, (data, target) in enumerate(dataloader):
        if i >= num_iterations:
            break
        _ = model(data)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return num_iterations / elapsed

def benchmark_scaling():
    """Benchmark and report scaling efficiency"""
    results = {}
    
    for num_gpus in [1, 2, 4, 8]:
        print(f"\nBenchmarking with {num_gpus} GPU(s)...")
        throughput = measure_throughput(model, dataloader)
        results[num_gpus] = throughput
        print(f"Throughput: {throughput:.2f} samples/sec")
    
    # Calculate scaling efficiency
    baseline = results[1]
    print("\nScaling Efficiency:")
    for n in [2, 4, 8]:
        ideal = baseline * n
        actual = results[n]
        efficiency = (actual / ideal) * 100
        print(f"{n} GPUs: {actual:.2f} samples/sec "
              f"(ideal: {ideal:.2f}, efficiency: {efficiency:.1f}%)")
```

### Example 3: Network Diagnostic Tools

**File:** `examples/ch10_network_diagnostics.py`

```python
"""
Network diagnostic tools for distributed training.
Run with: torchrun --nproc_per_node=2 examples/ch10_network_diagnostics.py
"""
import torch
import torch.distributed as dist
import time

def test_bandwidth(rank, world_size):
    """Test network bandwidth between nodes"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    sizes_mb = [1, 10, 100, 1000]
    
    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024 // 4
        tensor = torch.randn(size, device='cuda')
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        bandwidth = (size_mb * 2 * iterations) / elapsed
        
        if rank == 0:
            print(f"Size: {size_mb}MB, Bandwidth: {bandwidth:.2f} MB/s")
    
    dist.destroy_process_group()
```



## Best Practices and Common Pitfalls

This section consolidates the lessons learned throughout the chapter into actionable guidelines.

### The Benchmarking Checklist

Before running any benchmark, verify:

1. **Warmup:** At least 10-20 iterations before measurement
2. **Iterations:** At least 100 measurement iterations
3. **Runs:** At least 3-5 independent runs
4. **Synchronization:** `torch.cuda.synchronize()` before timing
5. **Documentation:** Hardware, software versions, configuration recorded
6. **Seeds:** Random seeds fixed for reproducibility

### Common Mistakes and Fixes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| No warmup | Cold start included in measurements | Add 10-20 warmup iterations |
| Single measurement | Results dominated by noise | Run 100+ iterations, report statistics |
| Including data loading | Inflated inference time | Pre-load data before timing |
| Ignoring variance | False confidence in results | Report std, P95, P99 |
| Wrong traffic pattern | Benchmark doesn't reflect production | Use realistic request distributions |

### Use Case: Comparing Inference Engines

**Scenario:** Choose between vLLM, SGLang, and TensorRT-LLM for production

**Approach:**
1. Define metrics: Throughput, latency (P50/P95/P99), memory usage
2. Create realistic workload using production request patterns
3. Run benchmarks with genai-bench for consistency
4. Analyze results across engines
5. Consider tradeoffs: latency vs throughput, memory vs speed

**Example Results:**
```
Engine      Throughput    P50 Latency    P95 Latency    Memory
vLLM        150 tok/s    0.15s          0.35s          24GB
SGLang      180 tok/s    0.12s          0.28s          22GB
TensorRT    200 tok/s    0.10s          0.25s          26GB
```

### Use Case: Optimizing Multi-Node Clusters

**Scenario:** Improve scaling efficiency of 8-node training cluster from 52% to 80%

**Diagnostic Steps:**
1. Profile communication vs compute time
2. Check data loading throughput
3. Measure GPU utilization
4. Test network bandwidth between nodes

**Common Fixes:**
- **Communication bottleneck:** Enable gradient compression, optimize bucket size
- **Data loading bottleneck:** Increase `num_workers`, use `pin_memory=True`
- **Compute bottleneck:** Check for CPU-GPU synchronization points


## Summary

This chapter has equipped you with the tools and techniques to systematically benchmark and optimize distributed AI systems. The key takeaways:

1. **Methodology matters:** Proper warmup, multiple runs, and variance analysis separate meaningful benchmarks from noise.

2. **Right tools for the job:** PyTorch Profiler and Nsight for training; genai-bench for inference; GLUE, MMLU, and HumanEval for accuracy.

3. **Dual benchmarking:** Both performance (speed, throughput) and accuracy (quality, correctness) must be measured—optimizations that degrade accuracy are not optimizations.

4. **Network is often the bottleneck:** Communication overhead grows with scale. Profile it, understand it, optimize it.

5. **Scaling has limits:** Amdahl's Law sets theoretical bounds. Measure your scaling efficiency to understand where you stand.

6. **Reproducibility enables progress:** Document everything. Version control benchmark scripts. Fix random seeds.

**Skills you've gained:**

- Design reproducible benchmark experiments with proper warmup and measurement
- Profile training workloads using PyTorch Profiler and Nsight Systems
- Benchmark inference systems using genai-bench with realistic traffic patterns
- Evaluate model accuracy using standard benchmarks
- Diagnose network bottlenecks using communication profiling
- Calculate and interpret scaling efficiency

Effective benchmarking is the foundation of performance optimization. Without accurate measurements, optimization efforts are blind guesses. The techniques in this chapter provide the visibility needed to make data-driven decisions about your distributed AI systems.

Throughout this book, we've covered the current state of distributed AI: DDP and FSDP for training, vLLM and SGLang for inference, Slurm for job scheduling, and production serving stacks. But the field is rapidly evolving. The final chapter explores emerging trends and future directions: MoE scaling, hybrid edge-cloud architectures, advanced parallelism strategies, and cost optimization techniques. Understanding where the field is heading will help you position yourself for the next wave of distributed AI innovations.



<!-- include: exercises/torch.md if include_math -->
<!-- include: exercises/torch.md if include_torch -->

## Further Reading

**Performance Benchmarking:**

- genai-bench Documentation: https://github.com/sgl-project/genai-bench
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- MLPerf: https://mlcommons.org/en/inference-edge-21/
- Amdahl's Law: https://en.wikipedia.org/wiki/Amdahl%27s_law
- [MLPerf Inference Benchmark](https://arxiv.org/pdf/1911.02549)
- [LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators](https://arxiv.org/html/2411.00136v1)
- [Meta-Metrics and Best Practices for System-Level Inference Performance Benchmarking](https://arxiv.org/html/2508.10251)
- https://github.com/IBM/fmwork
- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html 



**Accuracy Benchmarking:**

- GenAI-Bench (Text-to-Visual Evaluation): https://linzhiqiu.github.io/papers/genai_bench/
- GLUE Benchmark: https://gluebenchmark.com/
- MMLU Benchmark: https://github.com/hendrycks/test
- HumanEval (Code Generation): https://github.com/openai/human-eval
- HELM (Holistic Evaluation): https://crfm.stanford.edu/helm/
- Hugging Face Evaluate: https://huggingface.co/docs/evaluate/
- https://github.com/NVIDIA-NeMo/Evaluator
- https://huggingface.co/blog/nvidia/nemotron-3-nano-evaluation-recipe
- https://docs.nvidia.com/nim/benchmarking/llm/latest/index.html


