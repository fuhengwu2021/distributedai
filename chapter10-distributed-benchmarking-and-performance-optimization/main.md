# Chapter 10: Distributed Benchmarking and Performance Optimization

## Overview

This chapter teaches readers how to benchmark distributed training and inference systems rigorously using tools like genai-bench, MLPerf, and custom profiling scripts. It covers both **performance benchmarking** (throughput, latency, scaling efficiency) and **accuracy benchmarking** (model quality, output correctness). Topics include warmup methodology, scaling efficiency, network bottleneck identification, accuracy evaluation, and performance analysis. By the end of this chapter, readers will be able to design reproducible benchmark experiments, identify performance bottlenecks, evaluate model accuracy, and optimize distributed systems effectively.

**Chapter Length:** 28 pages



## 1. Benchmarking Methodology and Metrics

Benchmarking distributed AI systems is fundamentally different from benchmarking single-device workloads. The complexity arises from multiple dimensions: multiple GPUs, network communication, synchronization overhead, and system-level interactions. A rigorous benchmarking methodology is essential for making informed decisions about system design and optimization.

### Why Benchmarking Matters

**Key Reasons:**
- **Performance Validation:** Ensure systems meet latency and throughput requirements
- **Optimization Guidance:** Identify bottlenecks to prioritize optimization efforts
- **Cost Analysis:** Understand resource utilization to optimize costs
- **Comparison:** Fairly compare different systems, frameworks, and configurations
- **Regression Detection:** Catch performance regressions in CI/CD pipelines

### Core Metrics for Distributed Systems

**Throughput Metrics:**
- **Samples per Second (Training):** Number of training samples processed per second
- **Tokens per Second (Inference):** Number of tokens generated per second
- **Requests per Second (Serving):** Number of API requests handled per second
- **GPU Utilization:** Percentage of time GPUs are actively computing

**Latency Metrics:**
- **P50 (Median):** 50th percentile latency
- **P95:** 95th percentile latency (captures tail latency)
- **P99:** 99th percentile latency (worst-case scenarios)
- **Time to First Token (TTFT):** Latency for first output in inference
- **Time per Token (TPT):** Average time between tokens

**Efficiency Metrics:**
- **Scaling Efficiency:** How well performance scales with number of devices
- **Memory Efficiency:** Memory utilization vs available memory
- **Communication Overhead:** Time spent on synchronization vs computation
- **Cost per Token/Request:** Economic efficiency metric

### Benchmarking Methodology

**1. Warmup Phase:**
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

**2. Multiple Runs and Variance:**
- Always run multiple independent runs (minimum 3-5 runs)
- Report mean, standard deviation, and confidence intervals
- Use statistical significance tests when comparing systems

**3. Steady-State Measurement:**
- Discard initial iterations (warmup)
- Measure over sufficient duration to capture variance
- Account for system-level effects (thermal throttling, background processes)

**4. Reproducibility:**
- Fix random seeds
- Document hardware configuration
- Version control benchmark scripts
- Record system state (driver versions, CUDA version, etc.)

### Common Benchmarking Pitfalls

**Pitfall 1: Insufficient Warmup**
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
```python
# Wrong: Single measurement
time_taken = measure_once()

# Correct: Multiple measurements with statistics
times = [measure() for _ in range(10)]
mean_time = np.mean(times)
std_time = np.std(times)
print(f"Time: {mean_time:.3f} ± {std_time:.3f} seconds")
```

**Pitfall 3: Measuring Wrong Thing**
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



## 2. Training Benchmarking Tools and Procedures

Benchmarking distributed training requires understanding the full pipeline: data loading, forward pass, backward pass, gradient synchronization, and optimizer updates. Each component contributes to overall training time and must be measured separately.

### PyTorch Profiler

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

**Advanced Profiling:**
```python
# Profile with schedule
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

### Nsight Systems

**Command Line Usage:**
```bash
# Profile training script
nsys profile --trace=cuda,nvtx,osrt \
    --output=training_profile.nsys-rep \
    python train.py

# Generate report
nsys stats --report gputrace training_profile.nsys-rep
```

**Key Metrics from Nsight:**
- GPU kernel execution time
- Memory transfer time (H2D, D2H)
- CUDA API calls
- Synchronization points
- Communication operations (NCCL)

### Custom Benchmarking Script

**Comprehensive Training Benchmark:**
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
        """Benchmark a single training iteration"""
        if warmup:
            # Warmup iteration
            data, target = next(iter(self.dataloader))
            _ = self._training_step(data, target)
            torch.cuda.synchronize()
        
        # Actual measurement
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
        
        # Communication (if distributed)
        comm_start = time.time()
        if dist.is_initialized():
            # Gradients are synchronized automatically in DDP
            pass
        torch.cuda.synchronize()
        comm_time = time.time() - comm_start
        
        # Optimizer step
        optimizer_start = time.time()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        optimizer_time = time.time() - optimizer_start
        
        total_time = forward_time + backward_time + comm_time + optimizer_time
        
        return {
            'data_loading': data_time,
            'forward': forward_time,
            'backward': backward_time,
            'communication': comm_time,
            'optimizer': optimizer_time,
            'total': total_time
        }
    
    def benchmark(self, num_warmup=10, num_iterations=100):
        """Run full benchmark"""
        # Warmup
        for _ in range(num_warmup):
            self.benchmark_iteration(warmup=False)
        
        # Actual benchmark
        for _ in range(num_iterations):
            metrics = self.benchmark_iteration(warmup=False)
            for key, value in metrics.items():
                self.metrics[key].append(value)
        
        # Compute statistics
        stats = {}
        for key, values in self.metrics.items():
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        
        return stats
    
    def _training_step(self, data, target):
        """Internal training step"""
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

### Measuring Scaling Efficiency

**Scaling Efficiency Calculation:**
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

**Scaling Efficiency Analysis:**
- **>90%:** Excellent scaling
- **70-90%:** Good scaling (typical for well-optimized systems)
- **50-70%:** Moderate scaling (communication overhead significant)
- **<50%:** Poor scaling (bottleneck identified)



## 3. Inference Benchmarking with genai-bench

Inference benchmarking has unique challenges: variable request patterns, caching effects, and tail latency requirements. genai-bench provides a comprehensive framework for benchmarking inference systems with realistic workloads.

**Note:** This section covers the **performance benchmarking** tool from [sgl-project/genai-bench](https://github.com/sgl-project/genai-bench), which measures engineering metrics (throughput, latency, scaling). This is distinct from accuracy/quality benchmarking tools (e.g., [GenAI-Bench for text-to-visual evaluation](https://linzhiqiu.github.io/papers/genai_bench/)) that measure model output quality and alignment with prompts. This chapter focuses on performance optimization, not model accuracy evaluation.

### genai-bench Overview

**Key Features:**
- CLI-based benchmarking tool (not a Python API)
- Realistic prompt distributions via traffic scenarios
- Configurable load patterns (concurrency, traffic scenarios)
- Automatic result aggregation and storage
- Support for multiple inference engines (vLLM, SGLang, OpenAI, AWS Bedrock, Azure, GCP, OCI)
- Live UI dashboard for real-time progress monitoring
- Comprehensive Excel reports and plot generation
- Latency percentile tracking (TTFT, E2E, TPOT)

### Setting Up genai-bench

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
    --server-version "v0.9.2" \
    --server-gpu-count 4
```

**Key Parameters:**
- `--api-backend`: Backend type (openai, vllm, sglang, aws-bedrock, azure-openai, gcp-vertex, oci-genai, oci-cohere)
- `--api-base`: API endpoint URL
- `--api-model-name`: Model name for requests
- `--model-tokenizer`: Path to tokenizer (required for token counting)
- `--task`: Task type (text-to-text, text-to-embeddings, image-text-to-text, etc.)
- `--max-time-per-run`: Maximum duration per run in minutes
- `--max-requests-per-run`: Maximum requests to send per run
- `--num-concurrency`: Number of concurrent requests (can specify multiple values)
- `--traffic-scenario`: Traffic scenario definition (can specify multiple scenarios)
- `--server-engine`, `--server-gpu-type`, `--server-version`, `--server-gpu-count`: Server metadata

**Traffic Scenarios:**
Traffic scenarios define input/output token distributions:
- `D(input_tokens, output_tokens)`: Deterministic fixed tokens
- `I(input_tokens, output_tokens)`: Image-text input with fixed tokens
- `E(input_tokens)`: Embedding input with fixed tokens
- Custom distributions via dataset configs

**Example with Multiple Scenarios and Concurrencies:**
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
    --server-gpu-type "H100" \
    --server-version "v0.9.2"
```

**Analyzing Results:**
After running a benchmark, genai-bench saves results in an `experiments/` folder. You can analyze them:

**Generate Excel Report:**
```bash
genai-bench excel \
    --experiment-folder ./experiments/your_experiment \
    --excel-name benchmark_results \
    --metric-percentile mean \
    --metrics-time-unit s
```

**Generate Plots:**
```bash
genai-bench plot \
    --experiments-folder ./experiments \
    --group-key traffic_scenario \
    --preset 2x4_default
```

**Using Python to Call CLI (Programmatic Access):**
```python
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
    traffic_scenario: str = "D(100,100)"
):
    """Run genai-bench via CLI from Python"""
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
        "--traffic-scenario", traffic_scenario
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def analyze_results(experiment_folder: str):
    """Generate Excel and plots from experiment results"""
    # Excel report
    subprocess.run([
        "genai-bench", "excel",
        "--experiment-folder", experiment_folder,
        "--excel-name", "results",
        "--metric-percentile", "mean"
    ])
    
    # Plots
    subprocess.run([
        "genai-bench", "plot",
        "--experiments-folder", experiment_folder,
        "--group-key", "traffic_scenario",
        "--preset", "2x4_default"
    ])

# Usage
result = run_genai_benchmark(
    api_base="http://localhost:8000",
    api_key="your-key",
    model_name="llama-2-7b-chat",
    tokenizer_path="/path/to/tokenizer"
)

# Analyze results (experiment folder from genai-bench output)
# analyze_results("./experiments/your_experiment")
```

### Custom Inference Benchmark

**Comprehensive Inference Benchmark:**
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
    
    def generate_prompts(self, num_prompts, length_mean=50, length_std=20):
        """Generate prompts with specified distribution"""
        lengths = np.random.lognormal(
            mean=np.log(length_mean),
            sigma=length_std / length_mean,
            size=num_prompts
        ).astype(int)
        
        prompts = []
        for length in lengths:
            prompt = " ".join([f"word{i}" for i in range(length)])
            prompts.append(prompt)
        
        return prompts
    
    def benchmark_single_request(self, prompt, max_tokens=512):
        """Benchmark a single inference request"""
        # Tokenize
        tokenize_start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        tokenize_time = time.time() - tokenize_start
        
        # Warmup (if first request)
        if not hasattr(self, '_warmed_up'):
            _ = self.model.generate(**inputs, max_new_tokens=1)
            torch.cuda.synchronize()
            self._warmed_up = True
        
        # Time to first token
        torch.cuda.synchronize()
        ttft_start = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True
        )
        torch.cuda.synchronize()
        ttft = time.time() - ttft_start
        
        # Time per token
        num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        tpt = ttft / num_tokens if num_tokens > 0 else 0
        
        return {
            'tokenize_time': tokenize_time,
            'ttft': ttft,
            'tpt': tpt,
            'num_tokens': num_tokens
        }
    
    def benchmark_concurrent(self, prompts, concurrency=10):
        """Benchmark with concurrent requests"""
        async def process_request(prompt):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.benchmark_single_request,
                prompt
            )
            return result
        
        async def run_benchmark():
            tasks = []
            for prompt in prompts:
                task = asyncio.create_task(process_request(prompt))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        start = time.time()
        results = asyncio.run(run_benchmark())
        total_time = time.time() - start
        
        # Aggregate metrics
        for result in results:
            for key, value in result.items():
                self.metrics[key].append(value)
        
        return {
            'total_time': total_time,
            'throughput': len(prompts) / total_time,
            'results': results
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

**Cold Start Benchmarking:**
```python
def benchmark_cold_start(model, prompts):
    """Measure cold start performance"""
    # Clear cache
    torch.cuda.empty_cache()
    
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

Reasoning models (chain-of-thought, multi-step decision-making, or tool-augmented LLMs) behave differently from single-pass generation models. For these models you should measure both per-step latency and end-to-end session latency, and separate local generation time from external-call time (retrievals, tool calls, networked services).

Key measurement points:
- **Per-step latency:** Measure TTFT/TPT for each reasoning step separately to see whether some steps are significantly slower.
- **End-to-end session latency:** Total time to complete the whole reasoning session (sum of steps + external calls).
- **External-call breakdown:** Measure time spent waiting for retrievals, database queries, or tool responses vs local model generation.
- **Cache / KV effects:** Compare cold vs warm runs when KV cache or retrieval caches are warmed — important for long-context reasoning.
- **Distributed step overhead:** If steps require cross-device synchronization (e.g., large-model sharding, cross-node retrieval), measure inter-step communication cost.
- **Quality vs latency trade-off:** Track how adding more reasoning steps (or more complex tool calls) affects both latency and task quality (accuracy/F1/etc.).

Minimal example: measure per-step and total session times (replace `model.generate` and tool calls with real calls):

```python
import time
import numpy as np

def run_reasoning_step(model, step_input, max_new_tokens=64, do_tool_call=None):
    # Measure local generation
    gen_start = time.time()
    out = model.generate(step_input, max_new_tokens=max_new_tokens)
    # If using CUDA, synchronize here
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        pass
    gen_time = time.time() - gen_start

    tool_time = 0.0
    if do_tool_call:
        # Example: external retrieval or tool call
        t0 = time.time()
        tool_result = do_tool_call()
        tool_time = time.time() - t0

    return gen_time, tool_time, out

def measure_reasoning_session(model, session_steps, do_tool_call_fn=None):
    per_step = []
    total = 0.0
    for step_input in session_steps:
        gen_t, tool_t, out = run_reasoning_step(model, step_input, do_tool_call=(do_tool_call_fn if do_tool_call_fn else None))
        per_step.append({'gen_time': gen_t, 'tool_time': tool_t, 'step_total': gen_t + tool_t})
        total += gen_t + tool_t

    times = [s['step_total'] for s in per_step]
    return {
        'per_step': per_step,
        'total': total,
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }

# Example usage:
# stats = measure_reasoning_session(model, ['step1 prompt', 'step2 prompt', 'step3 prompt'], do_tool_call_fn=call_retriever)
# Print per-step and total latencies and store alongside quality metrics for analysis.
```

Guidance:
- Instrument and log timestamps for each reasoning step and every external dependency.
- Correlate latency traces with quality/accuracy to choose the optimal number of reasoning steps.
- When benchmarking across systems, include both per-step breakdowns and session-level aggregates in reports (P50/P95/P99 and mean/std).



## 4. Accuracy and Quality Benchmarking

While performance benchmarking measures speed, throughput, and efficiency, **accuracy benchmarking** evaluates the quality and correctness of model outputs. For distributed AI systems, accuracy benchmarking ensures that optimizations and scaling don't degrade model quality, and helps compare different models, configurations, and serving strategies.

### Why Accuracy Benchmarking Matters

**Key Reasons:**
- **Quality Assurance:** Ensure distributed optimizations don't degrade model accuracy
- **Model Comparison:** Fairly compare different models, quantization levels, or serving engines
- **Regression Detection:** Catch accuracy regressions in CI/CD pipelines
- **Optimization Trade-offs:** Understand accuracy vs performance trade-offs
- **Production Validation:** Verify models meet quality requirements before deployment

### Accuracy Metrics for LLMs

**Text Generation Quality Metrics:**
- **BLEU Score:** Measures n-gram overlap with reference text (common for translation)
- **ROUGE Score:** Measures overlap of n-grams, longest common subsequence (common for summarization)
- **METEOR:** Considers synonyms and word order
- **BERTScore:** Semantic similarity using BERT embeddings
- **Human Evaluation:** Gold standard but expensive (Likert scales, pairwise comparisons)

**Task-Specific Metrics:**
- **Classification Accuracy:** For classification tasks
- **F1 Score:** For tasks with precision/recall trade-offs
- **Exact Match (EM):** For question answering
- **Code Execution Accuracy:** For code generation (run code and check output)

**Compositional Reasoning Metrics:**
- **VQAScore:** Uses VQA models to evaluate image-text alignment
- **CLIPScore:** Measures image-text similarity using CLIP
- **PickScore:** Human preference prediction for image generation
- **HPSv2:** Human preference score for image quality

### GenAI-Bench for Text-to-Visual Evaluation

**GenAI-Bench** ([linzhiqiu.github.io/papers/genai_bench/](https://linzhiqiu.github.io/papers/genai_bench/)) is a comprehensive benchmark for evaluating compositional text-to-visual generation. Unlike performance benchmarks, it focuses on **model output quality** and alignment with complex prompts.

**Key Features:**
- **1,600 professionally-designed prompts** covering compositional reasoning
- **Human ratings** (38,400+ ratings) for benchmarking automated metrics
- **Compositional skills evaluation:** Objects, attributes, relationships, counting, logic, comparison
- **Multiple model evaluation:** DALL-E 3, Stable Diffusion, Midjourney, Pika, Gen2, etc.
- **VQAScore integration:** Automated evaluation metric that correlates well with human judgments

**Compositional Skills Evaluated:**
- **Basic Compositions:** Objects, scenes, attributes, spatial/action/part relationships
- **Advanced Reasoning:** Counting, comparison, differentiation, logic (negation/universality)

**Using VQAScore for Evaluation:**

VQAScore uses a VQA (Visual Question Answering) model to evaluate image-text alignment:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

def compute_vqascore(image, prompt):
    """
    Compute VQAScore for image-prompt alignment.
    
    VQAScore = P("Yes" | image, question="Does the image match: {prompt}?")
    """
    # Load VQA model
    processor = AutoProcessor.from_pretrained("model_name")
    model = AutoModelForVision2Seq.from_pretrained("model_name")
    
    # Create question
    question = f"Does the image match: {prompt}?"
    
    # Process inputs
    inputs = processor(images=image, text=question, return_tensors="pt")
    
    # Get probability of "Yes"
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract probability of "Yes" answer
        vqa_score = extract_yes_probability(outputs)
    
    return vqa_score

# Example usage
image = load_image("generated_image.png")
prompt = "A red apple on a wooden table"
score = compute_vqascore(image, prompt)
print(f"VQAScore: {score:.3f}")  # Higher is better
```

**Improving Generation with VQAScore:**

VQAScore can improve generation quality by selecting the best candidate from multiple generations:

```python
def generate_with_vqascore_selection(model, prompt, num_candidates=5):
    """Generate multiple candidates and select best using VQAScore"""
    candidates = []
    scores = []
    
    # Generate multiple candidates
    for _ in range(num_candidates):
        image = model.generate(prompt)
        score = compute_vqascore(image, prompt)
        candidates.append(image)
        scores.append(score)
    
    # Select highest scoring candidate
    best_idx = np.argmax(scores)
    return candidates[best_idx], scores[best_idx]

# Usage
best_image, best_score = generate_with_vqascore_selection(
    model, 
    "A red apple on a wooden table",
    num_candidates=9
)
```

### LLM Accuracy Benchmarking

**Standard Benchmarks:**

**1. GLUE (General Language Understanding Evaluation):**
```python
# Using Hugging Face Evaluate
from evaluate import load

glue = load("glue", "sst2")  # Sentiment classification
results = glue.compute(predictions=predictions, references=references)
print(f"Accuracy: {results['accuracy']:.3f}")
```

**2. SuperGLUE:**
- More challenging tasks than GLUE
- Includes reading comprehension, natural language inference, etc.

**3. MMLU (Massive Multitask Language Understanding):**
```python
# MMLU evaluates knowledge across 57 tasks
# Categories: STEM, humanities, social sciences, etc.
def evaluate_mmlu(model, dataset):
    """Evaluate model on MMLU benchmark"""
    correct = 0
    total = 0
    
    for example in dataset:
        prompt = example['question']
        choices = example['choices']
        answer = model.generate(prompt, choices)
        
        if answer == example['answer']:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy
```

**4. HumanEval (Code Generation):**
```python
def evaluate_humaneval(model, problems):
    """Evaluate code generation on HumanEval"""
    results = []
    
    for problem in problems:
        prompt = problem['prompt']
        solution = model.generate_code(prompt)
        
        # Execute and test
        passed = test_solution(solution, problem['test'])
        results.append(passed)
    
    pass_rate = sum(results) / len(results)
    return pass_rate
```

**5. HELM (Holistic Evaluation of Language Models):**
- Comprehensive evaluation across multiple scenarios
- Includes accuracy, robustness, fairness, efficiency

### Custom Accuracy Benchmarking

**Creating Custom Evaluation Scripts:**

```python
import json
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AccuracyBenchmark:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.results = []
    
    def evaluate_classification(self, examples: List[Dict]):
        """Evaluate classification accuracy"""
        correct = 0
        total = 0
        
        for example in examples:
            prompt = example['input']
            expected_label = example['label']
            
            # Generate prediction
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_label = self._extract_label(outputs)
            
            if predicted_label == expected_label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_generation(self, examples: List[Dict], metric='bleu'):
        """Evaluate text generation quality"""
        scores = []
        
        for example in examples:
            prompt = example['input']
            reference = example['reference']
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=512)
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Compute metric
            if metric == 'bleu':
                score = compute_bleu(generated, reference)
            elif metric == 'rouge':
                score = compute_rouge(generated, reference)
            elif metric == 'bertscore':
                score = compute_bertscore(generated, reference)
            
            scores.append(score)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }
    
    def evaluate_distributed_accuracy(self, num_gpus_list=[1, 2, 4, 8]):
        """Compare accuracy across different distributed configurations"""
        results = {}
        
        for num_gpus in num_gpus_list:
            # Setup distributed model
            model = setup_distributed_model(self.model, num_gpus)
            
            # Evaluate
            accuracy = self.evaluate_classification(self.dataset)
            results[num_gpus] = accuracy
        
        return results
    
    def _extract_label(self, outputs):
        """Extract predicted label from model outputs"""
        # Implementation depends on model architecture
        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1)
        return predicted_id.item()

# Usage
benchmark = AccuracyBenchmark(model, tokenizer, dataset)
results = benchmark.evaluate_classification(test_examples)
print(f"Accuracy: {results['accuracy']:.3f}")
```

### Evaluating Distributed Training Accuracy

**Ensuring Distributed Training Maintains Accuracy:**

```python
def compare_centralized_vs_distributed_accuracy(
    model_centralized,
    model_distributed,
    test_dataset
):
    """Compare accuracy between centralized and distributed training"""
    # Evaluate centralized model
    acc_centralized = evaluate_model(model_centralized, test_dataset)
    
    # Evaluate distributed model
    acc_distributed = evaluate_model(model_distributed, test_dataset)
    
    # Calculate accuracy drop
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

**Evaluating Quantization Impact:**

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

### Best Practices for Accuracy Benchmarking

**1. Use Representative Datasets:**
- Test on production-like data distributions
- Include edge cases and failure modes
- Ensure sufficient sample size for statistical significance

**2. Multiple Metrics:**
- Don't rely on a single metric
- Use task-appropriate metrics
- Include human evaluation when possible

**3. Baseline Comparison:**
- Always compare against a baseline (previous model, centralized training, etc.)
- Track accuracy over time to detect regressions

**4. Statistical Significance:**
```python
from scipy import stats

def compare_models_statistically(model1_scores, model2_scores):
    """Test if accuracy difference is statistically significant"""
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    if p_value < 0.05:
        print(f"Statistically significant difference (p={p_value:.4f})")
    else:
        print(f"No significant difference (p={p_value:.4f})")
    
    return t_stat, p_value
```

**5. Document Everything:**
- Dataset versions and splits
- Evaluation methodology
- Model versions and configurations
- Random seeds used



## 5. Network Bottleneck Diagnosis

Network bottlenecks are often the limiting factor in distributed training and inference. Identifying and diagnosing network issues requires understanding communication patterns, measuring bandwidth, and analyzing topology.

### Network Monitoring Tools

**1. iftop (Interface Top):**
```bash
# Monitor network traffic
sudo iftop -i eth0

# Filter by specific hosts
sudo iftop -i eth0 -f "host 192.168.1.10"
```

**2. nload:**
```bash
# Real-time network bandwidth monitor
nload eth0
```

**3. iperf3 (Bandwidth Testing):**
```bash
# Server side
iperf3 -s

# Client side
iperf3 -c server_ip -t 60 -i 1
```

### NCCL Communication Tests

**NCCL AllReduce Test:**
```python
import torch
import torch.distributed as dist
import time

def test_allreduce_bandwidth(rank, world_size):
    """Test AllReduce bandwidth"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Test different message sizes
    sizes = [1, 10, 100, 1000, 10000]  # MB
    
    for size_mb in sizes:
        size = size_mb * 1024 * 1024 // 4  # Convert to float32 elements
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
        
        # Calculate bandwidth
        data_transferred = size_mb * 2 * num_iterations  # Send + receive
        bandwidth = data_transferred / elapsed  # MB/s
        
        if rank == 0:
            print(f"Size: {size_mb}MB, Bandwidth: {bandwidth:.2f} MB/s")
    
    dist.destroy_process_group()
```

### Identifying Communication Patterns

**Communication Overhead Analysis:**
```python
def analyze_communication_overhead(model, dataloader, num_iterations=100):
    """Analyze communication vs computation time"""
    comm_times = []
    compute_times = []
    
    for i, (data, target) in enumerate(dataloader):
        if i >= num_iterations:
            break
        
        # Forward + backward (computation)
        compute_start = time.time()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.cuda.synchronize()
        compute_time = time.time() - compute_start
        
        # Communication (gradient sync)
        comm_start = time.time()
        # In DDP, this happens automatically during backward
        # But we can measure it separately
        torch.cuda.synchronize()
        comm_time = time.time() - comm_start
        
        compute_times.append(compute_time)
        comm_times.append(comm_time)
        
        optimizer.step()
        optimizer.zero_grad()
    
    # Analyze
    total_compute = sum(compute_times)
    total_comm = sum(comm_times)
    total_time = total_compute + total_comm
    
    print(f"Compute time: {total_compute:.2f}s ({total_compute/total_time*100:.1f}%)")
    print(f"Communication time: {total_comm:.2f}s ({total_comm/total_time*100:.1f}%)")
    print(f"Communication overhead: {total_comm/total_compute*100:.1f}%")
```

### Topology-Aware Communication

**Detecting Topology:**
```python
import torch.distributed as dist

def detect_topology():
    """Detect network topology"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Test communication with each peer
    topology = {}
    for peer_rank in range(world_size):
        if peer_rank == rank:
            continue
        
        # Measure latency
        tensor = torch.randn(1000, device='cuda')
        start = time.time()
        dist.send(tensor, dst=peer_rank)
        dist.recv(tensor, src=peer_rank)
        latency = time.time() - start
        
        topology[peer_rank] = latency
    
    return topology
```



## 6. Scaling Efficiency and Optimization

Scaling efficiency measures how well a system utilizes additional resources. Understanding scaling efficiency helps identify bottlenecks and guide optimization efforts.

### Scaling Efficiency Metrics

**Linear Scaling Efficiency:**
```python
def calculate_scaling_efficiency(throughput_1, throughput_n, n):
    """
    Calculate linear scaling efficiency.
    
    Efficiency = (Actual Throughput) / (Ideal Throughput) * 100%
    Ideal Throughput = Throughput_1 * N
    """
    ideal_throughput = throughput_1 * n
    efficiency = (throughput_n / ideal_throughput) * 100
    return efficiency

# Example
throughput_1gpu = 100  # samples/sec
throughput_8gpu = 650  # samples/sec
efficiency = calculate_scaling_efficiency(throughput_1gpu, throughput_8gpu, 8)
print(f"Scaling efficiency: {efficiency:.1f}%")  # 81.25%
```

**Amdahl's Law Application:**
```python
def amdahl_speedup(serial_fraction, n):
    """
    Calculate speedup using Amdahl's Law.
    
    Speedup = 1 / (S + P/N)
    where S = serial fraction, P = parallel fraction, N = processors
    """
    parallel_fraction = 1 - serial_fraction
    speedup = 1 / (serial_fraction + parallel_fraction / n)
    return speedup

# Example: If 10% of work is serial
serial_fraction = 0.10
n = 8
speedup = amdahl_speedup(serial_fraction, n)
print(f"Maximum speedup with {n} GPUs: {speedup:.2f}x")
```

### Identifying Scaling Bottlenecks

**Bottleneck Analysis:**
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
    bucket_cap_mb=25,  # Bucket size
    find_unused_parameters=False
)
```

**2. Optimize Data Loading:**
```python
# Use multiple workers and prefetching
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster H2D transfer
    prefetch_factor=2  # Prefetch batches
)
```

**3. Gradient Accumulation:**
```python
# Accumulate gradients to simulate larger batch size
accumulation_steps = 4
for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```



## Hands-On Examples

### Example 1: genai-bench Benchmarking

**File:** `examples/ch10_genai_bench.py`

```python
"""
Comprehensive inference benchmarking using genai-bench CLI.
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
    print(result.stdout)
    
    # Results are saved in experiments/ folder
    # You can parse the results from the experiment folder
    return result

def analyze_experiment_results(experiment_folder: str):
    """Analyze results from genai-bench experiment"""
    # Generate Excel report
    excel_cmd = [
        "genai-bench", "excel",
        "--experiment-folder", experiment_folder,
        "--excel-name", "benchmark_results",
        "--metric-percentile", "mean"
    ]
    
    subprocess.run(excel_cmd)
    
    # Generate plots
    plot_cmd = [
        "genai-bench", "plot",
        "--experiments-folder", experiment_folder,
        "--group-key", "traffic_scenario",
        "--preset", "2x4_default"
    ]
    
    subprocess.run(plot_cmd)

if __name__ == "__main__":
    # Run benchmark
    result = run_genai_benchmark(
        api_base="http://localhost:8000",
        api_key="your-api-key",
        model_name="llama-2-7b-chat",
        tokenizer_path="/path/to/tokenizer",
        max_requests=1000,
        max_time_minutes=15,
        concurrency=100,
        traffic_scenario="D(100,100)"
    )
    
    # Analyze results (experiment folder path from genai-bench output)
    # analyze_experiment_results("./experiments/your_experiment_folder")
```

### Example 2: Scaling Efficiency Measurement

**File:** `examples/ch10_scaling_efficiency.py`

```python
"""
Measure scaling efficiency across different GPU counts.
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
    
    throughput = num_iterations / elapsed
    return throughput

def benchmark_scaling():
    """Benchmark scaling across different GPU counts"""
    results = {}
    
    for num_gpus in [1, 2, 4, 8]:
        print(f"\nBenchmarking with {num_gpus} GPU(s)...")
        
        # Setup model and dataloader (simplified)
        # In practice, you'd use torchrun or similar
        throughput = measure_throughput(model, dataloader)
        results[num_gpus] = throughput
        
        print(f"Throughput: {throughput:.2f} samples/sec")
    
    # Calculate scaling efficiency
    baseline_throughput = results[1]
    print("\nScaling Efficiency:")
    print(f"1 GPU: {baseline_throughput:.2f} samples/sec (baseline)")
    
    for n in [2, 4, 8]:
        ideal = baseline_throughput * n
        actual = results[n]
        efficiency = (actual / ideal) * 100
        print(f"{n} GPUs: {actual:.2f} samples/sec "
              f"(ideal: {ideal:.2f}, efficiency: {efficiency:.1f}%)")

if __name__ == "__main__":
    benchmark_scaling()
```

### Example 3: Network Diagnostic Tools

**File:** `examples/ch10_network_diagnostics.py`

```python
"""
Network diagnostic tools for distributed training.
"""
import torch
import torch.distributed as dist
import time
import subprocess

def test_bandwidth(rank, world_size):
    """Test network bandwidth between nodes"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Test different message sizes
    sizes_mb = [1, 10, 100, 1000]
    
    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024 // 4  # float32 elements
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
        
        # Calculate bandwidth
        data_mb = size_mb * 2 * iterations  # send + receive
        bandwidth = data_mb / elapsed
        
        if rank == 0:
            print(f"Size: {size_mb}MB, Bandwidth: {bandwidth:.2f} MB/s")
    
    dist.destroy_process_group()

def check_network_health():
    """Check network health using system tools"""
    # Check latency
    result = subprocess.run(
        ['ping', '-c', '5', 'target_host'],
        capture_output=True,
        text=True
    )
    print("Network Latency:")
    print(result.stdout)
    
    # Check bandwidth (requires iperf3)
    # result = subprocess.run(
    #     ['iperf3', '-c', 'target_host', '-t', '10'],
    #     capture_output=True,
    #     text=True
    # )
    # print("Network Bandwidth:")
    # print(result.stdout)

if __name__ == "__main__":
    # This would be called with torchrun
    # torchrun --nproc_per_node=2 examples/ch10_network_diagnostics.py
    pass
```



## Best Practices

### 1. Avoiding Incorrect Benchmarking Methods

**Common Mistakes:**
- **Measuring cold start as typical performance:** Always warmup before measurement
- **Single measurement:** Always run multiple iterations and report statistics
- **Including setup time:** Measure only the operation of interest
- **Ignoring system state:** Document and control system configuration

**Correct Approach:**
```python
def correct_benchmark(func, num_warmup=10, num_iterations=100):
    # Warmup
    for _ in range(num_warmup):
        func()
    
    # Synchronize
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_iterations):
        start = time.time()
        func()
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    # Report statistics
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
```

### 2. Measuring Variance Correctly

**Why Variance Matters:**
- Performance can vary significantly between runs
- Tail latency (P95, P99) is critical for production systems
- Variance helps identify instability issues

**Correct Variance Measurement:**
```python
def measure_with_variance(func, num_runs=5, num_iterations=100):
    """Measure with proper variance analysis"""
    all_times = []
    
    for run in range(num_runs):
        run_times = []
        for _ in range(num_iterations):
            start = time.time()
            func()
            torch.cuda.synchronize()
            run_times.append(time.time() - start)
        all_times.append(run_times)
    
    # Calculate statistics across runs
    mean_times = [np.mean(times) for times in all_times]
    
    return {
        'mean': np.mean(mean_times),
        'std': np.std(mean_times),
        'min': np.min(mean_times),
        'max': np.max(mean_times),
        'coefficient_of_variation': np.std(mean_times) / np.mean(mean_times)
    }
```

### 3. Understanding Warmup Behavior

**Warmup Effects:**
- First iteration is often slower (kernel compilation, memory allocation)
- CUDA kernels are JIT compiled on first use
- Memory allocation patterns stabilize after warmup
- Caching effects (KV cache in inference)

**Proper Warmup:**
```python
def benchmark_with_warmup(func, warmup_iterations=20, measure_iterations=100):
    """Benchmark with sufficient warmup"""
    # Warmup phase
    for _ in range(warmup_iterations):
        func()
        torch.cuda.synchronize()
    
    # Measurement phase
    times = []
    for _ in range(measure_iterations):
        start = time.time()
        func()
        torch.cuda.synchronize()
        times.append(time.time() - start)
    
    return times
```



## Use Cases

### Use Case 1: Comparing Inference Engines

**Scenario:** Choose between vLLM, SGLang, and TensorRT-LLM for production

**Approach:**
1. **Define metrics:** Throughput, latency (P50/P95/P99), memory usage
2. **Create realistic workload:** Use production request patterns
3. **Run benchmarks:** Use genai-bench for consistency
4. **Analyze results:** Compare across engines
5. **Consider tradeoffs:** Latency vs throughput, memory vs speed

**Example Results:**
```
Engine      Throughput    P50 Latency    P95 Latency    Memory
vLLM        150 tok/s    0.15s          0.35s          24GB
SGLang      180 tok/s    0.12s          0.28s          22GB
TensorRT    200 tok/s    0.10s          0.25s          26GB
```

### Use Case 2: Optimizing Multi-Node Clusters

**Scenario:** Improve scaling efficiency of 8-node training cluster

**Approach:**
1. **Measure current efficiency:** Calculate scaling efficiency
2. **Identify bottlenecks:** Profile communication, compute, data loading
3. **Optimize:** Address identified bottlenecks
4. **Re-measure:** Validate improvements

**Optimization Steps:**
- **If communication is bottleneck:** Use gradient compression, optimize topology
- **If data loading is bottleneck:** Increase num_workers, use prefetching
- **If compute is bottleneck:** Check GPU utilization, optimize kernels



## Skills Learned

By the end of this chapter, readers will be able to:

1. **Design reproducible benchmark experiments**
   - Create benchmark scripts with proper warmup and measurement
   - Document system configuration and environment
   - Use version control for benchmark code

2. **Benchmark training workloads**
   - Use PyTorch profiler and Nsight Systems
   - Measure per-component timing (forward, backward, communication)
   - Calculate scaling efficiency

3. **Benchmark inference workloads**
   - Use genai-bench CLI for realistic inference benchmarks
   - Configure traffic scenarios and concurrency levels
   - Generate Excel reports and plots for analysis
   - Measure latency metrics (TTFT - Time to First Token, E2E - End-to-End, TPOT - Time Per Output Token) with percentiles from experiment results

4. **Benchmark model accuracy and quality**
   - Use standard benchmarks (GLUE, MMLU, HumanEval) for LLM evaluation
   - Evaluate text-to-visual models with GenAI-Bench and VQAScore
   - Compare accuracy across different distributed configurations
   - Measure accuracy impact of quantization and optimizations

5. **Identify communication bottlenecks**
   - Use network monitoring tools (iftop, nload)
   - Test NCCL communication patterns
   - Analyze topology impact on performance

6. **Optimize distributed performance**
   - Calculate and interpret scaling efficiency
   - Apply Amdahl's Law to understand limits
   - Implement optimization strategies based on bottleneck analysis



## Summary

This chapter has covered comprehensive benchmarking methodologies for distributed AI systems, covering both performance and accuracy evaluation. Key takeaways:

1. **Rigorous methodology:** Proper warmup, multiple runs, variance analysis
2. **Right tools:** PyTorch profiler, Nsight, genai-bench CLI for performance; GLUE, MMLU, GenAI-Bench for accuracy
3. **Dual benchmarking:** Both performance (speed, throughput) and accuracy (quality, correctness) are essential
4. **Network matters:** Communication bottlenecks are common in distributed systems
5. **Scaling efficiency:** Measure and optimize to maximize resource utilization
6. **Accuracy preservation:** Ensure distributed optimizations don't degrade model quality
7. **Reproducibility:** Document everything for fair comparisons

Effective benchmarking is the foundation of performance optimization and quality assurance. Without accurate measurements, optimization efforts are blind. The tools and techniques covered in this chapter provide a solid foundation for understanding and improving distributed AI system performance while maintaining model accuracy.



## Exercises

1. **Create a training benchmark:** Write a script that measures forward, backward, communication, and optimizer times separately. Run it with 1, 2, 4, and 8 GPUs and calculate scaling efficiency.

2. **Benchmark inference latency:** Use genai-bench CLI to benchmark an inference server with different traffic scenarios and concurrency levels. Generate Excel reports and plots to analyze TTFT, E2E latency, and TPOT metrics with percentiles (P50, P95, P99).

3. **Benchmark model accuracy:** Evaluate a model on a standard benchmark (e.g., MMLU or GLUE). Compare accuracy between centralized training, distributed training, and quantized versions. Use statistical tests to determine if accuracy differences are significant.

4. **Analyze communication overhead:** Profile a distributed training job and identify what percentage of time is spent on communication vs computation.

5. **Optimize a bottleneck:** Identify a bottleneck in a distributed system and implement an optimization. Measure both performance improvement and accuracy impact.



## Further Reading

**Performance Benchmarking:**
- genai-bench Documentation: https://github.com/sgl-project/genai-bench
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- MLPerf: https://mlcommons.org/en/inference-edge-21/
- Amdahl's Law: https://en.wikipedia.org/wiki/Amdahl%27s_law

**Accuracy Benchmarking:**
- GenAI-Bench (Text-to-Visual Evaluation): https://linzhiqiu.github.io/papers/genai_bench/
- GLUE Benchmark: https://gluebenchmark.com/
- MMLU Benchmark: https://github.com/hendrycks/test
- HumanEval (Code Generation): https://github.com/openai/human-eval
- HELM (Holistic Evaluation): https://crfm.stanford.edu/helm/
- Hugging Face Evaluate: https://huggingface.co/docs/evaluate/

