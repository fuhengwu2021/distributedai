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


## The Performance Gap in Distributed Systems

You've built the distributed AI system. DDP synchronizes gradients across your 8-GPU cluster. FSDP shards your 70B parameter model across nodes. vLLM serves inference requests with continuous batching. The production stack from Chapter 9 routes traffic to the right model instances. Everything *works*—but is it working *well*?

This is the question that separates functional systems from optimized ones. A training run that completes is not the same as a training run that efficiently utilizes your \$100,000 worth of GPUs. An inference endpoint that returns responses is not the same as one that meets your 200ms P99 latency SLA. The difference between "working" and "working well" can mean days of wasted training time, violated service agreements, and unnecessary cloud costs.

Consider this scenario: your team trains a model on 64 GPUs across 8 nodes. The training completes, the model performs well on benchmarks, and everyone celebrates. But hidden in the logs is a troubling pattern—GPU utilization hovers around 45%, and scaling efficiency from 8 to 64 GPUs is only 52%. You're paying for 64 GPUs but getting the effective compute of 33. Over a two-week training run, that's \$50,000 in wasted compute.

This chapter teaches you to find and fix these hidden inefficiencies. We'll cover the complete benchmarking lifecycle: the metrics that actually matter for distributed systems, profiling tools that reveal where time goes, accuracy evaluation to ensure optimizations don't degrade model quality, network diagnostics for communication bottlenecks, and scaling analysis to understand your system's limits. By the end, you'll have the skills to systematically identify performance bottlenecks, make data-driven optimization decisions, and validate that your distributed systems are running at peak efficiency.


## Why Benchmarking Matters

Benchmarking a single-GPU training loop is straightforward: time the forward pass, backward pass, and optimizer step. The numbers are reproducible, the methodology is simple, and the results are actionable. Distributed systems, however, introduce a fundamentally different challenge.

When you scale from one GPU to eight, you're not simply running eight copies of the same workload. Those eight GPUs must synchronize gradients after every backward pass, coordinate memory allocations to avoid fragmentation, and communicate over networks with varying bandwidths and latencies. Each of these coordination points introduces overhead that doesn't exist in single-device training. A 10% inefficiency in gradient synchronization might seem negligible in a single training step, but over a two-week training run, it compounds to days of wasted GPU time.

The challenge is compounded by the fact that distributed system performance depends on factors that are invisible to simple timing measurements. Network topology affects AllReduce performance. Memory fragmentation affects batch size. Data loading throughput affects GPU utilization. Without systematic benchmarking that accounts for these factors, optimization efforts become guesswork.

__The consequences of inadequate benchmarking manifest in several ways:__

- **Suboptimal technology selection:** A team evaluates vLLM and SGLang using benchmarks with fixed-length prompts. vLLM wins by a small margin. In production, however, request lengths vary dramatically—some prompts are 50 tokens, others are 5,000. SGLang's radix attention provides 40% better throughput for this workload. The benchmark methodology failed to capture production reality.

- **Resource over-provisioning:** Without accurate scaling efficiency measurements, capacity planning relies on conservative estimates. Teams provision twice the hardware they need "to be safe," doubling infrastructure costs without improving performance.

- **Unidentified bottlenecks:** A training job runs for two weeks on 64 GPUs. Post-hoc analysis reveals that GPU utilization averaged only 45%—data loading was the bottleneck, not compute. A \$50 SSD upgrade would have saved \$10,000 in wasted GPU time.

- **Production SLA violations:** An inference service passes load testing with synthetic prompts averaging 100 tokens. In production, real user queries include long documents that trigger different code paths. P99 latency spikes to 500ms—2.5x the promised SLA.

This chapter establishes systematic, reproducible measurement practices that address these challenges. We begin with the metrics that matter for distributed systems, then progress through profiling tools, accuracy evaluation, network diagnostics, and scaling analysis.


## Core Metrics for Distributed Systems

Not all metrics are created equal. Understanding which numbers to track—and which to ignore—is the foundation of effective benchmarking. Let's examine the metrics that actually matter for distributed AI systems.

### Throughput Metrics

Throughput measures how much work your system completes per unit time. For distributed systems, the key insight is that raw throughput must be normalized to be meaningful:

- **Samples per Second (Training):** Number of training samples processed per second *across all GPUs*. This is your primary training efficiency metric.

- **Tokens per Second (Inference):** Number of tokens generated per second. For LLM serving, this determines how many concurrent users you can support. Total TPS per system represents the total output tokens per second throughput, accounting for all requests happening simultaneously. As the number of requests increases, total TPS increases until it reaches a saturation point for all available GPU compute resources, beyond which it may decrease.

![TPS timeline showing concurrent request throughput](img/tps_timeline.png)

- **Requests per Second (Serving):** Number of API requests handled per second at the system level. Unlike tokens/second, this captures the full request lifecycle including queuing and routing.

- **GPU Utilization:** Percentage of time GPUs are actively computing vs waiting. Low utilization often indicates bottlenecks elsewhere (data loading, communication, CPU preprocessing).

### Latency Metrics

Latency measures how long individual operations take. For production systems, percentile latencies matter more than averages. The following diagram illustrates the key LLM inference latency metrics and their relationships:

![Overview of LLM inference performance metrics](img/inference_metrics_overview.png)

- **P50 (Median):** The latency experienced by a typical request. Useful for understanding normal behavior.

- **P95:** The latency experienced by 95% of requests. This is often the SLA target for production systems.

- **P99:** The worst-case latency for 99% of requests. Critical for user experience—even 1% of users experiencing 10x latency creates significant frustration.

- **End-to-End Request Latency (e2e_latency):** The total time from submitting a query to receiving the complete response. This metric captures the full request lifecycle including tokenization, prefill, all token generation, and de-tokenization.

![End-to-end request latency pipeline](img/e2e_latency_pipeline.png)

- **Time to First Token (TTFT):** For streaming inference, how long until the first token appears. Users perceive this as "response time." TTFT includes tokenization, the prefill phase (processing the entire input prompt), generating the first token, and de-tokenization.

![TTFT pipeline from input to first output token](img/ttft_pipeline.png)

- **Time per Output Token (TPOT):** Average time between consecutive output tokens. Determines the perceived "typing speed" of streaming responses. Also known as Inter-token Latency (ITL), this metric is calculated as:

$$
\text{ITL} = \frac{\text{e2e\_latency} - \text{TTFT}}{\text{Total\_output\_tokens} - 1}
$$

![ITL pipeline showing output token generation](img/itl_pipeline.png)

Understanding the distribution of latencies across requests is essential for setting realistic SLAs. Figure \ref{fig:latency-distribution} shows a typical latency distribution with both a histogram (showing frequency) and a cumulative distribution function (CDF). The histogram reveals the shape of the distribution—often long-tailed in production systems—while the CDF makes it easy to read off percentile values directly: the P50 is where the CDF crosses 50%, P95 at 95%, and P99 at 99%.

![Latency distribution histogram and CDF with percentile markers](img/latency_distribution.png){#fig:latency-distribution}

>NOTES: **Why P99 Matters More Than Average**

Average latency can be misleading. Consider two systems: System A has 100ms average latency with all requests between 90-110ms. System B has 80ms average latency, but 5% of requests take 500ms. System B has better average latency but worse user experience—those 5% of slow requests create frustrated users and potential timeouts. Always report P95 and P99 alongside averages.

>NOTEE

### Efficiency Metrics

Raw performance numbers tell only part of the story. A system that processes 10,000 tokens per second sounds impressive—until you learn it requires 64 GPUs to achieve that throughput. Efficiency metrics bridge this gap by measuring how well your system converts resources into useful work.

__Scaling Efficiency__ answers a fundamental question: when you double your hardware, do you double your performance? In an ideal world, adding 8 GPUs would yield 8x the throughput of a single GPU. Reality is less generous. Communication overhead, synchronization barriers, and load imbalances all conspire to reduce this ratio. If those 8 GPUs deliver only 6.5x throughput, your scaling efficiency is 81.25%—meaning nearly 20% of your additional hardware investment is lost to coordination costs. Understanding where this efficiency loss occurs is the first step toward recovering it.

__Memory Efficiency__ measures how effectively you utilize the most precious resource in modern AI: GPU memory. High utilization sounds desirable, but the picture is more nuanced. Memory fragmentation can leave you with 20GB "free" yet unable to allocate a 10GB tensor because that free space is scattered across non-contiguous regions. Effective memory efficiency considers not just utilization percentage, but whether that memory organization supports your target batch sizes and sequence lengths.

__Communication Overhead__ quantifies the hidden tax of distributed computing. Every gradient synchronization, every tensor transfer between devices, every collective operation consumes time that could otherwise be spent on computation. In well-optimized systems, communication overlaps with computation, hiding much of this cost. In poorly configured systems, GPUs sit idle waiting for data transfers to complete. The ratio of communication time to total time reveals how much room remains for optimization.

__Cost per Token/Request__ translates technical metrics into business reality. A system with superior throughput might still be economically inferior if it requires expensive hardware or consumes excessive power. This metric combines performance measurements with actual infrastructure costs—cloud instance pricing, electricity consumption, cooling requirements—to answer the question that ultimately matters: how much does each unit of useful work cost?

### The Benchmarking Methodology

Rigorous methodology separates meaningful benchmarks from noise. Here's the four-phase approach that ensures reproducible, accurate results:

**Phase 1: Setup**
- Fix random seeds for reproducibility
- Document hardware configuration (GPU model, memory, interconnect)
- Version control benchmark scripts
- Record system state (driver versions, CUDA version, framework versions)

**Phase 2: Warmup**

The warmup phase is critical because CUDA operations are lazily compiled—the first execution triggers JIT compilation, memory allocation, and cache population. Without warmup, your measurements include these one-time costs, making results unreliable. See `code/benchmark_warmup.py` for a complete implementation with proper synchronization and statistics reporting.

**Phase 3: Measurement**
- Run multiple independent iterations (minimum 100)
- Ensure CUDA synchronization before timing
- Measure over sufficient duration to capture variance

**Phase 4: Analysis**
- Report mean, standard deviation, and percentiles
- Use statistical significance tests when comparing systems
- Document any anomalies or outliers

![Four-phase benchmarking workflow with pitfalls and best practices](img/benchmarking_methodology.png)

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

![Training iteration time breakdown by phase and GPU count](img/training_breakdown.png)

The figure above illustrates a common pattern: as you scale from 1 GPU to 16 GPUs, communication overhead grows from 0% to over 50% of iteration time. This is why scaling efficiency decreases—you're spending more time synchronizing and less time computing. Understanding this breakdown is the first step toward optimization.

### PyTorch Profiler

PyTorch's built-in profiler is your first tool for understanding training performance. It captures CPU and CUDA operations, memory allocations, and can export traces for visualization.

**Basic Usage:**

The PyTorch profiler captures both CPU and CUDA operations, giving you a complete picture of where time is spent. The key is labeling your code regions with `record_function` so you can identify bottlenecks by phase. Key configuration options include:

- **`activities`:** Specify `CPU` and `CUDA` to capture both host and device operations
- **`record_shapes`:** Records tensor shapes, useful for understanding memory patterns
- **`profile_memory`:** Tracks memory allocations and deallocations
- **`with_stack`:** Captures Python call stacks for deeper debugging

The output table shows operations sorted by total CUDA time. Look for operations consuming disproportionate time—these are your optimization targets. The exported `trace.json` can be viewed in `chrome://tracing` for detailed timeline analysis.

**Advanced Profiling with Schedule:**

For multi-iteration analysis, use a profiling schedule that handles warmup automatically. The schedule parameters control the profiling lifecycle: `wait` skips cold start iterations, `warmup` runs iterations without recording, `active` profiles the specified number of iterations, and `repeat` cycles through this pattern multiple times. The `tensorboard_trace_handler` automatically saves traces for visualization with TensorBoard.

See `code/pytorch_profiler.py` for complete implementations of both basic and scheduled profiling.

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

For systematic benchmarking across configurations, a custom benchmark class provides more control than ad-hoc profiling. This class measures each training phase separately—data loading, forward pass, backward pass, and optimizer step—enabling you to pinpoint exactly where time is spent.

**Why separate phase timing matters:** If your backward pass takes 3x longer than your forward pass, you might have inefficient gradient computation or memory fragmentation. If data loading dominates, you need more DataLoader workers or faster storage. Without phase-level breakdown, you're optimizing blind.

**Interpreting the results:** The returned `stats` dictionary contains mean, std, and percentiles for each phase. If `data_loading` exceeds 10% of total time, increase DataLoader workers. If `backward` is more than 2x `forward`, check for gradient checkpointing opportunities or memory fragmentation. High variance (large gap between mean and P99) indicates system instability—investigate thermal throttling or competing processes.

See `code/training_benchmark.py` for the complete `TrainingBenchmark` class implementation.

### Measuring Scaling Efficiency

Scaling efficiency quantifies how well your system utilizes additional resources. Perfect linear scaling (100% efficiency) means doubling GPUs doubles throughput—but communication overhead makes this impossible in practice. Measuring scaling efficiency helps you decide when adding more GPUs is cost-effective versus when you've hit diminishing returns.

![Scaling efficiency: ideal vs actual throughput and efficiency percentages](img/scaling_efficiency.png)

**Interpreting Scaling Efficiency:**

- **>90%:** Excellent scaling—your system is well-optimized
- **70-90%:** Good scaling—typical for well-tuned distributed training
- **50-70%:** Moderate scaling—communication overhead is significant, investigate network
- **<50%:** Poor scaling—major bottleneck present, likely communication or data loading

Understanding your scaling efficiency helps with capacity planning. If you have 81% efficiency at 8 GPUs, you can predict that 16 GPUs will provide roughly 13x throughput (not 16x), helping you make informed hardware decisions.

See `code/scaling_efficiency.py` for functions to calculate and benchmark scaling efficiency.


## Inference Benchmarking

Inference benchmarking presents different challenges than training. Request patterns are variable, caching effects matter, and tail latency requirements are strict. A training job that's 10% slower is annoying; an inference endpoint that violates P99 SLA loses customers.

>NOTES: **Performance vs Accuracy Benchmarking**

This section covers **performance benchmarking** using tools like [genai-bench](https://github.com/sgl-project/genai-bench), which measures engineering metrics (throughput, latency, scaling). This is distinct from **accuracy benchmarking** tools like [GenAI-Bench for text-to-visual evaluation](https://linzhiqiu.github.io/papers/genai_bench/) that measure model output quality. We cover accuracy benchmarking in the next section.

>NOTEE

### genai-bench Overview

genai-bench is a CLI-based benchmarking tool designed for realistic inference workload testing. Unlike simple load generators that send identical requests, genai-bench supports configurable traffic patterns that mirror production workloads—variable prompt lengths, different concurrency levels, and realistic request distributions.

**Why use genai-bench over custom scripts?** Production inference traffic is highly variable. Users send prompts ranging from 10 tokens to 10,000 tokens. Load varies from 1 concurrent request to 1,000. Simple benchmarks with fixed-length prompts miss critical performance characteristics like how your system handles long-context requests under load, or how batching efficiency changes with request diversity.

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

**Understanding Traffic Scenarios:**

Traffic scenarios define the distribution of input and output token lengths:

- `D(100,100)`: Deterministic—all requests have exactly 100 input and 100 output tokens
- `D(512,512)`: Longer context scenario
- `I(input_tokens, output_tokens)`: Image-text input with fixed tokens
- `E(input_tokens)`: Embedding requests

For realistic benchmarking, test multiple scenarios that reflect your production traffic. A test matrix covering different concurrency levels and context lengths reveals critical performance characteristics:

- **Low concurrency + short context:** Baseline latency without batching effects
- **High concurrency + short context:** How well the system batches requests
- **Any concurrency + long context:** Memory pressure and KV cache behavior

Look for non-linear latency increases as context length grows—this indicates memory bandwidth bottlenecks.

See `code/genai_bench_example.py` for complete examples of running genai-bench programmatically, including single benchmarks, test matrices, and result analysis.

### Custom Inference Benchmark

For scenarios where genai-bench doesn't fit—custom models, non-standard APIs, or specialized metrics—a custom benchmark class provides fine-grained control over the measurement process. This is useful for CI/CD integration or specialized metrics.

### Measuring Cold vs Warm Performance

Cold start latency—the time for the first request after model loading—can be 10-100x slower than warm requests. This matters for autoscaling: if cold starts take 30 seconds but warm requests take 100ms, aggressive scale-down policies will cause user-facing latency spikes when traffic returns. A 16x cold/warm ratio is typical. Use this data to configure autoscaler minimum instances—keep enough warm instances to handle baseline traffic without cold starts.

### Measuring Reasoning & Multi-step Models

Reasoning models—chain-of-thought, tool-augmented LLMs, multi-step agents—require different benchmarking approaches. You need to measure both per-step latency and end-to-end session latency, separating local generation time from external calls.

**Key measurement points:**

- **Per-step latency:** Measure TTFT/TPOT for each reasoning step to identify slow steps
- **End-to-end session latency:** Total time for the complete reasoning session
- **External-call breakdown:** Time waiting for retrievals, tool calls, or APIs vs local generation
- **Cache effects:** Cold vs warm runs when KV cache or retrieval caches are populated

In agentic workflows, often 60%+ of latency comes from external calls, not model inference. Optimizing the model won't help—you need to optimize or parallelize the tool calls.

See `code/inference_benchmark.py` for complete implementations of `InferenceBenchmark`, cold start measurement, and reasoning session benchmarking.

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

**GLUE/SuperGLUE:** General language understanding tasks. The Hugging Face `evaluate` library provides easy access to standard benchmarks like SST-2 (sentiment classification).

**MMLU:** Knowledge across 57 tasks (STEM, humanities, social sciences)

**HumanEval:** Code generation with execution-based evaluation

**HELM:** Holistic evaluation including accuracy, robustness, fairness, and efficiency

### Evaluating Distributed Training Accuracy

A critical check: does your distributed training produce the same model quality as single-GPU training? Bugs in gradient synchronization, different batch size effects, or numerical precision issues can cause distributed training to converge to worse solutions.

### Evaluating Quantization Impact

Quantization trades precision for speed. Before deploying a quantized model, measure the accuracy cost to ensure the speedup is worth the quality tradeoff.

>NOTES: **Statistical Significance in Accuracy Comparisons**

Small accuracy differences may be noise, not signal. Use statistical tests to determine if differences are meaningful. A 0.5% accuracy drop with p=0.3 is probably noise. A 0.5% drop with p=0.001 is real.

>NOTEE

See `code/accuracy_benchmark.py` for functions to compare centralized vs distributed training accuracy, evaluate quantization impact, and test statistical significance.


## Network and Communication Profiling

In distributed systems, the network is often the bottleneck. A single slow link between nodes can limit the entire system's performance. Understanding communication patterns and diagnosing network issues is essential for scaling efficiently.

![Ring AllReduce pattern and multi-node bandwidth hierarchy](img/network_topology.png)

The bandwidth hierarchy matters enormously: NVLink between GPUs on the same node provides ~600 GB/s, InfiniBand between nodes provides ~200 GB/s, and Ethernet provides only ~12.5 GB/s (100 Gbps). Communication patterns that cross these boundaries pay significant latency penalties.

### Network Monitoring Tools

Before diving into NCCL-specific profiling, use standard network tools to establish baseline connectivity and bandwidth between nodes.

**iftop:** Real-time network traffic monitoring

```bash
sudo iftop -i eth0
sudo iftop -i eth0 -f "host 192.168.1.10"  # Filter by host
```

This shows live bandwidth usage per connection. Look for unexpected traffic patterns or connections that should be idle but aren't.

**nload:** Bandwidth monitoring

```bash
nload eth0
```

Displays incoming/outgoing bandwidth graphs. Useful for seeing if you're saturating your network link during training.

**iperf3:** Bandwidth testing between nodes

```bash
# Server side (run on node 1)
iperf3 -s

# Client side (run on node 2)
iperf3 -c server_ip -t 60 -i 1
```

This measures raw TCP bandwidth between nodes. If iperf3 shows 100 Gbps but your training only achieves 20 Gbps effective bandwidth, the bottleneck is in your communication pattern, not the network hardware.

### NCCL Communication Tests

AllReduce is the dominant communication operation in distributed training—it synchronizes gradients across all GPUs. Testing AllReduce bandwidth separately from training helps isolate network issues from compute issues.

Small messages have high overhead (latency-bound), while large messages approach peak bandwidth (bandwidth-bound). If your large-message bandwidth is significantly below hardware specs, check for topology issues or NCCL configuration problems.

### Communication Overhead Analysis

Measure how much time is spent communicating vs computing. This helps determine if your scaling bottleneck is network-related. In DDP, communication happens during the backward pass, so use the profiler to separate these components.

See `code/network_diagnostics.py` for AllReduce bandwidth tests and communication overhead analysis functions.

With network profiling complete, let's examine how to analyze and improve scaling efficiency.


## Scaling Efficiency Analysis

Scaling efficiency measures how well your system utilizes additional resources. Understanding scaling behavior helps with capacity planning, cost optimization, and identifying bottlenecks.

### Amdahl's Law

Amdahl's Law provides the theoretical limit on speedup from parallelization. Even with infinite processors, speedup is bounded by the serial (non-parallelizable) portion of your workload. With 10% serial work, even infinite GPUs can only provide 10x speedup. This is why identifying and reducing serial bottlenecks is critical—reducing serial fraction from 10% to 5% has more impact than doubling GPU count.

### Identifying Scaling Bottlenecks

Once you know your scaling efficiency is poor, the next step is identifying *why*. Common patterns include:

- **"Data loading not scaling well":** Your DataLoader can't keep up with multiple GPUs. Increase `num_workers` or use faster storage.
- **"High communication overhead":** Network is the bottleneck. Consider gradient compression, larger batch sizes, or better interconnect.
- **"Low GPU utilization":** GPUs are waiting for something—usually data or synchronization.

### Optimization Strategies

Once you've identified the bottleneck, apply the appropriate fix:

**1. Overlap Communication and Computation:** DDP uses gradient bucketing to overlap backward computation with gradient synchronization. Smaller buckets start communication earlier but have more overhead. Larger buckets have less overhead but delay communication. Start with 25MB and tune based on profiling.

**2. Optimize Data Loading:** Set `num_workers` to 2-4x your CPU cores per GPU. `pin_memory=True` enables faster CPU→GPU transfers. `prefetch_factor` controls how many batches each worker prefetches.

**3. Gradient Accumulation:** If communication overhead is high, reduce synchronization frequency by accumulating gradients over multiple micro-batches. This effectively increases batch size without increasing memory usage.

See `code/scaling_bottlenecks.py` for Amdahl's Law calculations, bottleneck analysis functions, and optimization strategy implementations.


\fancydividerwithicon[center]{python.png}

## Hands-On Examples

The following examples provide complete, runnable code for common benchmarking scenarios. Each example is self-contained and can be adapted to your specific use case. All code is available in the `code/` directory.

### Example 1: genai-bench Inference Benchmarking

Programmatically run genai-bench and analyze results. Use this when you need to integrate benchmarking into CI/CD pipelines or automate performance regression testing.

**File:** `code/genai_bench_example.py`

### Example 2: Scaling Efficiency Measurement

Measure how well your training scales across GPU counts. Run with different `--nproc_per_node` values to build a scaling curve and identify where efficiency drops off.

**File:** `code/scaling_efficiency.py`

Run with: `torchrun --nproc_per_node=N code/scaling_efficiency.py`

### Example 3: Network Diagnostic Tools

Test raw AllReduce bandwidth between GPUs. Use it to verify your network is performing as expected before debugging higher-level training issues.

**File:** `code/network_diagnostics.py`

Run with: `torchrun --nproc_per_node=2 code/network_diagnostics.py`

If bandwidth is significantly lower than expected, check: (1) NCCL environment variables, (2) GPU topology with `nvidia-smi topo -m`, (3) whether GPUs are on the same NUMA node.

### Additional Code Files

- `code/benchmark_warmup.py` - Proper warmup and timing utilities
- `code/pytorch_profiler.py` - PyTorch profiler examples
- `code/training_benchmark.py` - Training phase breakdown benchmark
- `code/inference_benchmark.py` - Custom inference and reasoning benchmarks
- `code/accuracy_benchmark.py` - Accuracy comparison utilities
- `code/scaling_bottlenecks.py` - Bottleneck analysis and optimization strategies


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


