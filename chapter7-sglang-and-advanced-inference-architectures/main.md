# Chapter 7: SGLang and Advanced Inference Architectures

## Overview

This chapter explains how SGLang's lightweight runtime, operator fusion, and scheduling mechanisms enable high-performance distributed inference. Readers learn about DeepSeek-style chunked prefill strategies, genai-bench integration for benchmarking, router-based distributed inference architectures, and hybrid CPU/GPU serving patterns. By the end of this chapter, readers will be able to deploy SGLang across multiple nodes, optimize operator fusion, and build router-based inference systems for high-QPS production workloads.

**Chapter Length:** 26 pages

## SGLang vs. vLLM: Key Differences

Understanding the differences between SGLang and vLLM helps in choosing the right inference engine for your use case. Both are high-performance inference systems, but they optimize for different aspects of distributed inference.

### Architecture Philosophy

**vLLM (Chapter 6):**
- **Scheduler-Executor-Worker Pattern**: Multi-threaded architecture with separate scheduler, executor, and worker components
- **PagedAttention**: Core innovation for efficient KV cache memory management using virtual memory paging concepts
- **Continuous Batching**: Dynamically groups requests into batches with padding elimination
- **Focus**: Memory efficiency and high throughput through sophisticated memory management

**SGLang:**
- **Graph-Based IR Runtime**: Compact intermediate representation with operator fusion at compile time
- **Lightweight Scheduler**: Single-threaded event loop with minimal scheduling overhead
- **Operator Fusion**: Aggressively fuses small operators into single kernels to reduce launch overhead
- **Focus**: Low latency and minimal scheduling overhead through runtime optimization

### Memory Management

**vLLM:**
- **PagedAttention**: Virtual memory-style paging for KV cache blocks
- **Block-based Allocation**: Fixed-size blocks allocated on-demand
- **Memory Efficiency**: Near-100% memory utilization, 2-4x more concurrent requests
- **Handles**: Variable-length sequences without padding overhead

**SGLang:**
- **Memory-Paged Primitives**: Efficient cache management with streaming attention
- **Chunked Prefill**: DeepSeek-style pattern for long contexts, streams prompts in chunks
- **KV Cache Offloading**: Can offload older KV pages to CPU/NVMe
- **Handles**: Long-context generation with reduced peak memory

### Scheduling and Batching

**vLLM:**
- **Continuous Batching**: Dynamic batching where batch composition changes every step
- **Multi-threaded**: Separate threads for scheduling, execution, and worker management
- **High Concurrency**: Built for high-churn workloads with frequent request arrivals
- **Throughput-Optimized**: Maximizes tokens per second

**SGLang:**
- **Single-threaded Event Loop**: Minimal overhead scheduler that batches small work items
- **Operator-Level Batching**: Fuses operators to reduce kernel launches
- **Low-Latency Optimized**: Minimizes time-to-first-token through reduced overhead
- **QPS-Optimized**: Maximizes queries per second for high-QPS workloads

### Distributed Inference

**vLLM:**
- **Tensor Parallelism (TP)**: Horizontal sharding of model weights across GPUs
- **Data Parallelism (DP)**: Replicates model across multiple GPUs/nodes
- **Pipeline Parallelism (PP)**: Vertical sharding across nodes for very large models
- **Ray Backend**: Uses Ray for multi-node coordination
- **Internal/External Load Balancing**: Supports both self-contained and external routing

**SGLang:**
- **Router-Based Architecture**: Central router with session affinity and cache-aware routing
- **Prefill/Decode Disaggregation**: Specialized workers for prefill vs decode phases
- **Multi-Node Coordination**: Central scheduler or decentralized broker
- **Session Affinity**: Routes requests to workers with hot KV cache
- **Hybrid CPU/GPU**: Can offload lightweight operators to CPU

### Performance Characteristics

**vLLM:**
- **Strengths:**
  - Excellent memory efficiency (PagedAttention)
  - High throughput for large batches
  - Strong support for very large models (TP/PP/DP)
  - Production-proven at scale
- **Best For:**
  - High-throughput serving
  - Large models requiring multi-GPU/node distribution
  - Memory-constrained environments
  - Variable-length sequence workloads

**SGLang:**
- **Strengths:**
  - Lower latency (operator fusion, lightweight scheduler)
  - Higher QPS for small-to-medium batches
  - Better for high-churn, low-latency workloads
  - Flexible CPU/GPU hybrid serving
- **Best For:**
  - Low-latency inference (chat applications)
  - High-QPS production workloads
  - Long-context generation with chunked prefill
  - Cost-optimized hybrid CPU/GPU deployments

### When to Choose Which

**Choose vLLM when:**
- You need maximum memory efficiency (PagedAttention)
- Throughput is the primary concern
- Serving very large models requiring TP/PP/DP
- You have variable-length sequences and want to eliminate padding
- You need proven production stability at scale

**Choose SGLang when:**
- Latency (especially TTFT) is critical
- You need high QPS for interactive applications
- You want to optimize for small-to-medium batch sizes
- You need router-based distributed inference with session affinity
- You want to explore hybrid CPU/GPU serving for cost optimization

### Complementary Use Cases

Both systems can coexist in the same infrastructure:
- **vLLM**: For batch processing, large model serving, high-throughput workloads
- **SGLang**: For interactive chat, low-latency APIs, high-QPS endpoints

## 1. SGLang Internals and Operator Fusion

SGLang is built around a compact graph-based Intermediate Representation (IR) and a runtime that emphasizes operator fusion and minimal scheduling overhead. Understanding these internals is crucial for optimizing inference performance.

### Runtime Architecture

**Key Components:**

1. **Graph-Based IR**
   - Compact intermediate representation
   - Compiles operator sequences into fused kernels
   - Reduces kernel launch latency significantly

2. **Lightweight Scheduler**
   - Single-threaded event loop
   - Batches small work items efficiently
   - Dispatches fused kernels with minimal overhead

3. **Memory-Friendly Operators**
   - Quantized kernels for reduced memory footprint
   - Streaming attention operators for long contexts
   - Memory-paged primitives for efficient cache management

### Operator Fusion

**Why Operator Fusion Matters:**

- **Reduces Kernel Launches:** Combining small operations into single kernels eliminates launch overhead
- **Lowers Memory Traffic:** Intermediate buffers are combined, reducing read/write operations
- **Enables Custom Optimizations:** Fused kernels (e.g., layernorm + linear + activation) outperform generic compositions

**Fusion Strategy:**

```python
# Example: Fused layernorm + linear + activation (runnable example)
import torch
import torch.nn as nn

class FusedLayerNormLinearActivation(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()

    def forward(self, x):
        # Fused operation reduces memory traffic
        x = self.layernorm(x)
        x = self.linear(x)
        x = self.activation(x)
        return x
```

**Profiling Operator Fusion:**

1. Use microbenchmarks to measure per-operator latency
2. Identify hot small operators (embeddings, layernorms, pointwise ops)
3. Implement fusion for top-3 hot operators
4. Re-measure and validate improvements

### Practical Exercise: Operator Fusion Profiling

**Steps:**

1. **Profile Baseline:**
```python
import torch
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    output = model(input)

# Analyze operator-level timing
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

2. **Identify Fusion Candidates:**
   - Look for sequences of small ops with high cumulative time
   - Focus on ops with high memory traffic

3. **Implement Fused Kernel:**
   - Write CUDA kernel or use Triton
   - Register in SGLang's kernel registry

4. **Validate Improvement:**
   - Measure latency reduction
   - Verify numerical correctness
   - Check memory usage reduction

## 2. Multi-Node SGLang Inference

Multi-node SGLang deployments coordinate execution across workers via a central scheduler or decentralized broker. This enables scaling beyond single-node capacity.

### Deployment Architecture

**Components:**

1. **Central Scheduler**
   - Coordinates request distribution
   - Manages worker health and capacity
   - Implements load balancing policies

2. **Model Sharding**
   - Partition model across nodes
   - Route requests by model shard
   - Co-locate hot shards to reduce cross-node communication

3. **KV Cache Management**
   - Distribute KV cache across nodes
   - Implement cache coherence protocols
   - Minimize cross-node KV transfers

### Multi-Node Setup

**Configuration:**

```python
# Multi-node SGLang configuration
config = {
    "num_nodes": 4,
    "nodes_per_shard": 2,
    "scheduler_url": "http://scheduler:8000",
    "worker_urls": [
        "http://worker1:8000",
        "http://worker2:8000",
        "http://worker3:8000",
        "http://worker4:8000"
    ],
    "routing_policy": "cache_aware"
}
```

**Deployment Steps:**

1. **Launch Workers:**
   ```bash
   # Node 1
   python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct \
       --port 8000 --tensor-parallel-size 2
   
   # Node 2
   python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct \
       --port 8000 --tensor-parallel-size 2
   ```

2. **Configure Router:**
   ```bash
   python -m sglang_router.launch_router \
       --worker-urls http://worker1:8000 http://worker2:8000 \
       --policy cache_aware
   ```

3. **Test Multi-Node Routing:**
   ```python
   import requests
   
   response = requests.post(
       "http://router:8000/v1/chat/completions",
       json={
           "model": "llama-3.1-8b",
           "messages": [{"role": "user", "content": "Hello!"}]
       }
   )
   ```

### Performance Considerations

- **Network Latency:** Minimize cross-node communication
- **Cache Locality:** Route requests to nodes with hot KV cache
- **Load Balancing:** Distribute load evenly across nodes
- **Fault Tolerance:** Handle node failures gracefully

## 3. Chunked Prefill and KV Cache Strategies

Long-context generation can exhaust KV cache memory on GPU. Chunked prefill (a DeepSeek-style pattern) streams the initial prompt in chunks so the full attention state need not be materialized at once.

### Chunked Prefill Pattern

**High-Level Approach:**

1. Split initial context into chunks (e.g., 512 tokens)
2. For each chunk: run forward to build partial KV entries
3. Optionally offload older KVs to CPU/NVMe
4. Keep sliding window of hot KVs on GPU for decoding

**Implementation:**

```python
def chunked_prefill(model, tokenizer, prompt, chunk_size=512):
    """Chunked prefill for long contexts"""
    chunks = [prompt[i:i+chunk_size] 
              for i in range(0, len(prompt), chunk_size)]
    
    kv_cache = []
    for chunk in chunks:
        tokens = tokenizer(chunk).to(device)
        # Forward with streaming/paged attention
        kv_chunk = model.forward_prefill(tokens)
        kv_cache.append(kv_chunk)
        
        # Optionally offload older KV pages to CPU/NVMe
        if len(kv_cache) > max_gpu_chunks:
            offload_to_cpu(kv_cache[0])
            kv_cache = kv_cache[1:]
    
    # Now decode with hot KV cache
    return kv_cache
```

### Tradeoffs and Tuning

**Chunk Size:**
- **Larger chunks:** Reduce total kernel overhead but increase peak memory
- **Smaller chunks:** Lower memory but more kernel launches

**Eviction Policy:**
- **LRU:** Evict least recently used KV pages
- **Size-based:** Evict largest KV pages first
- **Hybrid:** Combine both strategies

**Prefill Parallelism:**
- Overlap chunk prefill with IO when offloading to NVMe
- Use async operations for non-blocking transfers

### Memory Management

**KV Cache Offloading:**

```python
class KVCacheManager:
    def __init__(self, gpu_capacity, cpu_capacity):
        self.gpu_cache = {}
        self.cpu_cache = {}
        self.gpu_capacity = gpu_capacity
        self.cpu_capacity = cpu_capacity
    
    def store(self, session_id, kv_data):
        if len(self.gpu_cache) >= self.gpu_capacity:
            # Evict to CPU
            oldest = min(self.gpu_cache.keys(), 
                        key=lambda k: self.gpu_cache[k].last_access)
            self.cpu_cache[oldest] = self.gpu_cache.pop(oldest)
        
        self.gpu_cache[session_id] = kv_data
    
    def retrieve(self, session_id):
        if session_id in self.gpu_cache:
            return self.gpu_cache[session_id]
        elif session_id in self.cpu_cache:
            # Promote to GPU
            kv = self.cpu_cache.pop(session_id)
            self.store(session_id, kv)
            return kv
        return None
```

## 4. Benchmarking with genai-bench

Integrate SGLang with genai-bench to generate realistic, reproducible workloads. Benchmark scenarios should include cold-start, warm cache, and long-context streaming.

### genai-bench Setup

**Installation:**

```bash
pip install genai-bench
```

**Basic Benchmark:**

```bash
genai-bench benchmark \
    --api-backend sglang \
    --api-base "http://localhost:8000" \
    --api-key "your-api-key" \
    --api-model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --model-tokenizer "/path/to/tokenizer" \
    --task text-to-text \
    --max-time-per-run 15 \
    --max-requests-per-run 1000 \
    --num-concurrency 100 \
    --traffic-scenario "D(100,100)" \
    --server-engine "SGLang" \
    --server-gpu-type "H100" \
    --server-version "v0.5.5"
```

### Benchmark Scenarios

**1. Cold-Start Benchmark:**
- Restart server between runs
- Measure first-request latency
- Capture initialization overhead

**2. Warm-Cache Benchmark:**
- Pre-warm KV cache
- Measure steady-state performance
- Compare with cold-start

**3. Long-Context Streaming:**
- Use traffic scenarios with large input tokens
- Measure chunked prefill performance
- Validate memory usage

### Analyzing Results

**Generate Excel Report:**

```bash
genai-bench excel \
    --experiment-folder ./experiments/your_experiment \
    --excel-name sglang_benchmark \
    --metric-percentile mean
```

**Generate Plots:**

```bash
genai-bench plot \
    --experiments-folder ./experiments \
    --group-key traffic_scenario \
    --preset 2x4_default
```

**Key Metrics:**

- **TTFT (Time to First Token):** Latency until first token generation
- **E2E Latency:** End-to-end request latency
- **TPOT (Time Per Output Token):** Average time per generated token
- **Throughput:** Tokens per second
- **GPU Utilization:** GPU usage percentage

### Kernel-Level Profiling

**Using Nsight Systems:**

```bash
nsys profile --trace=cuda,nvtx \
    python -m sglang.launch_server --model your-model
```

**Using torch.profiler:**

```python
import torch
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    output = model.generate(input)

# Export to Chrome trace format
prof.export_chrome_trace("trace.json")
```

## 5. Router-Based Distributed Inference

Routers sit in front of model runners and implement session affinity, dynamic routing rules, and prefill routing to optimize distributed inference.

### Router Architecture

**Key Features:**

1. **Session Affinity / Stickiness**
   - Keep request sessions routed to runner with hot KV cache
   - Minimize KV cache transfers between nodes
   - Improve cache hit rates

2. **Dynamic Routing Rules**
   - Route by model size, user SLA, or experimental flags
   - Implement priority queues for different request types
   - Support A/B testing and canary deployments

3. **Prefill Routing**
   - Route heavy prefill work to specialized prefill workers
   - Avoid stalling decode workers with long prefill requests
   - Implement PD (Prefill/Decode) disaggregation

### Router Implementation

**Basic Router:**

```python
class InferenceRouter:
    def __init__(self, workers):
        self.workers = workers
        self.session_map = {}  # session_id -> worker_id
        self.worker_load = {w: 0 for w in workers}
    
    def route_request(self, request):
        session_id = request.get("session_id")
        
        # Check for existing session
        if session_id and session_id in self.session_map:
            worker_id = self.session_map[session_id]
            if self.workers[worker_id].is_available():
                return worker_id
        
        # Route to best available worker
        worker_id = self.select_best_worker(request)
        if session_id:
            self.session_map[session_id] = worker_id
        
        return worker_id
    
    def select_best_worker(self, request):
        # Select worker with lowest load and available capacity
        available = [w for w in self.workers if w.is_available()]
        if not available:
            raise Exception("No available workers")
        
        return min(available, key=lambda w: self.worker_load[w.id])
```

### Advanced Router Features

**1. Adaptive Batching:**

```python
class AdaptiveBatchingRouter:
    def __init__(self, min_batch_size=1, max_batch_size=32):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_latency = 0.0
        self.target_latency = 0.1  # 100ms
    
    def adjust_batch_size(self):
        if self.current_latency < self.target_latency:
            # Increase batch size
            return min(self.max_batch_size, self.current_batch_size * 2)
        else:
            # Decrease batch size
            return max(self.min_batch_size, self.current_batch_size // 2)
```

**2. Pre-warming:**

```python
def prewarm_workers(expected_traffic):
    """Pre-warm workers for expected traffic bursts"""
    for worker in workers:
        # Pre-load models
        worker.load_model()
        
        # Pre-allocate KV cache
        worker.allocate_kv_cache(capacity=expected_traffic)
        
        # Warm up with dummy requests
        for _ in range(10):
            worker.process_dummy_request()
```

**3. Multi-Model Routing:**

```python
class MultiModelRouter:
    def route(self, request):
        model_name = request.get("model")
        
        if model_name == "reranker":
            return self.reranker_workers[0]
        elif model_name == "tool-caller":
            return self.tool_workers[0]
        else:
            return self.main_model_workers[0]
```

### SGLang Router Configuration

**Using sgl-router:**

```bash
# Rust binary with PD disaggregation
./target/release/sglang-router \
    --pd-disaggregation \
    --prefill http://prefill1:30001 \
    --decode http://decode1:30011 \
    --policy cache_aware \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
```

**Python launcher:**

```bash
python3 -m sglang_router.launch_router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy cache_aware
```

### PD Quick Demo (3 commands)

The following minimal commands start a prefill worker, a decode worker, and a router locally so you can observe PD behavior. Adjust model names and ports as needed.

```bash
# 1) Start a prefill worker (HTTP) on port 30001
python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --port 30001 &

# 2) Start a decode worker (HTTP) on port 30011
python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --port 30011 &

# 3) Start the router in PD mode pointing to the workers (router listens on 8080)
python3 -m sglang_router.launch_router --pd-disaggregation \
    --prefill http://localhost:30001 --decode http://localhost:30011 --port 8080
```

Validate with a single request (curl):

```bash
curl -X POST "http://localhost:8080/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "input": "Hello world"}'
```

Note: run the workers in separate terminals or background them; terminate with `kill` when done.

## 6. Hybrid CPU/GPU Serving Strategies

To reduce cost, move low-priority or lightweight operators to CPU (or CPU with INT8) and reserve GPU for heavy attention matmuls.

### Operator Placement Strategy

**Profile Operators:**

```python
import time
import torch
import torch.nn as nn

def profile_operators(model, input):
    """Profile each operator to determine CPU vs GPU placement"""
    operator_times = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            start = time.time()
            _ = module(input)
            torch.cuda.synchronize()
            operator_times[name] = time.time() - start

    # Sort by execution time
    sorted_ops = sorted(operator_times.items(), key=lambda x: x[1], reverse=True)

    # Place top N on GPU, rest on CPU
    gpu_ops = [op[0] for op in sorted_ops[:10]]
    cpu_ops = [op[0] for op in sorted_ops[10:]]

    return gpu_ops, cpu_ops
```

### CPU/GPU Split

**Strategies:**

1. **Operator-Level Placement:**
   - Profile operators individually
   - Assign CPU vs GPU per-operator
   - Use quantized CPU kernels (INT8/FP16) where accuracy drop is acceptable

2. **Edge-of-GPU Pattern:**
   - Keep attention matmuls on GPU
   - Offload tokenization, logits post-processing, and simple fusions to CPU
   - Reduce GPU memory pressure

3. **Request-Level Routing:**
   - Route low-priority requests to CPU-only workers
   - Reserve GPU for high-priority, latency-sensitive requests
   - Implement quality-of-service (QoS) tiers

### Implementation Example

```python
class HybridCPUGPUServing:
    def __init__(self):
        self.gpu_model = load_model_on_gpu()
        self.cpu_model = load_model_on_cpu(quantized=True)
    
    def serve(self, request, priority="high"):
        if priority == "high":
            # Use GPU for low-latency
            return self.gpu_model.generate(request)
        else:
            # Use CPU for cost savings
            return self.cpu_model.generate(request)
```

### Monitoring and Correctness

**Validation:**

```python
def validate_cpu_gpu_outputs():
    """Compare CPU vs GPU path outputs"""
    test_prompts = load_golden_prompts()
    
    for prompt in test_prompts:
        gpu_output = gpu_model.generate(prompt)
        cpu_output = cpu_model.generate(prompt)
        
        # Check similarity
        similarity = compute_similarity(gpu_output, cpu_output)
        assert similarity > 0.95, "CPU/GPU outputs differ significantly"
```

**Monitoring:**

- Compare CPU vs GPU path outputs with unit tests
- Monitor tail latency to ensure CPU path doesn't introduce spikes
- Track accuracy metrics for quantized CPU kernels
- Alert on quality regressions

## Hands-on Examples

### Example 1: SGLang Multi-Node Deployment

**File:** `examples/ch07_sglang_multinode.py`

```python
"""
Multi-node SGLang deployment with router-based routing.
"""
import subprocess
import time
import requests

def deploy_multinode_sglang():
    """Deploy SGLang across multiple nodes"""
    
    # Launch workers
    workers = []
    for i in range(4):
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--port", str(8000 + i),
            "--tensor-parallel-size", "2"
        ]
        workers.append(subprocess.Popen(cmd))
    
    time.sleep(30)  # Wait for workers to start
    
    # Launch router
    router_cmd = [
        "python", "-m", "sglang_router.launch_router",
        "--worker-urls",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://localhost:8002",
        "http://localhost:8003",
        "--policy", "cache_aware"
    ]
    router = subprocess.Popen(router_cmd)
    
    time.sleep(10)  # Wait for router to start
    
    # Test routing
    response = requests.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "model": "llama-3.1-8b",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
    )
    
    print(f"Response: {response.json()}")
    
    return workers, router

if __name__ == "__main__":
    workers, router = deploy_multinode_sglang()
```

### Example 2: Operator Fusion Profiling

**File:** `examples/ch07_operator_fusion.py`

```python
"""
Profile and optimize operator fusion in SGLang.
"""
import torch
import torch.profiler as profiler
import time

def profile_baseline(model, input):
    """Profile baseline model without fusion"""
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        output = model(input)
    
    return prof.key_averages().table(sort_by="cuda_time_total")

def profile_fused(model, input):
    """Profile model with fused operators"""
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        output = model(input)
    
    return prof.key_averages().table(sort_by="cuda_time_total")

def compare_fusion():
    """Compare baseline vs fused performance"""
    model_baseline = load_baseline_model()
    model_fused = load_fused_model()
    input = torch.randn(1, 512, 4096).cuda()
    
    # Warmup
    for _ in range(10):
        _ = model_baseline(input)
        _ = model_fused(input)
    
    torch.cuda.synchronize()
    
    # Benchmark baseline
    start = time.time()
    for _ in range(100):
        _ = model_baseline(input)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 100
    
    # Benchmark fused
    start = time.time()
    for _ in range(100):
        _ = model_fused(input)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / 100
    
    print(f"Baseline: {baseline_time*1000:.2f}ms")
    print(f"Fused: {fused_time*1000:.2f}ms")
    print(f"Speedup: {baseline_time/fused_time:.2f}x")

if __name__ == "__main__":
    compare_fusion()
```

### Example 3: Chunked Prefill Implementation

**File:** `examples/ch07_chunked_prefill.py`

```python
"""
Chunked prefill implementation for long-context generation.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChunkedPrefillModel:
    def __init__(self, model_name, chunk_size=512):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.kv_cache = []
    
    def prefill_chunked(self, prompt):
        """Prefill prompt in chunks"""
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"]
        
        # Split into chunks
        chunks = [input_ids[:, i:i+self.chunk_size] 
                  for i in range(0, input_ids.size(1), self.chunk_size)]
        
        kv_cache = []
        for chunk in chunks:
            with torch.no_grad():
                outputs = self.model(chunk, use_cache=True)
                kv_cache.append(outputs.past_key_values)
        
        self.kv_cache = kv_cache
        return kv_cache
    
    def generate(self, max_new_tokens=100):
        """Generate using chunked KV cache"""
        # Use last token from prefill as starting point
        input_ids = self.kv_cache[-1][0][0][:, -1:].unsqueeze(0)
        
        generated = []
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids,
                past_key_values=self.kv_cache[-1] if self.kv_cache else None
            )
            
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token.item())
            
            input_ids = next_token.unsqueeze(0)
            self.kv_cache = outputs.past_key_values
        
        return self.tokenizer.decode(generated)

if __name__ == "__main__":
    model = ChunkedPrefillModel("meta-llama/Llama-3.1-8B-Instruct")
    
    long_prompt = "Your long prompt here..." * 100
    model.prefill_chunked(long_prompt)
    output = model.generate()
    print(output)
```

## Best Practices

1. **Profile at operator granularity before fusing**
   - Focus on kernels that reduce overall latency
   - Use `torch.profiler` or Nsight Systems for detailed analysis
   - Measure memory traffic reduction, not just compute time

2. **Use genai-bench with real prompt distributions**
   - Avoid microbenchmark bias
   - Test with production-like workloads
   - Include cold-start and warm-cache scenarios

3. **Design routers to prefer KV locality**
   - Moving KV across nodes is expensive
   - Implement session affinity
   - Use cache-aware routing policies

4. **Implement graceful degradation**
   - Allow CPU fallback on overload
   - Monitor for quality regressions
   - Set up alerts for accuracy drops

5. **Optimize chunked prefill parameters**
   - Tune chunk size based on memory constraints
   - Implement efficient eviction policies
   - Overlap prefill with IO operations

## Use Cases

### Use Case 1: High-QPS AI Chat Services

**Scenario:** Deploy a chat service handling thousands of requests per second

**Approach:**
1. Use SGLang's lightweight runtime for low latency
2. Implement router-based load balancing
3. Use chunked prefill for long conversations
4. Monitor with genai-bench

**Results:**
- Low P95 latency (<200ms)
- High throughput (>1000 QPS per node)
- Efficient memory usage

### Use Case 2: Enterprise Inference Gateways

**Scenario:** Multi-tenant inference gateway with different SLAs

**Approach:**
1. Implement router with priority queues
2. Route high-priority requests to GPU
3. Use CPU for low-priority batch jobs
4. Implement session affinity for better cache utilization

**Results:**
- Cost reduction through CPU/GPU hybrid serving
- SLA compliance for different tenant tiers
- Efficient resource utilization

## Skills Learned

By the end of this chapter, readers will be able to:

1. **Understand SGLang runtime optimizations**
   - Explain operator fusion benefits
   - Identify fusion opportunities
   - Implement custom fused kernels

2. **Deploy SGLang across multiple nodes**
   - Configure multi-node deployments
   - Implement model sharding
   - Manage distributed KV cache

3. **Benchmark inference workloads**
   - Use genai-bench for realistic workloads
   - Analyze performance metrics
   - Identify bottlenecks

4. **Build router-based inference systems**
   - Implement session affinity
   - Design routing policies
   - Handle prefill/decode disaggregation

5. **Use hybrid inference for cost/performance**
   - Profile operators for CPU/GPU placement
   - Implement quantized CPU kernels
   - Monitor quality regressions

## Exercises

1. **Operator Fusion Exercise:**
   - Profile a SGLang model and identify top-3 hot operators
   - Implement fused kernel for one operator
   - Measure latency and memory improvement

2. **Multi-Node Deployment:**
   - Deploy SGLang across 2 nodes
   - Configure router with cache-aware policy
   - Benchmark with genai-bench and compare with single-node

3. **Chunked Prefill:**
   - Implement chunked prefill for a long-context model
   - Measure memory usage vs latency tradeoffs
   - Compare with standard prefill

4. **Router Implementation:**
   - Build a simple router with session affinity
   - Implement adaptive batching
   - Test with realistic workload

## Further Reading

- SGLang Documentation: https://docs.sglang.ai/
- SGLang GitHub: https://github.com/sgl-project/sglang
- genai-bench Documentation: https://docs.sglang.ai/genai-bench/
- DeepSeek Chunked Prefill: Research papers on long-context generation
- Router Architecture Patterns: Load balancing and routing strategies
- Operator Fusion Techniques: CUDA and Triton kernel optimization
