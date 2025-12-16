# Chapter 7: SGLang and Advanced Inference Architectures

In the previous chapter, we covered vLLM, which uses model parallelism (TP/PP/DP/EP) to distribute large models across GPUs. vLLM excels at high-throughput workloads with large batches and is optimized for models that require multiple GPUs just to fit in memory.

But what if you need ultra-low latency for interactive chat applications? Or what if your models fit on a single GPU but you need to handle thousands of concurrent requests with session persistence? That's where SGLang comes in. This chapter introduces SGLang's distributed inference architecture, focusing on router-based request routing, prefill/decode disaggregation, and multi-node coordination. Unlike vLLM's model parallelism approach, SGLang emphasizes request-level routing and workload disaggregation for high-QPS, low-latency distributed inference.

## Introduction to SGLang and Setup

**SGLang** (Structured Generation Language) is a high-performance inference engine optimized for low-latency, high-QPS workloads. It uses a router-based distributed architecture that emphasizes request routing, session affinity, and workload disaggregation rather than model weight sharding.

### Prerequisites

Before installing SGLang, ensure you have:

- **OS**: Linux (required for GPU support)
- **Python**: 3.10+
- **NVIDIA GPU**: With CUDA support
- **CUDA**: Compatible CUDA version installed

### Installation

SGLang can be installed using multiple methods. It is recommended to use `uv` for faster installation.

#### Method 1: Install with uv (Recommended)

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang" --prerelease=allow
```

#### Method 2: Install with pip

```bash
pip install --upgrade pip
pip install "sglang[all]"
```

#### Method 3: Install from Source

```bash
# Clone the repository
git clone -b v0.5.6 https://github.com/sgl-project/sglang.git
cd sglang

# Install from source
pip install --upgrade pip
pip install -e "python"
```

#### Method 4: Using Docker

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<your-token>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --host 0.0.0.0 \
        --port 30000
```

**Note:** If you encounter `OSError: CUDA_HOME environment variable is not set`, set it with:
```bash
export CUDA_HOME=/usr/local/cuda-<your-cuda-version>
```

#### Verify Installation

```bash
# Check SGLang version
python -c "import sglang; print(sglang.__version__)"

# Test basic functionality
python -m sglang.launch_server --help
```

### Basic Usage

**Start SGLang Server:**

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

**Test with curl:**

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

## SGLang Core Theory

SGLang's architecture is built around three core concepts: graph-based Intermediate Representation (IR), operator fusion, and a lightweight scheduler. These enable low-latency inference by minimizing overhead.

### Graph-Based Intermediate Representation (IR)

SGLang uses a compact graph-based IR that:

- Compiles operator sequences into fused kernels at compile time
- Reduces kernel launch latency significantly
- Enables aggressive operator fusion optimizations

### Operator Fusion

**Why Operator Fusion Matters:**

- **Reduces Kernel Launches:** Combining small operations into single kernels eliminates launch overhead
- **Lowers Memory Traffic:** Intermediate buffers are combined, reducing read/write operations
- **Enables Custom Optimizations:** Fused kernels (e.g., layernorm + linear + activation) outperform generic compositions

**Example:**

```python
# Fused layernorm + linear + activation
class FusedLayerNormLinearActivation(nn.Module):
    def forward(self, x):
        # All operations fused into single kernel
        x = self.layernorm(x)
        x = self.linear(x)
        x = self.activation(x)
        return x
```

### Lightweight Scheduler

SGLang uses a single-threaded event loop scheduler that:

- Batches small work items efficiently
- Dispatches fused kernels with minimal overhead
- Optimizes for low latency rather than maximum throughput

## Router-Based Distributed Architecture

Unlike vLLM's model parallelism (TP/PP/DP) which focuses on splitting model weights, SGLang uses a **router-based architecture** that emphasizes request-level routing, session affinity, and workload disaggregation. This approach is optimized for high-QPS, low-latency workloads where request routing matters more than model sharding.

### Key Architectural Differences from vLLM

**vLLM (Chapter 6) focuses on:**
- **Model-level parallelism**: How to split model weights (TP/PP/DP/EP)
- **Memory efficiency**: PagedAttention for KV cache
- **Throughput optimization**: Continuous batching for large batches

**SGLang (This chapter) focuses on:**
- **Request-level routing**: How to route requests to optimize latency and cache locality
- **Workload disaggregation**: Separating prefill and decode workloads
- **Session management**: Maintaining KV cache locality through routing

### Router Architecture Components

**1. Central Router**
- Routes requests based on session affinity and cache locality
- Implements dynamic routing policies (cache-aware, load-based, priority-based)
- Manages session-to-worker mappings
- Supports A/B testing and canary deployments

**2. Session Affinity / Cache Locality**
- Routes requests from the same session to the same worker
- Keeps KV cache "hot" on specific workers
- Minimizes KV cache transfers between nodes
- Improves cache hit rates significantly (2-3x latency reduction)

**3. Prefill/Decode (PD) Disaggregation**
- Specialized prefill workers for heavy initial computation
- Dedicated decode workers for token generation
- Prevents decode workers from being blocked by long prefill requests
- Enables independent scaling of prefill vs decode capacity

### Why Router-Based Architecture?

**Advantages over Model Parallelism:**

1. **Cache Locality**: Session affinity keeps KV cache on the same worker, avoiding expensive transfers
2. **Independent Scaling**: Scale prefill and decode workers independently based on workload
3. **Fault Tolerance**: Router can route around failed workers without model re-sharding
4. **Flexible Routing**: Implement custom routing policies (priority, SLA, A/B testing)
5. **Lower Latency**: Avoids synchronization overhead of model parallelism for small-to-medium models

**When Router-Based Works Best:**
- Models fit on single GPU or small TP group (2-4 GPUs)
- High QPS with many concurrent sessions
- Latency-sensitive workloads (chat, interactive applications)
- Need for session persistence and cache locality

**When Model Parallelism (vLLM) is Better:**
- Very large models requiring TP/PP across many GPUs
- Throughput-optimized workloads with large batches
- Memory-constrained environments needing PagedAttention

## Prefill/Decode Disaggregation

Prefill/Decode (PD) disaggregation is a key distributed architecture pattern in SGLang that separates the computationally intensive prefill phase from the decode phase, enabling independent scaling and better resource utilization.

### Motivation for PD Disaggregation

**The Problem:**
- Prefill (initial prompt processing) is compute-intensive and can block decode workers
- Decode (token generation) requires low latency and high throughput
- In unified scheduling, prefill batches frequently interrupt ongoing decode batches, causing substantial delays
- In data-parallel attention, one DP worker may process prefill while another handles decode, leading to increased decode latency

**The Solution:**
- Separate prefill workers handle initial prompt processing
- Dedicated decode workers handle token generation
- Router intelligently routes requests to appropriate workers
- KV cache is transferred from prefill workers to decode workers using high-performance transfer engines

### PD Disaggregation Architecture

```
┌─────────────┐
│   Router    │
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       │              │              │
       ↓              ↓              ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Prefill W1  │  │ Prefill W2  │  │ Decode W1   │
│ (Heavy)     │  │ (Heavy)     │  │ (Light)     │
└─────────────┘  └─────────────┘  └─────────────┘
                                       │
                                       ↓
                                ┌─────────────┐
                                │ Decode W2   │
                                │ (Light)     │
                                └─────────────┘
```

### Benefits

1. **Independent Scaling**: Scale prefill and decode workers separately based on workload
2. **Better Resource Utilization**: Prefill workers can use more compute, decode workers optimize for latency
3. **Fault Isolation**: Failure in prefill workers doesn't affect decode workers
4. **Cost Optimization**: Use different hardware types for different workloads

### Transfer Engines

SGLang supports multiple transfer engines for KV cache transfer between prefill and decode workers:

- **Mooncake**: High-performance transfer engine using RDMA for efficient data transfers
- **NIXL**: UCX-based transfer engine for flexible deployment
- **ASCEND**: For Ascend NPU deployments

### PD Disaggregation Setup

**Prerequisites:**

For Mooncake backend:
```bash
uv pip install mooncake-transfer-engine
```

For NIXL backend:
```bash
pip install nixl
```

**1. Start Prefill Workers (Mooncake):**

```bash
# Prefill worker
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill \
    --port 30000 \
    --disaggregation-ib-device mlx5_roce0
```

**2. Start Decode Workers:**

```bash
# Decode worker
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --disaggregation-ib-device mlx5_roce0
```

**3. Start Router with PD Disaggregation:**

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --host 0.0.0.0 \
    --port 8000
```

**For NIXL backend**, replace `--disaggregation-ib-device` with `--disaggregation-transfer-backend nixl`.

## Multi-Node SGLang Deployment

SGLang supports multi-node deployment using tensor parallelism (TP) and expert parallelism (EP) for large models, similar to vLLM. For smaller models, router-based architecture with replicated workers provides an alternative approach.

### Multi-Node Architecture with Tensor Parallelism

For large models that don't fit on a single node, SGLang uses tensor parallelism across multiple nodes:

**Components:**

1. **Distributed Initialization**
   - Uses `--dist-init-addr` to specify master node address
   - `--nnodes` specifies total number of nodes
   - `--node-rank` identifies each node (0 for master, 1, 2, ... for workers)

2. **Tensor Parallelism**
   - `--tp` or `--tensor-parallel-size` splits model weights across GPUs
   - Works across nodes using NCCL for communication

3. **Expert Parallelism (for MoE models)**
   - `--ep` or `--expert-parallel-size` distributes experts across devices
   - Supports DeepEP and Mooncake backends for efficient all-to-all communication

### Multi-Node Setup with Tensor Parallelism

**Example: Llama 3.1 405B on Two Nodes**

```bash
# Node 0 (master node, replace 172.16.4.52:20000 with your master node IP:port)
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
    --tp 16 \
    --dist-init-addr 172.16.4.52:20000 \
    --nnodes 2 \
    --node-rank 0

# Node 1 (worker node)
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-405B-Instruct \
    --tp 16 \
    --dist-init-addr 172.16.4.52:20000 \
    --nnodes 2 \
    --node-rank 1
```

**Key Parameters:**
- `--dist-init-addr`: Master node IP address and port for NCCL initialization
- `--nnodes`: Total number of nodes
- `--node-rank`: Rank of this node (0 for master, 1, 2, ... for workers)
- `--tp`: Tensor parallel size (total GPUs = nnodes × tp)

### Router-Based Multi-Node Deployment

For smaller models, you can use router-based architecture with replicated workers:

**1. Launch Workers on Each Node:**

```bash
# Each node runs full model copy
# Node 1
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Node 2
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

**2. Configure Router (SGLang Model Gateway):**

```bash
python -m sglang_router.launch_router \
    --worker-urls \
        http://node1:8000 \
        http://node2:8000 \
        http://node3:8000 \
        http://node4:8000 \
    --policy cache_aware
```

**3. Test Multi-Node Routing:**

```python
import requests

# First request creates session
response1 = requests.post(
    "http://router:8080/v1/chat/completions",
    json={
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "session_id": "user-123"
    }
)

# Second request routes to same node (cache hit)
response2 = requests.post(
    "http://router:8080/v1/chat/completions",
    json={
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Continue"}],
        "session_id": "user-123"  # Same session ID
    }
)
```

### Performance Considerations

- **Session Affinity Benefits**: 2-3x latency reduction for follow-up requests in same session
- **Cache Locality**: Route requests to nodes with hot KV cache to minimize cache misses
- **Load Balancing**: Distribute load evenly while maintaining session affinity
- **Fault Tolerance**: Router automatically routes around failed nodes
- **Network Overhead**: Minimal compared to model parallelism (only request routing, no weight synchronization)

## Expert Parallelism for MoE Models

SGLang supports Expert Parallelism (EP) for Mixture-of-Experts (MoE) models, distributing expert weights across multiple devices. This is particularly important for large-scale MoE models like DeepSeek-V3 and DeepSeek-R1.

### Expert Parallelism Overview

**Key Features:**
- Distributes expert weights across multiple GPUs
- Optimized all-to-all communication for token routing
- Supports multiple backends: DeepEP, Mooncake, and native implementations
- Enables efficient scaling for high-performance MoE inference

### EP Backends

**All-to-All Communication Backends:**
- **`deepep`**: DeepEP library for efficient token shuffling in MoE models
- **`mooncake`**: Extension of DeepEP for elastic inference, leveraging RDMA
- **`none`**: Uses All-Reduce or All-Gather (for hybrid EP and TP setups)

**MoE Computation Backends:**
- **`auto`**: Automatically selects optimal backend based on hardware and model
- **`triton`**: Triton-based implementation for grouped GEMMs
- **`deep_gemm`**: DeepGEMM backend optimized for MoE matrix multiplications
- **`cutlass`**: CUTLASS-based backend for efficient GEMMs

### Expert Parallelism Setup

**Example: DeepSeek-V3 with EP**

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm \
    --tp 8 \
    --ep 8
```

**Key Parameters:**
- `--ep` or `--expert-parallel-size`: Number of GPUs for expert parallelism
- `--moe-a2a-backend`: Backend for all-to-all communication
- `--moe-runner-backend`: Backend for MoE computation
- `--enable-dp-attention`: Enable data-parallel attention (often used with EP)

### Computation and Communication Overlap

SGLang employs advanced overlap techniques to hide communication latency:

- **Two-Batch Overlap (TBO)**: Splits requests into micro-batches, interleaving attention computation with dispatch/combine operations. Enable with `--enable-two-batch-overlap` for up to 2x throughput improvement.
- **Single-Batch Overlap (SBO)**: Overlaps operations within a single batch. Enable with `--enable-single-batch-overlap`.

## Distributed KV Cache Management

In router-based distributed inference, KV cache is distributed across worker nodes. Each worker maintains its own KV cache, and the router ensures session affinity to maximize cache hit rates.

### KV Cache Distribution Strategy

**Per-Worker KV Cache:**
- Each worker node maintains independent KV cache
- No cross-node KV cache synchronization needed
- Session affinity ensures requests hit cached data

**Cache Locality Optimization:**
- Router routes requests to workers with matching session KV cache
- Minimizes cache misses and expensive recomputation
- Improves latency for conversational workloads

### KV Cache Offloading

For long-context workloads, SGLang supports KV cache offloading:

```python
class DistributedKVCacheManager:
    def __init__(self, workers):
        self.workers = workers
        self.session_cache_map = {}  # session_id -> worker_id
    
    def get_cache_location(self, session_id):
        """Get worker that has KV cache for this session"""
        if session_id in self.session_cache_map:
            return self.session_cache_map[session_id]
        return None
    
    def route_request(self, request):
        session_id = request.get("session_id")
        cache_location = self.get_cache_location(session_id)
        
        if cache_location:
            # Route to worker with cache
            return self.workers[cache_location]
        else:
            # Route to least loaded worker
            return self.select_least_loaded_worker()
```

### Chunked Prefill for Long Contexts

For very long contexts, SGLang uses chunked prefill to manage memory:

```python
def chunked_prefill(model, tokenizer, prompt, chunk_size=512):
    """Chunked prefill for long contexts"""
    chunks = [prompt[i:i+chunk_size] 
              for i in range(0, len(prompt), chunk_size)]
    
    kv_cache = []
    for chunk in chunks:
        tokens = tokenizer(chunk).to(device)
        kv_chunk = model.forward_prefill(tokens)
        kv_cache.append(kv_chunk)
        
        # Optionally offload older KV pages
        if len(kv_cache) > max_gpu_chunks:
            offload_to_cpu(kv_cache[0])
            kv_cache = kv_cache[1:]
    
    return kv_cache
```

## Hands-on Examples

### Example 1: Basic Multi-Node SGLang Deployment

**Deploy SGLang across 4 nodes with router:**

```bash
# Node 1-4: Start workers
for i in {1..4}; do
    ssh node$i "python -m sglang.launch_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --port 8000"
done

# Router node: Start router
python -m sglang_router.launch_router \
    --worker-urls \
        http://node1:8000 \
        http://node2:8000 \
        http://node3:8000 \
        http://node4:8000 \
    --policy cache_aware \
    --port 8080
```

### Example 2: PD Disaggregation Setup with Mooncake

**Separate prefill and decode workers:**

```bash
# Install Mooncake transfer engine
uv pip install mooncake-transfer-engine

# Prefill worker
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode prefill \
    --port 30000 \
    --disaggregation-ib-device mlx5_roce0

# Decode worker (in another terminal)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --disaggregation-ib-device mlx5_roce0

# Router with PD disaggregation
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --host 0.0.0.0 \
    --port 8000
```

### Example 3: Session Affinity Testing

```python
import requests
import time

router_url = "http://router:8080/v1/chat/completions"
session_id = "test-session-123"

# First request (creates session)
response1 = requests.post(
    router_url,
    json={
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "session_id": session_id
    }
)
print(f"First request latency: {response1.elapsed.total_seconds()}s")

# Second request (should hit cache)
response2 = requests.post(
    router_url,
    json={
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "What did I say?"}],
        "session_id": session_id
    }
)
print(f"Second request latency: {response2.elapsed.total_seconds()}s")
print(f"Cache hit improvement: {response1.elapsed / response2.elapsed:.2f}x")
```

### Example 4: Custom Router Implementation

```python
class CustomInferenceRouter:
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
        available = [w for w in self.workers if w.is_available()]
        if not available:
            raise Exception("No available workers")
        
        return min(available, key=lambda w: self.worker_load[w.id])
```

## Summary

### Key Takeaways

1. **Router-based architecture** enables distributed inference without model sharding
2. **Session affinity** provides 2-3x latency improvement for conversational workloads
3. **PD disaggregation** allows independent scaling of prefill and decode workloads
4. **Multi-node deployment** scales horizontally by adding worker nodes
5. **Cache locality** is critical for performance in distributed settings

### When to Use SGLang vs vLLM

**Use SGLang when:**
- Latency (especially TTFT) is critical
- High QPS with many concurrent sessions
- Models fit on single GPU or small TP group
- Need for session persistence and cache locality
- Want router-based distributed inference

**Use vLLM when:**
- Very large models requiring TP/PP across many GPUs
- Throughput-optimized workloads with large batches
- Memory-constrained environments needing PagedAttention
- Model parallelism is the primary concern

### Complementary Approaches

Both systems can coexist:

- **vLLM**: For batch processing, large model serving, high-throughput workloads
- **SGLang**: For interactive chat, low-latency APIs, high-QPS endpoints with session management

We've now covered both training (DDP, FSDP, DeepSpeed) and inference (vLLM, SGLang) systems. But understanding the theory is only part of the equation—you also need to know how to actually run these systems in practice. The next chapter provides a hands-on guide to running distributed AI training workloads using Slurm, the job scheduler used by most HPC clusters and cloud providers. We'll cover setting up Slurm clusters, submitting distributed training jobs, integrating with PyTorch DDP and FSDP, and best practices for production workloads.

## References

- SGLang Documentation: https://docs.sglang.io/
- SGLang GitHub: https://github.com/sgl-project/sglang
- SGLang Model Gateway (Router): https://docs.sglang.io/advanced_features/router.html
- PD Disaggregation: https://docs.sglang.io/advanced_features/pd_disaggregation.html
- Expert Parallelism: https://docs.sglang.io/advanced_features/expert_parallelism.html
- Multi-Node Deployment: https://docs.sglang.io/references/multi_node_deployment/multi_node.html
- Router Architecture Patterns: Load balancing and routing strategies
