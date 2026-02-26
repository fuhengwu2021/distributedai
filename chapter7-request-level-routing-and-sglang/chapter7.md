# Chapter 7: Request-Level Routing and SGLang {-}

*Ultra-low latency inference with request-level routing and workload disaggregation*

> The best way to predict the future is to invent it.
- Alan Kay, Computer Scientist

**Code Summary**

- `sglang.Runtime`: SGLang runtime for model execution
- `sglang.srt.server.LaunchEngine`: Launch SGLang inference engine
- `sglang.srt.hf_transformers_utils`: HuggingFace transformers integration
- `sglang.srt.engine`: SGLang engine for request processing
- `sglang.srt.server`: SGLang server for distributed serving
- `sglang.srt.model_config`: Model configuration for SGLang
- `sglang.srt.tokenizer`: Tokenizer utilities for SGLang
- `sglang.srt.router`: Request router for distributed SGLang
- `sglang.srt.sampling_params`: Sampling parameters for SGLang generation
- `sglang.srt.server_runner`: Server runner for SGLang deployment

## A Different Philosophy for Distributed Inference

In the previous chapter, we explored vLLM's approach to distributed inference: model parallelism. When a model is too large for a single GPU, vLLM splits the model weights across multiple GPUs using tensor parallelism (TP) or pipeline parallelism (PP). Workers must synchronize—all-reduce operations for TP, pipeline stages for PP—and the system optimizes for throughput by batching as many requests as possible.

This approach works well for large models and batch-oriented workloads, but it has limitations. The synchronization overhead between workers adds latency. Session state (like conversation history) doesn't naturally persist across requests. And for smaller models that fit on a single GPU, the model parallelism machinery becomes unnecessary overhead.

SGLang takes a fundamentally different approach. Instead of splitting model weights across workers, SGLang routes requests to independent workers, each running a complete model replica. There's no weight synchronization between workers—each processes requests independently. A router sits in front of the workers, intelligently directing requests based on cache locality, session affinity, and load balancing.

This request-level routing architecture excels at what vLLM's model parallelism struggles with: ultra-low latency for interactive applications, session persistence for multi-turn conversations, and efficient handling of high QPS (queries per second) workloads. When a user sends multiple messages in a conversation, SGLang's router ensures all messages go to the same worker, where the KV cache from previous turns is already warm. The result is 2-3x lower latency for follow-up requests compared to systems without session affinity.

Beyond routing, SGLang introduces innovations that complement its architecture: RadixAttention for sharing KV cache across requests with common prefixes, X-Grammar for efficient structured output generation, and a zero-overhead scheduler that overlaps CPU scheduling with GPU computation. These techniques work together to make SGLang particularly effective for interactive chat applications, AI agents with system prompts, and any workload where latency matters more than raw throughput.

**SGLang** (Structured Generation Language) emerged from the LMSYS team at UC Berkeley—the same group behind the Chatbot Arena leaderboard. While vLLM focused on memory efficiency through PagedAttention, SGLang's creators asked a different question: how can we make LLM serving faster for interactive applications where users expect near-instant responses?

The answer led to a fundamentally different architecture. Instead of optimizing how a single model instance handles requests, SGLang optimizes how requests flow through a distributed system. The result is an inference engine that excels at high-QPS, low-latency workloads—particularly multi-turn conversations where maintaining session state is critical.

### Prerequisites

SGLang runs on Linux with Python 3.10 or later. You'll need an NVIDIA GPU with CUDA support for GPU-accelerated inference. While SGLang also supports other accelerators, NVIDIA remains the most common deployment target.

### Installation

SGLang can be installed using several methods. Docker is the quickest way to try SGLang without installing dependencies locally.

#### Docker Setup

Pre-built Docker images are available on the [SGLang Docker Hub page](https://hub.docker.com/r/lmsysorg/sglang). These images bundle all dependencies, making it easy to get started.

SGLang supports a diverse range of model types, each optimized for different use cases. Base models like `meta-llama/Llama-3.2-1B` are pre-trained language models that use the `/v1/completions` endpoint for text completion. Chat models such as `Qwen/Qwen2.5-0.5B-Instruct` are fine-tuned for conversations and use `/v1/chat/completions`. Embedding models like `Qwen/Qwen3-Embedding-0.6B` generate vector representations through `/v1/embeddings`. For more specialized tasks, SGLang supports diffusion language models for non-autoregressive generation, multimodal models that accept images and videos alongside text, rerank models for search result ordering, and reward models for reinforcement learning applications. SGLang also accelerates diffusion models for image and video generation tasks.

The SGLang server exposes an OpenAI-compatible API, making it a drop-in replacement for applications already using the OpenAI API format. For a complete list of endpoints with detailed descriptions and usage examples, see the **OpenAI-Compatible API Endpoints** section in the Appendix.

__Pull the Latest Image__

Pull the latest Docker image. The runtime image requires approximately 16GB of disk space.

```bash
docker pull lmsysorg/sglang:latest-runtime
```

The `latest-runtime` tag provides a production-ready image with minimal dependencies. There's also a `latest` tag that includes development tools and build dependencies, but at around 35GB it's significantly larger. For most use cases, the runtime image is sufficient.

__Run the Docker Container__

The Docker image runs an OpenAI-compatible server. To avoid hardcoding the model name, set it as an environment variable:

```bash
export SGLANG_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

Then run the container:

```bash
docker run --runtime nvidia --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "SGLANG_MODEL=$SGLANG_MODEL" \
  -p 30000:30000 --ipc=host --shm-size 32g \
  lmsysorg/sglang:latest-runtime \
  python3 -m sglang.launch_server \
    --model-path $SGLANG_MODEL \
    --host 0.0.0.0 \
    --port 30000
```

Note that SGLang does not support OPT models (e.g., `facebook/opt-125m`). It supports modern architectures like Llama, Mistral, Qwen, Gemma, and Phi3. Some models may require the `--trust-remote-code` flag. See the [SGLang documentation](https://docs.sglang.io) for the full list of supported architectures.

Here are some small models suitable for learning purposes:

| Model Name | Type | Parameter Size |
|------------|------|----------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | Chat/Instruct | 0.5B |
| `meta-llama/Llama-3.2-1B-Instruct` | Chat/Instruct | 1B |
| `meta-llama/Llama-3.2-1B` | Base | 1B |
| `microsoft/Phi-tiny-MoE-instruct` | MoE/Instruct | 3.8B total, ~1.1B active |
| `Qwen/Qwen3-Embedding-0.6B` | Embedding | 0.6B |
| `Qwen/Qwen2-VL-2B-Instruct` | VLM/Multimodal | 2B |
| `BAAI/bge-reranker-v2-m3` | Rerank | 0.6B |
| `jason9693/Qwen2.5-1.5B-apeach` | Reward/Classify | 1.5B |

To use any of these models, replace the model name in the Docker command. For example, to serve `meta-llama/Llama-3.2-1B-Instruct`, change `--model-path $SGLANG_MODEL` to `--model-path meta-llama/Llama-3.2-1B-Instruct`.

The Docker flags deserve explanation. The `--runtime nvidia --gpus all` flag enables GPU access; replace `--gpus all` with `--gpus '"device=0"'` for a single GPU or `--gpus '"device=0,1"'` for specific GPUs. The volume mount `-v $HOME/.cache/huggingface:/root/.cache/huggingface` shares your local Hugging Face cache with the container, avoiding repeated model downloads. The `--ipc=host` flag allows the container to access the host's shared memory, which PyTorch uses for efficient data sharing during tensor parallel inference. Finally, `--shm-size 32g` sets the shared memory size, which is important for SGLang's KV cache management and RadixAttention features.

__Verify the Setup__

Once the container is running, verify it's working correctly. First, check that the server is responding:

```bash
curl -w "HTTP Status: %{http_code}\n" http://localhost:30000/health
```

List the available models:

```bash
curl http://localhost:30000/v1/models
```

Test a base model completion:

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SGLANG_MODEL"'",
    "prompt": "The result of 1+1 is",
    "max_tokens": 3
  }'
```

Test a chat model:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SGLANG_MODEL"'",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

For deterministic output (same result every time), add `"temperature": 0`:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SGLANG_MODEL"'",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0
  }'
```

For alternative installation methods (pip, uv, or from source), refer to the [SGLang installation guide](https://docs.sglang.io/get_started/install.html). The documentation also covers model-specific requirements and advanced configuration options.

## Overview of the SGLang Architecture

SGLang's architecture follows a frontend-backend design pattern. The frontend (API Server) handles client requests, while the backend (SGLang Runtime, or SRT) executes inference. This separation allows the system to optimize each layer independently.

```
┌────────────────────────────────────────┐
│           Clients                      │
│  ┌──────────────┐  ┌──────────────┐    │
│  │SGLang Program│  │ HTTP Client  │    │
│  │ + Interpreter│  │              │    │
│  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼────────────┘
          │                 │
          └────────┬────────┘
                   ↓
        ┌──────────────────────┐
        │    API Server        │ ← Entry point for requests
        └──────────┬───────────┘
                   │
                   ↓
    ┌──────────────────────────────────┐
    │   SGLang Runtime (SRT)           │
    │                                  │
    │  ┌──────────┐                    │
    │  │Tokenizer │ ← Converts text to tokens
    │  └────┬─────┘                    │
    │       │                          │
    │       ↓                          │
    │  ┌──────────────┐                │
    │  │Request Queue │ ← Batches requests
    │  └──────┬───────┘                │
    │         │                        │
    │         ↓                        │
    │  ┌──────────────┐                │
    │  │  Scheduler   │ ← Intelligent batching
    │  │              │   RadixAttention
    │  └──────┬───────┘                │
    │         │                        │
    │         ↓                        │
    │  ┌──────────────────────┐        │
    │  │   GPU Workers        │        │
    │  │  W0 → W1 → W2 → W3   │ ← Model execution
    │  └──────┬───────────────┘        │
    │         │                        │
    │         ↓                        │
    │  ┌──────────────┐                │
    │  │ Detokenizer  │ ← Converts tokens to text
    │  └──────┬───────┘                │
    └─────────┼────────────────────────┘
              │
              ↓
        ┌──────────────┐
        │  API Server  │ ← Returns responses
        └──────────────┘
```

The SGLang Runtime contains several key components working together. The **Tokenizer** converts incoming text into numerical tokens that the model can process. The **Request Queue** buffers these tokenized requests, managing concurrency and preparing them for batching. The **Scheduler** is where SGLang's intelligence lives—it batches requests intelligently, prioritizes tasks that can benefit from KV cache reuse through RadixAttention, and implements zero-overhead scheduling that overlaps CPU work with GPU computation. The **GPU Workers** execute the actual model inference, and can be organized for tensor parallelism (where workers collaborate on a single request) or as independent workers for data parallelism. Finally, the **Detokenizer** converts generated tokens back into human-readable text.

For distributed deployments, SGLang adds a **Router** (also called Model Gateway) layer above the API Server. The Router distributes requests across multiple SRT instances, maintaining session affinity so that requests from the same conversation go to the same worker (preserving KV cache locality). It includes a control plane for worker management, load monitoring, and health checking, plus a data plane that implements various load balancing policies.

The architectural difference from vLLM is fundamental. vLLM uses a scheduler-executor-worker pattern focused on model parallelism—splitting model weights across GPUs and coordinating them through all-reduce operations. SGLang's architecture focuses on request-level optimizations instead. Rather than sharding model weights, SGLang routes requests to independent workers, each running a complete model. This eliminates synchronization overhead and enables features like session affinity that are difficult to implement in model-parallel systems.

## SGLang Core Theory

SGLang's performance comes from several innovations working together: RadixAttention for KV cache reuse across requests, X-Grammar for efficient structured output generation, operator fusion to reduce kernel launch overhead, and a zero-overhead scheduler that overlaps CPU and GPU work. While vLLM focuses on memory efficiency within individual requests (PagedAttention) and model parallelism for large models, SGLang emphasizes optimizations that span multiple requests—making it particularly effective for high-QPS workloads where many requests share common patterns.

### RadixAttention: Prefix Cache Reuse

RadixAttention is perhaps SGLang's most distinctive innovation. While vLLM's PagedAttention optimizes memory management within a single request's KV cache, RadixAttention optimizes across multiple requests by sharing KV cache for common prefixes.

Consider a typical AI assistant deployment. Every request starts with the same system prompt: "You are a helpful assistant. You provide accurate, helpful responses..." This system prompt might be 500 tokens. In a traditional system, if 100 users send requests simultaneously, the system computes KV cache for that 500-token prefix 100 times—a massive waste of computation and memory.

RadixAttention solves this by organizing KV cache as a radix tree (also called a prefix tree). In this data structure, common prefixes are stored once and shared across all requests that use them. When a new request arrives, the system finds the longest matching prefix in the tree, reuses the existing KV cache for that prefix, and only computes KV cache for the new tokens. As requests complete, shared prefixes remain in the tree for future reuse while unique suffixes are evicted.

Here's a concrete example. Suppose three requests arrive:

```
Request 1: "You are helpful. What is Python?"
Request 2: "You are helpful. Explain ML."
Request 3: "You are helpful. Write code."

Radix Tree:
Root
 └─ "You are helpful. "
    ├─ "What is Python?" (Request 1)
    ├─ "Explain ML." (Request 2)
    └─ "Write code." (Request 3)
```

The KV cache for "You are helpful. " is computed once and shared by all three requests. Each unique suffix is computed separately. The savings compound as more requests share the same prefix.

The performance benefits are substantial. For workloads with shared prefixes (system prompts, few-shot examples), RadixAttention can reduce prefill computation by up to 90%. Memory efficiency improves because shared prefixes are stored once rather than per-request. Latency drops 2-3x for requests that hit the cache. And throughput increases because reduced per-request memory enables larger batch sizes.

SGLang's scheduler is aware of the radix cache and uses it to optimize batch formation. When selecting the next batch to run, the scheduler sorts requests by their longest matching prefix length and prioritizes requests with longer shared prefixes. This maximizes cache hit rates and GPU utilization.

The cache also integrates with session affinity. When requests from the same session are routed to the same worker, the radix tree on that worker accumulates the conversation history. Follow-up messages in a conversation benefit from the cached KV from previous turns, dramatically reducing latency for multi-turn interactions.

### Structured Output Decoding with X-Grammar

Many applications need LLMs to generate output in specific formats—JSON for API responses, SQL for database queries, or custom schemas for domain-specific tasks. The naive approach to constraint decoding checks each generated token against grammar rules and masks invalid tokens. But with vocabularies of 128K tokens (like Llama-3), checking every token at each step becomes computationally prohibitive.

SGLang's X-Grammar framework solves this efficiently. The key insight is that most grammar rules are context-free—the validity of a token depends only on the current state, not on the history of how we got there. For these rules, X-Grammar precompiles valid token sets using Finite State Machines (FSMs). When generating a boolean value, for instance, the only valid tokens are "true" and "false"—no runtime validation needed. This precompilation handles over 75% of tokens in typical grammars.

For rules that require context (like matching parentheses, where ")" is only valid if there's an unmatched "("), X-Grammar uses Pushdown Automata (PDAs) that extend FSMs with a stack. Traditional implementations snapshot the entire stack at each step, which is expensive. X-Grammar uses tree-based stack management with node reuse, reducing memory copies by 90%.

The result is structured output generation with guaranteed validity and minimal overhead. Here's a simple example:

```python
import sglang as sgl

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "city": {"type": "string"}
    },
    "required": ["name", "age"]
}

response = sgl.generate(
    prompt="Extract information: John is 30 years old, lives in NYC",
    grammar=schema,
    max_tokens=100
)
# Output guaranteed to be valid JSON: {"name": "John", "age": 30, "city": "NYC"}
```

X-Grammar supports any Context-Free Grammar, not just JSON. You can define custom grammars for SQL queries, domain-specific languages, or any structured format your application needs.

### Operator Fusion and Graph-Based IR

Modern GPUs are incredibly fast at computation, but launching kernels has overhead. Each small operation—layer normalization, linear projection, activation—requires a separate kernel launch, and the overhead adds up. SGLang addresses this through operator fusion, combining multiple operations into single kernels.

The most common fusion pattern combines layer normalization, linear projection, and activation into a single kernel. Instead of three kernel launches with intermediate memory writes and reads, the fused kernel does everything in one pass. Memory traffic drops because intermediate results stay in registers rather than being written to global memory.

SGLang uses a graph-based Intermediate Representation (IR) to identify fusion opportunities at compile time. The IR represents the computation as a graph of operations, and optimization passes identify sequences that can be fused. Common patterns include `layernorm + linear + activation`, `attention + output_projection`, and complete MLP fusion (`mlp_up + gelu + mlp_down`).

For MoE (Mixture of Experts) models, SGLang provides specialized fused kernels that combine expert routing with computation, reducing the overhead of the all-to-all communication pattern that MoE requires.

### Zero-Overhead Scheduler

Traditional inference systems execute scheduling and computation serially: the CPU schedules the next batch, then the GPU computes it, then the CPU processes results and schedules again. The GPU sits idle while the CPU works, and scheduler overhead can consume 50% or more of total time.

SGLang's zero-overhead scheduler eliminates this idle time by overlapping CPU scheduling with GPU computation. The key insight is that while the GPU processes batch N, the CPU can prepare batch N+1 and process results from batch N-1. The GPU never waits for the CPU.

The scheduler splits CPU work into two logical parts. The "Scheduler CPU" handles pre-scheduling (collecting requests, matching prefixes in the radix tree, allocating memory) and post-scheduling (checking completion conditions, removing finished requests, updating cache). The "Launch CPU" handles kernel launches and result processing. These can overlap because they operate on different batches.

```
Batch 1: Pre-schedule → Compute → Sample → Post-schedule
Batch 2:              Pre-schedule → Compute → Sample → Post-schedule
Batch 3:                           Pre-schedule → Compute → Sample
```

To make this overlap work, SGLang uses a token placeholder mechanism. When the scheduler dispatches a batch to the GPU, it doesn't wait for results. Instead, it allocates placeholder tokens and continues scheduling the next batch. A background thread monitors GPU completion and replaces placeholders with actual tokens when results are ready.

The performance benefits are substantial: up to 2x throughput improvement over serial scheduling, with lower end-to-end latency because the GPU is always busy.

## Router-Based Distributed Architecture

Now we arrive at SGLang's most distinctive feature: its router-based distributed architecture. While vLLM distributes inference by splitting model weights across GPUs (tensor parallelism, pipeline parallelism), SGLang takes a fundamentally different approach—it routes requests to independent workers, each running a complete model.

The difference is profound. In vLLM's model parallelism, workers must synchronize: all-reduce operations for tensor parallelism, pipeline handoffs for pipeline parallelism. This synchronization adds latency and couples workers together. In SGLang's router-based architecture, workers are independent. Each processes requests on its own, with no synchronization overhead. The router sits in front, directing traffic based on load, cache locality, and session affinity.

This architectural choice reflects different optimization targets. vLLM optimizes for large models that don't fit on a single GPU and for throughput-oriented batch processing. SGLang optimizes for high-QPS, low-latency workloads where models fit on individual GPUs (or small GPU groups) and where request routing decisions matter more than model weight distribution.

### SGLang Model Gateway Architecture

The SGLang Model Gateway (formerly called SGLang Router) is the component that implements request-level routing. It sits in front of multiple SRT instances, directing traffic based on sophisticated policies while providing enterprise-grade reliability features.

The gateway architecture has two main layers: a control plane and a data plane. The control plane manages worker lifecycle—discovering workers, tracking their load, monitoring health, and handling registration/removal. The data plane handles actual request routing across multiple protocols (HTTP, gRPC) with built-in reliability features like retries, circuit breakers, and rate limiting.

The control plane includes several components working together. The Worker Manager discovers worker capabilities and tracks real-time load statistics. The Health Checker continuously probes workers to verify availability, updating circuit breaker state when workers fail. The Load Monitor feeds routing policies with live statistics about pending requests, active sessions, and resource utilization. For Kubernetes deployments, Service Discovery automatically keeps the worker registry aligned with pod lifecycle.

The data plane implements multiple router types. The HTTP Router handles standard OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, etc.) with support for streaming and non-streaming modes. The Prefill/Decode Router coordinates disaggregated workloads, merging metadata from prefill and decode phases. The gRPC Router provides higher throughput for performance-critical deployments, with native tokenization and reasoning parser support built in.

For reliability, the gateway implements exponential backoff with jitter for retries, worker-scoped circuit breakers that automatically fail over when workers become unhealthy, and token-bucket rate limiting with queuing. Observability comes through Prometheus metrics (latency, throughput, cache hit rates), OpenTelemetry tracing, and structured logging.

### Why Router-Based Architecture?

The router-based approach offers several advantages over model parallelism for appropriate workloads.

Cache locality is perhaps the biggest win. With session affinity, all requests from the same conversation go to the same worker, where the KV cache from previous turns is already warm. There's no expensive cache transfer between workers.

Independent scaling becomes possible. You can scale prefill workers (which are compute-bound) separately from decode workers (which are memory-bound), optimizing resource allocation for your specific workload.

Fault tolerance is simpler. When a worker fails, the router simply routes around it. There's no need to re-shard model weights or rebuild distributed state—other workers continue operating independently.

Flexible routing policies enable advanced use cases: priority queues for premium users, A/B testing between model versions, SLA-based routing that directs latency-sensitive requests to less-loaded workers.

The trade-off is that router-based architecture works best when models fit on a single GPU or small TP group (2-4 GPUs). For very large models requiring TP/PP across many GPUs, vLLM's model parallelism approach remains more appropriate.

## Prefill/Decode Disaggregation

One of SGLang's most powerful distributed patterns is prefill/decode (PD) disaggregation. To understand why this matters, recall the two phases of autoregressive generation from Chapter 6: prefill processes the entire prompt in parallel (compute-bound), while decode generates tokens one at a time (memory-bound). These phases have fundamentally different resource requirements.

In a unified system, prefill and decode compete for the same resources. A long prefill can block decode workers, causing latency spikes for users waiting for tokens. In data-parallel setups, one worker might be doing prefill while another handles decode, leading to inconsistent latency.

PD disaggregation solves this by separating the workloads entirely. Dedicated prefill workers handle initial prompt processing, optimized for compute throughput. Dedicated decode workers handle token generation, optimized for low latency. The router directs new requests to prefill workers, then transfers the KV cache to decode workers for generation.

This is fundamentally different from vLLM's pipeline parallelism (PP), which splits model layers across stages. PP requires strict synchronization between stages—each stage must wait for the previous one. PD disaggregation separates workload types rather than model layers, allowing independent scaling without synchronization overhead.

### PD Disaggregation Architecture

The architecture is straightforward: the router sits in front of both prefill and decode workers, directing traffic appropriately.

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
│ (Compute)   │  │ (Compute)   │  │ (Latency)   │
└─────────────┘  └─────────────┘  └─────────────┘
                                       │
                                       ↓
                                ┌─────────────┐
                                │ Decode W2   │
                                │ (Latency)   │
                                └─────────────┘
```

The benefits are substantial. You can scale prefill and decode workers independently—add more prefill workers when prompt processing becomes a bottleneck, add more decode workers when generation latency matters. Different hardware can be used for different workloads: high-compute GPUs for prefill, memory-optimized GPUs for decode. Fault isolation improves because failure in prefill workers doesn't affect ongoing decode operations.

### Transfer Engines

The critical challenge in PD disaggregation is transferring KV cache from prefill workers to decode workers. For long contexts, this cache can be gigabytes of data, so efficient transfer is essential for maintaining low latency.

SGLang supports multiple transfer engines optimized for different network fabrics. **Mooncake** uses RDMA (Remote Direct Memory Access) for direct memory-to-memory transfers without CPU involvement, achieving the lowest latency on InfiniBand networks. **NIXL** provides a UCX-based interface that works across different network fabrics (InfiniBand, Ethernet), offering flexibility at the cost of some performance. **ASCEND** is specialized for Huawei Ascend NPU deployments.

### PD Disaggregation Setup

Setting up PD disaggregation requires starting prefill workers, decode workers, and a router configured to coordinate them. Here's a complete example using the Mooncake transfer engine.

First, install the transfer engine:

```bash
uv pip install mooncake-transfer-engine
```

Start a prefill worker. The `--disaggregation-mode prefill` flag configures the worker for prefill-only operation, and `--disaggregation-ib-device` specifies the InfiniBand device for RDMA transfers:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode prefill \
    --port 30000 \
    --disaggregation-ib-device mlx5_roce0
```

Start a decode worker on a different GPU:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --disaggregation-ib-device mlx5_roce0
```

Finally, start the router with PD disaggregation enabled:

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --host 0.0.0.0 \
    --port 30000
```

For NIXL backend (which works across different network fabrics), replace `--disaggregation-ib-device mlx5_roce0` with `--disaggregation-transfer-backend nixl`.

## Distributed Inference Architecture and Parallelism Strategies

While SGLang's router-based architecture is its distinguishing feature, the system also supports traditional parallelism strategies for cases where they're needed. Understanding when to use each approach—and how they can combine—is crucial for building scalable inference systems.

SGLang supports four parallelism dimensions, which can be combined as needed. **Tensor Parallelism (TP)** splits model weights across GPUs within a node, using all-reduce for synchronization. **Pipeline Parallelism (PP)** splits model layers across GPUs or nodes, using point-to-point communication between stages. **Data Parallelism (DP)** replicates the model across workers, each processing different requests independently. **Expert Parallelism (EP)** distributes MoE experts across devices, using all-to-all communication for token routing.

These dimensions multiply: with TP=8, PP=2, DP=4, and EP=2, you'd use 8 × 2 × 4 × 2 = 128 GPUs. But the key insight is that SGLang's router-based architecture provides an alternative to traditional DP—instead of replicating models with synchronized training-style data parallelism, you can run independent workers behind a router with no synchronization overhead.

### Tensor Parallelism: Weight Sharding

When a model is too large for a single GPU, tensor parallelism (TP) splits model weights across multiple GPUs. SGLang's TP implementation uses the same Megatron-style algorithm as vLLM—column-parallel and row-parallel linear layers with all-reduce communication—but is tailored to work with SGLang's scheduler and RadixAttention.

The core idea is straightforward: each layer's weights are partitioned across TP ranks, each GPU stores 1/TP of the weights, and all-reduce operations after each layer combine partial results. For attention layers, the QKV projections are split column-wise across ranks. For MLP layers, the up projection uses column parallelism and the down projection uses row parallelism.

The communication overhead is significant: all-reduce operations happen after every attention and MLP layer. This makes TP highly sensitive to network topology—it works best when all GPUs in a TP group are connected via NVLink within a single node. Cross-node TP is possible but requires high-bandwidth interconnects like InfiniBand.

Here's how to configure TP for a single node with 8 GPUs:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp 8
```

For multi-node TP (when the model is too large for a single node), you need to specify the distributed initialization address and node ranks:

```bash
# Node 0
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp 16 \
    --dist-init-addr 172.16.4.52:20000 \
    --nnodes 2 \
    --node-rank 0

# Node 1
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp 16 \
    --dist-init-addr 172.16.4.52:20000 \
    --nnodes 2 \
    --node-rank 1
```

### Pipeline Parallelism: Layer Sharding

Pipeline parallelism (PP) splits model depth across GPUs or nodes. Unlike TP which splits each layer horizontally, PP assigns contiguous layers to different stages. Stage 0 might handle layers 0-7, Stage 1 handles layers 8-15, and so on.

The communication pattern is simpler than TP: point-to-point transfers between adjacent stages, rather than all-reduce across all ranks. This makes PP more suitable for cross-node deployment where bandwidth is limited.

The challenge with PP is pipeline bubbles—idle time at the start and end of processing when the pipeline isn't full. During startup, later stages wait for earlier stages to produce output. During drain, earlier stages finish before later stages. Various techniques (micro-batching, virtual pipeline parallelism) help minimize these bubbles, but they can't be eliminated entirely.

PP is typically combined with TP: use TP within nodes (where NVLink provides high bandwidth) and PP across nodes (where lower-bandwidth interconnects are acceptable):

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp 8 \
    --pp 4 \
    --dist-init-addr 172.16.4.52:20000 \
    --nnodes 2 \
    --node-rank 0
```

### Data Parallelism: Request-Level Replication

Data parallelism (DP) replicates the model across multiple workers, with each worker processing different requests. Unlike TP and PP, there's no communication between workers during inference—each operates independently.

This is where SGLang's router-based architecture shines. Traditional DP (as used in training) requires gradient synchronization. But for inference, workers are truly independent. SGLang's router provides intelligent request distribution without any synchronization overhead.

The trade-offs between DP and TP are clear:

| Aspect | Data Parallelism | Tensor Parallelism |
|--------|------------------|-------------------|
| **Memory** | Full model per worker | 1/TP model per worker |
| **Communication** | None (inference) | All-reduce per layer |
| **Latency** | Lower (no sync) | Higher (sync overhead) |
| **Throughput** | Higher for small batches | Higher for large batches |
| **Scalability** | Limited by model size | Scales to very large models |

For models that fit on a single GPU, DP with router-based distribution is almost always better than TP. You get lower latency (no synchronization), better fault tolerance (workers are independent), and features like session affinity that TP can't provide.

Here's the simplest DP setup with a router:

```bash
# Worker 1
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000

# Worker 2
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30001

# Router
python -m sglang_router.launch_router \
    --worker-urls http://worker1:30000 http://worker2:30001 \
    --policy cache_aware
```

### Hybrid Parallelism: Combining Strategies

Real-world deployments often combine multiple parallelism strategies. The most common pattern is TP within nodes (leveraging NVLink) combined with either PP across nodes or router-based DP.

For large models that require multiple nodes, TP+PP is typical. With 32 GPUs across 4 nodes, you might use TP=8 (within each node) and PP=4 (across nodes). Each pipeline stage has an 8-GPU TP group, and stages communicate via point-to-point transfers.

For high-throughput deployments with models that fit on smaller GPU groups, TP+DP (via router) is often better. With 16 GPUs, you might run 4 independent workers, each using TP=4. The router distributes requests across workers with no synchronization overhead.

For MoE models, expert parallelism (EP) adds another dimension. A large MoE deployment might use TP=4 within nodes, PP=2 across node pairs, and EP=16 to distribute experts. The communication patterns become complex—all-reduce for TP, point-to-point for PP, all-to-all for EP—but the parallelism dimensions are orthogonal and can be configured independently.

### Communication Patterns

Each parallelism strategy has a characteristic communication pattern. Understanding these patterns helps you choose the right strategy and optimize performance.

**Tensor Parallelism** uses all-reduce after each layer. Every GPU sends its partial result to every other GPU, and they all compute the sum. The communication volume is O(hidden_size × batch_size) per layer, and it happens frequently—after every attention and MLP layer. This makes TP highly sensitive to interconnect bandwidth.

**Pipeline Parallelism** uses point-to-point communication between adjacent stages. Only neighboring stages communicate, and they only send activations (not gradients, since this is inference). The communication volume is similar to TP, but the frequency is lower—once per micro-batch rather than per layer.

**Expert Parallelism** uses all-to-all communication for token routing. Each GPU sends tokens destined for remote experts and receives tokens destined for local experts. The communication pattern is more complex than TP or PP because the routing is data-dependent—different tokens go to different experts.

**Router-based DP** has no communication during inference. Each worker operates independently, and the router handles request distribution. This is why router-based architecture achieves lower latency for appropriate workloads.

To optimize communication, SGLang supports overlapping communication with computation (enable with `--tp-comm-overlap`), topology-aware placement (keeping TP groups within NVLink domains), and multiple communication backends (NCCL for general use, DeepEP for MoE, Mooncake for RDMA).

## Multi-Node SGLang Deployment

SGLang supports two fundamentally different approaches to multi-node deployment. For large models that don't fit on a single node, you use tensor parallelism (TP) and/or pipeline parallelism (PP) across nodes—similar to vLLM. For smaller models, you use router-based architecture with replicated workers on each node—SGLang's distinctive approach.

The choice depends on your model size and workload characteristics. Router-based deployment avoids communication overhead and enables features like session affinity, but requires each node to hold a complete model. TP/PP deployment enables serving models too large for a single node, but adds synchronization overhead.

### Multi-Node with Tensor Parallelism

For models requiring TP across nodes, you need to configure distributed initialization. The `--dist-init-addr` parameter specifies the master node's address for NCCL initialization. Each node needs a unique `--node-rank` (0 for master, 1, 2, ... for workers), and `--nnodes` specifies the total node count.

Here's an example deploying a model across 2 nodes with TP=16 (8 GPUs per node):

```bash
# Node 0 (master node)
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp 16 \
    --dist-init-addr 172.16.4.52:20000 \
    --nnodes 2 \
    --node-rank 0

# Node 1 (worker node)
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp 16 \
    --dist-init-addr 172.16.4.52:20000 \
    --nnodes 2 \
    --node-rank 1
```

Replace `172.16.4.52:20000` with your master node's IP address and an available port. Both nodes must be able to reach this address for NCCL initialization.

### Router-Based Multi-Node Deployment

For models that fit on a single node (or small TP group), router-based deployment is usually better. Each node runs an independent worker with a complete model, and the router distributes requests across them.

```
                    ┌─────────────┐
                    │   Router    │
                    │  (Gateway)  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ↓                  ↓                  ↓
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ Worker 1│       │ Worker 2│       │ Worker 3│
   │ Node 1  │       │ Node 2  │       │ Node 3  │
   │ Full    │       │ Full    │       │ Full    │
   │ Model   │       │ Model   │       │ Model   │
   └─────────┘       └─────────┘       └─────────┘
```

The setup is straightforward. Launch a worker on each node:

```bash
# Node 1
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000

# Node 2
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000

# Node 3
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000
```

Then configure the router to distribute requests across workers:

```bash
python -m sglang_router.launch_router \
    --worker-urls \
        http://node1:30000 \
        http://node2:30000 \
        http://node3:30000 \
    --policy cache_aware \
    --port 8080
```

The `cache_aware` policy routes requests based on prefix matching, maximizing RadixAttention cache hits. For session-based workloads, requests with the same `session_id` are routed to the same worker, maintaining KV cache locality across conversation turns.

### Router Policies and Load Balancing

The router's policy determines how requests are distributed across workers. SGLang provides several policies optimized for different scenarios.

The **cache-aware policy** is recommended for most workloads. It maintains an approximate radix tree for each worker, tracking which prefixes are likely cached. When a request arrives, the router finds the worker with the best prefix match. If the match exceeds a threshold, the request goes to that worker (cache hit). Otherwise, it falls back to load balancing. This policy maximizes RadixAttention benefits while preventing any single worker from becoming overloaded.

```bash
python -m sglang_router.launch_router \
    --worker-urls http://node1:30000 http://node2:30000 \
    --policy cache_aware \
    --cache-threshold 0.5 \
    --balance-abs-threshold 10 \
    --balance-rel-threshold 1.5
```

The **round-robin policy** distributes requests evenly across workers without considering cache state. It's simple and predictable, useful when requests don't share prefixes or when you want uniform load distribution.

The **shortest-queue policy** routes to the worker with the fewest pending requests. This adapts to variable request lengths—workers processing long requests naturally receive fewer new requests.

The **power-of-two-choices policy** samples two random workers and picks the less loaded one. This provides good load distribution with lower overhead than checking all workers, a classic technique from load balancing literature.

### Session Affinity and Cache Locality

Session affinity is one of SGLang's most powerful features for conversational workloads. The idea is simple: requests from the same conversation should go to the same worker, where the KV cache from previous turns is already warm.

When a request arrives with a `session_id`, the router checks if there's an existing mapping for that session. If so, and the mapped worker is healthy, the request goes to that worker. If not, the router selects a worker using its policy and creates a new mapping.

The latency benefits are dramatic. The first message in a conversation requires full prefill—computing KV cache for the entire prompt. But follow-up messages can reuse the cached KV from previous turns, skipping most of the prefill computation. In practice, this means 2-3x lower latency for follow-up requests in a conversation.

This is something vLLM's model parallelism can't easily provide. In a TP/PP deployment, there's no natural "worker" to route to—the model is distributed across all GPUs. Session state would need to be explicitly managed and potentially transferred between requests. SGLang's router-based architecture makes session affinity a natural consequence of the design.

### Enterprise Extensions

The following sections cover enterprise extensions that extend SGLang's core runtime capabilities:

#### MCP (Model Context Protocol) Integration (Enterprise Extension)

SGLang Model Gateway includes native MCP client integration for advanced tooling and agentic workflows:

**1. MCP Transport Protocols**
- **STDIO**: Standard input/output communication
- **HTTP**: RESTful API communication
- **SSE**: Server-Sent Events for streaming
- **Streamable**: Custom streaming protocol

**2. Tool Execution Loops**
- Native tool-call parsers for JSON, Pythonic, XML, and custom schemas
- Streaming and non-streaming execution modes
- Tool result integration with model responses
- Multi-turn tool execution within conversation context

**3. Reasoning Parser Integration**
- Built-in reasoning parsers for structured-thought models:
  - DeepSeek, Qwen, Llama, Mistral, GPT-OSS
  - Step-3, GLM4, Kimi K2, and other reasoning-capable models
- Automatic parser selection based on model type
- Supports both streaming and non-streaming reasoning extraction

**4. Privacy-Preserving Tool Execution**
- Tool execution occurs within router boundary
- History and context remain local
- No data leakage to external tool providers
- Compliant multi-turn orchestration

#### Enterprise Features (Enterprise Extension)

**1. Security & Authentication**
- Configurable authentication mechanisms
- Secure communication protocols
- API key management
- Role-based access control (RBAC)

**2. Multi-Tenant Support**
- Inference Gateway Mode (`--enable-igw`) for multi-model deployments
- Per-model policy overrides
- Isolated routing stacks per tenant
- Resource quota management

**3. Advanced API Endpoints**
- `/v1/responses`: Agentic multi-turn orchestration
- `/v1/conversations`: Conversation management
- `/v1/rerank`: Reranking capabilities
- `/v1/embeddings`: Embedding generation
- Admin endpoints for worker management

**4. Dynamic Scaling**
- Worker lifecycle management
- Automatic worker registration and removal
- Kubernetes service discovery integration
- Horizontal scaling based on load

### Fault Tolerance and High Availability

SGLang Model Gateway provides comprehensive fault tolerance:

**1. Health Checking:**

Router continuously monitors worker health:

```python
class HealthChecker:
    def check_worker_health(self, worker):
        try:
            response = requests.get(f"{worker.url}/health", timeout=1.0)
            return response.status_code == 200
        except:
            return False
    
    def background_health_check(self):
        """Continuous background health monitoring"""
        while True:
            for worker in self.workers:
                is_healthy = self.check_worker_health(worker)
                worker.update_health_status(is_healthy)
            time.sleep(self.check_interval)
```

**2. Circuit Breakers:**

Worker-scoped circuit breakers prevent cascading failures:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func):
        if self.state == "open":
            if time.time() - self.last_failure > self.timeout:
                self.state = "half-open"
            else:
                raise CircuitBreakerOpenError()
        
        try:
            result = func()
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.last_failure = time.time()
            raise
```

**3. Automatic Failover:**

When worker fails, router automatically routes around it:

```python
def route_request(self, request):
    healthy_workers = [
        w for w in self.workers 
        if w.is_healthy() and not w.circuit_breaker.is_open()
    ]
    
    if not healthy_workers:
        raise Exception("No healthy workers available")
    
    # Route to healthy worker with retry
    return self.select_worker_with_retry(healthy_workers, request)
```

**4. Session Migration:**

For failed workers, sessions can be migrated:

```python
def handle_worker_failure(self, failed_worker_id):
    # Find all sessions on failed worker
    affected_sessions = [
        sid for sid, wid in self.session_map.items()
        if wid == failed_worker_id
    ]
    
    # Migrate sessions to healthy workers
    for session_id in affected_sessions:
        new_worker = self.select_worker(self.healthy_workers)
        self.session_map[session_id] = new_worker
        # Note: KV cache will be recomputed on new worker
        self.notify_session_migration(session_id, new_worker)
```

**5. Graceful Degradation:**

System continues operating with reduced capacity:

- Failed workers automatically removed from pool
- Remaining workers handle increased load
- Router automatically rebalances traffic
- Circuit breakers prevent routing to unhealthy workers
- Health checks continuously probe for recovery

### Performance Considerations

**Network Topology:**

- **Intra-Node**: Use TP within nodes (NVLink domain)
- **Inter-Node**: Use PP or router-based for cross-node communication
- **InfiniBand**: Recommended for multi-node deployments

**Communication Optimization:**

- **Overlap**: Enable `--tp-comm-overlap` to hide communication latency
- **Topology Awareness**: Configure NCCL to use optimal network paths
- **Bandwidth**: Ensure sufficient bandwidth for TP all-reduce

**Load Balancing Trade-offs:**

- **Cache-Aware**: Best for conversational workloads with session affinity
- **Round-Robin**: Best for uniform, stateless workloads
- **Shortest Queue**: Best for dynamic, variable-length requests

**Scaling Guidelines:**

- **Small Models (<10B)**: Router-based with DP
- **Medium Models (10B-100B)**: TP within nodes, router across nodes
- **Large Models (100B+)**: TP+PP across multiple nodes
- **MoE Models**: TP+EP, consider DP Attention for MLA models

## Expert Parallelism for MoE Models

Mixture-of-Experts (MoE) models present unique challenges for distributed inference. Unlike dense models where every parameter is used for every token, MoE models route each token to a subset of "expert" networks. This sparse activation pattern enables much larger models (in total parameters) while keeping computation manageable.

The challenge is that experts need to be distributed across GPUs, and tokens need to be routed to the right experts. This requires all-to-all communication—every GPU potentially sends tokens to every other GPU and receives tokens from every other GPU. SGLang's Expert Parallelism (EP) provides efficient implementations of this pattern.

### How Expert Parallelism Works

In an MoE layer, a router network examines each token and selects the top-K experts (typically K=2) to process it. With EP, experts are distributed across GPUs:

```
8 Experts, EP=4:
  GPU 0: Experts 0, 1
  GPU 1: Experts 2, 3
  GPU 2: Experts 4, 5
  GPU 3: Experts 6, 7
```

The token routing process involves several steps. First, the router computes routing scores for each token and selects the top-K experts. Then, an all-to-all operation dispatches tokens to the GPUs holding their assigned experts. Each GPU processes the tokens it received with its local experts. Another all-to-all gathers the expert outputs back. Finally, the outputs are combined using the routing weights.

### EP Backends

SGLang supports multiple backends for the all-to-all communication that EP requires. **DeepEP** is optimized for cross-node communication and recommended for multi-node MoE deployments. **Mooncake** extends DeepEP with RDMA support for even higher performance. For hybrid EP+TP setups, the **none** backend uses all-reduce or all-gather instead of all-to-all.

For the actual MoE computation, SGLang provides several backends. The **auto** setting automatically selects the best option based on hardware and model characteristics. **Triton** provides a flexible implementation using Triton kernels. **DeepGEMM** is optimized specifically for MoE matrix multiplications. **CUTLASS** uses NVIDIA's CUTLASS library for efficient GEMMs.

### Expert Parallelism Setup

Here's an example deploying an MoE model with EP:

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm \
    --tp 8 \
    --ep 8
```

For larger MoE models, you can combine EP with TP and PP:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --tp 4 \
    --ep 16 \
    --pp 2 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm \
    --enable-dp-attention
```

The key parameters are `--ep` for expert parallel size, `--moe-a2a-backend` for the all-to-all communication backend, `--moe-runner-backend` for the computation backend, and `--enable-dp-attention` which is often beneficial for MoE models with few KV heads.

### EP Communication Optimization

The all-to-all communication in EP can become a bottleneck, especially for large batch sizes. SGLang provides several optimization techniques.

**Two-Batch Overlap (TBO)** splits requests into micro-batches and interleaves attention computation with all-to-all operations. While one micro-batch is doing all-to-all, another is doing attention. This can provide up to 2x throughput improvement by hiding communication latency:

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --ep 8 \
    --enable-two-batch-overlap
```

**Single-Batch Overlap (SBO)** uses multiple CUDA streams to overlap operations within a single batch. Attention runs on one stream while all-to-all runs on another:

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --ep 8 \
    --enable-single-batch-overlap
```

For best results, keep EP groups within NVLink domains when possible, and batch tokens by expert to reduce communication overhead.

### Combining EP with Other Parallelism

EP can be combined with TP, PP, and DP Attention for maximum scalability. A typical large MoE deployment might use:

- **EP + TP**: 8 expert parallel groups, each with 4-GPU TP = 32 GPUs total. EP handles expert distribution, TP handles dense layer sharding.
- **EP + PP**: 2 pipeline stages, each with 8-GPU EP = 16 GPUs total. Useful when model depth requires pipeline parallelism.
- **EP + DP Attention**: For models with few KV heads (like MLA architectures), DP Attention avoids KV cache duplication while EP handles experts.

The communication patterns layer on top of each other: all-to-all for EP, all-reduce for TP, point-to-point for PP. TBO and SBO can overlap these communications to hide latency.

## Advanced KV Cache Management

SGLang's KV cache management builds on the RadixAttention concepts we discussed earlier, implementing them through a two-level memory pool system.

The first level, the **request-to-token pool**, maps each request to its tokens' KV cache indices. This is a simple 2D array where `req_to_token_pool[request_id][token_position]` gives the KV cache index for that token.

The second level, the **token-to-KV pool**, stores the actual KV cache data. It's organized as `[num_layers, max_tokens, num_heads, head_dim]`, allowing efficient access to KV cache for any token at any layer.

The **radix tree cache** sits on top of these pools, enabling prefix sharing. When a new request arrives, the system traverses the radix tree to find the longest matching prefix. For matched tokens, it reuses existing KV cache indices. For new tokens, it allocates fresh slots from the token-to-KV pool.

### Chunked Prefill for Long Contexts

For very long contexts that exceed available GPU memory, SGLang supports chunked prefill. Instead of processing the entire prompt at once, it splits the prompt into smaller chunks and processes them sequentially. Older chunks can be offloaded to CPU memory if needed, allowing SGLang to handle contexts longer than GPU memory can hold.

This is particularly useful for applications like document summarization or code analysis where prompts can be tens of thousands of tokens.

## Distributed KV Cache Management

In router-based distributed inference, each worker maintains its own independent KV cache. There's no cross-node KV cache synchronization—that would defeat the purpose of the router-based architecture. Instead, the router ensures session affinity so that requests from the same conversation always go to the same worker, where the relevant KV cache is already present.

This design has important implications. KV cache locality is maintained through routing decisions, not through cache transfer. When a worker fails, sessions mapped to it lose their cached state and must rebuild it on a new worker. The trade-off is simplicity and performance: no complex distributed cache coherence protocol, no cache transfer overhead, just intelligent routing.

## Speculative Decoding

Speculative decoding is an optimization technique that can significantly accelerate inference. The core idea is to use a smaller, faster "draft" model to predict multiple tokens, then verify them in parallel with the larger "target" model.

Traditional decoding generates one token at a time, with each step requiring a full forward pass through the model. This is memory-bound—the GPU spends most of its time loading model weights rather than computing. Speculative decoding changes this by batching multiple verification steps together.

Here's how it works. The draft model (typically 2-4x smaller than the target) generates N draft tokens quickly. Then the target model verifies all N tokens in a single forward pass. If all draft tokens match what the target would have generated, you get N tokens for the cost of one forward pass—an N-fold speedup. If some tokens don't match, the target model's output is used from the first mismatch point, and the rejected tokens' KV cache is evicted.

In practice, acceptance rates vary based on how well the draft model approximates the target. For well-matched model pairs (same family, same training data), acceptance rates of 70-90% are common, yielding 2-4x speedups.

SGLang supports speculative decoding with configurable draft models:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --speculative-draft-model-path Qwen/Qwen2.5-0.5B-Instruct \
    --speculative-num-draft-tokens 4
```

The draft and target models should share the same tokenizer and vocabulary. Using models from the same family (like Qwen2.5-0.5B as draft for Qwen2.5-7B) typically gives the best results.

## Data Parallel Attention

Data Parallel Attention (DP Attention) is SGLang's optimization for models with few KV heads, like those using Multi-Head Latent Attention (MLA). The problem it solves is subtle but important.

In traditional tensor parallelism, QKV projections are split across GPUs. But when the number of KV heads is less than the TP size, the KV heads must be replicated across GPUs. For a model with 1 KV head and TP=8, each of the 8 GPUs stores a complete copy of the KV cache—8x memory waste.

DP Attention takes a different approach. Instead of splitting the attention computation with TP, it uses data parallelism: each GPU processes different requests independently, with no KV cache duplication. Before the MLP layer, an all-gather combines the attention outputs from all GPUs. The MLP then runs with tensor parallelism (since MLP doesn't have the KV cache problem), and the outputs are sliced back to each GPU.

The result is up to 1.9x higher decoding throughput for models with few KV heads, because memory isn't wasted on duplicated KV cache. Enable it with:

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --enable-dp-attention \
    --dp-size 8 \
    --tp-size 8
```

DP Attention is most beneficial for large batch sizes where memory efficiency matters. For low-latency, small-batch scenarios, the all-gather overhead may outweigh the benefits.

## Scheduler Evolution

SGLang's scheduler has evolved through several generations, each reducing GPU idle time further.

The initial serial scheduler executed steps sequentially: CPU schedules, GPU computes, CPU processes results, repeat. The GPU sat idle during CPU work, and scheduler overhead could consume 50% or more of total time.

The current zero-overhead scheduler (discussed earlier in this chapter) overlaps CPU and GPU work completely. While the GPU processes batch N, the CPU prepares batch N+1 and processes results from batch N-1. The GPU never waits for the CPU.

The latest implementation uses multiple CUDA streams and a FutureMap for async result handling. The result is up to 2x throughput improvement over the serial scheduler, with lower end-to-end latency because the GPU is always busy.

## Continuous Batching

Like vLLM, SGLang implements continuous batching (also called dynamic batching) to handle variable-length sequences efficiently. Instead of waiting for all requests in a batch to complete before starting new ones, continuous batching adds new requests as they arrive and removes completed requests immediately.

The difference from vLLM is how continuous batching interacts with the router-based architecture. In SGLang, the router's cache-aware policy considers not just load balancing but also which workers have relevant prefixes cached. Requests are batched at each worker, but the routing decision already optimized for cache locality. This combination—intelligent routing plus continuous batching—achieves both high throughput and low latency.

SGLang's scheduler prioritizes prefill requests over decode. When a new request arrives, it can interrupt ongoing decode batches to start prefill immediately. This reduces time-to-first-token (TTFT) at the cost of slightly higher inter-token latency for in-progress requests—usually a good trade-off for interactive applications.

## Production Deployment Patterns

Different model sizes and workload characteristics call for different deployment patterns. Here's a guide to choosing the right approach.

### Small Models (<10B): Router-Based DP

For models that fit on a single GPU, router-based deployment with data parallelism is almost always the best choice. Each node runs a complete model, and the router distributes requests with cache-aware routing and session affinity.

```bash
# Workers on each node
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000

# Router
python -m sglang_router.launch_router \
    --worker-urls http://node1:30000 http://node2:30000 http://node3:30000 \
    --policy cache_aware \
    --port 8080
```

Expect 50-100ms TTFT, 1000+ QPS, and 60-80% cache hit rates for conversational workloads.

### Medium Models (10B-100B): TP Within Nodes + Router

For models requiring 2-8 GPUs, use tensor parallelism within each node and router-based distribution across nodes. Each node runs a TP group, and the router treats each TP group as a single worker.

```bash
# Each node runs TP=8
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 8 \
    --port 30000

# Router across nodes
python -m sglang_router.launch_router \
    --worker-urls http://node1:30000 http://node2:30000 \
    --policy cache_aware
```

### Large Models (100B+): TP+PP Across Nodes

For very large models, use tensor parallelism within nodes and pipeline parallelism across nodes. This is similar to vLLM's approach—the model is too large for router-based replication.

### MoE Models: EP+TP Hybrid

For MoE models, combine expert parallelism with tensor parallelism. Enable DP Attention if the model has few KV heads, and use TBO/SBO for communication overlap.

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --tp 8 --ep 16 \
    --moe-a2a-backend deepep \
    --enable-dp-attention \
    --enable-two-batch-overlap
```

### PD Disaggregation for Mixed Workloads

When prefill and decode have very different characteristics (long prompts with short generations, or vice versa), PD disaggregation lets you scale each phase independently.

## Performance Optimization Guide

### Choosing a Parallelism Strategy

The right parallelism strategy depends primarily on model size. For models under 10B parameters that fit on a single GPU, router-based data parallelism gives the best throughput and latency. For models between 10B and 100B, use tensor parallelism within nodes (TP=4-8) and router-based distribution across nodes. For models over 100B, you'll need TP+PP across nodes, similar to vLLM. MoE models add expert parallelism to the mix.

Model architecture also matters. Dense models use TP+PP. MoE models benefit from EP+TP, and models with few KV heads (like MLA architectures) should enable DP Attention. Long-context workloads benefit from chunked prefill.

### Communication Optimization

Keep tensor parallelism within NVLink domains to minimize communication overhead. Use pipeline parallelism for inter-node communication. Set NCCL environment variables appropriately for your network topology:

```bash
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
```

Enable communication overlap wherever possible: `--tp-comm-overlap` for tensor parallelism, `--enable-two-batch-overlap` for expert parallelism. These flags hide communication latency behind computation.

### Memory Optimization

RadixAttention is enabled by default and provides significant memory savings through prefix sharing. For long-context workloads, enable chunked prefill to avoid memory spikes during prefill. Quantization (FP8 or INT4/AWQ) reduces memory footprint and can improve throughput.

### Latency vs Throughput

For latency-sensitive workloads, use cache-aware routing to maximize prefix hits, enable speculative decoding for faster generation, and co-locate the router with workers to minimize network hops.

For throughput-sensitive workloads, tune batch sizes (larger batches amortize overhead but increase latency), scale data parallelism by adding more workers, and enable all overlap options to maximize GPU utilization.

## Hands-on Examples

### Basic Multi-Node Deployment

The simplest SGLang deployment runs workers on multiple nodes with a router for load balancing:

```bash
# Start workers on each node
for i in {1..4}; do
    ssh node$i "python -m sglang.launch_server \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --port 30000"
done

# Start router
python -m sglang_router.launch_router \
    --worker-urls http://node1:30000 http://node2:30000 http://node3:30000 http://node4:30000 \
    --policy cache_aware \
    --port 8080
```

### PD Disaggregation with Mooncake

For workloads with distinct prefill and decode characteristics, PD disaggregation separates these phases:

```bash
# Install transfer engine
uv pip install mooncake-transfer-engine

# Prefill worker
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode prefill \
    --port 30000 \
    --disaggregation-ib-device mlx5_roce0

# Decode worker
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --disaggregation-ib-device mlx5_roce0

# Router
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --port 8080
```

### Testing Session Affinity

To verify that session affinity improves cache hit rates:

```python
import requests

router_url = "http://router:8080/v1/chat/completions"
session_id = "test-session-123"

# First request creates session
response1 = requests.post(router_url, json={
    "model": "opt-125m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "session_id": session_id
})
print(f"First request: {response1.elapsed.total_seconds()}s")

# Second request should hit cache
response2 = requests.post(router_url, json={
    "model": "opt-125m",
    "messages": [{"role": "user", "content": "What did I say?"}],
    "session_id": session_id
})
print(f"Second request: {response2.elapsed.total_seconds()}s")
print(f"Speedup: {response1.elapsed / response2.elapsed:.2f}x")
```

### MoE Model with EP+TP

For MoE models, combine expert parallelism with tensor parallelism:

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --tp 8 --ep 16 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm \
    --enable-dp-attention \
    --enable-two-batch-overlap
```

## Distributed Coordination

Distributed inference requires coordination between components. The key synchronization points differ by parallelism type:

**Tensor Parallelism** uses all-reduce after each layer to combine partial results across GPUs. This is the most communication-intensive pattern.

**Pipeline Parallelism** uses point-to-point communication between stages. Each stage sends its output to the next stage and receives input from the previous stage.

**Expert Parallelism** uses all-to-all communication to route tokens to their assigned experts and gather results.

**Router-based DP** requires no synchronization between workers—each processes requests independently, and the router handles distribution.

### Fault Tolerance

SGLang's router architecture provides natural fault tolerance. When a worker fails, the router detects it through health checks and routes subsequent requests to healthy workers. Sessions may lose their cached KV state, but requests continue to be served.

For TP/PP deployments, failure is more disruptive since all ranks in a group must be available. Recovery typically requires restarting the entire group.

## Summary

This chapter explored SGLang's approach to distributed inference, which differs fundamentally from vLLM's model parallelism. Where vLLM shards model weights across GPUs and requires synchronization between workers, SGLang's router-based architecture distributes requests to independent workers, eliminating communication overhead.

The key innovations that make SGLang effective for high-QPS, low-latency workloads are:

**RadixAttention** enables KV cache sharing across requests with common prefixes. For workloads with shared system prompts or multi-turn conversations, this can save up to 90% of prefill computation.

**X-Grammar** provides efficient structured output decoding using FSMs and PDAs. Unlike naive constraint decoding that validates each token against the full grammar, X-Grammar precompiles token masks and achieves 75%+ cache hit rates.

**Zero-Overhead Scheduler** overlaps CPU scheduling with GPU computation, eliminating the idle time that plagued earlier scheduler designs.

**Router-based architecture** enables horizontal scaling without the communication overhead of tensor or pipeline parallelism. Each worker runs a complete model and processes requests independently.

**Session affinity** routes requests from the same conversation to the same worker, maintaining KV cache locality and improving latency by 2-3x for multi-turn interactions.

**PD disaggregation** separates prefill and decode into specialized workers, allowing independent scaling based on workload characteristics.

### When to Choose SGLang vs vLLM

The choice between SGLang and vLLM comes down to your primary optimization target.

**Choose SGLang when:**
- Latency is critical (especially TTFT under 50ms)
- You have high QPS with many concurrent sessions
- Models fit on 1-8 GPUs
- You need session affinity for conversational workloads
- Structured output decoding is required

**Choose vLLM when:**
- Very large models require 16+ GPUs
- Throughput is more important than latency
- You're doing batch processing
- Simpler deployment without router setup is preferred

The fundamental architectural difference: vLLM shards model weights across GPUs and requires synchronization between workers. SGLang routes requests to independent workers running complete models, eliminating communication overhead but limiting model size to what fits on a single worker (or small TP group).

### Complementary Deployment

Many production systems use both. vLLM handles batch processing and large model serving where model parallelism is necessary. SGLang handles interactive APIs and structured generation where low latency matters. A routing layer directs requests to the appropriate backend based on workload characteristics.

### Looking Ahead

We've now covered both training (DDP, FSDP, DeepSpeed) and inference (vLLM, SGLang) systems. The next chapter provides a hands-on guide to running distributed AI workloads using Slurm, the job scheduler used by most HPC clusters and cloud providers.

## References

__SGLang and RadixAttention__

- SGLang: Efficient Execution of Structured Language Model Programs (2023). https://arxiv.org/abs/2312.07104
- SGLang v0.4: Faster, Longer, and Scalable LLM Serving (2025). https://arxiv.org/abs/2506.21901
- SGLang Documentation: https://docs.sglang.io/
- SGLang GitHub: https://github.com/sgl-project/sglang

__Distributed Inference__

- SGLang Model Gateway (Router): https://docs.sglang.io/advanced_features/router.html
- PD Disaggregation: https://docs.sglang.io/advanced_features/pd_disaggregation.html
- Expert Parallelism: https://docs.sglang.io/advanced_features/expert_parallelism.html
- Multi-Node Deployment: https://docs.sglang.io/references/multi_node_deployment/multi_node.html

__Structured Output and Constrained Decoding__

- XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models (2024). https://arxiv.org/abs/2411.15100
- Constrained Decoding in SGLang: https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/constraint-decoding
- Understanding Constraint Decoding: https://www.aidancooper.co.uk/constrained-decoding/

__Tutorials and Walkthroughs__

- SGLang Code Walk Through: https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md
- SGLang Scheduler Evolution: https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/scheduler-evolution/SGLang%20Scheduler%20Evolution.md
- Why SGLang is a Game-Changer for LLM Workflows (Hugging Face, 2025). https://huggingface.co/blog/paresh2806/sglang-efficient-llm-workflows
- Use Cases Favoring vLLM vs SGLang (2025). https://kanerika.com/blogs/sglang-vs-vllm/
