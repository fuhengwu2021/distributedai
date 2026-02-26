# Chapter 7: Cross-Request Optimization with SGLang {-}

*RadixAttention, structured generation, and request-level routing for low-latency inference*

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

SGLang takes a different approach. It supports TP, PP, and EP just like vLLM, but elevates *cross-request optimization* to a first-class concern. While vLLM primarily focuses on intra-request execution efficiency (how efficiently a single request is processed), SGLang additionally optimizes inter-request execution efficiency (how requests interact and share resources).

The core innovations that enable this are **RadixAttention** and the **zero-overhead scheduler**. RadixAttention organizes KV cache as a radix tree, allowing requests with common prefixes to share cached computations. The scheduler overlaps CPU work with GPU computation, eliminating idle time. These optimizations work at the kernel and scheduler level, forming the foundation of SGLang's performance.

On top of this execution engine, SGLang introduces **request-level routing** as a scaling primitive. Instead of only scaling through model parallelism (splitting weights), SGLang can scale through request routing (distributing requests to independent workers). A router directs traffic based on cache locality, session affinity, and load balancing. This is particularly effective for workloads where models fit on a single GPU or small TP group.

The combination is powerful for specific workloads. For multi-turn conversations with shared system prompts, RadixAttention's prefix caching combined with session affinity can reduce latency by 2-3x on follow-up requests. But this benefit is conditional—it requires prefix reuse, conversational workloads, and non-batch-dominated scenarios. For batch throughput workloads or very large models requiring extensive model parallelism, vLLM's approach may be more appropriate.

**SGLang** (Structured Generation Language) emerged from the LMSYS team at UC Berkeley—the same group behind the Chatbot Arena leaderboard. While vLLM focused on memory efficiency through PagedAttention, SGLang's creators asked a different question: how can we optimize across requests, enabling them to share computation and benefit from each other?

The answer led to RadixAttention for cross-request KV cache sharing, X-Grammar for efficient structured output generation, and a zero-overhead scheduler that maximizes GPU utilization. Request-level routing emerged later as a production scaling layer that complements these core innovations.

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

![SGLang Architecture](img/sglang_architecture.png){#fig:sglang-arch .block width=70% align=center}

Figure~\ref{fig:sglang-arch} illustrates the flow of requests through SGLang. Clients (either SGLang programs using the native interpreter or standard HTTP clients) send requests to the API Server, which serves as the entry point. The SGLang Runtime (SRT) then processes these requests through a pipeline of components. The **Tokenizer** converts incoming text into numerical tokens that the model can process. The **Request Queue** buffers these tokenized requests, managing concurrency and preparing them for batching. The **Scheduler** is where SGLang's intelligence lives—it batches requests intelligently, prioritizes tasks that can benefit from KV cache reuse through RadixAttention, and implements zero-overhead scheduling that overlaps CPU work with GPU computation. The **GPU Workers** execute the actual model inference, and can be organized for tensor parallelism (where workers collaborate on a single request) or as independent workers for data parallelism. Finally, the **Detokenizer** converts generated tokens back into human-readable text, and the response flows back through the API Server to the client.

For distributed deployments, SGLang adds a **Router** (also called Model Gateway) layer above the API Server. The Router distributes requests across multiple SRT instances, maintaining session affinity so that requests from the same conversation go to the same worker (preserving KV cache locality). It includes a control plane for worker management, load monitoring, and health checking, plus a data plane that implements various load balancing policies.

The architectural difference from vLLM lies in optimization focus. vLLM uses a scheduler-executor-worker pattern optimized for intra-request efficiency—PagedAttention for memory management, model parallelism for large models. SGLang's architecture additionally optimizes inter-request efficiency through RadixAttention and cache-aware scheduling. Both systems support TP/PP/EP for model parallelism; SGLang adds request-level routing as a complementary scaling primitive for workloads where models fit on individual workers.

![vLLM vs SGLang architecture comparison](img/vllm_vs_sglang.png){#fig:vllm-vs-sglang .block width=85% align=center}

Figure~\ref{fig:vllm-vs-sglang} contrasts the two architectural approaches. On the left, vLLM's model parallelism requires workers to synchronize via all-reduce on every layer—necessary for splitting large models, but adding communication overhead. On the right, SGLang's router-based architecture distributes requests to independent workers, each holding a complete model. Workers process requests without inter-worker synchronization, and the router handles load balancing and cache-aware routing. This independence enables horizontal scaling without communication overhead, but requires each worker to hold the full model (or a small TP group).


## SGLang Core Theory

SGLang's performance advantages come primarily from its execution engine innovations. The core technologies are **RadixAttention** for KV cache reuse across requests, **zero-overhead scheduler** for eliminating CPU/GPU idle time, **X-Grammar** for efficient structured output generation, and **operator fusion** to reduce kernel launch overhead. These kernel-level and scheduler-level optimizations form the foundation, working effectively regardless of deployment topology.

While vLLM focuses on memory efficiency within individual requests (PagedAttention), SGLang emphasizes optimizations that span multiple requests. This makes SGLang particularly effective for workloads where many requests share common patterns—system prompts, few-shot examples, or multi-turn conversations.

### RadixAttention: Prefix Cache Reuse

RadixAttention is perhaps SGLang's most distinctive innovation. While vLLM's PagedAttention optimizes memory management within a single request's KV cache, RadixAttention optimizes across multiple requests by sharing KV cache for common prefixes.

Consider a typical AI assistant deployment. Every request starts with the same system prompt: "You are a helpful assistant. You provide accurate, helpful responses..." This system prompt might be 500 tokens. In a traditional system, if 100 users send requests simultaneously, the system computes KV cache for that 500-token prefix 100 times—a massive waste of computation and memory.

RadixAttention solves this by organizing KV cache as a radix tree (also called a prefix tree). In this data structure, common prefixes are stored once and shared across all requests that use them. When a new request arrives, the system finds the longest matching prefix in the tree, reuses the existing KV cache for that prefix, and only computes KV cache for the new tokens. As requests complete, shared prefixes remain in the tree for future reuse while unique suffixes are evicted.

![RadixAttention prefix sharing](img/radix_tree.png){#fig:radix-tree .block width=85% align=center}

Figure~\ref{fig:radix-tree} illustrates how RadixAttention shares KV cache across requests. Suppose three requests arrive with a common system prompt: "You are helpful. What is Python?", "You are helpful. Explain ML.", and "You are helpful. Write code." The radix tree stores the shared prefix "You are helpful. " once (green node), while each request's unique suffix is stored separately (yellow nodes).

The KV cache for "You are helpful. " is computed once and shared by all three requests. Each unique suffix is computed separately. The savings compound as more requests share the same prefix.

The performance benefits are substantial—but conditional on workload characteristics. For workloads with long shared prefixes (system prompts, few-shot examples, multi-turn conversations), RadixAttention can reduce prefill computation by up to 90%. For workloads without prefix sharing (unique prompts, single-turn interactions), the benefit is minimal. Memory efficiency improves because shared prefixes are stored once rather than per-request. Latency drops 2-3x for requests that hit the cache. And throughput increases because reduced per-request memory enables larger batch sizes.

SGLang's scheduler is aware of the radix cache and uses it to optimize batch formation. When selecting the next batch to run, the scheduler sorts requests by their longest matching prefix length and prioritizes requests with longer shared prefixes. This maximizes cache hit rates and GPU utilization.

The cache also integrates with session affinity. When requests from the same session are routed to the same worker, the radix tree on that worker accumulates the conversation history. Follow-up messages in a conversation benefit from the cached KV from previous turns, dramatically reducing latency for multi-turn interactions.

Under the hood, SGLang implements RadixAttention through a two-level memory pool (as of SGLang v0.5). The first level maps each request to its tokens' KV cache indices. The second level stores the actual KV cache data, organized as `[num_layers, max_tokens, num_heads, head_dim]`. The radix tree sits on top of these pools, tracking which prefixes are cached and enabling efficient lookup and sharing.

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

X-Grammar supports any Context-Free Grammar, including JSON, SQL queries, domain-specific languages, or any structured format your application needs.

### Operator Fusion and Graph-Based IR

Modern GPUs are incredibly fast at computation, but launching kernels has overhead. Each small operation—layer normalization, linear projection, activation—requires a separate kernel launch, and the overhead adds up. SGLang addresses this through operator fusion, combining multiple operations into single kernels.

The most common fusion pattern combines layer normalization, linear projection, and activation into a single kernel. Instead of three kernel launches with intermediate memory writes and reads, the fused kernel does everything in one pass. Memory traffic drops because intermediate results stay in registers rather than being written to global memory.

SGLang uses a graph-based Intermediate Representation (IR) to identify fusion opportunities at compile time. The IR represents the computation as a graph of operations, and optimization passes identify sequences that can be fused. Common patterns include `layernorm + linear + activation`, `attention + output_projection`, and complete MLP fusion (`mlp_up + gelu + mlp_down`).

For MoE (Mixture of Experts) models, SGLang provides specialized fused kernels that combine expert routing with computation, reducing the overhead of the all-to-all communication pattern that MoE requires.

### Zero-Overhead Scheduler

Traditional inference systems execute scheduling and computation serially: the CPU schedules the next batch, then the GPU computes it, then the CPU processes results and schedules again. The GPU sits idle while the CPU works, and scheduler overhead can consume 50% or more of total time.

SGLang's zero-overhead scheduler eliminates this idle time by overlapping CPU scheduling with GPU computation. The key insight is that while the GPU processes batch N, the CPU can prepare batch N+1 and process results from batch N-1. The GPU never waits for the CPU.

![Serial vs zero-overhead scheduler](img/scheduler_comparison.png){#fig:scheduler-comparison .block width=100% align=center}

Figure~\ref{fig:scheduler-comparison} illustrates the difference. In the serial scheduler (top), CPU and GPU alternate: the GPU sits idle during scheduling, and the CPU sits idle during computation. In the zero-overhead scheduler (bottom), CPU work (pre-scheduling, launching, post-processing) overlaps with GPU computation. While the GPU processes one batch, the CPU prepares the next batch and processes results from the previous batch. The GPU never waits.


The scheduler splits CPU work into two logical parts. The "Scheduler CPU" handles pre-scheduling (collecting requests, matching prefixes in the radix tree, allocating memory) and post-scheduling (checking completion conditions, removing finished requests, updating cache). The "Launch CPU" handles kernel launches and result processing. These can overlap because they operate on different batches.

To make this overlap work, SGLang uses a token placeholder mechanism. When the scheduler dispatches a batch to the GPU, it doesn't wait for results. Instead, it allocates placeholder tokens and continues scheduling the next batch. A background thread monitors GPU completion and replaces placeholders with actual tokens when results are ready.

The performance benefits are substantial: up to 2x throughput improvement over serial scheduling, with lower end-to-end latency because the GPU is always busy.

## Router-Based Distributed Architecture

Beyond its core execution engine, SGLang introduces request-level routing as a scaling primitive that complements traditional model parallelism. This is not a replacement for TP/PP—SGLang supports those just like vLLM. Rather, routing provides an additional scaling dimension for workloads where models fit on individual workers.

The key distinction: eliminating *intra-layer* synchronization is impossible when using TP/PP—you still need all-reduce for TP, pipeline handoffs for PP. What router-based scaling eliminates is *inter-request* synchronization. Each worker processes requests independently, with no coordination overhead between workers. The router directs traffic based on load, cache locality, and session affinity.

To be precise: router-based architecture eliminates cross-request synchronization (no coordination between workers processing different requests), but not intra-layer collective operations (TP still requires all-reduce, PP still requires pipeline handoffs, EP still requires all-to-all). If you use TP=8 within each worker, those 8 GPUs still synchronize on every layer—the router doesn't change that. What the router eliminates is the need for workers to coordinate with each other.

This is most valuable when models fit on a single GPU or small TP group (2-8 GPUs). For very large models requiring extensive TP/PP across many GPUs, the router adds little value—you're limited by model parallelism anyway. But for smaller models serving high QPS, routing enables horizontal scaling without the communication overhead of traditional data parallelism.

### SGLang Model Gateway Architecture

The SGLang Model Gateway (formerly called SGLang Router) is the component that implements request-level routing. It sits in front of multiple SRT instances, directing traffic based on sophisticated policies while providing enterprise-grade reliability features.

The gateway architecture has two main layers: a control plane and a data plane. The control plane manages worker lifecycle—discovering workers, tracking their load, monitoring health, and handling registration/removal. The data plane handles actual request routing across multiple protocols (HTTP, gRPC) with built-in reliability features like retries, circuit breakers, and rate limiting.

The control plane includes several components working together. The Worker Manager discovers worker capabilities and tracks real-time load statistics. The Health Checker continuously probes workers to verify availability, updating circuit breaker state when workers fail. The Load Monitor feeds routing policies with live statistics about pending requests, active sessions, and resource utilization. For Kubernetes deployments, Service Discovery automatically keeps the worker registry aligned with pod lifecycle.

The data plane implements multiple router types. The HTTP Router handles standard OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, etc.) with support for streaming and non-streaming modes. The Prefill/Decode Router coordinates disaggregated workloads, merging metadata from prefill and decode phases. The gRPC Router provides higher throughput for performance-critical deployments, with native tokenization and reasoning parser support built in.

For reliability, the gateway implements exponential backoff with jitter for retries, worker-scoped circuit breakers that automatically fail over when workers become unhealthy, and token-bucket rate limiting with queuing. Observability comes through Prometheus metrics (latency, throughput, cache hit rates), OpenTelemetry tracing, and structured logging.

### When Router-Based Architecture Excels

Router-based scaling shines in specific scenarios where its strengths align with workload characteristics. High-QPS interactive workloads—chatbot platforms, enterprise multi-tenant deployments, systems handling 100K+ concurrent sessions—benefit most from the router's cache-aware policies that maximize RadixAttention's prefix sharing.

Multi-turn conversations are another sweet spot. Session affinity keeps all turns of a conversation on the same worker, where KV cache from previous messages is already warm. When a user sends "What about Python?" as a follow-up, the worker doesn't need to recompute the system prompt or the previous exchange—it's all cached. Combined with RadixAttention's prefix sharing across users, this can reduce latency by 2-3x for follow-up requests. But this benefit is conditional: it requires conversational workloads with actual prefix reuse. Single-shot prompts won't see these gains.

PD disaggregation adds another dimension: you can scale prefill workers (compute-bound) independently from decode workers (memory-bound), matching hardware to workload characteristics. And fault tolerance comes naturally—when a worker fails, the router simply routes around it. No need to re-shard model weights or restart a distributed group.

That said, router-based architecture isn't universally superior. For batch inference, the router hop adds latency without providing cache locality benefits—you're better off with direct model parallelism. For very large models (70B+) that require extensive TP/PP across many GPUs, the router adds little value since you're constrained by model parallelism anyway. And for throughput-only workloads where latency doesn't matter, vLLM's continuous batching may achieve higher efficiency without the routing overhead.


## Prefill/Decode Disaggregation

One of SGLang's most powerful distributed patterns is prefill/decode (PD) disaggregation. To understand why this matters, recall the two phases of autoregressive generation from Chapter 6: prefill processes the entire prompt in parallel (compute-bound), while decode generates tokens one at a time (memory-bound). These phases have fundamentally different resource requirements.

![Prefill vs decode resource utilization](img/pd_disaggregation.png){#fig:pd-disaggregation .block width=95% align=center}

Figure~\ref{fig:pd-disaggregation} illustrates why disaggregation makes sense. Prefill processes many tokens in parallel, achieving high compute utilization (85%) but moderate memory bandwidth usage (40%). Decode generates one token at a time, resulting in low compute utilization (25%) but high memory bandwidth usage (90%) as it repeatedly loads KV cache. These opposite resource profiles mean that mixing the phases on the same hardware leads to suboptimal utilization—either compute or memory bandwidth is underutilized at any given time.

In a unified system, prefill and decode compete for the same resources. A long prefill can block decode workers, causing latency spikes for users waiting for tokens. In data-parallel setups, one worker might be doing prefill while another handles decode, leading to inconsistent latency.

PD disaggregation solves this by separating the workloads entirely. Dedicated prefill workers handle initial prompt processing, optimized for compute throughput. Dedicated decode workers handle token generation, optimized for low latency. The router directs new requests to prefill workers, then transfers the KV cache to decode workers for generation.

This is fundamentally different from vLLM's pipeline parallelism (PP), which splits model layers across stages. PP requires strict synchronization between stages—each stage must wait for the previous one. PD disaggregation separates workload types rather than model layers, allowing independent scaling without synchronization overhead.


### PD Disaggregation Architecture

The data flow in PD disaggregation involves both a control plane and a data plane, with the router orchestrating both.

![PD disaggregation architecture](img/pd_architecture.png){#fig:pd-architecture .block width=100% align=center}

Figure~\ref{fig:pd-architecture} shows the complete flow. A single router handles both ingress and egress. In Phase 1, requests arrive and the router dispatches them to a prefill worker (compute-bound). The router also selects which decode worker will handle generation—this is the control plane (dashed arrow). In Phase 2, the prefill worker transfers KV cache blocks directly to the selected decode worker via RDMA or Mooncake—this is the data plane, a worker-to-worker transfer that bypasses the router. The decode worker (memory-bound) then generates tokens and streams them back through the router to the client. Note that raw prompts never go to decode workers; they only receive KV cache blocks from prefill workers.

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

SGLang's core innovations (RadixAttention, zero-overhead scheduler) work at the execution engine level. For scaling, SGLang supports both traditional parallelism strategies (TP/PP/EP) and router-based request distribution. Understanding when to use each approach—and how they can combine—is crucial for building scalable inference systems.

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

![Router-based multi-node deployment](img/router_multi_node.png){#fig:router-multi-node .block width=70% align=center}

Figure~\ref{fig:router-multi-node} shows the router-based architecture. The cache-aware router distributes requests across worker nodes, each holding a complete model. Workers process requests independently with no inter-worker synchronization—this is the key advantage over model parallelism, where workers must coordinate on every layer.

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

### Fault Tolerance

Router-based architecture provides natural fault tolerance. The router continuously monitors worker health through periodic probes to each worker's `/health` endpoint. When a worker fails or becomes unresponsive, the router automatically removes it from the active pool and redistributes traffic to healthy workers.

Circuit breakers prevent cascading failures. If a worker fails repeatedly (typically 5 consecutive failures), the router "opens" the circuit and stops sending requests to that worker for a timeout period. After the timeout, it sends a single probe request—if successful, the worker rejoins the pool; if not, the circuit stays open.

Session migration handles the case where a worker holding active sessions fails. The router remaps affected sessions to healthy workers. The trade-off is that KV cache must be recomputed on the new worker, so the first request after migration incurs full prefill latency. But subsequent requests benefit from the warm cache on the new worker.

This fault tolerance comes naturally from the architecture. In contrast, TP/PP deployments require all ranks to be available—a single GPU failure can bring down the entire serving group.

## Expert Parallelism for MoE Models

We covered the fundamentals of MoE architecture and expert parallelism in Chapters 5 and 6—how tokens are routed to experts, how experts are distributed across GPUs, and the all-to-all communication pattern this requires. Here we focus on SGLang's specific optimizations for MoE inference.

The challenge with MoE inference is that all-to-all communication can dominate runtime, especially at large batch sizes. SGLang addresses this through specialized backends and communication overlap techniques.

For all-to-all communication, SGLang offers multiple backends. DeepEP is optimized for cross-node MoE deployments and is the recommended choice for multi-node setups. Mooncake extends DeepEP with RDMA support for even lower latency on InfiniBand networks. For hybrid EP+TP configurations where you want all-reduce instead of all-to-all, the "none" backend provides that option.

For the actual expert computation (the matrix multiplications inside each expert), SGLang provides DeepGEMM (optimized specifically for MoE patterns), Triton (flexible and portable), and CUTLASS (NVIDIA's high-performance GEMM library). The "auto" setting selects the best option based on your hardware.

A basic MoE deployment looks like this:

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --ep 8 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm
```

For larger models like Qwen3-235B, you'll combine EP with other parallelism dimensions:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --tp 4 --ep 16 --pp 2 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm \
    --enable-dp-attention
```

The `--enable-dp-attention` flag is particularly important for models with few KV heads (like those using MLA). It avoids duplicating KV cache across TP ranks, which would otherwise waste memory.

SGLang's most distinctive MoE optimization is communication overlap. Two-Batch Overlap (TBO) splits the batch into micro-batches and pipelines them: while one micro-batch performs all-to-all communication, another runs attention computation. This can nearly double throughput by hiding communication latency behind computation. Single-Batch Overlap (SBO) achieves similar benefits within a single batch using multiple CUDA streams. Enable these with `--enable-two-batch-overlap` or `--enable-single-batch-overlap`.

For best results, keep EP groups within NVLink domains when possible—all-to-all is bandwidth-intensive, and NVLink's 600+ GB/s far exceeds cross-node interconnects.

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

Speculative decoding compounds well with RadixAttention. RadixAttention reduces prefill time by reusing cached KV for shared prefixes, improving time-to-first-token (TTFT). Speculative decoding then accelerates the decode phase by generating multiple tokens per forward pass. Together, they can dramatically reduce end-to-end latency for conversational workloads: RadixAttention handles the prefix, speculative decoding handles the generation.

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

The difference from vLLM is how continuous batching interacts with the router-based architecture. In SGLang, the router's cache-aware policy considers both load balancing and which workers have relevant prefixes cached. Requests are batched at each worker, with the routing decision already optimized for cache locality. This combination—intelligent routing plus continuous batching—achieves both high throughput and low latency.

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

Throughout this chapter, we've explored SGLang's distinctive approach to LLM inference. The key insight is that SGLang doesn't reject model parallelism—it supports TP, PP, and EP just like vLLM. What sets SGLang apart is its elevation of cross-request optimization to a first-class concern. While vLLM focuses primarily on intra-request efficiency (how fast can we process one request?), SGLang additionally asks: how can multiple requests share work and benefit from each other?

The answer to that question led to SGLang's core technical innovations. RadixAttention organizes KV cache as a radix tree, allowing requests with common prefixes to share cached computations. When many users send requests with the same system prompt, that prompt's KV cache is computed once and reused—potentially saving up to 90% of prefill computation for workloads with long shared prefixes. For workloads without prefix sharing, this benefit diminishes, which is why understanding your workload characteristics matters when choosing between systems.

The zero-overhead scheduler addresses a different bottleneck: the traditional serial pattern where GPUs sit idle while CPUs schedule the next batch. By overlapping CPU scheduling with GPU computation—preparing batch N+1 while the GPU processes batch N—SGLang keeps the GPU continuously busy. This is a kernel-level optimization that works regardless of how you deploy the system.

X-Grammar tackles structured output generation, a common requirement for applications that need JSON, SQL, or other formatted outputs. Rather than validating each token against grammar rules at runtime (prohibitively expensive with 128K-token vocabularies), X-Grammar precompiles valid token sets using finite state machines. The result is guaranteed-valid structured output with minimal overhead.

Built on this execution engine, SGLang introduces router-based architecture as a scaling primitive. The router distributes requests across independent workers, each holding a complete model (or small TP group). This eliminates inter-request synchronization—workers don't coordinate with each other, only with the router. Combined with session affinity (routing conversation turns to the same worker) and cache-aware load balancing, this architecture excels for high-QPS interactive workloads where many concurrent sessions share common patterns.

PD disaggregation takes this further by separating prefill (compute-bound) from decode (memory-bound) into specialized workers. The router dispatches new requests to prefill workers, which transfer KV cache blocks directly to decode workers via RDMA. Each worker type can be scaled independently and tuned for its specific resource profile.

When should you choose SGLang over vLLM? The decision isn't about which system is "better"—it's about which optimization focus matches your workload. SGLang shines when cross-request optimization matters: chatbot platforms with shared system prompts, multi-turn conversations where session affinity preserves KV cache, high-QPS deployments where router-based scaling avoids model parallelism overhead, and applications requiring structured output. vLLM excels when you're serving very large models that require extensive TP/PP across many GPUs, when batch throughput matters more than latency, or when single-shot prompts without prefix reuse dominate your traffic.

It's worth noting that vLLM also supports prefix caching and session pinning—the systems aren't mutually exclusive in their capabilities. They represent different design tradeoffs, different answers to the question of what to optimize. Many production deployments use both: vLLM for batch processing and large model serving, SGLang for interactive APIs where RadixAttention and session affinity provide latency benefits. A routing layer directs traffic to the appropriate backend based on workload characteristics.

We've now covered both sides of distributed AI: training systems (DDP, FSDP, DeepSpeed, Megatron) that optimize for throughput and memory efficiency during model development, and inference systems (vLLM, SGLang) that optimize for latency and throughput when serving trained models. The next chapter provides a hands-on guide to running these workloads on HPC clusters using Slurm, the job scheduler that powers most research clusters and cloud GPU providers.

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
