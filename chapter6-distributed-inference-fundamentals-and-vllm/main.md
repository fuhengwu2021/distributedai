title: "Distributed Inference Fundamentals and vLLM"

# Chapter 6 — Distributed Inference Fundamentals and vLLM

This chapter introduces distributed inference concepts and vLLM internals, focusing on tensor parallelism, data parallelism, pipeline parallelism, and optimization techniques for serving large language models efficiently in production environments.

## Introduction to vLLM and Setup

**vLLM** (virtual Large Language Model) is a high-throughput, memory-efficient inference and serving engine for large language models. It was designed to address the critical challenges of serving LLMs in production: maximizing throughput while minimizing latency and memory usage.

### Prerequisites

Before installing vLLM, ensure you have:

- **OS**: Linux (required for GPU support)
- **Python**: 3.10+
- **NVIDIA GPU**: With CUDA support (for CUDA-based installation)
- **CUDA**: Compatible CUDA version installed

### Installation

vLLM can be installed using several methods. **Docker is the quickest way to try vLLM** without installing dependencies locally.

#### Docker Setup

Docker provides the quickest way to get started with vLLM without installing dependencies locally. Pre-built images are available on the [vLLM Docker Hub page](https://hub.docker.com/r/vllm/vllm-openai).

vLLM supports three main types of models. Base models like `facebook/opt-125m` are pre-trained language models without instruction tuning, and they use the `/v1/completions` endpoint with a `prompt` parameter. Chat models such as `Qwen/Qwen2.5-0.5B-Instruct` are fine-tuned for conversational tasks and use `/v1/chat/completions` with a `messages` parameter. Embedding models like `sentence-transformers/all-MiniLM-L6-v2` generate vector representations and use `/v1/embeddings` with an `input` parameter.

The vLLM server exposes an OpenAI-compatible API with the following endpoints:

| Endpoint | Method | Description | Usage |
|------------------|--------|-----------------------|---------------------------|
| `/v1/models` | GET | List available models | `curl http://localhost:8000/v1/models` |
| `/v1/completions` | POST | Text completion for base models | Use with `prompt` parameter for base models |
| `/v1/chat/completions` | POST | Chat completion for instruction-tuned models | Use with `messages` parameter for chat models |
| `/v1/embeddings` | POST | Generate embeddings from text | Use with `input` parameter for embedding models |
| `/health` | GET | Health check endpoint | `curl http://localhost:8000/health` |
| `/metrics` | GET | Prometheus metrics | `curl http://localhost:8000/metrics` |
| `/docs` | GET | API documentation (Swagger UI) | Open in browser: `http://localhost:8000/docs` |

**Pull the Latest Image**

Pull the latest Docker image. The image requires approximately 8GB of disk space.

```bash
docker pull vllm/vllm-openai:latest
```

**Run the Docker Container**

The Docker image runs an OpenAI-compatible server. To serve a base model like `facebook/opt-125m`, run:

```bash
docker run --runtime nvidia --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 --ipc=host vllm/vllm-openai:latest \
  facebook/opt-125m
```

Here are some small models suitable for learning purposes:

| Model Name | Type | Parameter Size |
|------------|------|----------------|
| `facebook/opt-125m` | Base | 125M |
| `Qwen/Qwen2.5-0.5B-Instruct` | Chat/Instruct | 0.5B |
| `meta-llama/Llama-3.2-1B-Instruct` | Chat/Instruct | 1B |
| `meta-llama/Llama-3.2-1B` | Base | 1B |
| `microsoft/Phi-tiny-MoE-instruct` | MoE/Instruct | ~500M (active) |
| `sentence-transformers/all-MiniLM-L6-v2` | Embedding | 22M |

To use any of these models, replace the model name in the Docker command:

```bash
... vllm/vllm-openai:latest <MODEL_NAME>
```

For example:
```bash
... vllm/vllm-openai:latest meta-llama/Llama-3.2-1B-Instruct
```

The `--runtime nvidia --gpus all` flag enables GPU access. To use specific GPUs, replace `--gpus all` with `--gpus '"device=0"'` for a single GPU or `--gpus '"device=0,1"'` for multiple GPUs. You can also set `--env "CUDA_VISIBLE_DEVICES=0,1"` to limit visible GPUs.

The `-v $HOME/.cache/huggingface:/root/.cache/huggingface` volume mount shares your local Hugging Face cache with the container, avoiding repeated model downloads. The container path `/root/.cache/huggingface` assumes the container runs as root. Adjust this path if your container uses a different user, or set the `HF_HOME` environment variable to customize the cache location.

The `--ipc=host` flag allows the container to access the host's shared memory, which PyTorch uses for efficient data sharing during tensor parallel inference.

The model name is specified as a positional argument after the image tag. You can append additional vLLM engine arguments after the model name.

**Verify the Setup**

Once the container is running, verify it's working correctly. First, check that the server is responding:

```bash
curl http://localhost:8000/health
```

List the available models:

```bash
curl http://localhost:8000/v1/models
```

Test a base model completion:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "facebook/opt-125m", "prompt": "The result of 1+1 is", "max_tokens": 3}'
```

Test a chat model:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

For deterministic output (same result every time), add `"temperature": 0`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0
  }'
```

**Note**: Setting `temperature=0` enables greedy sampling (always selects the highest probability token), which should produce the same output for the same input. However, vLLM does not guarantee complete reproducibility by default due to scheduling and batching optimizations. For fully deterministic results, you may need to set `VLLM_ENABLE_V1_MULTIPROCESSING=0` or enable batch invariance (see vLLM's reproducibility documentation).

Test an embedding model:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "This is a test sentence"
  }'
```

#### Install and Run from Package Manager (uv, conda, pip)

**Method 1: Using uv (Recommended)**

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new Python environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vLLM with automatic PyTorch backend detection
uv pip install vllm --torch-backend=auto
```

The `--torch-backend=auto` flag automatically selects the appropriate PyTorch index based on your CUDA driver version. You can also specify a specific backend (e.g., `--torch-backend=cu126` for CUDA 12.6).

Alternatively, use `uv run` to execute vLLM commands without creating a permanent environment:

```bash
uv run --with vllm vllm --help
```

**Method 2: Using conda**

```bash
# Create a conda environment
conda create -n vllm-env python=3.12 -y
conda activate vllm-env

# Install uv within conda (optional but recommended)
pip install --upgrade uv

# Install vLLM
uv pip install vllm --torch-backend=auto
```

**Method 3: Using pip directly**

```bash
# Create a virtual environment
python3.12 -m venv vllm-env
source vllm-env/bin/activate

# Install vLLM
pip install vllm
```

**Note**: When using pip directly, ensure you have the correct PyTorch version installed for your CUDA version.

**Verifying Installation**

After installation, verify that vLLM is correctly installed:

```bash
# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Test basic functionality
vllm --help
```

#### Compile, Install, and Run from Local Source

To build vLLM from source, clone the repository and install:

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install from source
pip install -e .
```

For development installations with editable mode:

```bash
# Install in development mode
pip install -e ".[dev]"
```

**Note**: Building from source requires all build dependencies and may take longer than package manager installation.

#### Offline Inference

Test your installation with a simple offline inference example:

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-125m")

# Define prompts and sampling parameters
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Generate outputs
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

#### Online Inference

Start an OpenAI-compatible API server:

```bash
# Start the server with a chat model
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

In another terminal, test the server:

```bash
# List available models
curl http://localhost:8000/v1/models

# Test chat completion (for instruction-tuned models)
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }'

# Test completion (for base models)
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "The result of 1+1 is",
        "max_tokens": 3
    }'
```

## KV Cache

### Decoder-Only Transformer Architecture

Modern large language models like GPT, LLaMA, and their variants use decoder-only transformer architectures. Decoder-only transformer is proved to be highly effective for autoregressive language modeling and text generation tasks. These models consist of a stack of identical decoder layers, each containing a self-attention sublayer with causal masking, a feed-forward network (MLP), and residual connections with layer normalization.

The self-attention mechanism uses three learned linear projections: **Query (Q)**, **Key (K)**, and **Value (V)**. For each token at position $i$, the model computes:

- $Q_i = x_i \times W_Q$ (Query vector)
- $K_i = x_i \times W_K$ (Key vector)
- $V_i = x_i \times W_V$ (Value vector)

where $x_i$ is the token embedding (or hidden state) and $W_Q$, $W_K$, $W_V$ are learned weight matrices. The attention scores are computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}}\right) \times V$$

where $d_k$ is the dimension of the key vectors. The causal mask ensures that tokens only attend to previous positions (set to $-\infty$ before softmax), preventing the model from "seeing" future tokens.

A typical architecutre is as follows:

![Decoder-only Transformer](img/decoder_only.png)

Note: This decoder is from the original Transformer papaer, so the `Add&Norm` should be `Norm&Add` for today's LLM.

**Notation**

We use the following notation throughout this section:

- $B$: batch size
- $L_1$: length of an attending sequence (query); $l_1$: index of a position in this sequence
- $L_2$: length of a being attended sequence (key/value); $l_2$: index of a position in this sequence
- $D$: model hidden size (dimension of hidden states/tokens)
- $H$: number of attention heads
- $D_{qk}$: dimension of a query/key vector
- $D_v$: dimension of a value vector, and $D = D_v \times H$
- $L$: notation used when $L_1 = L_2$

### Text Generation Process: Prefill and Decode

When serving generation requests, we typically batch multiple prompts for higher throughput. The input has shape $B \times L_2 \times D$, where $L_2$ is the prompt length.

![Input shape](img/input.png)

Text generation occurs in two distinct phases. The first phase, called **prefill**, processes the entire prompt sequence in parallel to initialize the model's internal state. During prefill, the model processes all prompt tokens simultaneously:

![Prefill Stage](img/prefill.png)

The input $X_0$ (the prompt) after transformer blocks generates $Y_0$, which is the first output token. This token then becomes part of the input for the next generation step.

### The Inefficiency Problem

The second phase is called **Decode**. During the decode phase, the model generates tokens one at a time. Consider generating the first new token after the prompt. The input to the query layer is $X_1 = Y_0$ (the newly generated token), but the input to the Key and Value layers must be a concatenation of the previous prompt $X_0$ and the new token $X_1$, resulting in shape $B \times (L_2 + 1) \times D$.

![Decode without KV Cache](img/decode_without_kvcache.png)

This is necessary because the query needs to attend to the entire context so far. For example, if our prompt $X_0$ is "Time flies" and $Y_0$ is "like", we use "like" to query the context "Time flies like" and predict the next token, probably "an". Then we use "an" to query "Time flies like an" and get "arrow". This process continues: each newly generated token must attend to all previous tokens (both the original prompt and all previously generated tokens) to maintain context and generate coherent text. However, at each step, we need to recompute the Key and Value vectors for the entire sequence history, even though most of these computations were already performed in previous steps.

From above figure, we can easily see the inefficiency arises because we need to construct $\hat{X} = [X_0, X_1]$ and multiply it with $W_K$ and $W_V$, even though __the $X_0$ part has already been multiplied with these weight matrices during the prefill phase__. This duplication occurs at every generation step, requiring recomputation of $Key$ and $Value$ vectors for all previous tokens through the entire transformer stack.

Without caching, this naive approach has time complexity $O(L_{\text{total}}^2)$ per step, where $L_{\text{total}}$ is the total sequence length $L_2 + L_t$ including prompt and generated tokens. For long sequences, this becomes computationally prohibitive.

### KV Cache Solution

KV cache solves this inefficiency by storing precomputed Key and Value vectors for all previously processed tokens. Instead of concatenating and recomputing, we can simply use $X_1$ as input for $K$ and $V$ calculation, as long as we cache the previous results. Take $Key$ vector as an example, we only calucate $K_{new}$ which has shape of $B1D_k$ and the time complexity reduced dramatically.

![Key Cache Grows](img/cache_grow.png)

With KV cache, the decoding stage becomes much more efficient.

The input shape is $B \times 1 \times D$ and output shape is also $B \times 1 \times D$. The key insight is that only the new token needs attention computation, while all previous tokens reuse their cached $K$ and $V$ vectors.

**Time Complexity Analysis:**

With KV cache, the computational complexity changes dramatically:

- **Prefill phase**: Processes all $L_2$ prompt tokens in parallel. The attention computation has complexity $O(L_2^2 \cdot D)$ for the self-attention over the prompt sequence. This is a one-time cost.

- **Decode phase (per token)**: 
  - **Without KV cache**: $O(L_{\text{total}}^2 \cdot D)$ per step, where $L_{\text{total}} = L_2 + L_t$ grows with each generated token. For $T$ generated tokens, total complexity is $O(T \cdot L_{\text{total}}^2 \cdot D)$, which becomes $O(T^3 \cdot D)$ when $L_t \gg L_2$.
  
  - **With KV cache**: Only compute $K$ and $V$ for the new token (shape $B \times 1 \times D_k$), then perform attention with cached keys/values. The complexity per step is $O(L_{\text{total}} \cdot D)$ for the attention computation, where $L_{\text{total}}$ is the current sequence length. For $T$ generated tokens, total complexity is $O(T \cdot L_{\text{total}} \cdot D) \approx O(T^2 \cdot D)$ when $L_t \gg L_2$.

The key improvement is reducing the quadratic dependency on sequence length in the decode phase to linear, making long-sequence generation feasible. However, this comes at the cost of memory: KV cache requires $O(L_{\text{total}} \cdot D)$ memory to store all cached Key and Value vectors.

![Decode with KV Cache](img/decode_with_kvcache.png)

**Summary of Inference Stages**

The following table summarizes the tensor shapes and operations during prefill and decoding phases:

| Stage | Shape | Notes |
|-------|-------|-------|
| **Prefill Phase** | | |
| Input $X_{\text{prompt}}$ | $B \times L_2 \times D$ | Prompt sequence |
| Process all tokens | Parallel | Same as training |
| Cache $K$, $V$ | $B \times L_2 \times D_{qk/v}$ | For reuse |
| **Decoding Phase** | | |
| Input $x_t$ | $B \times 1 \times D$ | Single new token |
| $Q_t = x_t W^Q$ | $B \times 1 \times D_{qk}$ | Query for new token |
| $K_{\text{cached}}$ | $B \times (L_2 + t) \times D_{qk}$ | All previous keys |
| $V_{\text{cached}}$ | $B \times (L_2 + t) \times D_v$ | All previous values |
| Attention scores | $B \times 1 \times (L_2 + t)$ | New token attends to all previous |
| Attention $A_t$ | $B \times 1 \times (L_2 + t)$ | Upper triangular |
| Output $Z_t$ | $B \times 1 \times D_v$ | |
| Concat heads | $B \times 1 \times (H \cdot D_v)$ | |
| Final output | $B \times 1 \times D$ | Single generated token |

## PagedAttention: Solving KV Cache Fragmentation

While KV cache dramatically improves computational efficiency, it introduces a critical challenge in production serving systems: **memory fragmentation**. When serving multiple concurrent requests, each with different sequence lengths that grow dynamically, traditional memory allocation strategies lead to significant waste and limit throughput.

vLLM's breakthrough innovation is **PagedAttention**, a memory management algorithm inspired by virtual memory paging in operating systems. PagedAttention is primarily designed to solve KV cache fragmentation in large-scale LLM serving by allocating the cache in fixed-size blocks. A crucial consequence of this block-based design is that attention computation no longer iterates over padded sequence ranges. Instead, it traverses only the blocks that actually exist for each request, thereby eliminating padding-related attention FLOPs.

### The KV Cache Fragmentation Problem

In production environments, serving systems must handle multiple concurrent requests simultaneously. Each request maintains its own KV cache that grows dynamically as tokens are generated during autoregressive decoding. Different prompts and generation lengths result in different cache sizes, creating a fundamental allocation challenge.

Traditional systems allocate contiguous memory blocks per request. When sequences finish or have different lengths, this approach leads to wasted space that cannot be efficiently reused. Consider a scenario where Request 1 finishes after 8 tokens, Request 2 is active with 4 tokens, and Request 3 is active with 10 tokens. The memory allocated for Request 1 sits unused but cannot be easily reclaimed for the other requests, leading to fragmentation.

```
Request 1: [========]  (8 tokens, finished)
Request 2: [====]      (4 tokens, active)
Request 3: [==========] (10 tokens, active)
           ↑ Memory fragmentation - can't reuse Request 1's space efficiently
```

Additionally, traditional batching introduces a less obvious but equally critical inefficiency: padding-induced attention computation waste. In batched decoding, different requests typically have different effective context lengths. Let the batch size be $B$, and let the cached context length for request $i$ be $(L_2^{(i)} + t^{(i)})$. To batch these requests together, conventional attention implementations must pad all sequences to a common maximum length:

$$(L_2 + t)_{\max} = \max_i (L_2^{(i)} + t^{(i)})$$

As a result, the cached keys and values used in attention have shape $K_{\text{cached}}, V_{\text{cached}} \in \mathbb{R}^{B \times (L_2 + t)_{\max} \times D_{qk/v}}$, and the attention scores for the decode step are computed as:

$$A_t = \text{softmax}\left(Q_t K_{\text{cached}}^T + \text{mask}\right), \qquad A_t \in \mathbb{R}^{B \times 1 \times (L_2 + t)_{\max}}$$

Although masking prevents padded positions from influencing the output, the dot products involving padded tokens are still fully computed. Consequently, a large fraction of attention FLOPs is spent on tokens that carry no semantic information, especially when the context lengths within a batch vary widely.

![PA](img/pa.svg)

### How PagedAttention Works

To solve the fragmentation problem, PagedAttention must use fixed-size blocks. Continuous memory allocation inevitably leads to fragmentation when sequences have different lengths and finish at different times. The only viable approach is to partition the KV cache into fixed-size blocks that can be allocated and freed independently.

PagedAttention divides the KV cache into fixed-size blocks, typically 16 tokens per block (denoted as $B_{\text{size}}$), similar to memory pages in OS virtual memory. Each block contains the Key and Value vectors for a contiguous segment of tokens. For request $i$, the cached KV is represented as $N_i = \lceil (L_2^{(i)} + t^{(i)}) / B_{\text{size}} \rceil$ blocks, each block storing keys and values with shape $\text{Block} \in \mathbb{R}^{B_{\text{size}} \times D_{qk/v}}$.

Each request maintains a block table that maps logical sequence positions to physical block addresses, similar to page tables in OS virtual memory. This allows sequences to be logically continuous while physically stored in discrete blocks scattered across GPU memory. Blocks are allocated and freed as sequences grow or complete. When a request finishes, its blocks are immediately returned to a shared block pool. Freed blocks can be immediately reused by new requests, eliminating fragmentation and enabling near-100% memory utilization.

```
Block Pool: [Block0][Block1][Block2][Block3][Block4][Block5]...
            ↓        ↓        ↓
Request 1:  [Block0][Block1]  (finished, blocks returned to pool)
Request 2:  [Block2]          (active)
Request 3:  [Block3][Block4]   (active)
            ↑ No fragmentation - blocks can be reused immediately
```

### Eliminating Padding FLOPs

Once blocks exist, the iteration semantics of attention computation fundamentally change. This is a crucial consequence of the block-based design, not an additional optimization. With block-based storage, attention computation no longer assumes KV cache is a continuous sequence from position 1 to $(L_2 + t)_{\max}$. Instead, attention iterates over the block table, and tokens that do not exist simply do not have corresponding blocks.

During decoding, the attention computation for request $i$ iterates only over the blocks listed in its block table:

$$A_t^{(i)} = \text{softmax}\left(Q_t^{(i)} \cdot \bigcup_{b \in \mathcal{B}_i} K_b^T\right)$$

where $\mathcal{B}_i$ denotes the set of blocks owned by request $i$. Crucially, this computation depends only on $(L_2^{(i)} + t^{(i)})$, not on $(L_2 + t)_{\max}$. Tokens that do not exist for a given request are never visited by the attention kernel because the corresponding blocks do not exist in the block table.

With PagedAttention, padding tokens are not masked after computation—they are never part of the computation. The attention kernel does not launch dot products for padded positions. As a result, the number of attention FLOPs for each request is proportional to its true context length rather than the maximum context length in the batch. This property eliminates padding-related FLOPs entirely and is a key enabler for efficient continuous batching in large-scale LLM serving systems.

The performance improvement in vLLM comes from both effects working together. Fragmentation reduction allows more concurrent requests to be served simultaneously, while zero padding FLOPs increases the effective compute utilization of decode attention. These are two sides of the same block-based design decision.

Unlike OS page management that focuses on address mapping and access correctness, PagedAttention is optimized for efficient attention computation. Custom CUDA kernels read KV cache from non-contiguous blocks while maintaining coalesced memory access patterns, ensuring high GPU utilization. This attention-aware design means the system understands how attention operations access memory and optimizes accordingly.

The benefits are substantial. Memory efficiency improves dramatically because fragmentation waste from variable-length sequences is eliminated, enabling near-100% memory utilization. Production deployments can serve 2-4x more concurrent requests with the same GPU memory compared to traditional approaches. More importantly, padding overhead is completely eliminated, meaning zero FLOPs are wasted on padding. This dramatically improves throughput in real serving scenarios.

Flexible batching becomes possible because requests with different sequence lengths can be batched efficiently without padding. The system supports dynamic batching where batch composition changes every step, enabling efficient serving of highly variable workloads. Long context support is also enhanced, as the system efficiently handles variable-length contexts without pre-allocating maximum memory. Very long sequences of 100K+ tokens can be served by allocating blocks on-demand. The design handles high concurrency naturally, built for high-churn workloads with frequent request arrivals and completions.

### Connection to Distributed Inference

While PagedAttention solves memory efficiency within a single GPU, **distributed inference techniques** (tensor parallelism, data parallelism, pipeline parallelism) are needed when:

- Models exceed single GPU memory capacity
- Throughput requirements exceed single GPU capabilities
- Models are too large for a single node

The combination of PagedAttention (memory efficiency) and distributed parallelism (scalability) enables vLLM to serve the largest models efficiently in production environments.

## Motivation: The "Out of Memory" Problem

As language models grow larger—reaching 400 billion parameters (like DeepSeek R1) or even 600 billion parameters—they exceed the memory capacity of a single GPU. Even with modern H100 GPUs, models of this scale cannot fit into a single device.

One approach to address this is **quantization**—reducing precision from FP16 to FP8 or even lower bit widths. While this helps, it has limitations:

1. **Accuracy trade-offs**: Lower precision can affect model accuracy, though modern quantization techniques have mitigated this significantly.
2. **Limited scalability**: Even with FP8 (half a byte per parameter), a 400B parameter model still requires 200GB of memory, which exceeds single GPU capacity.

The solution is to **distribute the model across multiple GPUs** using parallelism techniques. This provides much better scalability than quantization alone.

## Overview of the vLLM Architecture

vLLM's architecture centers around a **scheduler-executor-worker** pattern:

```
┌─────────────┐
│  Scheduler  │ ← Schedules requests
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  Executor   │ ← Manages workers, issues distributed commands
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Workers   │ ← Associated with accelerators (GPUs)
└─────────────┘
```

**Key Components**:

- **Scheduler**: Groups and schedules incoming requests
- **Executor**: Manages workers and coordinates distributed inference
  - Supports multiple backends: Ray, multi-processing, single GPU
  - Issues distributed inference commands to all workers
  - Returns results to the scheduler
- **Workers**: Execute computation on their associated accelerators

This architecture enables the distributed execution of inference across multiple GPUs and nodes.

## Overview of Parallelism Strategies in vLLM

![Parallelism strategies in vLLM](img/parallelism_strategies_overview.svg)

vLLM provides three fundamental parallelism strategies for distributing computation and memory across multiple GPUs:

1. **Tensor Parallelism (TP)**: Shards individual layers across multiple GPUs within a node. Each GPU processes a portion of each layer, with results synchronized through collective communication operations.

2. **Data Parallelism (DP)**: Creates multiple complete replicas of the model, each processing different requests independently. This increases throughput by handling multiple requests simultaneously.

3. **Pipeline Parallelism (PP)**: Splits the model's layers across multiple GPUs or nodes, with each GPU processing different layers sequentially. Data flows through these stages like an assembly line.

Additionally, vLLM provides **Expert Parallelism (EP)** as a special modifier flag for Mixture-of-Experts (MoE) models. EP is not a standalone strategy—it modifies how MoE layers are distributed and must be combined with TP or DP. The `--enable-expert-parallel` flag changes communication patterns and expert distribution for MoE models.

The following sections explore each strategy in detail, then discuss how they can be combined for optimal performance.

## Tensor Parallelism (TP)

**Tensor Parallelism (TP)** is a technique that shards model weights horizontally across multiple GPUs within a single node, allowing all GPUs to compute concurrently. This follows an **SPMD (Single Program, Multiple Data)** paradigm.

### Why Tensor Parallelism?

When models get larger, one GPU is insufficient. Even within a single node, you might have multiple GPUs available. The question is: how do we parallelize computation across these GPUs?

**Answer**: Shard weights horizontally and compute everything concurrently at the same time.

### Visual Example

For a model with 4 layers (for illustration):

- Each GPU gets a piece of each layer
- All GPUs work simultaneously on different parts of the computation
- Communication operations synchronize results between GPUs

### Linear Algebra Behind TP: Column & Row Parallelism

Tensor parallelism is built on two fundamental linear algebra operations:

### Column Parallelism

Splits the second matrix along its **columns**:

```
Y = X × A

Split A into [A₁ | A₂] (column-wise)
Then: Y = [X × A₁ | X × A₂]
```

- Each GPU computes one piece of the result
- Use **all-gather** operation to concatenate the pieces
- Result: Full output vector on all GPUs

### Row Parallelism

Splits the first vector and the matrix along its **rows**:

```
Y = X × A

Split X into [X₁; X₂] and A into [A₁; A₂] (row-wise)
Then: Y = X₁ × A₁ + X₂ × A₂
```

- Each GPU computes a partial result
- Use **all-reduce** operation to sum the partial results
- Result: Final output vector on all GPUs

### Communication Operations

- **All-gather**: Concatenates results from multiple GPUs
- **All-reduce**: Sums partial results from multiple GPUs

### Applying TP to a Multi-Layer Perceptron (MLP)

An MLP layer (as found in LLaMA) consists of:

```
Input → Up Projection → Activation → Down Projection → Output
```

**Step-by-step TP application**:

1. **Up Projection (Column Parallel)**:
   - Split weight matrix A along columns
   - Each GPU computes: `X × A₁` and `X × A₂`
   - Results are naturally sharded (no all-gather needed yet)

2. **Activation**:
   - Element-wise operation (e.g., SiLU, ReLU)
   - Operates on already-sharded data
   - Output remains sharded

3. **Down Projection (Row Parallel)**:
   - Split weight matrix B along rows
   - Each GPU computes partial result: `Activation₁ × B₁`
   - Use **all-reduce** to sum partial results
   - Final result available on all GPUs

**Key Insight**: Within an MLP, we can avoid the all-gather after column parallel because the next operation (row parallel) needs sharded data anyway. We only need all-reduce at the end.

### Scaling to More GPUs

To use more GPUs, simply shard each matrix into more sections. For N GPUs:

- Column parallel: Split into N columns
- Row parallel: Split into N rows

### Benefits of Tensor Parallelism

### 1. Weight Space Reduction

**Benefit**: Each GPU stores only a fraction of the weights.

**Example**: 

- 140B parameter model
- With TP=2: Each GPU stores ~70B parameters
- **Result**: Model can now fit on GPUs that couldn't hold the full model

### 2. KV Cache Space Increase

**Benefit**: More space available for KV cache per GPU.

**Example**:

- Single GPU: 160GB total, 140GB for weights → 20GB for KV cache
- With TP=2: Each GPU has 160GB, 70GB for weights → 90GB for KV cache
- **Result**: Super-linear increase in KV cache capacity

**Important**: Even when a model *can* fit on one or two GPUs, expanding to more GPUs can dramatically increase throughput by providing more KV cache space. This requires careful calculation of available memory.

### 3. Latency Reduction

**Benefit**: Faster computation and memory bandwidth utilization.

**Mechanism**:

- Each GPU loads fewer weights from HBM to compute
- Effectively doubles (or multiplies) memory bandwidth
- Prefill operations (often memory-bound) benefit significantly

**Trade-off**: Communication overhead between GPUs

### 4. Communication Cost

**Data transferred per layer**:

- Size: `batch_size × sequence_length × hidden_size`
- Occurs for both MLP and attention layers
- Repeated for every layer in the model

**Mitigation**: Good communication hardware (e.g., NVLink within a node) reduces this overhead.

### Trade-offs of Tensor Parallelism

#### Advantages

1. **Improves end-to-end latency**: By splitting weights, each GPU has less to load and compute
2. **Reduces memory pressure**: Enables larger models and more KV cache
3. **Simple implementation**: SPMD paradigm is straightforward

#### Disadvantages

1. **High communication overhead**: 
   - All-reduce operations for every layer
   - Can be 60%+ of time in prefill-heavy workloads
   - Especially problematic without NVLink (e.g., L4 GPUs over PCIe)

2. **Hardware requirements**:
   - Works best with NVLink within a node
   - Poor performance with PCIe interconnect for prefill-heavy workloads

3. **Constraint**: Attention heads must be divisible by tensor parallel size (or use padding)

#### When to Use TP

- **Good**: Models that don't fit on a single GPU, good interconnect (NVLink)
- **Caution**: Prefill-heavy workloads with poor interconnect
- **Best practice**: Profile your workload to understand communication vs. computation ratio






## Data Parallelism (DP) for Throughput Scaling

**Data Parallelism (DP)** is a technique that replicates model weights across separate instances or GPUs to process independent batches of requests. Unlike tensor parallelism (which splits a model across GPUs) or pipeline parallelism (which splits layers across GPUs), data parallelism creates multiple complete copies of the model, each handling different requests.

### Why Data Parallelism?

When a model fits on a single GPU (or a small TP group), but you need to serve more concurrent requests than a single replica can handle, data parallelism allows you to scale throughput horizontally by adding more model replicas.

**Key Use Cases**:

- **Throughput scaling**: Serve more requests simultaneously by adding replicas
- **Load balancing**: Distribute requests across multiple independent model instances
- **MoE models**: Combine DP attention layers with EP/TP expert layers for optimal MoE performance

### How Data Parallelism Works

In data parallelism:

- Each DP rank maintains a **complete copy** of the model weights
- Each rank processes **independent batches** of requests
- Each rank has its **own independent KV cache**
- Requests are distributed across ranks (load balancing)

```
DP Rank 0: [Complete Model] → Processes Batch 1
DP Rank 1: [Complete Model] → Processes Batch 2
DP Rank 2: [Complete Model] → Processes Batch 3
DP Rank 3: [Complete Model] → Processes Batch 4
```

### Data Parallelism vs. Other Parallelism Strategies

| Strategy | Model Replication | Communication | Use Case |
|----------|-------------------|---------------|----------|
| **Data Parallel (DP)** | Full model on each rank | None (independent) | Throughput scaling, load balancing |
| **Tensor Parallel (TP)** | Split weights horizontally | All-reduce per layer | Model too large for single GPU |
| **Pipeline Parallel (PP)** | Split layers vertically | Sequential data transfer | Model too large for single node |
| **Expert Parallel (EP)** | Split experts | All-to-all for routing | MoE models |

### Combining Data Parallelism with Other Strategies

Data parallelism can be combined with tensor parallelism and expert parallelism:

#### DP + TP

When using both DP and TP:

- Each DP rank contains a TP group
- Total GPUs = `DP_size × TP_size`
- Example: `DP=4, TP=2` requires 8 GPUs (4 replicas, each with 2-GPU TP)

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

#### DP + EP (for MoE Models)

For MoE models, data parallelism is particularly powerful:

- **Attention layers**: Use data parallel (each rank has full attention)
- **Expert layers**: Use expert parallel or tensor parallel
- Expert layers form a `(DP × TP)` sized group for synchronization

**Important**: For MoE models with DP, forward passes must be aligned across all ranks. Even if a rank has no requests, it must perform "dummy" forward passes to maintain synchronization with expert layers.

### Deployment Modes

vLLM supports two deployment modes for data parallelism:

#### 1. Internal Load Balancing (Self-Contained)

A single API endpoint with internal load balancing:

```bash
# Single node: DP=4, TP=2 (8 GPUs total)
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

**Multi-node example**:
```bash
# Node 0 (head node with API server)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345

# Node 1 (worker node)
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-start-rank 2 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

**Benefits**:

- Single HTTP endpoint
- Automatic load balancing based on queue lengths
- Simpler deployment

**Limitations**:

- API server can become a bottleneck at large DP sizes
- Use `--api-server-count` to scale out API servers

#### 2. External Load Balancing

Each DP rank is deployed as a separate vLLM instance with its own endpoint:

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 --port 8000

# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 --port 8001
```

An external load balancer (e.g., nginx, HAProxy) routes requests to different ranks based on:

- Real-time telemetry (queue lengths, KV cache usage)
- Request characteristics (prefix caching opportunities)
- Health status

**Benefits**:

- Better scalability for large DP deployments
- More sophisticated load balancing (KV cache aware)
- Independent scaling of each rank

### Benefits of Data Parallelism

1. **Linear throughput scaling**: Adding more replicas increases throughput proportionally
2. **Independent KV caches**: Each rank maintains its own KV cache, maximizing total cache capacity
3. **Fault tolerance**: If one rank fails, others continue serving
4. **Prefix caching optimization**: Load balancer can route requests with common prefixes to the same rank
5. **No communication overhead**: Unlike TP/PP, DP ranks operate independently

### Trade-offs of Data Parallelism

#### Advantages

1. **Simple and effective**: Easy to understand and deploy
2. **No communication overhead**: Ranks operate independently
3. **Linear scaling**: Throughput scales with number of replicas
4. **Flexible load balancing**: Can optimize routing based on cache state

#### Disadvantages

1. **Memory overhead**: Each rank stores full model weights (or TP-sharded weights)
2. **MoE synchronization**: For MoE models, ranks must synchronize even when idle
3. **Load balancing complexity**: External mode requires additional infrastructure
4. **Not for large models**: If model doesn't fit on a single GPU/TP group, DP alone won't help

### When to Use Data Parallelism

- **Good**: 
  - Model fits on single GPU (or small TP group)
  - Need to serve more concurrent requests
  - Throughput is the primary concern
  - MoE models (combine DP attention with EP experts)

- **Not suitable**:
  - Model too large for single GPU/TP group (use TP/PP first)
  - Latency-sensitive workloads (TP may be better)
  - Limited GPU memory (DP replicates weights)

### Best Practices

1. **Start with TP/PP**: Use TP/PP to fit the model, then add DP for throughput
2. **Profile load balancing**: For external mode, monitor and optimize routing
3. **Consider prefix caching**: Route requests with common prefixes to same rank
4. **Monitor KV cache**: Balance requests based on available KV cache per rank
5. **Use Ray backend**: For multi-node DP, Ray simplifies deployment

### vLLM Data Parallelism Source Code

The data parallelism implementation in vLLM is distributed across several key files:

**Core Implementation**:
- `vllm/v1/engine/core.py` (lines 1139-1457): 
  - `DPEngineCoreProc`: Main data parallel engine core process class
  - `DPEngineCoreActor`: Ray actor version for distributed execution
  - Handles DP rank initialization and step synchronization

**Parallel State Management**:
- `vllm/distributed/parallel_state.py` (line 1102+):
  - `get_dp_group()`: Returns the data parallel process group
  - Initializes DP groups and manages DP ranks

**DP Coordination and Synchronization**:
- `vllm/v1/worker/dp_utils.py`:
  - `coordinate_batch_across_dp()`: Coordinates batch processing across DP ranks
  - `_synchronize_dp_ranks()`: Synchronizes token counts and microbatching decisions
  - Handles DP padding and ubatch coordination via all-reduce operations

**DP Coordinator**:
- `vllm/v1/engine/coordinator.py` (line 22+):
  - `DPCoordinator`: Coordinates multiple DP engine ranks
  - Manages request waves, load balancing stats, and START_DP_WAVE messages
  - Collects statistics from DP engines for load balancing

**Worker Integration**:
- `vllm/v1/worker/gpu_worker.py`: GPU worker with DP support
- `vllm/v1/worker/gpu_model_runner.py`: Model runner with DP batch coordination
- `vllm/v1/worker/gpu/dp_utils.py`: GPU-specific DP utilities

**Configuration**:
- `vllm/config/parallel.py`:
  - `ParallelConfig` class with `data_parallel_size`, `data_parallel_rank`, etc.

**Examples**:
- `examples/offline_inference/data_parallel.py`: Offline batch inference example
- `examples/online_serving/multi_instance_data_parallel.py`: Online serving example

## Pipeline Parallelism (PP)

**Pipeline Parallelism (PP)** shards the model across multiple nodes when models are too large for a single node (e.g., DeepSeek R1, LLaMA 405B). 

**Note**: While PP can technically work on a single node, it's not the recommended approach:

- **Single node**: Use **Tensor Parallelism (TP)** when the model doesn't fit on one GPU
- **Multi-node**: Use **TP within nodes + PP across nodes** for very large models

PP is primarily designed for multi-node deployments where you need to distribute layers across nodes.

### How Pipeline Parallelism Works

Unlike tensor parallelism (which splits layers horizontally), pipeline parallelism **splits the model along layers**:

- Each GPU holds a **group of consecutive layers**

- Data flows sequentially through GPUs
- Each GPU computes its layers, then sends results to the next GPU

### Implementation in vLLM

```python
# Simplified concept
for layer_group in my_layers:
    if not first_stage:
        data = receive_from_previous_gpu()
    result = compute_layers(layer_group, data)
    if not last_stage:
        send_to_next_gpu(result)
```

**Key differences from TP**:

- **TP**: All GPUs work simultaneously (SPMD)
- **PP**: GPUs work sequentially (MPMD - Multiple Program, Multiple Data)
- **TP**: High communication, low latency
- **PP**: Low communication, but doesn't improve latency

### Communication Pattern

- **Data size**: `batch_size × sequence_length × hidden_size`
- **Frequency**: Once per layer group (much less than TP)
- **Trade-off**: Lower communication overhead, but sequential execution

### Solving Pipeline Bubbles with Request Groups

### The Pipeline Bubble Problem

In naive pipeline parallelism, GPUs sit idle between batches:

```
GPU 0: [====]     [====]     [====]
GPU 1:     [====]     [====]     [====]
GPU 2:         [====]     [====]     [====]
```

**Problem**: Each GPU is idle most of the time—huge waste of resources.

### Solution: Request Groups (Virtual Engines)

vLLM uses **request groups** (also called virtual engines) to keep all GPUs busy:

```
GPU 0: [Group1][Group2][Group3][Group4][Group1][Group2]...
GPU 1: [Group1][Group2][Group3][Group4][Group1][Group2]...
GPU 2: [Group1][Group2][Group3][Group4][Group1][Group2]...
```

**How it works**:
- Multiple independent request streams (groups) run simultaneously
- Each group is data-independent
- vLLM uses multiple schedulers and cache engines to maintain separation
- Locks ensure only one request group operates on each GPU at a time

### Trade-offs

1. **KV Cache Splitting**: KV cache is divided among request groups
   - Each group gets a fraction (e.g., 1/4 for 4 PP stages)
   - Supports smaller batch sizes per group

2. **Micro-batches**: Each request group acts as a micro-batch
   - All groups together form the full batch
   - Can reduce decode efficiency (memory-bound operations benefit from larger batches)

3. **Load Balancing**: vLLM balances by splitting KV cache evenly and routing requests to schedulers with most available KV cache

### Optimizing Pipelines with Chunked Prefill

### The Prefill vs. Decode Problem

- **Prefill**: Long sequences, compute-intensive, can take significant time
- **Decode**: Short sequences, memory-bound, fast

**Problem**: Long prefill can create bubbles in the pipeline when decode is much faster.

### Solution: Chunked Prefill

**Chunked prefill** amortizes the cost of prefill by:

1. Processing only a chunk of the prefill initially
2. Interleaving the remaining prefill with subsequent decode steps
3. Spreading prefill computation across multiple decode iterations

**Visual Example**:

Without chunked prefill:
```
Prefill: [==============]
Decode:  [=][=][=][=][=]
         ↑ Bubble here
```

With chunked prefill:
```
Prefill: [==][==][==][==][==]  (chunked)
Decode:  [=][=][=][=][=][=][=]
         ↑ Smooth, no bubbles
```

### Benefits

1. **Eliminates bubbles**: Smooth pipeline execution
2. **More KV cache space**: By limiting maximum batch size, vLLM can infer more available KV cache
3. **Better concurrency**: Prevents arbitrary large prefill from consuming all memory

### Choosing Chunk Size

**Important**: Default chunk size may not be optimal for your workload.

**Example** (LLaMA 13B on 2×L4 GPUs):
- **Large chunk size**: Creates bubbles, ~20% performance loss
- **Small chunk size**: Smooth execution, optimal performance

**Considerations**:
- Prefill-to-decode ratio
- Hardware characteristics
- Workload patterns

**Note**: Chunked prefill is enabled by default in vLLM v1.

## Expert Parallelism (EP): A Modifier Flag for MoE Models

**Expert Parallelism (EP)** is not a standalone parallelism strategy. Instead, it is a modifier flag (`--enable-expert-parallel`) that changes how MoE (Mixture-of-Experts) models distribute experts and communicate across GPUs. EP must be combined with TP or DP—it cannot be used alone.

**Critical constraint**: The EP flag only takes effect when `TP_SIZE × DP_SIZE > 1`. If both TP and DP are set to 1, the EP flag is ignored.

This section uses Phi-tiny-MoE-instruct (referred to as Phi-tiny) to illustrate how MoE architectures work and how the EP flag modifies parallelism behavior.

### Understanding MoE Architecture

In MoE models, the standard feed-forward network (FFN) is replaced with a **Mixture-of-Experts** layer that contains multiple expert networks. Each token is routed to a subset of experts (typically top-2) based on a learned routing mechanism.

![MOE Architecture](img/moe_arch.svg)

#### Decoder Layer Structure

In Phi-tiny, each decoder layer follows the standard Transformer architecture with attention and MoE components:

```python
class PhiMoEDecoderLayer(nn.Module):
    def __init__(self, config: PhiMoEConfig, layer_idx: int):
        super().__init__()
        self.self_attn = PHIMOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.block_sparse_moe = PhiMoESparseMoeBlock(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                past_key_value=None, output_attentions=False, 
                output_router_logits=False, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states, self_attn_weights if output_attentions else None,
                present_key_value if use_cache else None, router_logits if output_router_logits else None)
```

The MoE layer (`block_sparse_moe`) is invoked after the post-attention layer normalization, replacing the standard FFN with a routing mechanism that selects and combines outputs from multiple experts.

#### Expert Structure

Each expert in Phi-tiny is a complete MLP with three linear projections:

```python
class PhiMoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: PhiMoEConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
```

The expert uses a gated activation pattern: `w1` projects to the intermediate dimension and is activated, `w3` provides a gating signal, and `w2` projects back to the hidden dimension. This structure is similar to standard Transformer FFNs but with multiple specialized experts.

#### Routing Mechanism

The `PhiMoESparseMoeBlock` implements the routing logic that selects which experts process each token:

```python
class PhiMoESparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            PhiMoEBlockSparseTop2MLP(config) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = sparsemixer(
            router_logits, top_k=2, jitter_eps=self.router_jitter_noise, 
            training=self.training)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, device=hidden_states.device)
        
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            
            current_state = hidden_states[None, top_x.tolist()].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * \
                routing_weights[top_x.tolist(), idx.tolist(), None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits
```

The routing process works as follows:

1. **Gate computation**: A linear layer (`gate`) computes routing logits for each token over all experts.
2. **Expert selection**: The `sparsemixer` function selects the top-2 experts for each token and computes routing weights.
3. **Expert processing**: Each selected expert processes its assigned tokens, with outputs weighted by routing weights.
4. **Aggregation**: Expert outputs are aggregated using `index_add_` to combine contributions from multiple experts per token.

### How Expert Parallelism Modifies Behavior

The EP flag changes two key aspects of MoE model execution:

1. **Expert Distribution**:
   - **Without EP**: All experts are present on every GPU, but their weight tensors are sharded across GPUs (via `flatten_tp_across_dp`).
   - **With EP**: Experts are distributed across GPUs, with each GPU holding a different subset of complete experts (via `determine_expert_map`).

2. **Communication Pattern**:
   - **TP + EP**: Uses AllReduce communication (same as TP without EP, since `dp_size=1`).
   - **DP + EP**: Uses AllToAll communication, enabling DP Attention with partitioned KV cache.

### Expert Distribution Formula

For routed experts, the distribution follows:

```
EP_SIZE = TP_SIZE × DP_SIZE
Routed experts per GPU = Total Routed Experts / EP_SIZE
```

**Example**: DeepSeek-R1 has 256 routed experts:
- With `TP=8, DP=1, EP`: Each GPU holds 32 complete experts (256/8 = 32)
- With `TP=1, DP=8, EP`: Each GPU holds 32 complete experts (256/8 = 32)
- With `TP=4, DP=2, EP`: Each GPU holds 32 complete experts (256/8 = 32)

### When to Use Expert Parallelism

The EP flag provides benefits when:

1. **High expert activation density** (>3%): AllToAll communication overhead is offset by memory bandwidth gains.
2. **MLA/MQA models** (DeepSeek V2/V3/R1): EP with DP is essential for proper KV cache partitioning.
3. **Memory bandwidth is the bottleneck**: EP distributes experts to leverage aggregate memory bandwidth across GPUs.

**Note**: For ultra-sparse models (<1% activation density), EP may add overhead. The EP flag requires additional dependencies (DeepEP, pplx-kernels, DeepGEMM) and may not be fully stable for all model/quantization/hardware combinations. See the [vLLM Expert Parallel Deployment documentation](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) for details.

## Combining Parallelism Strategies

vLLM allows combining multiple parallelism strategies to efficiently distribute models across GPUs. Understanding which combinations work and their constraints is crucial for successful deployment.

### TP + PP: Tensor and Pipeline Parallelism

Since TP and PP operate along **different axes**, they can be combined effectively:

**Typical Configuration**:
- **Pipeline Parallelism**: Across nodes (where interconnect is slower)
- **Tensor Parallelism**: Within nodes (where NVLink provides fast communication)

**Benefits**:
1. **Reduced communication in PP**: Each GPU only sends its TP chunk between pipeline stages
   - Data size: `batch_size × sequence_length × hidden_size / TP_size`
   - Smaller transfers between pipeline stages

2. **Flexibility**: Can adapt to hardware constraints and interconnect characteristics

**Example**:
```bash
--tensor-parallel-size 4    # TP within each node
--pipeline-parallel-size 8  # PP across 8 nodes
```

### TP + EP: Tensor Parallelism with Expert Parallelism

When combining TP with EP for MoE models:

**Behavior**:
- Experts are distributed across TP ranks (split experts)
- Uses AllReduce communication (not AllToAll, since `dp_size=1`)
- KV cache is duplicated on each TP rank (same as TP without EP)

**Use case**: Large MoE models that don't fit on a single GPU, low-moderate concurrency workloads.

**Example**:
```bash
--tensor-parallel-size 8 --enable-expert-parallel
```

**Note**: For MLA/MQA models (DeepSeek), TP+EP has limited benefits due to KV cache duplication. Consider DP+EP instead for better memory efficiency.

### DP + EP: Data Parallelism with Expert Parallelism

When combining DP with EP for MoE models:

**Behavior**:
- Enables **DP Attention**: Request-level parallelism with partitioned KV cache
- Experts are distributed across DP ranks
- Uses AllToAll communication (requires `dp_size > 1`)
- KV cache is partitioned across GPUs (each GPU holds cache for its assigned requests)

**Use case**: 
- Essential for MLA/MQA models (DeepSeek) to avoid KV cache duplication
- High concurrency workloads where throughput matters
- When TP choices are not compatible (non-power-of-2 GPU counts)

**Example**:
```bash
--data-parallel-size 8 --enable-expert-parallel
```

**Critical**: Using `--data-parallel-size` alone (without EP) for MoE models uses traditional DP with sharded experts, not DP Attention. The EP flag is required to enable DP Attention behavior.

### TP + DP: Tensor and Data Parallelism

When combining TP with DP:

**Behavior**:
- Each DP rank contains a TP group
- Total GPUs = `DP_size × TP_size`
- Non-MoE layers: TP-sharded within each DP rank
- MoE layers: Behavior depends on EP flag

**Use case**: Large models that need both model sharding (TP) and throughput scaling (DP).

**Example**:
```bash
--tensor-parallel-size 4 --data-parallel-size 2  # 8 GPUs total
```

### TP + DP + EP: Combined Strategies for MoE Models

For MoE models, you can combine all three:

**Behavior**:
- EP_SIZE = TP_SIZE × DP_SIZE
- Experts distributed across all GPUs in the combined group
- Communication: AllToAll (since `dp_size > 1`)

**Example**:
```bash
--tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel
# EP_SIZE = 4 × 2 = 8, experts distributed across 8 GPUs
```

### PP + EP: Pipeline Parallelism with Expert Parallelism

**Critical constraint**: EP only activates if `TP_SIZE × DP_SIZE > 1` within each pipeline stage.

**Limitations**:
- `--pipeline-parallel-size 2 --enable-expert-parallel` → EP does NOT activate (TP=1, DP=1 per stage)
- `--pipeline-parallel-size 2 --tensor-parallel-size 4 --enable-expert-parallel` → EP activates (TP=4 per stage)
- Requires AITER (Advanced Inter-node Tensor-parallelism Engine Runtime) for stability

**Example**:
```bash
VLLM_ROCM_USE_AITER=1 vllm serve model-name \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel
```

### Expert Parallelism Activation Constraint

**Critical**: The EP flag only takes effect when `TP_SIZE × DP_SIZE > 1`.

| TP_SIZE | DP_SIZE | EP Flag | EP Active? | Communication |
|---------|---------|---------|------------|---------------|
| 8 | 1 | Yes | Yes | AllReduce |
| 1 | 8 | Yes | Yes | AllToAll |
| 4 | 2 | Yes | Yes | AllToAll |
| 8 | 1 | No | No | AllReduce |
| 1 | 1 | Yes | No | N/A (constraint violated) |

**Key insight**: AllToAll communication requires `dp_size > 1`. With TP-only configurations (`dp_size=1`), vLLM always uses AllReduce even when the EP flag is enabled.

## Hands-on Examples

### Example 1: Basic vLLM Setup with Tensor Parallelism

```bash
# Serve a model with tensor parallelism on 4 GPUs
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --port 8000
```

### Example 2: Multi-Node with TP and PP

```bash
# Serve DeepSeek R1 with combined parallelism
python -m vllm.entrypoints.api_server \
    --model deepseek-ai/DeepSeek-R1 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 8 \
    --enable-chunked-prefill \
    --chunked-prefill-size 2048 \
    --port 8000
```

### Example 3: Custom Chunked Prefill Configuration

```python
# vLLM configuration for optimal chunked prefill
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    enable_chunked_prefill=True,
    max_num_seqs=256,  # Adjust based on your KV cache
    chunked_prefill_size=1024,  # Tune based on workload
)

# Generate text
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, how are you?"], sampling_params)
```

### Example 4: Profiling with Nsight Systems

```bash
# Profile vLLM to understand communication overhead
nsys profile \
    --trace=cuda,nvtx \
    --output=profile.qdrep \
    python -m vllm.entrypoints.api_server \
        --model meta-llama/Llama-2-70b-hf \
        --tensor-parallel-size 4

# Analyze results
nsys-ui profile.qdrep
```

### Best Practices

1. **Profile first**: Use Nsight Systems to understand communication vs. computation
2. **Tune chunked prefill**: Adjust `chunked_prefill_size` based on your prefill/decode ratio
3. **Consider hardware**: 
   - NVLink → Good for TP
   - PCIe only → Consider PP even within a node
4. **Calculate KV cache**: Determine optimal TP size by calculating available KV cache space
5. **Experiment**: What works for one deployment may not work for another

## Summary

### Key Takeaways

1. **Three fundamental strategies**: TP, DP, and PP are the core parallelism strategies in vLLM
2. **EP is a modifier**: Expert Parallelism is a flag (`--enable-expert-parallel`) that modifies MoE behavior when combined with TP or DP
3. **Combine strategies strategically**: TP+PP for multi-node, TP+EP for MoE latency, DP+EP for MoE throughput
4. **Enable chunked prefill** for pipeline parallelism, but tune the chunk size carefully
5. **Profile your workload**: Don't set parameters arbitrarily—measure communication vs. computation ratios
6. **Consider trade-offs**:
   - TP: Lower latency, higher communication overhead
   - PP: Lower communication, doesn't improve latency, more complex
   - DP: Linear throughput scaling, no communication overhead, but replicates weights
   - EP: Modifies expert distribution and communication patterns for MoE models

### When to Use What

- **Tensor Parallelism**: 
  - Model doesn't fit on single GPU
  - Good interconnect (NVLink) available
  - Latency-sensitive workloads
  
- **Data Parallelism**:
  - Model fits on single GPU (or small TP group)
  - Need to serve more concurrent requests
  - Throughput is primary concern
  - MoE models (combine with EP for experts)
  
- **Pipeline Parallelism**:
  - Model doesn't fit on single node
  - Poor interconnect between nodes
  - Throughput-focused workloads

- **Combined TP+PP**:
  - Very large models (400B+ parameters)
  - Multi-node deployments
  - Need to balance latency and communication

- **Expert Parallelism (EP)**:
  - MoE models with high activation density (>3%)
  - MLA/MQA models (DeepSeek) require EP with DP for KV cache partitioning
  - Must be combined with TP or DP (not standalone)
  - Activation constraint: `TP_SIZE × DP_SIZE > 1`

- **Combined Strategies**:
  - **TP+PP**: Very large models (400B+), multi-node deployments
  - **TP+EP**: MoE models, low-moderate concurrency, latency-sensitive
  - **DP+EP**: MoE models, high concurrency, throughput-focused (enables DP Attention)
  - **TP+DP+EP**: Large MoE models needing both sharding and throughput scaling

### Future Developments

- **vLLM v1**: Enhanced features and optimizations
- **Disaggregated Prefill/Decode**: Advanced techniques for further optimization
- **Improved EP stability**: Better support across model/quantization/hardware combinations



## References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - Kwon et al., SOSP 2023 (Original vLLM paper)
- [When to Reason: Semantic Router for vLLM](https://arxiv.org/abs/2510.08731) - Wang et al., NeurIPS 2025 Workshop on ML for Systems
- [vLLM Documentation - Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM Documentation - Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment.html)
- [vLLM Documentation - Distributed Serving](https://docs.vllm.ai/en/v0.8.1/serving/distributed_serving.html)
- [NVIDIA Dynamo KV Cache Manager](https://docs.nvidia.com/dynamo/archive/0.2.0/architecture/kv_cache_manager.html)
- [Red Hat: Distributed Inference with vLLM](https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm#gpu_parallelism_techniques_in_vllm)
- SAR (Speculative, Approximate, and Recurrent) Decoding Papers
- vLLM Roadmap: https://roadmap.vllm.ai
