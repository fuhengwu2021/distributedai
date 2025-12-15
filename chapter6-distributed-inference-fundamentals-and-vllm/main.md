title: "Distributed Inference Fundamentals and vLLM"

# Chapter 6 — Distributed Inference Fundamentals and vLLM

This chapter introduces distributed inference concepts and vLLM internals, focusing on tensor parallelism, pipeline parallelism, and optimization techniques for serving large language models efficiently in production environments.

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

For a chat model, replace the model name:

```bash
... vllm/vllm-openai:latest Qwen/Qwen2.5-0.5B-Instruct
```

For an embedding model:

```bash
... vllm/vllm-openai:latest sentence-transformers/all-MiniLM-L6-v2
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

## The Core Innovation: PagedAttention

vLLM's breakthrough innovation is **PagedAttention**, a memory management algorithm inspired by virtual memory paging in operating systems. This technique fundamentally solves the **KV cache fragmentation problem** that plagues traditional LLM serving systems.



### The KV Cache Problem

During autoregressive generation, LLMs maintain a **Key-Value (KV) cache** that stores the computed key and value tensors for all previous tokens in a sequence. This cache:

- **Grows linearly** with sequence length
- **Varies per request**—different prompts have different lengths
- **Creates fragmentation**—traditional systems allocate contiguous memory blocks per request, leading to wasted space when sequences finish or have different lengths

**Traditional Approach Problems**:
```
Request 1: [========]  (8 tokens, finished)
Request 2: [====]      (4 tokens, active)
Request 3: [==========] (10 tokens, active)
           ↑ Memory fragmentation - can't reuse Request 1's space efficiently
```

### How PagedAttention Works

PagedAttention divides the KV cache into **fixed-size blocks** (similar to memory pages in OS virtual memory):

1. **Block-based storage**: KV cache is stored in fixed-size blocks (e.g., 16 tokens per block)
2. **Block tables**: Each request maintains a "block table" that maps logical sequence positions to physical block addresses
3. **Dynamic allocation**: Blocks are allocated and freed as sequences grow or complete
4. **Memory reuse**: Freed blocks can be immediately reused by new requests

**PagedAttention Approach**:
```
Block Pool: [Block0][Block1][Block2][Block3][Block4][Block5]...
            ↓        ↓        ↓
Request 1:  [Block0][Block1]  (finished, blocks returned to pool)
Request 2:  [Block2]          (active)
Request 3:  [Block3][Block4]   (active)
            ↑ No fragmentation - blocks can be reused immediately
```

### Key Benefits

1. **Memory Efficiency**: 
   - Eliminates fragmentation waste
   - Enables near-100% memory utilization
   - Can serve 2-4x more concurrent requests with the same GPU memory

2. **Flexible Batching**:
   - Requests with different sequence lengths can be batched efficiently
   - No need to pad sequences to the same length
   - Supports dynamic batching (continuous batching)

3. **Long Context Support**:
   - Efficiently handles variable-length contexts
   - Supports very long sequences without pre-allocating maximum memory
   - Enables serving models with long context windows (100K+ tokens)

4. **Performance**:
   - Custom CUDA kernels optimized for paged access patterns
   - Minimal overhead from block table lookups
   - Efficient memory access patterns

### vLLM Architecture Overview

vLLM's architecture is built around efficient memory management and distributed execution:

- **PagedAttention**: Core memory management for KV cache
- **Continuous Batching**: Dynamically groups requests into batches
- **Distributed Execution**: Supports tensor parallelism and pipeline parallelism
- **Optimized Kernels**: Custom CUDA kernels for attention computation

### Connection to Distributed Inference

While PagedAttention solves memory efficiency within a single GPU, **distributed inference techniques** (tensor parallelism, pipeline parallelism) are needed when:

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

## Contrasting Inference with Training

Understanding the differences between training and inference is crucial for designing effective distributed inference systems:

### Training Characteristics
- **Scale**: Very large scale, often using mega-clusters with many GPUs
- **Parallelism**: Employs data parallelism, pipeline parallelism, tensor parallelism, and FSDP
- **Objective**: Maximize **throughput** (tokens per second)
- **Workload**: Static—each training step is similar
- **Complexity**: Comes from parallelism algorithms and the forward/backward pass

### Inference Characteristics
- **Scale**: Typically smaller—allocate a small number of machines per model, scale horizontally based on load
- **Objective**: Balance **throughput** and **latency** (especially time-to-first-token)
- **Workload**: Dynamic—user requests arrive at varying rates
- **Complexity**: Comes from:
  - KV cache management
  - Speculative decoding (draft models)
  - Dynamic request scheduling
  - Multiple moving parts

### Key Problems in Large Language Model Inference

1. **Model cannot fit into a single GPU**: Requires tensor parallelism within a node
2. **Model cannot fit into a single node**: Requires pipeline parallelism across nodes
3. **Increased CPU overhead**: Requires optimized control plane (vLLM addresses this)

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

## Introduction to Tensor Parallelism (TP)

**Tensor Parallelism (TP)** is a technique that shards model weights horizontally across multiple GPUs within a single node, allowing all GPUs to compute concurrently. This follows an **SPMD (Single Program, Multiple Data)** paradigm.

### Why Tensor Parallelism?

When models get larger, one GPU is insufficient. Even within a single node, you might have multiple GPUs available. The question is: how do we parallelize computation across these GPUs?

**Answer**: Shard weights horizontally and compute everything concurrently at the same time.

### Visual Example

For a model with 4 layers (for illustration):
- Each GPU gets a piece of each layer
- All GPUs work simultaneously on different parts of the computation
- Communication operations synchronize results between GPUs

## Linear Algebra Behind TP: Column & Row Parallelism

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

## Applying TP to a Multi-Layer Perceptron (MLP)

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

## Benefits of Tensor Parallelism

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

## Expert Parallelism for Mixture-of-Experts (MoE) Models

For **Mixture-of-Experts (MoE)** models (e.g., DeepSeek R1), vLLM employs **Expert Parallelism**:

### How It Works

- Each GPU holds different experts
- Different tokens may need different experts
- Use **all-to-all** operations to:
  1. Send tokens to the correct expert GPU
  2. Compute expert outputs
  3. Send results back to original GPUs (another all-to-all)

### Status

Expert parallelism is **planned** for vLLM but not yet implemented. This is especially important for large MoE models like DeepSeek R1.

## Trade-offs of Tensor Parallelism

### Advantages

1. **Improves end-to-end latency**: By splitting weights, each GPU has less to load and compute
2. **Reduces memory pressure**: Enables larger models and more KV cache
3. **Simple implementation**: SPMD paradigm is straightforward

### Disadvantages

1. **High communication overhead**: 
   - All-reduce operations for every layer
   - Can be 60%+ of time in prefill-heavy workloads
   - Especially problematic without NVLink (e.g., L4 GPUs over PCIe)

2. **Hardware requirements**:
   - Works best with NVLink within a node
   - Poor performance with PCIe interconnect for prefill-heavy workloads

3. **Constraint**: Attention heads must be divisible by tensor parallel size (or use padding)

### When to Use TP

- **Good**: Models that don't fit on a single GPU, good interconnect (NVLink)
- **Caution**: Prefill-heavy workloads with poor interconnect
- **Best practice**: Profile your workload to understand communication vs. computation ratio

## Introduction to Pipeline Parallelism (PP) for Multi-Node Inference

When models are too large for a single node (e.g., DeepSeek R1, LLaMA 405B), **Pipeline Parallelism (PP)** shards the model across multiple nodes.

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

## Combining Tensor and Pipeline Parallelism

Since TP and PP operate along **different axes**, they can be combined:

### Typical Configuration

- **Pipeline Parallelism**: Across nodes (where interconnect is slower)
- **Tensor Parallelism**: Within nodes (where NVLink provides fast communication)

### Benefits of Combination

1. **Reduced communication in PP**: When using TP+PP, each GPU only sends its TP chunk
   - Data size: `batch_size × sequence_length × hidden_size / TP_size`
   - Smaller transfers between pipeline stages

2. **Flexibility**: Not a hard rule—sometimes PP within a node makes sense, sometimes TP across nodes works if interconnect is good enough

### Example Configuration

For DeepSeek R1:
```bash
--tensor-parallel-size 4    # TP within each node
--pipeline-parallel-size 8  # PP across 8 nodes
```

## Solving Pipeline Bubbles with Request Groups

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

## Optimizing Pipelines with Chunked Prefill

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

1. **Use TP and PP together** when applicable, but experiment to find what works best for your hardware and workload
2. **Enable chunked prefill** for pipeline parallelism, but tune the chunk size carefully
3. **Don't set parameters arbitrarily**: Profile and optimize based on your specific deployment
4. **Consider trade-offs**:
   - TP: Lower latency, higher communication overhead
   - PP: Lower communication, doesn't improve latency, more complex

### When to Use What

- **Tensor Parallelism**: 
  - Model doesn't fit on single GPU
  - Good interconnect (NVLink) available
  - Latency-sensitive workloads
  
- **Pipeline Parallelism**:
  - Model doesn't fit on single node
  - Poor interconnect between nodes
  - Throughput-focused workloads

- **Combined TP+PP**:
  - Very large models (400B+ parameters)
  - Multi-node deployments
  - Need to balance latency and communication

### Future Developments

- **Expert Parallelism**: Coming soon for MoE models
- **vLLM v1**: Enhanced features and optimizations
- **Disaggregated Prefill/Decode**: Advanced techniques for further optimization

---

## References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - Kwon et al., SOSP 2023 (Original vLLM paper)
- [When to Reason: Semantic Router for vLLM](https://arxiv.org/abs/2510.08731) - Wang et al., NeurIPS 2025 Workshop on ML for Systems
- [vLLM Documentation - Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM Documentation - Distributed Serving](https://docs.vllm.ai/en/v0.8.1/serving/distributed_serving.html)
- [NVIDIA Dynamo KV Cache Manager](https://docs.nvidia.com/dynamo/archive/0.2.0/architecture/kv_cache_manager.html)
- [Red Hat: Distributed Inference with vLLM](https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm#gpu_parallelism_techniques_in_vllm)
- SAR (Speculative, Approximate, and Recurrent) Decoding Papers
- vLLM Roadmap: https://roadmap.vllm.ai
