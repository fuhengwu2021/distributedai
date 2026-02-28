# Chapter 11: The Evolving Landscape of Distributed AI {-}

*Exploring emerging technologies and future directions in distributed AI*

> The best way to predict the future is to invent it.
- Alan Kay, Computer Scientist

**Code Summary**

- `torch.distributed.checkpoint.save`: Async distributed checkpointing with DCP
- `torch.distributed.checkpoint.load`: Load sharded checkpoints across ranks
- `torch.distributed.elastic.multiprocessing`: Elastic training with fault tolerance
- `torch.ao.quantization.quantize_dynamic`: Dynamic quantization for model compression
- `flwr.client.NumPyClient`: Flower federated learning client interface
- `flwr.server.strategy.FedAvg`: Federated averaging aggregation strategy
- `megatron.core.transformer.moe.router.TopKRouter`: Megatron MoE top-k routing
- `vllm.LLM`: vLLM inference engine with PagedAttention
- `sglang.Engine`: SGLang runtime with RadixAttention



## The Evolution of Distributed AI

Throughout this book, we've explored the foundations of distributed AI: from DDP's gradient synchronization to FSDP's memory optimization, from vLLM's PagedAttention to SGLang's RadixAttention. These technologies have enabled training and serving models that were unimaginable just a few years ago. But the field doesn't stand still—the landscape continues to shift in ways that surprise even seasoned practitioners.

### Where We Stand Today

The numbers tell a remarkable story. Training infrastructure has scaled to over 100,000 GPUs in a single cluster, enabled by new communication frameworks like NCCLX and Torchcomms. SGLang now powers more than 400,000 GPUs worldwide, running production workloads for xAI's Grok 3 and Microsoft Azure. Perhaps most striking is the economics: DeepSeek-V3, a 671-billion parameter MoE model with 37 billion active parameters per token, was trained for just $5.5 million on 2,048 H800 GPUs—a cost that would have seemed impossibly low just a few years earlier.

On-device AI has crossed a critical threshold. Models like Gemma 3n run comfortably in 2-3GB of RAM using techniques like Per-Layer Embeddings, making LLMs practical on smartphones. The 4-bit quantization methods that once seemed experimental are now standard practice.

But perhaps the most significant shift is one of priorities. For years, the AI industry focused obsessively on training: building bigger clusters, training larger models, pushing the frontier. That era hasn't ended, but it's no longer the dominant story. Inference workloads now consume over 55% of AI infrastructure spending, and that share continues to grow. The global AI inference market is projected to reach $1.3 trillion within the next several years.

This shift makes sense when you think about it. Training happens once; inference happens millions of times. As models mature and deployment scales, the economics inevitably favor inference optimization. Infrastructure costs have dropped dramatically over the past few years, making it economically viable to deploy AI in contexts that were previously impractical.

### The New Bottlenecks

With this growth come new constraints. Interestingly, the primary bottleneck is no longer silicon availability—it's power and cooling. Data centers are being designed around thermal limits rather than rack space. Organizations now factor cooling retrofits into total cost of ownership calculations and negotiate cloud capacity commitments months in advance.

The remaining technical challenges are equally interesting. Scaling beyond 100K GPUs efficiently requires new communication patterns and fault tolerance mechanisms. Balancing latency and throughput in production inference remains an art as much as a science. And cost optimization across hybrid cloud-edge architectures is becoming a discipline unto itself.

### Emerging Trends at a Glance

Before diving into each area, here's a roadmap of what's reshaping distributed AI:

__Advanced MoE Architectures__: LatentMoE brings hardware-software co-design for optimal accuracy per FLOP, adopted by Nvidia's Nemotron-3. MoSE (Mixture of Slimmable Experts) enables variable-width expert execution for continuous accuracy-compute trade-offs. Elastic MoE scales inference-time expert count to 2-3× training values. ReMoE introduces fully differentiable routing using ReLU instead of TopK+Softmax.

__On-Device and Edge AI__: Sub-billion parameter models now handle practical tasks effectively. Gemma 3n's Per-Layer Embeddings reduce RAM requirements—5B/8B models run with 2B/4B footprint. Memory bandwidth (50-90 GB/s on mobile vs 2-3 TB/s in data centers) is the real bottleneck, not compute. Speculative decoding delivers 2-3x speedups on edge devices.

__Next-Generation Communication__: Torchcomms, PyTorch's new API, is designed for 100K+ GPU scale with heterogeneous hardware support. NCCLX/RCCLX (Meta's enhanced backends) deliver 10-50% speedup on AllReduce operations. Async checkpointing is now 6x faster with cached plans and reduced GIL contention.

__Inference Engine Evolution__: SGLang achieves 16,215 tok/s on H100, with RadixAttention for automatic prefix reuse and pipeline parallelism for million-token contexts. vLLM's PagedAttention reduces KV cache waste from 60-80% to under 4%. Both now support AMD MI355/MI300, Intel Xeon, Google TPUs, and Ascend NPUs.



## Mixture of Experts: The Dominant Architecture

We covered MoE fundamentals in Chapter 5 (expert parallelism for training, see Figure~\ref{fig:expert-parallelism}) and Chapter 6 (MoE inference with vLLM). Here we focus on the research frontier pushing MoE capabilities further.

Research continues to push MoE capabilities beyond what we covered in earlier chapters:

__LatentMoE__[^latentmoe] uses hardware-software co-design to optimize accuracy per FLOP across different inference scenarios. Adopted by Nvidia's Nemotron-3 models, it demonstrates that the right co-design can significantly improve efficiency without sacrificing capability.

[^latentmoe]: LatentMoE: Toward Optimal Accuracy per FLOP. \url{https://arxiv.org/abs/2601.18089}

__MoSE (Mixture of Slimmable Experts)__[^mose] allows variable-width expert execution. Instead of fixed-size experts, MoSE can dynamically adjust expert width at inference time, enabling continuous accuracy-compute trade-offs. This is particularly valuable for deployment scenarios where latency requirements vary.

[^mose]: MoSE: Mixture of Slimmable Experts. \url{https://arxiv.org/abs/2602.06154}

__Elastic MoE__[^elastic-moe] scales the number of active experts beyond training-time values. A model trained with top-2 routing can use top-4 or top-6 at inference time, achieving 2-3x the training-time expert count while improving performance. This decouples training and inference configurations.

[^elastic-moe]: Elastic MoE: Inference-Time Expert Scaling. \url{https://arxiv.org/abs/2501.03140}

__ReMoE__[^remoe] replaces the non-differentiable TopK+Softmax routing with fully differentiable ReLU-based routing. This enables more efficient dynamic computation allocation and simplifies the training dynamics.

[^remoe]: ReMoE: Fully Differentiable Mixture-of-Experts with ReLU Routing. \url{https://arxiv.org/abs/2412.14711}

The code in `code/moe_layer.py` provides implementations that extend the concepts from Chapter 5, including load-balanced routing with the auxiliary-loss-free approach.



## The Edge-Cloud Continuum

One of the most profound shifts in distributed AI is the blurring of the boundary between cloud and edge. For years, the assumption was simple: train in the cloud, maybe serve in the cloud, and edge devices are for consumption. That model is breaking down.

### Why Edge AI Matters Now

The case for edge AI is compelling. Latency drops from hundreds of milliseconds to single digits when you eliminate network round-trips. Privacy improves dramatically when data never leaves the device. Costs decrease when you're not paying per-token for cloud inference. And offline capability opens entirely new use cases.

What changed to make this practical? Three things converged. First, model architectures got more efficient. Sub-billion parameter models now handle practical tasks that previously required 7B+ models. Second, quantization techniques matured. Going from 16-bit to 4-bit reduces both storage and memory traffic by 4x, with minimal quality loss using methods like GPTQ[^gptq] and AWQ[^awq]. Third, hardware improved. Mobile NPUs are now capable enough to run these quantized models at acceptable speeds.

[^gptq]: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. \url{https://arxiv.org/abs/2210.17323}
[^awq]: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. \url{https://arxiv.org/abs/2306.00978}

But there's a subtlety here that's often missed. The bottleneck on mobile devices isn't compute—it's memory bandwidth. A typical mobile device has 50-90 GB/s of memory bandwidth, compared to 2-3 TB/s in a data center GPU. That's a 30-50x gap. This is why quantization matters so much: it's not primarily about storage, it's about reducing the bytes that need to flow through that bandwidth bottleneck on every forward pass.

### Practical On-Device Models

The landscape of edge-capable models has matured rapidly:

| Model | Parameters | Memory | Use Cases |
|-------|-----------|--------|-----------|
| Gemma 3n | 5B/8B | 2-3GB | General assistant |
| Llama 3.2 | 1B/3B | 1-2GB | Text tasks |
| Phi-4 mini | 3.8B | 2GB | Reasoning |
| Qwen2.5 | 0.5B-1.5B | <1GB | Lightweight tasks |

Google's Gemma 3n uses Per-Layer Embeddings to run 5B and 8B parameter models with the memory footprint of 2B and 4B models respectively—about 2-3GB total. Meta's Llama 3.2 offers 1B and 3B variants specifically designed for edge deployment. Microsoft's Phi-4 mini packs strong reasoning capability into 3.8B parameters. Alibaba's Qwen2.5 ranges from 0.5B to 1.5B for the lightest-weight tasks.

The key insight is that architecture and training quality matter more than raw size at small scales. A well-trained 1B model can outperform a poorly-trained 3B model on many tasks.

### Edge-Cloud Coordination

The most interesting systems don't treat edge and cloud as separate worlds—they find ways to make them work together. Remember speculative decoding from Chapter 7? We used it to speed up inference on a single server by having a small draft model propose tokens that a larger model then verifies. The same idea works beautifully across the edge-cloud boundary.

Picture your phone running a tiny but fast model. It generates a sequence of candidate tokens—maybe "The cat sat on the"—and ships them to a powerful cloud model. The cloud doesn't generate anything; it just checks whether each token matches what it would have produced. Verification is cheap: the cloud model can evaluate all five tokens in a single forward pass, whereas generating them one by one would take five passes. When the drafts are mostly correct (and for predictable text, they often are), you get cloud-quality output at edge-like speed. Speedups of 2-3x are common in practice[^spec-decode].

![Edge-cloud speculative decoding workflow](img/speculative_decoding.png){#fig:speculative-decoding .block width=100% align=center}

Figure~\ref{fig:speculative-decoding} illustrates this dance. The edge device proposes tokens (yellow circles), sends them to the cloud, and the verifier stamps each one as accepted (green) or rejected (red). In this example, "on" gets rejected—perhaps the cloud model prefers a different preposition—so the edge will need to regenerate from that point. But four out of five tokens sailed through, saving significant latency.

[^spec-decode]: Fast Inference from Transformers via Speculative Decoding. \url{https://arxiv.org/abs/2211.17192}

Intelligent routing pushes this coordination further. Not every request actually needs the cloud. A well-designed system estimates how hard each request is, checks how confident the edge model feels, glances at network conditions, and decides: handle locally, or send to the cloud? Simple queries—"What time is it in Tokyo?"—stay on device. Complex reasoning tasks go to the cloud. The routing logic can even learn from past decisions, getting better at predicting which requests will succeed locally.

The `code/edge_cloud.py` file implements these patterns: `EdgeCloudSpeculativeDecoding` for the draft-verify loop, `IntelligentOffloading` for complexity-based routing, and `AdaptiveRouter` that improves over time.

## Parallelism at Scale

The parallelism strategies we covered in Chapters 3-5—data parallelism (DDP, FSDP), tensor parallelism, and pipeline parallelism—remain foundational. But at 100K+ GPU scale, new challenges emerge that require new solutions.

### The Communication Revolution

PyTorch's introduction of Torchcomms marked a significant evolution in distributed training infrastructure. The previous communication stack, while functional, was showing its age at massive scale. Torchcomms was designed from the ground up for 100K+ GPU deployments with several key innovations.

First, it decouples communication primitives from PyTorch's core, allowing researchers to iterate on new collectives and backends independently. Second, it uses eager initialization and model-specific hints to optimize communicator and resource allocation at massive scale. Third, it supports heterogeneous hardware—mixed deployments across multiple vendors and GPU generations within a single training job. Fourth, it builds in fault tolerance mechanisms that were previously afterthoughts.

Meta's NCCLX backend, released alongside Torchcomms, and RCCLX for AMD platforms, deliver 10-50% speedups on AllReduce operations through techniques like Direct Data Access (DDA). These aren't incremental improvements—they can mean the difference between a training run that's economically viable and one that isn't.

### Hierarchical Parallelism

At scale, you don't choose one parallelism strategy—you combine them. A typical large training run might use data parallelism across nodes, tensor parallelism within nodes, and pipeline parallelism across model stages. The math is straightforward: if you have `world_size` GPUs, then `dp_size × tp_size × pp_size = world_size`.

The art is in choosing the right combination. Tensor parallelism has low communication overhead within a node (NVLink is fast) but high overhead across nodes. Pipeline parallelism can hide communication latency but introduces bubble overhead. Data parallelism scales well but requires gradient synchronization. The optimal configuration depends on your model architecture, hardware topology, and batch size constraints.

The `HierarchicalParallelism` class in `code/parallelism.py` manages the process group creation for all three dimensions, ensuring that each GPU knows which groups it belongs to for each type of parallelism.

### Sequence Parallelism for Long Contexts

As context lengths grow—million-token contexts are now practical—sequence parallelism becomes essential. The idea is to split the sequence dimension across GPUs. Each GPU computes attention for its local chunk of the sequence, but attention requires seeing the full key and value tensors. This means all-gathering K and V from all sequence-parallel ranks.

SGLang's pipeline parallelism implementation achieves remarkable results: 3.31× prefill throughput improvement for DeepSeek-V3.1 and up to 81% TTFT reduction for million-token contexts, while maintaining 82.8% scaling efficiency. These numbers matter because long-context inference is increasingly common in production—RAG systems, document analysis, and code understanding all benefit from longer contexts.

### Ring Attention

Ring attention[^ring-attention] offers a memory-efficient alternative to standard sequence parallelism. Instead of all-gathering the full K and V tensors (which requires O(sequence_length) memory per GPU), ring attention passes K and V chunks around a ring of GPUs, computing partial attention scores at each step.

[^ring-attention]: Ring Attention with Blockwise Transformers for Near-Infinite Context. \url{https://arxiv.org/abs/2310.01889}

The algorithm works as follows: each GPU starts with its local Q, K, V chunks. In each round, GPUs compute attention between their local Q and the current K, V. Then K and V are passed to the next GPU in the ring. After N rounds (where N is the number of GPUs), each GPU has computed attention against all K, V chunks.

This trades communication rounds for memory efficiency—useful when you're memory-constrained but have communication bandwidth to spare. The `RingAttention` class in `code/parallelism.py` implements this pattern.



## When Things Go Wrong: Fault Tolerance

Here's a sobering calculation. Suppose you have a cluster of 100,000 GPUs, each with 99.9% reliability over a 24-hour period. The probability that all GPUs survive the day is 0.999^100,000 ≈ 0.00005. You'll see roughly 100 failures per day. At this scale, failures aren't exceptional—they're the norm.

### The Checkpointing Challenge

Traditional checkpointing is synchronous: stop training, save state to disk, resume. This was acceptable when checkpoints took seconds and training runs lasted hours. At scale, checkpoints can take minutes, and training runs last weeks. The overhead becomes significant.

PyTorch's Distributed Checkpointing (DCP) introduced several optimizations that achieve 6.5x faster checkpoint processing:

- __Cached save plans__: Reuse tensor metadata across checkpoints—the shapes and dtypes don't change, so why recompute them?
- __Background saving__: Reduce GIL contention by moving the actual I/O to separate threads
- __Incremental saves__: Only write tensors that have changed since the last checkpoint

The `AsyncCheckpointer` in `code/fault_tolerance.py` implements these ideas. It maintains a background thread that processes save requests from a queue, allowing training to continue while checkpoints are written. It also manages checkpoint retention, automatically cleaning up old checkpoints to save disk space.

### Detecting and Handling Failures

Failure detection sounds simple—just check if workers are responding—but the details matter. How long do you wait before declaring a worker dead? Too short, and you'll have false positives from network hiccups. Too long, and you waste time waiting for a genuinely dead worker.

The `FailureDetector` class uses a heartbeat mechanism. Workers send periodic heartbeats; missing heartbeats trigger failure callbacks. The system can then decide how to respond: continue with remaining workers, restart the failed worker, or checkpoint and abort.

Key design decisions include:

- __Heartbeat interval__: Typically 1-5 seconds, balancing responsiveness with overhead
- __Timeout threshold__: Usually 3-5 missed heartbeats before declaring failure
- __Failure callback__: What action to take—log, alert, checkpoint, or abort

### Elastic Training

The most sophisticated approach is elastic training, where the system adapts to worker changes without stopping. Workers can join or leave during training. When a worker fails, the system checkpoints, redistributes work among remaining workers, and continues. When a new worker joins, it receives a share of the work.

This requires careful coordination:

- __Gradient averaging__ must account for the changing number of workers
- __Data loading__ must redistribute shards dynamically
- __Learning rate schedules__ may need adjustment for different worker counts
- __Checkpointing__ must handle partial states during transitions

The `ElasticTrainer` class in `code/fault_tolerance.py` handles these concerns, providing automatic checkpoint on failure and graceful degradation.



## Beyond Text: Multimodal Distributed Training

The models capturing the most attention today aren't text-only—they're multimodal. Vision-Language Models (VLMs) that can see and read, models that process video, systems that understand audio alongside text. Distributing these models efficiently requires new patterns.

### The Cross-Modal Challenge

A VLM typically has a vision encoder (often ViT-based) that produces image features, and a language model that processes text while attending to those image features. The cross-modal attention—where text tokens attend to vision features—is where the modalities meet.

This cross-modal attention can be parallelized using tensor parallelism, sharding attention heads across GPUs. But there's a subtlety: the vision encoder and language model may have different optimal parallelism strategies. The vision encoder processes fixed-size images and benefits from different batch sizes than the language model, which handles variable-length text.

The `code/multimodal.py` file provides implementations of these components:

- `VisionEncoder`: ViT-style image encoder with patch embedding and transformer blocks
- `CrossModalAttention`: Text-to-vision attention with tensor parallelism support
- `VisionLanguageModel`: Complete VLM combining vision and language components

### Handling Multiple Modalities

Different modalities have different characteristics:

| Modality | Size | Characteristics |
|----------|------|-----------------|
| Images | Fixed after preprocessing | Batch-friendly, predictable memory |
| Text | Variable length | Requires padding, dynamic memory |
| Video | Multiple frames | High memory, temporal structure |
| Audio | Variable duration | Temporal structure, streaming |

`MultimodalDataParallel` handles scattering batches across workers while keeping modalities aligned. This is trickier than it sounds—you need to ensure that the image corresponding to a text prompt ends up on the same worker, even when batch sizes differ across modalities.



## Federated Learning: An Alternative Paradigm

Throughout this book, we've assumed that all GPUs can access a shared dataset—or at least shards of it stored in a common data center. But what if the data can't be moved? What if privacy regulations, competitive concerns, or sheer logistics make centralization impossible? This is where federated learning comes in.

### The Federated Paradigm

Federated learning flips the script: instead of bringing data to the model, it brings the model to the data. Each participant—whether a smartphone, a hospital, or a bank—trains on its local data and shares only model updates, never raw data. A central server aggregates these updates to produce a global model. The paradigm has been around since 2016, and it has matured into a practical solution for privacy-sensitive domains.

Figure~\ref{fig:federated-learning} illustrates this architecture. A central server at the top coordinates training and holds the global model. Four clients at the bottom—a hospital with patient records, a bank with transaction data, a mobile app with user behavior, and IoT devices with sensor readings—each train locally on their private data. Blue arrows show the server distributing the current model; green arrows show clients sending back their updates. The crucial point, highlighted at the bottom: data never leaves the client. Only gradients and weights travel across the network.

![Federated learning: data stays on clients](img/federated_learning.png){#fig:federated-learning .block width=80% align=center}

The canonical algorithm is FedAvg[^fedavg]. Each round, the server sends the current model to a subset of clients. Clients train locally for several epochs, then send updated weights back. The server averages these weights to produce a new global model. Simple as it sounds, this protocol has proven remarkably effective—and the field has built many refinements on top of it.

[^fedavg]: Communication-Efficient Learning of Deep Networks from Decentralized Data. \url{https://arxiv.org/abs/1602.05629}

### Where Federated Learning Thrives

Federated learning has found strong adoption in specific verticals:

__Healthcare__: Hospitals can collaboratively train diagnostic models without sharing patient records. Projects like NVIDIA FLARE[^flare] enable multi-institutional medical imaging research while maintaining HIPAA compliance. A model trained across 20 hospitals sees more diverse pathology than any single institution could provide.

[^flare]: NVIDIA FLARE: Federated Learning Application Runtime Environment. \url{https://github.com/NVIDIA/NVFlare}

__Mobile Keyboards__: Google's Gboard uses federated learning to improve next-word prediction without uploading what users type. The model learns from millions of devices while keeping text on-device. This was one of the first large-scale production deployments of federated learning.

__Financial Services__: Banks can collaborate on fraud detection models without sharing transaction data. Each institution contributes patterns from its customer base; the combined model catches fraud that no single bank's data would reveal.

__Edge IoT__: Industrial sensors, autonomous vehicles, and smart devices generate data that's expensive or impractical to centralize. Federated learning enables model improvement without massive data transfers.

### The Challenges of Heterogeneity

Federated learning faces challenges that don't exist in traditional distributed training:

__Non-IID Data__: In a data center, you can shuffle data to ensure each GPU sees a representative sample. In federated learning, each client's data reflects its local distribution. A keyboard model trained in Japan sees different text than one trained in Brazil. This statistical heterogeneity can cause model divergence and slow convergence.

__System Heterogeneity__: Clients have vastly different compute capabilities. A flagship smartphone and a three-year-old budget phone can't train at the same speed. Some clients may drop out mid-round due to battery constraints or network issues. The system must be robust to stragglers and partial participation.

__Communication Constraints__: Unlike NVLink-connected GPUs, federated clients communicate over mobile networks or the internet. Bandwidth is limited and expensive. Techniques like gradient compression, quantization, and infrequent synchronization become essential—not for performance, but for feasibility.

### Modern Federated Techniques

Research has addressed these challenges with increasingly sophisticated methods:

__FedProx__[^fedprox] adds a proximal term to the local objective, preventing clients from drifting too far from the global model. This stabilizes training when data is highly non-IID.

[^fedprox]: Federated Optimization in Heterogeneous Networks. \url{https://arxiv.org/abs/1812.06127}

__Scaffold__[^scaffold] uses control variates to correct for client drift, achieving faster convergence than FedAvg on heterogeneous data.

[^scaffold]: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. \url{https://arxiv.org/abs/1910.06378}

__Differential Privacy__ can be layered on top of federated learning to provide formal privacy guarantees. By adding calibrated noise to updates, even the aggregated model reveals limited information about any individual's data. This is particularly important in healthcare and finance.

__Secure Aggregation__[^secagg] ensures the server only sees the sum of client updates, not individual contributions. Even a compromised server can't extract any single client's model update.

[^secagg]: Practical Secure Aggregation for Privacy-Preserving Machine Learning. \url{https://eprint.iacr.org/2017/281}

### Federated Learning for LLMs

Applying federated learning to large language models presents unique challenges. A 7B parameter model can't fit on a smartphone, and even fine-tuning requires significant memory. Recent work explores several approaches:

__Federated Fine-Tuning with LoRA__: Instead of updating all parameters, clients train low-rank adapters. This reduces communication (only adapter weights are exchanged) and memory (adapters are small). FedIT[^fedit] and similar approaches make federated LLM fine-tuning practical.

[^fedit]: Federated Instruction Tuning of LLMs with Domain Coverage Augmentation. \url{https://arxiv.org/abs/2409.12568}

__Split Learning__: The model is split between client and server. Clients run early layers on their data and send activations (not raw data) to the server, which completes the forward pass. This enables training large models without requiring clients to hold the full model.

__On-Device Personalization__: Rather than training a single global model, each device maintains a personalized version. The global model provides a starting point; local fine-tuning adapts to individual users. This is particularly relevant for assistants that should reflect user preferences.

### The Flower Framework

Flower[^flower] has emerged as the leading open-source federated learning framework. It provides:

- A simple API for defining federated training loops
- Support for various aggregation strategies (FedAvg, FedProx, etc.)
- Integration with PyTorch, TensorFlow, and JAX
- Simulation capabilities for development and research
- Production deployment tools

[^flower]: Flower: A Friendly Federated Learning Framework. \url{https://flower.ai/}

The framework abstracts away much of the complexity, letting researchers focus on algorithms rather than infrastructure. A basic federated training setup requires just a few dozen lines of code.

### When to Consider Federated Learning

Federated learning isn't a replacement for traditional distributed training—it's a tool for specific situations:

| Consider Federated Learning When | Stick with Traditional Training When |
|----------------------------------|-------------------------------------|
| Data can't leave its source (privacy, regulation) | Data can be centralized |
| Data is naturally distributed (mobile, edge) | You control the infrastructure |
| Participants don't trust each other | Single organization training |
| Communication is expensive | High-bandwidth interconnect available |

The overhead of federated learning—slower convergence, communication constraints, heterogeneity challenges—means it's not the right choice when you can simply centralize data. But when privacy or logistics make centralization impossible, federated learning enables collaboration that wouldn't otherwise happen.



## Agentic AI: The Next Frontier

Perhaps the most exciting recent development is the rise of agentic AI—systems that don't just respond to prompts but plan, use tools, and take actions. These systems push distributed inference in new directions.

### Why Agents Need Distribution

Consider what happens when an agent decides to call a tool. The tool might be a code interpreter, a web browser, a database query, or an API call. Each tool has different resource requirements and latency characteristics. Some tools are stateful and need to run on specific workers. Some are embarrassingly parallel; others are sequential bottlenecks.

Multi-agent systems add another layer. Multiple agents might collaborate on a task, each with different capabilities. They need to communicate, coordinate, and sometimes disagree. This communication can happen within a single node or across a distributed system.

Long reasoning chains—the kind that models like o1 and DeepSeek-R1[^deepseek-r1] produce—require sustained inference over many steps. Each step might involve tool calls, memory lookups, or coordination with other agents. The inference isn't a single forward pass; it's an extended computation that might run for minutes.

[^deepseek-r1]: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. \url{https://arxiv.org/abs/2501.12948}

### Multi-Agent Patterns

The `code/agentic_inference.py` file implements several coordination patterns:

__Sequential Processing__: Agents process in order, passing output to the next agent. Useful when agents have complementary capabilities—one might extract information, another might reason about it, a third might format the response.

__Parallel Processing__: Multiple agents process the same task concurrently. Useful for ensemble approaches where different agents might find different solutions, or when you want to race multiple strategies.

__Hierarchical Processing__: A coordinator agent delegates subtasks to worker agents. Useful for complex tasks that decompose naturally—the coordinator handles high-level planning while workers handle specific subtasks.

### Distributed Tool Execution

The `DistributedToolExecutor` routes tool calls to appropriate workers:

- Tools are registered with optional worker assignments
- A code interpreter might only run on workers with proper sandboxing
- A simple calculator can run anywhere
- The executor tracks execution statistics for optimization

This enables sophisticated tool placement strategies. Frequently-used tools can be replicated across workers. Stateful tools can be pinned to specific workers. Resource-intensive tools can be isolated to prevent interference.

### Reasoning Agents

The `ReasoningAgent` implements multi-step reasoning with tool use. It maintains a reasoning trace—a record of thoughts, tool calls, and observations. At each step, it decides whether to think, call a tool, or produce a final answer.

The reasoning loop:

1. Generate a thought about the current state
2. Decide on an action (think more, call tool, or answer)
3. If tool call, execute and observe result
4. Repeat until answer or max steps reached

This is a simplified version of what production reasoning systems do, but it captures the core loop that enables complex problem-solving.



## The Economics of Scale

As distributed AI matures, economic considerations become increasingly important. The days of "just throw more GPUs at it" are ending; efficiency and cost optimization are now first-class concerns.

### Workload Distribution

Enterprise workload distribution shows no single dominant category:

| Workload Type | Share |
|--------------|-------|
| Inference at scale | 34.6% |
| Foundation model training | 24.9% |
| Domain-specific training | 23.3% |
| Fine-tuning | 17.2% |

This diversity means that infrastructure must be flexible—optimized for one workload type won't serve the full range of needs.

Organizations are adopting hybrid strategies:

- __Public cloud__: Handles elastic and variable workloads where demand is unpredictable
- __On-premises__: Serves high-volume production inference where predictable costs matter
- __Near-edge data centers__: Handle latency-sensitive, user-facing AI services

### Resource Management

Dynamic resource allocation is essential when workloads vary. The `AdaptiveGPUAllocator` in `code/resource_management.py` estimates GPU requirements based on:

- Model size (memory requirement)
- Batch size (throughput requirement)
- Job priority (business importance)

It maintains a queue for jobs that can't be immediately scheduled and processes the queue as resources become available.

### Multi-Tenant GPU Sharing

`MultiTenantGPUScheduler` implements time-sliced GPU sharing:

- Priority-weighted round-robin scheduling
- Fair distribution across tenants
- Queue management for pending work
- Preemption for high-priority jobs

This allows multiple teams or services to share GPU resources fairly, with priorities reflecting business importance.

### Gradient Compression

At scale, communication overhead can dominate computation time. `GradientCompression` in `code/resource_management.py` reduces this overhead:

__Top-k Sparsification__[^topk-sgd]: Keep only the k largest gradient values. With k=1% of gradients, you achieve 100x compression. Error feedback—accumulating dropped gradients for the next iteration—preserves convergence.

[^topk-sgd]: Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training. \url{https://arxiv.org/abs/1712.01887}

__Quantization__: Reduce precision to 8-bit. This achieves 4x compression with minimal impact on training dynamics. Can be combined with sparsification for even higher compression.

The key insight is that gradients are highly compressible. Most gradient values are small and contribute little to learning. By focusing communication on the largest values and accumulating the rest, you can dramatically reduce bandwidth requirements without sacrificing convergence.



## Preparing for the Future

### Skills to Develop

__MoE and Sparse Architectures__: Understanding expert routing strategies (TopK, ReLU-based), load balancing mechanisms, and elastic expert scaling at inference time is increasingly essential as MoE becomes the default architecture.

__Edge-Cloud Coordination__: Speculative decoding implementation, memory bandwidth optimization for mobile, and quantization techniques (GPTQ, AWQ) open new deployment options that weren't practical until recently.

__Large-Scale Communication__: Torchcomms API and NCCLX/RCCLX backends, async checkpointing strategies, and heterogeneous hardware deployments enable bigger training runs and more efficient infrastructure utilization.

__Inference Optimization__: RadixAttention and PagedAttention internals, pipeline parallelism for long contexts, and multi-tenant GPU scheduling improve serving efficiency and reduce costs.

__Agentic Systems__: Multi-agent orchestration patterns, distributed tool execution, and reasoning chain optimization represent the next application frontier where distributed AI meets complex problem-solving.

### Staying Current

The field moves fast—techniques that seem cutting-edge today may be standard practice in six months. Staying current requires building habits around a few key information sources.

For day-to-day learning, arXiv is indispensable. New papers in cs.DC and cs.LG drop daily, and the best work often appears here months before conference publication. Set up alerts for keywords like "distributed training," "inference optimization," and "mixture of experts." Open-source projects are equally important: vLLM, SGLang, DeepSpeed, and Megatron-LM are where theory meets practice. Watch their release notes and GitHub discussions—that's where you'll learn what actually works at scale. Industry blogs from teams at xAI, Anthropic, Google, and Meta regularly publish technical deep-dives that reveal practical insights never found in papers. And don't overlook communities: Hugging Face forums, PyTorch Discuss, and the SGLang Discord are where practitioners share war stories and debug tricky issues together.

For deeper dives, conferences remain essential. Research venues like NeurIPS, ICML, and ICLR are where new algorithms and architectures debut—pay special attention to workshops on efficient ML and large-scale systems. Systems conferences like MLSys, OSDI, and SOSP focus on infrastructure, communication optimization, and production deployment; if you care about making things fast, these are essential reading. Industry events like GTC and PyTorch Conference are where vendors announce new hardware and frameworks, giving you a preview of what's coming in the next 12-18 months.

The References section at the end of this chapter provides URLs for the key papers and projects mentioned throughout.



## Summary

This chapter has explored the current state of distributed AI and the trends shaping its future:

1. __The great shift to inference__: Inference workloads now consume over 55% of AI infrastructure spending, with training becoming a smaller fraction of total compute. This reshapes infrastructure priorities and optimization targets.

2. __MoE dominance__: Architectures like DeepSeek-V3 demonstrate that 671B parameter models can be trained for $5.5M with only 37B active parameters per token. Understanding expert routing, load balancing, and parallelism is essential.

3. __On-device AI is practical__: Sub-billion parameter models, 4-bit quantization, and techniques like Per-Layer Embeddings enable LLMs on mobile devices. Memory bandwidth, not compute, is the bottleneck.

4. __100K+ GPU scale__: New communication APIs (Torchcomms, NCCLX/RCCLX) and async checkpointing enable training at unprecedented scale. Heterogeneous hardware support is increasingly important.

5. __Fault tolerance is essential__: At scale, failures are expected—roughly 100 per day in a 100K GPU cluster. Elastic training and async checkpointing are mandatory, not optional.

6. __Multimodal and agentic__: VLMs and multi-agent systems require new distributed patterns for cross-modal attention and tool execution. These represent the next application frontier.

7. __Federated learning as alternative paradigm__: When data can't be centralized due to privacy or regulatory constraints, federated learning offers a mature alternative. Techniques like FedProx and secure aggregation address the unique challenges of non-IID data and untrusted participants.

8. __Inference engine evolution__: SGLang achieves 16,215 tok/s with RadixAttention; vLLM's PagedAttention reduces KV cache waste to under 4%. Both support diverse hardware platforms.

The technologies covered in this book—DDP, FSDP, DeepSpeed, vLLM, SGLang—remain foundational. But the landscape continues to evolve rapidly. Power and cooling constraints, not silicon availability, are now the primary bottleneck. Understanding these trends and staying current with new tools will prepare you for the next wave of distributed AI innovations.

The future of distributed AI is being written now, by researchers pushing the boundaries and practitioners deploying at scale. The best way to predict that future is to help invent it.



<!-- include: exercises/torch.md if include_math -->
<!-- include: exercises/torch.md if include_torch -->

## References

__MoE Architectures__

- DeepSeek-V3 Technical Report: \url{https://arxiv.org/abs/2412.19437}
- LatentMoE: Toward Optimal Accuracy per FLOP: \url{https://arxiv.org/abs/2601.18089}
- MoSE: Mixture of Slimmable Experts: \url{https://arxiv.org/abs/2602.06154}
- Elastic MoE: Inference-Time Expert Scaling: \url{https://arxiv.org/abs/2501.03140}
- ReMoE: Fully Differentiable MoE with ReLU Routing: \url{https://arxiv.org/abs/2412.14711}

__Communication and Scale__

- NVIDIA NCCL: \url{https://developer.nvidia.com/nccl}
- Torchcomms API: \url{https://pytorch.org/blog/torchcomms/}
- RCCLX for AMD Platforms: \url{https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/}
- PyTorch Distributed Checkpoint: \url{https://pytorch.org/docs/stable/distributed.checkpoint.html}
- Async Checkpointing Improvements: \url{https://pytorch.org/blog/6x-faster-async-checkpointing/}
- Deep Gradient Compression: \url{https://arxiv.org/abs/1712.01887}

__Inference Engines__

- SGLang (RadixAttention): \url{https://arxiv.org/abs/2312.07104}
- SGLang Pipeline Parallelism: \url{https://lmsys.org/blog/2026-01-15-chunked-pipeline/}
- SGLang Documentation: \url{https://sgl-project.github.io/}
- vLLM (PagedAttention): \url{https://arxiv.org/abs/2309.06180}
- vLLM Documentation: \url{https://docs.vllm.ai/}
- Speculative Decoding: \url{https://arxiv.org/abs/2211.17192}

__Memory-Efficient Attention__

- Multi-head Latent Attention (DeepSeek-V2): \url{https://arxiv.org/abs/2405.04434}
- Ring Attention: \url{https://arxiv.org/abs/2310.01889}
- FlashAttention-2: \url{https://arxiv.org/abs/2307.08691}

__On-Device AI and Quantization__

- GPTQ: Post-Training Quantization: \url{https://arxiv.org/abs/2210.17323}
- AWQ: Activation-aware Weight Quantization: \url{https://arxiv.org/abs/2306.00978}
- KV Cache Compression: \url{https://arxiv.org/abs/2405.12981}
- Google LiteRT: \url{https://ai.google.dev/edge/litert}
- Gemma 3n: \url{https://ai.google.dev/gemma/docs/gemma-3n}

__Reasoning and Agentic AI__

- DeepSeek-R1: \url{https://arxiv.org/abs/2501.12948}
- ReAct: Reasoning and Acting in Language Models: \url{https://arxiv.org/abs/2210.03629}

__Fault Tolerance and Elastic Training__

- PyTorch Elastic: \url{https://pytorch.org/docs/stable/elastic/run.html}
- TorchSnapshot: \url{https://pytorch.org/torchsnapshot/}

__Federated Learning__

- FedAvg: Communication-Efficient Learning from Decentralized Data: \url{https://arxiv.org/abs/1602.05629}
- FedProx: Federated Optimization in Heterogeneous Networks: \url{https://arxiv.org/abs/1812.06127}
- SCAFFOLD: Stochastic Controlled Averaging: \url{https://arxiv.org/abs/1910.06378}
- Secure Aggregation for Privacy-Preserving ML: \url{https://eprint.iacr.org/2017/281}
- Flower Framework: \url{https://flower.ai/}
- NVIDIA FLARE: \url{https://github.com/NVIDIA/NVFlare}
- FedIT: Federated Instruction Tuning of LLMs: \url{https://arxiv.org/abs/2409.12568}
