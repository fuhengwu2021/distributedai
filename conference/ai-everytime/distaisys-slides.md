---
marp: true
theme: default
paginate: true
size: 16:9
html: true
backgroundColor: #fbf9f4
color: #33393e
style: |
  section {
    font-family: "Helvetica Neue", Arial, sans-serif;
    padding: 52px 68px;
    background: #fbf9f4 !important;
    justify-content: flex-start !important;
  }
  section h1,
  section h2,
  section > h1:first-child,
  section > h2:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }
  section h1 {
    color: #1f7a72 !important;
    font-size: 1.85em;
    font-weight: 800;
    line-height: 1.2;
    border-bottom: 3px solid #f4a261;
    padding-bottom: 6px;
    margin: 0 0 20px 0 !important;
  }
  section h2 {
    color: #e76f51 !important;
    font-size: 1.45em;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 20px 0 !important;
    padding: 0 !important;
  }
  section img {
    display: block;
    margin: 0.3em auto;
  }
  .two-columns {
    display: grid !important;
    grid-template-columns: 1.12fr 0.88fr !important;
    gap: 32px !important;
    align-items: center !important;
    width: 100% !important;
    margin-top: 6px !important;
  }
  .two-columns .left-content {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
  }
  .two-columns .left-content ul {
    margin: 0 !important;
    padding-left: 20px !important;
  }
  .two-columns .left-content li {
    margin-bottom: 8px !important;
    font-size: 0.88em !important;
    line-height: 1.35 !important;
  }
  .two-columns .right-img {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
  }
  .two-columns img {
    max-height: 340px !important;
    max-width: 100% !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
    margin: 0 auto !important;
    display: block !important;
  }
  section h3 {
    color: #3d84a8 !important;
  }
  section a {
    color: #3d84a8 !important;
  }
  section code {
    background: #eef1ff !important;
    color: #6a4bbc !important;
    border-radius: 4px;
  }
  section pre {
    background: #f4f6ff !important;
    border: 1px solid #dfe4fb;
    border-radius: 10px;
  }
  section pre code {
    background: transparent !important;
    color: #3a3f6b !important;
  }
  section strong {
    color: #e07a1f !important;
  }
  section em {
    color: #2a9d8f !important;
  }
  section li::marker {
    color: #f4a261 !important;
  }
  section.lead {
    justify-content: center !important;
    align-items: flex-start !important;
    background: linear-gradient(135deg, #fff1e0 0%, #ffe3ee 35%, #e3edff 70%, #e0fbf1 100%) !important;
  }
  section.lead h1 {
    font-size: 1.8em;
    border-bottom: none;
    color: #1f7a72 !important;
  }
  section.lead h2 {
    color: #7454b3 !important;
    font-weight: 400;
  }
  section.lead strong {
    color: #d9622b !important;
  }
  section footer {
    color: #9a9fa8 !important;
    font-size: 0.55em;
  }
  section table {
    font-size: 0.75em;
    border-radius: 10px;
    overflow: hidden;
    border-collapse: separate;
    border-spacing: 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  }
  section thead th {
    background: #2a9d8f !important;
    color: #ffffff !important;
  }
  section tbody tr:nth-child(even) {
    background: #fdf1de !important;
  }
  section tbody tr:nth-child(odd) {
    background: #ffffff !important;
  }
  section td, section th {
    padding: 0.4em 0.7em;
  }
  section blockquote {
    border-left: 5px solid #f4a261;
    color: #5b6169 !important;
    background: #fff6ea !important;
    padding: 0.4em 1em;
    border-radius: 0 8px 8px 0;
  }
  section.cover {
    display: flex;
    flex-direction: column;
    justify-content: center;
    background: linear-gradient(135deg, #fff7ed 0%, #ffedf2 35%, #edf4ff 70%, #e6faf5 100%) !important;
    padding: 48px 64px;
  }
  section.cover .badge {
    display: inline-block;
    font-size: 0.6em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #ffffff;
    background: linear-gradient(135deg, #e76f51, #f4a261);
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
    width: fit-content;
    box-shadow: 0 2px 8px rgba(231, 111, 81, 0.25);
  }
  section.cover h1 {
    font-size: 2.1em;
    font-weight: 800;
    line-height: 1.15;
    color: #1a535c !important;
    border-bottom: none;
    padding-bottom: 0;
    margin: 0 0 10px 0;
  }
  section.cover h2 {
    font-size: 1.02em;
    font-weight: 500;
    line-height: 1.35;
    color: #636b78 !important;
    margin: 0 0 18px 0;
  }
  section.cover .divider {
    width: 48px;
    height: 4px;
    background: #f4a261;
    border-radius: 2px;
    margin-bottom: 20px;
  }
  section.cover .footer-meta {
    margin-top: 6px;
  }
  section.cover .author-name {
    font-size: 1.05em;
    font-weight: 700;
    color: #d9622b;
    margin-bottom: 2px;
  }
  section.cover .author-title {
    font-size: 0.78em;
    color: #5b6169;
    margin-bottom: 4px;
  }
  section.cover .author-link {
    font-size: 0.75em;
    margin-bottom: 8px;
  }
  section.cover .conf-info {
    font-size: 0.72em;
    color: #8b929e;
  }
---

<!-- _class: cover -->
<!-- _paginate: false -->

![bg right:34% 88%](../../cover/7x10/amazon_cover.jpg)

<span class="badge">Conference Keynote</span>

# From Single GPU to<br>Production Clusters

## Building Distributed AI Systems for Training and Inference

<div class="divider"></div>

<div class="footer-meta">
  <div class="author-name">Henry Fuheng Wu</div>
  <div class="author-title">Author of <em>Distributed AI Systems</em></div>
  <div class="author-link">🔗 <a href="https://www.linkedin.com/in/henrywoo/" style="color: #2b7a78 !important;">linkedin.com/in/henrywoo</a></div>
  <div class="conf-info">Global Data & AI Virtual Tech Conference 2026 · August 22–24</div>
</div>

<!--
Welcome everyone, thanks for having me. I'm Henry, and today I want to walk through what it actually takes to go from a single-GPU experiment to a production-grade distributed AI system, for both training and inference. This is drawn from hands-on production work at places like Oracle, Wells Fargo, Uber plus material from my book Distributed AI Systems. Let's get started.
-->

---

## About the Speaker

**Henry Fuheng Wu**

  <div class="author-link">🔗 <a href="https://www.linkedin.com/in/henrywoo/" style="color: #2b7a78 !important;">linkedin.com/in/henrywoo</a></div>
<hr>

- AI leader and distributed systems expert
- Author of **Distributed AI Systems**, *Above the Clouds*, *Mathematics for AI and Machine Learning*
- Amazon Best-Selling Author in AI · Judge, International AI Innovation Olympiad
- Production AI experience across **Wells Fargo, Oracle, Uber, and Oscar Health**
- Built large-scale inference, distributed training, GPU infrastructure, and real-time data platforms

<!--
A quick bit about my background so you know where this is coming from. I've spent my career building production AI systems, distributed training, large-scale inference, GPU infrastructure, across Oracle, Wells Fargo, Uber, Oscar Health. I've also written about this space, including my book Distributed AI Systems, and I judge AI competitions on the side. Everything in this talk is grounded in things I've actually shipped, not just theory.
-->

---

## The Most Lucrative Skill Set on the Planet

![h:350](meta-poached-pang-with-over-200M-dollars.png)

**Meta paid Apple's Ruoming Pang a package reportedly over $200M** — every major lab is aggressively competing for the narrow pool of talent who can scale these systems.

<!--
Let's ground this in reality. Last year, Meta poached Apple's Ruoming Pang with a compensation package reportedly north of two hundred million dollars. Reports quickly followed of Microsoft and OpenAI counter-raiding talent with nine-figure bonuses. Why? Because the bottleneck in AI today is not model design ideas — it's the specialized engineering discipline required to train, scale, and serve these models across thousands of GPUs without catastrophic slowdowns or failures.
-->

---

## I've Worked in These Trenches — Why I Built This Blueprint

![h:130](uber-ai-infra-mark-lee.png)

That's **Mark Lee**, my teammate at Uber ATG — same project, same interview panel, same team. Top engineers aren't wizards — they just had the rare opportunity to master the full production lifecycle. That knowledge has been siloed in a few labs. To bridge that gap, I wrote ***Distributed AI Systems*** and published the complete runnable recipes.

<!--
And that Mark Lee name isn't just a headline to me, this is personal. Back at Uber ATG, Mark Lee was on my team. We sat on the same interview panel together, running tech screens side by side. When I say this talent war is real, it's not from a news feed, it's right in my network. But here's the real point: engineers in these positions aren't superheroes, they just had the rare opportunity to work across the full lifecycle at scale. That knowledge is siloed inside a handful of top labs. It took me a decade across Oracle, Wells Fargo, and Uber to piece it all together — and that's exactly why I wrote Distributed AI Systems and open-sourced the code, to give every engineer that complete production playbook.
-->

---

## Agenda

### 1 · Core Bottlenecks & Hardware Limits
GPU memory wall · Interconnect bandwidth cliffs · Communication overhead

### 2 · Distributed Training in Practice
PyTorch DDP & FSDP2 · DeepSpeed ZeRO stages · Megatron 3D Parallelism

### 3 · Production Inference & Cluster Ops
KV cache & vLLM / SGLang · Real benchmark data · Failure postmortems

<!--
Here is our roadmap today in three concise blocks. First, why distribution is mandatory and where the real memory and network cliffs are. Second, the training strategy ladder from DDP to FSDP and Megatron 3D parallelism. Third, the shift to inference with vLLM and SGLang, ending with real benchmark numbers and hard-won production failure postmortems.
-->

---

## Why Modern AI Requires Distribution

- Model sizes have outpaced single-GPU memory for years — a 70B-parameter model alone needs **~140GB+** just for weights in FP16
- Training adds optimizer states, gradients, and activations on top of weights
- **The scale challenge is not optional** — it's a hard constraint of modern foundation models

**Rule of thumb:** if your model + optimizer state + activations don't fit in one GPU's memory, or a single GPU can't finish training in a reasonable time, you need a distributed strategy.

<!--
Let's ground this in numbers. A 70-billion-parameter model needs roughly 140GB just to hold the weights in FP16, before you've loaded a single batch. Once you add optimizer states, gradients, and activations for training, that number multiplies several times over. So the question isn't whether to distribute, it's when, and the rule of thumb I use is simple: if it doesn't fit in one GPU's memory, or training would take unreasonably long on one GPU, you're in distributed territory.
-->

---

## GPU Memory and Networking Bottlenecks

**Compute is rarely the bottleneck — memory bandwidth and communication are.**

- **Memory Bandwidth Cliff** — moving weights and activations on/off HBM dominates latency
- **Interconnect Hierarchy** — NVLink (~900 GB/s intra-node) vs. InfiniBand / RoCE (400–800 Gbps inter-node)
- **NUMA & CPU Affinity** — poor host pinning silently degrades PCIe transfer bandwidth
- **Collective Communication Overhead** — AllReduce and ReduceScatter traffic scales directly with model & cluster size

Understanding *where* the bytes move is the first step to scaling efficiently.

<!--
This is the slide I wish more engineers internalized before they start optimizing compute. In practice, compute is rarely your bottleneck, memory bandwidth and communication are. NVLink inside a node is fast, but the moment you cross node boundaries onto InfiniBand or RoCE, there's a steep bandwidth cliff. Bad NUMA or CPU pinning can quietly throttle your PCIe transfers too. And every collective operation, AllReduce, AllGather, ReduceScatter, costs you traffic that scales with both model size and cluster size. If you understand where the bytes move, you understand where your performance problems will come from.
-->

---

## Distributed Training Strategies

| Strategy | Shards | Best for |
|---|---|---|
| **Data Parallelism (DDP)** | Data batches | Model fits on one GPU |
| **Fully Sharded Data Parallel (FSDP)** | Params, grads, optimizer state | Model too large for one GPU |
| **Tensor Parallelism** | Individual layers | Very large layers, high-bandwidth links |
| **Pipeline Parallelism** | Model depth (stages) | Many GPUs, deep models |
| **Sequence / Context Parallelism** | Sequence dimension | Long-context training |

Production systems increasingly combine several of these — **hybrid parallelism**.

<!--
Here's the strategy landscape at a glance. Data parallelism, which DDP implements, is your starting point when the model fits on one GPU. FSDP shards parameters, gradients, and optimizer states when it doesn't. Tensor and pipeline parallelism shard the model itself, its layers or its depth, for very large models. Sequence and context parallelism handle long-context training. And in practice, production systems rarely use just one of these, they combine several, which is what we call hybrid parallelism. We'll walk through DDP and FSDP in detail next.
-->

---

## PyTorch DDP: How It Works

- Each rank holds a **full copy** of the model; only gradients are synchronized
- **Gradient bucketing** groups small gradients so communication overlaps with backward computation
- **Ring-AllReduce** averages gradients across ranks without a single bottleneck node
- Mixed precision + gradient scaling maintains throughput without loss of accuracy

```bash
torchrun --nproc_per_node=8 --nnodes=2 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 train.py
```

*Trade-off:* simple and robust, but every rank needs the **full model & optimizer** in memory.

<!--
DDP is the workhorse of distributed training, and it's worth understanding exactly what it does. Every rank keeps a full copy of the model, only gradients get synchronized, via AllReduce, after each backward pass. The clever part is gradient bucketing: PyTorch groups gradients into buckets so communication can overlap with backward computation instead of waiting for it to finish. Combined with mixed precision, this gets you excellent throughput. The catch is memory, every single rank needs the full model, optimizer states and all, which is exactly the wall we hit next.
-->

---

## FSDP: Breaking the Memory Wall

<div class="two-columns">
  <div class="left-content">
    <ul>
      <li><strong>Zero Redundancy</strong>: shards parameters, gradients, and optimizer states across ranks.</li>
      <li><strong>Just-in-Time Gather</strong>: un-shards layers on-the-fly for forward/backward, then frees them.</li>
      <li><strong>FSDP2 & DeviceMesh</strong>: clean 2D hybrid sharding (intra-node TP + inter-node FSDP).</li>
      <li><strong>Outcome</strong>: 70B+ models become trainable on standard GPU clusters.</li>
    </ul>
  </div>
  <div class="right-img">
    <img src="../../chapter4-scaling-with-fully-sharded-data-parallel-fsdp/img/ddp_fsdp_mem.png" />
  </div>
</div>

<!--
FSDP is the answer to that memory wall. Instead of every rank holding a full copy, FSDP shards parameters, gradients, and optimizer states across ranks, and only gathers the full parameters just-in-time for each layer's forward and backward pass, then frees them again. FSDP2 cleaned this up significantly with per-parameter sharding and the DeviceMesh API, which makes 1D and 2D, that is hybrid, sharding topologies much easier to reason about and configure. The net effect: you can train models that simply would not fit on any single GPU, using hardware you already have.
-->

---

## Going Further: DeepSpeed & Megatron

<div class="two-columns">
  <div class="left-content">
    <ul>
      <li><strong>DeepSpeed ZeRO</strong>: Stage 1 (optimizer) &rarr; Stage 2 (+ gradients) &rarr; Stage 3 (+ params).</li>
      <li><strong>ZeRO-Offload / Infinity</strong>: spills state to NVMe/host RAM for trillion-scale parameters.</li>
      <li><strong>Megatron-LM</strong>: shards computation along tensor (intra-node) and pipeline (inter-node) axes.</li>
      <li><strong>3D Parallelism</strong>: combines TP + PP + DP to train trillion-parameter models.</li>
    </ul>
  </div>
  <div class="right-img">
    <img src="../../chapter5-beyond-state-sharding-with-deepspeed-and-megatron/img/zero_stages_comparison.png" />
  </div>
</div>

<!--
When FSDP still isn't enough, there are two more levers. DeepSpeed's ZeRO takes state sharding all the way: Stage 1 shards optimizer states, Stage 2 adds gradients, Stage 3 adds parameters, which is functionally similar to FSDP. ZeRO-Offload and ZeRO-Infinity go further, spilling to CPU or even NVMe storage for models that exceed your entire cluster's GPU memory. Megatron-LM attacks the problem from a different axis entirely, it shards the computation itself, splitting individual layers with tensor parallelism and model depth with pipeline parallelism. Combine these with data and sequence parallelism, and this is genuinely how trillion-parameter models get trained.
-->

---

## From Training to Inference

Inference has a fundamentally different profile:

- **Prefill** (compute-bound) vs. **Decode** (memory-bound, one token at a time)
- The **KV cache** avoids recomputing attention for prior tokens — but grows with every generated token
- Naive KV cache allocation fragments GPU memory and wastes capacity
- This is the "out of memory" problem that shapes every modern inference engine

<!--
Now let's shift from training to inference, because it's a fundamentally different problem. Inference has two very different phases: prefill, which is compute-bound and processes your whole prompt at once, and decode, which is memory-bound and generates one token at a time. The KV cache is what makes decode efficient, it avoids recomputing attention over every previous token, but it grows continuously as generation proceeds. If you allocate that cache naively, you fragment GPU memory badly, and that's really the root of the out-of-memory problems that plague naive inference serving.
-->

---

## vLLM: Solving KV Cache Fragmentation

<div class="two-columns">
  <div class="left-content">
    <ul>
      <li><strong>PagedAttention</strong>: manages KV cache like OS virtual memory (non-contiguous fixed blocks).</li>
      <li><strong>Eliminates Waste</strong>: cuts down up to 80% memory waste from internal fragmentation & padding.</li>
      <li><strong>Higher Concurrency</strong>: dramatically increases batch density & tokens/sec per GPU.</li>
      <li><strong>Scale-Out Ready</strong>: combines with Tensor Parallelism for high-throughput serving.</li>
    </ul>
  </div>
  <div class="right-img">
    <img src="../../chapter6-distributed-inference-fundamentals-and-vllm/img/padding_vs_paged.png" />
  </div>
</div>

<!--
vLLM's core contribution is PagedAttention, which treats the KV cache the way an operating system treats virtual memory: fixed-size blocks, no fragmentation, and no wasted padding. That alone dramatically increases how many requests you can batch together on one GPU, which directly increases throughput. And vLLM scales out the same way training does, tensor parallelism for large models, plus pipeline and data parallelism for further scale-out. If you're serving open models at any real scale today, there's a good chance vLLM is already part of your stack.
-->

---

## SGLang: Cross-Request Optimization

A different philosophy — optimize *across* requests, not just within one:

- **RadixAttention** — reuses shared prompt prefixes across requests via a radix tree cache
- **Zero-overhead scheduler** and **operator fusion** for lower per-step latency
- **Prefill/Decode disaggregation** — separates compute-bound and memory-bound phases onto different workers
- **Router-based architecture** — a model gateway load-balances across replicas, TP/PP/DP workers

Best fit for high cache-reuse workloads: chat, agents, RAG, few-shot prompting.

<!--
SGLang takes a different angle: instead of optimizing a single request, it optimizes across requests. RadixAttention keeps a radix tree of shared prompt prefixes, so if two requests share a system prompt or few-shot examples, that computation gets reused instead of repeated. Add a zero-overhead scheduler, operator fusion, and prefill/decode disaggregation, which separates the compute-bound and memory-bound phases onto different workers, and you get real latency wins. This shines especially in chat, agent, and RAG workloads, where prefix reuse across requests is high.
-->

---

## Benchmarking: What Real Production Numbers Look Like

| Workload / Benchmark | Baseline (Naive PyTorch) | Optimized (FSDP / vLLM) | Production Gain |
|---|---|---|---|
| **70B Training VRAM / GPU** (8x H100) | OOM (>140 GB needed) | **~34 GB** (FSDP2 Full Shard) | **Trainable without offloading** |
| **DDP Scaling Efficiency** (16 GPUs) | 94.2% (Intra-node NVLink) | **71.8%** (100GbE inter-node) | **22.4% penalty without RoCE/IB** |
| **Inference TTFT (p99)** (4K context) | 1,840 ms (Unpaged) | **390 ms** (PagedAttention) | **4.7x faster first token** |
| **Serving Throughput** (Concurrent 64) | 48 tokens/sec/GPU | **215 tokens/sec/GPU** | **4.5x higher token density** |

<!--
None of this matters if you can't measure it. Here are real numbers from our benchmarks. For training a 70B model on 8 H100s, naive execution OOMs immediately; FSDP2 brings per-GPU memory down to 34GB. In scaling efficiency, crossing from intra-node NVLink to standard 100Gb Ethernet drops efficiency from 94% to 71% — that is your communication bottleneck in black and white. On the inference side, PagedAttention slashes p99 TTFT by nearly 5x and boosts concurrent throughput by 4.5x. All reproducible scripts are in the book's repository.
-->

---

## Production Failure Modes: Hard-Won Postmortems

- **NCCL Silent Throttling** — Inter-node fallback from RoCE to TCP due to MTU mismatch; throughput dropped 68% with zero error logs
- **KV Cache Memory Fragmentation** — Naive contiguous memory allocation caused serving OOM crashes at only 48% actual VRAM usage
- **Straggler Node Cascade** — A single degraded PCIe link stalled synchronous AllReduce rings, idling 63 other healthy GPUs
- **Checkpoint I/O Storms** — Unsharded multi-terabyte checkpoints saturated storage bandwidth, turning 10-minute saves into job timeouts

**Takeaway:** Design for failure from day one. Health checks, elastic recovery, and non-blocking I/O are mandatory at scale.

<!--
Let me close with four concrete production failure modes I've personally debugged in production clusters. First, NCCL silent throttling: a subtle network MTU mismatch caused inter-node traffic to quietly drop from RoCE to TCP, degrading throughput by 68% without throwing a single error. Second, KV cache fragmentation causing OOMs when half the GPU memory was still free. Third, a single straggler GPU with degraded PCIe bandwidth stalling the entire AllReduce ring. And fourth, checkpoint storms that took down shared storage. These aren't theoretical — they are the operational realities of distributed systems.
-->

---

## Key Takeaways

- Distribution is driven by **memory and communication limits**, not just raw compute
- Start simple — **DDP** — and move to **FSDP/DeepSpeed/Megatron** only when memory forces you to
- Inference is a different problem from training: **KV cache management and cross-request reuse** dominate
- **Benchmark relentlessly** — percentile latency and scaling efficiency, not just averages
- Production reliability requires designing for failure, not just for the happy path

<!--
So if you remember five things from this talk: distribution is driven by memory and communication limits, not just raw compute. Start with DDP, and only move to FSDP, DeepSpeed, or Megatron when memory actually forces your hand. Inference is a genuinely different problem from training, KV cache management and cross-request reuse dominate there. Benchmark relentlessly, and look at percentiles, not averages. And build for failure from the start, because at scale, something is always failing somewhere.
-->

---

<!-- _class: cover -->
<!-- _paginate: false -->

# Thank You!

## Continue Learning & Build with Us

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 16px 0;">

<div style="background: rgba(255, 255, 255, 0.75); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
  <div style="font-weight: 700; color: #1f7a72; font-size: 0.95em; margin-bottom: 4px;">📖 Read the Book</div>
  <div style="font-size: 0.78em; color: #555; line-height: 1.4;"><strong>Distributed AI Systems</strong><br>Hands-on guide to training, inference & cluster scaling</div>
</div>

<div style="background: rgba(255, 255, 255, 0.75); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
  <div style="font-weight: 700; color: #e76f51; font-size: 0.95em; margin-bottom: 4px;">💻 Clone & Run Code</div>
  <div style="font-size: 0.78em; color: #555; line-height: 1.4;"><strong>GitHub:</strong> <a href="https://github.com/PacktPublishing/Distributed-AI-Systems" style="color: #2a9d8f;">PacktPublishing/Distributed-AI-Systems</a><br>Full benchmarks, DDP, FSDP & vLLM scripts</div>
</div>

<div style="background: rgba(255, 255, 255, 0.75); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
  <div style="font-weight: 700; color: #3d84a8; font-size: 0.95em; margin-bottom: 4px;">🌐 Technical Articles</div>
  <div style="font-size: 0.78em; color: #555; line-height: 1.4;"><strong>Website:</strong> <a href="https://distaisys.com" style="color: #2a9d8f;">distaisys.com</a><br>Deep dives, architecture teardowns & updates</div>
</div>

</div>


<!--
Thank you all for your time today! We covered the foundational principles from GPU memory and interconnects to DDP, FSDP, and inference with vLLM/SGLang. If you want to go deeper and run every single script, test case, and benchmark yourself, head over to our GitHub repository at github.com/PacktPublishing/Distributed-AI-Systems and check out distaisys.com for ongoing deep dives. The complete blueprint is in Distributed AI Systems. Feel free to connect with me on LinkedIn, and I'd love to take your questions now!
-->
