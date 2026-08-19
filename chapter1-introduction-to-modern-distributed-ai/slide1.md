---
marp: true
theme: default
paginate: true
size: 16:9
backgroundColor: #fbf9f4
color: #33393e
style: |
  section {
    font-family: "Helvetica Neue", Arial, sans-serif;
    padding: 56px 72px;
    background: #fbf9f4 !important;
  }
  section.lead {
    justify-content: center !important;
    align-items: flex-start !important;
    background: linear-gradient(135deg, #fff1e0 0%, #ffe3ee 35%, #e3edff 70%, #e0fbf1 100%) !important;
  }
  section img {
    display: block;
    margin: 0.3em auto;
  }
  section h1 {
    color: #1f7a72 !important;
    font-size: 1.9em;
    border-bottom: 4px solid #f4a261;
    padding-bottom: 0.2em;
  }
  section h2 {
    color: #e76f51 !important;
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
  section.lead h1 {
    font-size: 2.5em;
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
    font-size: 0.7em;
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
    padding: 0.35em 0.6em;
  }
  section blockquote {
    border-left: 5px solid #f4a261;
    color: #5b6169 !important;
    background: #fff6ea !important;
    padding: 0.4em 1em;
    border-radius: 0 8px 8px 0;
  }
---

<!-- _class: lead -->

# Why Modern AI Needs Distributed Systems
## Resource Estimation, the Decision Framework, and Your First Distributed Training Run

**Henry Fuheng Wu**
Author of *Distributed AI Systems*

Global Data & AI Virtual Tech Conference 2026 · August 22–24

<!--
Welcome, everyone. This talk is a deep dive into Chapter 1 of my book, Distributed AI Systems — the chapter that answers a question I get asked constantly: how do you actually know when you need distributed training or inference, versus when a single GPU is enough? We're going to work through real memory math, a decision framework you can apply this week, and a hands-on distributed training example with real numbers. Let's get into it.
-->

---

## About the Speaker

**Henry Fuheng Wu**

- AI leader and distributed systems expert
- Author of *Above the Clouds*, *Mathematics for AI and Machine Learning*, and *Distributed AI Systems*
- Amazon Best-Selling Author in AI · Judge, International AI Innovation Olympiad
- Production AI experience across **Wells Fargo, Oracle, Uber, Oscar Health, Bloomberg, Google, and WorldQuant**
- Built large-scale inference, distributed training, GPU infrastructure, and real-time data platforms

<!--
Quick background so you know where this material comes from. I've spent my career building production AI systems, distributed training, large-scale inference, and GPU infrastructure at Wells Fargo, Oracle, Uber, Oscar Health, Bloomberg, Google, and WorldQuant. Everything in this chapter, and in this talk, is grounded in decisions I've actually had to make on real infrastructure budgets, not just theory.
-->

---

## Agenda

1. Why modern AI requires distribution
2. Estimating memory: training vs. inference
3. The modern AI model lifecycle
4. Decision framework — do you actually need distributed systems?
5. Hands-on: single GPU vs. multi-GPU, real numbers
6. The distributed AI stack, from framework to hardware
7. Process groups, ranks, and collective operations
8. Key takeaways

<!--
Here's the roadmap. We'll start with why distribution stopped being optional, then get concrete with memory math for both training and inference, because that's the calculation that should drive every infrastructure decision you make. From there, a decision framework you can literally apply on Monday morning. Then we'll get hands-on with real benchmark numbers from single GPU up to eight GPUs, both for training and inference. We'll close by going under the hood on how PyTorch actually moves those gradients around, process groups, ranks, and the collective operations that power all of this.
-->

---

## Why Modern AI Requires Distribution

- A few years ago, most models trained on a single GPU — ResNet-50 on ImageNet took a couple of days
- Today, a 70B-parameter language model would take **months** on a single GPU, if it even fits in memory
- Frontier models now span **1–10 trillion parameters**; training requires thousands of GPUs working together
- Model size and compute needs grew **exponentially** — single-GPU memory and compute grew **linearly** at best

**The era of single-machine AI is over. Modern AI systems are distributed by design.**

<!--
A few years ago, you could train most models people cared about on one GPU. ResNet-50 on ImageNet, a couple of days, done. Today, a 70-billion-parameter language model would take months on a single GPU, assuming it even fits in memory in the first place. And we're not talking about 70 billion as the ceiling anymore — frontier models are now in the one-to-ten-trillion-parameter range, and training them takes thousands of GPUs working in concert. The core problem is a mismatch: model size and compute requirements grew exponentially, while single-GPU memory and compute grew linearly at best. That gap is why distribution isn't a nice-to-have anymore, it's the default architecture.
-->

---

## The Scale Challenge, in Real Numbers

Take a 70B parameter model:

- **FP32:** 280 GB just for weights — no mainstream datacenter GPU holds that today
- Even an **H200 (141 GB)** or **B200 (192 GB)** falls short in FP32
- **BF16** cuts that to 140 GB — still needs multiple GPUs just to *load* the model
- And that's before adding gradients, optimizer states, and activations for training

**Rule of thumb:** if model + optimizer state + activations don't fit on one GPU, or training would take unreasonably long on one, you're in distributed territory.

<!--
Let's ground this in actual numbers instead of vibes. A 70-billion-parameter model in full FP32 precision needs 280 gigabytes just to hold the weights. There is no mainstream datacenter GPU that holds that today — even an H200 at 141 gigabytes, or a B200 at 192 gigabytes, falls short. Switch to BF16 and you halve that to 140 gigabytes, but you still need multiple GPUs just to load the model, before you've trained a single step. And that's before you add gradients, optimizer states, and activations. My rule of thumb is simple: if the model plus optimizer state plus activations doesn't fit on one GPU, or training would take unreasonably long on one, you're in distributed territory, full stop.
-->

---

## How Much Memory Per Parameter?

| Format | Bytes | Primary Usage |
|---|---|---|
| **FP32** | 4 | Training, high-precision inference |
| **BF16** | 2 | Training (preferred), inference |
| **FP16** | 2 | Inference |
| **FP8** (E4M3 / E5M2) | 1 | Inference weights / training gradients |
| **Int8** | 1 | Quantized inference |
| **Int4** | 0.5 | Extreme-compression quantized inference |

A 7B model: **28 GB** (FP32) → **14 GB** (BF16) → **7 GB** (Int8) → **3.5 GB** (Int4)

<!--
Before you estimate anything, you need to know bytes per parameter. FP32 is 4 bytes, BF16 and FP16 are 2 bytes, FP8 and Int8 are 1 byte, Int4 is half a byte. For training, BF16 is preferred over FP16 because it has the same dynamic range as FP32, just less precision — that makes it far more numerically stable during training. Concretely, a 7-billion-parameter model is 28 gigabytes in FP32, 14 in BF16, 7 in Int8, and 3.5 in Int4. This one table is the starting point for every capacity estimate we're about to do.
-->

---

## Training Memory: More Than Just Weights

Training needs weights **+ gradients + optimizer states + activations**:

- **Gradients**: one value per parameter — same size as the weights (1×)
- **Optimizer state** depends on the optimizer:
  - SGD: ~0× (just a scalar learning rate)
  - SGD + Momentum: 1×
  - **Adam / AdamW**: 2× — stores both a first-moment and second-moment estimate per parameter
- **Activations** scale with batch size and sequence length

Adam's adaptive learning rates usually converge faster — worth the 2× memory cost in most cases.

<!--
Model weights are just the starting point for training. You also need gradients, one value per parameter, so that's another full copy of the model size. Then optimizer state, and this is where it gets interesting: plain SGD needs almost nothing extra, just a scalar learning rate. SGD with momentum needs one extra copy. But Adam and AdamW, which is what most of you are actually using, need two extra copies, because Adam tracks both a first-moment and second-moment estimate per parameter. That's why Adam costs twice the model size in optimizer state alone. And on top of all that, activations from the forward pass, which scale with your batch size and sequence length. Despite the memory cost, Adam's adaptive learning rates usually converge fast enough to be worth it.
-->

---

## Where Peak Memory Actually Happens

![h:430](img/training_memory_timeline.png)

7B model, Adam, BF16: 14GB weights + 28GB optimizer + 14GB gradients + 12GB activations = **peak 68GB** during `backward()`

<!--
Here's the training memory timeline for a 7-billion-parameter model with Adam, all in BF16. During the forward pass, you've got weights, optimizer state, and growing activations. The peak happens during the backward pass, loss dot backward, because that's the one moment where activations and gradients both exist in memory at the same time: 14 gigabytes of weights, 28 of optimizer state, 14 of gradients, and 12 of activations, adding up to a 68 gigabyte peak. Once the backward pass finishes, activations get freed and memory drops back down for the optimizer step. This is exactly why gradient accumulation and activation checkpointing work — they specifically target that peak moment.
-->

---

## Inference Memory: Weights + KV Cache

- Inference only needs a **forward pass** — no gradients, no optimizer states
- But the **KV cache** grows with every generated token, scaling with batch size × sequence length × model depth
- A 70B model in BF16: **140 GB** for weights alone
- Add a batch of 32 at 2048 tokens: **+20–40 GB** of KV cache → **160–180 GB total**
- Ultra-long-context models can push KV cache alone past **1 TB** for a single sequence

**This is the out-of-memory problem that shapes every modern inference engine** — more in the vLLM/SGLang chapters.

<!--
Inference looks much lighter than training on paper — no gradients, no optimizer states, just a forward pass. But there's a catch: the KV cache. It stores key-value pairs from every previous token so you don't recompute attention over the whole sequence each time, and it keeps growing as generation proceeds. A 70-billion-parameter model in BF16 needs 140 gigabytes just for weights. Add a batch of 32 requests at 2048 tokens each, and the KV cache alone adds another 20 to 40 gigabytes, pushing you to 160 to 180 gigabytes total — already past a single A100. And with today's ultra-long-context models, the KV cache alone can blow past a terabyte for a single sequence. This is exactly the problem that PagedAttention in vLLM, and RadixAttention in SGLang, were built to solve — we cover both later in the book.
-->

---

## Quick Reference: Memory by Model Size

| Model Size | BF16 Weights | Training (BF16 + Adam) | Inference (BF16) |
|---|---|---|---|
| 1B | 2 GB | ~8 GB | 2–4 GB |
| 7B | 14 GB | ~60–70 GB | 14–20 GB |
| 13B | 26 GB | ~110–130 GB | 26–35 GB |
| 70B | 140 GB | ~600–700 GB | 140–180 GB |

**Get this wrong and you either hit OOM errors or waste money over-provisioning.**

<!--
This is the table I keep pinned to my desk. Notice the gap between the weight size and the training requirement — training a 7B model needs roughly four to five times the weight-only memory once you add gradients, optimizer states, and activations. That gap is exactly why people get surprised by out-of-memory errors: they size their cluster off the weights alone and forget everything training adds on top. Get this math wrong in either direction and you either hit OOM in production, or you over-provision and burn budget on GPUs you didn't need.
-->

---

## The Modern AI Model Lifecycle

![w:560](img/mdlc.png)

Data engineering → **Training** → Inference optimization → Benchmarking → **Production deployment** → Feedback loop

<!--
Building AI models isn't a one-shot process, it's a cycle. You start with data engineering, collecting and cleaning terabytes of data. Then training: forward passes, backprop, hyperparameter tuning, fine-tuning. Once trained, models go through inference optimization, quantization, operator fusion, kernel optimization. Before deployment, comprehensive benchmarking checks both accuracy and engineering performance. Production deployment brings autoscaling, load balancing, and observability into the picture. And production feedback flows back into data collection priorities, closing the loop. This book focuses specifically on the distributed-systems pieces of this cycle: training, inference, benchmarking, and deployment.
-->

---

## Training vs. Inference vs. Serving

| Aspect | Training | Inference | Serving |
|---|---|---|---|
| **Goal** | Learn parameters | Generate predictions | Provide access |
| **Memory** | High (activations + gradients) | Medium (weights + KV cache) | Variable |
| **Communication** | Frequent (gradients) | Medium | API-level |
| **Latency** | Hours to days | ms to seconds | Milliseconds |
| **Throughput** | Samples/sec | Tokens/sec | Requests/sec |

These are three genuinely different engineering problems — don't solve them with the same tools.

<!--
These three terms get used interchangeably, but they're genuinely different engineering problems. Training is about learning parameters, it's memory-hungry and needs frequent gradient communication, and it runs for hours to days. Inference is about generating predictions, lighter on memory, but still needs to hit millisecond-to-second latency. Serving is the production system around inference — the API gateway, the load balancer, the multi-model routing, the observability. If you design your serving infrastructure using training assumptions, or vice versa, you'll build the wrong thing.
-->

---

## Decision Framework: Do You Need Distributed Systems?

![h:460](img/decision_tree.png)

**Distributed systems add complexity, communication overhead, and cost. Use them when you have to, not when you want to.**

<!--
This is the decision tree from the chapter, and it's the single most useful slide in this talk. For training: does the model fit in memory? If not, you need model or parameter parallelism, FSDP or tensor parallelism. If it fits but training drags on for weeks, data parallelism speeds it up. If you're fine-tuning, LoRA or QLoRA change the equation entirely — QLoRA on a 70B model fits in a single 48 gigabyte GPU. For inference: does the model exceed single-GPU memory? If yes, model parallelism. If it fits but you need thousands of requests per second, distributed inference. And if both memory and throughput fit comfortably on one GPU, stay there and just use an optimized engine like vLLM or SGLang. Distributed systems add real complexity and cost — use them because the math says you have to, not because it sounds impressive.
-->

---

## Hands-On: Single-GPU Baseline

ResNet18 on FashionMNIST (70,000 grayscale images, 10 classes) — 3 epochs:

```
Epoch 1/3, Loss: 0.4164, Accuracy: 84.94%
Epoch 2/3, Loss: 0.2936, Accuracy: 89.17%
Epoch 3/3, Loss: 0.2531, Accuracy: 90.58%

Total training time: 8.78s
```

```bash
python code/single_gpu_baseline.py
```

<!--
Let's get hands-on. We start with a simple, fast baseline: ResNet18 on FashionMNIST, seventy thousand grayscale images across ten clothing categories. Small enough to iterate quickly, but real enough to demonstrate the concepts. Three epochs on a single GPU gets us to just over 90 percent accuracy in 8.78 seconds. This is our reference point — everything from here is measured as a speedup relative to this number.
-->

---

## Multi-GPU: The Speedup Curve

```bash
torchrun --nproc_per_node=2 code/multi_gpu_ddp.py
```

| GPUs | Training Time | Speedup |
|---|---|---|
| 1 | 8.78s | 1.00× |
| 2 | 5.91s | 1.48× |
| 4 | 3.69s | 2.38× |
| 6 | 2.92s | 3.01× |
| 8 | 2.44s | **3.60×** |

Same `torchrun` launcher, same model — just more processes, more GPUs.

<!--
Same model, same data, just wrapped with DistributedDataParallel and launched with torchrun. Two GPUs takes us from 8.78 seconds to 5.91, a 1.48x speedup. Keep adding GPUs and we get to 2.44 seconds at 8 GPUs, a 3.6x speedup. Notice the speedup isn't linear — going from 1 to 2 GPUs gets you 1.48x, not 2x, because gradient synchronization overhead eats into the gains. That gap between theoretical and actual speedup is exactly what we're managing throughout this book.
-->

---

## Bigger Workload, Better Scaling

CIFAR-10 (50,000 RGB images), 20 epochs — a more realistic workload:

| GPUs | Training Time | Speedup |
|---|---|---|
| 1 | 73.00s | 1.00× |
| 2 | 46.47s | 1.57× |
| 4 | 27.72s | 2.63× |
| 8 | 18.20s | **4.01×** |

**More computation per step → communication overhead shrinks as a fraction of total time → better scaling.**

<!--
Here's the important lesson: scaling efficiency isn't fixed, it depends on your workload. Same distributed setup, but now on CIFAR-10 with 20 epochs, a bigger, more realistic workload. Eight GPUs now gets us a 4.01x speedup, noticeably better than the 3.6x we saw on the smaller FashionMNIST job. Why? Because gradient synchronization is a fixed cost per step, and when each step does more computation, that fixed communication cost becomes a smaller fraction of total time. For billion-parameter models, this effect is even more pronounced — scaling gets closer to linear as computation dominates.
-->

---

## Distributed Inference: Throughput, Not Time-to-Convergence

ResNet18 inference on FashionMNIST, 1000 requests:

| GPUs | Pattern | Time | Throughput | Speedup |
|---|---|---|---|---|
| 1 | Baseline | 1.85s | 541 req/s | 1.00× |
| 2 | Data-split | 0.98s | 1025 req/s | **1.89×** |
| 2 | Request-split | 1.22s | 819 req/s | 1.51× |

No gradient sync in inference — each GPU processes independently, so scaling is **near-linear**.

<!--
Inference is a different optimization problem entirely: it's about throughput, not time-to-convergence. And because there's no gradient synchronization overhead, each GPU just processes its own requests independently, so scaling is close to linear. Two GPUs with a simple data-split pattern nearly doubles throughput, 1.89x. The request-split pattern, which simulates a real production queue with round-robin assignment, is slightly lower at 1.51x due to coordination overhead, but it's more realistic for how production traffic actually arrives.
-->

---

## The Distributed AI Stack

![h:330](img/distai-stack.png)

**Framework** → **Bucketing** → **Collectives** (AllReduce...) → **NCCL/GLOO** → **Topology** → **Physical Links** (NVLink, InfiniBand) → **Hardware**

<!--
When you call an operation like all_reduce, it flows down this entire stack. At the top, the framework layer, where you write training loops and PyTorch decides when communication needs to happen. Below that, bucketing — PyTorch groups small gradient tensors together so you make fewer, larger network calls instead of many small ones. Below that, the collective operations layer, which defines what needs to happen: AllReduce, AllGather, Broadcast. Then the actual implementation, NCCL for GPUs, GLOO for CPU testing. Then network topology, ring versus fat-tree versus mesh. Then physical links, NVLink inside a node, InfiniBand across nodes. And at the bottom, the physical hardware itself. When a distributed job is slow or hangs, this stack is your debugging checklist, top to bottom.
-->

---

## Process Groups, Ranks, and Collective Operations

- **World size**: total processes across the entire job (2 nodes × 4 GPUs = world size 8)
- **Global rank**: unique across the whole job (0 to world_size − 1)
- **Local rank**: unique within a node — usually maps to `torch.cuda.set_device(local_rank)`

| Operation | Use Case |
|---|---|
| **AllReduce** | Gradient sync (DDP default) |
| **AllGather / ReduceScatter** | Collecting/sharding (FSDP) |
| **Broadcast / Scatter** | One-to-all distribution |
| **AllToAll** | Tensor/expert parallelism |

<!--
A few definitions that everything else builds on. World size is the total number of processes in your job — two nodes with four GPUs each is a world size of eight. Global rank is unique across the entire job. Local rank is unique within a single node, and that's what usually maps directly to which GPU a process uses. On top of ranks, PyTorch gives you collective operations: AllReduce for gradient synchronization, which is what DDP calls automatically under the hood. AllGather and ReduceScatter, which FSDP uses for sharding parameters. Broadcast and Scatter for one-to-all distribution. And AllToAll, the most general pattern, which shows up in tensor parallelism and expert parallelism for MoE models. You rarely call these directly, but knowing what each one does is essential for debugging.
-->

---

## DDP in Three Lines

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = YourModel().cuda(rank)
model = DDP(model, device_ids=[rank])
```

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

```bash
torchrun --nproc_per_node=8 --nnodes=2 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 train.py
```

<!--
And here's the whole thing distilled to code. Wrap your model with DistributedDataParallel, give it your device id, and PyTorch handles gradient synchronization for you during loss.backward() — you don't call AllReduce yourself. Use a DistributedSampler so each process actually sees a different slice of the data, otherwise every GPU trains on the same batch and you've gained nothing. And launch it all with torchrun, which handles process spawning for you, locally or across multiple nodes. This is deliberately the simplest possible starting point — Chapter 3 goes much deeper into DDP internals, and later chapters cover what to do once a model no longer fits on a single GPU even with this pattern.
-->

---

## Key Takeaways

- Model size and compute needs grew **exponentially**; single-GPU capacity grew **linearly** — that gap is why distribution is now the default
- **Calculate before you provision**: weights + gradients + optimizer states + activations (training), weights + KV cache (inference)
- Use the **decision framework** — distribute because the math forces you to, not because it sounds impressive
- Scaling efficiency depends on workload size — **more compute per step means better scaling**
- Distributed inference scales **near-linearly**; distributed training is bounded by **gradient sync overhead**

<!--
If you take five things from this talk: the gap between exponential model growth and linear hardware growth is the entire reason distributed systems exist now. Always calculate before you provision — weights, gradients, optimizer states, and activations for training, weights and KV cache for inference. Use the decision framework, distribute because you have to, not because it's trendy. Scaling efficiency depends on your workload, bigger computation per step scales better because communication overhead shrinks as a fraction of the total. And remember that inference and training scale differently, inference is close to linear because there's no gradient sync, training is bounded by that synchronization cost. Chapter 2 picks up right here, going deep on the GPU hardware and networking that make all of this possible.
-->

---

<!-- _class: lead -->

# Thank You

**Henry Fuheng Wu**

Author of *Distributed AI Systems*, *Above the Clouds*, and *Mathematics for AI and Machine Learning*

Global Data & AI Virtual Tech Conference 2026 · August 22–24

<!--
Thank you all for your time. This was just Chapter 1 — the book goes on to cover GPU hardware and networking, DDP and FSDP in depth, DeepSpeed and Megatron for the largest models, vLLM and SGLang for production inference, and the benchmarking and failure modes you'll actually hit running this in production. Happy to take questions now, or connect afterward.
-->
