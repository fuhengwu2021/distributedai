# 第 6 章：分布式推理核心原理与 vLLM 架构实战 {-}

*以极致高吞吐与超低延迟在大规模集群上部署服务大语言模型*

> 推理是 AI 时代全新的 Web 应用程序。  
> —— 克莱顿·科尔曼（Clayton Coleman，Google 杰出工程师）

**核心代码速查**

- `vllm.LLM`：vLLM 离线模型加载与批量推理核心入口类
- `vllm.SamplingParams`：文本生成采样超参数配置类（Temperature, Top-p, Max Tokens 等）
- `vllm.engine.LLMEngine`：vLLM 底层核心推理执行引擎
- `vllm.worker.worker.Worker`：分布式推理 Worker 进程（单 GPU 计算实体）
- `vllm.distributed.parallel_state`：vLLM 推理端并行状态管理器
- `vllm.engine.async_llm_engine.AsyncLLMEngine`：专用于在线高并发服务的异步推理引擎
- `vllm serve`：一键启动兼容 OpenAI 标准协议的生产级高性能推理服务命令行


## 从分布式训练跨越到分布式推理

在前面的章节中，我们深入探讨了如何跨多 GPU 高效训练千亿大模型：利用 FSDP 和 ZeRO 进行显存状态切分，以及利用 Megatron 进行张量与流水线算力切分。然而，**模型训练仅仅是整个大模型工程生命周期的前半程**。一旦模型权重收敛，我们必须将其上线部署并面向海量用户提供在线推理服务——而服务化推理面临着截然不同的系统约束与技术挑战。

| 系统维度 | 分布式训练（Training） | 分布式推理与服务化（Inference & Serving） |
|:---|:---|:---|
| **核心优化目标** | **纯吞吐量优先**（Tokens/sec 越高越好，平摊在长周期任务中） | **延迟与吞吐兼顾**（首 Token 延迟 TTFT、生成延迟 TPOT 需在 ms 级） |
| **工作负载特征** | 固定 Batch Size、静态序列长度、可预测的计算图 | **高动态到达**、请求长度极度不一、随时开始与结束 |
| **容错与可用性** | 允许存 Checkpoint 后重启恢复 | **要求 $7 \times 24$ 小时高可用**、零服务中断 |
| **显存占用构成** | **优化器状态 (75%)** + 梯度 (15%) + 激活值 + 权重 | **模型静态权重 + KV Cache 动态缓存池（无优化器/梯度）** |
| **计算瓶颈属性** | 恒定的 GEMM 算力密集型（Compute-Bound） | **Prefill 阶段算力密集，Decode 阶段严重显存带宽受限（Memory-Bound）** |

本章将全面解构现代分布式推理的基石：**vLLM**。我们将深入剖析为何 **PagedAttention（分页注意力）** 与 **Continuous Batching（持续批处理）** 彻底重塑了工业级大模型推理，并探讨如何通过张量并行（TP）、数据并行（DP）、流水线并行（PP）以及针对 MoE 的专家并行（EP）在大规模集群上构建高并发推理引擎。

---

## vLLM 架构概述与环境搭建

2023 年，加州大学伯克利分校（UC Berkeley）研究团队针对当时 LLM 推理服务系统存在的严重显存浪费问题，提出了 **vLLM**（Virtual Large Language Model）。其核心洞察是：**传统推理系统将 KV Cache 视为必须在物理显存中连续分配的大内存块，导致了高达 60%–80% 的严重显存碎片与预留浪费**。vLLM 借鉴了操作系统操作系统的**虚拟内存分页机制（Paging）**，首创了 PagedAttention 算法，实现了显存利用率的革命性飞跃。

### vLLM 核心分层架构

![vLLM 调度器-执行器-工作进程（Scheduler-Executor-Worker）分层架构](img/vllm_architecture.png){#fig:vllm-arch .block width=70% align=center}

如 @fig:vllm-arch 所示，vLLM 内部遵循清晰的 **Scheduler-Executor-Worker** 架构模式：

1. **Scheduler（调度器）**：负责接收外部请求流，维护等待队列与活跃生成队列，执行 **持续批处理（Continuous Batching）**，动态管理 Block Table 逻辑块表分配；
2. **Executor（执行器）**：协调层，将调度器的 Batch 指令翻译为多进程分布式命令，支持 Single-GPU、Multi-Processing（单机多卡）以及基于 Ray 的跨机集群执行；
3. **Workers（计算工作节点）**：每个 Worker 独占一张 GPU，持有对应的模型切片（Weight Shard）与物理 KV Cache Block 显存池，通过 NCCL 执行底层算子与 AllReduce 同步。

### 快速安装与环境初始化

#### 1. 基于 Docker 容器一键部署（推荐）

```bash
# 拉取官方最新镜像
docker pull vllm/vllm-openai:latest

# 启动 OpenAI 兼容格式的推理服务器
docker run --runtime nvidia --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 --ipc=host vllm/vllm-openai:latest \
  facebook/opt-125m
```

#### 2. 基于 `uv` / `pip` 本地安装

```bash
# 使用现代化的高性能包管理器 uv（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

#### 3. 离线 Python 原生调用示例

```python
from vllm import LLM, SamplingParams

# 初始化离线推理引擎
llm = LLM(model="facebook/opt-125m")

# 定义采样超参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

# 批量生成文本
prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

---

## 自回归生成与 KV Cache 机理深度剖析

### Decoder-Only Transformer 结构回顾

![Decoder-Only 经典 Transformer 架构示意图](img/decoder_only.png){#fig:decoder-only .block width=30% align=top-right}

如 @fig:decoder-only 所示，现代大模型（LLaMA, GPT, Qwen, DeepSeek 等）均统一采用自回归 Decoder-Only 架构。

在自注意力机制中，每个 Token 位置 $i$ 的输入嵌入 $x_i$ 经过投影矩阵生成 **Query ($Q_i$)**、**Key ($K_i$)** 和 **Value ($V_i$)**：
$$Q_i = x_i W_Q, \quad K_i = x_i W_K, \quad V_i = x_i W_V$$
并通过因果掩码注意力计算关联权重：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{Mask}\right) V$$

### 文本生成的两大阶段：Prefill（预填充） vs Decode（解码）

大模型文本生成在计算特征上分裂为两个完全不同的物理阶段：

![输入张量结构：Batch × Sequence Length × Hidden Dimension](img/input.png){#fig:transformer-input .block width=40% align=top-left}

#### 1. Prefill 阶段（输入理解与上下文编码）

![Prefill 预填充阶段计算流示意图](img/prefill.png){#fig:prefill .block width=100% align=center}

如 @fig:prefill 所示，用户提交的 Prompt 序列（例如长度为 $L_2$ 的 Token 串）在第一个步长中被**全并行输入**模型。所有 $L_2$ 个 Token 在所有网络层中并发完成矩阵投影与自注意力计算，最终输出首个生成 Token 的概率分布 $Y_0$。

在 Prefill 阶段，模型计算复杂度为 $O(L_2^2)$，算子以稠密的大矩阵乘法（GEMM）为主，**GPU Tensor Core 算力被高度占满，处于计算受限（Compute-Bound）状态**。

#### 2. Decode 阶段（逐 Token 自回归生成与显存带宽受限）

在生成首个 Token $Y_0$ 之后，系统进入逐字生成的自回归循环：预测下一个 Token $Y_1$，再将其拼接回输入预测 $Y_2$。

![无 KV Cache 时的朴素解码：每步均需全量重复计算历史 Prompt 的 K/V 投影](img/decode_without_kvcache.png){#fig:decode-no-cache .block width=100% align=center}

如 @fig:decode-no-cache 所示，若不采取任何缓存优化，生成第 $t$ 个 Token 时，模型必须将全量历史上下文 $[X_0, X_1, \dots, X_{t-1}]$ 重新投影并重新计算注意力，单步复杂度高达 $O(L_{\text{total}}^2)$，计算量随长度呈立方级爆炸。

#### 3. KV Cache 机制：以空间换时间

![KV Cache 随 Decode 步长递增动态增长示意图](img/cache_grow.png){#fig:cache-grow .block width=100% align=center}

由于历史 Token 的 $K$ 和 $V$ 向量在后续所有步中是完全恒定不变的，系统只需在 Prefill 阶段将历史各层的 $K$ 和 $V$ 张量缓存在显存中。进入 Decode 阶段时：
- 每次仅输入**单个新 Token**（形状为 $B \times 1 \times D$）；
- 仅计算该单 Token 的 $Q_t, K_t, V_t$；
- 将 $K_t, V_t$ 追加写入 **KV Cache**；
- 当前 $Q_t$ 与全局缓存的 $K_{\text{cached}}, V_{\text{cached}}$ 做矩阵向量点积（GEMV）。

![带 KV Cache 的高效 Decode 解码流程](img/decode_with_kvcache.png){#fig:decode-with-cache .block width=100% align=center}

如 @fig:decode-with-cache 所示，Decode 阶段的时间复杂度从二次方骤降至 $O(L_{\text{total}})$。但此时计算形态退化为**极细碎的矩阵-向量乘法（GEMV）**，算术强度极低，**GPU 绝大部分时间处于从 HBM 读取权重的等待状态，处于典型的显存带宽受限（Memory-Bound）状态**。

---

## PagedAttention：彻底终结显存碎片与 Padding 浪费

### 传统连续显存分配的致命缺陷

在传统推理系统中，系统为每个请求预先分配一块**连续的物理显存**以存放 KV Cache。由于用户最终会生成多少 Token 事先不可知，系统通常按照模型支持的最大长度（如 2048 或 4096）进行预分配。

![传统连续分配导致的严重内部显存碎片与已完成请求的显存闲置](img/kv_cache_fragmentation.png){#fig:kv-cache-fragmentation .block width=85% align=center}

如 @fig:kv-cache-fragmentation 所示，这引发了三重灾难性的显存浪费：
1. **内部碎片（Internal Fragmentation）**：预分配了 2048 个 Token 显存，实际用户仅生成 50 个 Token 即结束，剩余显存被完全锁定闲置；
2. **外部碎片（External Fragmentation）**：不同请求长度不一且随时退出，显存被切割为零碎空隙，无法容纳新的连续大请求；
3. **Padding 带来的无效算力浪费（Padding FLOPs）**：为了批处理不同长度的请求，传统系统必须对齐短请求并填充大量无效的 Padding Token，导致 GPU 浪费高达 30%–50% 的算力计算无效的 Padding 注意力。

### PagedAttention 的分页核心原理

PagedAttention 从操作系统虚拟内存技术中汲取灵感：**将 KV Cache 切分为固定大小的物理块（Physical Blocks，默认每个 Block 容纳 16 个 Token）**。

![PagedAttention 逻辑 Block Table 与离散物理块映射机制](img/paged_attention_blocks.png){#fig:paged-attention-blocks .block width=90% align=center}

如 @fig:paged-attention-blocks 所示：
- **动态按需分配**：请求到达时仅分配 1 个 Block，当生成满 16 个 Token 时再动态从空闲池中申请下一个 Block；
- **物理离散、逻辑连续**：每个请求维护一个 **Block Table（块映射表）**，将逻辑 Token 索引映射到底层离散的物理块地址中；
- **即刻归还**：任何请求一旦触发结束符，其占用的物理块立即归还全局池并瞬间供给新请求复用；
- **显存碎片率降至 < 4%**，使单 GPU 能够容纳 **2–4 倍** 的并发请求量！

### 零 Padding 浪费（Zero Padding FLOPs）

![传统 Padding 批处理 vs PagedAttention 按块精准计算对比](img/padding_vs_paged.png){#fig:padding-vs-paged .block width=100% align=center}

如 @fig:padding-vs-paged 所示，由于 PagedAttention 的自研 CUDA Attention Kernel 是直接基于 Block Table 遍历有效物理块，**根本不需要在显存中构造包含 Padding 的对齐张量**。不存在的 Token 在物理上根本没有分配 Block，Attention 算子直接跳过，**彻底消除了 100% 的 Padding 无效算力消耗**。

---

## 持续批处理（Continuous Batching）

传统静态批处理（Static Batching）在处理多请求时必须等待当前 Batch 中**最长的一个请求完全生成完毕**才能统一返回并启动下一个 Batch，导致短请求用户被长请求强制拖慢（木桶短板效应）。

vLLM 实现了 **持续批处理（Continuous Batching / 迭代级调度）**：
- 调度器在**每一个 Decode Token 步长级别（Iteration-level）**动态进行请求重组；
- 生成结束的请求在当个 Step 结束后立即退出并释放显存；
- 等待队列中的新请求即刻在下一个 Step 插入 Batch 执行 Prefill 并与其余请求合并 Decode；
- 始终保持 GPU 处于满载无闲置状态。

---

## 分布式推理并行策略全景

当模型参数量超过单卡显存上限，或需要支撑极大规模的高并发在线吞吐时，必须采用分布式多 GPU 推理。

![vLLM 支持的核心分布式推理并行策略全景图](img/parallelism_strategies_overview.png){#fig:vllm-parallelism .block width=90% align=center}

### 1. 张量并行（Tensor Parallelism, TP）——首选低延迟利器

- **机制**：在单机多卡内部，将每层的注意力投影矩阵与 MLP 矩阵横向切分到多卡并发执行，每层通过一次 AllReduce 汇总结果；
- **核心收益**：
  - **显存均摊**：模型权重被等分，同时为每张卡腾出宝贵的 KV Cache 物理空间；
  - **显存带宽翻倍**：多张 GPU 的 HBM 显存带宽并发叠加（例如 4 卡 H100 提供高达 $4 \times 3.35\text{ TB/s} = \mathbf{13.4\text{ TB/s}}$ 聚合带宽），**使 Decode 阶段的每 Token 生成延迟（TPOT）呈近线性缩短**。
- **命令行启动**：
  ```bash
  vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4
  ```

### 2. 数据并行（Data Parallelism, DP）——高并发吞吐扩展

- **机制**：在多卡或多节点上独立部署多个完整的模型副本，每个副本处理不同的用户请求，彼此之间零通信；
- **核心收益**：**总吞吐量（Tokens/sec）随副本数呈 100% 纯线性扩展**，具备天然的高可用容灾能力；
- **命令行启动（配合 TP 混合使用）**：
  ```bash
  # 8 卡环境下：部署 2 个模型副本（DP=2），每个副本由 4 卡 TP 组成（TP=4）
  vllm serve meta-llama/Llama-3.1-70B-Instruct --data-parallel-size 2 --tensor-parallel-size 4
  ```

### 3. 流水线并行（Pipeline Parallelism, PP）——跨节点大模型切分

- **机制**：沿网络层深度将模型纵向拆分到不同机架节点，适用于无法通过 NVLink 组建大 TP 域的跨机超大模型（如 405B/671B）。

![流水线气泡（Pipeline Bubble）与虚拟引擎请求组调度优化](img/pipeline_bubble.png){#fig:pipeline-bubble .block width=85% align=center}

- **分块预填充（Chunked Prefill）消除气泡** {#sec:chunked-prefill}：
  长文本 Prefill（耗时数秒）与快速 Decode（耗时数毫秒）交织时会导致流水线严重停顿。vLLM 引入 **Chunked Prefill** 将长 Prompt 拆分为小分块（如 512 tokens/块）分批推进，使计算流平滑交错：
  ```bash
  vllm serve deepseek-ai/DeepSeek-R1 --tensor-parallel-size 4 --pipeline-parallel-size 8 \
      --enable-chunked-prefill --max-num-batched-tokens 2048
  ```

---

## 专家并行（Expert Parallelism, EP）：MoE 稀疏大模型推理

针对 Mixtral 8x7B、DeepSeek-V3/R1 等稀疏激活模型，vLLM 提供了专用修饰参数 `--enable-expert-parallel`。

![MoE 架构：Router 门控网络与稀疏专家计算流](img/moe_arch.png){#fig:moe-arch .block width=85% align=center}

如 @fig:moe-arch 所示，在开启 EP 后：
- 各卡独立持有完整的特定 Expert 网络实体；
- 配合 **DP+EP** 模式可实现 **DP Attention（KV Cache 跨卡完全切分不复制）**，结合 DeepEP 高性能通信库，完美解决超大 MoE 模型的显存与计算并发瓶颈。

```bash
# 8 卡并发部署 MoE 模型
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --data-parallel-size 8 --enable-expert-parallel
```

---

## 生产级 Nsight Systems 性能 Profiling 与诊断

在上线服务前，推荐使用 NVIDIA Nsight Systems 深度捕获 GPU 底层 CUDA Kernel 与 NCCL 通信开销：

```bash
# 使用 nsys 捕获推理服务运行时时序
nsys profile \
    --trace=cuda,nvtx \
    --output=vllm_profile.qdrep \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --tensor-parallel-size 4
```

在 Nsight UI 中核心检查：
1. **AllReduce 耗时占比**：若跨卡 AllReduce 占用总时间 $> 30\%$，说明可能缺乏 NVLink 直连（受限于 PCIe 带宽），应降低 TP 并转向 DP 或 PP；
2. **HBM 显存带宽饱和度**：确认 Decode 期间 GEMV 算子是否达到单卡物理显存带宽的理论极限。

---

## 本章小结

本章系统解构了现代分布式大模型在线推理的核心引擎：
- 剖析了推理在延迟、动态 Batch 与显存构成上与训练的根本差异；
- 揭示了 Prefill（算力受限）与 Decode（显存带宽受限）的物理本质；
- 深入推导了 PagedAttention 消除显存碎片与零 Padding FLOPs 的底层机制；
- 掌握了基于持续批处理（Continuous Batching）的高并发吞吐调度；
- 建立了以 TP（降延迟）、DP（扩吞吐）、PP（跨机扩展）与 EP（MoE 稀疏优化）为支柱的工业级推理架构。

在掌握了 vLLM 核心原理之后，在大模型复杂的 Agent 交互、多轮对话与长文档问答场景中，系统往往面临大量重复前缀（System Prompt / 历史上下文）的重复计算。在下一章中，我们将进入 **SGLang** 的世界，探索 **跨请求前缀缓存（RadixAttention）、计算分离架构与结构化输出优化** 的前沿技术。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
