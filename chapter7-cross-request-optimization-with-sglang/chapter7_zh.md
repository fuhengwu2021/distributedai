# 第 7 章：跨请求优化与 SGLang 架构实战 {-}

*RadixAttention 前缀树缓存、结构化约束生成与请求级智能路由*

> 预测未来的最好方式，就是去亲手创造它。  
> —— 艾伦·凯（Alan Kay，图灵奖得主、计算机科学家）

**核心代码速查**

- `sglang.Engine`：SGLang 高级离线批量推理 Python API 接口
- `sglang.launch_server`：一键启动 SGLang Runtime (SRT) 服务端入口
- `sglang.srt.server.LaunchEngine`：SGLang 底层高性能推理执行引擎
- `sglang.srt.router`：跨节点请求级路由网关（Model Gateway / Router）
- `sglang.srt.sampling_params`：SGLang 文本生成与结构化约束采样配置
- `sglang.function`：SGLang 前端声明式控制流编程装饰器
- `sgl.gen()` / `sgl.fork()`：结构化受控生成算子与对话状态分支克隆算子


## 分布式推理的全新哲学：从单请求内并行到跨请求协同

在上一章中，我们系统学习了 vLLM 基于模型并行（Model Parallelism）的分布式推理架构：通过张量并行（TP）在单机内切分单层矩阵，通过流水线并行（PP）跨节点切分网络深度，并通过 PagedAttention 消除单请求内的显存碎片。

然而，加州大学伯克利分校 LMSYS 团队（知名大模型排行榜 Chatbot Arena 的创立团队）在构建千万级高并发真实对话系统时发现：**在真实的在线交互与 Agent 业务中，海量并发请求之间绝非孤立存在，它们存在着极高比例的重复公共前缀（Shared Prefixes）与会话级状态关联**。例如：
- **智能客服 / 企业知识库**：数万个并发请求均共享同一个长达 1,000–5,000 Tokens 的 System Prompt 或产品知识库文档；
- **Few-shot / CoT 复杂推理**：海量请求共享长达数千字的多样本示例模板；
- **多轮交互对话（Multi-Turn Chat）**：用户的连续提问完全复用上一轮的完整对话上下文；
- **Agent 工作流与分支推演（MCTS / Tree-of-Thought）**：基于同一状态分叉并发推演多条不同的回答路径。

在传统推理架构下，每一个请求都被视作孤立的个体，导致系统对完全相同的 System Prompt 重复执行了成千上万次高昂的 Prefill 矩阵计算。

这正是 **SGLang（Structured Generation Language）** 诞生的核心驱动力[^radix-attention]。SGLang 在完全兼容 TP/PP/EP 传统模型并行的基础上，**将“跨请求协同优化（Cross-Request Optimization）”提升为系统设计的第一公民**。其核心支柱包括：
1. **RadixAttention（基数树前缀缓存）**：将 KV Cache 组织为全局基数树，自动实现任意请求间前缀的零冗余复用与 LRU 智能淘汰；
2. **Zero-Overhead Scheduler（零开销重叠调度器）**：实现 CPU 批调度逻辑与 GPU 算子计算的 100% 异步完全重叠，消除 GPU 气泡等待；
3. **XGrammar 结构化生成引擎**：基于有限状态机（FSM）预编译，实现毫秒级严格 JSON Schema / 正则语法约束解码；
4. **Router-Based 分布式路由网关**：基于 Cache-Aware（缓存感知）与 Session Affinity（会话粘性）分发流量，避免不必要的跨卡同步通信；
5. **PD 分离架构（Prefill/Decode Disaggregation）**：将算力受限的 Prefill 节点与显存带宽受限的 Decode 节点物理解耦，通过 RDMA 极速交换 KV Cache。

[^radix-attention]: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," arXiv:2312.07104, 2023. https://arxiv.org/abs/2312.07104

---

## SGLang 整体架构与运行环境搭建

### 前后端解耦的分层设计

![SGLang 前后端解耦架构：从 API Server 到 GPU Worker 计算流](img/sglang_architecture.png){#fig:sglang-arch .block width=70% align=center}

如 @fig:sglang-arch 所示，SGLang 体系由清晰的两个主要层面构成：
- **前端编程接口（Frontend Language）**：提供类似 Python 嵌入式 DSL 的声明式语法（支持 `@sgl.function`、`fork()`、`join()`、多轮状态分支控制）；
- **后端高性能运行时（SGLang Runtime, SRT）**：由 Tokenizer、Request Queue、Radix 缓存感知调度器、Fused CUDA Kernel 以及 Detokenizer 组成的极速执行流水线。

![vLLM 模型并行 vs SGLang 路由网关架构对比](img/vllm_vs_sglang.png){#fig:vllm-vs-sglang .block width=85% align=center}

如 @fig:vllm-vs-sglang 所示，在多 GPU 集群部署形态上：
- **vLLM 模型并行模式（左）**：每张 GPU 持有模型切片，各层之间必须频繁执行 AllReduce 同步（适合超大参数量单模型）；
- **SGLang 路由网关模式（右）**：各 Worker 独立部署完整副本（或小 TP 组），前端通过智能网关分发请求，各 Worker 之间**完全零通信**，吞吐与容灾能力极强。

### 快速启动与环境配置

#### 1. 基于 Docker 容器一键部署

```bash
# 拉取生产级轻量运行时镜像
docker pull lmsysorg/sglang:latest-runtime

# 启动兼容 OpenAI 标准协议的在线服务
export SGLANG_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
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

#### 2. 在线 API 验证

```bash
# 测试标准对话接口
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己。"}
    ],
    "temperature": 0
  }'
```

---

## RadixAttention：基数树跨请求前缀缓存原理

### 共享前缀复用的系统价值

![RadixAttention 跨请求公共前缀（System Prompt）零冗余共享机制](img/radix_tree.png){#fig:radix-tree .block width=85% align=center}

如 @fig:radix-tree 所示，当请求 1、2、3 到达时，它们均以相同的系统提示词（如 `"You are helpful. "`）开头：
- 传统系统：对该前缀独立执行 3 次完整的 Prefill 矩阵乘法；
- RadixAttention：仅在首个请求到达时计算一次该前缀的 KV Cache（绿色根节点），后续所有请求直接**零开销复用**已有 KV，仅对各自独有的后缀（黄色子节点）执行增量 Prefill 计算！

### Radix 树的演进与 LRU 自动剪枝生命周期

![RadixAttention 树的构建、分裂与分支演化过程](img/radix_attn1.jpg){#fig:radix-attention1 .block width=95% align=center}

![RadixAttention 树在显存压力下的 LRU 淘汰机制](img/radix_attn2.jpg){#fig:radix-attention2 .block width=95% align=center}

![RadixAttention 树高负载下的自适应动态剪枝](img/radix_attn3.jpg){#fig:radix-attention3 .block width=95% align=center}

如 @fig:radix-attention1、@fig:radix-attention2 与 @fig:radix-attention3 所示，Radix 树在运行时展现出精妙的自适应生命周期：
1. **自动节点分裂（Node Splitting）**：当后续提问延伸已有对话时，原始长节点自动在分叉点切分为“公共父节点”与“新子分支”；
2. **显存感知与 LRU 淘汰（Eviction）**：当 GPU 显存池逼近饱和时，调度器自动扫描各分支叶子节点，优先淘汰最久未被访问的陈旧会话（如标记橙色虚线框的失效分支），而常驻高频访问的 System Prompt 根节点则得以持久保留。

---

## 零开销重叠调度器（Zero-Overhead Scheduler）

传统推理框架在单步执行中通常采用串行同步模型：CPU 收集请求组装 Batch $\rightarrow$ 提交 GPU 发射 Kernel 计算 $\rightarrow$ CPU 阻塞等待 GPU 返回结果 $\rightarrow$ 处理 Token 后再调度下一步。GPU 有近 30%–50% 的时间处于空转等待状态。

![串行调度 vs SGLang 零开销重叠调度执行时序对比](img/scheduler_comparison.png){#fig:scheduler-comparison .block width=100% align=center}

如 @fig:scheduler-comparison 所示，SGLang 提出了**双阶段流水线重叠机制**：
- 在 GPU 正在满载执行当前 Batch $N$ 的矩阵计算时，CPU 异步并发准备 Batch $N+1$ 的前缀树匹配与显存分配，并异步解析上一步 Batch $N-1$ 的输出结果；
- 通过引入**占位符机制（Token Placeholder）**，CPU 与 GPU 实现极致重叠，**将调度器额外开销彻底压缩至 0 ms**，使整体系统吞吐提升高达 **2 倍**！

---

## XGrammar：极速结构化约束生成

在 Agent 函数调用（Function Calling）、信息抽取及代码生成中，要求模型严格输出符合 JSON Schema 规范的文本至关重要。传统通过在每步逐 Token 进行正则校验的方案极其缓慢。

SGLang 集成了 **XGrammar** 约束解码引擎[^xgrammar]：
- **离线语法编译**：在生成前将 JSON Schema 或上下文无关文法（CFG）预编译为**有限状态自动机（FSM / Pushdown Automata）**；
- **自适应 Bitmask 过滤**：在每个生成步长中，FSM 在微秒级别输出当前合法的 Token 掩码，直接在 GPU Logits 上施加掩码过滤，**实现 100% 格式合规且几乎零吞吐损失**。

[^xgrammar]: Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models," arXiv:2411.15100, 2024. https://arxiv.org/abs/2411.15100

```bash
# 客户端提交带严格 JSON Schema 约束的推理请求
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "提取信息：张伟，32岁，现居上海。"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "user_info",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "city": {"type": "string"}
          },
          "required": ["name", "age", "city"]
        }
      }
    }
  }'
```

---

## PD 计算分离架构（Prefill/Decode Disaggregation）

### 动因：Prefill 与 Decode 的物理资源错配

![Prefill 算力密集 vs Decode 显存带宽密集资源画像对比](img/pd_disaggregation.png){#fig:pd-disaggregation .block width=95% align=center}

如 @fig:pd-disaggregation 所示：
- **Prefill 阶段**：算力利用率高达 85%，显存带宽利用率仅 40%（计算密集型，对 Tensor Core 算力饥渴）；
- **Decode 阶段**：算力利用率仅 25%，显存带宽利用率高达 90%（显存受限型，对 HBM 传输带宽饥渴）。

将两类截然不同的计算混跑在同一张 GPU 上，必然导致互相干扰、互相拖慢（尤其是长 Prefill 会导致正在 Decode 的用户遭遇严重卡顿）。

### 基于 RDMA 的 PD 分离集群架构

![SGLang PD 分离架构：Router 控制面调度与 Mooncake RDMA 数据直传](img/pd_architecture.png){#fig:pd-architecture .block width=100% align=center}

如 @fig:pd-architecture 所示，SGLang 实现了完全物理隔离的 PD 分离：
1. **控制面（Control Plane）**：客户端请求到达统一智能路由网关（Router），网关将请求派发给 **Prefill 算力 Worker**，同时指定承接生成的 **Decode 显存 Worker**；
2. **数据面（Data Plane）**：Prefill Worker 在极速计算完 Context 编码后，通过 **Mooncake 引擎借助 RDMA (InfiniBand/RoCE)** 将生成的 KV Cache 物理块**直接以内存零拷贝方式跨机推送到 Decode Worker 的显存中**；
3. **Decode 启动**：Decode Worker 接收到就绪的 KV Cache 后，专注于自回归吐字，并通过网关流式返回给用户。

---

## SGLang Model Gateway（智能路由网关）

![基于 Cache-Aware 智能路由的多节点高并发集群架构](img/router_multi_node.png){#fig:router-multi-node .block width=70% align=center}

如 @fig:router-multi-node 所示，在多节点分布式集群部署中，SGLang 网关支持多种先进的流量路由策略：

1. **缓存感知策略（`--policy cache_aware`）**：网关在内存中维护全集群各 Worker 的前缀基数树拓扑镜像。当收到新请求时，自动将其精准路由到**拥有最长匹配已缓存前缀的 Worker 节点**，最大化全集群的 Radix 命中率；
2. **会话粘性（Session Affinity）**：针对多轮对话，通过 `session_id` 自动锁定同一个 Worker，彻底免除上下文重复传输与重新 Prefill 计算；
3. **熔断与动态降级（Circuit Breaking）**：秒级探测故障节点并自动平滑剔除，请求自动无缝迁移至健康节点。

---

## MoE 专家并行与 DP Attention 高阶优化

### 稀疏专家并行（EP）与通信计算重叠

在部署 Mixtral 8x7B、DeepSeek-V3 等大模型时，SGLang 提供了领先的通信掩盖技术：
- **DeepEP 与 DeepGEMM**：专为 MoE 定制的极低延迟跨节点 All-to-All 通信库与融合矩阵算子；
- **两批重叠（Two-Batch Overlap, TBO）**：将 Batch 拆分为两组交错流动：当 Batch 0 正在执行 All-to-All 路由通信时，GPU 算力并发执行 Batch 1 的自注意力计算，**彻底隐藏 All-to-All 网络通信开销**。

```bash
# 启动 8 卡 TP+EP 混合并行并开启 TBO 重叠
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --tp 8 --ep 8 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm \
    --enable-two-batch-overlap
```

### 数据并行注意力（DP Attention）

针对 DeepSeek-V3/R1 等采用 MLA（Multi-Head Latent Attention）架构的模型，由于其 KV 维度极小，传统 TP 会强行在各卡复制 KV Cache 造成严重浪费。SGLang 引入 **DP Attention**：在 Attention 阶段按请求数据并行切分 KV（各卡独立持有不同请求），而在 MLP 阶段转为张量并行，**使显存利用率提升 1.9 倍**！

---

## vLLM 与 SGLang 架构选型与工业落地指南

| 对比维度 | vLLM | SGLang |
|:---|:---|:---|
| **核心优化重心** | **请求内模型并行优化**（PagedAttention、TP/PP 多机大模型切分） | **跨请求跨会话协同优化**（RadixAttention、PD 分离、智能网关） |
| **前缀复用机制** | 块级 Hash 匹配（Block Hash Prefix Caching） | **全局显式 Radix Tree 基数树索引 + LRU 自动剪枝** |
| **调度器性能** | 迭代级持续批处理（Continuous Batching） | **零开销异步重叠调度器（Zero-Overhead Scheduler）** |
| **结构化生成** | 集成 Outlines（正则约束） | **原生深度集成 XGrammar 语法状态机（毫秒级极速）** |
| **多节点部署形态** | 传统的跨机 TP/PP 强同步拓扑 | **Router 智能网关 + Cache-Aware 分发 + PD 物理分离** |
| **最适业务场景** | **单次超长文档离线批处理、70B/405B 超大模型全切分** | **高并发 Agent 交互、多轮对话平台、Few-shot/RAG 问答服务** |

---

## 本章小结

本章系统解构了现代大语言模型跨请求优化的先锋架构 **SGLang**：
- 剖析了从“请求内并行”到“跨请求协同优化”的设计哲学转变；
- 深入推导了 RadixAttention 前缀树缓存结构、节点分裂与 LRU 自动剪枝机制；
- 阐明了零开销重叠调度器（Zero-Overhead Scheduler）消除 CPU/GPU 同步空转的原理；
- 掌握了基于 XGrammar 状态机的毫秒级结构化约束生成技术；
- 掌握了基于 Mooncake RDMA 的 Prefill/Decode 计算分离（PD Disaggregation）工业级架构；
- 掌握了 SGLang Router 智能网关的 Cache-Aware 路由、会话粘性与 MoE 优化体系。

在掌握了分布式训练（DDP/FSDP/Megatron）与分布式在线推理（vLLM/SGLang）两大核心引擎后，我们如何在真实的超算集群（HPC）或云原生数据中心中统一调度、分配上千张 GPU 资源并高效运行这些分布式作业？在下一章中，我们将进入 **Slurm 集群作业调度与分布式运维实战**，打通从底层基础设施到上层 AI 任务的关键闭环。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
