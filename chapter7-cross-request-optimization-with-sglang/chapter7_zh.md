# 第7章：使用 SGLang 的跨请求优化 {-}

*用于低延迟推理的 RadixAttention、结构化生成和请求级路由*

> 预测未来的最好方式是发明它。
- Alan Kay，计算机科学家

**Code Summary**

- `sglang.Engine`：高级离线推理 API
- `sglang.srt.server.LaunchEngine`：启动 SGLang 推理引擎
- `sglang.srt.hf_transformers_utils`：HuggingFace transformers 集成
- `sglang.srt.engine`：用于请求处理的 SGLang 引擎
- `sglang.srt.server`：用于分布式服务的 SGLang 服务器
- `sglang.srt.model_config`：SGLang 的模型配置
- `sglang.srt.tokenizer`：SGLang 的分词器工具
- `sglang.srt.router`：分布式 SGLang 的请求路由器
- `sglang.srt.sampling_params`：SGLang 生成的采样参数
- `sglang.srt.server_runner`：SGLang 部署的服务器运行器

## 分布式推理的不同哲学

在上一章，我们探索了 vLLM 的分布式推理方法：模型并行。当一个模型对单块 GPU 太大时，vLLM 用张量并行（TP）或流水线并行（PP）将模型权重跨多块 GPU 拆分。工作进程必须同步——TP 的 all-reduce 操作，PP 的流水线阶段——系统通过批处理尽可能多的请求来为吞吐量优化。

SGLang 采取不同的方法。它像 vLLM 一样支持 TP、PP 和专家并行（EP），但将 *跨请求优化* 提升为一等关切。虽然 vLLM 主要专注于请求内执行效率（单个请求被处理的效率），SGLang 额外优化请求间执行效率（请求如何交互和共享资源）。

实现这一点的核心创新是 **RadixAttention**[^radix-attention] 和 **零开销调度器**。RadixAttention 将 KV 缓存组织为一个基数树，允许有共同前缀的请求共享缓存的计算。调度器将 CPU 工作与 GPU 计算重叠，消除空闲时间。这些优化在内核和调度器级别工作，构成 SGLang 性能的基础。

[^radix-attention]: SGLang: Efficient Execution of Structured Language Model Programs. \url{https://arxiv.org/abs/2312.07104}

在这个执行引擎之上，SGLang 引入 **请求级路由** 作为一个扩展原语。SGLang 不只通过模型并行（拆分权重）扩展，还可以通过请求路由（将请求分发到独立的工作进程）扩展。一个路由器根据缓存局部性、会话亲和性和负载均衡引导流量。这对模型装进单块 GPU 或小 TP 组的工作负载特别有效。

这种组合对特定工作负载很强大。对于有共享系统提示的多轮对话，RadixAttention 的前缀缓存结合会话亲和性可以将后续请求的延迟减少 2-3 倍。但这个好处是有条件的——它需要前缀重用、对话式工作负载和非批处理主导的场景。对于批处理吞吐量工作负载或需要广泛模型并行的非常大的模型，vLLM 的方法可能更合适。

**SGLang**（结构化生成语言）从 UC Berkeley 的 LMSYS 团队出现——就是 Chatbot Arena 排行榜背后的同一个团队。虽然 vLLM 通过 PagedAttention 专注于内存效率，SGLang 的创造者问了一个不同的问题：我们如何跨请求优化，使它们能共享计算并彼此受益？

答案引导出用于跨请求 KV 缓存共享的 RadixAttention、用于约束结构化输出的 XGrammar，以及最大化 GPU 利用率的零开销调度器。请求级路由后来作为补充这些核心创新的生产扩展层出现。

### 先决条件

SGLang 运行在带 Python 3.10 或更高版本的 Linux 上。对于 GPU 加速推理，你需要一块支持 CUDA 的 NVIDIA GPU。虽然 SGLang 也支持其他加速器，但 NVIDIA 仍然是最常见的部署目标。

### 安装

SGLang 可以用几种方法安装。Docker 是尝试 SGLang 最快的方式，无需在本地安装依赖。

#### Docker 设置

预构建的 Docker 镜像在 [SGLang Docker Hub 页面](https://hub.docker.com/r/lmsysorg/sglang) 上可用。这些镜像捆绑所有依赖，使入门容易。

SGLang 支持多样的模型类型，每种为不同的用例优化。像 `meta-llama/Llama-3.2-1B` 这样的基础模型是使用 `/v1/completions` 端点进行文本补全的预训练语言模型。像 `Qwen/Qwen2.5-0.5B-Instruct` 这样的聊天模型为对话微调，使用 `/v1/chat/completions`。像 `Qwen/Qwen3-Embedding-0.6B` 这样的嵌入模型通过 `/v1/embeddings` 生成向量表示。对于更专门的任务，SGLang 支持用于非自回归生成的扩散语言模型、接受图像和视频与文本一起的多模态模型、用于搜索结果排序的重排模型，以及用于强化学习应用的奖励模型。SGLang 还为图像和视频生成任务加速扩散模型。

SGLang 服务器暴露一个 OpenAI 兼容的 API，使它成为已经使用 OpenAI API 格式的应用的即插即用替代品。关于带详细描述和使用示例的端点完整列表，见附录中的 **OpenAI 兼容 API 端点** 部分。

__拉取最新镜像__

拉取最新的 Docker 镜像。运行时镜像需要约 16GB 磁盘空间。

```bash
docker pull lmsysorg/sglang:latest-runtime
```

`latest-runtime` 标签提供一个带最小依赖的生产就绪镜像。还有一个包括开发工具和构建依赖的 `latest` 标签，但约 35GB 大得多。对大多数用例，运行时镜像足够。

__运行 Docker 容器__

Docker 镜像运行一个 OpenAI 兼容的服务器。为避免硬编码模型名，将它设为环境变量：

```bash
export SGLANG_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

然后运行容器：

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

SGLang 支持 Llama、Mistral、Qwen、Gemma、Phi 和其他 HuggingFace 架构；像 `facebook/opt-125m` 这样的传统因果 LM 也能工作，尽管聊天端点期待指令微调的检查点。一些模型需要 `--trust-remote-code`。见 [SGLang 文档](https://docs.sglang.io) 获取完整列表。

以下是一些适合学习目的的小模型：

| 模型名 | 类型 | 参数大小 |
|------------|------|----------------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 聊天/指令 | 0.5B |
| `meta-llama/Llama-3.2-1B-Instruct` | 聊天/指令 | 1B |
| `meta-llama/Llama-3.2-1B` | 基础 | 1B |
| `microsoft/Phi-tiny-MoE-instruct` | MoE/指令 | 总共 3.8B，约 1.1B 活跃 |
| `Qwen/Qwen3-Embedding-0.6B` | 嵌入 | 0.6B |
| `Qwen/Qwen2-VL-2B-Instruct` | VLM/多模态 | 2B |
| `BAAI/bge-reranker-v2-m3` | 重排 | 0.6B |
| `jason9693/Qwen2.5-1.5B-apeach` | 奖励/分类 | 1.5B |

要使用这些模型中的任何一个，替换 Docker 命令中的模型名。例如，要服务 `meta-llama/Llama-3.2-1B-Instruct`，将 `--model-path $SGLANG_MODEL` 改为 `--model-path meta-llama/Llama-3.2-1B-Instruct`。

Docker 标志值得解释。`--runtime nvidia --gpus all` 标志启用 GPU 访问；将 `--gpus all` 替换为单块 GPU 的 `--gpus '"device=0"'` 或特定 GPU 的 `--gpus '"device=0,1"'`。卷挂载 `-v $HOME/.cache/huggingface:/root/.cache/huggingface` 将你的本地 Hugging Face 缓存与容器共享，避免重复的模型下载。`--ipc=host` 标志允许容器访问主机的共享内存，PyTorch 在张量并行推理期间用它进行高效的数据共享。最后，`--shm-size 32g` 设置共享内存大小，这对 SGLang 的 KV 缓存管理和 RadixAttention 特性很重要。

__验证设置__

一旦容器运行，验证它正确工作。首先，检查服务器响应：

```bash
curl -w "HTTP Status: %{http_code}\n" http://localhost:30000/health
```

列出可用的模型：

```bash
curl http://localhost:30000/v1/models
```

测试基础模型补全：

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$SGLANG_MODEL"'",
    "prompt": "The result of 1+1 is",
    "max_tokens": 3
  }'
```

测试聊天模型：

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

对于确定性输出（每次相同结果），添加 `"temperature": 0`：

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

关于替代安装方法（pip、uv 或从源码），参考 [SGLang 安装指南](https://docs.sglang.io/get_started/install.html)。文档还涵盖模型特定需求和高级配置选项。

## SGLang 架构概述

SGLang 的架构遵循前端-后端设计模式。前端（API 服务器）处理客户端请求，而后端（SGLang 运行时，或 SRT）执行推理。这种分离允许系统独立优化每一层。

![SGLang 架构](img/sglang_architecture_zh.png){#fig:sglang-arch .block width=70% align=center}


图~\ref{fig:sglang-arch} 展示了请求通过 SGLang 的流程。客户端（使用原生解释器的 SGLang 程序或标准 HTTP 客户端）向 API 服务器发送请求，它作为入口点。SGLang 运行时（SRT）然后通过一个组件流水线处理这些请求。**分词器（Tokenizer）** 将传入文本转换为模型可以处理的数字 token。**请求队列（Request Queue）** 缓冲这些分词的请求，管理并发并为批处理准备它们。**调度器（Scheduler）** 是 SGLang 智能所在——它智能地批处理请求，通过 RadixAttention 优先处理可以从 KV 缓存重用中受益的任务，并实现将 CPU 工作与 GPU 计算重叠的零开销调度。**GPU 工作进程（GPU Workers）** 执行实际的模型推理，可以为张量并行（工作进程协作处理单个请求）或作为数据并行的独立工作进程组织。最后，**逆分词器（Detokenizer）** 将生成的 token 转换回人类可读的文本，响应通过 API 服务器流回客户端。

对于分布式部署，SGLang 在 API 服务器之上添加一个 **路由器（Router）**（也称为模型网关）层。路由器跨多个 SRT 实例分发请求，维持会话亲和性，使来自同一对话的请求去同一个工作进程（保持 KV 缓存局部性）。它包括一个用于工作进程管理、负载监控和健康检查的控制平面，加上一个实现各种负载均衡策略的数据平面。

与 vLLM 的架构差异在于优化焦点。vLLM 使用为请求内效率优化的调度器-执行器-工作进程模式——用于内存管理的 PagedAttention，用于大型模型的模型并行。SGLang 的架构额外通过 RadixAttention 和缓存感知调度优化请求间效率。两个系统都支持 TP/PP/EP 进行模型并行；SGLang 为模型装进单个工作进程的工作负载添加请求级路由作为补充的扩展原语。

![vLLM vs SGLang 架构比较](img/vllm_vs_sglang_zh.png){#fig:vllm-vs-sglang .block width=85% align=center}

图~\ref{fig:vllm-vs-sglang} 对比了两种架构方法。在左边，vLLM 的模型并行需要工作进程在每层通过 all-reduce 同步——对拆分大型模型是必要的，但增加通信开销。在右边，SGLang 的基于路由器的架构将请求分发到独立的工作进程，每个持有一个完整模型。工作进程在没有工作进程间同步的情况下处理请求，路由器处理负载均衡和缓存感知路由。这种独立性实现无通信开销的水平扩展，但需要每个工作进程持有完整模型（或一个小 TP 组）。


## SGLang 核心理论

SGLang 的性能优势主要来自它的执行引擎创新。核心技术是用于跨请求 KV 缓存重用的 **RadixAttention**、用于消除 CPU/GPU 空闲时间的 **零开销调度器**、用于结构化输出解码的 **XGrammar**，以及用于减少内核启动开销的 **算子融合**。这些内核级和调度器级优化构成基础，无论部署拓扑如何都有效工作。

虽然 vLLM 专注于单个请求内的内存效率（PagedAttention），SGLang 强调跨多个请求的优化。这使 SGLang 对许多请求共享共同模式的工作负载特别有效——系统提示、少样本示例或多轮对话。

### RadixAttention：前缀缓存重用

RadixAttention 也许是 SGLang 最独特的创新。虽然 vLLM 的 PagedAttention 优化单个请求 KV 缓存内的内存管理，RadixAttention 通过为共同前缀共享 KV 缓存跨多个请求优化。vLLM 的前缀缓存（在近期版本中默认开启）通过哈希共享的 token 块实现类似的重用；RadixAttention 改为保持一个显式的基数树并将前缀查找直接绑入批调度。当许多请求共享长的、相同的前缀时收益显现——系统提示、少样本模板或多轮聊天的早期轮次。

考虑一个典型的 AI 助手部署。每个请求以相同的系统提示开始："You are a helpful assistant. You provide accurate, helpful responses..."。这个系统提示可能是 500 个 token。在传统系统中，如果 100 个用户同时发送请求，系统为那个 500-token 前缀计算 KV 缓存 100 次——计算和内存的巨大浪费。

RadixAttention 通过将 KV 缓存组织为一个基数树（也称为前缀树）解决这个问题。在这个数据结构中，共同前缀被存储一次并跨所有使用它们的请求共享。当一个新请求到达时，系统在树中找到最长匹配前缀，重用那个前缀的现有 KV 缓存，只为新 token 计算 KV 缓存。随着请求完成，共享前缀保留在树中供未来重用，而唯一后缀被驱逐。

![RadixAttention 前缀共享](img/radix_tree_zh.png){#fig:radix-tree .block width=85% align=center}


图~\ref{fig:radix-tree} 展示了 RadixAttention 如何跨请求共享 KV 缓存。假设三个请求到达，带一个共同系统提示："You are helpful. What is Python?"、"You are helpful. Explain ML."，和 "You are helpful. Write code."。基数树将共享前缀 "You are helpful. " 存储一次（绿色节点），而每个请求的唯一后缀单独存储（黄色节点）。

"You are helpful. " 的 KV 缓存被计算一次并由所有三个请求共享。每个唯一后缀单独计算。随着更多请求共享同一前缀，节省累积。

性能好处是显著的——但取决于工作负载特征。对于有长共享前缀的工作负载（系统提示、少样本示例、多轮对话），RadixAttention 可以将预填充计算减少多达 90%。对于没有前缀共享的工作负载（唯一提示、单轮交互），好处极小。内存效率提高，因为共享前缀被存储一次而非每请求。对命中缓存的请求，延迟下降 2-3 倍。吞吐量增加，因为减少的每请求内存实现更大的批大小。

SGLang 的调度器知道基数缓存并用它来优化批形成。选择下一个要运行的批次时，调度器按它们的最长匹配前缀长度对请求排序，并优先处理有更长共享前缀的请求。这最大化缓存命中率和 GPU 利用率。

缓存还与会话亲和性集成。当来自同一会话的请求被路由到同一个工作进程时，那个工作进程上的基数树累积对话历史。对话中的后续消息受益于之前轮次缓存的 KV，大幅减少多轮交互的延迟。但随着对话增长和内存填满会发生什么？

![RadixAttention 树演化和 LRU 驱逐-1。来源：Zheng et al., 2023。](img/radix_attn1.jpg){#fig:radix-attention1 .block width=95% align=center}

图 \ref{fig:radix-attention1}、\ref{fig:radix-attention2}、\ref{fig:radix-attention3} 追踪一个基数树从诞生到成熟的生命周期。在面板 (1) 中，树是空的——还没有请求到达。面板 (2) 显示第一个聊天会话："You are a helpful assistant. User: Hello! Assistant: Hi!" 成为单个节点。当面板 (3) 中一个后续消息扩展这个对话时，发生一些有趣的事：树重构自己。原始内容拆分成一个共享前缀节点和一个用于续写 "User: Solve this problem..." 的新分支。

随着更多用户到达，树揭示它真正的力量。面板 (4) 显示多个聊天会话——每个以 "You are a helpful assistant." 开始——从单个共享前缀节点分支。一个用户问 "Hello!"，另一个问 "What can you do?"，第三个提出不同的问题。系统提示被存储一次并由所有共享。

![RadixAttention 树演化和 LRU 驱逐-2。来源：Zheng et al., 2023。](img/radix_attn2.jpg){#fig:radix-attention2 .block width=95% align=center}

但内存是有限的。面板 (5)、(8) 和 (9) 显示压力下发生什么。当一个新请求需要空间时，SGLang 的 LRU（最近最少使用）策略介入。在面板 (5) 中，节点 (c)——一个较旧、不活跃的对话——被驱逐（用橙色虚线框和 "X" 标记）以为 "Write a story..." 腾出空间。

![RadixAttention 树演化和 LRU 驱逐-3。来源：Zheng et al., 2023。](img/radix_attn3.jpg){#fig:radix-attention3 .block width=95% align=center}

随着面板 (8)-(9) 中压力增大，整个对话分支消失，但频繁访问的系统提示和活跃会话幸存。树自我修剪，保留重要的并丢弃不重要的。

在底层，SGLang 通过一个两级内存池实现 RadixAttention（截至 SGLang v0.5）。第一级将每个请求映射到它的 token 的 KV 缓存索引。第二级存储实际的 KV 缓存数据，组织为 `[num_layers, max_tokens, num_heads, head_dim]`。基数树位于这些池之上，跟踪哪些前缀被缓存并实现高效的查找和共享。

### 用 XGrammar 进行结构化输出解码

许多应用需要 LLM 以特定格式生成输出——API 响应的 JSON、数据库查询的 SQL，或领域特定任务的自定义模式。约束解码的天真方法根据语法规则检查每个生成的 token 并掩码无效 token。但用 128K token 的词汇表（如 Llama-3），在每一步检查每个 token 变得计算上难以承受。

SGLang 默认通过 **XGrammar** 路由结构化输出，Outlines 和 llguidance 作为替代语法后端。XGrammar 为上下文无关规则将有效 token 集预编译成有限状态机——例如，生成布尔值时，只有 "true" 和 "false" token 保持有效。这种预编译覆盖典型 JSON 或模式约束中的大多数 token。对于上下文敏感规则（如平衡括号），它使用带基于树的栈管理的下推自动机以避免昂贵的栈快照。

结果是保证有效性和最小开销的结构化输出。以下是在运行的服务器上的 OpenAI 兼容调用：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Extract: John is 30 years old, lives in NYC"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "city": {"type": "string"}
          },
          "required": ["name", "age"]
        }
      }
    }
  }'
```

对于离线批处理作业，通过 Python `Engine` 上的 `sampling_params` 传入相同的模式：

```python
import json
from sglang import Engine

llm = Engine(model_path="Qwen/Qwen2.5-0.5B-Instruct")
schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
    "required": ["name", "age"],
}
outputs = llm.generate(
    ["Extract: John is 30 years old, lives in NYC"],
    {"max_new_tokens": 100, "json_schema": json.dumps(schema)},
)
print(outputs[0]["text"])
```

XGrammar 广泛支持上下文无关语法——JSON、SQL、领域特定语言，以及你的应用需要的其他结构化格式。

### 算子融合

现代 GPU 计算快，但每次内核启动携带开销。层归一化、线性投影和激活作为单独的内核意味着多次启动和额外的内存流量。SGLang 为常见 transformer 模式提供融合的 CUDA 内核——层归一化加线性加激活、注意力输出投影，以及与专家 GEMM 融合的 MoE 路由——使中间结果保持在寄存器中，而非往返于全局内存。对于 MoE 模型，专门的内核将专家路由与计算结合以减少 all-to-all 开销。

### 零开销调度器

传统推理系统串行执行调度和计算：CPU 调度下一个批次，然后 GPU 计算它，然后 CPU 处理结果并再次调度。GPU 在 CPU 工作时闲置，调度器开销可以消耗总时间的 50% 或更多。

SGLang 的零开销调度器通过将 CPU 调度与 GPU 计算重叠来消除这种空闲时间。关键洞见是当 GPU 处理批次 N 时，CPU 可以准备批次 N+1 并处理批次 N-1 的结果。GPU 从不等待 CPU。

![串行 vs 零开销调度器](img/scheduler_comparison_zh.png){#fig:scheduler-comparison .block width=100% align=center}

图~\ref{fig:scheduler-comparison} 展示了区别。在串行调度器（上）中，CPU 和 GPU 交替：GPU 在调度期间闲置，CPU 在计算期间闲置。在零开销调度器（下）中，CPU 工作（预调度、启动、后处理）与 GPU 计算重叠。当 GPU 处理一个批次时，CPU 准备下一个批次并处理前一个批次的结果。GPU 从不等待。


调度器将 CPU 工作拆分成两个逻辑部分。"调度器 CPU" 处理预调度（收集请求、在基数树中匹配前缀、分配内存）和后调度（检查完成条件、移除完成的请求、更新缓存）。"启动 CPU" 处理内核启动和结果处理。这些可以重叠，因为它们在不同的批次上操作。

为使这种重叠工作，SGLang 使用一个 token 占位符机制。当调度器将一个批次分派到 GPU 时，它不等待结果。相反，它分配占位符 token 并继续调度下一个批次。一个后台线程监控 GPU 完成，并在结果就绪时用实际 token 替换占位符。

性能好处是显著的：相比串行调度多达 2 倍的吞吐量改进，以及更低的端到端延迟，因为 GPU 总是忙碌。

## 基于路由器的分布式架构

除了它的核心执行引擎，SGLang 引入请求级路由作为补充传统模型并行的扩展原语。这不是 TP/PP 的替代——SGLang 像 vLLM 一样支持那些。相反，路由为模型装进单个工作进程的工作负载提供额外的扩展维度。

关键区别：使用 TP/PP 时消除 *层内* 同步是不可能的——你仍然需要 TP 的 all-reduce、PP 的流水线交接。基于路由器的扩展消除的是 *请求间* 同步。每个工作进程独立处理请求，工作进程之间没有协调开销。路由器根据负载、缓存局部性和会话亲和性引导流量。

准确地说：基于路由器的架构消除跨请求同步（处理不同请求的工作进程之间无协调），但不消除层内集合操作（TP 仍需要 all-reduce，PP 仍需要流水线交接，EP 仍需要 all-to-all）。如果你在每个工作进程内用 TP=8，那 8 块 GPU 仍在每层同步——路由器不改变那个。路由器消除的是工作进程彼此协调的需要。

这在模型装进单块 GPU 或小 TP 组（2-8 块 GPU）时最有价值。对于需要跨许多 GPU 广泛 TP/PP 的非常大的模型，路由器增加的价值很小——你无论如何都受模型并行限制。但对于服务高 QPS 的较小模型，路由实现无传统数据并行通信开销的水平扩展。

### SGLang 模型网关架构

SGLang 模型网关（以前称为 SGLang Router）是实现请求级路由的组件。它位于多个 SRT 实例前面，根据复杂策略引导流量，同时提供企业级可靠性特性。

网关架构有两个主要层：一个控制平面和一个数据平面。控制平面管理工作进程生命周期——发现工作进程、跟踪它们的负载、监控健康，以及处理注册/移除。数据平面处理跨多个协议（HTTP、gRPC）的实际请求路由，带有像重试、断路器和限流这样的内建可靠性特性。

控制平面包括几个协同工作的组件。工作进程管理器发现工作进程能力并跟踪实时负载统计。健康检查器持续探测工作进程以验证可用性，在工作进程失败时更新断路器状态。负载监控器向路由策略馈送关于待处理请求、活跃会话和资源利用率的实时统计。对于 Kubernetes 部署，服务发现自动使工作进程注册表与 pod 生命周期对齐。

数据平面实现多种路由器类型。HTTP 路由器处理标准的 OpenAI 兼容端点（`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 等），支持流式和非流式模式。预填充/解码路由器协调解耦的工作负载，合并来自预填充和解码阶段的元数据。gRPC 路由器为性能关键部署提供更高吞吐量，内建原生分词和推理解析器支持。

为可靠性，网关实现带抖动的指数退避用于重试、工作进程范围的断路器（在工作进程变得不健康时自动故障转移），以及带排队的令牌桶限流。可观测性通过 Prometheus 指标（延迟、吞吐量、缓存命中率）、OpenTelemetry 追踪和结构化日志实现。

### 基于路由器的架构何时表现卓越

基于路由器的扩展在它的优势与工作负载特征对齐的特定场景中大放异彩。高 QPS 交互式工作负载——聊天机器人平台、企业多租户部署、处理 100K+ 并发会话的系统——最受益于路由器的缓存感知策略，这些策略最大化 RadixAttention 的前缀共享。

多轮对话是另一个最佳点。会话亲和性将对话的所有轮次保持在同一个工作进程上，那里之前消息的 KV 缓存已经预热。当用户发送 "What about Python?" 作为后续时，工作进程不需要重新计算系统提示或之前的交流——它全都被缓存。结合 RadixAttention 跨用户的前缀共享，这可以将后续请求的延迟减少 2-3 倍。但这个好处是有条件的：它需要有实际前缀重用的对话式工作负载。一次性提示不会看到这些收益。

PD 解耦让你独立于解码工作进程（内存受限）扩展预填充工作进程（计算受限）。SGLang 的路由器协调这两个池并在它们之间传输 KV 缓存。当一个工作进程失败时，路由器绕过它路由——无需重新分片模型权重或重启分布式组。

也就是说，基于路由器的架构并非普遍更优。对于批处理推理，路由器跳转增加延迟而不提供缓存局部性好处——你用直接模型并行更好。对于需要跨许多 GPU 广泛 TP/PP 的非常大的模型（70B+），路由器增加的价值很小，因为你无论如何受模型并行约束。而对于延迟无关紧要的仅吞吐量工作负载，vLLM 的连续批处理可能在没有路由开销的情况下实现更高效率。


## 预填充/解码解耦

SGLang 最强大的分布式模式之一是 **预填充/解码（PD）解耦**。相同的拆分出现在现代服务栈中——DistServe 和 Mooncake 早期记录了这个设计；vLLM 和 NVIDIA Dynamo 提供类似的模式——但动机是普遍的。回想第 6 章，预填充并行处理整个提示（计算受限），而解码一次生成一个 token（内存受限）。这些阶段有根本不同的资源需求。

![预填充 vs 解码资源利用率](img/pd_disaggregation_zh.png){#fig:pd-disaggregation .block width=95% align=center}

图~\ref{fig:pd-disaggregation} 展示了为什么解耦有意义。预填充并行处理许多 token，实现高计算利用率（85%）但中等内存带宽使用（40%）。解码一次生成一个 token，导致低计算利用率（25%）但高内存带宽使用（90%），因为它反复加载 KV 缓存。这些相反的资源特征意味着在同一硬件上混合两个阶段导致次优利用率——在任何给定时间，计算或内存带宽之一未被充分利用。

在统一系统中，预填充和解码竞争相同的资源。一个长预填充可以阻塞解码工作进程，为等待 token 的用户造成延迟尖峰。在数据并行设置中，一个工作进程可能在做预填充而另一个处理解码，导致不一致的延迟。

PD 解耦通过完全分离工作负载来解决这个问题。专用预填充工作进程处理初始提示处理，为计算吞吐量优化。专用解码工作进程处理 token 生成，为低延迟优化。路由器将新请求引导到预填充工作进程，然后将 KV 缓存传输到解码工作进程进行生成。

这与 vLLM 的流水线并行（PP）根本不同，后者跨阶段拆分模型层。PP 需要阶段之间严格的同步——每个阶段必须等待前一个。PD 解耦分离工作负载类型而非模型层，允许无同步开销的独立扩展。


### PD 解耦架构

PD 解耦中的数据流涉及一个控制平面和一个数据平面，路由器编排两者。

![PD 解耦架构](img/pd_architecture_zh.png){#fig:pd-architecture .block width=100% align=center}

图~\ref{fig:pd-architecture} 显示完整流程。单个路由器处理入口和出口。在阶段 1，请求到达，路由器将它们分派到一个预填充工作进程（计算受限）。路由器还选择哪个解码工作进程将处理生成——这是控制平面（虚线箭头）。在阶段 2，预填充工作进程通过 RDMA 或 Mooncake 将 KV 缓存块直接传输到选定的解码工作进程——这是数据平面，一个绕过路由器的工作进程到工作进程传输。解码工作进程（内存受限）然后生成 token 并通过路由器将它们流回客户端。注意原始提示从不去解码工作进程；它们只从预填充工作进程接收 KV 缓存块。

好处是显著的。你可以独立扩展预填充和解码工作进程——当提示处理成为瓶颈时添加更多预填充工作进程，当生成延迟重要时添加更多解码工作进程。不同硬件可以用于不同工作负载：高计算 GPU 用于预填充，内存优化 GPU 用于解码。故障隔离改善，因为预填充工作进程的失败不影响正在进行的解码操作。

### 传输引擎

PD 解耦中的关键挑战是将 KV 缓存从预填充工作进程传输到解码工作进程。对于长上下文，这个缓存可能是 GB 级的数据，所以高效传输对维持低延迟至关重要。

SGLang 支持为不同网络结构优化的多个传输引擎。**Mooncake** 使用 RDMA（远程直接内存访问）进行无 CPU 参与的直接内存到内存传输，在 InfiniBand 网络上实现最低延迟。**NIXL** 提供一个跨不同网络结构（InfiniBand、以太网）工作的基于 UCX 的接口，以一些性能为代价提供灵活性。**ASCEND** 专门用于华为昇腾 NPU 部署。

### PD 解耦设置

设置 PD 解耦需要启动预填充工作进程、解码工作进程和一个配置为协调它们的路由器。这是一个使用 Mooncake 传输引擎的完整示例。

首先，安装传输引擎：

```bash
uv pip install mooncake-transfer-engine
```

启动一个预填充工作进程。`--disaggregation-mode prefill` 标志配置工作进程为仅预填充操作，`--disaggregation-ib-device` 指定 RDMA 传输的 InfiniBand 设备：

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode prefill \
    --port 30000 \
    --disaggregation-ib-device mlx5_roce0
```

在不同的 GPU 上启动一个解码工作进程：

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --disaggregation-ib-device mlx5_roce0
```

最后，启动启用 PD 解耦的路由器：

```bash
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --host 0.0.0.0 \
    --port 30000
```

对于 NIXL 后端（它跨不同网络结构工作），将 `--disaggregation-ib-device mlx5_roce0` 替换为 `--disaggregation-transfer-backend nixl`。

## 分布式推理架构与并行策略

SGLang 的核心创新（RadixAttention、零开销调度器）在执行引擎级别工作。为扩展，SGLang 支持传统并行策略（TP/PP/EP）和基于路由器的请求分发。理解何时使用每种方法——以及它们如何组合——对构建可扩展的推理系统至关重要。

SGLang 支持四个并行维度，可以根据需要组合。**张量并行（TP）** 在节点内跨 GPU 拆分模型权重，用 all-reduce 同步。**流水线并行（PP）** 跨 GPU 或节点拆分模型层，用阶段之间的点对点通信。**数据并行（DP）** 跨工作进程复制模型，每个独立处理不同的请求。**专家并行（EP）** 跨设备分布 MoE 专家，用 all-to-all 通信进行 token 路由。

这些维度相乘：用 TP=8、PP=2、DP=4 和 EP=2，你会用 8 × 2 × 4 × 2 = 128 块 GPU。但关键洞见是 SGLang 的基于路由器的架构为传统 DP 提供替代——你不用带同步的训练式数据并行复制模型，而是可以在路由器后运行无同步开销的独立工作进程。

### 张量并行：权重分片

当一个模型对单块 GPU 太大时，张量并行（TP）跨多块 GPU 拆分模型权重。SGLang 的 TP 实现使用与 vLLM 相同的 Megatron 风格算法——带 all-reduce 通信的列并行和行并行线性层——但为与 SGLang 的调度器和 RadixAttention 协作而定制。

核心思想很直接：每层的权重跨 TP rank 分区，每块 GPU 存储 1/TP 的权重，每层后的 all-reduce 操作组合部分结果。对于注意力层，QKV 投影按列跨 rank 拆分。对于 MLP 层，上投影用列并行，下投影用行并行。

通信开销显著：all-reduce 操作在每个注意力和 MLP 层后发生。这使 TP 对网络拓扑高度敏感——当 TP 组中的所有 GPU 在单个节点内通过 NVLink 连接时工作最好。跨节点 TP 可能，但需要像 InfiniBand 这样的高带宽互连。

这是如何为带 8 块 GPU 的单节点配置 TP：

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp 8
```

对于多节点 TP（当模型对单个节点太大时），你需要指定分布式初始化地址和节点 rank：

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

### 流水线并行：层分片

流水线并行（PP）跨 GPU 或节点拆分模型深度。与水平拆分每层的 TP 不同，PP 将连续的层分配给不同的阶段。阶段 0 可能处理层 0-7，阶段 1 处理层 8-15，以此类推。

通信模式比 TP 更简单：相邻阶段之间的点对点传输，而非跨所有 rank 的 all-reduce。这使 PP 更适合带宽有限的跨节点部署。

PP 的挑战是流水线气泡——流水线未满时处理开始和结束的空闲时间。启动期间，较后的阶段等待较早的阶段产生输出。排空期间，较早的阶段在较后的阶段之前完成。各种技术（微批处理、虚拟流水线并行）有助于最小化这些气泡，但不能完全消除。

PP 通常与 TP 结合：节点内用 TP（NVLink 提供高带宽），跨节点用 PP（较低带宽互连可接受）：

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --tp 8 \
    --pp 4 \
    --dist-init-addr 172.16.4.52:20000 \
    --nnodes 2 \
    --node-rank 0
```

### 数据并行：请求级复制

数据并行（DP）跨多个工作进程复制模型，每个工作进程处理不同的请求。与 TP 和 PP 不同，推理期间工作进程之间没有通信——每个独立操作。

这是 SGLang 的基于路由器的架构大放异彩的地方。传统 DP（如训练中使用的）需要梯度同步。但对于推理，工作进程真正独立。SGLang 的路由器提供智能请求分发，无任何同步开销。

DP 和 TP 之间的权衡很清楚：

| 方面 | 数据并行 | 张量并行 |
|--------|------------------|-------------------|
| **内存** | 每工作进程完整模型 | 每工作进程 1/TP 模型 |
| **通信** | 无（推理） | 每层 all-reduce |
| **延迟** | 更低（无同步） | 更高（同步开销） |
| **吞吐量** | 小批次更高 | 大批次更高 |
| **可扩展性** | 受模型大小限制 | 扩展到非常大的模型 |

对于装进单块 GPU 的模型，带基于路由器分发的 DP 几乎总是比 TP 更好。你得到更低的延迟（无同步）、更好的容错（工作进程独立），以及 TP 不能提供的像会话亲和性这样的特性。

这是带路由器的最简单 DP 设置：

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

### 混合并行：组合策略

真实世界的部署常常组合多种并行策略。最常见的模式是节点内 TP（利用 NVLink）结合跨节点 PP 或基于路由器的 DP。

对于需要多节点的大型模型，TP+PP 是典型的。用跨 4 个节点的 32 块 GPU，你可能用 TP=8（每个节点内）和 PP=4（跨节点）。每个流水线阶段有一个 8-GPU TP 组，阶段通过点对点传输通信。

对于装进较小 GPU 组的模型的高吞吐量部署，TP+DP（通过路由器）常常更好。用 16 块 GPU，你可能运行 4 个独立工作进程，每个用 TP=4。路由器跨工作进程分发请求，无同步开销。

对于 MoE 模型，专家并行（EP）添加另一个维度。一个大型 MoE 部署可能用节点内 TP=4、跨节点对 PP=2，和 EP=16 分布专家。通信模式变得复杂——TP 的 all-reduce、PP 的点对点、EP 的 all-to-all——但并行维度是正交的，可以独立配置。

### 通信模式

每种并行策略有一个特征通信模式。理解这些模式有助于你选择正确的策略并优化性能。

**张量并行** 在每层后使用 all-reduce。每块 GPU 将它的部分结果发送给其他每块 GPU，它们都计算和。通信量是每层 `O(hidden_size × batch_size)`，且频繁发生——在每个注意力和 MLP 层后。这使 TP 对互连带宽高度敏感。

**流水线并行** 使用相邻阶段之间的点对点通信。只有相邻阶段通信，它们只发送激活值（而非梯度，因为这是推理）。通信量类似于 TP，但频率更低——每微批次一次而非每层。

**专家并行** 使用 all-to-all 通信进行 token 路由。每块 GPU 发送去往远程专家的 token 并接收去往本地专家的 token。通信模式比 TP 或 PP 更复杂，因为路由是数据相关的——不同 token 去不同专家。

**基于路由器的 DP** 推理期间没有通信。每个工作进程独立操作，路由器处理请求分发。这就是为什么基于路由器的架构为合适的工作负载实现更低延迟。

为优化通信，SGLang 支持将通信与计算重叠（用 `--tp-comm-overlap` 启用）、拓扑感知放置（将 TP 组保持在 NVLink 域内），以及多个通信后端（通用用 NCCL，MoE 用 DeepEP，RDMA 用 Mooncake）。

## 多节点 SGLang 部署

SGLang 支持两种根本不同的多节点部署方法。对于装不进单个节点的大型模型，你跨节点用张量并行（TP）和/或流水线并行（PP）——类似于 vLLM。对于较小的模型，你用基于路由器的架构，在每个节点上有复制的工作进程——SGLang 的独特方法。

选择取决于你的模型大小和工作负载特征。基于路由器的部署避免通信开销并实现像会话亲和性这样的特性，但需要每个节点持有完整模型。TP/PP 部署实现服务对单个节点太大的模型，但增加同步开销。

### 带张量并行的多节点

对于需要跨节点 TP 的模型，你需要配置分布式初始化。`--dist-init-addr` 参数指定主节点的地址用于 NCCL 初始化。每个节点需要一个唯一的 `--node-rank`（主节点 0，工作节点 1、2……），`--nnodes` 指定总节点数。

这是一个跨 2 个节点用 TP=16（每节点 8 块 GPU）部署模型的示例：

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

将 `172.16.4.52:20000` 替换为你的主节点的 IP 地址和一个可用端口。两个节点都必须能到达这个地址进行 NCCL 初始化。


### 基于路由器的多节点部署

对于装进单个节点（或小 TP 组）的模型，基于路由器的部署通常更好。每个节点运行一个带完整模型的独立工作进程，路由器跨它们分发请求。

![基于路由器的多节点部署](img/router_multi_node_zh.png){#fig:router-multi-node .block width=70% align=center}

图~\ref{fig:router-multi-node} 显示基于路由器的架构。缓存感知路由器跨工作节点分发请求，每个持有完整模型。工作进程独立处理请求，无工作进程间同步——这是相对模型并行的关键优势，后者中工作进程必须在每层协调。

设置很直接。在每个节点上启动一个工作进程：

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

然后配置路由器跨工作进程分发请求：

```bash
python -m sglang_router.launch_router \
    --worker-urls \
        http://node1:30000 \
        http://node2:30000 \
        http://node3:30000 \
    --policy cache_aware \
    --port 8080
```

`cache_aware` 策略根据前缀匹配路由请求，最大化 RadixAttention 缓存命中。对于基于会话的工作负载，带相同 `session_id` 的请求被路由到同一个工作进程，跨对话轮次维持 KV 缓存局部性。

### 路由器策略与负载均衡

路由器的策略决定请求如何跨工作进程分发。SGLang 提供几种为不同场景优化的策略。

**缓存感知策略** 推荐用于大多数工作负载。它为每个工作进程维护一个近似基数树，跟踪哪些前缀可能被缓存。当一个请求到达时，路由器找到有最佳前缀匹配的工作进程。如果匹配超过阈值，请求去那个工作进程（缓存命中）。否则，它回退到负载均衡。这个策略最大化 RadixAttention 好处，同时防止任何单个工作进程变得过载。

```bash
python -m sglang_router.launch_router \
    --worker-urls http://node1:30000 http://node2:30000 \
    --policy cache_aware \
    --cache-threshold 0.5 \
    --balance-abs-threshold 10 \
    --balance-rel-threshold 1.5
```

**轮询策略** 在不考虑缓存状态的情况下跨工作进程均匀分发请求。它简单且可预测，当请求不共享前缀或你想要均匀负载分布时有用。

**最短队列策略** 路由到待处理请求最少的工作进程。这适应可变的请求长度——处理长请求的工作进程自然接收更少的新请求。

**二选一策略** 采样两个随机工作进程并挑选负载较少的。这以比检查所有工作进程更低的开销提供良好的负载分布，是负载均衡文献中的经典技术。

### 会话亲和性与缓存局部性

会话亲和性是 SGLang 对对话式工作负载最强大的特性之一。想法很简单：来自同一对话的请求应该去同一个工作进程，那里之前轮次的 KV 缓存已经预热。

当一个带 `session_id` 的请求到达时，路由器检查是否有那个会话的现有映射。如果有，且映射的工作进程健康，请求去那个工作进程。如果没有，路由器用它的策略选择一个工作进程并创建一个新映射。

延迟好处是显著的。对话中的第一条消息需要完整预填充——为整个提示计算 KV 缓存。但后续消息可以重用之前轮次缓存的 KV，跳过大部分预填充计算。实践中，这意味着对话中后续请求的延迟低 2-3 倍。

这是 vLLM 的模型并行不能轻易提供的。在 TP/PP 部署中，没有自然的"工作进程"可路由——模型分布在所有 GPU 上。会话状态需要被显式管理并可能在请求之间传输。SGLang 的基于路由器的架构使会话亲和性成为设计的自然结果。

### 容错

基于路由器的架构提供自然的容错。路由器通过对每个工作进程的 `/health` 端点的周期性探测持续监控工作进程健康。当一个工作进程失败或变得无响应时，路由器自动从活跃池移除它并将流量重新分发给健康的工作进程。

断路器防止级联失败。如果一个工作进程反复失败（通常连续 5 次失败），路由器"打开"电路并在超时期停止向那个工作进程发送请求。超时后，它发送单个探测请求——如果成功，工作进程重新加入池；如果不成功，电路保持打开。

会话迁移处理持有活跃会话的工作进程失败的情况。路由器将受影响的会话重新映射到健康的工作进程。权衡是 KV 缓存必须在新工作进程上重新计算，所以迁移后的第一个请求承受完整预填充延迟。但后续请求受益于新工作进程上的预热缓存。

这种容错自然来自架构。相比之下，TP/PP 部署需要所有 rank 可用——单个 GPU 失败可以拖垮整个服务组。

## MoE 模型的专家并行

我们在第 5 章和第 6 章涵盖了 MoE 架构和专家并行的基础——token 如何路由到专家、专家如何跨 GPU 分布，以及这需要的 all-to-all 通信模式。这里我们专注于 SGLang 对 MoE 推理的特定优化。

MoE 推理的挑战是 all-to-all 通信可以主导运行时，特别是大批大小时。SGLang 通过专门的后端和通信重叠技术解决这个问题。

对于 all-to-all 通信，SGLang 提供多个后端。DeepEP 为跨节点 MoE 部署优化，是多节点设置的推荐选择。Mooncake 用 RDMA 支持扩展 DeepEP，在 InfiniBand 网络上实现更低延迟。对于你想要 all-reduce 而非 all-to-all 的混合 EP+TP 配置，"none" 后端提供那个选项。

对于实际的专家计算（每个专家内的矩阵乘法），SGLang 提供 DeepGEMM（专门为 MoE 模式优化）、Triton（灵活且可移植）和 CUTLASS（NVIDIA 的高性能 GEMM 库）。"auto" 设置根据你的硬件选择最佳选项。

一个基本的 MoE 部署看起来像这样：

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --ep 8 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm
```

对于像 Qwen3-235B 这样的较大模型，你会将 EP 与其他并行维度组合：

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --tp 4 --ep 16 --pp 2 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm \
    --enable-dp-attention
```

`--enable-dp-attention` 标志对有少量 KV 头的模型（如使用 MLA 的）特别重要。它避免跨 TP rank 复制 KV 缓存，否则会浪费内存。

SGLang 最独特的 MoE 优化是通信重叠。双批次重叠（TBO）将批次拆分成微批次并将它们流水线化：当一个微批次执行 all-to-all 通信时，另一个运行注意力计算。这可以通过将通信延迟隐藏在计算之后来几乎倍增吞吐量。单批次重叠（SBO）使用多个 CUDA 流在单个批次内实现类似好处。用 `--enable-two-batch-overlap` 或 `--enable-single-batch-overlap` 启用这些。

为获得最佳结果，尽可能将 EP 组保持在 NVLink 域内——all-to-all 是带宽密集型的，NVLink 的 600+ GB/s 远超跨节点互连。

## 推测解码

推测解码是一种可以显著加速推理的优化技术。核心思想是用一个更小、更快的"草稿"模型预测多个 token，然后用更大的"目标"模型并行验证它们。

传统解码一次生成一个 token，每步需要通过模型的完整前向传播。这是内存受限的——GPU 大部分时间花在加载模型权重而非计算。推测解码通过将多个验证步骤批处理在一起来改变这个。

它是这样工作的。草稿模型（通常比目标小 2-4 倍）快速生成 N 个草稿 token。然后目标模型在单个前向传播中验证所有 N 个 token。如果所有草稿 token 匹配目标会生成的，你以一次前向传播的成本得到 N 个 token——N 倍加速。如果一些 token 不匹配，从第一个不匹配点使用目标模型的输出，被拒绝 token 的 KV 缓存被驱逐。

实践中，接受率因草稿模型如何近似目标而异。对于匹配良好的模型对（同族、同训练数据），70-90% 的接受率常见，产生 2-4 倍加速。

SGLang 支持带可配置草稿模型的推测解码：

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --speculative-draft-model-path Qwen/Qwen2.5-0.5B-Instruct \
    --speculative-num-draft-tokens 4
```

草稿和目标模型应共享相同的分词器和词汇表。使用同族的模型（如用 Qwen2.5-0.5B 作为 Qwen2.5-7B 的草稿）通常给出最佳结果。

推测解码与 RadixAttention 良好复合。RadixAttention 通过为共享前缀重用缓存的 KV 减少预填充时间，改善首 token 时间（TTFT）。推测解码然后通过每次前向传播生成多个 token 加速解码阶段。它们一起可以大幅减少对话式工作负载的端到端延迟：RadixAttention 处理前缀，推测解码处理生成。

## 数据并行注意力

数据并行注意力（DP Attention）是 SGLang 对有少量 KV 头的模型（如使用多头潜在注意力（MLA）的）的优化。它解决的问题微妙但重要。

在传统张量并行中，QKV 投影跨 GPU 拆分。但当 KV 头的数量少于 TP 大小时，KV 头必须跨 GPU 复制。对于一个有 1 个 KV 头和 TP=8 的模型，8 块 GPU 中的每一块存储 KV 缓存的完整副本——8 倍内存浪费。

DP Attention 采取不同的方法。它不用 TP 拆分注意力计算，而是用数据并行：每块 GPU 独立处理不同的请求，无 KV 缓存复制。在 MLP 层之前，一个 all-gather 组合来自所有 GPU 的注意力输出。MLP 然后用张量并行运行（因为 MLP 没有 KV 缓存问题），输出被切回每块 GPU。

结果是对有少量 KV 头的模型高达 1.9 倍更高的解码吞吐量，因为内存不浪费在复制的 KV 缓存上。用以下命令启用它：

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --enable-dp-attention \
    --dp-size 8 \
    --tp-size 8
```

DP Attention 对内存效率重要的大批大小最有益。对于低延迟、小批次场景，all-gather 开销可能超过好处。

## 生产部署模式

正确的部署模式主要取决于模型大小。对于装进单块 GPU 的 10B 参数以下的模型，基于路由器的数据并行几乎总是最佳选择——每个节点运行一个完整模型，路由器用缓存感知路由和会话亲和性分发请求。这种配置对对话式工作负载通常实现 50-100ms TTFT、1000+ QPS 和 60-80% 的缓存命中率。

对于需要 2-8 块 GPU 的 10B 到 100B 参数之间的模型，在每个节点内用张量并行，跨节点用基于路由器的分发。每个节点运行一个 TP 组，路由器将每个 TP 组视为单个工作进程。对于 100B 参数以上的模型，你需要跨节点 TP+PP，类似于 vLLM 的方法——模型对基于路由器的复制太大。

MoE 模型将专家并行与张量并行组合。如果模型有少量 KV 头则启用 DP Attention，用 TBO/SBO 进行通信重叠。当预填充和解码有非常不同的特征（长提示带短生成，或反之）时，PD 解耦让你独立扩展每个阶段。

几个实用优化跨部署模式适用。对于通信，将张量并行保持在 NVLink 域内，并为你的网络适当配置 NCCL（`NCCL_IB_DISABLE=0`、`NCCL_IB_GID_INDEX=3`、`NCCL_SOCKET_IFNAME=ib0`）。对于内存效率，为长上下文工作负载启用分块预填充（`--enable-chunked-prefill` 配合 `--max-num-batched-tokens`）——这将长提示分成与解码操作交错的较小块，避免内存尖峰和流水线气泡（见第~\ref{chap:distributed-inference-fundamentals-and-vllm}章的第~\ref{sec:chunked-prefill}节）。用 FP8 或 INT4/AWQ 量化减少内存占用并可提高吞吐量。对于延迟敏感工作负载，用缓存感知路由和推测解码；对于吞吐量敏感工作负载，扩展数据并行并启用所有重叠选项（`--tp-comm-overlap`、`--enable-two-batch-overlap`）。

## 实操示例 {#sec:sglang-hands-on}

有了架构概念和优化策略，让我们走一遍常见场景的完整部署配置。

### 基本多节点部署 {#sec:sglang-basic-deployment}

最简单的 SGLang 部署在多个节点上运行工作进程，带一个路由器进行负载均衡。这个模式实现基于路由器的分布式架构一节描述的基于路由器的数据并行架构，其中每个工作进程持有一个完整模型并独立处理请求。

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

`--policy cache_aware` 标志启用路由器基于前缀缓存局部性的智能请求分发。当一个请求到达时，路由器检查它的前缀并将它路由到最可能有相关 KV 缓存条目的工作进程——这是 RadixAttention 的跨请求优化如何扩展到分布式部署。没有这个标志，路由器回退到轮询分发，它仍然提供负载均衡但失去缓存局部性好处。

对于生产部署，你会想在路由器级别添加健康检查（`--health-check-interval`）、连接超时和可能的 TLS 终止。路由器在 `/metrics` 暴露 Prometheus 指标，用于监控请求延迟、缓存命中率和每工作进程负载分布。

### 用 Mooncake 做 PD 解耦 {#sec:sglang-pd-example}

对于预填充和解码特征差异明显的工作负载——例如长提示配短生成，或上下文很大但响应简短的 RAG 应用——PD 解耦将这两个阶段分离到专门的工作进程上。这种架构在"预填充/解码解耦"一节中有详细介绍，它允许计算受限的预填充和内存受限的解码资源独立扩展。

```bash
# Install transfer engine
uv pip install mooncake-transfer-engine

# Prefill worker (compute-optimized)
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode prefill \
    --port 30000 \
    --disaggregation-ib-device mlx5_roce0

# Decode worker (memory-optimized)
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --disaggregation-ib-device mlx5_roce0

# Router with PD-aware scheduling
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --port 8080
```

`--disaggregation-ib-device mlx5_roce0` 标志指定了预填充和解码工作进程之间传输 KV 缓存所用的 RDMA 设备。这对性能至关重要——没有 RDMA，KV 缓存块必须经过网络协议栈，带来显著的延迟。Mooncake 传输引擎负责零拷贝的数据搬运，正如"传输引擎"一节所述。

在生产环境中，你通常会运行多个预填充和解码工作进程，比例由你工作负载的预填充与解码计算比决定。对于长上下文工作负载（预填充密集型），你可能使用 2:1 或 3:1 的预填充与解码比例；对于聊天机器人工作负载（解码密集型），1:2 的比例可能更合适。

### 测试会话亲和性 {#sec:sglang-session-example}

会话亲和性将来自同一对话的连续请求路由到同一个工作进程，最大化 KV 缓存复用。这个示例演示如何使用会话 ID 并测量由此带来的加速——其机制在"会话亲和性与缓存局部性"一节中有描述。

```python
import requests

router_url = "http://router:8080/v1/chat/completions"
session_id = "test-session-123"
model = "Qwen/Qwen2.5-0.5B-Instruct"

# First request creates session and computes KV cache
response1 = requests.post(router_url, json={
    "model": model,
    "messages": [{"role": "user", "content": "Hello!"}],
    "session_id": session_id
})
print(f"First request: {response1.elapsed.total_seconds()}s")

# Second request reuses KV cache from first request
response2 = requests.post(router_url, json={
    "model": model,
    "messages": [{"role": "user", "content": "What did I say?"}],
    "session_id": session_id
})
print(f"Second request: {response2.elapsed.total_seconds()}s")
print(f"Speedup: {response1.elapsed / response2.elapsed:.2f}x")
```

你观察到的加速取决于请求之间的重叠程度。如果第二个请求的前缀与第一个请求的完整上下文（系统提示 + 对话历史）匹配，RadixAttention 可以为共享部分完全跳过预填充。在系统提示较长的多轮对话中，这可以将 TTFT 降低 50%-90%。路由器的会话亲和性确保这些请求落在 KV 缓存所在的同一个工作进程上。

对于没有天然会话边界的应用（例如批处理），你仍然可以通过将有共同前缀的请求排序分组，再配合缓存感知路由策略，从前缀共享中受益。

### 用 EP+TP 部署 MoE 模型 {#sec:sglang-moe-example}

MoE（混合专家）模型需要用专家并行将专家分布到多块 GPU 上，并结合张量并行处理稠密层。这个配置在"MoE 模型的专家并行"一节中有详细介绍，展示了 SGLang 对混合并行策略的支持。

```bash
python -m sglang.launch_server \
    --model-path microsoft/Phi-tiny-MoE-instruct \
    --tp 8 --ep 16 \
    --moe-a2a-backend deepep \
    --moe-runner-backend deep_gemm \
    --enable-dp-attention \
    --enable-two-batch-overlap
```

这些标志的作用分解如下：`--tp 8` 用张量并行将稠密层分片到 8 块 GPU 上；`--ep 16` 将专家分布到 16 块 GPU 上（本例中每个专家组 2 块 GPU）。`--moe-a2a-backend deepep` 标志为专家路由期间的 all-to-all 通信选择 DeepEP 后端——它针对 MoE 的稀疏激活模式做了优化。`--moe-runner-backend deep_gemm` 标志为专家计算使用融合的 GEMM 内核。

还启用了两项额外优化：`--enable-dp-attention` 激活数据并行注意力（"数据并行注意力"一节有描述），这对使用 MLA 或 GQA 等少 KV 头的模型有利。`--enable-two-batch-overlap` 标志启用双批次重叠（TBO），它通过将 all-to-all 通信与相邻批次的计算重叠来隐藏通信延迟。

## 小结

在本章中，我们探索了 SGLang 对 LLM 推理的独特方法。核心洞察是：SGLang 并不排斥模型并行——它像 vLLM 一样支持 TP、PP 和 EP。SGLang 的与众不同之处在于，它把跨请求优化提升为头等关注点。vLLM 主要专注于请求内效率（我们能多快处理一个请求？），而 SGLang 还进一步追问：多个请求如何共享工作并相互受益？

对这个问题的回答，催生了 SGLang 的核心技术创新。RadixAttention 将 KV 缓存组织为一棵基数树，让有共同前缀的请求可以共享已缓存的计算结果。当许多用户发送带有相同系统提示的请求时，该提示的 KV 缓存只需计算一次并被复用——对于有长共享前缀的工作负载，这可能节省高达 90% 的预填充计算。对于没有前缀共享的工作负载，这种收益会减弱，这正是为什么在系统之间做选择时，理解你的工作负载特征很重要。

零开销调度器解决的是另一个瓶颈：传统的串行模式中，GPU 在 CPU 调度下一个批次时处于空闲状态。通过将 CPU 调度与 GPU 计算重叠——在 GPU 处理批次 N 的同时准备批次 N+1——SGLang 让 GPU 持续保持忙碌。这是一种内核级优化，无论你如何部署系统都能生效。

XGrammar 解决的是结构化输出生成问题，这是需要 JSON、SQL 或其他格式化输出的应用的常见需求。它不是在运行时逐 token 验证——在 128K token 词表规模下这样做代价高得令人却步——而是将合法的 token 集合预编译为有限状态机。结果是以极小的开销获得保证合法的结构化输出。

在这个执行引擎的基础上，SGLang 引入了基于路由器的架构作为扩展的基本单元。路由器将请求分发到各个独立的工作进程，每个工作进程持有一个完整模型（或一个小的 TP 组）。这消除了请求间同步——工作进程之间不互相协调，只与路由器协调。再结合会话亲和性（将同一对话的多轮请求路由到同一工作进程）和缓存感知负载均衡，这种架构在高 QPS、许多并发会话共享相同模式的交互式工作负载上表现出色。

PD 解耦更进一步，将预填充（计算受限）与解码（内存受限）分离到专门的工作进程上。路由器将新请求派发给预填充工作进程，后者通过 RDMA 将 KV 缓存块直接传输给解码工作进程。每种类型的工作进程都可以独立扩展，并针对其特定的资源特征进行调优。

什么时候应该选择 SGLang 而不是 vLLM？这个决定不在于哪个系统"更好"——而在于哪种优化重点匹配你的工作负载。当跨请求优化很重要时，SGLang 大放异彩：拥有共享系统提示的聊天机器人平台、会话亲和性能保留 KV 缓存的多轮对话、基于路由器的扩展能避免模型并行开销的高 QPS 部署，以及需要结构化输出的应用。而当你需要在许多 GPU 上大规模使用 TP/PP 服务超大模型时，当批处理吞吐量比延迟更重要时，或者当你的流量以没有前缀复用的单次提示为主时，vLLM 更为出色。

值得指出的是，vLLM 也支持前缀缓存和会话固定——这两个系统的能力并非互斥。它们代表了不同的设计取舍，对"该优化什么"这个问题给出了不同的答案。许多生产部署两者都用：vLLM 用于批处理和大模型服务，SGLang 用于 RadixAttention 和会话亲和性能带来延迟优势的交互式 API。一个路由层根据工作负载特征将流量引导到合适的后端。

至此，我们已经覆盖了分布式 AI 的两个方面：在模型开发阶段为吞吐量和内存效率做优化的训练系统（DDP、FSDP、DeepSpeed、Megatron），以及在服务已训练模型时为延迟和吞吐量做优化的推理系统（vLLM、SGLang）。下一章提供一份实战指南，介绍如何用 Slurm——为大多数研究集群和云 GPU 提供商提供支持的作业调度器——在 HPC 集群上运行这些工作负载。

## 有用的链接

__SGLang 与 RadixAttention__

- SGLang: Efficient Execution of Structured Language Model Programs (2023)：\url{https://arxiv.org/abs/2312.07104}
- SGLang v0.4: Faster, Longer, and Scalable LLM Serving (2025)：\url{https://arxiv.org/abs/2506.21901}
- SGLang 文档：\url{https://docs.sglang.io/}
- SGLang GitHub：\url{https://github.com/sgl-project/sglang}

__分布式推理__

- SGLang Model Gateway（路由器）：\url{https://docs.sglang.io/advanced_features/router.html}
- PD 解耦：\url{https://docs.sglang.io/advanced_features/pd_disaggregation.html}
- 专家并行：\url{https://docs.sglang.io/advanced_features/expert_parallelism.html}
- 多节点部署：\url{https://docs.sglang.io/references/multi_node_deployment/multi_node.html}

__结构化输出与受限解码__

- XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models (2024)：\url{https://arxiv.org/abs/2411.15100}
- SGLang 中的受限解码：\url{https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/constraint-decoding}
- 理解受限解码：\url{https://www.aidancooper.co.uk/constrained-decoding/}

__教程与讲解__

- SGLang 代码走读：\url{https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/readme.md}
- SGLang 调度器演进：\url{https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/scheduler-evolution/SGLang%20Scheduler%20Evolution.md}
- Why SGLang is a Game-Changer for LLM Workflows (Hugging Face, 2025)：\url{https://huggingface.co/blog/paresh2806/sglang-efficient-llm-workflows}
- Use Cases Favoring vLLM vs SGLang (2025)：\url{https://kanerika.com/blogs/sglang-vs-vllm/}

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
