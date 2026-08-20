\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. 从零实现基于分块的分页 KV Cache 管理器

实现一个类似 vLLM PagedAttention 的简化版分页 KV 缓存管理器，深入理解块式显存分配与地址映射机制。

__要求：__

- 类签名：
```python
class KVCacheManager:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, 
                 block_size: int, num_blocks: int):
        """使用固定大小物理分块初始化 KV 缓存池。"""
        pass
    
    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """为指定序列动态分配所需物理块，返回分配的块索引列表。"""
        pass
    
    def free(self, seq_id: int):
        """释放指定序列占用的全部物理块并归还缓存池。"""
        pass
    
    def get_cache(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """根据 Block Table 组装并获取该序列的 K 与 V 物理张量。"""
        pass
```
- 基于固定大小物理块（Block-based Allocation，如每块 16 Tokens）进行分配
- 维护全局物理块空闲池（Free Pool）与已分配块跟踪表
- 支持自回归生成过程中序列长度动态增长时的按需追加分配
- 实现释放逻辑，序列完成生成后立即归还全部物理块消除碎片

__测试代码：__
```python
import torch

# 初始化 KV 缓存管理器
cache_mgr = KVCacheManager(
    num_layers=32,
    num_heads=32,
    head_dim=128,
    block_size=16,
    num_blocks=1000
)

# 为不同长度的并发请求动态分配物理块
seq1_blocks = cache_mgr.allocate(seq_id=1, num_tokens=100)
seq2_blocks = cache_mgr.allocate(seq_id=2, num_tokens=50)

print(f"序列 1 占用物理块数: {len(seq1_blocks)}")
print(f"序列 2 占用物理块数: {len(seq2_blocks)}")
print(f"当前剩余空闲块数: {cache_mgr.num_free_blocks}")

# 释放序列 1
cache_mgr.free(seq_id=1)
print(f"释放后可用空闲块数: {cache_mgr.num_free_blocks}")
```

### 2. 实现持续批处理调度器（Continuous Batching）

编写一个支持请求动态插入与完成即刻退出的连续批处理（Continuous Batching / 迭代级调度）引擎。

__要求：__

- 类签名：
```python
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size: int, max_seq_len: int):
        pass
    
    def add_request(self, request_id: int, prompt_tokens: list[int]):
        """将新到达的推理请求加入等待队列。"""
        pass
    
    def step(self) -> dict:
        """执行单步迭代生成，返回本步已完成生成的请求 ID 列表。"""
        pass
    
    def get_batch(self) -> list[int]:
        """获取当前正在活跃执行 Decode 的请求 ID 列表。"""
        pass
```
- 维护待处理请求队列与活跃 Batch 集合
- 在每个 Decode Step 结束后，检查是否有请求生成了结束符（EOS）或达到最大长度；一旦完成立即剔除
- 若当前 Batch 存在空闲槽位，立即从等待队列中吸纳新请求执行 Prefill 并合流
- 消除传统静态批处理中“木桶短板”导致的 GPU 空转

__测试代码：__
```python
import random
import time

scheduler = ContinuousBatchingScheduler(max_batch_size=8, max_seq_len=2048)

# 模拟异步到达的并发请求流
for i in range(20):
    prompt_len = random.randint(10, 100)
    scheduler.add_request(request_id=i, prompt_tokens=list(range(prompt_len)))
    
    # 模拟执行若干步自回归生成
    for _ in range(5):
        completed = scheduler.step()
        if completed:
            print(f"已完成请求: {completed}")
    
    print(f"Step {i}: 当前活跃 Batch 大小 = {len(scheduler.get_batch())}")
```

### 3. vLLM 高并发推理吞吐与延迟基准评测

编写一个系统化的基准测试脚本，量化评测 vLLM 在不同并发配置下的性能表现。

__要求：__

- 测试变量维度：
  - Batch Size：1, 4, 8, 16, 32
  - 输入 Prompt 长度：128, 512, 1024, 2048 tokens
  - 输出生成长度：64, 128, 256 tokens
  - 张量并行度（TP Size）：1, 2, 4 GPUs
- 精确测量并输出：
  - 系统总体吞吐量（Tokens/sec）
  - 首 Token 生成延迟（TTFT, Time to First Token）
  - 每个输出 Token 平均生成延迟（TPOT, Time Per Output Token）
  - GPU 显存利用率与 KV Cache 占用率
- 生成量化对比分析曲线

__测试代码：__
```python
from vllm import LLM, SamplingParams
import time

def benchmark_vllm(
    model_name: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    tp_size: int = 1
) -> dict:
    """评测特定配置下的 vLLM 性能指标。"""
    llm = LLM(model=model_name, tensor_parallel_size=tp_size)
    
    # 构建测试 Prompt
    prompts = ["Hello " * (input_len // 2)] * batch_size
    sampling_params = SamplingParams(max_tokens=output_len)
    
    # 热身 Warmup
    _ = llm.generate(prompts[:1], sampling_params)
    
    # 计时评测
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    
    return {
        "throughput": throughput,
        "latency": elapsed / batch_size,
        "ttft": outputs[0].metrics.first_token_time,
    }

# 运行评测矩阵
for batch_size in [1, 4, 8, 16]:
    for input_len in [128, 512, 1024]:
        metrics = benchmark_vllm(
            "Qwen/Qwen2.5-0.5B-Instruct",
            batch_size, input_len, output_len=128
        )
        print(f"Batch={batch_size:2d}, InputLen={input_len:4d}: 吞吐量={metrics['throughput']:6.1f} tok/s")
```

### 4. 手动实现投机采样（Speculative Decoding）算法

实现一个基于小型 Draft Model 预测与大型 Target Model 并行验证的投机采样加速器。

__要求：__

- 使用轻量级小模型（如 Qwen2.5-0.5B）快速推测生成 $K$ 个候选 Token
- 将包含 $K$ 个候选 Token 的完整序列一次性输入大模型（如 LLaMA-3.2-1B）执行并行前向验证
- 实现基于概率比率的拒绝采样（Rejection Sampling），保证输出分布与纯大模型生成严格等价
- 测量并对比投机采样相比传统自回归生成的端到端加速比（Speedup）

__测试代码：__
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    def __init__(self, target_model, draft_model, tokenizer, num_speculative: int = 4):
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.num_speculative = num_speculative
    
    def generate(self, prompt: str, max_tokens: int) -> str:
        """执行投机采样文本生成。"""
        pass
    
    def _draft_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """使用 Draft 模型自回归生成候选 Token。"""
        pass
    
    def _verify_tokens(self, input_ids: torch.Tensor, 
                       draft_tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        """使用 Target 模型并行验证候选 Token，返回接受的 Token 序列与接受数量。"""
        pass

# 初始化模型进行对比验证
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").cuda()
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").cuda()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

decoder = SpeculativeDecoder(target, draft, tokenizer, num_speculative=4)
output = decoder.generate("人工智能在现代工业体系中的核心价值是", max_tokens=50)
print(f"投机采样输出结果: {output}")
```

### 5. 部署兼容 OpenAI 标准协议的 vLLM 服务端并编写异步客户端

搭建并启动 vLLM 在线推理服务，编写支持流式传输（Streaming）与异常重试的 Python 异步客户端。

__要求：__

- 启动 vLLM OpenAI 兼容服务端
- 使用 `httpx` 编写异步客户端，支持 `/v1/chat/completions` 与 `/v1/models`
- 实现 Server-Sent Events (SSE) 异步流式输出解析
- 添加请求超时重试与网络抖动退避机制

__测试代码：__
```bash
# 启动服务端（在终端运行）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --port 8000
```

```python
import httpx
import asyncio
from typing import AsyncIterator

class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def chat(self, messages: list[dict], stream: bool = False) -> str | AsyncIterator[str]:
        """发送对话请求并支持流式接收。"""
        pass
    
    async def list_models(self) -> list[str]:
        """获取当前服务端就绪的模型列表。"""
        pass

async def main():
    client = VLLMClient()
    models = await client.list_models()
    print(f"在线就绪模型列表: {models}")
    
    messages = [{"role": "user", "content": "什么是机器学习？"}]
    # 异步流式打印
    print("流式输出: ", end="")
    async for chunk in await client.chat(messages, stream=True):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 深刻理解自回归生成中 Prefill（计算受限）与 Decode（显存带宽受限）的本质区别
- 掌握 PagedAttention 消除 KV Cache 显存碎片与零 Padding FLOPs 的底层运行机制
- 熟练实现并调优连续批处理（Continuous Batching）调度器
- 科学配置 vLLM 的张量并行（TP）、数据并行（DP）、流水线并行（PP）与专家并行（MoE EP）
- 掌握分块预填充（Chunked Prefill）平滑长文本请求时延的调优技术
- 掌握投机采样（Speculative Decoding）的数学原理与工程加速实战
- 熟练使用 Nsight Systems 诊断分布式推理中的通信与计算瓶颈
