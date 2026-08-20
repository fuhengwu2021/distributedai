\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. 从零实现 RadixAttention 基数树前缀缓存管理器

实现一个类似 SGLang RadixAttention 的基数树（Radix Tree / Trie）KV 缓存管理器，深入理解跨请求前缀复用与 LRU 淘汰机制。

__要求：__

- 类签名：
```python
class RadixCache:
    def __init__(self, max_size: int):
        """初始化基数树前缀缓存池。"""
        pass
    
    def insert(self, token_ids: list[int], kv_cache: torch.Tensor) -> int:
        """将 Token 序列及其对应的 KV Cache 插入基数树，返回缓存条目 ID。"""
        pass
    
    def lookup(self, token_ids: list[int]) -> tuple[int, torch.Tensor | None]:
        """查询最长匹配公共前缀，返回 (匹配 Token 长度, 匹配到的 KV Cache)。"""
        pass
    
    def evict(self, num_entries: int):
        """依据 LRU（最近最少使用）策略淘汰非共享叶子节点以释放显存。"""
        pass
```
- 基于前缀树（Trie / Radix Tree）组织存储
- 支持多分支共享公共前缀节点
- 记录每个节点的访问时间戳以实现 LRU 淘汰
- 统计前缀命中率（Cache Hit Rate）

__测试代码：__
```python
import torch

cache = RadixCache(max_size=1000)

# 插入多条存在重叠前缀的序列
seq1 = [1, 2, 3, 4, 5]
seq2 = [1, 2, 3, 6, 7]
seq3 = [1, 2, 8, 9]

kv1 = torch.randn(5, 32, 128)  # 5 tokens, 32 heads, 128 dim
kv2 = torch.randn(5, 32, 128)
kv3 = torch.randn(4, 32, 128)

cache.insert(seq1, kv1)
cache.insert(seq2, kv2)
cache.insert(seq3, kv3)

# 测试前缀查询
test_seq = [1, 2, 3, 4, 10, 11]
match_len, kv = cache.lookup(test_seq)
print(f"查询序列: {test_seq}")
print(f"最长匹配前缀长度: {match_len}")  # 应为 4（匹配 [1, 2, 3, 4]）
print(f"命中复用的 KV Cache 形状: {kv.shape if kv is not None else None}")
```

### 2. 跨请求共享前缀基准性能对比测试

编写测试脚本，量化评测 SGLang 在不同前缀共享程度下的首 Token 延迟（TTFT）与吞吐量收益。

__要求：__

- 构造不同前缀共享特征的合成请求流：
  - 零共享（完全随机独立的 Prompts）
  - 中度共享（共享标准 System Prompt）
  - 高度共享（Few-shot 示例或长上下文单文档多轮提问）
- 精确测量并对比：
  - 首 Token 生成时间（TTFT, Time to First Token）
  - Radix 缓存命中率（Cache Hit Rate）
  - GPU 显存消耗峰值与请求并发度
- 绘制加速曲线图

__测试代码：__
```python
import sglang as sgl
import time

def create_shared_prefix_workload(num_requests: int, prefix_len: int, unique_len: int):
    """构造包含指定共享前缀长度的工作负载。"""
    shared_prefix = "你是一个拥有丰富专业知识的人工智能助手。" * (prefix_len // 30)
    prompts = [
        shared_prefix + f"\n问题 {i}：请解释第 {i} 个概念？" 
        for i in range(num_requests)
    ]
    return prompts

def benchmark_sglang(prompts: list[str]) -> dict:
    """评测 SGLang 前缀缓存性能。"""
    runtime = sgl.Runtime(model_path="Qwen/Qwen2.5-0.5B-Instruct")
    
    start = time.time()
    for prompt in prompts:
        response = runtime.generate(prompt, max_new_tokens=50)
    elapsed = time.time() - start
    
    return {
        "total_time": elapsed,
        "cache_hit_rate": runtime.get_cache_stats()["hit_rate"],
    }

# 评测不同前缀长度下的加速比
for prefix_len in [0, 100, 500, 1000]:
    prompts = create_shared_prefix_workload(100, prefix_len, unique_len=50)
    results = benchmark_sglang(prompts)
    print(f"前缀长度={prefix_len:4d}: 总耗时={results['total_time']:5.2f}s, "
          f"缓存命中率={results['cache_hit_rate']:6.2%}")
```

### 3. 基于 XGrammar 的 JSON Schema 结构化约束解码

使用 SGLang 编写结构化生成程序，利用 XGrammar 语法状态机强制模型生成符合 Pydantic Schema 的合法 JSON 数据。

__要求：__

- 定义严格的 Pydantic 数据模型（包含嵌套对象或列表）
- 使用 SGLang 约束解码接口生成结构化输出
- 验证生成文本的 JSON 合法性与字段解析正确率
- 对比约束解码与无约束自由生成的额外延迟开销（验证 XGrammar 的极低开销）

__测试代码：__
```python
import sglang as sgl
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

@sgl.function
def extract_person(s, text: str):
    s += "请从以下文本中提取人物信息：" + text + "\n"
    s += "输出格式严格遵守 JSON：\n"
    s += sgl.gen("json_output", max_tokens=200, regex=Person.model_json_schema())

# 测试结构化生成
runtime = sgl.Runtime(model_path="Qwen/Qwen2.5-0.5B-Instruct")

text = "张伟今年35岁，是一名资深软件工程师，居住在北京。"
state = extract_person.run(text=text)

print(f"生成的 JSON 字符串: {state['json_output']}")

# 验证 Schema 解析
try:
    person = Person.model_validate_json(state['json_output'])
    print(f"成功解析对象: 姓名={person.name}, 年龄={person.age}, 职业={person.occupation}")
except Exception as e:
    print(f"JSON 验证失败: {e}")
```

### 4. 构建多轮带状态对话与 Fork 并行分支生成

使用 SGLang 原生状态管理与 `fork()` 机制，实现高效复用历史 KV Cache 的多轮对话与 Best-of-N 候选生成。

__要求：__

- 实现多轮对话程序，验证第 2 轮与第 3 轮自动复用前置轮次的 KV Cache
- 使用 `s.fork(N)` 瞬间克隆当前对话状态，并行探索多个不同的推理回答分支
- 测量 Fork 并行相比纯串行多次生成的加速比

__测试代码：__
```python
import sglang as sgl
import time

@sgl.function
def parallel_generation(s, prompt: str, num_samples: int = 4):
    s += sgl.user(prompt)
    
    # 克隆当前对话状态为 4 个独立分支并发生成
    forks = s.fork(num_samples)
    for i, fork in enumerate(forks):
        fork += sgl.assistant(
            sgl.gen(f"response_{i}", max_tokens=200, temperature=0.8)
        )
    
    # 汇总分支
    s += sgl.join(forks)

# 运行基准对比
runtime = sgl.Runtime(model_path="Qwen/Qwen2.5-0.5B-Instruct")

# 串行生成 4 次
start = time.time()
for _ in range(4):
    _ = runtime.generate("请简要解释量子计算的基本原理", max_new_tokens=200)
sequential_time = time.time() - start

# Fork 并发生成 4 个分支（共享 Prompt KV）
start = time.time()
state = parallel_generation.run(prompt="请简要解释量子计算的基本原理", num_samples=4)
parallel_time = time.time() - start

print(f"串行耗时: {sequential_time:.2f}s")
print(f"Fork 并发耗时: {parallel_time:.2f}s")
print(f"加速比: {sequential_time / parallel_time:.2f}x")
```

### 5. 部署 PD 分离（Prefill/Decode Disaggregation）与智能路由架构

配置基于 Mooncake RDMA 引擎的 Prefill/Decode 计算分离集群，并启动 Cache-Aware 智能网关。

__要求：__

- 分别启动独立的 Prefill 算力节点与 Decode 显存节点
- 配置 Mooncake RDMA 传输通道实现 KV Cache 零拷贝跨卡/跨机直传
- 启动 `sglang_router` 并配置 `cache_aware` 负载均衡策略
- 发送测试请求流，验证 Prefill 算力节点向 Decode 节点无缝交接 KV Cache

__测试代码：__
```bash
# 1. 启动 Prefill 计算密集型 Worker
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode prefill \
    --port 30000 \
    --disaggregation-ib-device mlx5_roce0

# 2. 启动 Decode 显存密集型 Worker（使用 GPU 1）
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --disaggregation-ib-device mlx5_roce0

# 3. 启动 PD 分离智能路由网关
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --policy cache_aware \
    --port 8080
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 深刻理解 SGLang 与 vLLM 在设计哲学上的差异（跨请求跨会话优化 vs 请求内模型并行）
- 掌握 RadixAttention 基数树前缀缓存的生命周期与 LRU 自动剪枝机制
- 掌握零开销调度器（Zero-Overhead Scheduler）CPU/GPU 重叠执行机理
- 熟练使用 XGrammar 状态机实现极低开销的 JSON/正则结构化约束输出
- 熟练编排并部署基于 Mooncake RDMA 的 Prefill/Decode 计算分离（PD Disaggregation）工业集群
- 掌握 SGLang Router 的 Cache-Aware 路由策略、会话粘性（Session Affinity）与自动故障转移
- 熟练调优 MoE 模型的专家并行（EP）、DP Attention 与两批重叠（Two-Batch Overlap）流水线
