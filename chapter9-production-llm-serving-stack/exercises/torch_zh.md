\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. 构建支持多模型分发的 API 网关

基于 FastAPI 和 `httpx` 实现一个统一入口网关，根据客户端请求体中的 `model` 字段将流量反向代理到不同的后端推理服务。

__要求：__

- 类签名：
```python
class ModelRouter:
    def __init__(self, model_endpoints: dict[str, str]):
        """初始化模型名称与后端服务地址的映射字典。"""
        pass
    
    async def route_request(self, request: dict) -> dict:
        """根据 model 字段反向代理并将请求转发给正确的后端。"""
        pass
```
- 解析请求体中的 `model` 字段
- 将请求异步转发给对应的后端 vLLM/SGLang 实例
- 若请求的模型未在网关注册，返回 404 错误与清晰的提示
- 对后端实例进行心跳健康探测
- 同时支持 `/v1/chat/completions` 与 `/v1/completions`

__测试代码：__
```python
import asyncio
import httpx

# 初始化网关路由表
router = ModelRouter({
    "llama-3.2-1b": "http://localhost:8001",
    "qwen-0.5b": "http://localhost:8002",
})

# 测试模型路由
request = {
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "你好！"}],
    "max_tokens": 50
}

response = asyncio.run(router.route_request(request))
print(f"[{request['model']}] 后端响应: {response}")

# 测试未知模型
unknown_request = {"model": "unknown-model", "messages": []}
try:
    asyncio.run(router.route_request(unknown_request))
except Exception as e:
    print(f"捕获预期异常: {e}")
```

### 2. 实现基于令牌桶（Token Bucket）的限流中间件

实现一个高并发异步限流器中间件，支持突发流量缓冲（Burst）与基于 API Key 的配额管理。

__要求：__

- 类签名：
```python
class RateLimiter:
    def __init__(self, requests_per_minute: int, burst_size: int = 10):
        """初始化令牌桶限流器配置。"""
        pass
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """检查请求是否允许通过（True: 放行, False: 触发限流）。"""
        pass
    
    def get_remaining_requests(self, client_id: str) -> int:
        """查询指定客户端当前剩余可用请求配额。"""
        pass
```
- 使用令牌桶（Token Bucket）算法进行速率平滑
- 按客户端唯一标识（API Key 或 IP）独立维护状态
- 支持 Burst 容量应对瞬时流量尖峰
- 在 HTTP 响应头中注入 `X-RateLimit-Remaining` 与 `X-RateLimit-Reset`
- 定期清理陈旧无请求的客户端记录，避免内存泄漏

__测试代码：__
```python
import asyncio
import time

limiter = RateLimiter(requests_per_minute=10, burst_size=5)

async def test_rate_limiting():
    client_id = "test-client-123"
    
    # 突发流量测试（允许前 5 个请求瞬间通过）
    for i in range(5):
        allowed = await limiter.check_rate_limit(client_id)
        remaining = limiter.get_remaining_requests(client_id)
        print(f"请求 {i+1}: 放行={allowed}, 剩余配额={remaining}")
    
    # 超过突发限制后触发限流
    for i in range(10):
        allowed = await limiter.check_rate_limit(client_id)
        if not allowed:
            print(f"在第 {i+5+1} 个请求时成功触发限流（429 Too Many Requests）")
            break

asyncio.run(test_rate_limiting())
```

### 3. 实现金丝雀灰度发布（Canary Deployment）与自动化回滚

构建一个支持动态流量百分比切分与根据实时错误率自动回滚的灰度控制器。

__要求：__

- 类签名：
```python
class CanaryDeployment:
    def __init__(self, stable_endpoint: str, canary_endpoint: str):
        """初始化稳定版与金丝雀后端地址。"""
        pass
    
    def set_canary_percentage(self, percentage: float):
        """设置分配给金丝雀后端的流量比例（0–100）。"""
        pass
    
    async def route_request(self, request: dict) -> dict:
        """根据设定比例动态路由到稳定版或金丝雀。"""
        pass
    
    def record_result(self, is_canary: bool, success: bool, latency_ms: float):
        """记录推理成功率与时延指标。"""
        pass
    
    def should_rollback(self, error_threshold: float = 0.05) -> bool:
        """金丝雀版本异常率超过阈值时触发自动回滚。"""
        pass
```
- 支持流量梯度切分（10% $\rightarrow$ 25% $\rightarrow$ 50% $\rightarrow$ 100%）
- 独立统计稳定版与金丝雀版本的错误率与延迟分位数
- 当金丝雀错误率显著高于基线阈值时，自动将流量切回 0% 并告警

__测试代码：__
```python
import asyncio
import random

canary = CanaryDeployment(
    stable_endpoint="http://localhost:8001",
    canary_endpoint="http://localhost:8002"
)

# 初始分配 10% 流量给金丝雀版本
canary.set_canary_percentage(10)

# 模拟 100 次线上请求
for i in range(100):
    request = {"model": "test", "messages": []}
    is_canary = random.random() < 0.1
    # 模拟金丝雀版本存在异常高错误率
    success = random.random() > (0.15 if is_canary else 0.01)
    latency = random.uniform(50, 200)
    
    canary.record_result(is_canary, success, latency)
    
    if canary.should_rollback(error_threshold=0.05):
        print(f"在处理到第 {i+1} 个请求时检测到异常，成功触发自动回滚（Rollback）！")
        canary.set_canary_percentage(0)
        break

print(f"最终金丝雀流量比例: {canary.canary_percentage}%")
```

### 4. 基于 OpenTelemetry 实现端到端分布式全链路追踪

在多组件微服务服务架构（网关、Tokenizer、模型引擎）中接入 OpenTelemetry SDK 进行分布式链路追踪。

__要求：__

- 为请求生命周期的关键阶段创建 Spans：
  - `gateway.receive`：网关接收请求
  - `gateway.route`：路由决策判定
  - `tokenizer.encode`：输入 Prompt 分词
  - `model.inference`：GPU 批处理推理计算
  - `tokenizer.decode`：输出 Token 解码
  - `gateway.respond`：网关完成响应
- 为 Span 附加业务与性能属性：`model.name`、`request.tokens`、`response.tokens`、`latency.ttft_ms`、`latency.total_ms`
- 支持 HTTP Header 上下文透传与导出

__测试代码：__
```python
import asyncio
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# 初始化 Tracer
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-serving")

async def process_request(request: dict):
    with tracer.start_as_current_span("gateway.receive") as span:
        span.set_attribute("model.name", request.get("model", "unknown"))
        
        with tracer.start_as_current_span("tokenizer.encode"):
            tokens = [101, 2054, 2003, 102]
            span.set_attribute("request.tokens", len(tokens))
        
        with tracer.start_as_current_span("model.inference"):
            await asyncio.sleep(0.05)  # 模拟 GPU 计算
            output_tokens = [101, 7592, 102]
            span.set_attribute("response.tokens", len(output_tokens))
        
        with tracer.start_as_current_span("tokenizer.decode"):
            response_text = "机器学习是人工智能的一个核心分支。"
        
        return response_text

request = {"model": "llama-3.2-1b", "messages": [{"role": "user", "content": "什么是机器学习？"}]}
response = asyncio.run(process_request(request))
print(f"输出结果: {response}")
```

### 5. 基于 k3d / Kubernetes 编排多模型与多引擎推理集群

使用 k3d 搭建支持 GPU 直通的本地 Kubernetes 集群，部署多模型路由与 API Gateway。

__要求：__

1. 创建支持 GPU 直通的 k3d 集群：
```bash
k3d cluster create llm-serving \
  --gpus=all \
  --volume /path/to/models:/models \
  --port "8080:80@loadbalancer"
```
2. 部署两个独立的 vLLM 后端服务：
   - 模型 1：`Qwen/Qwen2.5-0.5B-Instruct`（服务端口 8001）
   - 模型 2：`meta-llama/Llama-3.2-1B-Instruct`（服务端口 8002）
3. 部署统一的 API Gateway 反向代理并配置路由映射 ConfigMap
4. 通过 `curl` 发送请求验证根据 `model` 字段自动分发到正确的后端的完整链路

__测试代码：__
```bash
# 1. 验证模型列表聚合
curl http://localhost:8080/v1/models

# 2. 请求 Qwen 模型
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'

# 3. 请求 Llama 模型（同一网关，不同后端）
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 深刻理解生产级 LLM 服务系统的分层架构（Gateway, Model Runner, Observability, Tokenizer）
- 掌握多模型与多引擎（vLLM vs SGLang）基于 API Gateway 的统一反向代理与动态分发
- 熟练实现基于令牌桶（Token Bucket）的请求限流与反压（Backpressure）机制
- 掌握金丝雀灰度发布（Canary Deployment）与自动化熔断回滚控制策略
- 运用 OpenTelemetry 与 Prometheus 搭建涵盖 TTFT、TPOT、GPU 利用率与 KV Cache 命中率的全栈可观测性监控体系
- 掌握基于 Kubernetes / k3d 与云原生生态（llm-d, KServe）部署千万级高并发大模型推理服务的标准范式
