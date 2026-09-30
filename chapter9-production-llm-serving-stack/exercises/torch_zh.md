\fancydividerwithicon[center]{hand.png}


## 实战演练


### 构建支持多模型动态路由的 API 网关

实现一个高性能 API 网关，根据请求体中的 `model` 字段将流量智能反向代理至不同的推理引擎后端。

__实战要求：__

- 类签名定义：
```python
class ModelRouter:
    def __init__(self, model_endpoints: dict[str, str]):
        """基于模型名称到后端真实 URL 的映射表初始化路由器。"""
        pass
    
    async def route_request(self, request: dict) -> dict:
        """根据 model 字段分发请求并透明返回后端响应。"""
        pass
```

- 解析入站请求中的 `model` 参数字段
- 将请求高效转发至对应模型实例的后端 URL
- 妥善处理所请求模型不存在或离线的情况（返回标准 404/503 异常）
- 增加针对各模型后端的异步周期性健康探测（Health Check）机制
- 同时兼容 `/v1/chat/completions` 与 `/v1/completions` 标准接口规范

__测试验证：__
```python
import asyncio
import httpx

# 初始化多模型网关
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
print(f"来自 {request['model']} 的响应: {response}")

# 测试未知模型异常拦截
unknown_request = {"model": "unknown-model", "messages": []}
try:
    asyncio.run(router.route_request(unknown_request))
except Exception as e:
    print(f"符合预期的捕获异常: {e}")
```

### 实现限流保护中间件（Rate Limiting Middleware）

编写基于令牌桶算法（Token Bucket）的请求限流拦截器，为不同客户端租户实施配额防护。

__实战要求：__

- 类签名定义：
```python
class RateLimiter:
    def __init__(self, requests_per_minute: int, burst_size: int = 5):
        """初始化速率上限与突发缓冲容量。"""
        pass
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """判定请求是否予以放行，超出配额返回 False。"""
        pass
    
    def get_remaining_requests(self, client_id: str) -> int:
        """查询当前客户端窗口内剩余可用令牌额度。"""
        pass
```

- 采用经典的令牌桶（Token Bucket）算法实现平滑限流
- 支持以 API Key 或客户端 IP 标识独立统计租户配额
- 支持突发流量缓冲（Burst Capacity），允许短时微峰值流量通过
- 返回符合 HTTP 标准的限流响应头（`X-RateLimit-Remaining`、`X-RateLimit-Reset`）
- 具备对过期闲置客户端的自动内存回收机制，防止长周期运行下的内存泄露

__测试验证：__
```python
import asyncio
import time

limiter = RateLimiter(requests_per_minute=10, burst_size=5)

async def test_rate_limiting():
    client_id = "test-client-123"
    
    # 突发流量应予以顺畅放行
    for i in range(5):
        allowed = await limiter.check_rate_limit(client_id)
        remaining = limiter.get_remaining_requests(client_id)
        print(f"请求 {i+1}: 放行状态={allowed}, 剩余配额={remaining}")
    
    # 配额耗尽后应触发限流拦截
    for i in range(10):
        allowed = await limiter.check_rate_limit(client_id)
        if not allowed:
            print(f"已在累计处理 {i+5} 个请求后成功触发限流保护")
            break

asyncio.run(test_rate_limiting())
```

### 实现具备自动故障回滚的金丝雀灰度发布

构建一个具备异常监控闭环的金丝雀（Canary）灰度发布系统，实现按比例切流与异常自动熔断回滚。

__实战要求：__

- 类签名定义：
```python
class CanaryDeployment:
    def __init__(self, stable_endpoint: str, canary_endpoint: str):
        """以稳定版与金丝雀灰度版服务入口初始化。"""
        pass
    
    def set_canary_percentage(self, percentage: float):
        """设定分配给灰度实例的流量百分比 (0-100)。"""
        pass
    
    @property
    def canary_percentage(self) -> float:
        """当前实际灰度切流百分比。"""
        pass
    
    async def route_request(self, request: dict) -> dict:
        """根据切流比例透明分发流量至对应后端。"""
        pass
    
    def record_result(self, is_canary: bool, success: bool, latency_ms: float):
        """记录请求质量指标供实时熔断策略决策。"""
        pass
    
    def should_rollback(self, error_threshold: float = 0.05) -> bool:
        """当灰度服务错误率超出阈值时返回 True 触发保护。"""
        pass
```

- 基于设定的百分比严格执行加权随机分流
- 实时滑动窗口统计 Stable 与 Canary 后端的错误率及 P99 时延
- 一旦检测到 Canary 错误率超出阈值（如相较基线高出 5%），自动触发熔断降级并清零流量（Rollback）
- 支持平滑阶梯式放量推进（10% -> 25% -> 50% -> 100%）

__测试验证：__
```python
import asyncio
import random

canary = CanaryDeployment(
    stable_endpoint="http://localhost:8001",
    canary_endpoint="http://localhost:8002"
)

# 初始切入 10% 探测流量
canary.set_canary_percentage(10)

# 模拟持续业务请求
for i in range(100):
    request = {"model": "test", "messages": []}
    
    # 模拟实际调用（假设金丝雀新版本存在缺陷，错误率显著偏高）
    is_canary = random.random() < 0.1
    success = random.random() > (0.1 if is_canary else 0.02)
    latency = random.uniform(50, 200)
    
    canary.record_result(is_canary, success, latency)
    
    if canary.should_rollback(error_threshold=0.05):
        print(f"在请求第 {i+1} 步检测到异常，触发自动保护回滚！")
        canary.set_canary_percentage(0)
        break

print(f"当前生效的金丝雀切流比例: {canary.canary_percentage}%")
```

### 基于 OpenTelemetry 实现端到端分布式全链路追踪

为多微服务 LLM 推理系统集成 OpenTelemetry 埋点，实现跨网关、分词器与推理引擎的调用链路追踪。

__实战要求：__

- 为请求生命周期的各核心阶段注入精细化 Span 追踪：
  - `gateway.receive`: 网关接收到入站请求
  - `gateway.route`: 动态路由解析决策
  - `tokenizer.encode`: 文本 Tokenizer 编码
  - `model.inference`: GPU 模型前向推理计算
  - `tokenizer.decode`: 生成 Token 流的解码还原
  - `gateway.respond`: 网关封装响应并交付给客户端
- 为各个 Span 挂载关键维度的可观测性属性标签（Attributes）：
  - `model.name`: 请求的模型名称
  - `request.tokens`: Prompt 输入 Token 总数
  - `response.tokens`: 输出生成 Token 总数
  - `latency.ttft_ms`: 首字生成耗时
  - `latency.total_ms`: 整体端到端链路耗时
- 跨服务进程通过 HTTP Header 透传 W3C Trace Context 追踪上下文
- 将 Trace 数据导出至控制台或 Jaeger 收集器

__测试验证：__
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# 初始化追踪框架
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-serving")

async def process_request(request: dict):
    with tracer.start_as_current_span("gateway.receive") as span:
        span.set_attribute("model.name", request.get("model", "unknown"))
        
        with tracer.start_as_current_span("tokenizer.encode"):
            tokens = tokenize(request["messages"])
            span.set_attribute("request.tokens", len(tokens))
        
        with tracer.start_as_current_span("model.inference"):
            output = await model.generate(tokens)
        
        with tracer.start_as_current_span("tokenizer.decode"):
            response = detokenize(output)
            span.set_attribute("response.tokens", len(output))
        
        return response

# 发起测试请求验证追踪导出
request = {
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "什么是机器学习？"}]
}
response = asyncio.run(process_request(request))
```

### 基于 k3d 构建本地多模型 GPU Kubernetes 集群

在轻量级 k3d 集群中配置 NVIDIA Container Toolkit GPU 直通，部署支持多模型路由的微服务服务栈。

__实战要求：__

1. 创建支持 GPU 透传的本地 k3d 集群环境：
```bash
k3d cluster create llm-serving \
  --gpus=all \
  --volume /path/to/models:/models \
  --port "8080:80@loadbalancer"
```

2. 部署两个独立的 vLLM 推理实例服务：
   - 实例 1：运行 `Qwen/Qwen2.5-0.5B-Instruct`，监听端口 8001
   - 实例 2：运行 `meta-llama/Llama-3.2-1B-Instruct`，监听端口 8002

3. 编写并部署统一的反向代理 API 网关，根据请求体中的 `model` 字段分发至对应服务

4. 验证多模型接入与服务可用性：
```bash
# 查询当前集群注册的模型列表
curl http://localhost:8080/v1/models

# 验证模型 1 补全接口
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "你好！"}]}'

# 验证模型 2 补全接口
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "你好！"}]}'
```

__交付成果清单：__

- 部署两个 vLLM 推理服务的 Kubernetes Deployment 与 Service YAML 配置
- API 网关的 Deployment、Service 与 Ingress 配置文件
- 维护模型名称至集群内部 DNS 映射的 ConfigMap 配置文件
- 一键自动化部署、健康巡检与功能验证的 Shell 自动化脚本


## 预期学习目标

完成本章实战练习后，你将能够：

- 熟练设计并实现支持异构大模型动态路由与反向代理的统一 API 网关
- 掌握令牌桶等高吞吐限流算法，有效抵御突发流量对 GPU 后端的雪崩冲击
- 领会微服务架构下金丝雀蓝绿灰度发布与自动化熔断回滚的工程闭环设计
- 使用 OpenTelemetry 全面观测分布式推理链路中的端到端延迟与 Token 吞吐细节
- 具备在 Kubernetes/k3d 容器化集群上编排、暴露与运维 GPU 推理服务栈的生产实操技能
- 构建具备自我防护、弹性伸缩与 SLA 质量兜底的工业级云原生 LLM 托管系统
