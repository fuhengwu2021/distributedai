# 第 9 章：生产级大语言模型服务化架构实战 {-}

*构建高可用、可观测、自动化弹性伸缩的工业级大模型在线服务体系*

> 所有系统在所有时刻都处于可能失效的状态。  
> —— 沃纳·威格尔斯（Werner Vogels，Amazon CTO）

**核心代码速查**

- `fastapi.responses.StreamingResponse`：基于 Server-Sent Events (SSE) 向客户端异步流式推送生成的 Token
- `fastapi.FastAPI`：构建统一推理 API 网关与后端模型服务的 HTTP 接口框架
- `uvicorn`：运行网关、模型推理引擎与 Tokenizer 微服务的高性能 ASGI 生产服务器
- `httpx.AsyncClient`：用于网关向后端多节点模型实例高效异步反向代理转发的客户端
- `pydantic.BaseModel`：定义并强校验推理 API 的请求与响应 JSON Schema 规范
- `transformers.AutoTokenizer`：独立的分词/逆分词服务，用于请求 Token 预估、计费与长度校验
- `prometheus_client`：暴露首 Token 延迟（TTFT）、每 Token 生成时延（TPOT）、吞吐量及 GPU 显存监控指标
- `opentelemetry.trace`：跨微服务全链路分布式追踪，记录请求生命周期内的精确耗时 Spans
- `kubectl`：在 Kubernetes 集群上编排、部署、扩缩容与运维 vLLM / SGLang 推理 Pod


## 生产级大模型推理服务系统的架构解构

在前面的章节中，我们深入掌握了从分布式训练（DDP, FSDP, DeepSpeed, Megatron-LM）到高性能分布式推理引擎（vLLM, SGLang）的全部核心技术。然而，**将模型加载进 GPU 显存并成功生成 Token，仅仅迈出了服务化的第一步**。

面对现实业务中瞬息万变、日请求量超千万的高并发在线流量，一套真正生产可用的大模型服务体系必须解决传统 Web 工程与 AI 物理算力碰撞带来的独特系统挑战：
1. **生成长度的极端动态性**：与输入输出大小固定的传统分类模型不同，LLM 生成长度事先不可知（可能输出 2 个 Token，也可能输出 2,000 个 Token），导致请求耗时相差百倍，传统的静态资源规划与线程池模型彻底失效；
2. **KV Cache 显存瓶颈与有状态调度**：随着请求并发增长与上下文延伸，KV Cache 显存消耗急剧膨胀，系统必须在显存不溢出（OOM）的前提下最大化吞吐，并针对公共前缀进行智能亲和性调度；
3. **流式低延迟响应（Streaming）**：用户期望以数十毫秒的首 Token 延迟（TTFT）即时看到打字机式的连续输出；
4. **服务治理与容灾**：需要具备模型多版本灰度（Canary Deployment）、A/B 测试、令牌桶限流（Rate Limiting）、过载反压（Backpressure）以及跨机故障秒级自愈。

![生产级大模型服务化系统标准分层架构](img/serving_architecture.png){#fig:serving-architecture .block width=80% align=center}

如 @fig:serving-architecture 所示，一个现代化的工业级大模型服务栈由以下核心中枢协同构成：

- **统一 API 网关（API Gateway）**：系统的全局单一入口，负责身份认证（Authentication）、令牌桶限流、跨模型智能路由、负载均衡以及金丝雀流量切分；
- **模型推理执行引擎（Model Runner）**：部署在专用 GPU 节点上的推理算力后端（基于 vLLM 或 SGLang），管理模型权重与 KV Cache 物理块，执行持续批处理（Continuous Batching）；
- **全栈可观测性中枢（Monitoring & Observability）**：集成 Prometheus 指标收集、OpenTelemetry 分布式全链路追踪与结构化日志，实时监控 TTFT、TPOT、KV Cache 命中率与 GPU 物理指标；
- **独立分词服务（Tokenizer Service，可选）**：轻量级 CPU 无状态微服务，用于在请求进入 GPU 前完成 Token 预估、计费鉴权与长度安全校验。

---

## 流量治理与智能路由中枢

在生产集群中，企业通常需要同时托管多种不同尺寸、不同专长（通用对话、代码生成、多模态）的模型，并频繁进行权重热更新与算法评测。

### 1. 多模型路由策略

- **特征匹配路由（Feature-based Routing）**：网关分析输入 Prompt 特征，代码类任务路由至专用 Code 模型，通用问答路由至标准对话模型；
- **A/B 测试一致性哈希路由（A/B Testing）**：使用用户唯一标识（User ID / Session ID）进行一致性哈希，确保同一用户在实验周期内始终稳定访问固定模型版本，保证评测因果有效性；
- **动态成本感知路由（Dynamic Model Selection）**：根据用户 SLA 等级与系统当前负载动态决策——简单请求派发给轻量模型（如 0.5B/1.5B），复杂深度推理请求派发给大模型（如 70B）。

### 2. 生产级负载均衡算法

- **轮询（Round-Robin）**：无状态平均分发，适合后端各实例规格与请求负载高度一致的场景；
- **最小活跃连接数（Least Connections）**：将请求派发给当前正在处理请求数最少的 Worker，天然平滑长短请求混杂带来的负载倾斜；
- **加权与缓存感知调度（Weighted / Cache-Aware）**：结合节点 GPU 算力规格（如 H100 vs A100）及前缀缓存（Radix Cache）命中情况进行打分调度。

### 3. 金丝雀灰度发布（Canary Deployment）与自动化熔断回滚

![金丝雀灰度发布流量渐进切分与自动化回滚机制](img/canary_deployment.png){#fig:canary-deployment .block width=80% align=center}

如 @fig:canary-deployment 所示，当上线新模型权重或升级推理引擎版本时：
1. **梯度流量切分**：按照 $10\% \rightarrow 25\% \rightarrow 50\% \rightarrow 100\%$ 的比例逐步将生产流量导入 Canary 金丝雀实例；
2. **指标比对与安全门禁**：实时对比稳定版（Stable）与金丝雀版（Canary）的 P99 延迟与 HTTP 5xx 错误率；
3. **自动化熔断回滚**：一旦金丝雀实例的错误率超过基线阈值（例如 $> 2\times$ 稳定版），控制器在秒级内**自动将金丝雀流量归零并发出告警**，实现对终端用户零感知的安全演进。

---

## 运维中枢：可观测性、高可用与成本控制

### 1. 全栈可观测性体系（Observability）

- **分布式全链路追踪（OpenTelemetry）**：生成覆盖 `gateway.receive` $\rightarrow$ `tokenizer.encode` $\rightarrow$ `model.inference` $\rightarrow$ `tokenizer.decode` 的完整调用链 Spans；
- **大模型核心指标监控（Prometheus & Grafana）**：
  - **首 Token 生成时间（TTFT, Time to First Token）**：衡量 Prefill 阶段与网关调度的响应敏捷度；
  - **每个输出 Token 平均时延（TPOT, Time Per Output Token）**：衡量 Decode 阶段每生成一个字耗费的毫秒数；
  - **KV Cache 命中率与显存利用率**：评估系统缓存复用与并发承载水位。

### 2. 高可用与过载反压（Reliability & Backpressure）

- **冷启动预热（Cold Start Warmup）**：GPU 实例拉起后，先由自动化探针发送若干模拟 Prompt 激活 CUDA Graph 与模型编译，预热完成后才向网关宣告 Ready 接受流量；
- **有界队列与快速失败反压（Backpressure）**：当所有 GPU 显存与请求槽位满载时，网关立即向超额请求返回 `HTTP 503 Service Unavailable / 429 Too Many Requests`，拒绝盲目排队导致的全局雪崩。

### 3. 极致成本优化（Cost Optimization）

- **竞价实例（Spot/Preemptible Instances）混合部署**：使用 50%–70% 超低成本的 Spot GPU 实例承载日常吞吐，配合按需（On-Demand）实例作为兜底，结合弹性伸缩节省 60%–80% 算力开销。

---

## 基于 Kubernetes 与 k3d 编排大模型服务 {#sec:k8s-deployment}

![](img/k8s_icon.png){.wrap align=top-right width=20%}

Kubernetes（K8s）已成为全球云原生大模型推理服务的事实标准底座。它提供了声明式配置、故障自动重启自愈、服务发现与弹性伸缩能力。

![Kubernetes 云原生大模型服务化体系核心组件映射](img/k8s.png){#fig:k8s-architecture .block width=90% align=center}

如 @fig:k8s-architecture 所示，Kubernetes 原生抽象完美契合大模型服务治理：
- **Ingress / Gateway API**：承载全局流量入口与跨模型路由；
- **Deployment & Pod**：基于 NVIDIA Device Plugin 请求 `nvidia.com/gpu: 1` 独占调度 GPU 物理卡，拉起包含健康检查（Liveness/Readiness Probes）的推理容器；
- **HPA / KEDA**：根据实时 QPS 与请求排队深度执行毫秒级 Pod 弹性伸缩。

### 1. 本地 k3d GPU 集群快速构建

使用 k3d（Docker 内运行的轻量级 Kubernetes）构建开发测试沙箱：

![k3d 本地轻量化 GPU Kubernetes 集群架构](img/k3d_architecture.png){#fig:k3d-architecture .block width=90% align=center}

```bash
cd code/k3d
./install-prerequisites.sh   # 安装 NVIDIA Container Toolkit 与 k3d
./build.sh                   # 构建支持 CUDA 的自定义 k3s 镜像
./create-cluster.sh          # 创建支持 GPU 直通的 Kubernetes 集群
```

通过 `kubectl describe nodes` 确认 `nvidia.com/gpu` 资源已成功就绪注册：

![`kubectl describe nodes` 成功识别物理 GPU 资源](img/k_desc_node.png){#fig:k-desc-node .wrap align=top-right}

### 2. 多模型路由实战（Multi-Model Routing）

![API Gateway 多模型反向代理与动态分发架构](img/multi_model_routing.png){#fig:multi-model-routing width=80%}

如 @fig:multi-model-routing 所示，客户端向统一网关发送不同 `model` 的请求，网关根据路由表分发到独立的 Kubernetes Service：

```yaml
# code/k3d/gateway/routing-config.yaml
routing:
  - model: "meta-llama/Llama-3.2-1B-Instruct"
    service_name: "vllm-llama-32-1b-service.multi-models.svc.cluster.local"
  - model: "microsoft/Phi-tiny-MoE-instruct"
    service_name: "vllm-phi-tiny-moe-service.multi-models.svc.cluster.local"
```

```bash
# 启动多模型集群
./manage-cluster-multi-models.sh start

# 发送请求至 Llama 模型
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "你好！"}]}'

# 发送请求至 Phi 模型（同一网关端口）
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "microsoft/Phi-tiny-MoE-instruct", "messages": [{"role": "user", "content": "你好！"}]}'
```

### 3. 多引擎路由实战（Multi-Engine Routing：vLLM vs SGLang）

![多引擎统一路由架构：支持客户端自由指定推理后端](img/multi_engine_routing.png){#fig:multi-engine-routing width=80%}

如 @fig:multi-engine-routing 所示，网关允许在请求中附加 `inference_server: "vllm"` 或 `"sglang"`，实现同模型在不同底层推理引擎间的无缝对比与平滑迁移。

---

## 生产级云原生推理框架：llm-d 深度探索

对于大规模工业部署，手工编写原始 Kubernetes YAML 维护成本极高。开源项目 **llm-d**（基于 vLLM、Envoy 与 NIXL 的云原生全栈大模型服务框架）提供了开箱即用的工业级解决方案[^llmd]。

[^llmd]: llm-d 生产级大模型 Kubernetes 服务化开源框架：https://github.com/llm-d/llm-d

### llm-d 核心技术突破

![llm-d 云原生生产级推理服务架构全景图](img/llmd_architecture.png){#fig:llmd-architecture width=80%}

如 @fig:llmd-architecture 所示，llm-d 集成了当今最先进的大模型工程实践：

1. **Inference Gateway（IGW 智能推理网关）**[^igw]：基于 Kubernetes Gateway API 标准，原生感知每个请求的 Prompt 长度与 Token 复杂度，预测耗时并实现 **前缀缓存感知路由（Prefix-Cache Aware Routing）**；
2. **PD 计算分离与 RDMA 直传（Prefill/Decode Disaggregation）**：Prefill Pod 算完后，通过 **NIXL 库借助 InfiniBand RDMA** 将千兆级 KV Cache 在毫秒内直传至 Decode Pod；
3. **分层分级分布式缓存（Distributed Prefix Caching）**：结合本地 Host RAM 与高速 NVMe SSD，实现跨 Pod 的全局 KV Cache 共享检索；
4. **变体容量感知弹性伸缩（Variant Autoscaling）**：根据当前活跃 Token 吞吐与生成延迟动态伸缩，而非单纯依赖 CPU/内存利用率。

[^igw]: Kubernetes Gateway API Inference Extension: https://github.com/kubernetes-sigs/gateway-api-inference-extension

![llm-d 多模型自动服务发现与智能分发架构](img/llmd_multi_model.png){#fig:llmd-multi-model width=80%}

### 手工 k3d 架构 vs llm-d 生产级框架全方位对比

::: {width=85%}

| 系统维度 | k3d 手工构建方案（适合学习研发） | llm-d 生产级框架（适合千万级业务） |
|:---|:---|:---|
| **配置与部署** | 手工编写 Service/Deployment YAML | Helm Charts 一键自动化声明式拉起 |
| **网关与路由** | 自定义 Python 代理网关 | 专为大模型设计的 Kubernetes 原生 IGW |
| **负载均衡** | 朴素轮询 / 最小连接数 | **前缀感知（Prefix-Cache Aware）智能调度** |
| **KV Cache 共享** | 单 Pod 独立隔离 | **支持基于 RDMA/NVMe 的跨 Pod 缓存直传共享** |
| **弹性伸缩** | 静态副本配置 | **基于 Token 吞吐与生成延迟的变体自动扩缩容** |
| **多模型管理** | 手工在网关路由表映射 | **基于 ModelService CRD 声明式自动发现** |

Table: k3d 手工方案与 llm-d 工业级生产框架核心对比 {#tbl:k3d-llmd-comparison}

:::

---

## 本章小结

本章构建了将大模型稳定推向生产环境的完整系统全貌：
- 剖析了生产级 LLM 服务系统的四大中枢（网关、推理引擎、可观测性、分词服务）；
- 掌握了特征匹配、A/B 测试一致性哈希以及金丝雀灰度发布的流量治理策略；
- 深入推导了基于 OpenTelemetry 与 Prometheus 构建全链路 TTFT、TPOT 与 KV 命中率监控的实战方法；
- 掌握了在 Kubernetes / k3d 上编排多模型与多引擎（vLLM vs SGLang）服务集群的标准范式；
- 探索了基于 **llm-d** 的先进云原生架构（智能 IGW、PD 计算分离与 RDMA 缓存直传）。

当全套服务基础设施部署上线后，我们如何精准量化系统的真实极限？如何系统性测量不同并发下的吞吐、延迟分布、首字耗时与多卡强扩展效率？在下一章中，我们将进入 **分布式性能基准测试与极限调优**，利用专业压测工具与底层 Profiler 压榨系统最后一滴算力性能。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
