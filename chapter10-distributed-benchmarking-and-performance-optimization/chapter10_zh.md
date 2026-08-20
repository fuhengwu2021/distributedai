# 第 10 章：分布式基准测试与系统性能调优 {-}

*精准度量与深度优化分布式 AI 系统的算力极限、通信瓶颈与吞吐延迟*

> 如果你无法度量它，你就无法改进它。  
> —— 彼得·德鲁克（Peter Drucker，现代管理学之父）

**核心代码速查**

- `torch.profiler`：PyTorch 官方内置的深度性能分析器，追踪 CPU/CUDA 算子耗时与显存波动
- `torch.utils.benchmark`：PyTorch 官方高精度微基准测试工具，提供自动热身与精确计时
- `nvidia-ml-py`：NVIDIA NVML Python 绑定库，实时采集 GPU 显存、核心利用率与功率温度
- `psutil`：底层系统与操作系统进程监控库，排查 CPU 瓶颈与主机内存泄漏
- `mlperf_logging`：MLPerf 官方合规性指标打点库，输出国际标准化 AI 训练/推理基准日志
- `tensorboard`：深度可视化训练指标、CUDA 执行时间线（Timeline）与内存分片剖析
- `wandb`：Weights & Biases 实验追踪云平台，记录多机多卡训练超参数与性能曲线
- `py-spy`：无需重启代码即可实时采样的低开销 Python 进程火焰图剖析器
- `nsys`：NVIDIA Nsight Systems，系统级（CUDA/NVLink/NCCL/OS）全栈底层性能追踪工具
- `ncu`：NVIDIA Nsight Compute，芯片 Kernel 级指令流、Tensor Core 利用率与显存带宽深度分析器


## 分布式系统中的性能黑洞

在前面的章节中，我们完整构建了现代分布式 AI 系统的每一个模块：从 DDP、FSDP、DeepSpeed、Megatron-LM 到 vLLM、SGLang、SLURM 编排以及云原生推理网关。至此，代码在集群上已经能够**完整跑通（Working）**。

然而在工业生产中，**“能跑通”与“高效运行”之间隔着一道巨大的鸿沟**：
- 一个耗时两周完成的 70B 模型训练任务，如果平均 GPU 利用率（GPU Utilization）只有 45%，意味着团队为价值上百万元的算力集群买单，却有一大半算力在空转等待；
- 一个在线部署的推理服务虽然能返回答案，但其 P99 尾部延迟高达 800ms，频繁导致下游业务超时熔断；
- 在 64 张 GPU 上训练时，扩展效率（Scaling Efficiency）暴跌至 52%，实际有效算力仅仅相当于 33 张卡。

本章将系统性揭开这些隐藏在分布式系统深处的“性能黑洞”。我们将全面建立分布式 AI 领域的科学评测方法论：涵盖训练与推理的核心评价指标、PyTorch Profiler 与 Nsight 底层追踪、多卡精度一致性校验、跨机网络通信压测以及阿姆达尔定律扩展瓶颈分析。

---

## 科学基准测试的方法论基石

分布式测试绝非“在 Python 循环外加一个 `time.time()`”那么简单。由于 GPU 异步执行、动态 JIT 编译以及网络通信抖动，不严谨的测试往往会得出完全错误的结论。

### 1. 为什么 P95 / P99 分位数比平均值（Average）重要百倍？

>NOTES: **平均值的谎言与长尾延迟陷阱**
>
> 假设系统 A 所有请求耗时均稳定在 100ms；系统 B 平均耗时 80ms，但有 5% 的长尾请求耗时超过 500ms。  
> 尽管系统 B 的“平均值”看起来更优，但在真实业务中，这 5% 的慢请求会严重阻塞并发流水线、导致用户流失或触发网关超时。在 SLA 约束下，**P95 与 P99 延迟分位数才是衡量系统稳定性的唯一金标准**。
>NOTEE

### 2. 核心效率度量指标

- **扩展效率（Scaling Efficiency）**：
  $$
  \text{Scaling Efficiency} = \frac{\text{Throughput}_N}{N \times \text{Throughput}_1}
  $$
  如果 8 张 GPU 的吞吐仅为单卡的 6.5 倍，则扩展效率为 $6.5 / 8 = 81.25\%$，意味着近 20% 的硬件投资被进程通信与同步开销吞噬。
- **显存碎片率（Memory Fragmentation）**：可用显存总量充足，但由于缺乏大块连续地址空间而触发 OOM。
- **每 Token 成本（Cost per Token）**：结合物理云实例租金与有效吞吐，量化每生成 100 万 Token 的真实美元成本。

### 3. 三大铁律：消除测量污染

1. **显式热身（Warmup）**：GPU 首次执行算子会触发 CUDA Context 初始化、JIT 编译与内存池分配。测试前必须先执行 10–50 次热身迭代；
2. **GPU 异步屏障同步（Synchronization）**：CUDA 算子是异步发射到流（Stream）中的。计时开始与结束前**必须调用 `torch.cuda.synchronize()`** 强制等待物理计算完毕；
3. **统计学多轮采样**：至少运行 100 轮以上，剔除极值并输出均值、标准差与各分位数。

---

## 分布式训练基准测试与性能剖析

分布式训练不仅包含前向计算（Forward）与反向传播（Backward），还深度交织着跨机梯度同步（AllReduce/Reduce-Scatter）与数据加载（DataLoader I/O）。

![分布式训练单步迭代耗时分解与随 GPU 卡数扩充的变化趋势](img/training_breakdown.png)

如上图所示，随着 GPU 数量由 1 卡增加至 16 卡：虽然单步绝对时间从 155ms 降至 33ms，但**通信同步耗时占比从 0% 激增至 55%**！超过一半的时间被跨机网络等待消耗，这就是扩展效率下降的根本原因。

### 1. PyTorch Profiler 实战

通过 `torch.profiler.profile` 细粒度捕获 Host CPU 与 Device GPU 的每一个算子：

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(5):
        with record_function("forward_pass"):
            out = model(inputs)
        with record_function("backward_pass"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()

# 导出 Chrome Tracing 轨迹并在 chrome://tracing 或 Perfetto 中可视化
if dist.get_rank() == 0:
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 2. NVIDIA Nsight Systems 系统级底层分析

当需要深入诊断 NCCL 环路通信、CUDA Kernel 启动开销与操作系统调度时，使用 `nsys`：
```bash
nsys profile --trace=cuda,nvtx,osrt --output=profile.nsys-rep python train.py
```

### 3. 多卡扩展效率与网络拓扑诊断

![多卡扩展效率评估：理想线性加速 vs 实际性能曲线](img/scaling_efficiency.png)

![跨节点网络带宽层级与 Ring AllReduce 通信拓扑](img/network_topology.png)

- **机内核间通信（NVLink）**：单向带宽高达 450–900 GB/s；
- **跨机跨节点通信（InfiniBand / RoCE）**：单向带宽约为 25–50 GB/s（200–400 Gbps）；
- **标准以太网（Ethernet）**：仅约 1.25–12.5 GB/s（10–100 Gbps）。

若跨机网络带宽不足，集合通信将瞬间成为全集群最大的木桶短板。

>IMPORS: **多节点集群性能优化排查 SOP**
>
> 1. **排查通信 vs 计算比例**：使用 PyTorch Profiler 确认 `nccl:all_reduce` 耗时是否超过 30%；
> 2. **优化梯度分桶（Bucket Size）**：合理调整 DDP 的 `bucket_cap_mb`（通常设置为 25MB–50MB），让上一层梯度的 AllReduce 与下一层的反向求导最大化重叠（Overlap）；
> 3. **排查 DataLoader 瓶颈**：确认 Host 端 CPU 预处理与磁盘读取速度，设置 `num_workers = 2~4 * GPU数` 并开启 `pin_memory=True`；
> 4. **精度一致性双重校验**：在优化性能的同时，定期对比分布式训练与单卡 Baseline 的测试集 Loss 与收敛精度，严防精度静默劣化。
>IMPORE

---

## 大模型在线推理基准测试（LLM Serving Benchmarks）

与训练不同，大模型自回归推理具有显著的阶段非对称性（Compute-bound Prefill vs Memory-bound Decode）与用户主观交互感知要求。

![大语言模型推理核心性能指标全景关系图](img/inference_metrics_overview.png)

### 1. 推理核心度量全景

![每秒生成 Token 吞吐量（TPS）时间线分布](img/tps_timeline.png)

- **首 Token 生成时间（TTFT, Time to First Token）**：从发起请求到客户端收到第一个文字的耗时，包含分词、全量 Prompt 预填充（Prefill）与首字生成；
  ![TTFT 首字延迟全链路时序拆解](img/ttft_pipeline.png)

- **Token 生成间隔时间（ITL / TPOT, Time Per Output Token）**：自回归解码（Decode）阶段生成连续字符的平均间隔，决定了客户端“打字机吐字速度”：
  $$
  \text{ITL} = \frac{\text{E2E\_Latency} - \text{TTFT}}{\text{Total\_Output\_Tokens} - 1}
  $$
  ![ITL 连续字符输出间隔时间流水线](img/itl_pipeline.png)

- **端到端总时延（E2E Latency）**：
  $$
  \text{E2E\_Latency} = \text{TTFT} + (\text{Output\_Tokens} - 1) \times \text{ITL}
  $$
  ![端到端总延迟 E2E 完整处理时序](img/e2e_latency_pipeline.png)

![推理时延直方图（Histogram）与累积分布函数（CDF）](img/latency_distribution.png){#fig:latency-distribution}

### 2. 专业压测框架：genai-bench

使用 SGLang 官方维护的工业级压测工具 `genai-bench` 进行真实流量分布压测：

```bash
pip install genai-bench

# 压测指定并发下的 Prompt 与 Output 长度分布
genai-bench \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --base-url http://localhost:8000/v1 \
  --dataset-name sharegpt \
  --concurrency 16 \
  --num-prompts 500
```

---

## 性能与精度的双轮驱动验证

**脱离精度的性能优化毫无价值**。在执行以下深度优化时，必须同步运行标准化精度评测集（MMLU, HumanEval, HELM）：
1. **模型权重量化（FP8 / INT8 / INT4）**：在享受 2–3 倍吞吐提升与显存减半的同时，严格监控下游任务准确率波动；
2. **长文本前缀缓存（Radix Caching / KV Eviction）**：验证 KV Cache 动态裁剪与驱逐策略是否破坏长文档检索的召回准确率；
3. **推测解码（Speculative Decoding）**：验证小草稿模型与大验证模型的协同输出是否与原始大模型贪婪解码严格等价。

---

## 本章小结

本章为分布式 AI 研发与运维人员构建了完整的量化调优方法论：
- 确立了以 P95/P99 尾部延迟、扩展效率与 GPU 利用率为核心的度量体系；
- 掌握了基于 PyTorch Profiler 与 Nsight Systems 诊断 CPU 阻塞、通信等待与显存碎片的技能；
- 深入推导了 LLM 推理的 TTFT、ITL/TPOT、E2E 与 TPS 之间的数学关系；
- 掌握了运用 `genai-bench` 模拟真实世界高并发长短文本流量的方法；
- 建立了“性能基准压测”与“精度一致性验证”并行的严密工程防线。

在掌握了分布式训练、推理、集群运维以及性能调优的全部核心工程体系之后，分布式 AI 的未来又将走向何方？在全书的最后一章中，我们将展望 **前沿趋势与下一代分布式系统设计**，探索 MoE 稀疏扩展、端云异构协同以及具身智能分布式架构的无限可能。

<!-- include: exercises/torch_zh.md if include_math -->
<!-- include: exercises/torch_zh.md if include_torch -->
