\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 单项选择题

__1. 在分布式性能基准测试中，进行热身（Warmup）迭代的核心目的是什么？__

A) 增加基准测试的总运行时间  
B) 消除 JIT 动态编译、CUDA Graph 捕获与显存缓存分配等冷启动干扰  
C) 降低 GPU 显存占用  
D) 测试异常处理代码分支  

__2. 测量首 Token 生成时间（TTFT, Time to First Token）时，包含以下哪些执行阶段？__

A) 仅包含模型前向传播计算  
B) 包含文本 Tokenizer 分词、全量 Prompt 预填充（Prefill）计算与首个 Token 生成  
C) 包含生成全量 Token 直至结束符的全部时间  
D) 仅包含网络传输往返延迟  

__3. 分布式训练作业在 8 张 GPU 扩展下的“扩展效率（Scaling Efficiency）”为 85%，这意味着什么？__

A) 85% 的 GPU 显存被成功利用  
B) 相比单卡实现了理论线性加速比的 85%（即实际加速比为 $8 \times 0.85 = 6.8\times$，而非理想的 $8\times$）  
C) 85% 的时间花费在纯算力计算上  
D) 15% 的梯度在传输过程中被丢弃  

__4. 在流式交互大模型应用中，哪项指标最能直接反映用户感知到的“即时响应灵敏度”？__

A) 每秒总 Token 数（Tokens Per Second, TPS）  
B) 端到端总时延（End-to-End Latency）  
C) 首 Token 生成时间（Time to First Token, TTFT）  
D) GPU 物理利用率（GPU Utilization）  

__5. 在 PyTorch Profiler 的追踪结果中，以下哪组算子直接反映了跨卡通信开销？__

A) `aten::matmul` 和 `aten::linear`  
B) `nccl:all_reduce` 和 `nccl:all_gather`  
C) `aten::relu` 和 `aten::gelu`  
D) `cuda::memcpy` 仅内存拷贝  

__6. 阿姆达尔定律（Amdahl's Law）对分布式系统的扩展极限做出了什么核心预测？__

A) 只要不断增加 GPU 卡数，系统总能维持完美的线性加速  
B) 程序中无法并行的串行执行部分，决定了系统所能达到的理论最大加速比上限  
C) 跨卡通信开销会随 GPU 卡数的增加而自然减小  
D) 显存占用随 GPU 数量呈纯线性增长  

__7. 在评测在线推理吞吐时，为什么必须在不同的请求并发度（Concurrency Levels）下进行梯度压测？__

A) 为了找出最大静态 Batch Size  
B) 为了找出吞吐量达到饱和的临界点（Saturation Point），识别显存带宽与算力瓶颈  
C) 为了测量冷启动加载耗时  
D) 为了测试模型文本准确率  

__8. 在 PyTorch DDP 训练中，以下哪种方式能够精准隔离并度量梯度同步通信耗时？__

A) 测量总训练耗时并直接除以 GPU 卡数  
B) 在梯度同步前后插入 `torch.cuda.synchronize()` 阻断屏障并进行精确打点计时  
C) 计算模型包含的总参数量  
D) 仅测量前向传播（Forward）的执行耗时  

__9. Token 生成间隔时间（ITL, Inter-token Latency）与单输出 Token 生成时延（TPOT, Time Per Output Token）的关系是什么？__

A) ITL 仅测量首字生成，TPOT 测量后续字符生成  
B) 两者在物理本质上等价，均表示自回归生成阶段连续两个 Token 之间的平均输出间隔时间  
C) ITL 专用于训练评估，TPOT 专用于推理评估  
D) TPOT 包含网络传输，ITL 不包含网络传输  

__10. 在对比单卡与分布式多卡训练的模型精度时，出现以下哪种现象说明系统存在异常？__

A) 单卡与多卡的 Loss 收敛曲线完全吻合  
B) 验证集精度出现显著下降且明显超出正常的随机浮动方差（Statistical Significance）  
C) 多卡训练收敛速度更快  
D) 多卡单 GPU 显存占用更低  

__11. 在生产级大模型在线推理 SLA 规范中，哪项分位数指标对保障服务质量最关键？__

A) P50（中位数）时延  
B) P95 或 P99 长尾延迟分位数（Tail Latency）  
C) 全局平均延迟（Mean Latency）  
D) 最好情况下的最小延迟（Min Latency）  

__12. 计算多卡训练扩展效率（Scaling Efficiency）的正确数学公式是什么？__

A) $\text{Efficiency} = \text{Throughput}_N / \text{Throughput}_1$  
B) $\text{Efficiency} = (\text{Throughput}_N \times N) / \text{Throughput}_1$  
C) $\text{Efficiency} = \text{Throughput}_N / (\text{Throughput}_1 \times N)$  
D) $\text{Efficiency} = N / \text{Throughput}_N$  

__13. 在诊断分布式 GPU 互联拓扑时，执行命令 `nvidia-smi topo -m` 显示的核心信息是什么？__

A) GPU 实时显存分配状态  
B) 各 GPU 芯片之间的物理互联拓扑矩阵（NVLink、NVSwitch、PCIe 通道）  
C) 外部网络带宽吞吐量  
D) CUDA Kernel 算子执行耗时  

__14. 为什么严肃的基准评测必须多次重复运行并进行统计学方差分析？__

A) 仅仅为了拉长测试脚本的运行时间  
B) 为了消除温控降频（Thermal Throttling）、网络抖动与后台竞争导致的偶然误差，确保测试结果具备可复现性与统计显著性  
C) 为了给 GPU 充分预热  
D) 为了测试不同的模型架构配置  

---

### 简答与深度思考题

15. **请从通信量与网络拓扑的角度，深入解释为什么在分布式训练中随着 GPU 节点数量增加，通信开销占比通常会显著上升？**  
16. **某推理集群压测显示 P50 延迟为 50ms，但 P99 延迟高达 500ms（高达 10 倍差距）。这种长尾分布通常暴露出系统的哪些潜在工程隐患？**  
17. **在不修改模型网络结构的前提下，列举至少两种能够有效降低多节点分布式训练通信开销的工程优化手段。**  
18. **为什么针对同一个模型权重进行 INT8 量化后，在不同的推理引擎（如 vLLM 与 SGLang）上运行可能会观察到微小的输出精度与答案差异？**  
19. **请阐述分布式训练基准测试中的“Samples Per Second（每秒样本数）”与在线推理基准测试中的“Tokens Per Second（每秒 Token 数）”的根本区别。**  

---

## 参考答案与解析

1. **B** —— 热身旨在消除动态 JIT 编译、权重载入与显存池建立带来的冷启动测量污染。  
2. **B** —— TTFT 完整覆盖从输入分词、全序列 Prompt 编码（Prefill）到产出首个 Token 的全过程。  
3. **B** —— 85% 扩展效率代表取得了理想 $8\times$ 理论线性加速比的 85%（即 $6.8\times$）。  
4. **C** —— 首 Token 延迟（TTFT）决定了用户等待模型开始吐字的主观响应体验。  
5. **B** —— `nccl:all_reduce` 与 `nccl:all_gather` 是 NCCL 集合通信的核心底层算子。  
6. **B** —— 阿姆达尔定律表明系统中不可并行的串行比例从根本上限制了最大理论加速比。  
7. **B** —— 递增并发压测能够精准测出 GPU 计算核心与显存带宽饱和的吞吐拐点。  
8. **B** —— 在通信前后插入显式 `torch.cuda.synchronize()` 阻断可以精确捕获真实的 GPU 同步时间。  
9. **B** —— ITL 与 TPOT 在物理上均衡量自回归解码阶段连续输出 Token 间的间隔时间。  
10. **B** —— 超过正常统计波动的精度显著跌落，通常暗示梯度累加、数值精度溢出或进程间数据分发存在 Bug。  
11. **B** —— P95/P99 衡量极端长尾请求的耗时，是 SLA 违约风险的核心监测指标。  
12. **C** —— $\text{Scaling Efficiency} = \text{Throughput}_N / (\text{Throughput}_1 \times N)$。  
13. **B** —— 展示 GPU 间的物理链路关系（如 NVLink 直连 NV12 还是跨 NUMA 节点的 PCIe 通信）。  
14. **B** —— 统计学多次采样能够平滑硬件抖动与瞬时网络拥塞，确保数据科学可信。  

__简答题要点解析：__

15. **解析要点**：随着 GPU 卡数扩充，跨机跨节点通信不可避免地从机内高带宽 NVLink（900 GB/s）降级为跨机低带宽 InfiniBand（200–400 Gbps）或以太网；同时参与 Ring/Tree AllReduce 的节点增多，集合通信的环路延迟与同步开销累加。  
16. **解析要点**：高达 10 倍的 P99 差距暴露出严重的长尾延迟（Tail Latency），通常由以下原因引起：长 Prompt 请求阻塞短请求、KV Cache 显存不足触发缺页/换出（Swap）、Python GC 垃圾回收瞬时停顿、或个别慢节点（Straggler）拖慢全局。  
17. **解析要点**：（1）**梯度累加（Gradient Accumulation）**：聚合多个 Micro-batch 梯度后再执行单次通信；（2）**计算与通信重叠（Bucket Overlapping）**：在反向传播计算上一层导数的同时异步触发下一层梯度的 Reduce-Scatter。  
18. **解析要点**：不同推理引擎采用的底层 CUDA INT8 GEMM 矩阵算子实现（如 CUTLASS、Triton 或 TensorRT-LLM 专用 Kernel）、激活值反量化截断策略以及浮点累加顺序存在细微数学差异。  
19. **解析要点**：训练端 Samples/sec 衡量固定 Batch 下整条样本的端到端吞吐；推理端 Tokens/sec 衡量自回归解码逐字生成的物理速率，其受动态输入输出长度、KV Cache 命中率与并发 Batch 状态动态影响。  


## 学习成果自测

完成本章所有练习后，你应当能够：

- 建立科学、严谨的分布式基准性能评测方法论（Warmup、统计学方差分析、同步屏障隔离）
- 熟练解读与区分大模型核心性能指标（TTFT, ITL/TPOT, TPS, Scaling Efficiency）
- 运用 PyTorch Profiler 与 NVIDIA Nsight Systems 精准定位算力、显存与通信瓶颈
- 运用阿姆达尔定律与强/弱扩展模型评估万卡集群的实际算力利用率（MFU）
- 建立性能优化与模型精度质量（MMLU, HumanEval, HELM）协同验证的双轮驱动机制
