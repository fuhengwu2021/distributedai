\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. GPU 硬件信息探测器

实现一个全面的 GPU 硬件信息探测函数，自动提取当前环境可用 GPU 的详细硬件指标。

__要求：__

- 函数签名：
```python
inspect_gpu_hardware()
```
- 为每张 GPU 返回一个包含如下键的字典：
  - `name`：GPU 型号名称
  - `memory_total_gb`：总显存（单位：GB）
  - `memory_free_gb`：当前可用空闲显存（单位：GB）
  - `compute_capability`：计算能力字符串（如 "9.0"）
  - `multiprocessor_count`：流式多处理器（SM）数量
  - `cuda_version`：CUDA 版本
- 处理 CUDA 不可用的异常情况（返回空列表）
- 格式化打印包含所有 GPU 硬件信息的汇总表格

__测试代码：__
```python
import torch

gpu_info = inspect_gpu_hardware()
print(f"检测到 {len(gpu_info)} 张可用 GPU")

for i, info in enumerate(gpu_info):
    print(f"\nGPU {i}:")
    print(f"  型号: {info['name']}")
    print(f"  显存: 总计 {info['memory_total_gb']:.1f} GB, "
          f"空闲 {info['memory_free_gb']:.1f} GB")
    print(f"  计算能力: {info['compute_capability']}")
    print(f"  SM 处理器数量: {info['multiprocessor_count']}")
```

### 2. 训练显存开销计算器

实现一个根据模型参数量、数值精度以及优化器配置，精确推导训练峰值显存需求的计算函数。

__要求：__

- 函数签名：
```python
calculate_training_memory(num_params, precision='bf16', optimizer='adam', batch_size=1, seq_length=2048)
```
- 计算以下各部分的显存占用：
  - 模型参数权重（Weights）
  - 梯度（Gradients，与参数尺寸一致）
  - 优化器内部状态（Optimizer States）：
    - Adam：参数量的 2 倍（一阶矩动量 + 二阶矩方差）
    - SGD：参数量的 1 倍（仅动量）
  - 中间激活值（Activations）：估算为 `batch_size × seq_length × hidden_size × num_layers × bytes_per_element`
- 支持的精度选项：'fp32' (4 字节), 'bf16'/'fp16' (2 字节), 'int8' (1 字节)
- 返回包含显存分解的字典：`{'parameters': ..., 'gradients': ..., 'optimizer': ..., 'activations': ..., 'total': ...}`（单位均为 GB）

__测试代码：__
```python
# 70B 参数模型在 BF16 精度、Adam 优化器下的显存估算
memory = calculate_training_memory(
    num_params=70e9,
    precision='bf16',
    optimizer='adam',
    batch_size=4,
    seq_length=2048
)

print("显存需求详细分解 (GB):")
for key, value in memory.items():
    print(f"  {key}: {value:.2f} GB")
```

### 3. GPU 互联拓扑分析器

实现一个自动解析 `nvidia-smi topo -m` 命令输出并推断系统 GPU 互联拓扑结构的函数。

__要求：__

- 函数签名：
```python
analyze_gpu_topology()
```
- 使用 `subprocess` 运行 `nvidia-smi topo -m` 并捕获标准输出
- 解析拓扑矩阵，提取识别：
  - 通过 NVLink 直连的 GPU 节点对（识别 NV18, NV12, NV4 等标记）
  - 仅通过 PCIe 连接的 GPU 节点对（识别 PIX, PXB 等标记）
  - 是否具备全互联 NVLink（All-to-All NVLink，即所有卡间均可通过 NVLink 全速通信）
- 返回包含如下字段的字典：
  - `num_gpus`：GPU 总数量
  - `nvlink_pairs`：具备 NVLink 互联的 GPU 编号对列表
  - `pcie_only_pairs`：仅通过 PCIe 互联的 GPU 编号对列表
  - `has_all_to_all_nvlink`：指示是否支持全互联 NVLink 的布尔值
  - `recommended_parallelism`：基于当前硬件拓扑给出的推荐并行策略

__测试代码：__
```python
import subprocess

topology = analyze_gpu_topology()
print(f"GPU 数量: {topology['num_gpus']}")
print(f"NVLink 直连对: {topology['nvlink_pairs']}")
print(f"仅 PCIe 连接对: {topology['pcie_only_pairs']}")
print(f"全互联 NVLink 支持: {topology['has_all_to_all_nvlink']}")
print(f"推荐并行策略: {topology['recommended_parallelism']}")
```

### 4. 分布式并行策略智能推荐器

实现一个根据模型规模、硬件拓扑结构以及任务场景，自动给出最优并行策略推荐的决策函数。

__要求：__

- 函数签名：
```python
recommend_parallelism_strategy(model_size_gb, num_gpus, topology_info, has_nvlink_all_to_all, training_type='training')
```
- 综合评估：
  - 模型体积与单卡显存容量的对比
  - 硬件互联拓扑（NVLink 全互联 vs 仅 PCIe）
  - 训练场景与推理场景的差异化约束
- 返回包含如下字段的字典：
  - `primary_strategy`：推荐的主选并行策略（DDP, FSDP, TP, PP 或 混合并行）
  - `reasoning`：该选型决策的核心技术依据与权衡分析
  - `alternative_strategies`：备选可行方案列表
  - `estimated_gpu_count`：推荐的最少 GPU 数量
  - `memory_per_gpu_gb`：预估每张 GPU 的显存开销

__测试代码：__
```python
# 70B 模型 (BF16 需 140 GB 权重) 在 8 卡全互联 NVLink 集群上的训练推荐
strategy = recommend_parallelism_strategy(
    model_size_gb=140,
    num_gpus=8,
    topology_info={'has_all_to_all_nvlink': True},
    has_nvlink_all_to_all=True,
    training_type='training'
)

print(f"推荐策略: {strategy['primary_strategy']}")
print(f"选型依据: {strategy['reasoning']}")
print(f"预估所需 GPU 卡数: {strategy['estimated_gpu_count']}")
print(f"单卡预估显存占用: {strategy['memory_per_gpu_gb']:.1f} GB")
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 熟练通过 PyTorch 与底层 CUDA API 编程探测 GPU 硬件各项核心属性
- 准确推导大模型在不同数值精度与优化器下的显存开销组成
- 熟练解析 `nvidia-smi topo -m` 拓扑矩阵并诊断跨卡通信链路
- 准确区分 NVLink、NVSwitch、PCIe 以及跨节点 InfiniBand/RoCE 互联差异
- 根据模型参数规模与集群互联拓扑，科学决策最优分布式并行策略（DDP/FSDP/TP/PP/EP）
- 深刻理解硬件网络带宽瓶颈对各并行策略扩展效率的制约机理
