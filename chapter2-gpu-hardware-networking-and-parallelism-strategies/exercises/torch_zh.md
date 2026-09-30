\fancydividerwithicon[center]{hand.png}


## 实战演练


### GPU 硬件信息探测器

实现一个完备的 GPU 硬件信息探测函数，收集并格式化输出系统中所有可用 GPU 的关键硬件参数。

__实战要求：__

- 函数签名：
```python
inspect_gpu_hardware()
```

- 为系统中每块 GPU 返回包含以下键的字典列表：
  - `name`: GPU 型号名称
  - `memory_total_gb`: 显存总量（单位：GB）
  - `memory_free_gb`: 空闲显存容量（单位：GB）
  - `compute_capability`: 计算能力字符串（例如 "9.0"）
  - `multiprocessor_count`: 流式多处理器（SM）核心数量
  - `cuda_version`: 当前 CUDA 运行时版本
- 正确处理系统中无可用 CUDA 设备的情况（返回空列表）
- 打印格式整齐的汇总表格，展示各项 GPU 硬件指标

__测试验证：__
```python
import torch

gpu_info = inspect_gpu_hardware()
print(f"检测到 {len(gpu_info)} 块 GPU")

for i, info in enumerate(gpu_info):
    print(f"\nGPU {i}:")
    print(f"  型号: {info['name']}")
    print(f"  显存: 总量 {info['memory_total_gb']:.1f} GB, "
          f"空闲 {info['memory_free_gb']:.1f} GB")
    print(f"  计算能力: {info['compute_capability']}")
    print(f"  SM 处理器数: {info['multiprocessor_count']}")
```

### 训练显存开销核算器

实现一个显存计算函数，精确预估不同精度与优化器配置下模型训练阶段的静态与动态显存占用。

__实战要求：__

- 函数签名：
```python
calculate_training_memory(num_params, precision='bf16', optimizer='adam', batch_size=1, seq_length=2048)
```

- 分别核算以下显存分量：
  - 模型参数权重（Parameters）
  - 梯度张量（Gradients，与参数规模一致）
  - 优化器状态（Optimizer States）：
    - Adam: 参数大小的 2 倍（一阶动量与二阶动量）
    - SGD: 参数大小的 1 倍（仅动量）
  - 激活值（Activations）：近似按 `batch_size × seq_length × hidden_size × num_layers × bytes_per_element` 估算
- 支持的精度选项：'fp32'（每元素 4 字节）、'bf16'/'fp16'（每元素 2 字节）、'int8'（每元素 1 字节）
- 返回细分字典：`{'parameters': ..., 'gradients': ..., 'optimizer': ..., 'activations': ..., 'total': ...}`
- 所有数值统一折算为 GB

__测试验证：__
```python
# 针对 70B 参数量模型，采用 BF16 精度与 Adam 优化器
memory = calculate_training_memory(
    num_params=70e9,
    precision='bf16',
    optimizer='adam',
    batch_size=4,
    seq_length=2048
)

print("训练显存开销拆解预估 (GB):")
for key, value in memory.items():
    print(f"  {key}: {value:.2f} GB")
```

### 拓扑探测与互连分析

实现一个解析 `nvidia-smi topo -m` 命令输出的函数，用于识别多卡拓扑结构与互连带宽特性。

__实战要求：__

- 函数签名：
```python
analyze_gpu_topology()
```

- 利用 `subprocess` 调用 `nvidia-smi topo -m` 并捕获标准输出
- 解析拓扑矩阵，辨识以下互连类型：
  - 通过 NVLink 直连的 GPU 间通道（识别 NV18、NV12、NV4 等标记）
  - 仅通过 PCIe 互连的 GPU 间通道（识别 PIX、PXB 等标记）
  - 全连接拓扑判定（判断是否任意两块 GPU 间均具备 NVLink 直连通道）
- 返回包含以下字段的分析字典：
  - `num_gpus`: GPU 总数
  - `nvlink_pairs`: 具备 NVLink 直连的 GPU 卡号对列表
  - `pcie_only_pairs`: 仅能通过 PCIe 互连的 GPU 卡号对列表
  - `has_all_to_all_nvlink`: 布尔值，指示是否具备全互联 NVLink 网格拓扑
  - `recommended_parallelism`: 基于拓扑特征推荐的最佳并行策略建议

__测试验证：__
```python
import subprocess

topology = analyze_gpu_topology()
print(f"GPU 总数: {topology['num_gpus']}")
print(f"NVLink 直连卡对: {topology['nvlink_pairs']}")
print(f"仅 PCIe 互连卡对: {topology['pcie_only_pairs']}")
print(f"是否具备全互联 NVLink: {topology['has_all_to_all_nvlink']}")
print(f"推荐并行策略: {topology['recommended_parallelism']}")
```

### 并行策略决策选择器

根据模型参数量、物理硬件拓扑以及训练/推理任务属性，实现一个自动推荐最适并行策略的决策函数。

__实战要求：__

- 函数签名：
```python
recommend_parallelism_strategy(model_size_gb, num_gpus, topology_info, has_nvlink_all_to_all, training_type='training')
```

- 综合权衡考量：
  - 模型显存需求与单卡物理显存容量对比
  - 硬件互连拓扑（是否具备高带宽 NVLink 全互联，或是受限的 PCIe 通道）
  - 任务类型特征（分布式训练阶段的状态同步需求 vs. 在线推理阶段的延迟吞吐平衡）
- 返回包含以下字段的建议字典：
  - `primary_strategy`: 核心并行方案（如 DDP、FSDP、TP、PP 或混合并行）
  - `reasoning`: 方案决策背后的关键工程逻辑与原理分析
  - `alternative_strategies`: 备选可行方案列表
  - `estimated_gpu_count`: 满足运行的最低 GPU 卡数预估
  - `memory_per_gpu_gb`: 单卡预估显存负载

__测试验证：__
```python
# 70B 参数量模型（BF16 精度下权重占 140 GB），运行于配备全互联 NVLink 的 8 卡集群
strategy = recommend_parallelism_strategy(
    model_size_gb=140,
    num_gpus=8,
    topology_info={'has_all_to_all_nvlink': True},
    has_nvlink_all_to_all=True,
    training_type='training'
)

print(f"推荐核心策略: {strategy['primary_strategy']}")
print(f"决策逻辑: {strategy['reasoning']}")
print(f"最少所需 GPU 卡数: {strategy['estimated_gpu_count']}")
print(f"单卡预估显存占用: {strategy['memory_per_gpu_gb']:.1f} GB")
```


## 预期学习目标

完成本章实战练习后，你将能够：

- 使用 PyTorch 与底层 CUDA API 编写脚本，自动化探查 GPU 计算规格与显存状态
- 熟练核算不同精度（FP32/BF16/FP16/INT8）与优化器状态下的训练显存开销
- 解析并理解 `nvidia-smi` 拓扑矩阵输出，准确识别系统内的互连拓扑瓶颈
- 区分 NVLink 与 PCIe 互连的带宽特征及其对不同集合通信模式的影响
- 结合模型规模与物理硬件拓扑，为实际工作负载科学选择数据并行、张量并行与流水线并行策略
- 深刻理解 GPU 互连物理拓扑如何直接制约分布式并行算法的执行效能
