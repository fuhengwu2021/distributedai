\fancydividerwithicon[center]{hand.png}


## 实战演练


### 对比 DeepSpeed ZeRO 各阶段特性

基于统一模型基准，实现对比 DeepSpeed ZeRO Stage 1、Stage 2 与 Stage 3 的端到端评测流水线。

__实战要求：__

- 选取 7B 参数量模型（或根据当前测试环境显存容量选择合适体量的网络）
- 配置评测 ZeRO Stage 1（仅分片优化器状态）
- 配置评测 ZeRO Stage 2（分片优化器状态 + 梯度张量）
- 配置评测 ZeRO Stage 3（深度分片优化器状态、梯度与模型参数）
- 针对各阶段度量以下指标：
  - 每块 GPU 上的峰值物理显存占用
  - 训练吞吐率（tokens/second）
  - 网络通信耗时与开销占比
- 生成直观的显存开销拆解对比表

__测试验证：__
```python
import deepspeed
import torch

def benchmark_zero_stage(model, stage: int, num_steps: int = 100):
    """评测特定 ZeRO 阶段的显存占用与吞吐表现。"""
    ds_config = {
        "train_batch_size": 32,
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "bf16": {"enabled": True},
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    
    # 执行基准压测
    memory, throughput = run_training(model_engine, num_steps)
    return memory, throughput

# 依次对比 Stage 1, 2, 3
results = {}
for stage in [1, 2, 3]:
    model = create_model()  # 为每个阶段重新初始化独立模型副本
    memory, throughput = benchmark_zero_stage(model, stage)
    results[f"ZeRO-{stage}"] = {"memory_gb": memory, "throughput": throughput}

print("优化阶段   | 峰值显存 (GB) | 训练吞吐 (tok/s)")
for name, metrics in results.items():
    print(f"{name:9} | {metrics['memory_gb']:13.1f} | {metrics['throughput']:12.1f}")
```

### 实现主机内存卸载（CPU Offloading）

配置并评测 DeepSpeed ZeRO-Offload 机制，突破 GPU 物理显存限制训练超大规模模型。

__实战要求：__

- 配置启用 CPU Offload 的 ZeRO Stage 3
- 评测优化器状态卸载至主机内存（Host Memory）
- 评测模型参数权重卸载至主机内存
- 度量与分析：
  - 单卡所能承载的最大可训练参数量上限
  - 相较纯 GPU 方案的吞吐损耗比率
  - 主机系统 CPU 内存占用曲线
  - PCIe 总线带宽饱和率
- 使用 `pin_memory` 锁页内存以最大化 host-to-device 传输吞吐

__测试验证：__
```python
import deepspeed

# 配置全量 CPU 卸载的 ZeRO-3 引擎
offload_config = {
    "train_batch_size": 8,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "bf16": {"enabled": True},
}

# 递增模型参数规模进行边界极限测试
model_sizes = ["1B", "3B", "7B", "13B"]
for size in model_sizes:
    try:
        model = create_model(size)
        engine, _, _, _ = deepspeed.initialize(model=model, config=offload_config)
        
        # 统计显存、内存与吞吐
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        cpu_mem = get_cpu_memory_usage()
        throughput = benchmark_throughput(engine, num_steps=10)
        
        print(f"{size}: GPU显存={gpu_mem:.1f}GB, CPU内存={cpu_mem:.1f}GB, 吞吐={throughput:.1f} tok/s")
    except RuntimeError as e:
        print(f"{size}: 触发 OOM 异常 - {e}")
        break
```

### 实现张量并行（Tensor Parallelism）

手动实现一个简化版的列并行与行并行线性层，领悟 Megatron-LM 张量切分的核心设计。

__实战要求：__

- 类签名定义：
```python
class ColumnParallelLinear(nn.Module):
    """沿输出特征维度按列切分的并行线性层。"""
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class RowParallelLinear(nn.Module):
    """沿输入特征维度按行切分的并行线性层。"""
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

- 列并行（Column Parallel）：将权重矩阵按输出特征维度 $out\_features$ 均分给各 rank
- 行并行（Row Parallel）：将权重矩阵按输入特征维度 $in\_features$ 均分给各 rank
- 在行并行线性层的输出端执行 AllReduce，将各卡局部结果累加聚合
- 验证张量并行级联结构与标准单卡 Linear 层前向输出的数值精确等价性

__测试验证：__
```python
import torch
import torch.distributed as dist

# 实例化张量并行层
world_size = dist.get_world_size()
rank = dist.get_rank()

col_linear = ColumnParallelLinear(1024, 4096, world_size, rank).cuda()
row_linear = RowParallelLinear(4096, 1024, world_size, rank).cuda()

# 执行前向传播测试
x = torch.randn(32, 1024).cuda()
y = col_linear(x)  # 张量形状: (32, 4096 // world_size)
z = row_linear(y)  # 经过内部 AllReduce 后恢复形状: (32, 1024)

print(f"Rank {rank}: 输入={x.shape}, 列切分后={y.shape}, 行切分求和后={z.shape}")

# 校验正确性（收集各卡输出并与单卡标准 Linear 进行数值绝对误差对比）
```

### 实现流水线并行（Pipeline Parallelism）

构建一个基于微批次（Micro-batching）与 1F1B 调度算法的简易流水线并行训练系统。

__实战要求：__

- 将多层模型切分为 $N$ 个顺序流水线阶段（Pipeline Stages）
- 实现经典的 1F1B（One Forward, One Backward）稳态调度逻辑
- 处理微批次累加与梯度反向传播
- 精确测量流水线气泡（Pipeline Bubble）时间开销占比
- 与纯数据并行方案对比扩展效率

__测试验证：__
```python
class PipelineStage(nn.Module):
    """封装单个流水线计算阶段。"""
    def __init__(self, layers: nn.ModuleList, stage_id: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PipelineParallel:
    def __init__(self, model: nn.Module, num_stages: int, num_microbatches: int):
        self.stages = self.split_model(model, num_stages)
        self.num_microbatches = num_microbatches
    
    def split_model(self, model, num_stages) -> list:
        """将深度模型均匀切分为流水线阶段。"""
        pass
    
    def forward_backward(self, batch):
        """执行 1F1B 稳态交替调度执行。"""
        pass

# 评测流水线并行
model = create_transformer(num_layers=24)
pp = PipelineParallel(model, num_stages=4, num_microbatches=8)

batch = torch.randn(64, 512, 1024).cuda()
loss = pp.forward_backward(batch)
print(f"流水线计算 Loss: {loss.item():.4f}")
```

### 配置 3D 混合并行架构

构建融合数据并行（DP）、张量并行（TP）与流水线并行（PP）的 3D 混合并行拓扑通信环境。

__实战要求：__

- 配置满足 $\text{DP} \times \text{TP} \times \text{PP} = \text{World Size}$ 的通信正交网格
- 以 8 卡环境为例，配置 $\text{DP}=2, \text{TP}=2, \text{PP}=2$ 拓扑
- 正确初始化并绑定各正交维度的独立通信进程组（Process Groups）
- 量化多维混合并行的实际扩展效率，对标单维度扩展方案
- 绘制并分析各通信维度（DP 梯度同步、TP 激活规约、PP 点对点传递）在物理链路上的分布特点

__测试验证：__
```python
import torch.distributed as dist

def setup_3d_parallelism(world_size: int, dp: int, tp: int, pp: int):
    """初始化 3D 混合并行所需的各正交进程组。"""
    assert dp * tp * pp == world_size, "DP × TP × PP 乘积必须严格等于 world_size"
    
    rank = dist.get_rank()
    
    # 计算当前 rank 在 3D 网格中的坐标位置
    dp_rank = rank // (tp * pp)
    tp_rank = (rank // pp) % tp
    pp_rank = rank % pp
    
    # 构建进程组：
    # 数据并行组：相同 TP 与 PP 坐标的 rank 组成组
    # 张量并行组：相同 DP 与 PP 坐标的 rank 组成组
    # 流水线并行组：相同 DP 与 TP 坐标的 rank 组成组
    
    return dp_rank, tp_rank, pp_rank, dp_group, tp_group, pp_group

# 以 8 卡集群为例测试: DP=2, TP=2, PP=2
dp_rank, tp_rank, pp_rank, dp_group, tp_group, pp_group = setup_3d_parallelism(
    world_size=8, dp=2, tp=2, pp=2
)

print(f"Rank {dist.get_rank()}: DP坐标={dp_rank}, TP坐标={tp_rank}, PP坐标={pp_rank}")
print(f"DP 组大小: {dist.get_world_size(dp_group)}")
print(f"TP 组大小: {dist.get_world_size(tp_group)}")
print(f"PP 组大小: {dist.get_world_size(pp_group)}")
```


## 预期学习目标

完成本章实战练习后，你将能够：

- 熟练评测并为生产集群选择最优的 DeepSpeed ZeRO 阶段（Stage 1/2/3）
- 掌握 CPU Offloading 显存卸载技术，利用廉价内存训练远超物理显存规模的模型
- 深刻理解 Megatron-LM 张量并行的代数原理，亲手实现高效的行列张量切分算子
- 实现流水线并行架构，深刻领会 1F1B 调度算法在降低显存峰值与缩减气泡上的精妙设计
- 熟练搭建 3D 混合并行（DP + TP + PP）通信进程组网格，应对万卡级超大模型分布式训练
- 全面掌握各并行维度对网络拓扑（跨节点以太网/InfiniBand vs. 节点内 NVLink）的带宽敏感度并做针对性拓扑编排
