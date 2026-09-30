\fancydividerwithicon[center]{hand.png}


## 实战演练


### 手动实现参数分片（Parameter Sharding）

实现一个简化版的 FSDP 参数分片机制，深入理解其如何突破单卡物理显存上限承载超大模型。

__实战要求：__

- 函数签名：
```python
def shard_parameters(model: nn.Module, world_size: int, rank: int) -> dict:
    """跨所有 rank 分片模型参数，各 rank 仅保留本地分片。"""
    pass

def gather_parameters(sharded_params: dict, world_size: int) -> dict:
    """在前向/反向计算前收集所有分片，动态重构完整参数。"""
    pass
```

- 将模型所有参数张量展平拼接为一个连续的扁平一维张量
- 将展平后的全局参数张量均分为 `world_size` 份独立分片
- 每个 rank 仅在本地持久保存属于自身的分片（占全局参数量的 $1/\text{world\_size}$）
- 实现 gather 逻辑，在执行前向传播时临时通过通信聚合并重构完整参数

__测试验证：__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 100)
)

# 执行参数分片
sharded = shard_parameters(model, world_size=4, rank=dist.get_rank())
print(f"Rank {dist.get_rank()}: 本地分片元素数量 = {sharded['shard'].numel()}")

# 临时全收集以支持前向传播
full_params = gather_parameters(sharded, world_size=4)
print(f"重构后的全局参数元素总量: {sum(p.numel() for p in full_params.values())}")
```

### 对比 FSDP 分片策略与权衡

编写基准测试脚本，量化评测 FSDP 不同分片策略下的显存占用与吞吐性能权衡。

__实战要求：__

- 对比评测三种典型分片策略：
  - `FULL_SHARD`（ZeRO-3）：深度分片模型参数、梯度与优化器状态
  - `SHARD_GRAD_OP`（ZeRO-2）：仅分片梯度与优化器状态，参数保持复制
  - `NO_SHARD`（等同于 DDP）：不进行任何状态分片
- 针对每种策略度量：
  - 峰值显存占用（Peak GPU Memory）
  - 训练吞吐率（samples/second）
  - 网络通信吞吐总量
- 选用一个在 `NO_SHARD` 模式下单卡直接触发 OOM 的大模型进行实验
- 自动生成对比汇总表格

__测试验证：__
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

strategies = [
    ShardingStrategy.FULL_SHARD,
    ShardingStrategy.SHARD_GRAD_OP,
    ShardingStrategy.NO_SHARD,
]

results = {}
for strategy in strategies:
    model = create_large_model()  # 例如 7B 参数量模型
    fsdp_model = FSDP(model, sharding_strategy=strategy)
    
    memory, throughput = benchmark_training(fsdp_model, num_steps=100)
    results[strategy.name] = {"memory_gb": memory, "throughput": throughput}

# 打印评测对比结果
print("分片策略        | 峰值显存 (GB) | 训练吞吐 (samples/s)")
for name, metrics in results.items():
    print(f"{name:15} | {metrics['memory_gb']:10.1f} | {metrics['throughput']:10.1f}")
```

### 自定义 FSDP 层级封装策略（Wrap Policy）

编写针对 Transformer 架构的自定义自动封装策略，实现 Transformer Block 维度的精细化显存释放。

__实战要求：__

- 函数签名：
```python
def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    min_num_params: int = 1e6
) -> bool:
    """针对 Transformer 模型的自定义层级封装判定策略。"""
    pass
```

- 将每一个独立的 Transformer Block（包含注意力机制与前馈网络）包装为独立的 FSDP 单元
- 避免对词嵌入层（Embedding）进行碎片化封装（保留在根 FSDP 单元中统一通信）
- 避免对顶层输出投影层（LM Head）进行过度嵌套
- 支持配置 `min_num_params` 参数阈值，忽略过小的轻量子模块

__测试验证：__
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import _module_wrap_policy
from transformers import AutoModelForCausalLM
import functools

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

fsdp_model = FSDP(
    model,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        min_num_params=1e6
    )
)

# 递归统计封装生成的 FSDP 单元总数
def count_fsdp_units(module, depth=0):
    count = 1 if isinstance(module, FSDP) else 0
    for child in module.children():
        count += count_fsdp_units(child, depth + 1)
    return count

print(f"生成的独立 FSDP 单元总数: {count_fsdp_units(fsdp_model)}")
```

### 配置 FSDP 混合精度策略

配置并评测 FSDP 在不同混合精度策略（Mixed Precision）下的显存收益与训练收敛性。

__实战要求：__

- 评测三种典型的数值精度策略：
  - 全 FP32 精度（作为基线对照）
  - BF16 计算 + FP32 参数与规约缓冲
  - 全流程 BF16（参数、梯度通信与前向计算均采用 BF16）
- 度量对比：
  - 相较全 FP32 的静态与动态显存节省幅度
  - 训练损失收敛轨迹
  - 实际算力吞吐提升比
- 若采用 FP16，增加梯度缩放器（GradScaler）处理防止下溢

__测试验证：__
```python
from torch.distributed.fsdp import MixedPrecision
import torch

# 定义各精度策略
fp32_policy = MixedPrecision()

bf16_compute_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

bf16_full_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

policies = {
    "FP32 全精度": fp32_policy,
    "BF16 仅计算": bf16_compute_policy,
    "BF16 全流程": bf16_full_policy,
}

for name, policy in policies.items():
    model = FSDP(create_model(), mixed_precision=policy)
    memory, loss, throughput = train_and_measure(model, num_steps=100)
    print(f"{name}: 显存占用={memory:.1f}GB, Loss={loss:.4f}, 吞吐={throughput:.1f}")
```

### 实现 FSDP 分布式检查点存取与重新分片

构建一套健壮的 FSDP 模型 Checkpoint 保存与恢复系统，支持异构集群拓扑与跨卡数弹性重新分片（Resharding）。

__实战要求：__

- 保存的检查点能够跨不同 GPU 数量与拓扑平滑恢复（例如 4 卡保存，8 卡加载恢复）
- 兼容完整状态字典（Full State Dict）与分片状态字典（Sharded State Dict）两种模式
- 实现检查点完整性校验机制（确保保存前后张量哈希与数值严格对齐）
- 完整包含优化器分布式状态的持久化与反序列化
- 支持从任意保存步数无缝恢复训练

__测试验证：__
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

class FSDPCheckpointer:
    def __init__(self, model: FSDP, optimizer, checkpoint_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
    
    def save(self, step: int, full_state: bool = True):
        """持久化保存分布式检查点。"""
        pass
    
    def load(self, step: int) -> dict:
        """从磁盘加载指定步数的检查点并还原状态。"""
        pass
    
    def validate(self, step: int) -> bool:
        """校验所保存检查点的数值一致性与完整性。"""
        pass

# 测试保存与恢复流水线
checkpointer = FSDPCheckpointer(fsdp_model, optimizer, "./checkpoints")

# 训练若干迭代步
train_steps(fsdp_model, optimizer, num_steps=10)

# 保存断点
checkpointer.save(step=10)

# 校验断点正确性
assert checkpointer.validate(step=10), "检查点完整性校验未通过！"

# 重新加载并恢复训练
metadata = checkpointer.load(step=10)
print(f"成功恢复训练，当前步数: Step {metadata['step']}")
```


## 预期学习目标

完成本章实战练习后，你将能够：

- 透彻理解 FSDP 核心参数分片、AllGather 动态收集与 ReduceScatter 梯度规约的交互逻辑
- 熟练根据硬件互连与模型显存规模，精准选择 `FULL_SHARD`、`SHARD_GRAD_OP` 或 `NO_SHARD`
- 针对各类 Transformer 架构定制高效的 Auto-Wrap 封装策略，最大化显存与通信重叠效率
- 熟练配置 BF16/FP16 混合精度策略，以极低的显存开销获取 Tensor Core 极限算力
- 构建支持异构拓扑与跨卡数重新分片的高可用 Checkpoint 恢复系统
- 在保障高吞吐的前提下，极致压缩超大参数量模型在单卡上的显存驻留
