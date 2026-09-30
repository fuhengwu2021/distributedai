\fancydividerwithicon[center]{hand.png}


## 实战演练


### 初始化分布式进程组

实现一个具备完备异常处理机制的分布式进程组初始化函数。

__实战要求：__

- 函数签名：`setup_distributed(rank, world_size, backend='nccl')`
- 使用 `torch.distributed.init_process_group()` 初始化进程组
- 为当前 rank 绑定对应的 CUDA 计算设备：`torch.cuda.set_device(rank)`
- 增加异常捕获机制，处理 CUDA 环境不可用的情况
- 初始化成功返回 `True`，失败返回 `False`
- 打印初始化成功的日志，包含当前的 rank 与 `world_size`

__测试验证：__
```python
import torch.distributed as dist
import os

# 从环境变量中读取模拟的 rank 与 world_size
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

if setup_distributed(rank, world_size):
    print(f"Rank {rank}/{world_size} 初始化成功")
    dist.destroy_process_group()
```

### 手动实现 AllReduce 梯度规约

通过底层 AllReduce 原语手动实现多卡梯度平均函数，模拟 DDP 底层通信的核心行为。

__实战要求：__

- 函数签名：`average_gradients(model, world_size)`
- 遍历模型中所有可训练参数
- 针对每个 `requires_grad=True` 的参数：
  - 调用 `dist.all_reduce()` 并指定 `op=dist.ReduceOp.SUM`，将所有 rank 上的梯度求和
  - 将累加后的梯度除以 `world_size` 获得全局平均梯度
- 正确处理梯度可能为 `None` 的参数（跳过未参与反向传播的参数）

__测试验证：__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

# 构建一个简单的单层线性模型
model = nn.Linear(10, 1).cuda()
loss_fn = nn.MSELoss()

# 前向传播与反向传播
x = torch.randn(32, 10).cuda()
y = torch.randn(32, 1).cuda()
output = model(x)
loss = loss_fn(output, y)
loss.backward()

# 跨 rank 手动同步平均梯度
average_gradients(model, world_size=2)

# 验证所有 rank 上的梯度是否一致（执行 all_reduce 后各卡梯度应完全相等）
print(f"Rank {rank} 上的权重梯度: {model.weight.grad}")
```

### 带一致性校验的 Broadcast 广播

实现一个将 Rank 0 的张量广播至所有其他进程组节点，并自动执行数值一致性校验的函数。

__实战要求：__

- 函数签名：`broadcast_and_verify(tensor, root=0)`
- 若当前 rank 为 root：使用传入的源张量
- 若当前 rank 非 root：在本地创建一个相同形状的全零张量占位
- 调用 `dist.broadcast()` 将 root 节点的张量分发至所有 rank
- 广播完成后，校验各 rank 上的数据是否与 root 保持严格一致
- 返回广播后的张量以及指示校验是否通过的布尔值

__测试验证：__
```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

if rank == 0:
    # Root 节点创建源数据
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
else:
    data = None

result, verified = broadcast_and_verify(data, root=0)
print(f"Rank {rank}: {result}, 校验状态: {verified}")
```

### 自动处理 Epoch 的 DistributedSampler 封装

封装 `DistributedSampler` 类，使其在每个 epoch 开始时自动调用 `set_epoch()`，避免手动设置遗漏导致的数据重排失效。

__实战要求：__

- 类名：`AutoEpochDistributedSampler`
- 继承自 `torch.utils.data.distributed.DistributedSampler`
- 重写 `__iter__()` 方法，在内部维护 epoch 计数器并在每次迭代开始时自动调用 `set_epoch()`
- 增加 `reset()` 方法重置 epoch 计数器
- 增加 `get_epoch()` 方法获取当前的 epoch 序号
- 每次触发 `__iter__()` 时自增内部 epoch 计数

__测试验证：__
```python
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# 构造模拟数据集
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.data = list(range(size))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = DummyDataset(100)
sampler = AutoEpochDistributedSampler(
    dataset, 
    num_replicas=world_size,
    rank=rank
)
dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

# 遍历多个 epoch —— sampler 内部应自动自增 epoch
for epoch in range(3):
    for batch in dataloader:
        pass  # 模拟批次处理
    print(f"Epoch {sampler.get_epoch()} 训练完成")
```

### 梯度同步耗时基准度量

编写测试脚本，精确度量不同规模张量在执行 AllReduce 梯度规约时的网络通信耗时。

__实战要求：__

- 函数签名：`measure_sync_time(model, num_iterations=10)`
- 创建可配置参数量的模型结构
- 执行前向与反向传播
- 精确测量所有梯度张量调用 `dist.all_reduce()` 的纯通信耗时
- 返回多次迭代的平均同步耗时（单位：毫秒）
- 针对不同模型体量（1M、10M、100M 参数）及不同进程组规模（world size）分别测试
- 打印测试对比结果，分析同步耗时随参数规模与卡数扩展的变化趋势

__测试验证：__
```python
import torch
import torch.nn as nn
import time
import torch.distributed as dist

def create_model(num_params):
    """构建一个参数量近似为 num_params 的简易网络"""
    layers = []
    # 简易参数近似计算：针对 Linear(in, out)，参数量约为 in * out + out
    return nn.Sequential(
        nn.Linear(1000, num_params // 2000),
        nn.ReLU(),
        nn.Linear(num_params // 2000, 10)
    ).cuda()

model_sizes = [1e6, 10e6, 100e6]  # 分别测试 1M、10M、100M 参数规模

for size in model_sizes:
    model = create_model(int(size))
    sync_time = measure_sync_time(model, num_iterations=10)
    print(f"模型参数规模: {size/1e6:.1f}M, "
          f"平均同步耗时: {sync_time:.2f} ms, "
          f"World Size: {dist.get_world_size()}")
```


## 预期学习目标

完成本章实战练习后，你将能够：

- 针对不同模型体量与精度配置，精确估算训练显存开销与理论带宽瓶颈
- 熟练编写具备异常处理能力的分布式进程组初始化与生命周期管理代码
- 深入掌握 DDP 底层梯度同步与 AllReduce 集合通信的工作原理
- 灵活运用 AllReduce、Broadcast 等通信原语实现自定义协同逻辑
- 正确使用并扩展 DistributedSampler，保障分布式数据分片与跨 Epoch 洗牌的确定性
- 独立编写通信基准测试脚本，量化多卡网络传输耗时与扩展瓶颈
