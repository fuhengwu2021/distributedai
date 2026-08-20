\fancydividerwithicon[center]{hand.png}


## 课后实战习题


### 1. 进程组初始化与容错封装

编写一个用于初始化分布式进程组的函数，并实现完善的异常与环境容错处理。

__要求：__

- 函数签名：`setup_distributed(rank, world_size, backend='nccl')`
- 使用 `torch.distributed.init_process_group()` 初始化进程组
- 将当前进程绑定至对应的 CUDA GPU 设备：`torch.cuda.set_device(rank)`
- 增加对当前环境 CUDA 是否可用的异常捕获与错误处理
- 初始化成功返回 `True`，失败返回 `False`
- 成功初始化后打印包含当前 rank 与 world_size 的提示信息

__测试代码：__
```python
import torch.distributed as dist
import os

# 从环境变量中模拟获取 rank 和 world_size
rank = int(os.environ.get('RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))

if setup_distributed(rank, world_size):
    print(f"Rank {rank}/{world_size} 初始化成功")
    dist.destroy_process_group()
```

### 2. 手动实现 AllReduce 梯度同步

使用 `dist.all_reduce()` 编写一个手动的梯度平均同步函数，模拟 DDP 底层在反向传播时所执行的同步逻辑。

__要求：__

- 函数签名：`average_gradients(model, world_size)`
- 遍历模型中的所有参数
- 对于每一个 `requires_grad=True` 的参数：
  - 使用 `dist.all_reduce()` 配合 `op=dist.ReduceOp.SUM` 对所有 Rank 上的梯度求和
  - 将求和后的梯度除以 `world_size` 得到平均梯度
- 正确处理梯度可能为 `None` 的情况（跳过无梯度的参数）

__测试代码：__
```python
import torch
import torch.nn as nn
import torch.distributed as dist

# 构建简单模型
model = nn.Linear(10, 1).cuda()
loss_fn = nn.MSELoss()

# 前向传播与反向传播
x = torch.randn(32, 10).cuda()
y = torch.randn(32, 1).cuda()
output = model(x)
loss = loss_fn(output, y)
loss.backward()

# 跨卡同步平均梯度
average_gradients(model, world_size=2)

# 验证梯度是否已平均同步（AllReduce 之后各卡梯度值应完全一致）
print(f"Rank {rank} 上的权重梯度: {model.weight.grad}")
```

### 3. 带校验的 Broadcast 广播

实现一个将张量从 Rank 0 广播至所有其他 Rank、并在各卡校验数据一致性的函数。

__要求：__

- 函数签名：`broadcast_and_verify(tensor, root=0)`
- 若当前进程为根节点（Rank 0）：创建或使用传入的源张量
- 若当前进程非根节点：创建相同形状的全 0 张量接收数据
- 使用 `dist.broadcast()` 将数据从根节点广播至所有 Rank
- 广播完成后，校验所有卡上的张量数据是否完全一致
- 返回广播后的张量以及指示校验是否通过的布尔值（`verified`）

__测试代码：__
```python
import torch
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

if rank == 0:
    # 根节点创建源数据
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
else:
    data = None

result, verified = broadcast_and_verify(data, root=0)
print(f"Rank {rank}: {result}, 校验通过: {verified}")
```

### 4. 自动递增 Epoch 的 DistributedSampler 封装

封装一个 `DistributedSampler` 包装类，自动管理并递增 Epoch 计数，免去在外部手动调用的遗漏风险。

__要求：__

- 类名：`AutoEpochDistributedSampler`
- 继承自 `torch.utils.data.distributed.DistributedSampler`
- 重写 `__iter__()` 方法，在每次迭代前使用内部的 Epoch 计数器自动调用 `set_epoch()`
- 提供 `reset()` 方法重置 Epoch 计数器
- 提供 `get_epoch()` 方法获取当前的 Epoch 序号
- 每次调用 `__iter__()` 时内部计数器自动累加 1

__测试代码：__
```python
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# 模拟虚拟数据集
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

# 遍历多个 Epoch —— Epoch 序号应自动递增
for epoch in range(3):
    for batch in dataloader:
        pass  # 处理 Batch
    print(f"Epoch {sampler.get_epoch()} 完成")
```

### 5. 梯度同步耗时 Profiling 评测

编写一个脚本，精确度量在不同模型规模与 World Size 下 AllReduce 梯度同步所需的时间开销。

__要求：__

- 函数签名：`measure_sync_time(model, num_iterations=10)`
- 构造参数量可配置的简单模型
- 执行前向与反向传播计算
- 精确度量所有参数梯度执行 `dist.all_reduce()` 的耗时
- 返回平均同步耗时（单位：毫秒 ms）
- 测试不同参数规模（1M、10M、100M 参数）在不同 World Size 下的耗时表现
- 打印输出结果，分析通信耗时随参数量与节点数增加的变化趋势

__测试代码：__
```python
import torch
import torch.nn as nn
import time
import torch.distributed as dist

def create_model(num_params):
    """创建约包含 num_params 参数量的测试模型"""
    return nn.Sequential(
        nn.Linear(1000, num_params // 2000),
        nn.ReLU(),
        nn.Linear(num_params // 2000, 10)
    ).cuda()

model_sizes = [1e6, 10e6, 100e6]  # 1M, 10M, 100M 参数量

for size in model_sizes:
    model = create_model(int(size))
    sync_time = measure_sync_time(model, num_iterations=10)
    print(f"模型规模: {size/1e6:.1f}M 参数, "
          f"平均同步耗时: {sync_time:.2f} ms, "
          f"World Size: {dist.get_world_size()}")
```


## 学习成果自测

完成本章所有练习后，你应当能够：

- 熟练计算不同规模与精度下的模型显存需求（权重、优化器状态、激活值、KV 缓存）
- 正确配置并初始化 PyTorch 分布式进程组与 CUDA 设备绑定
- 深刻理解 DDP 底层梯度同步与反向传播的协作机制
- 熟练运用核心集合通信原语（AllReduce、AllGather、Broadcast、ReduceScatter 等）
- 掌握 `DistributedSampler` 的数据切分机制与跨 Epoch 随机 Shuffle 逻辑
- 具备对分布式集群网络通信开销进行量化测量与瓶颈分析的能力
