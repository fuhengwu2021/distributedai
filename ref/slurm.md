4. 最简 SLURM 单机安装指南（可以直接写进书）
────────────────────────────────────

安装：

sudo apt update
sudo apt install slurm-wlm munge


初始化 munge：

sudo /usr/sbin/create-munge-key
sudo systemctl enable --now munge


配置 slurm.conf（最关键的 Demo 文件）

/etc/slurm/slurm.conf：

ClusterName=demo-cluster
ControlMachine=localhost

SlurmUser=slurm
StateSaveLocation=/var/spool/slurm
SlurmdSpoolDir=/var/spool/slurmd

NodeName=compute[1-1] CPUs=64 RealMemory=256000 Gres=gpu:8 State=UNKNOWN
PartitionName=gpu Nodes=compute[1-1] Default=YES MaxTime=INFINITE State=UP


启用 SLURM 服务：

sudo systemctl enable --now slurmctld
sudo systemctl enable --now slurmd


检查服务：

sinfo


你会看到：

PARTITION AVAIL TIMELIMIT NODES STATE NODELIST
gpu up infinite 1 idle compute1

如果你添加更多 fake nodes：

NodeName=compute[1-4] …

SLURM 会以为有 4 个节点。

────────────────────────────────────
5. GPU 训练 Demo（可直接放书里）
────────────────────────────────────

Example: 用一台 H200 模拟 4 节点 PyTorch DDP 训练

job.sbatch：

#!/bin/bash
#SBATCH --job-name=ddp-demo
#SBATCH --nodes=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2

srun python train.py


PyTorch 环境中：

torchrun --standalone --nnodes=4 --nproc_per_node=2 train.py


虽然只有一台机器，SLURM 会让它“看起来像 4 节点，每个节点 2 GPU”。

对写书来说：这是“完美的教学示例”。