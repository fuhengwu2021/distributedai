明白——如果你的目的只是**写书做 Demo（示范用）**，而不是给读者搭生产环境，那么你需要的 Kubernetes 环境必须满足：

1. **一台机器即可运行**（读者无需多机）
2. **安装步骤足够短，几分钟可完成**
3. **能表现出“真实 Kubernetes GPU 集群”的关键特性**
4. **适合截图、讲解、跑简单 GPU workload（比如 vLLM / PyTorch）**

基于这些要求，最适合写书用的是：

───────────────────────────────
最佳选择：**k3s + NVIDIA GPU plugin**（强烈推荐）
───────────────────────────────

为什么 k3s 是最适合写书 Demo 的方案？

轻量：
一个二进制文件，安装一步完成，不需要 kubeadm 那么复杂。

真实：
API 是正宗 Kubernetes（CNCF 认证），写的 YAML 与普通 K8s 完全一致。

简单：
几行命令搞定，非常适合书籍教学，不会吓到读者。

支持 GPU：
可以加载 NVIDIA k8s-device-plugin，展示 GPU scheduling、pod 要求 GPU 等用法。

不容易出错：
kubeadm 很容易因为 CNI、swap、iptables 等读者机器差异导致失败。

───────────────────────────────
最简 Demo 方案（可以直接放进书里）
───────────────────────────────

**前置条件（读者需要先完成）：**

1. 安装 NVIDIA 驱动（如果还没有）
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-535  # 或更新版本
   sudo reboot
   ```

2. 安装 NVIDIA Container Toolkit
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart containerd  # 如果已有 containerd
   ```

Step 1. 安装 k3s（读者只需一条命令）

```
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik" sh -
```

Step 2. 配置 k3s 使用 NVIDIA 运行时

编辑 k3s containerd 配置模板：

```bash
sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd
sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl > /dev/null <<EOF
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
    BinaryName = "/usr/bin/nvidia-container-runtime"
EOF

sudo systemctl restart k3s
```

Step 3. 配置 kubectl（设置 kubeconfig）

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config
kubectl get nodes
```

Step 4. 安装 NVIDIA GPU 插件

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.3/nvidia-device-plugin.yml
```

等待插件就绪：

```bash
kubectl wait --for=condition=ready pod -l name=nvidia-device-plugin-ds -n kube-system --timeout=60s
```

验证 GPU 可见：

```bash
kubectl describe node | grep nvidia.com/gpu
```

Step 5. 写一个简单的 GPU Pod（书中可配图）

gpu-pod.yaml：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: OnFailure
  containers:
  - name: cuda
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

运行：

```bash
kubectl apply -f gpu-pod.yaml
kubectl wait --for=condition=Ready pod/gpu-test --timeout=60s
kubectl logs gpu-test
```

读者看到 nvidia-smi 输出 → Demo 成功。

───────────────────────────────
在用户主目录安装（无需 root，适合受限环境）
───────────────────────────────

如果你没有 sudo 权限或想在主目录下运行，有两个选择：

**方案 A：使用 k3d（推荐，最简单）**

k3d 是 k3s in Docker，完全在用户空间运行，支持 GPU：

```bash
# 1. 安装 k3d（只需下载二进制，无需 root）
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# 2. 创建支持 GPU 的 k3s 集群
# 选项 A：使用所有 GPU
k3d cluster create mycluster \
  --image rancher/k3s:v1.33.6-k3s1 \
  --gpus=all \
  --volume ~/.k3d:/var/lib/rancher/k3s

# 选项 B：只使用特定 GPU（例如 GPU 4 和 5）
k3d cluster create mycluster \
  --image rancher/k3s:v1.33.6-k3s1 \
  --gpus "device=4,5" \
  --volume ~/.k3d:/var/lib/rancher/k3s

# 选项 C：为不同节点分配不同 GPU
# 先创建集群，然后为每个节点单独指定 GPU
k3d cluster create mycluster \
  --image rancher/k3s:v1.33.6-k3s1 \
  --servers 1 \
  --agents 1
k3d node create server-0 --cluster mycluster --role server --gpus "device=4"
k3d node create agent-0 --cluster mycluster --role agent --gpus "device=5"

# 3. 配置 kubectl
kubectl cluster-info

# 4. 安装 NVIDIA GPU 插件（与标准 k3s 相同）
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.3/nvidia-device-plugin.yml

# 5. 验证 GPU
kubectl wait --for=condition=ready pod -l name=nvidia-device-plugin-ds -n kube-system --timeout=60s
kubectl describe node | grep nvidia.com/gpu
```

**清理集群：**
```bash
k3d cluster delete mycluster
```

**创建多节点集群（2 节点示例）：**

```bash
# 创建包含 1 个 control-plane 和 1 个 worker 节点的集群
# 使用所有 GPU
k3d cluster create mycluster \
  --image rancher/k3s:v1.33.6-k3s1 \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume ~/.k3d:/var/lib/rancher/k3s

# 或者只使用特定 GPU（例如 GPU 4 和 5）
k3d cluster create mycluster \
  --image rancher/k3s:v1.33.6-k3s1 \
  --gpus "device=4,5" \
  --servers 1 \
  --agents 1

# 验证节点
kubectl get nodes

# 应该看到类似输出：
# NAME                    STATUS   ROLES           AGE   VERSION
# k3d-mycluster-server-0 Ready    control-plane   30s   v1.33.6+k3s1
# k3d-mycluster-agent-0   Ready    <none>          25s   v1.33.6+k3s1
```

**GPU 分配选项说明：**

```bash
# 查看可用 GPU
nvidia-smi --query-gpu=index,gpu_name --format=csv

# 使用所有 GPU
--gpus=all

# 使用特定 GPU（单个）
--gpus "device=4"

# 使用多个特定 GPU
--gpus "device=4,5"

# 使用 GPU 范围（如果支持）
--gpus "device=4-7"  # GPU 4, 5, 6, 7
```

**创建更多节点：**

```bash
# 创建 3 节点集群（1 control-plane + 2 workers）
k3d cluster create mycluster \
  --image rancher/k3s:v1.33.6-k3s1 \
  --gpus=all \
  --servers 1 \
  --agents 2

# 或者创建高可用集群（3 control-planes + 2 workers）
k3d cluster create mycluster \
  --image rancher/k3s:v1.33.6-k3s1 \
  --gpus=all \
  --servers 3 \
  --agents 2
```

**在现有集群中添加节点：**

```bash
# 添加一个 worker 节点（使用所有 GPU）
k3d node create new-worker --cluster mycluster --role agent --gpus=all

# 添加一个 worker 节点（使用特定 GPU）
k3d node create new-worker --cluster mycluster --role agent --gpus "device=5"

# 添加一个 control-plane 节点（用于高可用）
k3d node create new-server --cluster mycluster --role server --gpus=all
```

**查看集群信息：**

```bash
# 列出所有集群
k3d cluster list

# 查看集群详细信息
k3d cluster get mycluster

# 查看所有节点
k3d node list
```

**方案 B：k3s Rootless 模式（实验性）**

k3s 支持 rootless 模式，但需要一些额外配置：

```bash
# 1. 安装 rootlesskit 和 slirp4netns（如果还没有）
sudo apt-get install -y rootlesskit slirp4netns

# 2. 下载 k3s 二进制到主目录
mkdir -p ~/bin
curl -sfL https://github.com/k3s-io/k3s/releases/download/v1.33.6%2Bk3s1/k3s -o ~/bin/k3s
chmod +x ~/bin/k3s
export PATH=$HOME/bin:$PATH

# 3. 设置环境变量
export K3S_DATA_DIR=$HOME/.local/share/k3s
export K3S_CONFIG_FILE=$HOME/.config/k3s/config.yaml

# 4. 启动 k3s rootless（在后台运行）
k3s server --rootless --data-dir $K3S_DATA_DIR &
sleep 10

# 5. 配置 kubectl
mkdir -p ~/.kube
KUBECONFIG=$HOME/.config/k3s/k3s.yaml kubectl get nodes
```

**注意：** Rootless 模式对 GPU 支持有限，如果主要目的是演示 GPU workload，**强烈推荐使用 k3d**。

───────────────────────────────
额外可展示的 Demo（非常适合书籍）
───────────────────────────────
