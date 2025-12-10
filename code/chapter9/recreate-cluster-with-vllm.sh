#!/bin/bash
# Script to recreate k3d cluster with both model and vLLM source code mounts

set -e

CLUSTER_NAME="mycluster-gpu"
VLLM_SOURCE_PATH="/home/fuhwu/workspace/distributedai/resources/vllm"
MODELS_PATH="/raid/models"

echo "=== Recreating cluster with vLLM source code mount ==="

# Delete existing cluster
echo "Deleting existing cluster..."
k3d cluster delete $CLUSTER_NAME 2>/dev/null || true

# Wait a moment for cleanup
sleep 2

# Create cluster with both volume mounts
echo "Creating cluster with volume mounts..."
echo "  - Models: $MODELS_PATH -> /models"
echo "  - vLLM source: $VLLM_SOURCE_PATH -> /vllm"

k3d cluster create $CLUSTER_NAME \
  --image k3s-cuda:v1.33.6-cuda-12.2.0 \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume $MODELS_PATH:/models \
  --volume $VLLM_SOURCE_PATH:/vllm

# Wait for cluster to be ready
echo "Waiting for cluster to be ready..."
sleep 10

# Merge kubeconfig
echo "Merging kubeconfig..."
k3d kubeconfig merge $CLUSTER_NAME --kubeconfig-merge-default

# Fix kubeconfig server address if needed
export KUBECONFIG=$HOME/.kube/config
KUBE_SERVER=$(kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-mycluster-gpu")].cluster.server}' 2>/dev/null || echo "")
if [[ "$KUBE_SERVER" == *"0.0.0.0"* ]]; then
  echo "Fixing kubeconfig server address..."
  kubectl config set-cluster k3d-mycluster-gpu --server=$(echo $KUBE_SERVER | sed 's/0.0.0.0/127.0.0.1/')
fi

# Verify cluster
echo ""
echo "=== Cluster Status ==="
kubectl get nodes

echo ""
echo "=== Verifying Volume Mounts ==="
echo "Checking models mount:"
docker exec k3d-mycluster-gpu-server-0 ls -la /models/Phi-tiny-MoE-instruct/ 2>/dev/null | head -3 || echo "Models not found"

echo ""
echo "Checking vLLM source mount:"
docker exec k3d-mycluster-gpu-server-0 ls -la /vllm/ 2>/dev/null | head -5 || echo "vLLM source not found"

echo ""
echo "✅ Cluster ready! You can now deploy vLLM with:"
echo "   kubectl apply -f vllm-phi-tiny-moe.yaml"
