#!/usr/bin/env bash
set -euo pipefail
echo "Installing NVIDIA GPU Operator via Helm (namespace: gpu-operator-system)"
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update
helm install --namespace gpu-operator-system --create-namespace gpu-operator nvidia/gpu-operator
