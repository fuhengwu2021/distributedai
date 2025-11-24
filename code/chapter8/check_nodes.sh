#!/usr/bin/env bash
# Check node GPU allocatable resources
set -euo pipefail
echo "Listing nodes and GPU allocatable resources"
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPUS:.status.allocatable
