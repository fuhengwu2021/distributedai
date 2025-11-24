# Chapter 8 code snippets

This folder contains runnable snippets and manifests extracted from Chapter 8 (`Kubernetes for AI Workloads`). Use these as starting points — replace image names and tweak resource requests for your environment.

Files:

- `check_nodes.sh`: simple script to list nodes and GPU allocatable resources. Run: `bash check_nodes.sh`.
- `install_gpu_operator.sh`: installs NVIDIA GPU Operator via Helm. Run on a cluster admin machine with `helm` configured.
- `train_multi_gpu_pod.yaml`: example Pod manifest that requests 4 GPUs and runs `torchrun`.
- `pytorch_job.yaml`: example `Job` manifest for single-GPU training.
- `keda_scaledobject.yaml`: example KEDA `ScaledObject` manifest to scale an `inference-deployment` based on a Prometheus metric.

Quick commands:

```bash
# Check nodes
bash check_nodes.sh

# Install GPU operator (requires helm) - cluster admin
bash install_gpu_operator.sh

# Apply Kubernetes manifests (replace images and names first)
kubectl apply -f train_multi_gpu_pod.yaml
kubectl apply -f pytorch_job.yaml
kubectl apply -f keda_scaledobject.yaml
```

Notes:
- Update `image:` fields to point to your container registry.
- For multi-node training, prefer using StatefulSets or a launcher that performs rendezvous (see chapter text).
