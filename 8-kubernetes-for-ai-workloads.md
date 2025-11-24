---
title: "Kubernetes for AI Workloads"
---

# Chapter 8 — Kubernetes for AI Workloads

This chapter provides a practical guide to deploying, operating, and scaling AI workloads on Kubernetes. It covers GPU operators and device plugins, scheduling patterns for multi-GPU jobs, autoscaling (HPA, KEDA, queue-based), running distributed training and inference, observability, and hands-on manifests and recipes.

Audience: platform engineers and ML engineers who want reproducible, cost-effective Kubernetes patterns for training and inference workloads.

## 1. Enabling GPU Support in Kubernetes

Install the NVIDIA GPU Operator (or equivalent for your vendor) to manage drivers, container toolkits, and device plugins. Basic checklist:

- Ensure the node OS has compatible drivers for your GPU and CUDA version.
- Install the NVIDIA GPU Operator via Helm or OLM to manage drivers and the `nvidia-dcgm-exporter` for metrics.
- Confirm the device plugin is running and that nodes advertise `nvidia.com/gpu` resources:

```bash
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPUS:.status.allocatable
```

Tip: use node feature discovery to label nodes with accelerator types (A100, H100) to support heterogeneous clusters.

## 2. Scheduling Patterns for GPU Workloads

Common patterns:

- Pod-per-GPU: simplest model where each Pod requests `resources: requests.gpu: 1`. Use when jobs can be sharded with one process per GPU.
- Multi-GPU Pod: request multiple GPUs in a single Pod (e.g., `limits: nvidia.com/gpu: 4`) for frameworks that support intra-pod multi-GPU communication (e.g., `torchrun --nproc_per_node`). Prefer when NVLink or shared PCIe locality matters.
- DaemonSets for auxiliary services: run DCGM exporter or node-local services as DaemonSets.

Affinity and topology:

- Use `podAntiAffinity` and `nodeAffinity` to control pod placement.
- Use `topology.kubernetes.io/zone` and custom labels to keep multi-pod jobs within a failure domain or to exploit NVLink locality.

Example snippet (multi-GPU Pod):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: train-multi-gpu
spec:
  containers:
  - name: trainer
    image: my-registry/my-train:latest
    resources:
      limits:
        nvidia.com/gpu: 4
    command: ["/bin/bash", "-c", "torchrun --nproc_per_node=4 train.py"]
  restartPolicy: Never
```

## 3. Autoscaling Strategies (HPA, KEDA, Queue-Based)

Inference autoscaling:

- Horizontal Pod Autoscaler (HPA): scale based on custom metrics (latency, QPS) reported via the Metrics API or Prometheus Adapter.
- KEDA: use event-driven autoscaling for queue-based workloads (e.g., Kafka, RabbitMQ, Redis streams). KEDA integrates with metrics to scale to zero when idle.

Training autoscaling:

- Prefer queue-based scaling: a controller watches a job queue (Argo Events, a database, or a custom CRD) and scales a worker pool up and down. This is more predictable for long-lived training jobs and avoids noisy autoscaling triggers.

Example: scale inference deployment with KEDA (conceptual):

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: inference-scaledobject
spec:
  scaleTargetRef:
    name: inference-deployment
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: inference_queue_length
      threshold: '50'
```

## 4. Running Distributed Training on Kubernetes

Patterns and tools:

- Use `kubectl` Jobs for single-process training, or custom controllers like Kubeflow's MPIJob, or the Kubeflow TFJob operator for TensorFlow.
- For PyTorch, use `torchrun` with a launcher `init_method` that supports multi-node discovery (e.g., rendezvous via etcd, kubernetes API, or a headless service).
- Consider using a controller that manages lifecycle and checkpointing (e.g., a custom Job CRD that attaches persistent volumes).

Example: minimal headless service + StatefulSet pattern for rendezvous (conceptual):

1. Create a headless service to allow pods to discover each other.
2. Launch a StatefulSet with `replicas: N` that runs `torchrun --nproc_per_node=<gpus>`.

Checkpointing and storage:

- Use an object store (S3-compatible) or networked filesystem (NFS, Ceph, PV + PVC) for checkpoint persistence.
- Implement periodic checkpoint uploads and a restart/resume mechanism in your training script.

## 5. Inference Deployment Patterns

- Serve models as Deployments or Knative services for autoscale-to-zero semantics.
- Use routers (described in Chapter 7) in front of runners to implement session affinity for KV cache locality.
- Consider using NodePools for different instance types (GPU fast instances, CPU-only cheap instances) and route requests appropriately.

Batching and adaptive batching:

- Use sidecar or middleware components to accept requests and batch them before invoking the model.
- Implement dynamic batching algorithms that trade latency vs throughput and can be tuned per model.

## 6. Observability, Profiling and Cost Controls

Metrics and logs:

- Export GPU metrics via `nvidia-dcgm-exporter`.
- Instrument application metrics (QPS, latency, queue depth) and ship to Prometheus.
- Collect traces and spans with OpenTelemetry for request flows across services.

Profiling:

- Use `torch.profiler` for operator-level hotspots during training.
- Use Nsight Systems for system-level tracing across CUDA kernels and CPU threads.

Cost controls:

- Use Cluster Autoscaler with node group limits and spot/spot-fallback strategies.
- Implement preemption-aware jobs to take advantage of cheaper capacity while building robust checkpointing.

## 7. Security and Multi-Tenancy

- Use namespaces and ResourceQuota to isolate tenants.
- Apply Pod Security admission and network policies to limit access.
- Restrict access to GPU nodes with RBAC and node selectors.

## 8. Hands-on Examples

1. Installing NVIDIA GPU Operator via Helm (commands):

```bash
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update
helm install --namespace gpu-operator-system --create-namespace gpu-operator nvidia/gpu-operator
```

2. Minimal `torchrun` Job manifest (Job controller):

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-train
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: train
        image: my-registry/pytorch-train:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        command: ["/bin/bash", "-c", "torchrun --nproc_per_node=1 train.py"]
```

## 9. Troubleshooting Guide

- Pod stuck in `ContainerCreating`: check device plugin logs and node driver health.
- `Insufficient nvidia.com/gpu`: verify node allocatable resources and scheduling constraints.
- Networking issues with rendezvous: inspect headless service and DNS resolution between pods.

## 10. Next Steps and Automation

- Automate cluster provisioning with tools like Cluster API or Terraform.
- Add CI jobs to validate manifests and run small integration tests (smoke tests) for training and inference pipelines.
- Consider building a custom operator for lifecycle and cost-aware scheduling of long-running training jobs.

---

References: NVIDIA GPU Operator, KEDA, Kubernetes Cluster Autoscaler, Kubeflow docs.
---
title: "Kubernetes for AI Workloads"
---

# Chapter 8 — Kubernetes for AI Workloads

This chapter teaches how to deploy, schedule, and operate AI workloads on Kubernetes with GPUs. Topics include GPU operators, device plugins, node labeling, taints/tolerations, autoscaling (HPA, KEDA), and queue-based scaling for training and inference.

## 1. Enabling GPU Support in Kubernetes

Install GPU operators (NVIDIA GPU Operator) and device plugins. Ensure drivers and container runtimes are configured for GPU access in pods.

## 2. Scheduling GPU Workloads Effectively

Use node labels and taints to isolate GPU nodes; use resource requests/limits and device plugins to request GPUs. For multi-GPU jobs, use pod-per-GPU or cudnn-device plugin strategies depending on your launcher.

## 3. Autoscaling Strategies (HPA, KEDA, Queue-Based)

For inference, use HPA or KEDA with custom metrics (queue length, latency). For training, queue-based scaling (scale workers based on job queue) provides better cost-control.

## 4. Distributed Training on Kubernetes

Patterns: use Kubernetes Job/TFJob/Custom controllers to launch multi-pod training; consider using MPI-operator or RunPod-style tools for orchestration. Ensure shared storage for checkpoints.

## 5. Observability and Troubleshooting

Set up Prometheus + Grafana for metrics, and use NVIDIA DCGM exporter for GPU metrics. Capture pod logs and use `kubectl describe` to diagnose scheduling issues.

## Hands-on Examples

1. GPU operator install YAML and Helm chart example.
2. Kubernetes manifest for a multi-pod training job using `kubectl` and `torchrun`.

## Best Practices

- Isolate GPU workloads with node pools and taints.  
- Use affinity rules to keep multi-GPU pods on the same host when NVLink matters.  
- Automate backup and rotated checkpoints to shared storage.

---

References: NVIDIA GPU Operator docs, K8s device plugin docs, KEDA docs.
---
title: "Kubernetes for AI Workloads"
---

# Chapter 8 — Kubernetes for AI Workloads

Status: TODO — draft placeholder

Chapter headings:
1. Enabling GPU Support in Kubernetes
2. Scheduling GPU Workloads Effectively
3. Autoscaling Strategies (HPA, KEDA, Queue-Based)
4. Distributed Training on Kubernetes
5. Observability and Troubleshooting

TODO: Provide manifests, helm charts, and operator instructions.
