# Chapter 9: Production LLM Serving Stack {-}

*Building end-to-end production systems for reliable LLM serving*

> Everything fails all the time.
- Werner Vogels, CTO of Amazon

**Code Summary**

- `fastapi`: Python web framework for building API gateways
- `uvicorn`: ASGI server for running FastAPI applications
- `prometheus_client`: Prometheus metrics client for observability
- `opentelemetry`: OpenTelemetry for distributed tracing
- `kubernetes.client`: Kubernetes Python client for orchestration
- `httpx`: HTTP client library for service communication
- `redis`: Redis client for rate limiting and caching
- `pydantic`: Data validation library for API request/response models
- `grpc`: gRPC framework for high-performance service communication
- `docker`: Docker SDK for container management


## Anatomy of a Production LLM Serving System

In the previous chapters, we explored how to train large models at scale using distributed techniques like DDP, FSDP, DeepSpeed, and Megatron-LM. We also examined inference engines like vLLM and SGLang that optimize single-node and multi-node inference. Now it's time to put these pieces together into a complete production system.

AI model serving encompasses a broad spectrum of workloads. Traditional machine learning models—gradient boosted trees, linear models, and small neural networks—are typically served on CPUs with frameworks like TensorFlow Serving or Triton Inference Server. Computer vision models for image classification or object detection can also run efficiently on CPUs using optimized runtimes like ONNX Runtime, which has powered production CV workloads for years; GPUs are only necessary for the larger models or when throughput requirements are higher. These workloads have relatively predictable latency since input sizes are fixed. Diffusion models for image generation (Stable Diffusion, DALL-E) present unique challenges with their iterative denoising process, requiring careful batching strategies and often benefiting from techniques like classifier-free guidance caching.

LLM serving, however, has its own distinct characteristics that set it apart from these other workloads. The autoregressive nature of text generation means that output length is unpredictable—a simple "yes" or "no" question might generate 2 tokens, while a code generation request might produce 2,000. This variability makes batching and resource allocation fundamentally different from fixed-output models. The KV cache grows linearly with sequence length, creating memory pressure that doesn't exist in traditional ML serving. And the streaming nature of chat applications means users expect to see tokens as they're generated, not just a final response.

This chapter focuses specifically on LLM serving, building on the inference engines we covered in Chapters 6 and 7. The architectural patterns we discuss—routing, load balancing, observability—apply broadly to AI serving, but the specific implementations and trade-offs are tailored to the unique demands of large language models.

A production LLM serving system is more than just a model running on a GPU. It's a complex distributed system with multiple components working together to provide reliable, scalable, and cost-effective inference services. Understanding this architecture is essential before diving into specific deployment strategies.

The core components of a production serving stack include:

**Tokenizer Service.** This is a stateless, lightweight service that handles text tokenization and detokenization. Because it's stateless and CPU-bound, it can be scaled independently from the GPU-heavy model runners. The latency requirement is typically under 10ms—fast enough that it doesn't become a bottleneck in the request pipeline.

**Model Runner.** This is the heart of the system: a stateful, GPU-backed inference engine that manages model loading, KV cache, and continuous batching. In production, you'll typically use vLLM or SGLang (covered in Chapters 6 and 7) as the model runner, but the architectural principles apply regardless of which engine you choose.

**API Gateway.** The gateway sits between clients and the backend services, handling request routing, authentication, rate limiting, and load balancing. It's the single entry point that abstracts away the complexity of multiple model runners and routing decisions.

**Monitoring and Observability.** Production systems need visibility into what's happening. This includes metrics collection (typically with Prometheus), distributed tracing (OpenTelemetry), structured logging, and alerting. Without observability, debugging production issues becomes nearly impossible.

![Production LLM serving architecture.](img/serving_architecture.png){#fig:serving-architecture .block width=80% align=center}

Figure~\ref{fig:serving-architecture} shows how these components fit together. Clients send requests to the API Gateway, which routes them to the appropriate model runner based on the requested model and current load. The tokenizer service handles text-to-token conversion, and all components report metrics and traces to the observability stack.

The complete implementation of these components is available in `code/basic/`. The examples use Llama-2-7B-Chat as the default model, which requires a GPU with at least 16GB VRAM (e.g., NVIDIA RTX 4090, A100, or H100). For machines with less VRAM, you can substitute a smaller model like TinyLlama-1.1B or Qwen2-0.5B by modifying the model name in the code.

To try them out, first install the dependencies:

```bash
pip install fastapi uvicorn httpx pydantic transformers vllm
```

Then start the tokenizer service:

```bash
cd code/basic
uvicorn tokenizer_service:app --host 0.0.0.0 --port 8001
```

In another terminal, start the model runner (this will download and load the model):

```bash
# For GPU with 16GB+ VRAM (default: Llama-2-7B-Chat)
python model_runner.py

# For smaller GPUs, edit model_runner.py to use a smaller model:
# self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

In a third terminal, start the API gateway:

```bash
uvicorn api_gateway:app --host 0.0.0.0 --port 8000
```

You can then send requests to the gateway:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_tokens": 100}'
```

The tokenizer service (`code/basic/tokenizer_service.py`) demonstrates a FastAPI-based service for text-to-token conversion. The model runner (`code/basic/model_runner.py`) wraps vLLM for inference. The API gateway (`code/basic/api_gateway.py`) shows routing, rate limiting, and request forwarding.

## Request Routing and Traffic Management

In production systems, you often need to serve multiple models simultaneously, route requests intelligently, and balance load across instances. Beyond basic routing, you also need strategies for safely rolling out new model versions and running experiments. This section covers the complete traffic management picture: routing, load balancing, canary deployments, and A/B testing.

### Routing Strategies

There are three common approaches to routing requests in multi-model deployments:

**Feature-based routing** examines the request content to determine which model is best suited. For example, code-related prompts might route to Code Llama, while general chat goes to a conversational model. This approach works well when different models have distinct strengths, but requires careful keyword matching or classification logic.

**A/B traffic splits** use consistent hashing to assign users to model variants. The key insight is using the user ID (or session ID) as the hash input, ensuring the same user always gets the same model. This consistency is crucial for meaningful A/B comparisons—if users bounced between models randomly, you couldn't attribute performance differences to the model itself.

**Dynamic model selection** considers runtime factors like current load, latency budgets, and cost constraints. A cost-sensitive request might route to a smaller, cheaper model, while a latency-critical request goes to the fastest available option. This approach requires tracking model performance metrics in real-time.

The implementation of all three strategies is available in `code/basic/routing.py`. The `FeatureBasedRouter`, `ABRouter`, and `DynamicRouter` classes demonstrate these patterns.

### Load Balancing

Once you've decided which model handles a request, you need to choose which instance of that model receives it. The three standard approaches are:

**Round-robin** distributes requests evenly across instances by cycling through them in order. It's simple and works well when instances have similar capacity and requests have similar cost.

**Least connections** routes to the instance with the fewest active requests. This naturally handles varying request durations—slow requests don't cause one instance to fall behind while others sit idle.

**Weighted balancing** assigns different capacities to instances, useful when you have heterogeneous hardware (e.g., some instances on A100s, others on H100s).

All three balancers, plus a health checker that integrates with them, are implemented in `code/basic/load_balancer.py` and `code/basic/health_check.py`.

### Canary Deployments and A/B Testing

Beyond routing to existing models, you need strategies for safely introducing new model versions. Canary deployments allow you to gradually roll out new models while monitoring for issues. A/B testing enables comparing model performance in production. Both techniques are essential for safe, data-driven model updates.

**Canary Deployment.** The idea behind canary deployment is simple: instead of switching all traffic to a new model at once, you start by sending a small percentage (say, 10%) to the new "canary" model while the rest continues to the stable version. You monitor both versions, comparing error rates and latency. If the canary performs well, you gradually increase its traffic share. If it performs poorly, you roll back immediately with minimal user impact. The key metrics to track are error rate and latency—a reasonable promotion policy might allow the canary to have up to 10% higher error rate and 20% higher latency than the stable version, with rollback triggered if the canary's error rate exceeds twice the stable version's rate.

**Traffic Shifting.** This is the mechanism for gradually moving users from stable to canary. A typical progression might be: 10% → 25% → 50% → 75% → 100%. At each step, you wait for enough requests to accumulate (statistical significance) before deciding whether to proceed or roll back.

**A/B Testing.** A/B testing differs from canary deployment in its goal: canaries are about safe rollouts, while A/B tests are about comparing alternatives to make data-driven decisions. An A/B test might compare two different models, two different prompt templates, or two different inference configurations. The critical requirement is consistent assignment—the same user must always see the same variant, achieved through consistent hashing of the user ID.

The complete implementation of canary deployment, traffic shifting, and A/B testing is available in `code/basic/canary.py`. Here's how to use these classes in practice:

```python
from canary import CanaryDeployment, TrafficShifter, ABTestFramework, ABTestConfig

# Example 1: Canary deployment for a new model version
canary = CanaryDeployment(
    stable_model="llama-2-7b-v1",
    canary_model="llama-2-7b-v2",
    traffic_percent=0.1  # Start with 10% to canary
)

# Route a request
model = canary.route({"prompt": "Hello"})  # Returns stable or canary model
# After getting response, record metrics
canary.record_metrics(model, latency=0.15, error=False)

# Check if canary should be promoted or rolled back
if canary.should_promote():
    print("Canary performing well, increase traffic")
elif canary.should_rollback():
    print("Canary failing, rolling back")

# Example 2: Gradual traffic shifting
shifter = TrafficShifter("model-v1", "model-v2")
shifter.increase_traffic()  # 0% -> 10%
shifter.increase_traffic()  # 10% -> 25%
# ... continue based on metrics

# Example 3: A/B testing two models
ab = ABTestFramework()
ab.register_test(ABTestConfig(
    test_name="model_comparison",
    variants={"llama-7b": 0.5, "mistral-7b": 0.5},
    metrics=["latency", "quality_score"]
))

# Assign user to variant (consistent across requests)
variant = ab.assign_variant("model_comparison", user_id="user123")
# Record metrics after serving
ab.record_metric("model_comparison", variant, "latency", 0.12)
# Get aggregated results
results = ab.get_results("model_comparison")
```

These patterns integrate with the API gateway—in production, you'd wire the routing logic into your request handling pipeline.

## Operations: Observability, Reliability, and Cost

Beyond traffic management, production systems require operational capabilities: monitoring system health, handling failures gracefully, and optimizing costs. These cross-cutting concerns apply to every component in the serving stack.

### Observability

In a distributed LLM serving system, a single request might touch the API gateway, tokenizer service, and model runner—without proper observability, debugging issues becomes nearly impossible. Production LLM serving requires three types of observability:

**Distributed tracing** with OpenTelemetry tracks requests as they flow through multiple services. Each service creates "spans" that record timing and metadata, linked together by a trace ID that propagates through HTTP headers. When a request is slow, you can see exactly which service contributed the latency.

**Metrics collection** with Prometheus tracks aggregate statistics: request counts, latency histograms, error rates, and resource utilization. Unlike traces (which are sampled), metrics capture every request, making them essential for alerting and SLO monitoring. Key metrics for LLM serving include requests per second by model, latency percentiles (p50, p95, p99), active request count, and GPU utilization.

**Structured logging** captures detailed information about individual requests in a machine-parseable format (typically JSON). Unlike traditional logs, structured logs can be queried and aggregated—for example, finding all requests for a specific user that took longer than 5 seconds.

The complete implementation of all three is available in `code/basic/observability.py`.

### Reliability and Fault Tolerance

**Cold start mitigation.** LLM inference has significant cold start latency—loading a model and warming up the GPU can take 30-60 seconds. Two strategies help: warmup on startup (sending dummy requests through the model immediately after loading) and keep-alive requests (periodic dummy requests to prevent the model from going cold during idle periods).

**Autoscaling.** Request-based autoscaling adjusts replica count based on traffic. Key parameters include target RPS, scale-up threshold (typically 120% of target), scale-down threshold (typically 50% of target), and cooldown period. For LLM serving, be conservative with scale-down—spinning up a new GPU instance takes minutes, so it's better to have slightly excess capacity than to be caught short.

**Backpressure.** When traffic exceeds capacity, a bounded request queue provides backpressure—when the queue is full, new requests are rejected immediately with a 503 error rather than timing out after a long wait. This gives clients a clear signal to retry later or route to a different backend.

### Cost Optimization

GPU instances are expensive, so cost optimization matters. **Spot instances** (or preemptible VMs) cost 60-90% less than on-demand but can be terminated with short notice—a typical strategy uses 50% spot instances for baseline capacity, with on-demand instances absorbing traffic when spot instances are preempted. **Model selection** routes cost-sensitive requests to smaller, cheaper models; a 7B model might cost half as much per token as a 13B model, and for many use cases the quality difference doesn't justify the cost.

The implementation of warmup, autoscaling, request queuing, and cost-optimized routing is available in `code/basic/fault_tolerance.py`.

## Deploying LLM Serving on Kubernetes

The concepts we've covered so far—routing, load balancing, canary deployments, observability, and fault tolerance—are platform-agnostic patterns. You could implement them on bare metal servers, with Docker Compose, or on any cloud platform. However, Kubernetes (K8s) has emerged as the dominant platform for production LLM serving.

Kubernetes^[Kubernetes official site: \url{https://kubernetes.io/}] is an open-source container orchestration system originally developed by Google and now maintained by the Cloud Native Computing Foundation (CNCF). At its core, Kubernetes manages containerized workloads across a cluster of machines, handling scheduling, scaling, networking, and storage. You describe your desired state in YAML manifests—how many replicas of a service you want, how much CPU and memory each needs, how they should be exposed to the network—and Kubernetes continuously works to make the actual state match your desired state. This declarative model, combined with self-healing capabilities (automatically restarting failed containers, rescheduling workloads when nodes die), makes Kubernetes well-suited for production systems that need high availability.

For LLM serving specifically, Kubernetes provides native primitives for many of the patterns we've discussed:

| Concept | Kubernetes Primitive | In Practice |
|---------------|----------------------|---------------------------|
| Load Balancing | Services, Ingress | Model-aware routing via gateway |
| Autoscaling | HPA^[Horizontal Pod Autoscaler: \url{https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/}], KEDA^[Kubernetes Event-driven Autoscaling: \url{https://keda.sh/}] | Scale on requests-per-second or queue depth |
| Health Checks | Liveness/Readiness Probes | `/health`, `/ready` endpoints |
| Canary Deployments | Ingress traffic splitting | Weighted routing between model versions |
| Observability | Prometheus, OpenTelemetry | Latency, throughput, GPU metrics |
| Fault Tolerance | Pod restart, PDB^[PodDisruptionBudget: \url{https://kubernetes.io/docs/tasks/run-application/configure-pdb/}] | Graceful shutdown with request draining |
| GPU Scheduling | Device plugin, requests/limits | `nvidia.com/gpu: 1` in pod spec |

Rather than implementing these patterns from scratch, Kubernetes lets you declare your desired state and handles the implementation details. This declarative approach—combined with a rich ecosystem of operators and tools—is why production LLM serving stacks like vLLM Production Stack and llm-d are built on Kubernetes.

We'll explore Kubernetes-based LLM serving in two steps. First, we'll set up a local development environment using k3d^[k3d - k3s in Docker: \url{https://k3d.io/}], a lightweight Kubernetes distribution that runs entirely in Docker. This gives us a safe playground to experiment with configurations before deploying to production. Then, we'll deploy llm-d^[llm-d - Production LLM Serving on Kubernetes: \url{https://llm-d.ai/}], a production-ready serving stack that implements all the patterns we've discussed—routing, autoscaling, observability, and fault tolerance—as Kubernetes-native resources.

### Local Development with k3d

k3d wraps k3s (a lightweight Kubernetes distribution) inside Docker containers, giving you a fully functional Kubernetes cluster in minutes. It's lightweight (no VMs needed), supports GPU passthrough, and produces manifests that work unchanged on production clusters. This makes it ideal for developing and testing LLM serving configurations before deploying to production.

The complete k3d setup scripts are available in `code/k3d/`. The setup involves three main steps: installing prerequisites, building a custom GPU-enabled k3s image, and creating the cluster.

```bash
cd code/k3d

# Step 1: Install prerequisites (NVIDIA Container Toolkit, k3d)
./install-prerequisites.sh

# Step 2: Build custom k3s-cuda image
./build.sh

# Step 3: Create cluster with GPU support
./create-cluster.sh
```

The `install-prerequisites.sh` script checks for NVIDIA drivers, installs the NVIDIA Container Toolkit, and installs k3d. The `build.sh` script creates a custom k3s image with CUDA and NVIDIA runtime support—necessary because the default k3s image doesn't include GPU support.

The custom image combines k3s with CUDA and the NVIDIA Container Toolkit. The key files are `code/k3d/Dockerfile` and `code/k3d/device-plugin-daemonset.yaml`. The Dockerfile uses a multi-stage build to copy k3s binaries into an NVIDIA CUDA base image, then installs the container toolkit and configures containerd to use the NVIDIA runtime. The device plugin manifest is automatically deployed when the cluster starts, making GPUs visible to Kubernetes as `nvidia.com/gpu` resources.

Build the image by running `./build.sh` from the `code/k3d/` directory, or manually:

```bash
cd code/k3d
export DOCKER_BUILDKIT=1
docker build -t k3s-cuda:v1.33.6-cuda-12.2.0 .
```

The build requires Docker BuildKit for the `--exclude` flag in the multi-stage copy. If you encounter issues, ensure `docker buildx` is available.

### Creating a 2-Node GPU Cluster

Figure \ref{fig:k3d-architecture} shows the architecture of a k3d GPU cluster. The host machine runs Docker Engine, which contains the k3d network with two nodes: a control plane (server-0) running Kubernetes control plane services, and a worker node (agent-0) running application workloads like vLLM pods. Both nodes use the custom k3s-cuda image and have GPU passthrough via `--gpus=all`. The NVIDIA device plugin runs as a DaemonSet, exposing physical GPUs to Kubernetes as `nvidia.com/gpu` resources.

![k3d GPU cluster architecture showing host machine, Docker Engine, k3d network with control plane and worker nodes, GPU passthrough, and volume mounts.](img/k3d_architecture.png){#fig:k3d-architecture}

The `create-cluster.sh` script creates a 2-node cluster (1 control-plane + 1 worker) with GPU passthrough. You can also create the cluster manually with custom options:

```bash
# Basic cluster with all GPUs
k3d cluster create mycluster-gpu \
  --image k3s-cuda:v1.33.6-cuda-12.2.0 \
  --gpus=all \
  --servers 1 --agents 1

# With model directory mounted
k3d cluster create mycluster-gpu \
  --image k3s-cuda:v1.33.6-cuda-12.2.0 \
  --gpus=all \
  --servers 1 --agents 1 \
  --volume /path/to/models:/models

# With specific GPUs only
k3d cluster create mycluster-gpu \
  --image k3s-cuda:v1.33.6-cuda-12.2.0 \
  --gpus "device=0,1" \
  --servers 1 --agents 1
```

After creation, k3d automatically configures kubectl. Verify the cluster:

```bash
kubectl get nodes
# NAME                         STATUS   ROLES           AGE   VERSION
# k3d-mycluster-gpu-server-0   Ready    control-plane   30s   v1.33.6+k3s1
# k3d-mycluster-gpu-agent-0    Ready    <none>          25s   v1.33.6+k3s1

kubectl describe nodes | grep nvidia.com/gpu
```

You should see `nvidia.com/gpu: N` in the output, where N is the number of GPUs. To test GPU access end-to-end, use the provided test script:

```bash
cd code/k3d
./verify-gpu.sh
```

This runs a test pod that executes `nvidia-smi` inside the cluster, confirming that GPUs are accessible to Kubernetes workloads.

### Deploying vLLM on k3d

With the GPU cluster ready, deploying vLLM is straightforward. The `code/k3d/vllm/` directory contains ready-to-use deployment manifests for different models.

**Quick Start:**

```bash
cd code/k3d/vllm

# For gated models (like Llama), create a Hugging Face token secret first
kubectl create secret generic hf-token-secret --from-literal=token="$HF_TOKEN"

# Deploy Llama-3.2-1B (requires ~8GB GPU memory)
kubectl apply -f llama-3.2-1b.yaml

# Or deploy Phi-tiny-MoE (smaller, good for testing)
./deploy-phi-tiny-moe.sh
```

The deployment manifests handle GPU resource requests, health probes, model caching, and service exposure. Monitor the deployment:

```bash
kubectl get pods -l app=vllm -w
kubectl logs -l app=vllm --follow
```

Once running, test the OpenAI-compatible API:

```bash
kubectl port-forward svc/vllm-llama-32-1b-service 8000:8000 &

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Key Configuration Options:**

The vLLM deployment manifests demonstrate important configuration patterns:

- **GPU Memory**: `--gpu-memory-utilization 0.2` reserves 20% of GPU memory, allowing multiple models on shared GPUs
- **Health Probes**: Long `initialDelaySeconds` (120-180s) accounts for model loading time
- **Model Caching**: Host volume mounts (`/models`) cache downloaded models across pod restarts
- **Shared Memory**: `/dev/shm` mount required for tensor parallel inference

For larger models or multi-GPU setups, adjust `--tensor-parallel-size` and GPU resource limits. See `code/k3d/README.md` for detailed configuration and troubleshooting.

### Cleanup

```bash
k3d cluster delete mycluster-gpu
docker rmi k3s-cuda:v1.33.6-cuda-12.2.0  # optional
```

### Additional k3d Examples

The `code/k3d/` directory contains additional examples:

- `sglang/`: SGLang deployment manifests
- `tensorrt/`: TensorRT-LLM deployment
- `gateway/`: API gateway configuration
- `manage-cluster-multi-models.sh`: Script for managing multiple model deployments

Refer to `code/k3d/README.md` for comprehensive documentation on all available configurations and deployment patterns.

<!-- The following sections are preserved for reference but the main deployment flow uses code/k3d/ -->

<!--
**Step 1: Create Persistent Volume Claim (Optional)**

A PVC is used to cache downloaded models, reducing startup time on subsequent deployments:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mistral-7b
  namespace: default
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  volumeMode: Filesystem
```

**Step 2: Create Hugging Face Token Secret (Optional)**

Only required if you're using gated models that require authentication:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
  namespace: default
type: Opaque
stringData:
  token: ""  # Add your Hugging Face token here
```

**Step 3: Create vLLM Deployment**

Deploy vLLM using the official image with a model from Hugging Face:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mistral-7b
  namespace: default
  labels:
    app: mistral-7b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mistral-7b
  template:
    metadata:
      labels:
        app: mistral-7b
    spec:
      runtimeClassName: nvidia
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: mistral-7b
      # vLLM needs shared memory for tensor parallel inference
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "2Gi"
      containers:
      - name: mistral-7b
        image: vllm/vllm-openai:latest
        command: ["/bin/sh", "-c"]
        args:
        - |
          vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
            --trust-remote-code \
            --enable-chunked-prefill \
            --max-num-batched-tokens 1024 \
            --gpu-memory-utilization 0.4 \
            --kv-cache-memory-bytes 20G \
            --port 9876
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
        ports:
        - containerPort: 9876
          name: http
        resources:
          limits:
            cpu: "10"
            memory: 20Gi
            nvidia.com/gpu: "1"
          requests:
            cpu: "2"
            memory: 6Gi
            nvidia.com/gpu: "1"
        volumeMounts:
        - mountPath: /root/.cache/huggingface
          name: cache-volume
        - name: shm
          mountPath: /dev/shm
        livenessProbe:
          httpGet:
            path: /health
            port: 9876
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 9876
          initialDelaySeconds: 60
          periodSeconds: 5
```

**Key Configuration Points:**

- **Image:** `vllm/vllm-openai:latest` - Official vLLM OpenAI-compatible API server
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3` - Downloaded from Hugging Face on first startup
- **GPU Memory Utilization:** `0.4` (40% of GPU memory) - Adjust based on available GPU memory and other workloads
- **KV Cache:** `20G` - Fixed KV cache size in absolute units (overrides percentage-based allocation)
- **Shared Memory:** Required for tensor parallel inference (`/dev/shm` with 2Gi limit)
- **Model Cache:** PVC stores downloaded models in `/root/.cache/huggingface`
- **Health Probes:** Configured with 60s initial delay to allow model loading

**Step 4: Create Kubernetes Service**

Expose the vLLM deployment:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mistral-7b
  namespace: default
spec:
  ports:
  - name: http-mistral-7b
    port: 9876
    protocol: TCP
    targetPort: 9876
  selector:
    app: mistral-7b
  sessionAffinity: None
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: mistral-7b-nodeport
  namespace: default
spec:
  type: NodePort
  selector:
    app: mistral-7b
  ports:
  - port: 9876
    targetPort: 9876
    nodePort: 30082
    name: http
```

**Step 5: Deploy and Monitor**

Apply the configuration:

```bash
# Apply all resources
kubectl apply -f vllm-mistral-7b.yaml

# Monitor pod status (model download may take several minutes)
kubectl get pods -l app=mistral-7b -w

# Watch logs to see model download and server startup
kubectl logs -l app=mistral-7b --follow
```

**Step 6: Test the Deployment**

Once the pod is ready and the model has loaded:

```bash
# Port-forward to access the service
kubectl port-forward svc/mistral-7b 9876:9876

# Test health endpoint
curl http://localhost:9876/health

# Test completions endpoint
curl http://localhost:9876/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
  }'

# Test chat completions
curl http://localhost:9876/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# List available models
curl http://localhost:9876/v1/models
```

**Alternative: Access via NodePort**

```bash
# Get node IP
NODE_IP=$(kubectl get node k3d-mycluster-gpu-server-0 -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')

# Test via NodePort
curl http://$NODE_IP:30082/health
curl http://$NODE_IP:30082/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mistral-7B-Instruct-v0.3", "prompt": "What is AI?", "max_tokens": 50}'
```

**Using Different Models**

To use a different model, simply change the model name in the deployment args:

```yaml
args:

- |
  vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --trust-remote-code \
    --port 9876
```

**Troubleshooting:**

**Issue: Pod stuck in ContainerCreating**
- Check if PVC is bound: `kubectl get pvc`
- Verify storage class exists: `kubectl get storageclass`
- For k3d, the `local-path` storage class should be available

**Issue: Model download fails**
- Check network connectivity from pod: `kubectl exec <pod-name> -- curl -I https://huggingface.co`
- Verify Hugging Face token if using gated models: `kubectl get secret hf-token-secret -o jsonpath='{.data.token}' | base64 -d`
- Check logs for specific error: `kubectl logs -l app=mistral-7b`

**Issue: Out of memory errors**
- Reduce model size or increase pod memory limits
- Adjust `--gpu-memory-utilization` parameter (e.g., reduce from 0.4 to 0.2-0.3 if GPU is shared with other processes)
- Reduce `--kv-cache-memory-bytes` (e.g., from `20G` to `10G` or `15G`)
- Check available GPU memory: `nvidia-smi` on the host
- Consider using a smaller model variant

**Issue: Health probe failures**
- Increase `initialDelaySeconds` if model loading takes longer
- Check if server is actually running: `kubectl exec <pod-name> -- curl http://localhost:9876/health`

**Performance Tuning:**

For better performance, you can adjust vLLM parameters:

```yaml
args:

- |
  vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.4 \
    --kv-cache-memory-bytes 20G \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --port 9876
```

**Memory Configuration Notes:**

- `--gpu-memory-utilization`: Fraction (0.0-1.0) of total GPU memory to use. Lower values (e.g., 0.4 = 40%) allow sharing GPU with other processes. For a 140GB GPU, 0.4 means ~56GB total allocation.
- `--kv-cache-memory-bytes`: Absolute KV cache size in human-readable format (`20G`, `30G`, etc.). When specified, overrides percentage-based KV cache allocation from `gpu-memory-utilization`. Useful for precise memory control in multi-tenant environments.
- **Memory breakdown example** (140GB GPU, 0.4 utilization, 20G KV cache):
  - Total available: 56GB (0.4 × 140GB)
  - Model weights: ~14GB (Mistral-7B)
  - KV cache: 20GB (fixed)
  - Remaining: ~22GB (for activations, batching, etc.)
- **When to adjust**: Reduce `gpu-memory-utilization` to 0.2-0.3 if GPU is heavily shared. Reduce `kv-cache-memory-bytes` if you need more memory for other operations or have limited GPU memory.

**Multi-GPU Configuration:**

For larger models requiring multiple GPUs:

```yaml
args:

- |
  vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --tensor-parallel-size 2 \
    --port 9876
resources:
  limits:
    nvidia.com/gpu: "2"
  requests:
    nvidia.com/gpu: "2"
```

**Reference:**

- Official vLLM Kubernetes documentation: https://docs.vllm.ai/en/stable/deployment/k8s/#deployment-with-gpus
- vLLM configuration options: https://docs.vllm.ai/en/stable/serving/server_args.html

### Deploying a GPU Model Serving Example

Now let's deploy a simple GPU-enabled inference server that can serve as a foundation for LLM model serving.

**Step 1: Create Deployment Manifest**

Create `gpu-inference-server.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-inference-server
  labels:
    app: inference-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: inference-server
  template:
    metadata:
      labels:
        app: inference-server
    spec:
      runtimeClassName: nvidia
      containers:
      - name: inference
        image: nvidia/cuda:12.2.0-base-ubuntu22.04
        command:
        - /bin/bash
        - -c
        - |
          apt-get update && apt-get install -y python3 python3-pip curl || true
          python3 << 'PYTHON_EOF'
          from http.server import HTTPServer, BaseHTTPRequestHandler
          import json
          import subprocess
          
          class GPUHandler(BaseHTTPRequestHandler):
              def do_GET(self):
                  if self.path == '/health':
                      self.send_response(200)
                      self.send_header('Content-type', 'application/json')
                      self.end_headers()
                      self.wfile.write(json.dumps({"status": "healthy"}).encode())
                  elif self.path == '/gpu':
                      try:
                          result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], 
                                                capture_output=True, text=True, timeout=5)
                          gpu_info = result.stdout.strip().split(', ')
                          response = {
                              "gpu_name": gpu_info[0],
                              "memory_total_mb": gpu_info[1],
                              "memory_used_mb": gpu_info[2],
                              "utilization_percent": gpu_info[3]
                          }
                      except:
                          response = {"error": "Could not query GPU"}
                      
                      self.send_response(200)
                      self.send_header('Content-type', 'application/json')
                      self.end_headers()
                      self.wfile.write(json.dumps(response).encode())
                  else:
                      self.send_response(404)
                      self.end_headers()
              
              def log_message(self, format, *args):
                  pass
          
          server = HTTPServer(('0.0.0.0', 8000), GPUHandler)
          print("GPU Inference Server running on port 8000")
          server.serve_forever()
          PYTHON_EOF
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
          name: http
---
apiVersion: v1
kind: Service
metadata:
  name: inference-server-service
spec:
  selector:
    app: inference-server
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: inference-server-nodeport
spec:
  type: NodePort
  selector:
    app: inference-server
  ports:
  - port: 8000
    targetPort: 8000
    nodePort: 30080
    name: http
```

**Step 2: Deploy the Service**

```bash
# Deploy
kubectl apply -f gpu-inference-server.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod -l app=inference-server --timeout=120s

# Check pod status
kubectl get pods -l app=inference-server

# Note: The Python server inside the pod needs time to install packages and start
# Wait an additional 60-90 seconds after pod is Ready, then check logs:
kubectl logs -l app=inference-server | tail -5
# You should see "GPU Inference Server running on port 8000" when ready
```

**Step 3: Access the Service**

Set up port-forward to access the service:

```bash
# Ensure pod is fully started (wait for Python server to be ready)
sleep 60  # Give time for apt-get and Python server startup

# Port-forward in background
kubectl port-forward svc/inference-server-service 8000:8000 > /tmp/port-forward.log 2>&1 &
sleep 3  # Wait for port-forward to establish

# Test health endpoint
curl --max-time 3 http://localhost:8000/health

# Test GPU endpoint
curl --max-time 3 http://localhost:8000/gpu
```

Expected responses:

```json
# /health
{"status": "healthy"}

# /gpu
{
    "gpu_name": "NVIDIA H200",
    "memory_total_mb": "143771 MiB",
    "memory_used_mb": "1 MiB",
    "utilization_percent": "0 %"
}
```

**Alternative: Access via NodePort**

You can also access the service via NodePort:

```bash
# Get node IP
NODE_IP=$(kubectl get node k3d-mycluster-gpu-server-0 -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')

# Access via NodePort
curl http://$NODE_IP:30080/health
curl http://$NODE_IP:30080/gpu
```

### GPU Allocation Options

k3d supports flexible GPU allocation:

```bash
# Use all GPUs
--gpus=all

# Use specific GPU (single)
--gpus "device=4"

# Use multiple specific GPUs
--gpus "device=4,5"

# Use GPU range (if supported)
--gpus "device=4-7"  # GPUs 4, 5, 6, 7
```

To see available GPUs:

```bash
nvidia-smi --query-gpu=index,gpu_name --format=csv
```

### Troubleshooting

**Issue: Device plugin reports "No devices found"**

- Ensure the custom k3s image was built correctly
- Verify NVIDIA Container Toolkit is installed on the host
- Check that `--gpus` flag was used when creating the cluster
- Review device plugin logs: `kubectl logs -n kube-system -l name=nvidia-device-plugin-ds`

**Issue: Pod cannot access GPU**

- Verify `runtimeClassName: nvidia` is set in pod spec
- Check GPU resource requests/limits are specified
- Ensure device plugin DaemonSet is running: `kubectl get ds -n kube-system nvidia-device-plugin-daemonset`

**Issue: Port-forward not working**

- Ensure the pod is fully started (check logs: `kubectl logs -l app=inference-server`)
- Wait for Python server to finish installing packages and start listening (60-90 seconds after pod is Ready)
- Verify server is listening inside pod: `kubectl exec <pod-name> -- curl -s http://localhost:8000/health`
- Restart port-forward: `pkill -f "port-forward"` then recreate
- Use NodePort as alternative access method (more reliable)

**Issue: kubectl connection refused (localhost:8080)**

- The kubeconfig may have an incorrect server address pointing to `localhost:8080`
- Fix by updating the server address:
  ```bash
  KUBE_SERVER=$(kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-mycluster-gpu")].cluster.server}')
  kubectl config set-cluster k3d-mycluster-gpu --server=$(echo $KUBE_SERVER | sed 's/0.0.0.0/127.0.0.1/')
  ```
- Ensure `export KUBECONFIG=$HOME/.kube/config` is set

**Issue: Build fails with "unknown flag: --exclude"**

- Enable Docker BuildKit: `export DOCKER_BUILDKIT=1`
- Install buildx: `docker buildx version` or install the plugin
- Use `docker buildx build` instead of `docker build`

### Cleaning Up

To remove the cluster:

```bash
# Delete cluster
k3d cluster delete mycluster-gpu

# Remove custom image (optional)
docker rmi k3s-cuda:v1.33.6-cuda-12.2.0
```

### Serving Local Models with vLLM

To serve a model from your local filesystem (e.g., `/raid/models/Phi-tiny-MoE-instruct/`), you need to:

1. **Mount the model directory** into the k3d cluster
2. **Deploy vLLM** with the mounted model path
3. **Expose the service** for API access

**Step 1: Ensure Cluster Has Model Volume Mount**

The model directory must be accessible to pods. **Important:** When using k3d, the `hostPath` in your pod YAML must point to the path **inside the k3d node container**, not the host path.

**Create Cluster with Volume Mount:**

```bash
# Delete existing cluster (if it exists)
k3d cluster delete mycluster-gpu 2>/dev/null || true

# Create cluster with model directory mounted
# The --volume flag mounts /raid/models (host) to /models (inside node)
k3d cluster create mycluster-gpu \
  --image k3s-cuda:v1.33.6-cuda-12.4.1 \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume /raid/models:/models

# Verify mount by checking inside the node container
docker exec k3d-mycluster-gpu-server-0 ls -la /models/Phi-tiny-MoE-instruct/ | head -5
```

**Important Note on hostPath:**

In your pod YAML, use `/models` (the path inside the k3d node) as the hostPath, **not** `/raid/models`:

```yaml
volumes:

- name: models
  hostPath:
    path: /models        # ✅ Correct: path inside k3d node
    type: Directory
    # NOT /raid/models  # ❌ Wrong: this path doesn't exist in the node container
```

**Verify Volume Access:**

```bash
# Test by creating a temporary pod
kubectl run test-mount --image=busybox --rm -i --restart=Never -- \
  ls -la /models/Phi-tiny-MoE-instruct/ 2>&1 | head -5

# Should show model files if volume mount is correct
```

**Step 2: Create vLLM Deployment**

Create `vllm-phi-tiny-moe.yaml`:

**Important Image Distinction:**

There are **two different images** used in this setup:

1. **Cluster Node Image** (`k3s-cuda:v1.33.6-cuda-12.2.0`): 
   - Used when creating the k3d cluster with `--image k3s-cuda:...`
   - Provides GPU support for Kubernetes nodes
   - Built in "Step 3: Build the Custom Image" above
   - This is the Kubernetes distribution image

2. **Application Pod Image** (`vllm-dev:latest` or `vllm/vllm-openai:latest`):
   - Used for the vLLM application container running inside pods
   - Contains the vLLM Python application and dependencies
   - This is your application runtime image

**Use your custom built vLLM image** (`vllm-dev:latest`) which already has vLLM installed:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-phi-tiny-moe
  labels:
    app: vllm
    model: phi-tiny-moe
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
      model: phi-tiny-moe
  template:
    metadata:
      labels:
        app: vllm
        model: phi-tiny-moe
    spec:
      runtimeClassName: nvidia
      containers:
      - name: vllm-server
        image: vllm-dev:latest  # ✅ Use your custom built image (recommended)
        # Alternative: image: vllm/vllm-openai:latest  # Official vLLM image
        command:
        - python3
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model
        - /models/Phi-tiny-MoE-instruct
        - --host
        - "0.0.0.0"
        - --port
        - "9876"
        - --tensor-parallel-size
        - "1"
        - --gpu-memory-utilization
        - "0.9"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
        ports:
        - containerPort: 9876
          name: http
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
      volumes:
      - name: models
        hostPath:
          path: /models
          type: Directory
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-phi-tiny-moe-service
  labels:
    app: vllm
    model: phi-tiny-moe
spec:
  type: ClusterIP
  selector:
    app: vllm
    model: phi-tiny-moe
  ports:
  - port: 9876
    targetPort: 9876
    name: http
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-phi-tiny-moe-nodeport
  labels:
    app: vllm
    model: phi-tiny-moe
spec:
  type: NodePort
  selector:
    app: vllm
    model: phi-tiny-moe
  ports:
  - port: 9876
    targetPort: 9876
    nodePort: 30081
    name: http
```

**Step 3: Deploy vLLM**

```bash
# Apply the deployment
kubectl apply -f vllm-phi-tiny-moe.yaml

# Wait for pod to be ready (this may take a few minutes as vLLM loads the model)
kubectl wait --for=condition=Ready pod -l app=vllm,model=phi-tiny-moe --timeout=600s

# Check pod status
kubectl get pods -l app=vllm,model=phi-tiny-moe

# View logs to see model loading progress
kubectl logs -f -l app=vllm,model=phi-tiny-moe
```

**Step 4: Test the vLLM API**

Once the pod is ready, test the OpenAI-compatible API. vLLM provides an OpenAI-compatible API server.

**Method 1: Port-Forward (Recommended for Local Testing)**

```bash
# Start port-forward in background
kubectl port-forward svc/vllm-phi-tiny-moe-service 9876:9876 > /tmp/vllm-port-forward.log 2>&1 &

# Wait a moment for port-forward to establish
sleep 2

# Test health endpoint
curl --max-time 5 http://localhost:9876/health

# Expected response:
# {"status":"ok"}
```

**Test Completions Endpoint:**

```bash
# Simple completion request
curl --max-time 30 http://localhost:9876/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Pretty-print JSON response
curl --max-time 30 -s http://localhost:9876/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "prompt": "Explain neural networks in one sentence.",
    "max_tokens": 50,
    "temperature": 0.5
  }' | python3 -m json.tool
```

**Test Chat Completions Endpoint:**

```bash
# Chat completion (if model supports chat format)
curl --max-time 30 http://localhost:9876/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Multi-turn conversation
curl --max-time 30 -s http://localhost:9876/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50
  }' | python3 -m json.tool
```

**List Available Models:**

```bash
# List models endpoint
curl --max-time 5 http://localhost:9876/v1/models

# Expected response shows available models
curl --max-time 5 -s http://localhost:9876/v1/models | python3 -m json.tool
```

**Method 2: Access via NodePort**

```bash
# Get node IP
NODE_IP=$(kubectl get node k3d-mycluster-gpu-server-0 -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
echo "Node IP: $NODE_IP"

# Test health endpoint via NodePort
curl --max-time 5 http://$NODE_IP:30081/health

# Test completions via NodePort
curl --max-time 30 http://$NODE_IP:30081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "prompt": "What is AI?",
    "max_tokens": 50
  }'
```

**Method 3: Direct Pod Access (For Debugging)**

```bash
# Get pod name
POD_NAME=$(kubectl get pod -l app=vllm,model=phi-tiny-moe -o jsonpath='{.items[0].metadata.name}')

# Port-forward directly to pod
kubectl port-forward pod/$POD_NAME 8001:8000 > /tmp/pod-port-forward.log 2>&1 &

# Test
curl --max-time 5 http://localhost:8001/health
```

**Complete Testing Script:**

Create a test script `test-vllm-api.sh`:

```bash
#!/bin/bash
set -e

API_URL="${1:-http://localhost:8000}"
echo "Testing vLLM API at: $API_URL"
echo ""

# Test health
echo "1. Testing health endpoint..."
curl --max-time 5 -s "$API_URL/health" | python3 -m json.tool || echo "Health check failed"
echo ""

# List models
echo "2. Listing available models..."
curl --max-time 5 -s "$API_URL/v1/models" | python3 -m json.tool || echo "List models failed"
echo ""

# Test completion
echo "3. Testing completion endpoint..."
curl --max-time 30 -s "$API_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 50,
    "temperature": 0.7
  }' | python3 -m json.tool || echo "Completion failed"
echo ""

# Test chat (if supported)
echo "4. Testing chat completion endpoint..."
curl --max-time 30 -s "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 30
  }' | python3 -m json.tool || echo "Chat completion failed"
```

Make it executable and run:

```bash
chmod +x test-vllm-api.sh

# Test via port-forward
./test-vllm-api.sh http://localhost:8000

# Or test via NodePort
NODE_IP=$(kubectl get node k3d-mycluster-gpu-server-0 -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')
./test-vllm-api.sh http://$NODE_IP:30081
```

**Step 5: Monitor GPU Usage**

Check GPU utilization:

```bash
# Exec into pod and check nvidia-smi
kubectl exec -it $(kubectl get pod -l app=vllm,model=phi-tiny-moe -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi

# Or check from host
nvidia-smi

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi
```

**Quick Test Script:**

For convenience, use the provided test script:

```bash
# Make sure port-forward is running
kubectl port-forward svc/vllm-phi-tiny-moe-service 9876:9876 > /tmp/vllm-port-forward.log 2>&1 &

# Run the test script
cd /home/fuhwu/workspace/distributedai/code/chapter9
./test-vllm-api.sh http://localhost:9876
```

The script will test:
1. Health endpoint
2. List models endpoint
3. Completion endpoint
4. Chat completion endpoint (if supported)

**Example API Responses:**

**Health Check:**
```json
{"status":"ok"}
```

**List Models:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "/models/Phi-tiny-MoE-instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vllm"
    }
  ]
}
```

**Completion Response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "/models/Phi-tiny-MoE-instruct",
  "choices": [
    {
      "text": "Machine learning is a subset of artificial intelligence...",
      "index": 0,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 50,
    "total_tokens": 55
  }
}
```

### Advanced vLLM Configuration

For production use, you may want to adjust vLLM parameters:

```yaml
args:

- --model
- /models/Phi-tiny-MoE-instruct
- --host
- "0.0.0.0"
- --port
- "8000"
- --tensor-parallel-size
- "1"                    # Increase for multi-GPU
- --gpu-memory-utilization
- "0.4"                  # Fraction (0.0-1.0) of GPU memory to use
- --kv-cache-memory-bytes
- "20G"                  # KV cache size in absolute units (20G, 30G, etc.)
- --max-model-len
- "4096"                 # Maximum sequence length
- --dtype
- "auto"                 # Auto-detect dtype
- --enable-prefix-caching
- "true"                 # Enable prefix caching
- --max-num-seqs
- "256"                  # Maximum concurrent sequences
- --max-num-batched-tokens
- "8192"                 # Maximum batched tokens
```

**Multi-GPU Configuration:**

For models that need multiple GPUs:

```yaml
args:

- --tensor-parallel-size
- "2"                    # Use 2 GPUs
env:

- name: CUDA_VISIBLE_DEVICES
  value: "0,1"           # Specify GPU IDs
resources:
  limits:
    nvidia.com/gpu: 2    # Request 2 GPUs
  requests:
    nvidia.com/gpu: 2
```

### Troubleshooting vLLM Deployment

**Issue: Pod fails to start / OOM errors**

- Increase memory limits: `memory: 64Gi` or higher
- Reduce `--gpu-memory-utilization` (e.g., from `0.4` to `0.2-0.3` for shared GPU environments)
- Reduce `--kv-cache-memory-bytes` (e.g., from `20G` to `10G-15G`)
- Check available GPU memory on host: `nvidia-smi` (ensure sufficient free memory)
- Check model size vs. available GPU memory

**Issue: Model not found**

- Verify volume mount: `kubectl describe pod <pod-name> | grep -A 5 Mounts`
- Check model path: `kubectl exec <pod-name> -- ls -la /models/`
- Ensure cluster was created with `--volume /raid/models:/models`
- **Important:** In pod YAML, use `path: /models` (not `/raid/models`) as the hostPath points to the path inside the k3d node container
- Test volume access: `docker exec k3d-mycluster-gpu-server-0 ls -la /models/Phi-tiny-MoE-instruct/`

**Issue: vLLM fails with "libcuda.so.1: cannot open shared object file"**

- This indicates the pod cannot access GPU devices
- Ensure you're using the **custom k3s image** with NVIDIA Container Toolkit: `k3s-cuda:v1.33.6-cuda-12.2.0`
- Verify `runtimeClassName: nvidia` is set in the pod spec
- Check that NVIDIA device plugin is running: `kubectl get pods -n kube-system | grep nvidia`
- If using standard k3s image, GPU workloads will not work - you must build and use the custom CUDA-enabled image

**Issue: Slow model loading**

- This is normal for large models (can take 2-5 minutes)
- Monitor logs: `kubectl logs -f <pod-name>`
- Check disk I/O: `iostat -x 1`

**Issue: API returns errors**

- Verify model is fully loaded (check logs for "Uvicorn running on")
- Test health endpoint first: `curl http://localhost:8000/health`
- Check pod logs for specific error messages

**Issue: Custom k3s image fails during cluster creation**

- If you see "exec /usr/bin/sh: no such file or directory" when creating cluster with custom image, the custom image is missing `/usr/bin/sh` which k3d needs for exec operations
- **Root cause:** The custom k3s-cuda image was built from CUDA base and may not have included all necessary shell binaries
- **Solutions:**
  1. **Fix the custom image:** Ensure `/usr/bin/sh` (or symlink to `/bin/sh`) exists in the custom image
  2. **Use standard k3s:** Use `rancher/k3s:v1.33.6-k3s1` and manually configure NVIDIA runtime (complex)
  3. **Use production K8s:** Use kubeadm, k0s, or managed Kubernetes for full GPU support
  4. **Workaround:** If you have a working custom image, ensure it includes: `/bin/sh`, `/usr/bin/sh`, and basic shell utilities

**Note:** If using your own custom vLLM image (e.g., `vllm-dev:latest`), update the deployment YAML:

```yaml
containers:

- name: vllm-server
  image: vllm-dev:latest  # Use your custom image instead of vllm/vllm-openai:latest
```

### Next Steps

With a working GPU-enabled Kubernetes cluster, you can:

1. **Deploy vLLM:** Use the patterns above to deploy vLLM for LLM inference
2. **Scale workloads:** Test horizontal pod autoscaling with GPU workloads
3. **Multi-model serving:** Deploy multiple model instances with different GPU allocations
4. **Production patterns:** Practice canary deployments, A/B testing, and observability
5. **Upgrade to llm-d:** Use the cluster to deploy llm-d (covered in the next section)

This local setup provides a safe environment to experiment with production patterns before deploying to cloud Kubernetes services.
-->

## Kubernetes Deployment with llm-d

While building custom serving stacks provides flexibility, production deployments often benefit from standardized solutions that handle the operational complexity of distributed inference. [llm-d](https://github.com/llm-d/llm-d) is an open-source project that provides production-ready Helm charts and deployment patterns for running LLM inference on Kubernetes with modern accelerators.

### What is llm-d?

llm-d is a comprehensive solution for deploying state-of-the-art inference performance on Kubernetes. It integrates industry-standard open technologies:

- **vLLM** as the default model server and engine
- **Inference Gateway (IGW)** as request scheduler and balancer
- **Kubernetes** as infrastructure orchestrator
- **Envoy Proxy** for load balancing and routing
- **NIXL** for fast interconnects (IB/RoCE RDMA, TPU ICI)

### Key Features

**1. Intelligent Inference Scheduling**

llm-d builds on IGW's pattern of leveraging Envoy proxy and extensible balancing policies to make customizable "smart" load-balancing decisions for LLMs:

- **Predicted latency balancing** (experimental): Predicts request latency and routes accordingly
- **Prefix-cache aware routing**: Routes requests to instances with hot KV cache
- **SLA-aware scheduling**: Routes based on service level agreements
- **Load-aware balancing**: Distributes load based on current instance capacity

**2. Prefill/Decode Disaggregation**

Reduces Time to First Token (TTFT) and provides more predictable Time Per Output Token (TPOT) by splitting inference:

- **Prefill servers**: Handle prompt processing
- **Decode servers**: Handle response generation
- **KV cache transfer**: Uses NIXL for efficient KV cache transfer over fast interconnects
- **Sidecar coordination**: Coordinates transactions via sidecar alongside decode instances

**3. Disaggregated Prefix Caching**

llm-d uses vLLM's KVConnector abstraction to configure a pluggable KV cache hierarchy:

- **Independent (N/S) caching**: Offloads KVs to local memory and disk
- **Shared (E/W) caching**: KV transfer between instances with shared storage
- **Global indexing**: Enables higher performance at operational complexity cost

**4. Variant Autoscaling**

Traffic- and hardware-aware autoscaler that:

- Measures capacity of each model server instance
- Derives load function considering different request shapes and QoS
- Assesses recent traffic mix (QPS, QoS, shapes)
- Calculates optimal mix of instances for prefill, decode, and latency-tolerant requests
- Enables HPA for SLO-level efficiency

### Hardware Support

llm-d directly tests and validates multiple accelerator types:

- **NVIDIA GPUs**: A100, L4, or newer
- **AMD GPUs**: MI250 or newer
- **Google TPUs**: TPU v5e or newer
- **Intel XPUs**: Data Center GPU Max (Ponte Vecchio) series or newer

### Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│              Kubernetes Cluster                  │
│                                                  │
│  ┌──────────────┐      ┌──────────────────┐    │
│  │   Envoy      │      │  Inference        │    │
│  │   Proxy      │──────│  Gateway (IGW)    │    │
│  │              │      │  (Scheduler)      │    │
│  └──────┬───────┘      └──────────────────┘    │
│         │                                        │
│         ├──────────────┬──────────────────┐    │
│         ▼              ▼                  ▼     │
│  ┌──────────┐   ┌──────────┐      ┌──────────┐ │
│  │ Prefill  │   │ Decode   │      │ Decode   │ │
│  │ Server 1 │   │ Server 1 │      │ Server 2 │ │
│  │ (vLLM)   │   │ (vLLM)   │      │ (vLLM)   │ │
│  └──────────┘   └──────────┘      └──────────┘ │
│         │              │                  │     │
│         └──────────────┴──────────────────┘    │
│                    │                            │
│                    ▼                            │
│         ┌──────────────────────┐               │
│         │   KV Cache Storage    │               │
│         │   (NIXL/NVMe)         │               │
│         └──────────────────────┘               │
└─────────────────────────────────────────────────┘
```

### Getting Started with llm-d

**Prerequisites:**

- Kubernetes 1.29+ cluster
- Accelerators capable of running large models
- Fast interconnects (NVLINK, IB/RoCE RDMA, TPU ICI, DCN)

**Installation:**

llm-d provides Helm charts for easy deployment:

```bash
# Add llm-d Helm repository
helm repo add llm-d https://llm-d.github.io/llm-d
helm repo update

# Install llm-d with inference scheduling
helm install llm-d llm-d/llm-d \
  --set inferenceGateway.enabled=true \
  --set vllm.enabled=true \
  --set vllm.model=meta-llama/Llama-3.1-70B-Instruct
```

**Configuration Example:**

```yaml
# values.yaml
inferenceGateway:
  enabled: true
  replicas: 2
  policy: cache_aware
  routing:
    prefixCacheAware: true
    latencyPrediction: true

vllm:
  enabled: true
  model: meta-llama/Llama-3.1-70B-Instruct
  tensorParallelSize: 4
  gpuMemoryUtilization: 0.9
  
prefillDecode:
  enabled: true
  prefill:
    replicas: 2
    resources:
      requests:
        nvidia.com/gpu: 4
  decode:
    replicas: 4
    resources:
      requests:
        nvidia.com/gpu: 4

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetQPS: 100
```

### Well-Lit Paths

llm-d provides three tested and benchmarked deployment paths:

**1. Intelligent Inference Scheduling**

Deploy vLLM behind the Inference Gateway to decrease serving latency and increase throughput:

```bash
helm install llm-d llm-d/llm-d \
  --set path=inference-scheduling \
  --set inferenceGateway.enabled=true
```

**Benefits:**

- Reduced latency through predicted latency balancing
- Higher throughput with prefix-cache aware routing
- Customizable scheduling policies

**2. Prefill/Decode Disaggregation**

Split inference into prefill and decode servers:

```bash
helm install llm-d llm-d/llm-d \
  --set path=prefill-decode \
  --set prefillDecode.enabled=true
```

**Benefits:**

- Reduced TTFT (Time to First Token)
- More predictable TPOT (Time Per Output Token)
- Better resource utilization for large models and long prompts

**3. Wide Expert-Parallelism**

Deploy very large MoE models with Data Parallelism and Expert Parallelism:

```bash
helm install llm-d llm-d/llm-d \
  --set path=expert-parallelism \
  --set model.type=moe \
  --set expertParallelism.enabled=true
```

**Benefits:**

- Reduced end-to-end latency for MoE models
- Increased throughput
- Efficient scaling over fast accelerator networks

### Monitoring and Observability

llm-d integrates with standard Kubernetes monitoring:

```yaml
monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
  metrics:
    - request_latency
    - throughput
    - gpu_utilization
    - kv_cache_hit_rate
```

### Best Practices

1. **Start with Inference Scheduling Path**
   - Simplest production-ready deployment
   - Good for most use cases
   - Easy to extend later

2. **Use Prefill/Decode for Large Models**
   - Especially effective for models like Llama-70B
   - Best for very long prompts
   - Requires fast interconnects

3. **Monitor KV Cache Hit Rates**
   - High hit rates indicate effective routing
   - Low hit rates may need cache warming
   - Adjust routing policies based on metrics

4. **Tune Autoscaling Parameters**
   - Set appropriate min/max replicas
   - Configure target QPS based on workload
   - Monitor scaling behavior

5. **Leverage Hardware-Specific Optimizations**
   - Use NVLINK for NVIDIA GPUs
   - Configure IB/RoCE for AMD GPUs
   - Optimize for TPU ICI when using TPUs

### Comparison with Custom Stacks

| Feature | Custom Stack | llm-d |
|---------|--------------|-------|
| Setup Time | Weeks | Hours |
| Operational Complexity | High | Low |
| Flexibility | High | Medium |
| Production Readiness | Requires extensive testing | Pre-tested |
| Hardware Support | Manual configuration | Multi-accelerator support |
| Autoscaling | Custom implementation | Built-in |
| KV Cache Management | Custom | Integrated |

### When to Use llm-d

**Use llm-d when:**

- You need production-ready deployment quickly
- You're deploying on Kubernetes
- You want standardized best practices
- You need multi-accelerator support
- You want prefill/decode disaggregation

**Consider custom stack when:**

- You need very specific customizations
- You're not using Kubernetes
- You have unique requirements not covered by llm-d



## Hands-On Examples: LLM serving in Kubernetes with k3d

This section demonstrates two practical examples of deploying LLM serving systems in Kubernetes with API Gateway routing:

1. **Example 1**: Deploying different models using the same inference engine (vLLM)
2. **Example 2**: Deploying the same model using different inference engines (vLLM vs SGLang)

### How Routing Works

This is the final routing solution that supports routing based on both model and inference engine. The gateway routes requests using both the `model` field and the `owned_by` field.

1. **Client sends request** to API Gateway with both `model` and `owned_by` fields:
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B-Instruct",
     "owned_by": "sglang",
     "messages": [...]
   }
   ```

2. **Gateway parses request** and extracts both the `model` field and the `owned_by` field from the JSON body.

3. **Gateway looks up service** using both fields:
   - `("meta-llama/Llama-3.2-1B-Instruct", "vllm")` → `vllm-llama-32-1b`
   - `("meta-llama/Llama-3.2-1B-Instruct", "sglang")` → `sglang-llama-32-1b`
   - `("meta-llama/Llama-3.2-1B-Instruct", None)` → `vllm-llama-32-1b` (default)

4. **Gateway forwards request** to the target service based on the inference engine specified.

5. **Gateway returns response** to client (with streaming support if requested).

### Example 1: Different Models, Same Inference Engine (vLLM)

This example demonstrates how to deploy multiple vLLM models in a Kubernetes cluster and use an API Gateway to provide a unified interface that automatically routes requests to the correct model based on the `model` field in the request.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' field
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              API Gateway (Unified Entry Point)              │
│  - Parses 'model' field from request body                   │
│  - Routes to appropriate vLLM service                       │
│  - Returns response to client                               │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  vLLM Service 1      │    │  vLLM Service 2     │
    │  (Llama-3.2-1B)      │    │  (Phi-tiny-MoE)     │
    │                      │    │                     │
    │ Pod: vllm-llama-32-1b│    │  Pod: vllm-phi-tiny │
    │ Service: vllm-llama- │    │  Service: vllm-phi- │
    │    32-1b:8000        │    │    tiny-moe:8000    │
    └──────────────────────┘    └─────────────────────┘
```

This diagram shows two vLLM services serving different models:
- **vLLM Service 1**: Serves Llama-3.2-1B-Instruct (owned_by: "vllm")
- **vLLM Service 2**: Serves Phi-tiny-MoE-instruct (owned_by: "vllm")

Both use the same inference engine (vLLM) but serve different models. The gateway routes based on the `model` field in the request.

#### Step 1: Deploy First Model (Llama-3.2-1B-Instruct)

**File:** `code/vllm/llama-3.2-1b.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-llama-32-1b
  labels:
    app: vllm
    model: llama-32-1b
spec:
  runtimeClassName: nvidia
  containers:
  - name: vllm-server
    image: vllm/vllm-openai:v0.12.0
    command:
    - python3
    - -m
    - vllm.entrypoints.openai.api_server
    args:
    - --model
    - meta-llama/Llama-3.2-1B-Instruct
    - --host
    - "0.0.0.0"
    - --port
    - "8000"
    - --tensor-parallel-size
    - "1"
    - --gpu-memory-utilization
    - "0.9"
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: 8Gi
      requests:
        nvidia.com/gpu: 1
        memory: 6Gi
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama-32-1b
spec:
  type: ClusterIP
  selector:
    app: vllm
    model: llama-32-1b
  ports:
  - port: 8000
    targetPort: 8000
```

**Deploy:**

```bash
# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"

# Deploy model
kubectl apply -f code/vllm/llama-3.2-1b.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/vllm-llama-32-1b --timeout=300s
```

#### Step 2: Deploy Second Model (Phi-tiny-MoE-instruct)

**File:** `code/vllm/phi-tiny-moe.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-phi-tiny-moe
  labels:
    app: vllm
    model: phi-tiny-moe
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
      model: phi-tiny-moe
  template:
    metadata:
      labels:
        app: vllm
        model: phi-tiny-moe
    spec:
      runtimeClassName: nvidia
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:v0.12.0
        command:
        - python3
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model
        - /models/Phi-tiny-MoE-instruct
        - --host
        - "0.0.0.0"
        - --port
        - "8000"
        - --tensor-parallel-size
        - "1"
        - --gpu-memory-utilization
        - "0.9"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        hostPath:
          path: /models
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-phi-tiny-moe-service
spec:
  type: ClusterIP
  selector:
    app: vllm
    model: phi-tiny-moe
  ports:
  - port: 8000
    targetPort: 8000
```

**Deploy:**

```bash
# Deploy model
kubectl apply -f code/vllm/phi-tiny-moe.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod -l app=vllm,model=phi-tiny-moe --timeout=600s
```

#### Step 3: Deploy API Gateway

The API Gateway provides a unified entry point that automatically routes requests to the correct vLLM service based on the `model` field in the request body.

**File:** `code/api-gateway.yaml`

**Key Features:**
- **Automatic Routing**: Routes requests based on `model` field in request body
- **Unified Interface**: Single endpoint for all models (`/v1/chat/completions`)
- **Model Discovery**: Aggregates model lists from all services (`/v1/models`)
- **Streaming Support**: Supports streaming responses

**Deploy:**

```bash
# Deploy API Gateway
kubectl apply -f code/api-gateway.yaml

# Wait for gateway to be ready
kubectl wait --for=condition=Ready pod/vllm-api-gateway --timeout=60s
```

#### Step 4: Using the Unified Interface

Once deployed, all requests go through the API Gateway, which automatically routes to the correct model service.

**1. List Available Models:**

```bash
# Port forward gateway
kubectl port-forward svc/vllm-api-gateway 8000:8000

# List all models
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    },
    {
      "id": "/models/Phi-tiny-MoE-instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    }
  ]
}
```

**2. Chat Completion with Llama Model:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**3. Chat Completion with Phi-tiny-MoE Model:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [
      {"role": "user", "content": "What is a mixture of experts?"}
    ],
    "max_tokens": 100
  }'
```

**4. Streaming Response:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "stream": true
  }'
```

### Example 2: Same Model, Different Inference Engines (vLLM vs SGLang)

This example demonstrates how to deploy the same model (Llama-3.2-1B-Instruct) using different inference engines (vLLM and SGLang) and use an API Gateway to route requests based on both the `model` field and the `owned_by` field.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' and 
                            │ 'owned_by' fields
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              API Gateway (Unified Entry Point)              │
│  - Parses 'model' field from request body                   │
│  - Parses 'owned_by' field                                  │
│  - Routes to appropriate service based on both fields       │
│  - Returns response to client                               │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  vLLM Service        │    │  SGLang Service     │
    │  (Llama-3.2-1B)      │    │  (Llama-3.2-1B)     │
    │                      │    │                     │
    │ Pod: vllm-llama-32-1b│    │  Pod: sglang-llama- │
    │ Service: vllm-llama- │    │    32-1b            │
    │    32-1b:8000        │    │  Service: sglang-   │
    │                      │    │    llama-32-1b:8000 │
    │ owned_by: "vllm"     │    │  owned_by: "sglang" │
    │                      │    │                     │
    │ Image: vllm/vllm-    │    │  Image: lmsysorg/   │
    │   openai:v0.12.0     │    │    sglang:v0.5.6    │
    └──────────────────────┘    └─────────────────────┘
```

**Routing Logic:**

1. **Request with vLLM engine:**
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B-Instruct",
     "owned_by": "vllm",
     "messages": [...]
   }
   ```
   → Routes to `vllm-llama-32-1b:8000`

2. **Request with SGLang engine:**
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B-Instruct",
     "owned_by": "sglang",
     "messages": [...]
   }
   ```
   → Routes to `sglang-llama-32-1b:8000`

3. **Request without owned_by (defaults to vLLM):**
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B-Instruct",
     "messages": [...]
   }
   ```
   → Routes to `vllm-llama-32-1b:8000` (default)

**Key Differences:**
- **Same Model**: Both services serve `meta-llama/Llama-3.2-1B-Instruct`
- **Different Engines**: vLLM vs SGLang (different inference engines)
- **Different owned_by**: "vllm" vs "sglang" (used for routing)
- **Different Images**: Different container images for each engine
- **Different Performance**: Each engine has different characteristics (latency, throughput, etc.)

This setup allows clients to:
- Compare performance between inference engines
- A/B test different engines
- Route based on workload characteristics (e.g., use SGLang for structured generation, vLLM for chat)

#### Step 1: Deploy vLLM Service for Llama-3.2-1B-Instruct

Deploy the vLLM service serving Llama-3.2-1B-Instruct. This is the same deployment as in Example 1, Step 1.

**File:** `code/vllm/llama-3.2-1b.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-llama-32-1b
  labels:
    app: vllm
    model: llama-32-1b
spec:
  runtimeClassName: nvidia
  containers:
  - name: vllm-server
    image: vllm/vllm-openai:v0.12.0
    command:
    - python3
    - -m
    - vllm.entrypoints.openai.api_server
    args:
    - --model
    - meta-llama/Llama-3.2-1B-Instruct
    - --host
    - "0.0.0.0"
    - --port
    - "8000"
    - --tensor-parallel-size
    - "1"
    - --gpu-memory-utilization
    - "0.9"
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: 8Gi
      requests:
        nvidia.com/gpu: 1
        memory: 6Gi
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama-32-1b
spec:
  type: ClusterIP
  selector:
    app: vllm
    model: llama-32-1b
  ports:
  - port: 8000
    targetPort: 8000
```

**Deploy:**

```bash
# Create HuggingFace token secret (if not already created)
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"

# Deploy model
kubectl apply -f code/vllm/llama-3.2-1b.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/vllm-llama-32-1b --timeout=300s
```

#### Step 2: Deploy SGLang Service for Llama-3.2-1B-Instruct

Deploy the SGLang service serving the same model (Llama-3.2-1B-Instruct) but using a different inference engine.

**File:** `code/sglang/llama-3.2-1b.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sglang-llama-32-1b
  labels:
    app: sglang
    model: llama-32-1b
spec:
  runtimeClassName: nvidia
  containers:
  - name: sglang-server
    image: lmsysorg/sglang:v0.5.6.post2-runtime
    command:
    - python3
    - -m
    - sglang.launch_server
    args:
    - --model-path
    - meta-llama/Llama-3.2-1B-Instruct
    - --host
    - "0.0.0.0"
    - --port
    - "8000"
    - --trust-remote-code
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: 8Gi
      requests:
        nvidia.com/gpu: 1
        memory: 6Gi
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-llama-32-1b
spec:
  type: ClusterIP
  selector:
    app: sglang
    model: llama-32-1b
  ports:
  - port: 8000
    targetPort: 8000
```

**Deploy:**

```bash
# Deploy SGLang service
kubectl apply -f code/sglang/llama-3.2-1b.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/sglang-llama-32-1b --timeout=300s
```

#### Step 3: Update API Gateway Configuration

The API Gateway needs to be configured to route based on both `model` and `owned_by` fields. The gateway code already supports this routing logic (see `code/api-gateway.py`).

**Routing Configuration:**

The gateway uses the following routing logic:
- `("meta-llama/Llama-3.2-1B-Instruct", "vllm")` → `vllm-llama-32-1b`
- `("meta-llama/Llama-3.2-1B-Instruct", "sglang")` → `sglang-llama-32-1b`
- `("meta-llama/Llama-3.2-1B-Instruct", None)` → `vllm-llama-32-1b` (default)

**Deploy/Update Gateway:**

```bash
# Deploy or update API Gateway
kubectl apply -f code/api-gateway.yaml

# Wait for gateway to be ready
kubectl wait --for=condition=Ready pod/vllm-api-gateway --timeout=60s
```

#### Step 4: Test the Unified Interface

Test routing to both inference engines using the `owned_by` field.

**1. List Available Models:**

```bash
# Port forward gateway
kubectl port-forward svc/vllm-api-gateway 8000:8000

# List all models (should show both vllm and sglang versions)
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    },
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "sglang"
    }
  ]
}
```

**2. Request with vLLM Engine:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**3. Request with SGLang Engine:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "sglang",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**4. Request without owned_by (defaults to vLLM):**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

### Production Considerations

**1. Expose Gateway Externally:**

For production, expose the gateway using Ingress:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-api-gateway-ingress
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-api-gateway
            port:
              number: 8000
```

**2. Add Authentication:**

Add API key authentication to the gateway:

```python
API_KEYS = {"client-1": "key-abc123", "client-2": "key-def456"}

@app.middleware("http")
async def authenticate(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in API_KEYS.values():
        raise HTTPException(401, "Invalid API key")
    return await call_next(request)
```

**3. Add Rate Limiting:**

Implement per-client rate limiting:

```python
from collections import defaultdict
import time

rate_limiter = defaultdict(list)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_id = request.headers.get("X-Client-ID", "default")
    now = time.time()
    # Clean old requests
    rate_limiter[client_id] = [t for t in rate_limiter[client_id] if now - t < 60]
    if len(rate_limiter[client_id]) >= 100:
        raise HTTPException(429, "Rate limit exceeded")
    rate_limiter[client_id].append(now)
    return await call_next(request)
```

**4. Add Monitoring:**

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

request_count = Counter('gateway_requests_total', 'Total requests', ['model', 'status'])
request_latency = Histogram('gateway_request_duration_seconds', 'Request latency')

@app.middleware("http")
async def metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    model = extract_model_from_request(request)
    request_count.labels(model=model, status=response.status_code).inc()
    request_latency.observe(duration)
    return response
```


## Hands-On Examples: LLM serving in Kubernetes with llm-d

This section demonstrates how to achieve the same architecture as Example 1 (different models, same inference engine) using llm-d's production-ready Helm charts and intelligent inference scheduling.

### Architecture Overview

llm-d provides a production-grade solution using:
- **Inference Gateway (IGW)**: Kubernetes-native gateway with intelligent load balancing
- **InferencePool**: Routes requests to appropriate ModelService instances
- **ModelService**: Helm chart for deploying vLLM model servers
- **Intelligent Scheduler**: Load-aware and prefix-cache aware routing

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' field
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Inference Gateway (Kubernetes Gateway API)          │
│  - Load balancing with prefix-cache awareness               │
│  - Intelligent request scheduling                           │
│  - Traffic routing to InferencePool                         │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              InferencePool (Routing Layer)                  │
│  - Routes requests to appropriate ModelService              │
│  - Model-aware request distribution                         │
│  - Health checking and load balancing                       │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌──────────▼──────────┐
    │  ModelService 1      │    │  ModelService 2     │
    │  (Llama-3.2-1B)      │    │  (Phi-tiny-MoE)     │
    │                      │    │                     │
    │  vLLM Pods (2x)      │    │  vLLM Pods (2x)     │
    │  - Intelligent       │    │  - Intelligent      │
    │    load balancing    │    │    load balancing   │
    │  - Prefix cache      │    │  - Prefix cache     │
    │    aware routing     │    │    aware routing    │
    └──────────────────────┘    └─────────────────────┘
```

### Prerequisites

1. **Install Client Tools:**
   ```bash
   cd resources/llm-d/guides/prereq/client-setup
   ./install-deps.sh
   ```
   This installs: `helm`, `helmfile`, `kubectl`, `yq`, `git`

2. **Create Namespace:**
   ```bash
   export NAMESPACE=llm-d-multi-model
   kubectl create namespace ${NAMESPACE}
   ```

3. **Create HuggingFace Token Secret:**
   ```bash
   export HF_TOKEN='your_huggingface_token_here'
   kubectl create secret generic llm-d-hf-token \
     --from-literal="HF_TOKEN=${HF_TOKEN}" \
     --namespace "${NAMESPACE}" \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

4. **Deploy Gateway Provider:**
   ```bash
   cd resources/llm-d/guides/prereq/gateway-provider
   ./install-gateway-provider-dependencies.sh
   
   # For Istio (default)
   helmfile apply -f istio.helmfile.yaml
   
   # Or for kGateway
   # helmfile apply -f kgateway.helmfile.yaml
   ```

### Step 1: Deploy First Model (Llama-3.2-1B-Instruct)

Create a custom values file for the first model:

**File:** `llama-3.2-1b-values.yaml`

```yaml
multinode: false

modelArtifacts:
  uri: "hf://meta-llama/Llama-3.2-1B-Instruct"
  name: "meta-llama/Llama-3.2-1B-Instruct"
  size: 20Gi
  authSecretName: "llm-d-hf-token"

routing:
  servicePort: 8000
  proxy:
    image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.4.0-rc.1
    connector: nixlv2
    secure: false

decode:
  create: true
  replicas: 2
  containers:
  - name: "vllm"
    image: ghcr.io/llm-d/llm-d-cuda:v0.3.1
    modelCommand: vllmServe
    args:
      - "--disable-uvicorn-access-log"
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: 8Gi
      requests:
        nvidia.com/gpu: "1"
        memory: 6Gi
    mountModelVolume: true

prefill:
  create: false
```

**Deploy:**

```bash
cd resources/llm-d/guides/inference-scheduling

# Deploy first model with custom release name
RELEASE_NAME_POSTFIX=llama-3.2-1b \
helmfile apply -n ${NAMESPACE} \
  --set-file ms-llama-3.2-1b.values[0]=llama-3.2-1b-values.yaml
```

### Step 2: Deploy Second Model (Phi-tiny-MoE-instruct)

Create a custom values file for the second model:

**File:** `phi-tiny-moe-values.yaml`

```yaml
multinode: false

modelArtifacts:
  uri: "hf://microsoft/Phi-3.5-MoE-instruct"  # Or local path: "/models/Phi-tiny-MoE-instruct"
  name: "/models/Phi-tiny-MoE-instruct"
  size: 30Gi
  authSecretName: "llm-d-hf-token"

routing:
  servicePort: 8000
  proxy:
    image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.4.0-rc.1
    connector: nixlv2
    secure: false

decode:
  create: true
  replicas: 2
  containers:
  - name: "vllm"
    image: ghcr.io/llm-d/llm-d-cuda:v0.3.1
    modelCommand: vllmServe
    args:
      - "--disable-uvicorn-access-log"
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: 32Gi
      requests:
        nvidia.com/gpu: "1"
        memory: 16Gi
    mountModelVolume: true

prefill:
  create: false
```

**Deploy:**

```bash
# Deploy second model with different release name
RELEASE_NAME_POSTFIX=phi-tiny-moe \
helmfile apply -n ${NAMESPACE} \
  --set-file ms-phi-tiny-moe.values[0]=phi-tiny-moe-values.yaml
```

### Step 3: Configure InferencePool for Multi-Model Routing

The InferencePool automatically discovers ModelService instances and routes requests based on the `model` field. Create or update the InferencePool configuration:

**File:** `inferencepool-multi-model-values.yaml`

```yaml
provider:
  name: istio  # or kgateway, gke, etc.

inferencePool:
  apiVersion: inference.networking.k8s.io/v1
  metadata:
    name: multi-model-pool
  spec:
    # InferencePool automatically discovers ModelService instances
    # and routes based on model field in requests
```

**Deploy InferencePool:**

```bash
helm install multi-model-pool \
  -n ${NAMESPACE} \
  -f inferencepool-multi-model-values.yaml \
  --set "provider.name=istio" \
  --set "inferenceExtension.monitoring.prometheus.enable=true" \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool \
  --version v1.2.0-rc.1
```

### Step 4: Deploy HTTPRoute

Create an HTTPRoute to expose the InferencePool:

**File:** `httproute-multi-model.yaml`

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: multi-model-route
  namespace: ${NAMESPACE}
spec:
  parentRefs:
  - name: gateway  # Your Gateway name
    namespace: istio-system
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: multi-model-pool-epp
      port: 8000
```

**Deploy:**

```bash
kubectl apply -f httproute-multi-model.yaml -n ${NAMESPACE}
```

### Step 5: Test Multi-Model Routing

**1. Get Gateway External IP:**

```bash
kubectl get gateway -n istio-system
GATEWAY_IP=$(kubectl get svc -n istio-system -l istio=gateway -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}')
echo "Gateway IP: ${GATEWAY_IP}"
```

**2. List Available Models:**

```bash
curl http://${GATEWAY_IP}/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    },
    {
      "id": "/models/Phi-tiny-MoE-instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    }
  ]
}
```

**3. Request to Llama Model:**

```bash
curl http://${GATEWAY_IP}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**4. Request to Phi-tiny-MoE Model:**

```bash
curl http://${GATEWAY_IP}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [
      {"role": "user", "content": "What is a mixture of experts?"}
    ],
    "max_tokens": 100
  }'
```

### Key Advantages of llm-d Approach

1. **Production-Ready**: Tested and benchmarked configurations
2. **Intelligent Routing**: Prefix-cache aware and load-aware balancing
3. **Automatic Discovery**: InferencePool automatically discovers ModelService instances
4. **Monitoring**: Built-in Prometheus metrics and Grafana dashboards
5. **Scalability**: Easy to add more models or scale replicas
6. **Multi-Hardware Support**: Works with NVIDIA, AMD, Intel XPU, Google TPU
7. **Advanced Features**: Supports prefill/decode disaggregation, expert parallelism, etc.

### Comparison: k3d vs llm-d

| Feature | k3d (Manual) | llm-d (Production) |
|---------|--------------|---------------------|
| **Setup Complexity** | Manual YAML files | Helm charts (automated) |
| **Routing** | Custom API Gateway | Inference Gateway (K8s native) |
| **Load Balancing** | Basic round-robin | Intelligent (prefix-cache aware) |
| **Monitoring** | Manual setup | Built-in Prometheus/Grafana |
| **Scaling** | Manual pod management | HPA-ready, autoscaling support |
| **Multi-Model** | Manual service mapping | Automatic discovery |
| **Production Features** | Limited | Full production stack |



## Best Practices

### 1. Handling Cold Starts

**Strategies:**

- Pre-warm models on startup
- Use keepalive requests to prevent idle shutdown
- Implement graceful degradation (fallback to cached responses)
- Consider model quantization for faster loading

**Example:**
```python
# Pre-warm on startup
@app.on_event("startup")
async def startup():
    await warmup_model()

# Keepalive
async def keepalive_loop():
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        await model.generate(["keepalive"], max_tokens=1)
```

### 2. Designing Multi-Model Routing

**Guidelines:**

- Use consistent hashing for user-based routing
- Implement health checks for all model endpoints
- Support feature-based routing (code, chat, etc.)
- Allow dynamic model selection based on load

**Example:**
```python
# Consistent user routing
def route_user(user_id: str) -> str:
    hash_value = hash(user_id) % 100
    return "model-a" if hash_value < 50 else "model-b"
```

### 3. Monitoring End-to-End Latency

**Approach:**

- Instrument all components (gateway, tokenizer, model)
- Use distributed tracing to see full request path
- Track percentiles (P50, P95, P99)
- Set up alerts for latency violations

**Example:**
```python
# Track latency at each stage
with tracer.start_as_current_span("request"):
    with tracer.start_as_current_span("tokenize"):
        tokens = await tokenize(prompt)
    
    with tracer.start_as_current_span("inference"):
        result = await infer(tokens)
    
    with tracer.start_as_current_span("detokenize"):
        text = await detokenize(result)
```

## Use Cases

### Use Case 1: Cloud-Based LLM APIs

**Scenario:** Provide LLM API service to external customers

**Requirements:**

- High availability (99.9%+)
- Rate limiting per customer
- Multi-model support
- Cost optimization
- Observability

**Implementation:**

- API gateway with authentication
- Per-customer rate limiting
- Model routing based on customer tier
- Autoscaling based on load
- Comprehensive monitoring

### Use Case 2: Internal Enterprise AI Platform

**Scenario:** Internal platform for company-wide AI services

**Requirements:**

- Integration with internal systems
- A/B testing for model improvements
- Cost tracking and optimization
- Security and compliance

**Implementation:**

- Single sign-on integration
- Canary deployments for new models
- Cost tracking per department
- Audit logging



## Skills Learned

By the end of this chapter, readers will be able to:

1. **Build complete serving stacks**
   - Design and implement tokenizer service
   - Deploy model runners with proper configuration
   - Set up API gateway with routing and rate limiting

2. **Implement routing and model selection**
   - Design feature-based routing
   - Implement A/B traffic splits
   - Build dynamic model selection based on load and latency

3. **Use canary rollouts safely**
   - Implement gradual traffic shifting
   - Monitor canary performance
   - Automate rollback on SLO violations

4. **Add tracing and monitoring**
   - Instrument services with OpenTelemetry
   - Collect metrics with Prometheus
   - Build monitoring dashboards

5. **Improve reliability and cost efficiency**
   - Handle cold starts gracefully
   - Implement autoscaling
   - Optimize costs with spot instances and model selection

6. **Deploy with llm-d on Kubernetes**
   - Use Helm charts for production deployment
   - Configure intelligent inference scheduling
   - Set up prefill/decode disaggregation
   - Monitor and optimize Kubernetes-based serving



## Summary

This chapter has covered building a complete production LLM serving stack. Key takeaways:

1. **Production systems are complex:** Multiple components work together (tokenizer, model runner, gateway, monitoring)
2. **Routing is critical:** Intelligent routing improves performance and cost
3. **Canary deployments enable safe rollouts:** Gradual traffic shifting with automated rollback
4. **Observability is essential:** Distributed tracing and metrics are crucial for debugging and optimization
5. **Cost optimization matters:** Spot instances, model selection, and autoscaling reduce costs

Building production LLM serving systems requires careful attention to reliability, scalability, and cost. The patterns and techniques covered in this chapter provide a solid foundation for building such systems.

Once you've built your distributed training and inference systems, you need to know how well they're performing. Are you getting the throughput you expect? Is latency acceptable? How efficiently are you using your GPUs? The next chapter teaches you how to benchmark distributed training and inference systems rigorously. We'll cover both performance benchmarking (throughput, latency, scaling efficiency) and accuracy benchmarking (model quality, output correctness), using tools like genai-bench, PyTorch profiler, and custom scripts. By the end, you'll be able to identify bottlenecks, evaluate model accuracy, and optimize your systems effectively.

## Exercises

1. **Build API Gateway:** Implement an API gateway with routing, rate limiting, and health checks.

2. **Implement Canary Deployment:** Create a canary deployment system with automated rollback based on error rates.

3. **Add Distributed Tracing:** Instrument a multi-service LLM serving system with OpenTelemetry.

4. **Optimize Costs:** Design a cost-optimized serving system using spot instances and intelligent model selection.

5. **Deploy with llm-d:** Deploy a production LLM serving stack on Kubernetes using llm-d Helm charts, configure prefill/decode disaggregation, and monitor performance.

## References

__LLM Serving Frameworks__

- vLLM GitHub: \url{https://github.com/vllm-project/vllm}
- vLLM Production Stack: \url{https://github.com/vllm-project/production-stack}
- SGLang GitHub: \url{https://github.com/sgl-project/sglang}

__Kubernetes and llm-d__

- llm-d GitHub: \url{https://github.com/llm-d/llm-d}
- llm-d Documentation: \url{https://www.llm-d.ai/}
- Inference Gateway: \url{https://github.com/kserve/inference-gateway}
- k3d (k3s in Docker): \url{https://k3d.io/}
- NVIDIA Device Plugin for Kubernetes: \url{https://github.com/NVIDIA/k8s-device-plugin}

__Observability and Monitoring__

- OpenTelemetry: \url{https://opentelemetry.io/}
- Prometheus: \url{https://prometheus.io/}
- Grafana: \url{https://grafana.com/}

__API Gateway and Routing__

- Envoy Proxy: \url{https://www.envoyproxy.io/}
- FastAPI: \url{https://fastapi.tiangolo.com/}
- Kubernetes Gateway API: \url{https://gateway-api.sigs.k8s.io/}

__Tutorials and Guides__

- vLLM Kubernetes Deployment: \url{https://docs.vllm.ai/en/stable/deployment/k8s/}
- NVIDIA Container Toolkit: \url{https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/}