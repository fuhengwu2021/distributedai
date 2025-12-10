# vLLM Deployment for Phi-tiny-MoE-instruct

## Current Status

✅ **Deployment YAML ready:** `vllm-phi-tiny-moe.yaml`
- Uses custom `vllm-dev:latest` image
- Configured for GPU support
- Model mounted from `/raid/models/Phi-tiny-MoE-instruct`

❌ **Cluster Issue:** Custom k3s-cuda image has exec problem
- Missing `/usr/bin/sh` which k3d needs
- Cluster creation fails during post-start preparation

## Quick Start (Once Cluster is Fixed)

```bash
# 1. Ensure cluster exists with GPU support
k3d cluster list

# 2. Deploy vLLM
kubectl apply -f vllm-phi-tiny-moe.yaml

# 3. Wait for pod to be ready
kubectl wait --for=condition=Ready pod -l app=vllm,model=phi-tiny-moe --timeout=600s

# 4. Port-forward
kubectl port-forward svc/vllm-phi-tiny-moe-service 8000:8000 &

# 5. Test
curl http://localhost:8000/health
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/Phi-tiny-MoE-instruct", "prompt": "What is AI?", "max_tokens": 30}'
```

## Fixing the Custom k3s-cuda Image

The custom image needs `/usr/bin/sh`. To fix:

1. **Option 1: Add symlink in Dockerfile**
   ```dockerfile
   RUN ln -s /bin/sh /usr/bin/sh
   ```

2. **Option 2: Use busybox or ensure shell is in /usr/bin**
   ```dockerfile
   RUN apt-get update && apt-get install -y busybox-static
   ```

3. **Option 3: Rebuild with proper base**
   - Ensure the base CUDA image has shell utilities
   - Or copy them from the k3s image properly

## Testing Script

Use `test-vllm-api.sh` to test all endpoints:

```bash
./test-vllm-api.sh http://localhost:8000
```
