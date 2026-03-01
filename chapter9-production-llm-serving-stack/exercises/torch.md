\fancydividerwithicon[center]{hand.png}


## Exercises


### Build an API Gateway with Model Routing

Implement an API gateway that routes requests to different model backends based on the `model` field in the request body.

__Requirements:__

- Function signature:
```python
class ModelRouter:
    def __init__(self, model_endpoints: dict[str, str]):
        """Initialize with mapping of model names to backend URLs."""
        pass
    
    async def route_request(self, request: dict) -> dict:
        """Route request to appropriate backend based on model field."""
        pass
```
- Parse the `model` field from incoming requests
- Forward requests to the correct backend URL
- Handle cases where the requested model is not available (return 404)
- Add basic health checking for backends
- Support both `/v1/chat/completions` and `/v1/completions` endpoints

__Test your implementation:__
```python
import asyncio
import httpx

# Initialize router with model endpoints
router = ModelRouter({
    "llama-3.2-1b": "http://localhost:8001",
    "qwen-0.5b": "http://localhost:8002",
})

# Test routing
request = {
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
}

response = asyncio.run(router.route_request(request))
print(f"Response from {request['model']}: {response}")

# Test unknown model
unknown_request = {"model": "unknown-model", "messages": []}
try:
    asyncio.run(router.route_request(unknown_request))
except Exception as e:
    print(f"Expected error: {e}")
```

### Implement Rate Limiting Middleware

Create a rate limiting middleware that enforces per-client request quotas.

__Requirements:__

- Class signature:
```python
class RateLimiter:
    def __init__(self, requests_per_minute: int, burst_size: int = 10):
        """Initialize rate limiter with limits."""
        pass
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Return True if request is allowed, False if rate limited."""
        pass
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Return number of remaining requests for client."""
        pass
```
- Use token bucket algorithm for rate limiting
- Track requests per client (identified by API key or IP)
- Support burst capacity for handling traffic spikes
- Return appropriate headers (`X-RateLimit-Remaining`, `X-RateLimit-Reset`)
- Clean up stale client entries to prevent memory leaks

__Test your implementation:__
```python
import asyncio
import time

limiter = RateLimiter(requests_per_minute=10, burst_size=5)

async def test_rate_limiting():
    client_id = "test-client-123"
    
    # Should allow burst of requests
    for i in range(5):
        allowed = await limiter.check_rate_limit(client_id)
        remaining = limiter.get_remaining_requests(client_id)
        print(f"Request {i+1}: allowed={allowed}, remaining={remaining}")
    
    # Should start rate limiting
    for i in range(10):
        allowed = await limiter.check_rate_limit(client_id)
        if not allowed:
            print(f"Rate limited after {i+5} requests")
            break

asyncio.run(test_rate_limiting())
```

### Implement Canary Deployment with Automated Rollback

Create a canary deployment system that gradually shifts traffic and automatically rolls back on errors.

__Requirements:__

- Class signature:
```python
class CanaryDeployment:
    def __init__(self, stable_endpoint: str, canary_endpoint: str):
        """Initialize with stable and canary endpoints."""
        pass
    
    def set_canary_percentage(self, percentage: float):
        """Set percentage of traffic to route to canary (0-100)."""
        pass
    
    async def route_request(self, request: dict) -> dict:
        """Route request to stable or canary based on percentage."""
        pass
    
    def record_result(self, is_canary: bool, success: bool, latency_ms: float):
        """Record request result for monitoring."""
        pass
    
    def should_rollback(self, error_threshold: float = 0.05) -> bool:
        """Return True if canary error rate exceeds threshold."""
        pass
```
- Route traffic based on configured percentage
- Track success/failure rates for both stable and canary
- Calculate error rate difference between canary and stable
- Trigger rollback when canary error rate exceeds threshold
- Support gradual traffic increase (10% -> 25% -> 50% -> 100%)

__Test your implementation:__
```python
import asyncio
import random

canary = CanaryDeployment(
    stable_endpoint="http://localhost:8001",
    canary_endpoint="http://localhost:8002"
)

# Start with 10% canary traffic
canary.set_canary_percentage(10)

# Simulate requests
for i in range(100):
    request = {"model": "test", "messages": []}
    
    # Simulate response (canary has higher error rate)
    is_canary = random.random() < 0.1
    success = random.random() > (0.1 if is_canary else 0.02)
    latency = random.uniform(50, 200)
    
    canary.record_result(is_canary, success, latency)
    
    if canary.should_rollback(error_threshold=0.05):
        print(f"Rollback triggered after {i+1} requests!")
        canary.set_canary_percentage(0)
        break

print(f"Final canary percentage: {canary.canary_percentage}%")
```

### Add Distributed Tracing with OpenTelemetry

Instrument a multi-service LLM serving system with OpenTelemetry for end-to-end request tracing.

__Requirements:__

- Create spans for each stage of request processing:
  - `gateway.receive`: Request received at gateway
  - `gateway.route`: Routing decision
  - `tokenizer.encode`: Tokenization
  - `model.inference`: Model inference
  - `tokenizer.decode`: Detokenization
  - `gateway.respond`: Response sent
- Add attributes to spans:
  - `model.name`: Name of the model
  - `request.tokens`: Number of input tokens
  - `response.tokens`: Number of output tokens
  - `latency.ttft_ms`: Time to first token
  - `latency.total_ms`: Total latency
- Propagate trace context across service boundaries
- Export traces to console or Jaeger

__Test your implementation:__
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Setup tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-serving")

async def process_request(request: dict):
    with tracer.start_as_current_span("gateway.receive") as span:
        span.set_attribute("model.name", request.get("model", "unknown"))
        
        with tracer.start_as_current_span("tokenizer.encode"):
            tokens = tokenize(request["messages"])
            span.set_attribute("request.tokens", len(tokens))
        
        with tracer.start_as_current_span("model.inference"):
            output = await model.generate(tokens)
        
        with tracer.start_as_current_span("tokenizer.decode"):
            response = detokenize(output)
            span.set_attribute("response.tokens", len(output))
        
        return response

# Test with sample request
request = {
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
}
response = asyncio.run(process_request(request))
```

### Deploy Multi-Model Serving on k3d

Deploy a multi-model LLM serving system on a local k3d cluster with GPU support.

__Requirements:__

1. Create a k3d cluster with GPU passthrough:
```bash
k3d cluster create llm-serving \
  --gpus=all \
  --volume /path/to/models:/models \
  --port "8080:80@loadbalancer"
```

2. Deploy two vLLM instances serving different models:
   - Model 1: `Qwen/Qwen2.5-0.5B-Instruct` on port 8001
   - Model 2: `meta-llama/Llama-3.2-1B-Instruct` on port 8002

3. Create an API gateway deployment that routes based on model name

4. Verify the deployment:
```bash
# List available models
curl http://localhost:8080/v1/models

# Test Model 1
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test Model 2
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

__Deliverables:__

- Kubernetes YAML files for vLLM deployments
- API gateway deployment and service YAML
- ConfigMap for model-to-service routing
- Shell script to deploy and test the system


## Expected Learning Outcomes

After completing these exercises, you should be able to:

- Build API gateways that route requests to multiple model backends
- Implement rate limiting to protect services from overload
- Deploy canary releases with automated rollback capabilities
- Instrument distributed systems with OpenTelemetry for observability
- Deploy and manage LLM serving infrastructure on Kubernetes
- Design production-ready serving systems with proper monitoring and fault tolerance
