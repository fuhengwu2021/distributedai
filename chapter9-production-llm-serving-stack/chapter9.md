# Chapter 9: Production LLM Serving Stack

## Overview

We've covered training systems (DDP, FSDP, DeepSpeed), inference engines (vLLM, SGLang), and how to run them with Slurm. But building a production LLM serving system requires more than just running an inference engine. You need a complete stack: model runners that load and execute models efficiently, tokenizers that handle text preprocessing, API gateways that route and load balance requests, rate limiting to prevent abuse, observability to monitor performance, and deployment strategies like A/B testing and canary rollouts to ensure reliability.

This chapter builds a complete end-to-end production LLM serving stack, including all these components. Readers implement A/B testing, canary rollouts, and distributed tracing to ensure reliability and maintainability at scale. By the end of this chapter, readers will be able to design, deploy, and operate production-grade LLM serving systems.

**Chapter Length:** 35 pages



## 1. Anatomy of a Production LLM Serving System

A production LLM serving system is more than just a model running on a GPU. It's a complex distributed system with multiple components working together to provide reliable, scalable, and cost-effective inference services. Understanding the architecture is crucial for building robust systems.

### System Components

**Core Components:**

1. **Tokenizer Service**
   - Stateless, lightweight service
   - Handles text tokenization and detokenization
   - Can be scaled independently
   - Low latency requirement (<10ms)

2. **Model Runner**
   - Stateful, GPU-backed inference engine
   - Manages model loading, KV cache, batching
   - Handles continuous batching and scheduling
   - High throughput requirement

3. **API Gateway**
   - Request routing and load balancing
   - Authentication and authorization
   - Rate limiting and throttling
   - Request/response transformation

4. **Monitoring and Observability**
   - Metrics collection (Prometheus)
   - Distributed tracing (OpenTelemetry)
   - Logging and alerting
   - Performance dashboards

### System Architecture

**High-Level Architecture:**
```
┌─────────────┐
│   Clients   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ API Gateway │ (Routing, Auth, Rate Limiting)
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│Tokenizer │   │ Model    │   │ Model    │
│ Service  │   │ Runner 1 │   │ Runner 2 │
└──────────┘   └──────────┘   └──────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ Monitoring  │
              │ & Tracing   │
              └──────────────┘
```

### Tokenizer Service

**Implementation:**
```python
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
import asyncio

app = FastAPI()

class TokenizerService:
    def __init__(self):
        self.tokenizers = {}
        self.load_tokenizer("llama-2-7b", "meta-llama/Llama-2-7b-chat-hf")
    
    def load_tokenizer(self, model_name, tokenizer_path):
        """Load tokenizer for a model"""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizers[model_name] = tokenizer
    
    def encode(self, model_name, text):
        """Tokenize text"""
        if model_name not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_name} not found")
        
        tokenizer = self.tokenizers[model_name]
        tokens = tokenizer.encode(text, return_tensors="pt")
        return tokens.tolist()[0]
    
    def decode(self, model_name, token_ids):
        """Detokenize tokens"""
        if model_name not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_name} not found")
        
        tokenizer = self.tokenizers[model_name]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

tokenizer_service = TokenizerService()

@app.post("/tokenize")
async def tokenize(request: dict):
    """Tokenize endpoint"""
    model_name = request.get("model", "llama-2-7b")
    text = request.get("text", "")
    
    try:
        tokens = tokenizer_service.encode(model_name, text)
        return {"tokens": tokens, "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detokenize")
async def detokenize(request: dict):
    """Detokenize endpoint"""
    model_name = request.get("model", "llama-2-7b")
    token_ids = request.get("tokens", [])
    
    try:
        text = tokenizer_service.decode(model_name, token_ids)
        return {"text": text, "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Model Runner

**Basic Model Runner:**
```python
import torch
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
import asyncio
from typing import List, Dict

class ModelRunner:
    def __init__(self, model_name: str, gpu_id: int = 0):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.llm = None
        self.load_model()
    
    def load_model(self):
        """Load model for inference"""
        print(f"Loading model {self.model_name} on GPU {self.gpu_id}...")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            trust_remote_code=True
        )
        print("Model loaded successfully")
    
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text from prompts"""
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 512),
            stop=kwargs.get("stop", [])
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    async def generate_async(self, prompts: List[str], **kwargs) -> List[str]:
        """Async generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompts, **kwargs)

# Usage
runner = ModelRunner("meta-llama/Llama-2-7b-chat-hf")
prompts = ["What is machine learning?", "Explain neural networks."]
results = runner.generate(prompts, max_tokens=100)
```

### API Gateway

**Basic API Gateway with FastAPI:**
```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import time
from collections import defaultdict

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class GenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "llama-2-7b"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9

class GenerationResponse(BaseModel):
    text: str
    model: str
    latency_ms: float
    tokens_generated: int

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests
        client_requests[:] = [t for t in client_requests if now - t < self.window_seconds]
        
        # Check limit
        if len(client_requests) >= self.max_requests:
            return False
        
        # Add current request
        client_requests.append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Model routing
MODEL_ENDPOINTS = {
    "llama-2-7b": "http://localhost:8002",
    "llama-2-13b": "http://localhost:8003",
    "mistral-7b": "http://localhost:8004"
}

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, client_id: str = "default"):
    """Generate text endpoint"""
    # Rate limiting
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Route to appropriate model
    model_endpoint = MODEL_ENDPOINTS.get(request.model)
    if not model_endpoint:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    # Forward request to model runner
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{model_endpoint}/generate",
            json={
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p
            },
            timeout=60.0
        )
    
    latency_ms = (time.time() - start_time) * 1000
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    result = response.json()
    
    return GenerationResponse(
        text=result["text"],
        model=request.model,
        latency_ms=latency_ms,
        tokens_generated=result.get("tokens_generated", 0)
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Request Flow

**Complete Request Flow:**
```python
async def handle_request(request: GenerationRequest):
    """Handle a complete generation request"""
    # 1. Tokenize (if needed)
    tokenizer_response = await tokenize_request(request.prompt)
    
    # 2. Route to model
    model_response = await route_to_model(request.model, tokenizer_response)
    
    # 3. Detokenize
    final_response = await detokenize_response(model_response)
    
    # 4. Log and trace
    await log_request(request, final_response)
    
    return final_response
```

## 2. Multi-Model Routing and Load Balancing

In production systems, you often need to serve multiple models simultaneously, route requests intelligently, and balance load across instances. This section covers routing strategies and load balancing techniques.

### Routing Strategies

**1. Feature-Based Routing:**
```python
class FeatureBasedRouter:
    def __init__(self):
        self.routes = {
            "code": "code-llama-7b",
            "chat": "llama-2-7b-chat",
            "summarization": "mistral-7b",
            "default": "llama-2-7b"
        }
    
    def route(self, request: dict) -> str:
        """Route based on request features"""
        # Check explicit model parameter
        if "model" in request:
            return request["model"]
        
        # Route based on prompt content
        prompt = request.get("prompt", "").lower()
        
        if any(keyword in prompt for keyword in ["code", "function", "class", "def"]):
            return self.routes["code"]
        elif any(keyword in prompt for keyword in ["hello", "hi", "chat", "conversation"]):
            return self.routes["chat"]
        elif any(keyword in prompt for keyword in ["summarize", "summary", "brief"]):
            return self.routes["summarization"]
        else:
            return self.routes["default"]
```

**2. A/B Traffic Splits:**
```python
import random
import hashlib

class ABRouter:
    def __init__(self, split_ratio: float = 0.5):
        self.split_ratio = split_ratio
        self.model_a = "llama-2-7b"
        self.model_b = "llama-2-13b"
    
    def route(self, request: dict) -> str:
        """Route based on A/B split"""
        # Use consistent hashing for same user
        user_id = request.get("user_id", "anonymous")
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        
        # Consistent assignment
        if (hash_value % 100) < (self.split_ratio * 100):
            return self.model_a
        else:
            return self.model_b
```

**3. Dynamic Model Selection:**
```python
class DynamicRouter:
    def __init__(self):
        self.models = {
            "llama-2-7b": {
                "endpoint": "http://localhost:8002",
                "latency_budget_ms": 500,
                "cost_per_token": 0.001
            },
            "llama-2-13b": {
                "endpoint": "http://localhost:8003",
                "latency_budget_ms": 1000,
                "cost_per_token": 0.002
            }
        }
        self.model_loads = {model: 0 for model in self.models}
    
    def route(self, request: dict) -> str:
        """Route based on load and latency budget"""
        latency_budget = request.get("latency_budget_ms", 1000)
        cost_sensitive = request.get("cost_sensitive", False)
        
        # Filter models by latency budget
        available_models = [
            model for model, config in self.models.items()
            if config["latency_budget_ms"] <= latency_budget
        ]
        
        if not available_models:
            # Fallback to fastest
            return min(self.models.keys(), key=lambda m: self.models[m]["latency_budget_ms"])
        
        # Select based on cost or load
        if cost_sensitive:
            return min(available_models, key=lambda m: self.models[m]["cost_per_token"])
        else:
            return min(available_models, key=lambda m: self.model_loads[m])
    
    def update_load(self, model: str, delta: int):
        """Update model load"""
        if model in self.model_loads:
            self.model_loads[model] += delta
```

### Load Balancing

**Round-Robin Load Balancer:**
```python
from collections import deque

class RoundRobinBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = deque(endpoints)
        self.health_status = {endpoint: True for endpoint in endpoints}
    
    def get_endpoint(self) -> str:
        """Get next endpoint in round-robin fashion"""
        attempts = 0
        while attempts < len(self.endpoints):
            endpoint = self.endpoints[0]
            self.endpoints.rotate(1)
            
            if self.health_status.get(endpoint, False):
                return endpoint
            
            attempts += 1
        
        # All unhealthy, return first anyway
        return self.endpoints[0]
    
    def mark_unhealthy(self, endpoint: str):
        """Mark endpoint as unhealthy"""
        self.health_status[endpoint] = False
    
    def mark_healthy(self, endpoint: str):
        """Mark endpoint as healthy"""
        self.health_status[endpoint] = True
```

**Least Connections Load Balancer:**
```python
class LeastConnectionsBalancer:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.connection_counts = {endpoint: 0 for endpoint in endpoints}
        self.lock = asyncio.Lock()
    
    async def get_endpoint(self) -> str:
        """Get endpoint with least connections"""
        async with self.lock:
            endpoint = min(
                self.endpoints,
                key=lambda e: self.connection_counts[e]
            )
            self.connection_counts[endpoint] += 1
            return endpoint
    
    async def release_endpoint(self, endpoint: str):
        """Release connection from endpoint"""
        async with self.lock:
            if endpoint in self.connection_counts:
                self.connection_counts[endpoint] = max(0, self.connection_counts[endpoint] - 1)
```

**Weighted Load Balancer:**
```python
import random

class WeightedBalancer:
    def __init__(self, endpoints: List[tuple]):
        """
        endpoints: List of (endpoint, weight) tuples
        """
        self.endpoints = endpoints
        self.total_weight = sum(weight for _, weight in endpoints)
    
    def get_endpoint(self) -> str:
        """Get endpoint based on weights"""
        r = random.uniform(0, self.total_weight)
        cumulative = 0
        
        for endpoint, weight in self.endpoints:
            cumulative += weight
            if r <= cumulative:
                return endpoint
        
        # Fallback to last endpoint
        return self.endpoints[-1][0]
```

### Health Checks

**Health Check Implementation:**
```python
import asyncio
import httpx

class HealthChecker:
    def __init__(self, endpoints: List[str], check_interval: int = 30):
        self.endpoints = endpoints
        self.check_interval = check_interval
        self.health_status = {endpoint: True for endpoint in endpoints}
        self.running = False
    
    async def check_endpoint(self, endpoint: str) -> bool:
        """Check if endpoint is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{endpoint}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def check_all(self):
        """Check all endpoints"""
        tasks = [self.check_endpoint(endpoint) for endpoint in self.endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for endpoint, is_healthy in zip(self.endpoints, results):
            self.health_status[endpoint] = is_healthy if isinstance(is_healthy, bool) else False
    
    async def start(self):
        """Start health checking loop"""
        self.running = True
        while self.running:
            await self.check_all()
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop health checking"""
        self.running = False
    
    def is_healthy(self, endpoint: str) -> bool:
        """Check if endpoint is currently healthy"""
        return self.health_status.get(endpoint, False)
```

## 3. Canary Deployments and A/B Testing

Canary deployments allow you to gradually roll out new model versions while monitoring for issues. A/B testing enables comparing model performance in production. This section covers both techniques.

### Canary Deployment

**Basic Canary Implementation:**
```python
class CanaryDeployment:
    def __init__(self, stable_model: str, canary_model: str, traffic_percent: float = 0.1):
        self.stable_model = stable_model
        self.canary_model = canary_model
        self.traffic_percent = traffic_percent
        self.metrics = {
            "stable": {"requests": 0, "errors": 0, "latency_sum": 0.0},
            "canary": {"requests": 0, "errors": 0, "latency_sum": 0.0}
        }
    
    def route(self, request: dict) -> str:
        """Route request to stable or canary"""
        import random
        if random.random() < self.traffic_percent:
            return self.canary_model
        else:
            return self.stable_model
    
    def record_metrics(self, model: str, latency: float, error: bool = False):
        """Record metrics for model"""
        if model in self.metrics:
            self.metrics[model]["requests"] += 1
            self.metrics[model]["latency_sum"] += latency
            if error:
                self.metrics[model]["errors"] += 1
    
    def get_error_rate(self, model: str) -> float:
        """Get error rate for model"""
        if model not in self.metrics:
            return 0.0
        m = self.metrics[model]
        if m["requests"] == 0:
            return 0.0
        return m["errors"] / m["requests"]
    
    def get_avg_latency(self, model: str) -> float:
        """Get average latency for model"""
        if model not in self.metrics:
            return 0.0
        m = self.metrics[model]
        if m["requests"] == 0:
            return 0.0
        return m["latency_sum"] / m["requests"]
    
    def should_promote(self) -> bool:
        """Check if canary should be promoted"""
        canary_error_rate = self.get_error_rate(self.canary_model)
        stable_error_rate = self.get_error_rate(self.stable_model)
        canary_latency = self.get_avg_latency(self.canary_model)
        stable_latency = self.get_avg_latency(self.stable_model)
        
        # Promote if canary is better or similar
        if canary_error_rate <= stable_error_rate * 1.1:  # Allow 10% tolerance
            if canary_latency <= stable_latency * 1.2:  # Allow 20% latency increase
                return True
        
        return False
    
    def should_rollback(self) -> bool:
        """Check if canary should be rolled back"""
        canary_error_rate = self.get_error_rate(self.canary_model)
        stable_error_rate = self.get_error_rate(self.stable_model)
        
        # Rollback if canary error rate is significantly worse
        if canary_error_rate > stable_error_rate * 2.0:
            return True
        
        return False
```

### Traffic Shifting

**Gradual Traffic Shifting:**
```python
class TrafficShifter:
    def __init__(self, stable_model: str, canary_model: str):
        self.stable_model = stable_model
        self.canary_model = canary_model
        self.canary_percent = 0.0
        self.shift_steps = [0.1, 0.25, 0.5, 0.75, 1.0]  # Gradual steps
        self.current_step = 0
    
    def route(self, request: dict) -> str:
        """Route based on current traffic percentage"""
        import random
        if random.random() < self.canary_percent:
            return self.canary_model
        else:
            return self.stable_model
    
    def increase_traffic(self) -> bool:
        """Increase canary traffic to next step"""
        if self.current_step < len(self.shift_steps) - 1:
            self.current_step += 1
            self.canary_percent = self.shift_steps[self.current_step]
            return True
        return False
    
    def decrease_traffic(self):
        """Decrease canary traffic (rollback)"""
        if self.current_step > 0:
            self.current_step -= 1
            self.canary_percent = self.shift_steps[self.current_step]
```

### A/B Testing Framework

**A/B Testing Implementation:**
```python
import hashlib
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ABTestConfig:
    test_name: str
    variants: Dict[str, float]  # variant_name -> traffic_percent
    metrics: List[str]  # Metrics to track

class ABTestFramework:
    def __init__(self):
        self.tests: Dict[str, ABTestConfig] = {}
        self.results: Dict[str, Dict[str, Dict]] = {}  # test_name -> variant -> metrics
    
    def register_test(self, config: ABTestConfig):
        """Register an A/B test"""
        self.tests[config.test_name] = config
        self.results[config.test_name] = {
            variant: {metric: [] for metric in config.metrics}
            for variant in config.variants.keys()
        }
    
    def assign_variant(self, test_name: str, user_id: str) -> str:
        """Assign user to a variant"""
        if test_name not in self.tests:
            return "default"
        
        config = self.tests[test_name]
        
        # Consistent hashing for same user
        hash_value = int(hashlib.md5(f"{test_name}:{user_id}".encode()).hexdigest(), 16)
        cumulative = 0.0
        
        for variant, percent in config.variants.items():
            cumulative += percent
            if (hash_value % 100) < (cumulative * 100):
                return variant
        
        # Fallback to first variant
        return list(config.variants.keys())[0]
    
    def record_metric(self, test_name: str, variant: str, metric: str, value: float):
        """Record a metric value"""
        if test_name in self.results:
            if variant in self.results[test_name]:
                if metric in self.results[test_name][variant]:
                    self.results[test_name][variant][metric].append(value)
    
    def get_results(self, test_name: str) -> Dict:
        """Get test results"""
        if test_name not in self.results:
            return {}
        
        results = {}
        for variant, metrics in self.results[test_name].items():
            results[variant] = {
                metric: {
                    "mean": sum(values) / len(values) if values else 0,
                    "count": len(values)
                }
                for metric, values in metrics.items()
            }
        
        return results

# Usage
ab_framework = ABTestFramework()

# Register test
ab_framework.register_test(ABTestConfig(
    test_name="model_comparison",
    variants={"llama-2-7b": 0.5, "mistral-7b": 0.5},
    metrics=["latency", "quality_score", "error_rate"]
))

# Assign variant
user_id = "user123"
variant = ab_framework.assign_variant("model_comparison", user_id)

# Record metrics
ab_framework.record_metric("model_comparison", variant, "latency", 0.15)
ab_framework.record_metric("model_comparison", variant, "quality_score", 0.85)

# Get results
results = ab_framework.get_results("model_comparison")
```

### Automated Rollback

**SLO-Based Rollback:**
```python
class SLOMonitor:
    def __init__(self, slo_config: dict):
        self.slo_config = slo_config  # e.g., {"error_rate": 0.01, "p95_latency_ms": 500}
        self.metrics = []
    
    def record_request(self, latency_ms: float, error: bool):
        """Record a request"""
        self.metrics.append({
            "latency_ms": latency_ms,
            "error": error,
            "timestamp": time.time()
        })
        
        # Keep only recent metrics (last hour)
        cutoff = time.time() - 3600
        self.metrics = [m for m in self.metrics if m["timestamp"] > cutoff]
    
    def check_slo(self) -> dict:
        """Check if SLOs are being met"""
        if not self.metrics:
            return {"status": "unknown", "violations": []}
        
        violations = []
        
        # Check error rate
        error_rate = sum(1 for m in self.metrics if m["error"]) / len(self.metrics)
        if error_rate > self.slo_config.get("error_rate", 0.01):
            violations.append(f"Error rate {error_rate:.3f} exceeds SLO {self.slo_config['error_rate']}")
        
        # Check latency
        latencies = [m["latency_ms"] for m in self.metrics]
        p95_latency = np.percentile(latencies, 95)
        if p95_latency > self.slo_config.get("p95_latency_ms", 500):
            violations.append(f"P95 latency {p95_latency:.1f}ms exceeds SLO {self.slo_config['p95_latency_ms']}ms")
        
        return {
            "status": "violated" if violations else "met",
            "violations": violations,
            "error_rate": error_rate,
            "p95_latency_ms": p95_latency
        }

# Automated rollback
class AutoRollback:
    def __init__(self, canary_deployment: CanaryDeployment, slo_monitor: SLOMonitor):
        self.canary = canary_deployment
        self.slo_monitor = slo_monitor
        self.rollback_threshold = 3  # Number of consecutive violations
    
    def check_and_rollback(self):
        """Check SLO and rollback if needed"""
        slo_status = self.slo_monitor.check_slo()
        
        if slo_status["status"] == "violated":
            # Rollback canary
            self.canary.traffic_percent = 0.0
            print(f"Auto-rollback triggered: {slo_status['violations']}")
            return True
        
        return False
```

## 4. Observability and Distributed Tracing

Observability is crucial for understanding system behavior, debugging issues, and optimizing performance. This section covers metrics collection, distributed tracing, and monitoring dashboards.

### OpenTelemetry Integration

**Basic OpenTelemetry Setup:**
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

def setup_tracing(service_name: str = "llm-serving"):
    """Setup OpenTelemetry tracing"""
    resource = Resource.create({"service.name": service_name})
    
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(
        endpoint="http://localhost:4317",
        insecure=True
    ))
    provider.add_span_processor(processor)
    
    trace.set_tracer_provider(provider)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument()
    HTTPXClientInstrumentor.instrument()
    
    return trace.get_tracer(__name__)

# Usage
tracer = setup_tracing("llm-serving")

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate with tracing"""
    with tracer.start_as_current_span("generate_request") as span:
        span.set_attribute("model", request.model)
        span.set_attribute("prompt_length", len(request.prompt))
        
        # Tokenize
        with tracer.start_as_current_span("tokenize"):
            tokens = await tokenize(request.prompt)
        
        # Model inference
        with tracer.start_as_current_span("model_inference") as inference_span:
            start_time = time.time()
            result = await run_model(request.model, tokens)
            latency = time.time() - start_time
            
            inference_span.set_attribute("latency_ms", latency * 1000)
            inference_span.set_attribute("tokens_generated", result["tokens"])
        
        # Detokenize
        with tracer.start_as_current_span("detokenize"):
            text = await detokenize(result["token_ids"])
        
        span.set_attribute("response_length", len(text))
        
        return {"text": text}
```

### Metrics Collection

**Prometheus Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
request_count = Counter(
    'llm_requests_total',
    'Total number of requests',
    ['model', 'status']
)

request_latency = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

active_requests = Gauge(
    'llm_active_requests',
    'Number of active requests',
    ['model']
)

gpu_utilization = Gauge(
    'llm_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

# Instrument endpoints
@app.post("/generate")
async def generate(request: GenerationRequest):
    active_requests.labels(model=request.model).inc()
    
    start_time = time.time()
    try:
        result = await handle_generation(request)
        request_count.labels(model=request.model, status="success").inc()
        return result
    except Exception as e:
        request_count.labels(model=request.model, status="error").inc()
        raise
    finally:
        latency = time.time() - start_time
        request_latency.labels(model=request.model).observe(latency)
        active_requests.labels(model=request.model).dec()

# Start metrics server
start_http_server(8000)  # Metrics available at http://localhost:8000/metrics
```

### Distributed Tracing

**End-to-End Tracing:**
```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract

async def traced_request(request: GenerationRequest, headers: dict):
    """Handle request with distributed tracing"""
    tracer = trace.get_tracer(__name__)
    
    # Extract trace context from headers
    context = extract(headers)
    
    with tracer.start_as_current_span("llm_serving_request", context=context) as span:
        span.set_attribute("model", request.model)
        span.set_attribute("prompt_length", len(request.prompt))
        
        # Tokenize service call
        with tracer.start_as_current_span("tokenizer_service") as tokenize_span:
            tokenizer_headers = {}
            inject(tokenizer_headers)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://tokenizer:8001/tokenize",
                    json={"text": request.prompt},
                    headers=tokenizer_headers
                )
                tokens = response.json()["tokens"]
                tokenize_span.set_attribute("num_tokens", len(tokens))
        
        # Model runner call
        with tracer.start_as_current_span("model_runner") as model_span:
            model_headers = {}
            inject(model_headers)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://model-{request.model}:8002/generate",
                    json={"tokens": tokens},
                    headers=model_headers
                )
                result = response.json()
                model_span.set_attribute("tokens_generated", result["tokens_generated"])
        
        return result
```

### Logging

**Structured Logging:**
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_request(self, request_id: str, model: str, latency_ms: float, error: bool = False):
        """Log request with structured format"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR" if error else "INFO",
            "request_id": request_id,
            "model": model,
            "latency_ms": latency_ms,
            "error": error
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_metric(self, metric_name: str, value: float, tags: dict = None):
        """Log metric"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "metric",
            "metric": metric_name,
            "value": value,
            "tags": tags or {}
        }
        self.logger.info(json.dumps(log_entry))

logger = StructuredLogger("llm-serving")
```

## 5. Fault Tolerance and Cost Optimization

Production systems must handle failures gracefully and optimize costs. This section covers cold start handling, autoscaling, request queuing, and cost optimization strategies.

### Handling Cold Starts

**Warmup Strategy:**
```python
class ModelWarmup:
    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.warmed_up = False
    
    async def warmup(self):
        """Warmup model with dummy requests"""
        if self.warmed_up:
            return
        
        print("Warming up model...")
        dummy_prompts = ["warmup"] * 10
        
        # Run warmup requests
        for prompt in dummy_prompts:
            try:
                await self.model_runner.generate_async([prompt], max_tokens=1)
            except Exception as e:
                print(f"Warmup error: {e}")
        
        self.warmed_up = True
        print("Model warmed up")

# Pre-warm on startup
@app.on_event("startup")
async def startup():
    warmup = ModelWarmup(model_runner)
    await warmup.warmup()
```

**Keep-Alive Strategy:**
```python
class KeepAliveManager:
    def __init__(self, model_runner: ModelRunner, keepalive_interval: int = 300):
        self.model_runner = model_runner
        self.keepalive_interval = keepalive_interval
        self.running = False
    
    async def keepalive_loop(self):
        """Periodic keepalive requests"""
        self.running = True
        while self.running:
            await asyncio.sleep(self.keepalive_interval)
            try:
                await self.model_runner.generate_async(["keepalive"], max_tokens=1)
            except Exception as e:
                print(f"Keepalive error: {e}")
    
    def start(self):
        """Start keepalive loop"""
        asyncio.create_task(self.keepalive_loop())
    
    def stop(self):
        """Stop keepalive loop"""
        self.running = False
```

### Autoscaling

**Request-Based Autoscaling:**
```python
class Autoscaler:
    def __init__(self, min_replicas: int = 1, max_replicas: int = 10, target_rps: int = 100):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_rps = target_rps
        self.current_replicas = min_replicas
        self.request_queue = []
    
    def record_request(self):
        """Record incoming request"""
        self.request_queue.append(time.time())
        # Keep only last minute
        cutoff = time.time() - 60
        self.request_queue = [t for t in self.request_queue if t > cutoff]
    
    def get_current_rps(self) -> float:
        """Get current requests per second"""
        if not self.request_queue:
            return 0.0
        return len(self.request_queue) / 60.0
    
    def should_scale_up(self) -> bool:
        """Check if should scale up"""
        current_rps = self.get_current_rps()
        if current_rps > self.target_rps * 1.2:  # 20% over target
            if self.current_replicas < self.max_replicas:
                return True
        return False
    
    def should_scale_down(self) -> bool:
        """Check if should scale down"""
        current_rps = self.get_current_rps()
        if current_rps < self.target_rps * 0.5:  # 50% under target
            if self.current_replicas > self.min_replicas:
                return True
        return False
    
    async def scale(self):
        """Scale replicas"""
        if self.should_scale_up():
            self.current_replicas += 1
            await self.add_replica()
        elif self.should_scale_down():
            self.current_replicas -= 1
            await self.remove_replica()
    
    async def add_replica(self):
        """Add a new replica"""
        # Implementation depends on deployment platform
        print(f"Scaling up to {self.current_replicas} replicas")
    
    async def remove_replica(self):
        """Remove a replica"""
        print(f"Scaling down to {self.current_replicas} replicas")
```

### Request Queuing

**Queue with Backpressure:**
```python
import asyncio
from collections import deque

class RequestQueue:
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.max_size = max_size
    
    async def enqueue(self, request: dict) -> bool:
        """Enqueue request, return False if queue is full"""
        try:
            await asyncio.wait_for(self.queue.put(request), timeout=0.1)
            return True
        except asyncio.TimeoutError:
            return False  # Queue full
    
    async def dequeue(self) -> dict:
        """Dequeue request"""
        return await self.queue.get()
    
    def size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.queue.qsize() >= self.max_size

# Worker pool
class WorkerPool:
    def __init__(self, num_workers: int, queue: RequestQueue, model_runner: ModelRunner):
        self.num_workers = num_workers
        self.queue = queue
        self.model_runner = model_runner
        self.workers = []
    
    async def worker(self, worker_id: int):
        """Worker that processes requests"""
        while True:
            try:
                request = await self.queue.dequeue()
                result = await self.model_runner.generate_async([request["prompt"]])
                # Send result back
                await request["response_queue"].put(result)
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    def start(self):
        """Start worker pool"""
        for i in range(self.num_workers):
            worker = asyncio.create_task(self.worker(i))
            self.workers.append(worker)
    
    def stop(self):
        """Stop worker pool"""
        for worker in self.workers:
            worker.cancel()
```

### Cost Optimization

**Spot Instance Usage:**
```python
class SpotInstanceManager:
    def __init__(self, spot_ratio: float = 0.5):
        self.spot_ratio = spot_ratio
        self.spot_instances = []
        self.on_demand_instances = []
    
    def allocate_instances(self, total_instances: int):
        """Allocate mix of spot and on-demand instances"""
        num_spot = int(total_instances * self.spot_ratio)
        num_ondemand = total_instances - num_spot
        
        # Allocate spot instances (cheaper but can be preempted)
        for i in range(num_spot):
            instance = self.create_spot_instance()
            self.spot_instances.append(instance)
        
        # Allocate on-demand instances (reliable)
        for i in range(num_ondemand):
            instance = self.create_ondemand_instance()
            self.on_demand_instances.append(instance)
    
    def handle_preemption(self, instance_id: str):
        """Handle spot instance preemption"""
        # Remove preempted instance
        self.spot_instances = [i for i in self.spot_instances if i.id != instance_id]
        
        # Replace with on-demand if needed
        if len(self.spot_instances) < int(len(self.on_demand_instances) * self.spot_ratio):
            new_spot = self.create_spot_instance()
            self.spot_instances.append(new_spot)
```

**Model Selection for Cost:**
```python
class CostOptimizedRouter:
    def __init__(self):
        self.models = {
            "llama-2-7b": {"cost_per_token": 0.001, "latency_ms": 100},
            "llama-2-13b": {"cost_per_token": 0.002, "latency_ms": 200},
            "mistral-7b": {"cost_per_token": 0.0015, "latency_ms": 150}
        }
    
    def route(self, request: dict) -> str:
        """Route to cost-optimized model"""
        cost_sensitive = request.get("cost_sensitive", False)
        latency_budget = request.get("latency_budget_ms", 1000)
        
        if cost_sensitive:
            # Select cheapest model within latency budget
            available = [
                (model, config) for model, config in self.models.items()
                if config["latency_ms"] <= latency_budget
            ]
            if available:
                return min(available, key=lambda x: x[1]["cost_per_token"])[0]
        
        # Default to fastest
        return min(self.models.items(), key=lambda x: x[1]["latency_ms"])[0]
```

## 6. Local Kubernetes Setup with k3d for GPU Model Serving

While production deployments often use managed Kubernetes services, setting up a local Kubernetes cluster is invaluable for development, testing, and learning. This section covers setting up a GPU-enabled Kubernetes cluster using k3d (k3s in Docker), which provides a lightweight, production-like environment that runs entirely in Docker containers.

### Why k3d for Local Development?

**Advantages:**

- **Lightweight:** Runs in Docker, no need for VMs or complex setup
- **Fast setup:** Create a cluster in minutes
- **Real Kubernetes API:** Fully compatible with standard Kubernetes manifests
- **GPU support:** Can expose host GPUs to containers
- **Multi-node:** Easy to create multi-node clusters for testing
- **No root required:** Runs in user space (with proper Docker permissions)

**Use cases:**

- Local development and testing of GPU workloads
- Learning Kubernetes concepts
- Prototyping production deployments
- CI/CD pipeline testing

### Prerequisites

Before setting up k3d, ensure your system has the following:

**1. NVIDIA Drivers**

Verify NVIDIA drivers are installed:

```bash
nvidia-smi
```

If not installed, install drivers for your system.

**2. NVIDIA Container Toolkit**

The NVIDIA Container Toolkit enables containers to access GPUs:

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**3. Docker and k3d**

Install Docker (if not already installed) and k3d:

```bash
# Install k3d
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Verify installation
k3d --version
```

### Building a Custom GPU-Enabled k3s Image

The default k3s image doesn't include NVIDIA Container Toolkit support. We need to build a custom image that includes both k3s and NVIDIA runtime support.

**Step 1: Create Dockerfile**

Create a Dockerfile that combines k3s with CUDA and NVIDIA Container Toolkit:

```dockerfile
ARG K3S_TAG="v1.33.6-k3s1"
ARG CUDA_TAG="12.4.1-base-ubuntu22.04"

FROM rancher/k3s:$K3S_TAG as k3s
FROM nvcr.io/nvidia/cuda:$CUDA_TAG

# Install the NVIDIA container toolkit
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update && apt-get install -y nvidia-container-toolkit \
    && nvidia-ctk runtime configure --runtime=containerd

# Copy k3s binaries
COPY --from=k3s / / --exclude=/bin
COPY --from=k3s /bin /bin

# Deploy the nvidia device plugin on startup
COPY device-plugin-daemonset.yaml /var/lib/rancher/k3s/server/manifests/nvidia-device-plugin-daemonset.yaml

VOLUME /var/lib/kubelet
VOLUME /var/lib/rancher/k3s
VOLUME /var/lib/cni
VOLUME /var/log

ENV PATH="$PATH:/bin/aux"

ENTRYPOINT ["/bin/k3s"]
CMD ["agent"]
```

**Step 2: Create NVIDIA Device Plugin Manifest**

Create `device-plugin-daemonset.yaml`:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  updateStrategy:
    type: RollingUpdate
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      priorityClassName: "system-node-critical"
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.17.3
        name: nvidia-device-plugin-ctr
        env:
        - name: FAIL_ON_INIT_ERROR
          value: "false"
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
      nodeSelector:
        accelerator: nvidia-tesla-k80
```

**Step 3: Build the Custom Image**

Build the image using Docker BuildKit (required for the `--exclude` flag):

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build the image
docker build \
  --build-arg K3S_TAG=v1.33.6-k3s1 \
  --build-arg CUDA_TAG=12.2.0-base-ubuntu22.04 \
  -t k3s-cuda:v1.33.6-cuda-12.2.0 \
  .

# Verify the image
docker images | grep k3s-cuda
```

**Note:** If you encounter issues with the `--exclude` flag, ensure Docker BuildKit is enabled. You may need to install `docker buildx`:

```bash
# Install buildx
docker buildx version || docker plugin install --grant-all-permissions moby/buildx

# Use buildx for building (recommended)
docker buildx build \
  --build-arg K3S_TAG=v1.33.6-k3s1 \
  --build-arg CUDA_TAG=12.2.0-base-ubuntu22.04 \
  -t k3s-cuda:v1.33.6-cuda-12.2.0 \
  --load .

# Note: CUDA 12.4.1 also works, but 12.2.0 is tested and recommended
```

### Creating a 2-Node GPU Cluster

**Step 1: Create the Cluster**

**Important:** You must use the custom `k3s-cuda` image built in the previous section. This image provides GPU support for the Kubernetes cluster nodes.

Create a 2-node cluster (1 control-plane + 1 worker) with GPU support:

```bash
# Option 1: Use all available GPUs
k3d cluster create mycluster-gpu \
  --image k3s-cuda:v1.33.6-cuda-12.2.0 \
  --gpus=all \
  --servers 1 \
  --agents 1

# Option 2: Use all GPUs + mount model directory (for vLLM deployment)
k3d cluster create mycluster-gpu \
  --image k3s-cuda:v1.33.6-cuda-12.2.0 \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume /raid/models:/models

# Option 3: Use specific GPUs (e.g., GPU 4 and 5)
k3d cluster create mycluster-gpu \
  --image k3s-cuda:v1.33.6-cuda-12.2.0 \
  --gpus "device=4,5" \
  --servers 1 \
  --agents 1

# Note: CUDA version can be 12.2.0 or 12.4.1 (both tested and working)
# The --image flag uses the k3s-cuda image you built in Step 3 above
```

**Step 2: Configure kubectl**

Set up kubeconfig to access the cluster:

```bash
# Merge k3d kubeconfig
k3d kubeconfig merge mycluster-gpu --kubeconfig-merge-default

# Verify access
export KUBECONFIG=$HOME/.kube/config
kubectl get nodes
```

**Note:** If you see errors like `The connection to the server localhost:8080 was refused`, the kubeconfig may have an incorrect server address. Fix it:

```bash
# Check current server address
kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-mycluster-gpu")].cluster.server}'

# If it shows 0.0.0.0, update to 127.0.0.1
KUBE_SERVER=$(kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-mycluster-gpu")].cluster.server}')
kubectl config set-cluster k3d-mycluster-gpu --server=$(echo $KUBE_SERVER | sed 's/0.0.0.0/127.0.0.1/')

# Verify again
kubectl get nodes
```

You should see both nodes:

```
NAME                         STATUS   ROLES           AGE   VERSION
k3d-mycluster-gpu-server-0   Ready    control-plane   30s   v1.33.6+k3s1
k3d-mycluster-gpu-agent-0    Ready    <none>          25s   v1.33.6+k3s1
```

**Step 3: Verify GPU Visibility**

The NVIDIA device plugin is automatically deployed by the custom k3s image (via the manifest in `/var/lib/rancher/k3s/server/manifests/`). Check if GPUs are visible to Kubernetes:

```bash
# Wait for device plugin to be ready (it should start automatically)
kubectl wait --for=condition=ready pod -l name=nvidia-device-plugin-ds -n kube-system --timeout=120s

# Verify device plugin pods are running
kubectl get pods -n kube-system | grep nvidia

# Check GPU resources on nodes
kubectl describe node k3d-mycluster-gpu-server-0 | grep nvidia.com/gpu
kubectl describe node k3d-mycluster-gpu-agent-0 | grep nvidia.com/gpu
```

You should see output like:

```
Capacity:
  nvidia.com/gpu:  8
Allocatable:
  nvidia.com/gpu:  8
```

**Step 4: Test GPU Access**

Create a simple test pod to verify GPU access:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: OnFailure
  runtimeClassName: nvidia
  containers:
  - name: cuda
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
```

Apply and verify:

```bash
kubectl apply -f gpu-test.yaml
kubectl wait --for=condition=Ready pod/gpu-test --timeout=60s
kubectl logs gpu-test
```

You should see `nvidia-smi` output showing GPU information.

### Deploying vLLM with Official Image and Hugging Face Models

This section demonstrates deploying vLLM using the official `vllm/vllm-openai:latest` Docker image with models downloaded from Hugging Face. This is the recommended approach for production deployments as it uses the official, tested vLLM image.

**Prerequisites:**

- GPU-enabled Kubernetes cluster (as set up in previous sections)
- Access to Hugging Face (for model downloads)
- Optional: Hugging Face token for gated models

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

## 7. Kubernetes Deployment with llm-d

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



## Hands-On Examples

### Example 1: API Gateway with Routing

**File:** `examples/ch09_api_gateway.py`

```python
"""
Complete API gateway implementation with routing and rate limiting.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time
from collections import defaultdict

app = FastAPI()

# Rate limiter
class RateLimiter:
    def __init__(self, max_requests=100, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        client_reqs = self.requests[client_id]
        client_reqs[:] = [t for t in client_reqs if now - t < self.window]
        
        if len(client_reqs) >= self.max_requests:
            return False
        
        client_reqs.append(now)
        return True

rate_limiter = RateLimiter()

# Model routing
MODEL_ENDPOINTS = {
    "llama-2-7b": "http://localhost:8002",
    "mistral-7b": "http://localhost:8003"
}

@app.post("/generate")
async def generate(request: dict, client_id: str = "default"):
    # Rate limiting
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(429, "Rate limit exceeded")
    
    # Route to model
    model = request.get("model", "llama-2-7b")
    endpoint = MODEL_ENDPOINTS.get(model)
    if not endpoint:
        raise HTTPException(404, f"Model {model} not found")
    
    # Forward request
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{endpoint}/generate", json=request)
        return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Example 2: Canary Deployment

**File:** `examples/ch09_canary_deployment.py`

```python
"""
Canary deployment with automated rollback.
"""
import random
from collections import defaultdict

class CanaryDeployment:
    def __init__(self, stable="stable-v1", canary="canary-v2", traffic=0.1):
        self.stable = stable
        self.canary = canary
        self.traffic_percent = traffic
        self.metrics = defaultdict(lambda: {"requests": 0, "errors": 0})
    
    def route(self) -> str:
        return self.canary if random.random() < self.traffic_percent else self.stable
    
    def record(self, model: str, error: bool):
        self.metrics[model]["requests"] += 1
        if error:
            self.metrics[model]["errors"] += 1
    
    def should_rollback(self) -> bool:
        canary_error_rate = self.metrics[self.canary]["errors"] / max(
            self.metrics[self.canary]["requests"], 1
        )
        stable_error_rate = self.metrics[self.stable]["errors"] / max(
            self.metrics[self.stable]["requests"], 1
        )
        return canary_error_rate > stable_error_rate * 2.0

# Usage
canary = CanaryDeployment(traffic=0.1)
model = canary.route()
# ... process request ...
canary.record(model, error=False)

if canary.should_rollback():
    canary.traffic_percent = 0.0  # Rollback
```

### Example 3: OpenTelemetry Tracing

**File:** `examples/ch09_tracing.py`

```python
"""
OpenTelemetry instrumentation example.
"""
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Instrument code
@app.post("/generate")
async def generate(request: dict):
    with tracer.start_as_current_span("generate") as span:
        span.set_attribute("model", request["model"])
        
        with tracer.start_as_current_span("tokenize"):
            tokens = tokenize(request["prompt"])
        
        with tracer.start_as_current_span("inference") as inf_span:
            result = await model_inference(tokens)
            inf_span.set_attribute("tokens", result["tokens"])
        
        return result
```

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

## Further Reading

- OpenTelemetry: https://opentelemetry.io/
- Envoy Proxy: https://www.envoyproxy.io/
- Prometheus: https://prometheus.io/
- FastAPI: https://fastapi.tiangolo.com/
- vLLM: https://github.com/vllm-project/vllm
- llm-d: https://github.com/llm-d/llm-d - Production-ready Kubernetes deployment for LLM inference
- llm-d Documentation: https://www.llm-d.ai/ - Complete guides and well-lit paths
- Inference Gateway: https://github.com/kserve/inference-gateway - Request scheduler and balancer
