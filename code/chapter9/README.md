# Chapter 9: Production LLM Serving Stack - Code Examples

This directory contains code examples for building a complete production LLM serving stack.

## Files

- `tokenizer_service.py` - Tokenizer service implementation with FastAPI
- `model_runner.py` - Model runner using vLLM for inference
- `ch09_api_gateway.py` - Complete API gateway with routing and rate limiting
- `ch09_canary_deployment.py` - Canary deployment with automated rollback
- `ch09_tracing.py` - OpenTelemetry tracing instrumentation example

## Prerequisites

```bash
pip install fastapi uvicorn transformers vllm httpx opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

## Running Examples

### Tokenizer Service
```bash
python tokenizer_service.py
# Service runs on http://localhost:8001
```

### API Gateway
```bash
python ch09_api_gateway.py
# Gateway runs on http://localhost:8000
```

### Canary Deployment
```bash
python ch09_canary_deployment.py
```

### Tracing
```bash
# Requires OpenTelemetry collector running
python ch09_tracing.py
```

