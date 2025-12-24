# Appendix

## OpenAI-Compatible API Endpoints

Both vLLM and SGLang expose OpenAI-compatible APIs.

The following table lists common endpoints and their usage:

| Endpoint | Method | Description | Usage |
|------------------|--------|-----------------------|---------------------------|
| `/v1/models` | GET | List available models | `curl http://localhost:8000/v1/models` |
| `/v1/completions` | POST | Text completion for base models | Use with `prompt` parameter for base models |
| `/v1/chat/completions` | POST | Chat completion for instruction-tuned models | Use with `messages` parameter for chat models |
| `/v1/embeddings` | POST | Generate embeddings from text | Use with `input` parameter for embedding models |
| `/v1/rerank` | POST | Rerank documents by relevance to a query | Use with `query` and `documents` parameters (SGLang only) |
| `/v1/classify` | POST | Classify text inputs | Use with `input` parameter for classification models |
| `/generate` | POST | Generate text using diffusion models | Use with `text` parameter for diffusion language models (SGLang only) |
| `/health` | GET | Health check endpoint | `curl http://localhost:8000/health` |
| `/metrics` | GET | Prometheus metrics | `curl http://localhost:8000/metrics` |
| `/docs` | GET | API documentation (Swagger UI) | Open in browser: `http://localhost:8000/docs` |

**Note:** The default port is 8000 for vLLM and 30000 for SGLang. Adjust the port in the examples above accordingly. For detailed API documentation and model-specific requirements, refer to the respective documentation:

- [vLLM API Reference](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [SGLang Documentation](https://docs.sglang.io/)
