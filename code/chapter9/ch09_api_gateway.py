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

