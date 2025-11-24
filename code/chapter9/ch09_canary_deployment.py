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
if __name__ == "__main__":
    canary = CanaryDeployment(traffic=0.1)
    model = canary.route()
    # ... process request ...
    canary.record(model, error=False)
    
    if canary.should_rollback():
        canary.traffic_percent = 0.0  # Rollback

