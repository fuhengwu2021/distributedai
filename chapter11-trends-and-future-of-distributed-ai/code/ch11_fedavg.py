"""
Complete FedAvg implementation from scratch.
"""
import torch
import torch.nn as nn
from collections import OrderedDict

def fedavg_aggregate(client_models, client_weights=None):
    """Federated averaging"""
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    
    averaged_state = OrderedDict()
    param_names = client_models[0].keys()
    
    for param_name in param_names:
        weighted_sum = None
        total_weight = 0.0
        
        for model_state, weight in zip(client_models, client_weights):
            param = model_state[param_name]
            
            if weighted_sum is None:
                weighted_sum = param * weight
            else:
                weighted_sum += param * weight
            
            total_weight += weight
        
        averaged_state[param_name] = weighted_sum / total_weight
    
    return averaged_state

# Example usage
if __name__ == "__main__":
    # Example: Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
    
    # Create dummy client models
    client1 = SimpleModel()
    client2 = SimpleModel()
    client3 = SimpleModel()
    
    # Get their state dicts
    client_models = [
        client1.state_dict(),
        client2.state_dict(),
        client3.state_dict()
    ]
    
    # Aggregate
    averaged_state = fedavg_aggregate(client_models)
    
    # Load into global model
    global_model = SimpleModel()
    global_model.load_state_dict(averaged_state)
    
    print("FedAvg aggregation completed")

