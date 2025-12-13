"""
Federated learning using Flower framework.
"""
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Client implementation
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # loss, accuracy = test(self.model, self.valloader)
        return 0.0, len(self.valloader), {"accuracy": 0.0}

# Example usage
if __name__ == "__main__":
    # Start Flower server
    # fl.server.start_server(
    #     server_address="0.0.0.0:8080",
    #     config=fl.server.ServerConfig(num_rounds=100),
    # )
    
    # Start Flower client
    # fl.client.start_numpy_client(
    #     server_address="localhost:8080",
    #     client=FlowerClient(model, trainloader, valloader)
    # )
    print("Flower federated learning example - requires Flower framework")

