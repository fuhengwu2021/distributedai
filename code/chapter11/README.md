# Chapter 11: Federated Learning and Edge Distributed Systems - Code Examples

This directory contains code examples for federated learning and edge deployment.

## Files

- `ch11_flower_federated.py` - Federated learning using Flower framework
- `ch11_fedavg.py` - FedAvg (Federated Averaging) implementation from scratch
- `ch11_edge_deployment.py` - Edge deployment pipeline for NVIDIA Jetson

## Prerequisites

### For Federated Learning
```bash
pip install flwr torch torchvision
```

### For Edge Deployment
```bash
pip install torch torchvision onnx
# For TensorRT (optional, requires NVIDIA GPU)
# pip install tensorrt
```

## Running Examples

### Flower Federated Learning
```bash
# Start server
python ch11_flower_federated.py --server

# Start client (in another terminal)
python ch11_flower_federated.py --client
```

### FedAvg
```bash
python ch11_fedavg.py
```

### Edge Deployment
```bash
# Prepare model for edge
python ch11_edge_deployment.py
# Output: model.onnx
```

