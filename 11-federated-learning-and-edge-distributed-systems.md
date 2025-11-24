# Chapter 11: Federated Learning and Edge Distributed Systems

## Overview

Readers learn how federated learning distributes model training across clients while preserving data privacy. This chapter covers aggregation algorithms, differential privacy, secure aggregation, non-IID data, and deployment on edge devices. It concludes with an edge-cloud hybrid inference system that combines the benefits of on-device and cloud-based inference.

**Chapter Length:** 30 pages

---

## 1. Federated Learning Fundamentals

Federated learning is a distributed machine learning paradigm that enables training models across multiple clients without centralizing their data. Instead of sending raw data to a central server, clients train models locally and share only model updates. This approach addresses privacy concerns, reduces communication costs, and enables training on edge devices.

### Why Federated Learning?

**Key Motivations:**
- **Privacy Preservation:** Data never leaves the client device
- **Regulatory Compliance:** Meets GDPR, HIPAA, and other privacy regulations
- **Reduced Communication:** Only model updates are transmitted, not raw data
- **Edge Deployment:** Enables training on resource-constrained devices
- **Data Ownership:** Clients maintain control over their data

### Federated Learning Architecture

**Centralized Federated Learning:**
```
┌─────────┐
│ Server  │ (Coordinates training)
└────┬────┘
     │
     ├───┐
     │   │
┌────▼───▼────┐
│   Clients   │ (Train locally)
│  (Devices)  │
└─────────────┘
```

**Decentralized Federated Learning:**
```
┌──────┐     ┌──────┐     ┌──────┐
│Client│◄───►│Client│◄───►│Client│
└──────┘     └──────┘     └──────┘
   ▲                          │
   └──────────────────────────┘
   (Peer-to-peer aggregation)
```

### Federated Averaging (FedAvg) Algorithm

**Basic FedAvg:**
```python
import torch
import torch.nn as nn
from collections import OrderedDict

def federated_averaging(client_models, client_weights=None):
    """
    Federated averaging algorithm.
    
    Args:
        client_models: List of model state dicts from clients
        client_weights: Optional weights for each client (by default, equal weight)
    
    Returns:
        Averaged model state dict
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    
    # Initialize averaged model
    averaged_state = OrderedDict()
    
    # Get parameter names from first model
    param_names = client_models[0].keys()
    
    # Weighted average of each parameter
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
client_1_model = model_1.state_dict()
client_2_model = model_2.state_dict()
client_3_model = model_3.state_dict()

averaged_model = federated_averaging(
    [client_1_model, client_2_model, client_3_model]
)

# Load averaged model
global_model.load_state_dict(averaged_model)
```

### Federated Learning Workflow

**Complete FedAvg Implementation:**
```python
class FederatedLearningServer:
    def __init__(self, model, num_clients):
        self.global_model = model
        self.num_clients = num_clients
        self.client_updates = []
        self.client_weights = []
    
    def select_clients(self, fraction=0.1):
        """Select a fraction of clients for this round"""
        num_selected = max(1, int(self.num_clients * fraction))
        selected = np.random.choice(
            self.num_clients,
            size=num_selected,
            replace=False
        )
        return selected
    
    def aggregate_updates(self):
        """Aggregate client updates using FedAvg"""
        if not self.client_updates:
            return
        
        # Weighted average
        averaged_state = federated_averaging(
            self.client_updates,
            self.client_weights
        )
        
        # Update global model
        self.global_model.load_state_dict(averaged_state)
        
        # Clear updates for next round
        self.client_updates = []
        self.client_weights = []
    
    def distribute_model(self):
        """Send global model to clients"""
        return self.global_model.state_dict()

class FederatedLearningClient:
    def __init__(self, model, local_data, client_id):
        self.model = model
        self.local_data = local_data
        self.client_id = client_id
    
    def train_local(self, global_state, num_epochs=1, lr=0.01):
        """Train model on local data"""
        # Load global model
        self.model.load_state_dict(global_state)
        
        # Local training
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(num_epochs):
            for data, target in self.local_data:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Return updated model
        return self.model.state_dict()
    
    def get_data_size(self):
        """Return size of local dataset (for weighted averaging)"""
        return len(self.local_data.dataset)

# Federated learning loop
def federated_training(server, clients, num_rounds=100):
    """Main federated learning loop"""
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        
        # Select clients
        selected_clients = server.select_clients(fraction=0.1)
        
        # Distribute global model
        global_state = server.distribute_model()
        
        # Local training
        for client_id in selected_clients:
            client = clients[client_id]
            updated_state = client.train_local(global_state)
            data_size = client.get_data_size()
            
            # Send update to server
            server.client_updates.append(updated_state)
            server.client_weights.append(data_size)
        
        # Aggregate updates
        server.aggregate_updates()
        
        # Evaluate (optional)
        if (round_num + 1) % 10 == 0:
            evaluate_global_model(server.global_model)
```

### Communication Efficiency

**Reducing Communication Costs:**
```python
def compress_updates(model_state, compression_ratio=0.1):
    """Compress model updates to reduce communication"""
    compressed = {}
    
    for name, param in model_state.items():
        # Top-k sparsification
        k = int(param.numel() * compression_ratio)
        flat_param = param.flatten()
        _, indices = torch.topk(flat_param.abs(), k)
        
        # Only send top-k values and indices
        compressed[name] = {
            'values': flat_param[indices],
            'indices': indices,
            'shape': param.shape
        }
    
    return compressed

def decompress_updates(compressed, original_shape):
    """Decompress model updates"""
    decompressed = {}
    
    for name, comp_data in compressed.items():
        # Reconstruct parameter
        param = torch.zeros(comp_data['shape'])
        flat_param = param.flatten()
        flat_param[comp_data['indices']] = comp_data['values']
        decompressed[name] = param.reshape(comp_data['shape'])
    
    return decompressed
```

---

## 2. Privacy and Secure Aggregation

Privacy is a fundamental concern in federated learning. Even though raw data doesn't leave clients, model updates can leak information about the training data. This section covers techniques to protect privacy while maintaining model utility.

### Differential Privacy

**DP-SGD (Differentially Private Stochastic Gradient Descent):**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class DPSGD:
    def __init__(self, model, lr=0.01, noise_scale=1.0, clip_norm=1.0):
        self.model = model
        self.lr = lr
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
    
    def clip_gradients(self, model):
        """Clip gradients to bound sensitivity"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        clip_coef = self.clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise(self, model):
        """Add Gaussian noise to gradients for differential privacy"""
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=self.noise_scale,
                    size=param.grad.shape
                ).to(param.device)
                param.grad.data.add_(noise)
    
    def step(self, model):
        """DP-SGD step"""
        # Clip gradients
        self.clip_gradients(model)
        
        # Add noise
        self.add_noise(model)
        
        # Update parameters
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= self.lr * param.grad.data

# Usage in training
def train_with_dp(model, dataloader, dp_sgd, num_epochs=10):
    """Train model with differential privacy"""
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for data, target in dataloader:
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # DP-SGD step
            dp_sgd.step(model)
            
            # Zero gradients
            model.zero_grad()
```

### Privacy Budget Tracking

**Epsilon-Delta Privacy:**
```python
class PrivacyAccountant:
    def __init__(self, delta=1e-5):
        self.delta = delta
        self.epsilon = 0.0
        self.steps = 0
    
    def add_step(self, noise_multiplier, sample_rate):
        """
        Add a training step to privacy budget.
        
        Uses composition theorem for DP-SGD.
        """
        # Simplified privacy accounting
        # In practice, use more sophisticated methods (RDP, etc.)
        step_epsilon = noise_multiplier * sample_rate
        self.epsilon += step_epsilon
        self.steps += 1
    
    def get_privacy_spent(self):
        """Get total privacy spent"""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'steps': self.steps
        }
    
    def check_budget(self, max_epsilon):
        """Check if privacy budget is exceeded"""
        return self.epsilon < max_epsilon

# Usage
accountant = PrivacyAccountant(delta=1e-5)
noise_multiplier = 1.0
sample_rate = 0.01  # 1% of data per batch

for step in range(1000):
    accountant.add_step(noise_multiplier, sample_rate)
    
    if not accountant.check_budget(max_epsilon=10.0):
        print("Privacy budget exceeded!")
        break

print(f"Privacy spent: {accountant.get_privacy_spent()}")
```

### Secure Aggregation

**Cryptographic Secure Aggregation:**
```python
import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

class SecureAggregation:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.keys = self.generate_key_pairs()
    
    def generate_key_pairs(self):
        """Generate RSA key pairs for each client"""
        keys = {}
        for i in range(self.num_clients):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            keys[i] = {
                'private': private_key,
                'public': public_key
            }
        return keys
    
    def encrypt_update(self, client_id, update):
        """Encrypt model update"""
        public_key = self.keys[client_id]['public']
        
        # Serialize update
        serialized = self.serialize_update(update)
        
        # Encrypt
        encrypted = public_key.encrypt(
            serialized,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted
    
    def aggregate_encrypted(self, encrypted_updates):
        """Aggregate encrypted updates (homomorphic encryption)"""
        # Simplified: In practice, use proper homomorphic encryption
        # This is a placeholder for the concept
        aggregated = {}
        for update in encrypted_updates:
            # Homomorphic addition would happen here
            pass
        return aggregated
    
    def serialize_update(self, update):
        """Serialize model update for encryption"""
        # Convert to bytes
        import pickle
        return pickle.dumps(update)
```

### Privacy-Utility Tradeoff

**Analyzing Tradeoffs:**
```python
def evaluate_privacy_utility_tradeoff(model, test_data, epsilon_values):
    """Evaluate model performance at different privacy levels"""
    results = []
    
    for epsilon in epsilon_values:
        # Calculate noise scale for target epsilon
        noise_scale = calculate_noise_scale(epsilon)
        
        # Train with DP
        dp_sgd = DPSGD(model, noise_scale=noise_scale)
        train_with_dp(model, train_data, dp_sgd)
        
        # Evaluate
        accuracy = evaluate(model, test_data)
        
        results.append({
            'epsilon': epsilon,
            'noise_scale': noise_scale,
            'accuracy': accuracy
        })
    
    return results

# Plot tradeoff
import matplotlib.pyplot as plt

results = evaluate_privacy_utility_tradeoff(model, test_data, [0.1, 1.0, 10.0, 100.0])
epsilons = [r['epsilon'] for r in results]
accuracies = [r['accuracy'] for r in results]

plt.plot(epsilons, accuracies)
plt.xlabel('Privacy (epsilon)')
plt.ylabel('Accuracy')
plt.title('Privacy-Utility Tradeoff')
plt.show()
```

---

## 3. Non-IID Data and Robust Aggregation

Real-world federated learning scenarios often involve non-IID (non-independent and identically distributed) data, where clients have different data distributions. This section covers techniques to handle heterogeneous data and robust aggregation methods.

### Non-IID Data Challenges

**Types of Non-IID:**
1. **Label Skew:** Different clients have different class distributions
2. **Feature Skew:** Same labels but different feature distributions
3. **Quantity Skew:** Vastly different amounts of data per client
4. **Temporal Skew:** Data collected at different times

**Example: Label Skew:**
```python
def create_non_iid_data(num_clients=10, num_classes=10):
    """Create non-IID data distribution"""
    # Each client has data from only 2 classes
    client_data = {}
    
    for client_id in range(num_clients):
        # Assign 2 random classes to each client
        classes = np.random.choice(num_classes, size=2, replace=False)
        
        # Generate data for assigned classes
        data = []
        labels = []
        for class_id in classes:
            # Generate 100 samples per class
            class_data = np.random.randn(100, 784)  # MNIST-like
            class_labels = np.full(100, class_id)
            data.append(class_data)
            labels.append(class_labels)
        
        client_data[client_id] = {
            'data': np.vstack(data),
            'labels': np.hstack(labels)
        }
    
    return client_data
```

### Robust Aggregation Methods

**1. Median Aggregation (Robust to Outliers):**
```python
def median_aggregation(client_models):
    """Aggregate using median (robust to outliers)"""
    averaged_state = OrderedDict()
    param_names = client_models[0].keys()
    
    for param_name in param_names:
        # Stack all parameters
        stacked = torch.stack([
            model[param_name] for model in client_models
        ])
        
        # Compute median
        median_param, _ = torch.median(stacked, dim=0)
        averaged_state[param_name] = median_param
    
    return averaged_state
```

**2. Trimmed Mean (Remove Outliers):**
```python
def trimmed_mean_aggregation(client_models, trim_ratio=0.1):
    """Trimmed mean aggregation"""
    averaged_state = OrderedDict()
    param_names = client_models[0].keys()
    num_clients = len(client_models)
    trim_count = int(num_clients * trim_ratio)
    
    for param_name in param_names:
        # Stack all parameters
        stacked = torch.stack([
            model[param_name] for model in client_models
        ])
        
        # Sort and trim
        sorted_stack, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_stack[trim_count:-trim_count]
        
        # Compute mean of trimmed values
        mean_param = torch.mean(trimmed, dim=0)
        averaged_state[param_name] = mean_param
    
    return averaged_state
```

**3. Adaptive Weighting:**
```python
def adaptive_weighted_aggregation(client_models, client_losses):
    """
    Weight clients based on their performance.
    Better performing clients get higher weights.
    """
    # Convert losses to weights (inverse relationship)
    # Lower loss = higher weight
    losses = np.array(client_losses)
    weights = 1.0 / (losses + 1e-6)  # Add small epsilon to avoid division by zero
    weights = weights / weights.sum()  # Normalize
    
    # Weighted average
    averaged_state = OrderedDict()
    param_names = client_models[0].keys()
    
    for param_name in param_names:
        weighted_sum = None
        
        for model, weight in zip(client_models, weights):
            param = model[param_name]
            
            if weighted_sum is None:
                weighted_sum = param * weight
            else:
                weighted_sum += param * weight
        
        averaged_state[param_name] = weighted_sum
    
    return averaged_state
```

### FedProx: Handling Non-IID Data

**FedProx Algorithm:**
```python
class FedProxClient:
    def __init__(self, model, local_data, mu=0.01):
        self.model = model
        self.local_data = local_data
        self.mu = mu  # Proximal term coefficient
        self.global_state = None
    
    def train_local(self, global_state, num_epochs=1, lr=0.01):
        """FedProx local training with proximal term"""
        self.global_state = global_state
        self.model.load_state_dict(global_state)
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(num_epochs):
            for data, target in self.local_data:
                optimizer.zero_grad()
                
                # Standard loss
                output = self.model(data)
                loss = criterion(output, target)
                
                # Proximal term (penalize deviation from global model)
                proximal_term = 0.0
                for local_param, global_param in zip(
                    self.model.parameters(),
                    self._get_global_params()
                ):
                    proximal_term += torch.norm(local_param - global_param) ** 2
                
                # Total loss
                total_loss = loss + (self.mu / 2) * proximal_term
                total_loss.backward()
                optimizer.step()
        
        return self.model.state_dict()
    
    def _get_global_params(self):
        """Get global model parameters"""
        global_model = type(self.model)()
        global_model.load_state_dict(self.global_state)
        return global_model.parameters()
```

### Handling Quantity Skew

**Weighted Averaging by Data Size:**
```python
def weighted_fedavg(client_models, client_data_sizes):
    """FedAvg with data-size weighting"""
    total_size = sum(client_data_sizes)
    weights = [size / total_size for size in client_data_sizes]
    
    return federated_averaging(client_models, weights)
```

---

## 4. Edge Deployment and Optimization

Deploying models on edge devices requires careful optimization to fit resource constraints while maintaining acceptable performance. This section covers quantization, pruning, distillation, and hardware-specific optimizations.

### Model Quantization

**Post-Training Quantization:**
```python
import torch.quantization as quantization

def quantize_model(model, calibration_data):
    """Quantize model to INT8"""
    # Set quantization config
    model.eval()
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    
    # Prepare model
    quantization.prepare(model, inplace=True)
    
    # Calibrate with sample data
    with torch.no_grad():
        for data, _ in calibration_data:
            _ = model(data)
    
    # Convert to quantized model
    quantized_model = quantization.convert(model, inplace=False)
    
    return quantized_model

# Dynamic quantization (simpler, no calibration needed)
def dynamic_quantize(model):
    """Dynamic quantization"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model
```

**Quantization-Aware Training:**
```python
def quantization_aware_training(model, train_data, num_epochs=10):
    """Train model with quantization awareness"""
    # Set QAT config
    model.train()
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    quantization.prepare_qat(model, inplace=True)
    
    # Train normally
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for data, target in train_data:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Convert to quantized
    model.eval()
    quantized_model = quantization.convert(model, inplace=False)
    
    return quantized_model
```

### Model Pruning

**Magnitude-Based Pruning:**
```python
def prune_model(model, pruning_ratio=0.5):
    """Prune model by removing smallest magnitude weights"""
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    # Prune
    pruning.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning.L1Unstructured,
        amount=pruning_ratio
    )
    
    return model

# Iterative pruning
def iterative_pruning(model, train_data, num_iterations=5, pruning_ratio=0.2):
    """Iteratively prune and fine-tune"""
    for iteration in range(num_iterations):
        # Prune
        prune_model(model, pruning_ratio)
        
        # Fine-tune
        fine_tune(model, train_data, num_epochs=5)
    
    return model
```

### Knowledge Distillation

**Teacher-Student Distillation:**
```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Hard targets
        student_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss

def distill(teacher_model, student_model, train_data, num_epochs=20):
    """Knowledge distillation training"""
    criterion = DistillationLoss(temperature=3.0, alpha=0.7)
    optimizer = torch.optim.Adam(student_model.parameters())
    
    teacher_model.eval()
    student_model.train()
    
    for epoch in range(num_epochs):
        for data, target in train_data:
            optimizer.zero_grad()
            
            # Teacher predictions
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            
            # Student predictions
            student_logits = student_model(data)
            
            # Distillation loss
            loss = criterion(student_logits, teacher_logits, target)
            loss.backward()
            optimizer.step()
    
    return student_model
```

### Hardware-Specific Optimizations

**TensorRT Optimization:**
```python
# TensorRT conversion (pseudo-code, actual API may differ)
def convert_to_tensorrt(model, example_input):
    """Convert PyTorch model to TensorRT"""
    import tensorrt as trt
    
    # Export to ONNX first
    torch.onnx.export(model, example_input, "model.onnx")
    
    # Convert ONNX to TensorRT
    # (Actual conversion requires TensorRT SDK)
    # trt_engine = build_engine("model.onnx")
    
    return trt_engine
```

**ONNX Runtime:**
```python
import onnx
import onnxruntime as ort

def convert_to_onnx(model, example_input, output_path="model.onnx"):
    """Convert PyTorch model to ONNX"""
    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def run_onnx_inference(model_path, input_data):
    """Run inference with ONNX Runtime"""
    session = ort.InferenceSession(model_path)
    
    outputs = session.run(
        None,
        {'input': input_data.numpy()}
    )
    
    return outputs[0]
```

---

## 5. Edge-Cloud Hybrid Inference Systems

Edge-cloud hybrid systems combine the benefits of on-device inference (low latency, privacy) with cloud inference (high accuracy, complex models). This section covers architectures and implementation strategies.

### Hybrid Architecture

**System Design:**
```
┌─────────────┐
│   Client     │
│  (Edge)      │
│              │
│ ┌──────────┐│
│ │Lightweight││
│ │  Model   ││
│ └──────────┘│
└──────┬───────┘
       │
       │ (if confidence < threshold)
       ▼
┌─────────────┐
│   Cloud     │
│  Server     │
│              │
│ ┌──────────┐│
│ │  Large   ││
│ │  Model   ││
│ └──────────┘│
└─────────────┘
```

### Confidence-Based Routing

**Implementation:**
```python
class HybridInferenceSystem:
    def __init__(self, edge_model, cloud_model, confidence_threshold=0.8):
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.confidence_threshold = confidence_threshold
    
    def infer(self, input_data):
        """Hybrid inference with confidence-based routing"""
        # Edge inference
        edge_output = self.edge_model(input_data)
        edge_probs = F.softmax(edge_output, dim=1)
        edge_confidence = torch.max(edge_probs, dim=1)[0]
        
        # Check confidence
        if edge_confidence.item() >= self.confidence_threshold:
            # Use edge result
            return edge_output, 'edge'
        else:
            # Fallback to cloud
            cloud_output = self.cloud_model(input_data)
            return cloud_output, 'cloud'
    
    def batch_infer(self, input_batch):
        """Batch inference with routing"""
        results = []
        routing_decisions = []
        
        for input_data in input_batch:
            output, routing = self.infer(input_data)
            results.append(output)
            routing_decisions.append(routing)
        
        return results, routing_decisions
```

### Caching and Prefetching

**KV Cache Management:**
```python
class EdgeCloudCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key):
        """Get from cache"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Add to cache with LRU eviction"""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 0
```

### Load Balancing

**Intelligent Load Balancing:**
```python
class HybridLoadBalancer:
    def __init__(self, cloud_servers):
        self.cloud_servers = cloud_servers
        self.server_loads = {server: 0 for server in cloud_servers}
    
    def select_server(self, request):
        """Select least loaded server"""
        # Consider latency, load, and request complexity
        best_server = min(
            self.cloud_servers,
            key=lambda s: self.server_loads[s]
        )
        self.server_loads[best_server] += 1
        return best_server
    
    def update_load(self, server, delta):
        """Update server load"""
        self.server_loads[server] += delta
```

---

## Hands-On Examples

### Example 1: Flower Federated Training

**File:** `examples/ch11_flower_federated.py`

```python
"""
Federated learning using Flower framework.
"""
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=100),
)

# Start Flower client
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=FlowerClient(model, trainloader, valloader)
)
```

### Example 2: FedAvg Implementation

**File:** `examples/ch11_fedavg.py`

```python
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

# Usage
global_model = SimpleModel()
client_models = [client.train_local(global_model.state_dict()) for client in clients]
client_weights = [len(client.local_data) for client in clients]

averaged_state = fedavg_aggregate(client_models, client_weights)
global_model.load_state_dict(averaged_state)
```

### Example 3: Edge Deployment (Jetson)

**File:** `examples/ch11_edge_deployment.py`

```python
"""
Edge deployment pipeline for NVIDIA Jetson.
"""
import torch
import torchvision.transforms as transforms
from PIL import Image

def prepare_for_edge(model, example_input):
    """Prepare model for edge deployment"""
    # 1. Quantize
    quantized_model = dynamic_quantize(model)
    
    # 2. Convert to ONNX
    convert_to_onnx(quantized_model, example_input, "model.onnx")
    
    # 3. Optimize for TensorRT (if available)
    # trt_engine = convert_to_tensorrt("model.onnx")
    
    return quantized_model

def edge_inference(model, input_image):
    """Run inference on edge device"""
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_image).unsqueeze(0)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
    
    return probs

# Deployment script
if __name__ == "__main__":
    model = load_pretrained_model()
    example_input = torch.randn(1, 3, 224, 224)
    
    # Prepare for edge
    edge_model = prepare_for_edge(model, example_input)
    
    # Test inference
    test_image = Image.open("test_image.jpg")
    result = edge_inference(edge_model, test_image)
    print(f"Prediction: {result.argmax().item()}")
```

---

## Best Practices

### 1. Handling Non-IID Data

**Strategies:**
- Use FedProx or similar algorithms with proximal terms
- Implement robust aggregation (median, trimmed mean)
- Consider client-specific learning rates
- Use data augmentation to increase diversity

**Example:**
```python
# Adaptive client selection for non-IID
def select_clients_adaptive(clients, num_selected=10):
    """Select clients with diverse data distributions"""
    # Score clients by data diversity
    diversity_scores = [client.calculate_diversity() for client in clients]
    
    # Select top diverse clients
    selected = np.argsort(diversity_scores)[-num_selected:]
    return selected
```

### 2. Avoiding Privacy Leakage

**Guidelines:**
- Always use differential privacy for sensitive data
- Implement secure aggregation when possible
- Monitor privacy budget carefully
- Use gradient clipping to bound sensitivity
- Consider federated analytics instead of raw data sharing

**Example:**
```python
# Privacy-preserving federated learning
def privacy_preserving_fl(client_models, epsilon=1.0, delta=1e-5):
    """Federated learning with differential privacy"""
    # Clip gradients
    clipped_models = [clip_gradients(model) for model in client_models]
    
    # Add noise
    noise_scale = calculate_noise_scale(epsilon, delta)
    noisy_models = [add_noise(model, noise_scale) for model in clipped_models]
    
    # Aggregate
    aggregated = fedavg_aggregate(noisy_models)
    
    return aggregated
```

### 3. Reducing Communication Cost

**Techniques:**
- Gradient compression (top-k, quantization)
- Federated dropout
- Periodic aggregation (not every round)
- Structured updates (low-rank, sketched)

**Example:**
```python
# Compressed federated learning
def compressed_fedavg(client_models, compression_ratio=0.1):
    """FedAvg with gradient compression"""
    compressed_updates = []
    
    for model in client_models:
        # Compress gradients
        compressed = compress_gradients(model, compression_ratio)
        compressed_updates.append(compressed)
    
    # Aggregate compressed updates
    aggregated = aggregate_compressed(compressed_updates)
    
    # Decompress
    decompressed = decompress_gradients(aggregated)
    
    return decompressed
```

---

## Use Cases

### Use Case 1: Healthcare and Finance Privacy Workloads

**Scenario:** Train medical diagnosis model across hospitals without sharing patient data

**Requirements:**
- Strong privacy guarantees (differential privacy)
- Regulatory compliance (HIPAA, GDPR)
- Robust aggregation (handle different hospital data distributions)
- Secure communication channels

**Implementation:**
- Use DP-SGD with carefully tuned epsilon
- Implement secure aggregation
- Use FedProx for non-IID hospital data
- Regular privacy audits

### Use Case 2: Edge AI for IoT and Robotics

**Scenario:** Deploy lightweight models on edge devices (Jetson, Raspberry Pi)

**Requirements:**
- Low latency inference
- Limited memory and compute
- Offline capability
- Cloud fallback for complex queries

**Implementation:**
- Quantize models to INT8
- Use knowledge distillation (teacher-student)
- Implement hybrid edge-cloud system
- Optimize with TensorRT/ONNX Runtime

---

## Skills Learned

By the end of this chapter, readers will be able to:

1. **Implement federated training loops**
   - Use Flower framework or implement from scratch
   - Implement FedAvg and FedProx algorithms
   - Handle client selection and aggregation

2. **Apply privacy-preserving techniques**
   - Implement DP-SGD with gradient clipping and noise
   - Track privacy budget (epsilon-delta)
   - Use secure aggregation protocols

3. **Handle heterogeneous and non-IID data**
   - Implement robust aggregation methods
   - Use FedProx for non-IID scenarios
   - Adapt to quantity and distribution skew

4. **Deploy models on resource-constrained edge devices**
   - Quantize models (post-training and QAT)
   - Prune models for efficiency
   - Use knowledge distillation
   - Optimize with hardware-specific tools

5. **Build hybrid edge-cloud inference pipelines**
   - Implement confidence-based routing
   - Design caching strategies
   - Balance load between edge and cloud
   - Optimize for latency and accuracy tradeoffs

---

## Summary

This chapter has covered federated learning and edge distributed systems, two critical paradigms for privacy-preserving and resource-efficient AI. Key takeaways:

1. **Federated learning enables privacy-preserving training** without centralizing data
2. **Differential privacy and secure aggregation** protect against privacy leakage
3. **Non-IID data requires robust aggregation** methods like FedProx
4. **Edge deployment needs optimization** (quantization, pruning, distillation)
5. **Hybrid systems combine** edge and cloud benefits

Federated learning and edge AI represent the future of distributed AI systems, enabling AI applications in privacy-sensitive domains and resource-constrained environments. The techniques covered in this chapter provide a solid foundation for building such systems.

---

## Exercises

1. **Implement FedAvg:** Write a complete federated learning system using FedAvg. Test it with non-IID data and measure convergence.

2. **Add Differential Privacy:** Extend your FedAvg implementation with DP-SGD. Measure the privacy-utility tradeoff.

3. **Edge Deployment:** Take a pre-trained model, quantize it, and deploy it on an edge device (or simulate). Measure latency and accuracy.

4. **Hybrid System:** Build a confidence-based routing system that uses a lightweight edge model and falls back to cloud for low-confidence predictions.

---

## Further Reading

- Flower Framework: https://flower.dev/
- FedAvg Paper: https://arxiv.org/abs/1602.05629
- DP-SGD: https://arxiv.org/abs/1607.00133
- FedProx: https://arxiv.org/abs/1812.06127
- Edge AI Survey: https://arxiv.org/abs/1906.05049
