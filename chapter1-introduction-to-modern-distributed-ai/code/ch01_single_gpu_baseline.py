"""
Single-GPU training baseline for comparison with distributed training.
"""
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self, input_size=1000, hidden_size=512, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

def train_single_gpu():
    device = torch.device("cuda:0")
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy dataset
    dataset = TensorDataset(
        torch.randn(1000, 1000),
        torch.randint(0, 10, (1000,))
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(10):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/10, Loss: {epoch_loss/len(dataloader):.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Peak memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    train_single_gpu()
