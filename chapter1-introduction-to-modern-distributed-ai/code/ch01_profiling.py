"""
Memory and latency profiling for model training.
"""
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model():
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda()
    
    inputs = torch.randn(32, 1000).cuda()
    targets = torch.randint(0, 10, (32,)).cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Memory profiling
    torch.cuda.reset_peak_memory_stats()
    
    # Time profiling with PyTorch profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("forward"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        with record_function("backward"):
            loss.backward()
        
        with record_function("optimizer_step"):
            optimizer.step()
    
    # Print results
    print("=" * 80)
    print("CUDA Time Summary:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))
    
    print("\n" + "=" * 80)
    print("Memory Summary:")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=20
    ))
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak GPU Memory: {peak_memory:.2f} GB")

if __name__ == "__main__":
    profile_model()
