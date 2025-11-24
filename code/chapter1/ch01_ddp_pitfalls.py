import os

def wrong_master_port(rank):
    # Wrong: Each process uses different port
    os.environ['MASTER_PORT'] = str(12355 + rank)  # ❌

def correct_master_port():
    # Correct: All processes use same port
    os.environ['MASTER_PORT'] = '12355'  # ✅

def wrong_dataloader(dataset):
    # Wrong: Each process sees all data
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32)  # ❌
    return dataloader

def correct_dataloader(dataset, world_size, rank):
    # Correct: Each process sees subset of data
    from torch.utils.data import DataLoader, DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)  # ✅
    return dataloader

def set_epoch_for_shuffling(sampler, epoch):
    # Correct: Shuffle data each epoch
    sampler.set_epoch(epoch)  # ✅
