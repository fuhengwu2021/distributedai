import os
import torch
import torch.distributed as dist

def init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank

def wrap_model_for_ddp(model, local_rank):
    model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model

if __name__ == '__main__':
    # This file shows the minimal setup; integrate into train.py
    lr = init_distributed()
    print(f"Initialized distributed process on local_rank={lr}")
