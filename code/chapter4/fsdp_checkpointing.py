from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.checkpoint import checkpoint

def fsdp_wrap_and_checkpoint(model, inp):
    wrapped = FSDP(model)

    def forward_step(x):
        return wrapped(x)

    # Use checkpointing on large blocks to save activation memory
    out = checkpoint(forward_step, inp)
    return out
