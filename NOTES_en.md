# Technical Notes & Code Synchronization for *Distributed AI Systems*

This document provides technical notes and companion repository updates for readers extending the hands-on examples from *Distributed AI Systems* to broader multi-node environments.

---

### Chapter 3: Multi-Node Compatibility Note for DDP Profiling

- **Target Context**:
  The `profile_ddp.py` example in Chapter 3 is designed for a single-node multi-GPU environment, launched via:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 code/profile_ddp.py
  ```
  Following the steps in the book on a single-node dual-GPU workstation runs cleanly as printed, producing complete profiling summaries and Chrome trace files.

- **Multi-Node Cluster Extension**:
  For readers who wish to experiment with this profiling script across a multi-node cluster (such as two distinct single-GPU nodes where each machine only possesses local device `cuda:0`), the companion code repository has been updated to bind tensors using `LOCAL_RANK`:
  ```python
  local_rank = int(os.environ.get("LOCAL_RANK", rank))
  data = data.cuda(local_rank, non_blocking=True)
  target = target.cuda(local_rank, non_blocking=True)
  ```
  Output reporting and trace exports (`if rank == 0:`) remain tied to global `rank`. The companion repository has synchronized this implementation so readers can run the code seamlessly across both single-node and multi-node environments.

