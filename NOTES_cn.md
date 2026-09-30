# 《分布式 AI 系统》技术补充说明与代码同步 (Notes & Clarifications)

本文档提供《分布式 AI 系统》（*Distributed AI Systems*）读者在将书中实战示例拓展至多节点环境时的技术说明与代码库同步提示。

---

### 第 3 章：DDP 性能分析实战的多节点兼容性提示

- **使用场景说明**:
  书中的 `profile_ddp.py` 示例针对单机多卡环境设计，启动命令为：
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 code/profile_ddp.py
  ```
  按照书中步骤在单机双卡环境下运行完全正常，能够顺利输出完整的性能分析报表与 Chrome trace。

- **多节点集群扩展**:
  若读者在学完后续章节后，将该示例迁移至跨节点环境（例如两台独立的单卡机器，每台机器只有 `cuda:0`），配套代码库已同步更新为基于 `LOCAL_RANK` 进行设备绑定：
  ```python
  local_rank = int(os.environ.get("LOCAL_RANK", rank))
  data = data.cuda(local_rank, non_blocking=True)
  target = target.cuda(local_rank, non_blocking=True)
  ```
  控制台报表与 trace 导出逻辑（`if rank == 0:`）则继续基于全局 `rank`。配套代码仓库已同步采用此写法，以方便读者在单机与多节点环境下自由切换运行。

