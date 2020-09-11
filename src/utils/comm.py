import torch.distributed as dist


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()