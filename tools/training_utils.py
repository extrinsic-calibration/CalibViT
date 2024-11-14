import os
import torch
import torch.distributed as dist
import builtins
import datetime
from typing import Any

def setup_for_distributed(is_master: bool) -> None:
    """
    Disables printing when not in master process.

    Args:
        is_master (bool): Flag indicating if the current process is the master process.
    """
    builtin_print = builtins.print

    def print(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print(f'[{now}] ', end='')  # print with timestamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def get_world_size() -> int:
    """
    Get the number of processes in the current distributed group.

    Returns:
        int: The number of processes in the current distributed group.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized() -> bool:
    """
    Check if the distributed package is available and initialized.

    Returns:
        bool: True if distributed package is available and initialized, False otherwise.
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """
    Get the rank of the current process in the distributed group.

    Returns:
        int: The rank of the current process.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """
    Check if the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return get_rank() == 0


def dist_barrier() -> None:
    """
    Synchronize all processes in the distributed group.
    """
    if is_dist_avail_and_initialized():
        dist.barrier()


def init_distributed_mode(args: Any) -> int:
    """
    Initialize the distributed mode.

    Args:
        args (Any): Argument object that contains necessary distributed settings.

    Returns:
        int: The rank of the current process.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, 'rank'):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return 0

    args.distributed = True
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    return args.rank

def setup_logger_for_distributed(is_master: bool, logger: Any) -> None:
    """
    Disables logging when not in master process.

    Args:
        is_master (bool): Flag indicating if the current process is the master process.
        logger (Any): Logger object to control logging output.
    """
    logger_info = logger.info

    def info(*args: Any, **kwargs: Any) -> None:
        force = kwargs.pop('force', False)
        if is_master or force:
            logger_info(*args, **kwargs)

    logger.info = info
