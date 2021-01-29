import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import ReduceOp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def spawn(demo_fn):
    world_size = torch.cuda.device_count() 
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def all_reduce_mean(value, world_size, device):
    value = torch.tensor([value]).to(device=device)
    dist.all_reduce(value, op=ReduceOp.SUM)
    value = value.item() / world_size
    return value