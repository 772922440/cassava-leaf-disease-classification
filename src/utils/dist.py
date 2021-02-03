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


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

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


def all_reduce_scalar(value, device, world_size=1, mean=False):
    value = torch.tensor([value]).to(device=device)
    dist.all_reduce(value, op=ReduceOp.SUM)
    if mean:
        value /= world_size
    return value.item()


def all_reduce_array(array, device, world_size=1, mean=False):
    array = torch.from_numpy(array).to(device=device)
    dist.all_reduce(array, op=ReduceOp.SUM)
    if mean:
        array /= world_size
    return array.to('cpu').numpy()