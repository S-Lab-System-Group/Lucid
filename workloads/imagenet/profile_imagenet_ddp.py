from __future__ import print_function
import argparse
import timeit
from cvxpy import mixed_norm
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
import sys
import numpy as np
import os
import pandas as pd
import torchvision
import time
import torch.multiprocessing as mp
sys.path.append('/home/mzhang/work/ASPLOS23/collect_metric/')

from torch.nn import DataParallel
from multiprocessing import Process, Manager, Value
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from models import *

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--warmup_iter', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=50, help='number of training benchmark epochs')
parser.add_argument('--data_dir', type=str, default="~/data/", help='Data directory')
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')
parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Total time to run the code')
parser.add_argument('--master_port', type=str, default='47020', help='Total time to run the code')


args = parser.parse_args()


# ------ Setting up the distributed environment -------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successfully communicate across multiple process
    # involving multiple GPUs.


def cleanup():
    dist.destroy_process_group()


def benchmark_imagenet_ddp(rank, model_name, batch_size, mixed_precision, gpu_id, t_start):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)
    print(f"Running Distributed ResNet on rank {rank}.")
    setup(rank, len(gpu_id))
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    # Model
    # print('==> Building model..')
    model = getattr(torchvision.models, model_name)()
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # Dataset
    data = torch.randn(batch_size, 3, 224, 224)
    target = torch.LongTensor(batch_size).random_() % 1000
    data, target = data.to(rank), target.to(rank)

    # data, target = next(iter(trainloader))
    # data, target = data.cuda(), target.cuda()

    # Train
    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    iter_num = 0
    model.train()
    # Prevent total batch number < warmup+benchmark situation
    while True:
        # Warm-up: previous 10 iters
        if iter_num == args.warmup_iter-1:
            t_warmend = time.time()
        # Reach timeout: exit benchmark
        if time.time() - t_start >= args.total_time:
            t_end = time.time()
            t_pass = t_end - t_warmend
            break
        optimizer.zero_grad()
        if mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        iter_num += 1

    img_sec = len(gpu_id) * (iter_num - args.warmup_iter) * batch_size / t_pass
    if rank == 0:
        print(f'master port: {args.master_port}, speed: {img_sec}')

    cleanup()

if __name__ == '__main__':
    model_name = 'resnet50'
    batch_size = 64
    mixed_precision = 0
    gpu_id = [0,1,2,3]
    # world_size = 4
    t_start = time.time()
    mp.spawn(benchmark_imagenet_ddp, args=(model_name, batch_size, mixed_precision, gpu_id, t_start, ), nprocs=len(gpu_id), join=True)
