from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import numpy as np
import os
import pandas as pd
import torchvision
import time
import workloads.settings as settings

from torch.nn import DataParallel
from torchvision import transforms
from pointnet.dataset import ShapeNetDataset
from pointnet.pointnet import PointNetCls, feature_transform_regularizer


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch Profile pointnet", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use. Only work when use single gpu.")
parser.add_argument(
    "--num-warmup-batches", type=int, default=1, help='number of warm-up batches that don"t count towards benchmark'
)
parser.add_argument("--num-batches-per-iter", type=int, default=1, help="number of batches per benchmark iteration")
parser.add_argument("--num-iters", type=int, default=1, help="number of benchmark iterations")
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--warmup_epoch', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=20, help='number of training benchmark epochs')
parser.add_argument("--feature_transform", action="store_true", help="use feature transform")
parser.add_argument("--num_points", type=int, default=2500, help="num of points for dataset")
parser.add_argument('--data_dir', type=str, default="/home/mzhang/data/shapenetcore/", help='Data directory')
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')

args = parser.parse_args()

args.data_dir = settings.data_dir + 'shapenetcore/'
args.total_time = settings.total_time

def build_dataset():
    # Dataset: shapenet
    trainset = ShapeNetDataset(root=args.data_dir, classification=True, npoints=args.num_points,)
    return trainset


def build_dataloader(trainset, batch_size):
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=2, drop_last=True,
    )
    return trainloader


def loss_fn(output, label, trans_feat):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)
    if args.feature_transform:
        loss += feature_transform_regularizer(trans_feat) * 0.001
    return loss

def benchmark_pointnet(model_name, batch_size, mixed_precision, gpu_id, bench_list, warm_signal):
    t_start = time.time()
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # specify dataset
    # print('==> Preparing data..')
    trainset = build_dataset()
    trainloader = build_dataloader(trainset, batch_size)
    num_classes = len(trainset.classes)
    # print("classes", num_classes)

    # Model
    # print('==> Building model..')
    model = PointNetCls(k=num_classes, feature_transform=args.feature_transform)
    model = model.to(device)

 
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    if len(gpu_id) > 1:
        model = DataParallel(model)
    
     # Train
    def benchmark_step():
        iter_num = 0
        exit_flag = False
        model.train()
        # Prevent total batch number < warmup+benchmark situation
        while True:
            for inputs, targets in trainloader:
                # Warm-up: previous 10 iters
                if iter_num == args.warmup_epoch-1:
                    warm_signal.value = 1
                    t_warmend = time.time()
                # Reach timeout: exit benchmark
                if time.time() - t_start >= args.total_time:
                    t_end = time.time()
                    t_pass = t_end - t_warmend
                    exit_flag = True
                    break
                optimizer.zero_grad()
                targets = targets[:, 0]
                inputs = inputs.transpose(2, 1)
                if mixed_precision:
                    inputs, targets = inputs.to(device), targets.to(device)
                    with torch.cuda.amp.autocast():
                        pred, trans, trans_feat = model(inputs)
                        loss = loss_fn(pred, targets, trans_feat)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    inputs, targets = inputs.to(device), targets.to(device)
                    pred, trans, trans_feat = model(inputs)
                    loss = loss_fn(pred, targets, trans_feat)
                    loss.backward()
                    optimizer.step()
                iter_num += 1
            if exit_flag:
                break
        return t_pass, iter_num


    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass, iter_num = benchmark_step()
    img_sec = len(gpu_id) * (iter_num - args.warmup_epoch) * batch_size / t_pass
  
    # Results
    bench_list.append(img_sec)
