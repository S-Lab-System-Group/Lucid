from __future__ import print_function
import argparse
import timeit
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import numpy as np
import time
import os
import torchvision
import workloads.settings as settings

from torch.nn import DataParallel
from models import *


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
parser.add_argument("--gpu", default=1, type=int, help="GPU id to use. Only work when use single gpu.")


parser.add_argument('--warmup_epoch', type=int, default=10, help='number of warmup epochs')
parser.add_argument("--num-batches-per-iter", type=int, default=30, help="number of batches per benchmark iteration")
parser.add_argument("--num-iters", type=int, default=30, help="number of benchmark iterations")
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')


args = parser.parse_args()
# args.total_time = settings.total_time

# Training
def benchmark_imagenet(model_name, batch_size, mixed_precision, gpu_id, bench_list, warm_signal):
    t_start = time.time()
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True

    model = getattr(torchvision.models, model_name)()
    model = model.cuda()

    data = torch.randn(batch_size, 3, 224, 224)
    target = torch.LongTensor(batch_size).random_() % 1000
    data, target = data.cuda(), target.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    
    if len(gpu_id) > 1:
        model = DataParallel(model)
    
    def benchmark_step():
        iter_num = 0
        while True:
            optimizer.zero_grad()
            if iter_num == args.warmup_epoch-1:
                warm_signal.value = 1
                t_warmend = time.time()
            # Reach timeout: exit profiling
            if time.time() - t_start >= args.total_time:
                t_end = time.time()
                t_pass = t_end - t_warmend
                break
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            iter_num += 1
        return t_pass, iter_num

    # Benchmark
    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass, iter_num = benchmark_step()
    img_sec = len(gpu_id) * (iter_num - args.warmup_epoch) * batch_size / t_pass

    bench_list.append(img_sec)

    
        

