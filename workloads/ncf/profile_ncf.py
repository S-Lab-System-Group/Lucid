from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import os
import pandas as pd
import time
import workloads.settings as settings

from torchvision import transforms
from torch.nn import DataParallel
import ncf.models as models
import ncf.config as config
import ncf.data_utils as data_utils


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use. Only work when use single gpu.")
parser.add_argument(
    "--num-warmup-batches", type=int, default=1, help='number of warm-up batches that don"t count towards benchmark'
)
parser.add_argument("--num-batches-per-iter", type=int, default=1, help="number of batches per benchmark iteration")
parser.add_argument("--num-iters", type=int, default=1, help="number of benchmark iterations")
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--warmup_epoch', type=int, default=100, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=7000, help='number of training benchmark epochs')
parser.add_argument("--lr", 
  type=float, 
  default=0.001, 
  help="learning rate")
parser.add_argument("--dropout", 
  type=float,
  default=0.0,  
  help="dropout rate")
parser.add_argument("--epochs", 
  type=int,
  default=20,
  help="training epoches")
parser.add_argument("--top_k", 
  type=int, 
  default=10, 
  help="compute metrics@top_k")
parser.add_argument("--factor_num", 
  type=int,
  default=32, 
  help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
  type=int,
  default=3, 
  help="number of layers in MLP model")
parser.add_argument("--num_ng", 
  type=int,
  default=4, 
  help="sample negative items for training")
parser.add_argument("--test_num_ng", 
  type=int,
  default=99, 
  help="sample part of negative items for testing")
parser.add_argument("--out", 
  default=True,
  help="save model or not")
parser.add_argument("--adaptdl",
  action="store_true",
  help="enable adaptdl")
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')
args = parser.parse_args()

args.total_time = settings.total_time

def benchmark_ncf(model_name, batch_size, mixed_precision, gpu_id, bench_list, warm_signal):
    t_start = time.time()
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, args.num_ng, True)
    train_loader = data.DataLoader(train_dataset,
        batch_size, shuffle=True, num_workers=2)

    ########################### CREATE MODEL #################################
    if model_name == 'NeuMF-end':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    model = models.NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model, GMF_model, MLP_model)
    model.cuda()

    loss_function = nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    if len(gpu_id) > 1:
        model = DataParallel(model)
    ########################### TRAINING #####################################
    def benchmark_step():
        iter_num = 0
        exit_flag = False
        model.train()
        train_loader.dataset.ng_sample()
        print(len(train_loader))
        while True:
            for idx, (user, item, label) in enumerate(train_loader):
                # Warm-up: previous 10 iters
                if iter_num == args.warmup_epoch-1:
                    warm_signal.value = 1
                    t_warmed = time.time()
                # Reach timeout: exit profiling
                if time.time() - t_start >= args.total_time:
                    t_end = time.time()
                    t_pass = t_end - t_warmed
                    exit_flag = True
                    break
                user = user.cuda()
                item = item.cuda()
                label = label.float().cuda()
                optimizer.zero_grad()
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        prediction = model(user, item)
                        loss = loss_function(prediction, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    prediction = model(user, item)
                    loss = loss_function(prediction, label)
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
