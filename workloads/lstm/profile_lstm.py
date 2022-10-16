from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import sys
import numpy as np
import os
import pandas as pd
import torchvision
import time
import lstm.data as data
import lstm.models as models
import workloads.settings as settings

from torch.nn import DataParallel
from torchvision import transforms


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
parser.add_argument('--warmup_epoch', type=int, default=80, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=170, help='number of training benchmark epochs')
parser.add_argument('--data_dir', type=str, default="/home/mzhang/data/wikitext-2", help='Data directory')
parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--nhead', type=int, default=2, help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', default=False, help='tie the word embedding and softmax weights')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')

args = parser.parse_args()

args.data_dir = settings.data_dir + 'wikitext-2'
args.total_time = settings.total_time

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def benchmark_lstm(model_name, batch_size, mixed_precision, gpu_id, bench_list, warm_signal):
    t_start = time.time()
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset
    # print('==> Preparing data..')
    corpus = data.Corpus(args.data_dir)


    class CorpusDataset(torch.utils.data.Dataset):
        def __init__(self, data, batch_size, bptt):
            self._data = data.narrow(0, 0, (data.size(0) // batch_size) * batch_size)
            # Evenly divide the data across the bsz batches.
            self._data = self._data.view(batch_size, -1).t().contiguous().to(device)
            self._data_length = data.size(0)
            self._batch_size = batch_size
            self._bptt = bptt
        
        def get_input(self, row_idx, col_idx):
            row_idx = row_idx % len(self._data)
            seq_len = min(self._bptt, len(self._data) - 1 - row_idx)
            data = self._data[row_idx: row_idx+seq_len, col_idx]
            target = self._data[row_idx+1: row_idx+1+seq_len, col_idx].view(data.size())
            data = torch.cat([data, data.new_zeros(self._bptt - data.size(0))])
            target = torch.cat([target, target.new_zeros(self._bptt - target.size(0))])
            return data, target

        def __len__(self):
            return self._data_length // self._bptt

        def __getitem__(self, idx):
            return self.get_input((idx // self._batch_size) * self._bptt,
                                idx % self._batch_size)


    trainset = CorpusDataset(corpus.train,
                              batch_size,
                              args.bptt)

    trainloader = torch.utils.data.DataLoader(trainset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           sampler=None,
                                           drop_last=True)

    # Model
    # print('==> Building model..')
    ntokens = len(corpus.dictionary)
    if model_name == 'Transformer':
        model = models.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        model = models.RNNModel(model_name, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)


    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

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
        ntokens = len(corpus.dictionary)
        if model_name != 'Transformer':
            hidden = model.init_hidden(batch_size)
        # Prevent total batch number < warmup+benchmark situation
        while True:
            for data, targets in trainloader:
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
                data = data.t()
                targets = targets.t()
                optimizer.zero_grad()
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        if model_name == 'Transformer':
                            outputs = model(data)
                            outputs = outputs.view(-1, ntokens)
                        else:
                            hidden = repackage_hidden(hidden)
                            outputs, hidden = model(data, hidden)
                        loss = criterion(outputs, targets.flatten())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if model_name == 'Transformer':
                        outputs = model(data)
                        outputs = outputs.view(-1, ntokens)
                    else:
                        hidden = repackage_hidden(hidden)
                        outputs, hidden = model(data, hidden)
                    loss = criterion(outputs, targets.flatten())
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