from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
# import sys
# sys.path.append('./translation/')

import numpy as np
import os
import pandas as pd
import time
import workloads.settings as settings
from translation.transformer import Constants
from tqdm import tqdm 

from translation.dataset import TranslationDataset, paired_collate_fn
from translation.transformer.Models import Transformer
from translation.transformer.Optim import ScheduledOptim


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
parser.add_argument("--gpu", default=1, type=int, help="GPU id to use. Only work when use single gpu.")
parser.add_argument(
    "--num-warmup-batches", type=int, default=1, help='number of warm-up batches that don"t count towards benchmark'
)
parser.add_argument("--num-batches-per-iter", type=int, default=1, help="number of batches per benchmark iteration")
parser.add_argument("--num-iters", type=int, default=1, help="number of benchmark iterations")
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--warmup_epoch', type=int, default=60, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=300, help='number of training benchmark epochs')
parser.add_argument('--data_dir', type=str, default="~/data/", help='Data directory')

parser.add_argument('-data', default='/home/mzhang/data/multi30k/multi30k.atok.low.pt')

parser.add_argument('-epoch', type=int, default=None)
parser.add_argument('-step', type=int, default=None)
parser.add_argument('-batch_size', type=int, default=64)

parser.add_argument('-d_word_vec', type=int, default=512)
parser.add_argument('-d_model', type=int, default=512)
parser.add_argument('-d_inner_hid', type=int, default=2048)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-n_layers', type=int, default=6)
# NOTE(keshav2): This just refers to the learning rate schedule,
#                nothing performance related.
parser.add_argument('-n_warmup_steps', type=int, default=4000)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-embs_share_weight', action='store_true')
parser.add_argument('-proj_share_weight', action='store_true')

parser.add_argument('-log', default=None)
parser.add_argument('--checkpoint_dir', type=str,
                    default='/lfs/1/keshav2/checkpoints/transformer')
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')

parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-label_smoothing', action='store_true')

parser.add_argument('--dist-url', default='env://', type=str,
                            help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='Distributed backend')
parser.add_argument('--local_rank', default=0, type=int,
                            help='Local rank')
parser.add_argument('--rank', default=None, type=int,
                            help='Rank')
parser.add_argument('--world_size', default=None, type=int,
                            help='World size')
parser.add_argument('--master_addr', default=None, type=str,
                            help='Master address to use for distributed run')
parser.add_argument('--master_port', default=None, type=int,
                            help='Master port to use for distributed run')

parser.add_argument('--throughput_estimation_interval', type=int,
                    default=None,
                    help='Steps between logging steps completed')
parser.add_argument('--max_duration', type=int, default=None,
                    help='Maximum duration in seconds')
parser.add_argument('--enable_gavel_iterator', action='store_true',
                    default=False, help='If set, use Gavel iterator')
args = parser.parse_args()

args.data = settings.data_dir + 'multi30k/multi30k.atok.low.pt'

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

def prepare_dataloaders(data, opt, distributed, batch_size):
    # ========= Preparing DataLoader =========#
    train_dataset = TranslationDataset(
        src_word2idx=data['dict']['src'],
        tgt_word2idx=data['dict']['tgt'],
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'])
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=paired_collate_fn,
        shuffle=train_sampler is None,
        sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


def benchmark_transformer(model_name, batch_size, mixed_precision, gpu_id, bench_list, warm_signal):
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)


    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #========= Loading Dataset =========#
    data = torch.load(args.data)
    args.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, args, args.master_addr is not None, batch_size)

    args.src_vocab_size = training_data.dataset.src_vocab_size
    args.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if args.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    model = Transformer(
        args.src_vocab_size,
        args.tgt_vocab_size,
        args.max_token_seq_len,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        emb_src_tgt_weight_sharing=args.embs_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        d_inner=args.d_inner_hid,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.d_model, args.n_warmup_steps)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    # Train
    def benchmark_step():
        iter_num = 0
        model.train()
        for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
            # Warm-up: previous 10 iters
            if iter_num == args.warmup_epoch-1:
                warm_signal.value = 1
                t_start = time.time()
            # Benchmark: 50 iters
            if iter_num == args.warmup_epoch+args.benchmark_epoch-1:
                t_end = time.time()
                t_pass = t_end - t_start
                break
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]
            optimizer.zero_grad()
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                    loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
                # backward
                loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
                loss.backward()
                # update parameters
                optimizer.step_and_update_lr()
            iter_num += 1
        return t_pass

    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass = benchmark_step()
    img_sec = args.benchmark_epoch * batch_size / t_pass
  
    # Results
    bench_list.append(img_sec)
