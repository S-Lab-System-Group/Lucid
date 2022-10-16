from __future__ import print_function
import argparse
import json
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import torchvision
import time

from torch.nn import DataParallel
from models import DeepSpeech, supported_rnns
from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler

# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use. Only work when use single gpu.")
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--warmup_epoch', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=30, help='number of training benchmark epochs')
parser.add_argument('--data_dir', type=str, default="~/data/", help='Data directory')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--num-workers', default=2, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=80, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true', help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--seed', default=None, type=int, help='Seed to generators')

args = parser.parse_args()


def benchmark_deepspeech(model_name, batch_size, mixed_precision, bench_list, warm_signal):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.num_gpus))

    cudnn.benchmark = True 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model
    # print('==> Building model..')
    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))
    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=args.bidirectional)
    model = model.to(device)

    # Dataset
    # print('==> Preparing data..')
    trainset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.speed_volume_perturb, spec_augment=args.spec_augment)
    trainloader = AudioDataLoader(trainset, batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    criterion = torch.nn.CTCLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=False)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    model = DataParallel(model)

    # Train
    def benchmark_step():
        iter_num = 0
        model.train()
        for i, data in enumerate(trainloader):
            # Warm-up: previous 10 iters
            if iter_num == args.warmup_epoch-1:
                warm_signal.value = 1
                t_start = time.time()
            # Benchmark: 50 iters
            if iter_num == args.warmup_epoch+args.benchmark_epoch-1:
                t_end = time.time()
                t_pass = t_end - t_start
                break
            inputs, targets, input_sizes, target_sizes = data
            inputs = inputs.to(device)
            input_sizes = input_sizes.to(device)
            optimizer.zero_grad()
            if mixed_precision:
                with torch.cuda.amp.autocast():
                    out, output_sizes = model(inputs, input_sizes)
                    out = out.transpose(0, 1) 
                    float_out = out.float()  # ensure float32 for loss
                    loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
                    loss_value = loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out, output_sizes = model(inputs, input_sizes)
                out = out.transpose(0, 1) 
                float_out = out.float()  # ensure float32 for loss
                loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
                loss.backward()
                optimizer.step()
            iter_num += 1
        return t_pass

    print(f'==> Training {model_name} model with {batch_size} batchsize..')
    t_pass = benchmark_step()
    img_sec = args.benchmark_epoch * batch_size / t_pass
  
    # Results
    bench_list.append(img_sec)
