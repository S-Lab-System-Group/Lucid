
from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torch.utils.data.distributed
import sys
import numpy as np
import os
import pandas as pd
import torchvision
from torch.nn import DataParallel
from torchvision import transforms
import workloads.settings as settings


# Benchmark settings
parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--num-gpus", type=int, default=10, help="number of gpus")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use. Only work when use single gpu.")
parser.add_argument('--warmup_epoch', type=int, default=20, help='number of warmup epochs')
parser.add_argument('--benchmark_epoch', type=int, default=180, help='number of training benchmark epochs')
parser.add_argument("--amp-fp16", action="store_true", default=False, help="Enables FP16 training with Apex.")
parser.add_argument('--dataroot', type=str, default="~/data/", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')

args = parser.parse_args()
args.total_time = settings.total_time

# Training
def benchmark_dcgan(model_name, batch_size, mixed_precision, gpu_id, bench_list, warm_signal):
    t_start = time.time()
    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True

    dataset = dset.FakeData(size=30000, image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())
    nc = 3
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # print(len(dataloader))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ngpu = int(args.ngpu)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output


    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))

    if mixed_precision:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCELoss()
    fixed_noise = torch.randn(args.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    if args.dry_run:
     args.niter = 1

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
    
    if len(gpu_id) > 1:
        netD = DataParallel(netD)

    def benchmark_step():
        iter_num = 0
        exit_flag = False
        # Prevent total batch number < warmup+benchmark situation
        while True:
            for i, data in enumerate(dataloader, 0):
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
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        # train with real
                        netD.zero_grad()
                        real_cpu = data[0].to(device)
                        batch_size = real_cpu.size(0)
                        label = torch.full((batch_size,), real_label,
                                        dtype=real_cpu.dtype, device=device)
                        output = netD(real_cpu)
                        errD_real = criterion(output, label)
                    scaler.scale(errD_real).backward()

                    with torch.cuda.amp.autocast():
                        # train with fake
                        noise = torch.randn(batch_size, nz, 1, 1, device=device)
                        fake = netG(noise)
                        label.fill_(fake_label)
                        output = netD(fake.detach())
                        errD_fake = criterion(output, label)
                        errD = errD_real + errD_fake
                    scaler.scale(errD_fake).backward()
                    scaler.step(optimizerD)

                    with torch.cuda.amp.autocast():
                        ############################
                        # (2) Update G network: maximize log(D(G(z)))
                        ###########################
                        netG.zero_grad()
                        label.fill_(real_label)  # fake labels are real for generator cost
                        output = netD(fake)
                        errG = criterion(output, label)   
                    scaler.scale(errG).backward()    
                    scaler.step(optimizerG)
                    scaler.update()
                else:
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    # train with real
                    netD.zero_grad()
                    real_cpu = data[0].to(device)
                    batch_size = real_cpu.size(0)
                    label = torch.full((batch_size,), real_label,
                                    dtype=real_cpu.dtype, device=device)

                    output = netD(real_cpu)
                    errD_real = criterion(output, label)
                    errD_real.backward()

                    # train with fake
                    noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake = netG(noise)
                    label.fill_(fake_label)
                    output = netD(fake.detach())
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    output = netD(fake)
                    errG = criterion(output, label)
                    errG.backward()
                    optimizerG.step()
                iter_num += 1
            if exit_flag:
                break
        return t_pass, iter_num

    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    t_pass, iter_num = benchmark_step()
    img_sec = len(gpu_id) * (iter_num - args.warmup_epoch) * batch_size / t_pass
  
    # Results
    bench_list.append(img_sec)
    
        

