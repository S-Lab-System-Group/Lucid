''' run with bug: adaptdl on multiple GPUs'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import os
import argparse
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append('../')

from models import *

import adaptdl
import adaptdl.torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
# from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup(rank, world_size):
    os.environ['ADAPTDL_MASTER_ADDR'] = 'localhost'
    os.environ['ADAPTDL_MASTER_PORT'] = '12355'
    os.environ['ADAPTDL_NUM_REPLICAS'] = '2'
    os.environ['ADAPTDL_REPLICA_RANK'] = str(rank)
    # initialize the process group
    adaptdl.torch.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_func(rank, args):
    print(f"Running Distributed ResNet on rank {rank}.")
    setup(rank, args.world_size)
    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    # Model
    print('==> Building model..')
    if args.model == 'VGG':
        net = VGG('VGG11')
    else:
        net = eval(args.model)()
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    net = net.to(rank)
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(net.parameters(),
                        lr=args.lr, momentum= 0.67, weight_decay=0.0045200)
    # lr_scheduler = ExponentialLR(optimizer, 0.0133 ** (1.0 / args.epochs))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.2)

    net = adaptdl.torch.AdaptiveDataParallel(net, optimizer, lr_scheduler, device_ids=[rank])

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root="/home/mzhang/data/", train=True, download=True, transform=transform_train)
    trainloader = adaptdl.torch.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=True)
    if args.autoscale_bsz:
        trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024), gradient_accumulation=False)

    validset = torchvision.datasets.CIFAR10(root="/home/mzhang/data/", train=False, download=False, transform=transform_test)
    validloader = adaptdl.torch.AdaptiveDataLoader(validset, batch_size=100, shuffle=False, num_workers=8)

    # Training
    # for epoch in adaptdl.torch.remaining_epochs_until(args.epochs):
    for epoch in range(args.epochs):
        print('\nEpoch: %d' % epoch)
        net.train()
        stats = adaptdl.torch.Accumulator()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            stats["loss_sum"] += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            stats["total"] += targets.size(0)
            stats["correct"] += predicted.eq(targets).sum().item()

        #     trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data")
        #     net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model")
        # print(f'batch size: {len(inputs)}')

        with stats.synchronized():
            stats["loss_avg"] = stats["loss_sum"] / stats["total"]
            stats["accuracy"] = stats["correct"] / stats["total"]
            # writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
            # writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
            # report_train_metrics(epoch, stats["loss_avg"], accuracy=stats["accuracy"])
            if rank == 0:
                print("Train:", stats)
        lr_scheduler.step()

    stats = adaptdl.torch.Accumulator()
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            stats["loss_sum"] += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            stats["total"] += targets.size(0)
            stats["correct"] += predicted.eq(targets).sum().item()

        with stats.synchronized():
            stats["loss_avg"] = stats["loss_sum"] / stats["total"]
            stats["accuracy"] = stats["correct"] / stats["total"]
            # writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
            # writer.add_scalar("Accuracy/Valid", stats["accuracy"], epoch)
            if rank == 0: 
                print("Valid:", stats)
    cleanup()

def run_train_model(train_model, world_size):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0605, type=float, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--model', default='MobileNetV2', type=str, help='model')
    parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False, action='store_true', help='autoscale batchsize')
    parser.add_argument('--gpuid', default=0, type=int, help='run on which gpu')
    parser.add_argument('--world_size', type=int, default=world_size, help='total number of processes')
    args = parser.parse_args()

    mp.spawn(train_model, args=(args,), nprocs=args.world_size, join=True)


if __name__ == '__main__':
   
    # os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpuid}'

    # tensorboard_dir = os.path.join('./result/', args.model, str(args.autoscale_bsz))
    # with SummaryWriter(tensorboard_dir) as writer:
    n_gpus = 2
    run_train_model(train_func, n_gpus)
        