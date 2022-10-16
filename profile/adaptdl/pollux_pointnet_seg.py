from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import sys
sys.path.append('/home/mzhang/work/ASPLOS23/collect_metric/workloads/pointnet/')
from dataset import ShapeNetDataset
from pointnet import  PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

import adaptdl
import adaptdl.torch

from torch.utils.tensorboard import SummaryWriter
# from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default="/home/mzhang/data/shapenetcore/", help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False, action='store_true', help='autoscale batchsize')
parser.add_argument('--gpuid', default=0, type=int, help='run on which gpu')
opt = parser.parse_args()
# print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = f'{opt.gpuid}'

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])

    test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
else:
    exit('wrong dataset type')

# Data
print('==> Preparing data..')
dataloader = adaptdl.torch.AdaptiveDataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True)
if opt.autoscale_bsz:
    dataloader.autoscale_batch_size(4096, local_bsz_bounds=(8, 100), gradient_accumulation=True)
testdataloader = adaptdl.torch.AdaptiveDataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=int(opt.workers))

# testdataloader = torch.utils.data.DataLoader(
#         test_dataset,
#         batch_size=opt.batchSize,
#         shuffle=True,
#         num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

# if opt.model != '':
#     classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

adaptdl.torch.init_process_group("nccl")
classifier = adaptdl.torch.AdaptiveDataParallel(classifier, optimizer, scheduler)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    classifier.train()
    stats = adaptdl.torch.Accumulator()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        stats["loss_sum"] += loss.item() * points.shape[0]
        stats["total"] += points.shape[0]
        stats["correct"] += correct.item()
    print(f'batch size: {points.shape[0]}')

    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / (stats["total"]*2500)
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        print('Train: ', stats)

def valid(epoch):
    global classifier
    classifier.eval()
    stats = adaptdl.torch.Accumulator()
    with torch.no_grad():
        for i, data in enumerate(testdataloader, 0):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            stats["loss_sum"] += loss.item() * points.shape[0]
            stats["total"] += points.shape[0]
            stats["correct"] += correct.item()
            
    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / (stats["total"]*2500)
        writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Valid", stats["accuracy"], epoch)
        print('Valid: ', stats)

tensorboard_dir = os.path.join('/home/mzhang/work/ASPLOS23/collect_metric/adaptive/result/pointnet_seg', str(opt.autoscale_bsz))
with SummaryWriter(tensorboard_dir) as writer:
    for epoch in adaptdl.torch.remaining_epochs_until(opt.nepoch):
        train(epoch)
        valid(epoch)
        scheduler.step()
