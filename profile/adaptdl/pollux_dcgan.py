"""
DCGAN Tutorial
==============
**Author**: `Nathan Inkawhich <https://github.com/inkawhich>`_

**MODIFIED**: Petuum Inc., 2020 for AdaptDL
"""
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from PIL import Image
# from tfrecord.torch.dataset import TFRecordDataset
import time
import adaptdl
import adaptdl.torch as adl
import adaptdl.torch._metrics as metrics
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch DCGAN Training')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False, action='store_true', help='autoscale batchsize')
args = parser.parse_args()

# Set random seed for reproducibility
#manualSeed = 9999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 10

# Learning rate for optimizers
lr = 0.0005

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Root directory for dataset
dataroot = '/home/mzhang/data/celeba'

# transform=transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# dataset = dset.CelebA(dataroot, split='train', target_type='attr', transform=transform, target_transform=None, download=False)

# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = adl.AdaptiveDataLoader(dataset, batch_size=batch_size,
                                  num_workers=workers, shuffle=False)
if args.autoscale_bsz:
    dataloader.autoscale_batch_size(8*batch_size, local_bsz_bounds=(8, 1024), gradient_accumulation=True)

# Decide which device we want to run on
device = torch.device("cuda:3" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

#########################################################################
# Discriminator Code

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
        return self.main(input)

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=0.9, nesterov=True)
optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=0.9, nesterov=True)
scheduleD = optim.lr_scheduler.StepLR(optimizerD, 15, 1)
scheduleG = optim.lr_scheduler.StepLR(optimizerG, 15, 1)

adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available()
                                     else "gloo")
netD = adl.AdaptiveDataParallel(netD, optimizerD, scheduleD, name="netD")
netG = adl.AdaptiveDataParallel(netG, optimizerG, scheduleG, name="netG")

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

def train(epoch):
    # For each batch in the dataloader
    stats = adl.Accumulator()
    for i, data in enumerate(dataloader, 0):
        data = data[0]
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device, dtype=torch.float)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        stats["g_loss_sum"] += errG.item()
        stats["d_loss_sum"] += errD.item()
    print(f'batch size: {len(data)}')

    stats["norm"] += metrics._metrics_state().grad_params[0]
    stats["var"] += metrics._metrics_state().grad_params[1]
    stats["replicas"] += 1.0
    scheduleD.step()
    scheduleG.step()

    
    with stats.synchronized():
        writer.add_scalar("Loss/G", stats["g_loss_sum"]/stats["replicas"], epoch)
        writer.add_scalar("Loss/D", stats["d_loss_sum"]/stats["replicas"], epoch)
        writer.add_scalar("Performance/GlobalBatchsize", b_size*stats["replicas"], epoch)
        writer.add_scalar("Performance/Replicas", stats["replicas"], epoch)
        writer.add_scalar("Stats/Variance", stats["norm"]/stats["replicas"], epoch)
        writer.add_scalar("Stats/Norm", stats["var"]/stats["replicas"], epoch)
        # print("Train:", stats)


tensorboard_dir = os.path.join('./result/dcgan', str(args.autoscale_bsz))
with SummaryWriter(tensorboard_dir) as writer: 
    for epoch in adl.remaining_epochs_until(num_epochs):
        train(epoch)