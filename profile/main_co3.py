#NOTE: CUDA_LAUNCH_BLOCKING=1 python main_co.py will slow down the speed
from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import os
import pandas as pd
import time
import sys
sys.path.append('./workloads/')

from multiprocessing import Process, Manager, Value
from workloads.lstm.profile_lstm import benchmark_lstm
from workloads.imagenet.profile_imagenet import benchmark_imagenet
from workloads.cifar.profile_cifar import benchmark_cifar
from workloads.pointnet.profile_pointnet import benchmark_pointnet
from workloads.dcgan.profile_dcgan import benchmark_dcgan
from workloads.rl.profile_rl_lunarlander import benchmark_rl
from workloads.rl.profile_rl_walker import benchmark_rl2
from workloads.bert.profile_bert import benchmark_bert
from workloads.ncf.profile_ncf import benchmark_ncf
from smi import smi_getter
from co_collect import collect

# model_list_imagenet = ['resnet18', 'resnet50', 'mobilenet_v3_small', 'efficientnet_b0', 'shufflenet_v2_x0_5', 'vgg11', 'alexnet']
# model_list_cifar = ['AlexNet', 'EfficientNetB0', 'MobileNetV2', 'ResNet18', 'ResNet50', 'ShuffleNetV2', 'VGG']
metric_list = []
model_name1 = 'ResNet18'
model_name2 = 'ResNet18'
model_name3 = 'ResNet18'
batch_size1 = 32
batch_size2 = 32
batch_size3 = 32

gpu_id = [0]

start_record = time.time()
with Manager() as manager:
    smi_list = manager.list()
    speed_list1 = manager.list()
    speed_list2 = manager.list()
    speed_list3 = manager.list()
    signal1 = Value('i', 0)
    signal2 = Value('i', 0)
    signal3 = Value('i', 0)

    p1 = Process(target=benchmark_cifar, args=(model_name1, batch_size1, 0, gpu_id, speed_list1, signal1, ))
    p2 = Process(target=benchmark_cifar, args=(model_name2, batch_size2, 0, gpu_id, speed_list2, signal2, ))
    p3 = Process(target=benchmark_cifar, args=(model_name3, batch_size3, 0, gpu_id, speed_list3, signal3, ))
    p4 = Process(target=smi_getter, args=(sys.argv[1:], smi_list, gpu_id, ))

    p1.start()
    p2.start()
    p3.start()

    while True:
        if signal1.value == 1 and signal2.value == 1 and signal3.value == 1:
        # if signal1.value == 1:
            p4.start()
            break

    p1.join()
    p2.join()
    p3.join()
    p4.terminate()

    smi_df = pd.DataFrame(list(smi_list))
    
    print(f'1: {list(speed_list1)}, 2: {list(speed_list2)}, 3:{list(speed_list3)}')
    # print(f'1: {list(speed_list1)}')
    print(smi_df)

# mlist_imagenet = ['mobilenet_v3_small']
# print('imagenet + imagenet')
# metric_list1 = collect(benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, gpu_id)
# df = pd.DataFrame(metric_list1)
# df.to_csv('./1.csv')

# smi_list = []
# smi_getter(sys.argv[1:], smi_list, gpu_id)


end_record = time.time()
print(f'time usage: {end_record - start_record}')