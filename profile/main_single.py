#NOTE: CUDA_LAUNCH_BLOCKING=1 python main.py
from __future__ import print_function
import os
import pandas as pd
import time

import sys
sys.path.append('./workloads/')

from workloads.lstm.profile_lstm import benchmark_lstm
from workloads.imagenet.profile_imagenet import benchmark_imagenet
from workloads.cifar.profile_cifar import benchmark_cifar
from workloads.pointnet.profile_pointnet import benchmark_pointnet
from workloads.dcgan.profile_dcgan import benchmark_dcgan
from workloads.rl.profile_rl_lunarlander import benchmark_rl
from workloads.rl.profile_rl_walker import benchmark_rl2
from workloads.bert.profile_bert import benchmark_bert
from workloads.ncf.profile_ncf import benchmark_ncf
from workloads.translation.profile_transformer import benchmark_transformer

from single_collect import s_collect

model_list_imagenet = ['resnet50', 'mobilenet_v3_small']
# model_list_cifar = ['ResNet18', 'MobileNetV2', 'EfficientNetB0', 'VGG']
model_list_cifar = ['ResNet18']
bs_list = [64]
gpu_id = [0]
metric_list = []
mp_list = [0, 1]

os.makedirs('result/', exist_ok=True)

# # Single: imagenet metric
# print('Classification: imagenet')
# dataset = 'imagenet'
# for model_name in model_list_imagenet:
#     for batch_size in bs_list:
#         for mp in mp_list:
#             # collect single job gpu info
#             metric_dict = s_collect(benchmark_imagenet, dataset, model_name, batch_size, mp, gpu_id)
#             metric_list.append(metric_dict)
#             time.sleep(2)

# Single: cifar10 metric
print('Classification: cifar')
dataset = 'cifar10'
for model_name in model_list_cifar:
    for batch_size in bs_list:
        for mp in mp_list:
            # collect single job gpu info
            metric_dict = s_collect(benchmark_cifar, dataset, model_name, batch_size, mp, gpu_id)
            metric_list.append(metric_dict)
            time.sleep(2)

# # Single: pointnet
# print('3D: pointnet')
# for batch_size in bs_list:
#     for mp in mp_list:
#         metric_dict = s_collect(benchmark_pointnet, dataset='shapenet', model_name='pointnet', batch_size=batch_size, mp=mp, gpu_id=gpu_id)
#         metric_list.append(metric_dict)
#         time.sleep(2)

# # Single: dcgan
# print('CV: dcgan')
# for batch_size in bs_list:
#     for mp in mp_list:
#         metric_dict = s_collect(benchmark_dcgan, dataset='LSUN', model_name='dcgan', batch_size=batch_size, mp=mp, gpu_id=gpu_id)
#         metric_list.append(metric_dict)
#         time.sleep(2)

# # Single: rl-lunalander
# print('RL: LunarLander-v2')
# for batch_size in bs_list:
#     metric_dict = s_collect(benchmark_rl, dataset='LunarLander-v2', model_name='PPO', batch_size=batch_size, mp=0, gpu_id=gpu_id)
#     metric_list.append(metric_dict)
#     time.sleep(2)

# # Single: rl-Bipedal Walker
# print('RL: Bipedal Walker')
# for batch_size in bs_list:
#     metric_dict = s_collect(benchmark_rl2, dataset='BipedalWalker-v3', model_name='TD3', batch_size=batch_size, mp=0, gpu_id=gpu_id)
#     metric_list.append(metric_dict)
#     time.sleep(2)

# # Single: ncf
# print('Recommendation: ncf')
# for batch_size in [64, 128]:
#     for mp in mp_list:
#         metric_dict = s_collect(benchmark_ncf, dataset='MovieLens', model_name='NeuMF-pre', batch_size=batch_size, mp=mp, gpu_id=gpu_id)
#         metric_list.append(metric_dict)
#         time.sleep(2)

# # Single: lstm
# print('Language Modeling: lstm')
# for batch_size in [64, 128]:
#     for mp in mp_list:
#         metric_dict = s_collect(benchmark_lstm, dataset='Wikitext2', model_name='LSTM', batch_size=batch_size, mp=mp, gpu_id=gpu_id)
#         metric_list.append(metric_dict)
#         time.sleep(2)

# # Single: bert
# print('Question Answering: bert')
# for batch_size in [32]:
#     for mp in mp_list:
#         metric_dict = s_collect(benchmark_bert, dataset='SQUAD', model_name='bert', batch_size=batch_size, mp=mp, gpu_id=gpu_id)
#         metric_list.append(metric_dict)
#         time.sleep(2)

# # Single: transformer
# print('Translation: tranformer')
# for batch_size in [32, 64]:
#         metric_dict = s_collect(benchmark_transformer, dataset='multi30k', model_name='transformer', batch_size=batch_size, mp=0, gpu_id=gpu_id)
#         metric_list.append(metric_dict)
#         time.sleep(2)

# print(pd.DataFrame(metric_list))
df = pd.DataFrame(metric_list)
# df.replace(
# ['imagenet', 'cifar10', 'shapenet', 'LSUN', 'LunarLander-v2', 'BipedalWalker-v3', 'MovieLens', 'Wikitext2', 'SQUAD', 'multi30k'],
# ['ImageNet', 'CIFAR-10', 'ShapeNet', 'LSUN', 'LunarLander', 'BipedalWalker', 'MovieLens', 'Wikitext2', 'SQuAD', 'Multi30k'], inplace=True)

# df.replace(
# ['resnet50', 'mobilenet_v3_small', 'EfficientNetB0', 'pointnet', 'dcgan', 'NeuMF-pre', 'bert', 'transformer'],
# ['ResNet50', 'MobileNetV3', 'EfficientNet', 'PointNet', 'DCGAN', 'NeuMF', 'BERT', 'Transformer'], inplace=True)

df.to_csv('./result/single_cifar.csv')



