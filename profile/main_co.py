#NOTE: CUDA_LAUNCH_BLOCKING=1 python main_co.py will slow down the speed
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

from co_collect import collect


mlist_imagenet = ['resnet50', 'mobilenet_v3_small']
# mlist_cifar = ['ResNet18', 'MobileNetV2', 'EfficientNetB0', 'VGG']
mlist_cifar = ['VGG']
bs_list = [32, 64, 128]
bs_list1 = [32, 64]
gpu_id = [0]

os.makedirs('result/colocate/', exist_ok=True)
os.makedirs('result/vgg/', exist_ok=True)
t_begin = time.time()

# # colocate: imagenet + imagenet
# print('imagenet + imagenet')
# metric_list1 = collect(benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, gpu_id)
# df = pd.DataFrame(metric_list1)
# df.to_csv('./result/colocate/1.csv')

# # colocate: imagenet + cifar10
# print('imagenet + cifar10')
# metric_list2 = collect(benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, benchmark_cifar, mlist_cifar, 'CIFAR-10', bs_list, gpu_id)
# df = pd.DataFrame(metric_list2)
# df.to_csv('./result/colocate/2.csv')

# # colocate: imagenet + pointnet
# print('imagenet + pointnet')
# metric_list3 = collect(benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, benchmark_pointnet, ['PointNet'], 'ShapeNet', bs_list, gpu_id)
# df = pd.DataFrame(metric_list3)
# df.to_csv('./result/colocate/3.csv')

# # colocate: imagenet + dcgan
# print('imagenet + dcgan')
# metric_list4 = collect(benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, benchmark_dcgan, ['DCGAN'], 'LSUN', bs_list, gpu_id)
# df = pd.DataFrame(metric_list4)
# df.to_csv('./result/colocate/4.csv')

# # colocate: imagenet + rl
# print('imagenet + rl')
# metric_list5 = collect(benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, benchmark_rl, ['PPO'], 'LunarLander', bs_list, gpu_id)
# df = pd.DataFrame(metric_list5)
# df.to_csv('./result/colocate/5.csv')

# # colocate: imagenet + rl2
# print('imagenet + rl2')
# metric_list6 = collect(benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, benchmark_rl2, ['TD3'], 'BipedalWalker', bs_list, gpu_id)
# df = pd.DataFrame(metric_list6)
# df.to_csv('./result/colocate/6.csv')

# # colocate: imagenet + lstm
# print('imagenet + lstm')
# metric_list7 = collect(benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, benchmark_lstm, ['LSTM'], 'Wikitext2', [64, 128], gpu_id)
# df = pd.DataFrame(metric_list7)
# df.to_csv('./result/colocate/7.csv')

# colocate: cifar10 + cifar10
print('cifar10 + cifar10')
metric_list8 = collect(benchmark_cifar, mlist_cifar, 'CIFAR-10', bs_list, benchmark_cifar, mlist_cifar, 'CIFAR-10', bs_list, gpu_id)
df = pd.DataFrame(metric_list8)
df.to_csv('./result/vgg/8.csv')

# colocate: cifar10 + pointnet
print('cifar10 + pointnet')
metric_list9 = collect(benchmark_cifar, mlist_cifar, 'CIFAR-10', bs_list, benchmark_pointnet, ['PointNet'], 'ShapeNet', bs_list, gpu_id)
df = pd.DataFrame(metric_list9)
df.to_csv('./result/vgg/9.csv')

# colocate: cifar10 + dcgan
print('cifar10 + dcgan')
metric_list10 = collect(benchmark_cifar, mlist_cifar, 'CIFAR-10', bs_list, benchmark_dcgan, ['DCGAN'], 'LSUN', bs_list, gpu_id)
df = pd.DataFrame(metric_list10)
df.to_csv('./result/vgg/10.csv')

# colocate: cifar10 + rl
print('cifar10 + rl')
metric_list11 = collect(benchmark_cifar, mlist_cifar, 'CIFAR-10', bs_list, benchmark_rl, ['PPO'], 'LunarLander', bs_list, gpu_id)
df = pd.DataFrame(metric_list11)
df.to_csv('./result/vgg/11.csv')

# colocate: cifar10 + rl2
print('cifar10 + rl2')
metric_list12 = collect(benchmark_cifar, mlist_cifar, 'CIFAR-10', bs_list, benchmark_rl2, ['TD3'], 'BipedalWalker', bs_list, gpu_id)
df = pd.DataFrame(metric_list12)
df.to_csv('./result/vgg/12.csv')

# colocate: cifar10 + lstm
print('cifar10 + lstm')
metric_list13 = collect(benchmark_cifar, mlist_cifar, 'CIFAR-10', bs_list, benchmark_lstm, ['LSTM'], 'Wikitext2', [64, 128], gpu_id)
df = pd.DataFrame(metric_list13)
df.to_csv('./result/vgg/13.csv')

# # colocate: pointnet + pointnet
# print('pointnet + pointnet')
# metric_list14 = collect(benchmark_pointnet, ['PointNet'], 'ShapeNet', bs_list, benchmark_pointnet, ['PointNet'], 'ShapeNet', bs_list, gpu_id)
# df = pd.DataFrame(metric_list14)
# df.to_csv('./result/colocate/14.csv')

# # colocate: pointnet + dcgan
# print('pointnet + dcgan')
# metric_list15 = collect(benchmark_pointnet, ['PointNet'], 'ShapeNet', bs_list, benchmark_dcgan, ['DCGAN'], 'LSUN', bs_list, gpu_id)
# df = pd.DataFrame(metric_list15)
# df.to_csv('./result/colocate/15.csv')

# # colocate: pointnet + rl
# print('pointnet + rl')
# metric_list16 = collect(benchmark_pointnet, ['PointNet'], 'ShapeNet', bs_list, benchmark_rl, ['PPO'], 'LunarLander', bs_list, gpu_id)
# df = pd.DataFrame(metric_list16)
# df.to_csv('./result/colocate/16.csv')

# # colocate: pointnet + rl2
# print('pointnet + rl2')
# metric_list17 = collect(benchmark_pointnet, ['PointNet'], 'ShapeNet', bs_list, benchmark_rl2, ['TD3'], 'BipedalWalker', bs_list, gpu_id)
# df = pd.DataFrame(metric_list17)
# df.to_csv('./result/colocate/17.csv')

# # colocate: pointnet + lstm
# print('pointnet + lstm')
# metric_list18 = collect(benchmark_pointnet, ['PointNet'], 'ShapeNet', bs_list, benchmark_lstm, ['LSTM'], 'Wikitext2', [64, 128], gpu_id)
# df = pd.DataFrame(metric_list18)
# df.to_csv('./result/colocate/18.csv')

# # colocate: dcgan + dcgan
# print('dcgan + dcgan')
# metric_list19 = collect(benchmark_dcgan, ['DCGAN'], 'LSUN', bs_list, benchmark_dcgan, ['DCGAN'], 'LSUN', bs_list, gpu_id)
# df = pd.DataFrame(metric_list19)
# df.to_csv('./result/colocate/19.csv')

# # colocate: dcgan + rl
# print('dcgan + rl')
# metric_list20 = collect(benchmark_dcgan, ['DCGAN'], 'LSUN', bs_list, benchmark_rl, ['PPO'], 'LunarLander', bs_list, gpu_id)
# df = pd.DataFrame(metric_list20)
# df.to_csv('./result/colocate/20.csv')

# # colocate: dcgan + rl2
# print('dcgan + rl2')
# metric_list21 = collect(benchmark_dcgan, ['DCGAN'], 'LSUN', bs_list, benchmark_rl2, ['TD3'], 'BipedalWalker', bs_list, gpu_id)
# df = pd.DataFrame(metric_list21)
# df.to_csv('./result/colocate/21.csv')

# # colocate: dcgan + lstm
# print('dcgan + rl2')
# metric_list22 = collect(benchmark_dcgan, ['DCGAN'], 'LSUN', bs_list, benchmark_lstm, ['LSTM'], 'Wikitext2', [64, 128], gpu_id)
# df = pd.DataFrame(metric_list22)
# df.to_csv('./result/colocate/22.csv')

# # colocate: ncf + ncf
# print('ncf + ncf')
# metric_list23 = collect(benchmark_ncf, ['NeuMF-pre'], 'MovieLens', [64, 128], benchmark_ncf, ['NeuMF-pre'], 'MovieLens', [64, 128], gpu_id)
# df = pd.DataFrame(metric_list23)
# df.to_csv('./result/colocate/23.csv')

# # colocate: rl + rl
# print('rl + rl')
# metric_list24 = collect(benchmark_rl, ['PPO'], 'LunarLander', bs_list, benchmark_rl, ['PPO'], 'LunarLander', bs_list, gpu_id)
# df = pd.DataFrame(metric_list24)
# df.to_csv('./result/colocate/24.csv')

# # colocate: rl + rl2
# print('rl + rl2')
# metric_list25 = collect(benchmark_rl, ['PPO'], 'LunarLander', bs_list, benchmark_rl2, ['TD3'], 'BipedalWalker', bs_list, gpu_id)
# df = pd.DataFrame(metric_list25)
# df.to_csv('./result/colocate/25.csv')

# # colocate: rl + lstm
# print('rl + lstm')
# metric_list26 = collect(benchmark_rl, ['PPO'], 'LunarLander', bs_list, benchmark_lstm, ['LSTM'], 'Wikitext2', [64, 128], gpu_id)
# df = pd.DataFrame(metric_list26)
# df.to_csv('./result/colocate/26.csv')

# # colocate: rl2 + rl2
# print('rl2 + rl2')
# metric_list27 = collect(benchmark_rl2, ['TD3'], 'BipedalWalker', bs_list, benchmark_rl2, ['TD3'], 'BipedalWalker', bs_list, gpu_id)
# df = pd.DataFrame(metric_list27)
# df.to_csv('./result/colocate/27.csv')

# # colocate: rl2 + lstm
# print('rl2 + lstm')
# metric_list28 = collect(benchmark_rl2, ['TD3'], 'BipedalWalker', bs_list, benchmark_lstm, ['LSTM'], 'Wikitext2', [64, 128], gpu_id)
# df = pd.DataFrame(metric_list28)
# df.to_csv('./result/colocate/28.csv')

# # colocate: lstm + lstm
# print('lstm + lstm')
# metric_list29 = collect(benchmark_lstm, ['LSTM'], 'Wikitext2', [64, 128], benchmark_lstm, ['LSTM'], 'Wikitext2', [64, 128], gpu_id)
# df = pd.DataFrame(metric_list29)
# df.to_csv('./result/colocate/29.csv')

# # colocate: ncf + ncf
# print('ncf + ncf')
# metric_list30 = collect(benchmark_ncf, ['NeuMF-pre'], 'MovieLens', [64, 128], benchmark_ncf, ['NeuMF-pre'], 'MovieLens', [64, 128], gpu_id)
# df = pd.DataFrame(metric_list30)
# df.to_csv('./result/colocate/30.csv')

# # colocate: ncf + mobilenetv3
# print('ncf + mobilenetv3')
# metric_list31 = collect(benchmark_ncf, ['NeuMF-pre'], 'MovieLens', [64, 128], benchmark_imagenet, mlist_imagenet, 'ImageNet', bs_list, gpu_id)
# df = pd.DataFrame(metric_list31)
# df.to_csv('./result/colocate/31.csv')

# # colocate: transformer + transformer
# print('transformer + transformer')
# metric_list32 = collect(benchmark_transformer, ['Transformer'], 'Multi30k', [32, 64], benchmark_transformer, ['Transformer'], 'Multi30k', [32, 64], gpu_id)
# df = pd.DataFrame(metric_list32)
# df.to_csv('./result/colocate/32.csv')



t_end = time.time()
print(f'Time usage: {t_end - t_begin}s')