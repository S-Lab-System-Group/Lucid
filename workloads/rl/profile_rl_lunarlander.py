import gym
import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import workloads.settings as settings

from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.env_util import make_vec_env


parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--total_time', type=int, default=30, help='Total time to run the code')

args = parser.parse_args()

args.total_time = settings.total_time

warmup_epoch = 200
benchmark_epoch = 1000


def benchmark_rl(model_name, batch_size, mixed_precision, gpu_id, bench_list, warm_signal):
    t_start = time.time()

    if len(gpu_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id[0]}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_id)

    cudnn.benchmark = True 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Environments & Model
    env = make_vec_env("LunarLander-v2", n_envs=1)
    if model_name == 'PPO':
        model = PPO("MlpPolicy", env, verbose=0, batch_size=batch_size, device=device)
    elif model_name == 'TD3':
        model = TD3("MlpPolicy", env, verbose=0, batch_size=batch_size, device=device)
    
    # Warm-up
    model.learn(total_timesteps=warmup_epoch)
    warm_signal.value = 1
    t_warmend = time.time()

    # Benchmark
    print(f'==> Training {model_name} model with {batch_size} batchsize, {mixed_precision} mp..')
    iter_num = 0
    while True:
        if time.time() - t_start >= args.total_time:
            t_end = time.time()
            t_pass = t_end - t_warmend
            exit_flag = True
            break
        model.learn(total_timesteps=1)
        iter_num += 1

    img_sec = iter_num * batch_size / t_pass
  
    # Results
    bench_list.append(img_sec)
