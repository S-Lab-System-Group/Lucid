#!/bin/bash
worker_num=2
model=$2
batch_size=$3

last_rank=`expr $worker_num - 1`

# nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes='127.0.0.1'
nodes_array=( $nodes )
node1=${nodes_array[0]}

#export ADAPTDL_CHECKPOINT_PATH=cifar-checkpoint
# export ADAPTDL_SHARE_PATH=data
# export ADAPTDL_JOB_ID=$SLURM_JOB_ID
export ADAPTDL_MASTER_ADDR=$node1
export ADAPTDL_MASTER_PORT=47020
export ADAPTDL_NUM_REPLICAS=$worker_num


ADAPTDL_REPLICA_RANK=0 python3 -u pollux_mnist.py &
ADAPTDL_REPLICA_RANK=1 python3 -u pollux_mnist.py

# # batch_size=128
# for ((  i=0; i < $worker_num; i++ ))
# do
#     # node=${nodes_array[$i]}
#     node=${nodes_array[0]}
#     if [[ $i -lt `expr $worker_num-1` ]]
#     then
#         ADAPTDL_REPLICA_RANK=$i python3 -u pollux_cifar.py &
#     else
#         ADAPTDL_REPLICA_RANK=$i python3 -u pollux_cifar.py 
#     fi
# done