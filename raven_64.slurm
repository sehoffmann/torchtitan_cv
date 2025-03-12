#!/bin/bash

#SBATCH --job-name=titan-test
#SBATCH --time=00:06:00
#SBATCH --constraint="gpu"

#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000


module purge
module load condainer

ENV="$HOME/condainer-envs/torch"


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO


# Ensure that P2P is available
# export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
#export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1


CONFIG_FILE=${CONFIG_FILE:-"./configs/benchmark_340M.toml"}


# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun cnd -d "$ENV" exec -- torchrun --nnodes 16 --nproc_per_node 4 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" ./torchtitan/train.py --job.config_file ${CONFIG_FILE}

