#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --qos=high
#SBATCH --partition=hopper-prod  # Adjust this for your cluster
#SBATCH --output=/fsx/philipp/logs/%x-%j.out # Adjust this for your cluster
#SBATCH --err=/fsx/philipp/logs/%x-%j.err    # Adjust this for your cluster

set -x -e

source ~/.bashrc
micromamba activate dpo
echo "START TIME: $(date)"

CONFIG_FILE=$1

# Training setup
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export CMD=" \
    scripts/dpo/run_dpo.py --config $CONFIG_FILE
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
   --config_file configs/accelerate_configs/deepspeed_zero3.yaml  \
   --num_machines $NUM_NODES \
   --num_processes $WORLD_SIZE \
   --main_process_ip $MASTER_ADDR \
   --main_process_port $MASTER_PORT \
   --machine_rank \$SLURM_PROCID \
   --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
   --max_restarts 1 \
   --role \$(hostname -s): \
   --tee 3 \
   "

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"