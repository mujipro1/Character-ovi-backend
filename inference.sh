#!/bin/bash

#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --partition=high
#SBATCH --job-name=inference
#SBATCH --open-mode=append 

## move to repo directory
# cd /home/${USER}/video_audio_training/multimodal-generation
export NCCL_DEBUG=INFO

cd /home/weiminwang/open_source/MovieMachines

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

scontrol show hostnames $SLURM_JOB_NODELIST | awk '{print $1 " slots=8"}' > hostfile

## activate your venv
source /home/chetwinlow/video_audio_training/multimodal-generation/mmgen-env-26/bin/activate

NCCL_TREE_THRESHOLD=0 \
    deepspeed \
    -H hostfile \
    inference.py \
    --config ovi/configs/inference/inference_fusion.yaml \
    --deepspeed