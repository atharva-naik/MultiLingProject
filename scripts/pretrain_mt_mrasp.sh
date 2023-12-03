#!/bin/bash

#SBATCH --job-name=mt_mrasp_pretraining
#SBATCH --partition=babel-shared-long
#SBATCH --time=05-01:00:00
#SBATCH --gres gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=./logs/mt_mrasp_single_gpu_10k.log
#SBATCH --error=./errors/mt_mrasp_single_gpu_10k.err

source activate lora_env

python3 -m src.models.train_mt_mrasp \
--model_name_or_path t5-small \
--learning_rate 1e-5 \
--weight_decay 1e-2 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--max_train_steps 100000 \
--checkpointing_steps 2500 \
--lr_scheduler_type linear \
--warmup_ratio 0.06 \
--output_dir ./models/mt_mrasp_single_gpu_10k/ \