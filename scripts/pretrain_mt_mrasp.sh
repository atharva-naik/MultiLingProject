#!/bin/bash

#SBATCH --job-name=mt_mrasp_pretraining
#SBATCH --partition=babel-shared-long
#SBATCH --time=05-01:00:00
#SBATCH --gres gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=./logs/mt_mrasp_pretraining_test.log
#SBATCH --error=./errors/mt_mrasp_pretraining_test.err

source activate lora_env

python3 -m src.models.train_mt_mrasp \
--model_name_or_path t5-small \
--learning_rate 1e-5 \
--weight_decay 1e-2 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--max_train_steps 50000 \
--checkpointing_steps 100 \
--lr_scheduler_type linear \
--warmup_ratio 0.1 \
--output_dir ./models/mt_mrasp/ \