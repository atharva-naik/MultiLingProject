#!/bin/bash

#SBATCH --job-name=mt_mrasp_pretraining
#SBATCH --partition=babel-shared-long
#SBATCH --exclude=babel-0-37
#SBATCH --time=05-01:00:00
#SBATCH --gres gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --output=./logs/mt_mrasp_multi_gpu.log
#SBATCH --error=./errors/mt_mrasp_multi_gpu.err

source activate lora_env

# --model_name_or_path Salesforce/codet5p-770m \

accelerate launch --main_process_port 29502 --num_processes 2 -m src.models.train_mt_mrasp \
--model_name_or_path Salesforce/codet5p-770m \
--learning_rate 5e-5 \
--weight_decay 1e-2 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--max_train_steps 100000 \
--checkpointing_steps 500 \
--lr_scheduler_type linear \
--warmup_ratio 0.1 \
--output_dir ./models/mt_mrasp_multi_gpu/ \