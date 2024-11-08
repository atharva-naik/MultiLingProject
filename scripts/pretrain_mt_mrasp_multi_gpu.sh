#!/bin/bash

#SBATCH --job-name=mt_mrasp_pretraining
#SBATCH --partition=babel-shared-long
#SBATCH --time=05-01:00:00
#SBATCH --gres gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --output=./logs/mt_mrasp_single_gpu_conala_to_10k_per_task_100k_steps_500k.log
#SBATCH --error=./errors/mt_mrasp_single_gpu_conala_to_10k_per_task_100k_steps_500k.err

source activate lora_env

accelerate launch --main_process_port 65235 --num_processes 1 -m src.models.train_mt_mrasp \
--model_name_or_path ./models/mt_mrasp_single_gpu_conala_to_10k_per_task_100k/ \
--learning_rate 5e-5 \
--weight_decay 1e-2 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--max_train_steps 400000 \
--checkpointing_steps 500 \
--lr_scheduler_type linear \
--warmup_ratio 0 \
--output_dir ./models/mt_mrasp_single_gpu_conala_to_10k_per_task_100k_steps_500k/