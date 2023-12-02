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
--model_name_or_path Salesforce/codet5p-770m \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--checkpointing_steps 5000 \