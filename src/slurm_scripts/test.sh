#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=llama_test
#SBATCH --mem=60GB
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1-00:00:00
#SBATCH -o ../out/llama_test.out
python3 llama_transcoder.py