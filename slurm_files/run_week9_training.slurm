#!/bin/bash
#SBATCH --job-name=cesnet_week9_train
#SBATCH --output=logs/week9_train_%j.out
#SBATCH --error=logs/week9_train_%j.err
#SBATCH --time=16:30:00
#SBATCH --partition=gpu-h100-killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

$PYTHON scripts/train/train_per_week_cesnet.py \
    --week 9 \
    --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22 \
    --train_data_frac 1.0 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 50

echo "Completed training for week 9"
