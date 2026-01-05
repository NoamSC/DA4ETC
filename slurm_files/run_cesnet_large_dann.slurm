#!/bin/bash
#SBATCH --job-name=cesnet_large_dann
#SBATCH --output=logs/cesnet_large_dann_%A_%a.out
#SBATCH --error=logs/cesnet_large_dann_%A_%a.err
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

# Run training for specific week based on SLURM_ARRAY_TASK_ID
# Configuration 1: Large DANN
$PYTHON train_per_week_cesnet.py \
    --week 33 \
    --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22 \
    --exp_name "cesnet_v4_dann_sanity/normal_{}" \
    --lambda_rgl 1e0 \
    --lambda_dann 1e0 \
    --train_data_frac 1.0 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 50

echo "Completed training for week ${SLURM_ARRAY_TASK_ID} (large DANN)"
