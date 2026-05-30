#!/bin/bash
#SBATCH --job-name=cesnet_weekly_train_coral
#SBATCH --output=logs/weekly_train_coral_%A_%a.out
#SBATCH --error=logs/weekly_train_coral_%A_%a.err
#SBATCH --time=16:30:00
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-52%9

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Run CORAL training for specific week based on SLURM_ARRAY_TASK_ID
$PYTHON scripts/train/train_per_week_cesnet.py \
    --week ${SLURM_ARRAY_TASK_ID} \
    --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22_v2 \
    --exp_name "cesnet_multimodal_each_week_train_coral_v01/{}" \
    --multimodal \
    --lambda_coral 1.0 \
    --lambda_dann 0.0 \
    --train_data_frac 1.0 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 50

echo "Completed CORAL training for week ${SLURM_ARRAY_TASK_ID}"
