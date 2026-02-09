#!/bin/bash
#SBATCH --job-name=cesnet_mm_array
#SBATCH --output=logs/cesnet_mm_array/slurm-%A_%a.out.tmp
#SBATCH --error=logs/cesnet_mm_array/slurm-%A_%a.err.tmp
#SBATCH --time=16:30:00
#SBATCH --partition=gpu-h100-killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-52%5

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs subdirectory
mkdir -p logs/cesnet_mm_array

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Week is determined by array task ID
WEEK=${SLURM_ARRAY_TASK_ID}

# Extract experiment name for log files
EXP_NAME_FULL="cesnet_multimodal_each_week_train_v01/week_${WEEK}"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')

# Create logs subdirectory with job ID and experiment name
LOG_SUBDIR="logs/${SLURM_ARRAY_JOB_ID}_cesnet_mm_array"
mkdir -p "$LOG_SUBDIR"

# Rename log files with timestamp for chronological sorting
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTFILE="${LOG_SUBDIR}/week_${WEEK}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="${LOG_SUBDIR}/week_${WEEK}_${SLURM_ARRAY_JOB_ID}.err"

mv "logs/cesnet_mm_array/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/cesnet_mm_array/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

# Redirect output to the properly named files
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Run multimodal training for this week
$PYTHON train_per_week_cesnet.py \
    --week ${WEEK} \
    --val_week 40 \
    --exp_name "$EXP_NAME_FULL" \
    --lambda_rgl 0.0 \
    --lambda_dann 0.0 \
    --train_data_frac 1.0 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 100 \
    --multimodal \
    --dataset_root '/home/anatbr/dataset/CESNET-TLS-Year22_v2'

echo "Completed multimodal training for week ${WEEK}"
