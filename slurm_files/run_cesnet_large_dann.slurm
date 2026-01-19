#!/bin/bash
#SBATCH --job-name=cesnet_large_dann
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=16:30:00
#SBATCH --partition=gpu-h100-killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Extract experiment name for log files
EXP_NAME_FULL="cesnet_v4_dann_sanity/normal_v13_grl_effect_test_crazy_rgl_{}"
# Convert slashes to underscores and remove brackets for filename
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')

# Rename log files with timestamp for chronological sorting
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"

mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

# Redirect output to the properly named files
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Run training for specific week based on SLURM_ARRAY_TASK_ID
# Configuration 1: Large DANN
$PYTHON train_per_week_cesnet.py \
    --week 33 \
    --exp_name "$EXP_NAME_FULL" \
    --lambda_rgl 1e4 \
    --lambda_dann 1e0 \
    --train_data_frac 1.0 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 50

echo "Completed training for week ${SLURM_ARRAY_TASK_ID} (large DANN)"
