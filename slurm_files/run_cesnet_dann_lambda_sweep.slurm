#!/bin/bash
#SBATCH --job-name=dann_lambda_sweep
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=16:30:00
#SBATCH --partition=gpu-h100-killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Define lambda_rgl values array
LAMBDA_RGL_VALUES=(1e-8 1e-1 1e0 1e1 1e4 1e8)
LAMBDA_RGL=${LAMBDA_RGL_VALUES[$SLURM_ARRAY_TASK_ID]}

# Extract experiment name for log files
EXP_NAME_FULL="cesnet_v4_dann_sanity/lambda_sweep_v1_rgl_${LAMBDA_RGL}_{}"
# Convert slashes to underscores and remove brackets for filename
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')

# Rename log files with timestamp for chronological sorting
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

# Redirect output to the properly named files
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "Running with lambda_rgl=${LAMBDA_RGL} (task ${SLURM_ARRAY_TASK_ID})"

# Run training with varying lambda_rgl
$PYTHON train_per_week_cesnet.py \
    --week 33 \
    --val_week 40 \
    --exp_name "$EXP_NAME_FULL" \
    --lambda_rgl $LAMBDA_RGL \
    --lambda_dann 1.0 \
    --train_data_frac 1.0 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 50

echo "Completed training with lambda_rgl=${LAMBDA_RGL}"
