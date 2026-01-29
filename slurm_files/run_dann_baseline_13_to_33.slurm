#!/bin/bash
#SBATCH --job-name=dann_baseline_13_to_33
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
#SBATCH --mem-per-cpu=4G

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Extract experiment name for log files
EXP_NAME_FULL="cesnet_v10_dann_search_13_to_33"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')

# Rename log files with timestamp for chronological sorting
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_baseline_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_baseline_${SLURM_JOB_ID}.err"

mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

# Redirect output to the properly named files
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Run baseline (no DANN)
$PYTHON bayesian_dann_search.py \
    --trial_index 0 \
    --train_week 13 \
    --val_week 33 \
    --exp_name "$EXP_NAME_FULL" \
    --study_name "dann_search_13_to_33" \
    --storage "sqlite:////home/anatbr/students/noamshakedc/da4etc/exps/${EXP_NAME_FULL}/optuna_study.db" \
    --n_random_trials 3000 \
    --baseline

echo "Completed baseline trial"
