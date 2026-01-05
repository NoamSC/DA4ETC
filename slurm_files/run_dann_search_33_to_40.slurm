#!/bin/bash
#SBATCH --job-name=dann_search_33_40
#SBATCH --output=logs/dann_search_33_40_%A_%a.out
#SBATCH --error=logs/dann_search_33_40_%A_%a.err
#SBATCH --time=16:30:00
#SBATCH --partition=gpu-h100-killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-49%6

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Run Bayesian hyperparameter search
# First 30 trials will use random sampling, then switch to Bayesian optimization
$PYTHON bayesian_dann_search.py \
    --trial_index ${SLURM_ARRAY_TASK_ID} \
    --train_week 33 \
    --val_week 40 \
    --exp_name "cesnet_v6_dann_33_to_40" \
    --study_name "dann_search_33_to_40" \
    --storage "sqlite:////home/anatbr/students/noamshakedc/da4etc/exps/cesnet_v6_dann_33_to_40/optuna_study.db" \
    --n_random_trials 30

echo "Completed trial ${SLURM_ARRAY_TASK_ID}"
