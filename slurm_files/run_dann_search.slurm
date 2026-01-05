#!/bin/bash
#SBATCH --job-name=dann_search
#SBATCH --output=exps/dann_search/logs/trial_%A_%a.out
#SBATCH --error=exps/dann_search/logs/trial_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-h100-killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-79%4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p exps/dann_search/logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Run the Bayesian search trial
# Each array task runs the same script, which uses Optuna to pick the next hyperparameters
$PYTHON bayesian_dann_search.py --trial_index ${SLURM_ARRAY_TASK_ID}

echo "Completed trial ${SLURM_ARRAY_TASK_ID}"
