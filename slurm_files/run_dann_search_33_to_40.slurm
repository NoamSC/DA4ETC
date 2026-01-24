#!/bin/bash
#SBATCH --job-name=dann_search_13_to_33
#SBATCH --output=logs/dann_search_13_to_33/slurm-%A_%a.out.tmp
#SBATCH --error=logs/dann_search_13_to_33/slurm-%A_%a.err.tmp
#SBATCH --time=16:30:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99%10

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Extract experiment name for log files
EXP_NAME_FULL="cesnet_v10_dann_search_13_to_33"
# Convert slashes to underscores and remove brackets for filename
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')

# Create logs subdirectory with job ID and experiment name
LOG_SUBDIR="logs/${SLURM_ARRAY_JOB_ID}_${EXPNAME}"
mkdir -p "$LOG_SUBDIR"

# Rename log files with timestamp for chronological sorting
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTFILE="${LOG_SUBDIR}/trial_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="${LOG_SUBDIR}/trial_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.err"

mv "logs/dann_search_13_to_33/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/dann_search_13_to_33/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

# Redirect output to the properly named files
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Run hyperparameter search
# First array task (0) runs baseline (no DANN), rest run random search
BASELINE_FLAG=""
if [ ${SLURM_ARRAY_TASK_ID} -eq 0 ]; then
    BASELINE_FLAG="--baseline"
fi

$PYTHON bayesian_dann_search.py \
    --trial_index ${SLURM_ARRAY_TASK_ID} \
    --train_week 13 \
    --val_week 33 \
    --exp_name "$EXP_NAME_FULL" \
    --study_name "dann_search_13_to_33" \
    --storage "sqlite:////home/anatbr/students/noamshakedc/da4etc/exps/${EXP_NAME_FULL}/optuna_study.db" \
    --n_random_trials 3000 \
    $BASELINE_FLAG

echo "Completed trial ${SLURM_ARRAY_TASK_ID}"
