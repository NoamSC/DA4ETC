#!/bin/bash
#SBATCH --job-name=labelshift_sweep_est_w1
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=08:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="labelshift_sweep_estimators_w1"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- job command ---
# Week-1 half of the estimator-variant severity sweep, run in parallel with
# job 540003 to remove its timeout risk (whichever finishes first writes the
# cache; the other loads it or recomputes harmlessly).
$PYTHON scripts/analysis/labelshift_severity_sweep.py \
    --inference_dir results/inference/week_1_inference \
    --reference_week 1 \
    --output_dir figs/isolation \
    --no_mfwdd \
    --variant_estimators \
    --suffix _estimators

echo "Completed labelshift_sweep_estimators_w1"
