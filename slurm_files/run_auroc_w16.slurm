#!/bin/bash
#SBATCH --job-name=auroc_w16
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=06:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="auroc_anomaly_week16"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- Week-16 viral-spike AUROC / FAR@95 table (Sec VI, clean-regime reference).
#     Reference week = WEEK-2022-16 (vanilla frozen source, train_week_16 CNN
#     inference); forward-only eval (weeks 17-52); >=5 seeds {1,2,3,4,42},
#     mean +/- std. Moves the viral false-alarm headline onto the SAME reference
#     week as the tracking-rho / 52-week monitoring analyses (removes the silent
#     Week-1/Week-16 mix). Versioned outputs under figs/ with the ref16 tag. ---
$PYTHON -u scripts/analysis/auroc_anomaly_week16.py \
    --inference_dir results/inference/cesnet-tls-year22/cnn/train_week_16 \
    --reference_week 16 \
    --seeds 1 2 3 4 42 \
    --output_dir figs \
    --tag ref16

echo "Completed auroc_w16"
