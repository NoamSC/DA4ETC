#!/bin/bash
#SBATCH --job-name=labelshift_sweep_fwd
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=03:00:00
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
EXP_NAME_FULL="labelshift_sweep_fwd"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- job command ---
# Forward-only clean weeks for the week-16 reference (17-20): negatives drawn
# only from weeks after deployment; week 16 itself excluded because its npz is
# the model-selection validation set AND the reference-statistics data.
# n_trials=14 keeps the negative cell count (~56) comparable to the 13-20 run.
$PYTHON scripts/analysis/labelshift_severity_sweep.py \
    --inference_dir results/inference/week_16_inference \
    --reference_week 16 \
    --output_dir figs/isolation_w16 \
    --clean_weeks 17 20 \
    --n_trials 14 \
    --suffix _fwd

echo "Completed labelshift_sweep_fwd"
