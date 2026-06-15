#!/bin/bash
#SBATCH --job-name=mirror_ext_w16
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=02:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="mirror_extreme_w16"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- Mirror effect under extreme injected label shift, Week-16 source fwd ---
$PYTHON scripts/analysis/mirror_extreme_labelshift.py \
    --severities 10 100

echo "Completed mirror_ext_w16"
