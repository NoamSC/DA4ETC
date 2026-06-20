#!/bin/bash
#SBATCH --job-name=eset_shap
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=01:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

EXPNAME="eset_shap"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Case-study SHAP attribution of eset-edtd (class 57) latent drift around the
# frozen Week-16 recall cliff at ISO week 17. Produces figs/fig_shap_eset_edtd.png.
$PYTHON scripts/drift_validation/validate_eset_shap.py

echo "Completed eset_shap"
