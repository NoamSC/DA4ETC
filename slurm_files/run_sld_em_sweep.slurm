#!/bin/bash
#SBATCH --job-name=sld_em_sweep
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=1:00:00
#SBATCH --partition=killable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-15
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

cd /home/anatbr/students/noamshakedc/da4etc

mkdir -p logs

PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

EXPNAME="sld_em_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

$PYTHON test_sld_em_sweep.py --combo-idx $SLURM_ARRAY_TASK_ID

echo "Done"
