#!/bin/bash
#SBATCH --job-name=plot_paper_figs
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=5:00:00
#SBATCH --partition=gpu-h100-killable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

cd /home/anatbr/students/noamshakedc/da4etc

mkdir -p logs

PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

EXPNAME="plot_paper_figs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"

mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

$PYTHON plot_paper_figures.py \
    --inference_dir 'figs/week_1_inference' \
    --dataset_root '/home/anatbr/dataset/CESNET-TLS-Year22_v2' \
    --reference_week 1 \
    --grid_classes 9 \
    --output_dir 'figs' \
    --recompute

echo "Done"
