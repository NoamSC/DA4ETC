#!/bin/bash
#SBATCH --job-name=tsne_w16
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=04:00:00
#SBATCH --partition=killable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# Week-16 model t-SNE figures, ALL from ONE shared fit so the point cloud is
# identical and only the colouring changes between figures:
#   - per-app temporal drift for the top-10 apps
#   - a by-class map of the training week (16) over the all-weeks background
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_tsne_w16_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_tsne_w16_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

$PYTHON scripts/viz/plot_tsne_temporal_multi.py \
    --inference_dir results/inference/week_16_inference \
    --min_week 16 \
    --classes 19 49 56 57 97 98 101 102 140 167 \
    --highlight_week 16 \
    --out_dir figs/drift

echo "Done: week-16 per-app drift + by-class map (single shared fit) in figs/drift"
