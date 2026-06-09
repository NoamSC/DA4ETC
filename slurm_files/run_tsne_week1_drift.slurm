#!/bin/bash
#SBATCH --job-name=tsne_w1
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=04:00:00
#SBATCH --partition=killable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Week-1 model t-SNE drift figures:
#  (1) per-app shared-fit temporal drift for the SAME 10 apps as the week-16 run
#  (2) a "training-week background" map: week-1 samples coloured per class on top
#      of the grey all-weeks cloud (no focal class).

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Rename logs with a timestamp for chronological sorting
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_tsne_w1_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_tsne_w1_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

INFERENCE_DIR=results/inference/week_1_inference
# Same top-10 apps that the week-16 run produced, pinned explicitly so the
# week-1 figures cover the identical class set (no per-week re-ranking).
CLASSES="19 49 56 57 97 98 101 102 140 167"

# Single shared fit -> per-app drift plots AND the week-1 by-class map, so the
# point cloud is identical across figures and only the colouring changes.
$PYTHON scripts/viz/plot_tsne_temporal_multi.py \
    --inference_dir "$INFERENCE_DIR" \
    --classes $CLASSES \
    --highlight_week 1 \
    --out_dir figs/drift

echo "Done: week-1 per-app drift + by-class map (single shared fit) in figs/drift"
