#!/bin/bash
#SBATCH --job-name=tsne_top10
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=04:00:00
#SBATCH --partition=killable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Shared-fit t-SNE: fit ONCE on background + top-10 apps, then render 10 plots
# (one per app) from the same embedding. NOT a job array — the fit is shared.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Rename logs with a timestamp for chronological sorting
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_tsne_top10_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_tsne_top10_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Inference to visualise (week-16 model, weeks 16->52). Override by editing here.
INFERENCE_DIR=results/inference/week_16_inference
MIN_WEEK=16
TOP=10

$PYTHON scripts/viz/plot_tsne_temporal_multi.py \
    --inference_dir "$INFERENCE_DIR" \
    --min_week "$MIN_WEEK" \
    --top "$TOP" \
    --out_dir figs/drift

echo "Done: top-$TOP shared-fit t-SNE plots from $INFERENCE_DIR (weeks >= $MIN_WEEK)"
