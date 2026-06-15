#!/bin/bash
#SBATCH --job-name=ref1_cache
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=03:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="ref1_cache_build"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- Build the reference-week-1 metrics cache WITH per-week estimator vectors
#     (vec_*), required for the TV%-units estimation-accuracy figure. Recompute
#     is mandatory: the existing ref1 cache predates vec_*. We skip ALL figure
#     plotting here (a..i) — this job only regenerates
#     metrics_cache_ref1_grid9.npz so the per-figure jobs can plot cheaply from
#     it (with --exclude_weeks 0 to hide the untrustable week 0). ---
$PYTHON archive/plot_paper_figures.py \
    --inference_dir results/inference/week_1_inference \
    --reference_week 1 --em --recompute \
    --skip a b c c2 d e f g h i

echo "Completed ref1_cache_build"
