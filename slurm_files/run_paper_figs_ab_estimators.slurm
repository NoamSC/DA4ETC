#!/bin/bash
#SBATCH --job-name=figs_ab_est
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
EXP_NAME_FULL="paper_figs_ab_estimators"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- Fig A (paper Fig 8, estimation accuracy): keeps its original ref week 0 ---
$PYTHON archive/plot_paper_figures.py \
    --inference_dir results/inference/week_1_inference \
    --reference_week 0 --em --recompute \
    --skip b c c2 d e f g h i

# --- Fig B (paper Fig 10, mirror effect): keeps its original ref week 1 ---
$PYTHON archive/plot_paper_figures.py \
    --inference_dir results/inference/week_1_inference \
    --reference_week 1 --em --recompute \
    --skip a c c2 d e f g h i

echo "Completed figs_ab_est"
