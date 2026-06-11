#!/bin/bash
#SBATCH --job-name=mirror_w16
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=01:00:00
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
EXP_NAME_FULL="mirror_effect_w16"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- Fig B (mirror effect) with the Week-16 source/reference.
#     Reuses the ref16 metrics cache just rebuilt by the Fig-7 job (no
#     --recompute), so this is load + plot only. Separate output dir so the
#     Week-1 fig_mirror_effect.png is not overwritten. ---
mkdir -p figs/ref16
$PYTHON archive/plot_paper_figures.py \
    --inference_dir results/inference/week_16_inference \
    --reference_week 16 --em \
    --output_dir figs/ref16 \
    --skip a c c2 d e f g h i

echo "Completed mirror_w16"
