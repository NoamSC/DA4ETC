#!/bin/bash
#SBATCH --job-name=figs_w1
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
mkdir -p logs figs/week1_start

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="paper_figs_w1"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- Regenerate the four week-1-reference paper figures from the freshly-built
#     ref1 metrics cache (NO --recompute), with week 0 hidden everywhere via
#     --exclude_weeks 0. Non-destructive: writes into figs/week1_start/.
#       a -> Fig 8  estimation accuracy  (fig_estimation_accuracy{,_debiased}.png)
#       c -> Fig 9  app degradation grid (fig_app_degradation_grid.png)
#       b -> Fig 10 mirror effect        (fig_mirror_effect.png)
#       g -> Fig 4  decomposition        (fig_combined_penalties.png)
#     Skip c2 d e f h i. --debias emits the debiased Fig-8 variant alongside.
#     The cache already holds the SLD-EM vectors, so --em is unnecessary here. ---
$PYTHON archive/plot_paper_figures.py \
    --inference_dir results/inference/week_1_inference \
    --reference_week 1 --exclude_weeks 0 --debias \
    --output_dir figs/week1_start \
    --skip c2 d e f h i

echo "Completed paper_figs_w1"
