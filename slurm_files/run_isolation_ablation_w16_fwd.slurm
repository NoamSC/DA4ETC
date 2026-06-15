#!/bin/bash
#SBATCH --job-name=iso_ablation_w16_fwd
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=02:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="isolation_ablation_w16_fwd"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- job command ---
# Week-16-source precision-diagnostics isolation ablation, FORWARD-ONLY:
# evaluate only weeks >16 (clean-regime story from the healthy source week),
# label-shift robustness calibrated on forward clean weeks 17-20.
# Produces figs/isolation_w16/panels/fig_isolation_A_roc_detection_fwd.png (+ B-E,
# combined fig, isolation_metrics_fwd.json, *_fwd caches).
$PYTHON scripts/analysis/precision_isolation_ablation.py \
    --inference_dir results/inference/week_16_inference \
    --reference_week 16 \
    --checkpoint exps/cesnet_multimodal_each_week_train_v01/week_16/weights/best_model.pth \
    --output_dir figs/isolation_w16 \
    --forward \
    --clean_weeks 17 20 \
    --suffix _fwd

echo "Completed iso_ablation_w16_fwd"
