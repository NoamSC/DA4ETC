#!/bin/bash
#SBATCH --job-name=auroc_w16_mm
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=06:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="auroc_anomaly_week16_mm"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- Week-16 viral-spike AUROC / FAR@95 table on the MULTIMODAL backbone.
#     Identical protocol to run_auroc_w16.slurm but pointed at the multimodal
#     mm-CESNET frozen-source Week-16 vanilla inference (results/inference_auditfix/
#     week_16_vanilla_bs64) instead of the FlowPic CNN. This is the backbone the
#     paper actually claims (Multimodal_CESNET) and the one used by the negative-
#     transfer tables + stats-rigor (Table III), so it makes the Week-16 viral
#     anomaly figure/table consistent with the rest of the paper.
#     12 GB of per-week npz must all be held in RAM -> 96G request (48G OOM'd).
#     forward-only eval (weeks 17-52); seeds {1,2,3,4,42}; mean +/- std.
#     Versioned outputs under figs/ with the ref16_mm tag (does NOT overwrite the
#     CNN-backbone ref16 artifacts). ---
$PYTHON -u scripts/analysis/auroc_anomaly_week16.py \
    --inference_dir results/inference_auditfix/week_16_vanilla_bs64 \
    --reference_week 16 \
    --seeds 1 2 3 4 42 \
    --output_dir figs \
    --tag ref16_mm

echo "Completed auroc_w16_mm"
