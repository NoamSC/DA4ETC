#!/bin/bash
#SBATCH --job-name=stats_rigor
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=02:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

EXPNAME="stats_rigor"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Bootstrap CIs + F1-threshold sensitivity for the anomaly-monitor headline.
# Primary: Week-16 forward-only (clean / matched-protocol).  Variants: Week-16
# full-stream, and Week-1 (sensor-event validation, flagged).  Seeds 1,2,3,4,42.

echo "=== Week-16 FORWARD-ONLY (primary / matched protocol) ==="
$PYTHON scripts/analysis/stats_rigor.py \
    --inference_dir results/inference_auditfix/week_16_vanilla_bs64 \
    --reference_week 16 --forward \
    --out_fig figs/fig_threshold_sensitivity.png \
    --tag ref16_fwd

echo "=== Week-16 FULL-STREAM (deployed-setting variant) ==="
$PYTHON scripts/analysis/stats_rigor.py \
    --inference_dir results/inference_auditfix/week_16_vanilla_bs64 \
    --reference_week 16 \
    --out_fig figs/fig_threshold_sensitivity_ref16_full.png \
    --tag ref16_full

echo "=== Week-1 FORWARD-ONLY (sensor-event validation, NOT the clean headline) ==="
$PYTHON scripts/analysis/stats_rigor.py \
    --inference_dir results/inference_auditfix/week_1_vanilla_bs64 \
    --reference_week 1 --forward --exclude_weeks 0 \
    --out_fig figs/fig_threshold_sensitivity_ref1_fwd.png \
    --tag ref1_fwd

echo "Completed stats_rigor"
