#!/bin/bash
#SBATCH --job-name=quic_summary
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=00:30:00
#SBATCH --partition=cpu-killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Compare vanilla / TENT / CoTTA inference of the week-44 model across weeks 44-47:
# per-week accuracy + macro-F1, method-vanilla deltas, JSON summary, and a plot.
# Submit with afterok dependency on the vanilla + TTA inference jobs so it runs only
# once all .npz outputs exist. CPU-only, fast.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_quic_summary_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_quic_summary_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Summarizing QUIC week-44 inference (vanilla vs TENT vs CoTTA) ==="
$PYTHON scripts/analysis/quic_tta_summary.py \
    --results_dir results/inference \
    --out_dir results/inference/quic_w44_summary
STATUS=$?
echo "=== summary exit ${STATUS} -> results/inference/quic_w44_summary/ ==="
exit $STATUS
