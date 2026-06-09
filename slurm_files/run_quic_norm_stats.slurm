#!/bin/bash
#SBATCH --job-name=quic_norm
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# Compute (3,30) PPI + (44,) flowstats normalization statistics over the prepared
# CESNET-QUIC22 weeks. Writes <ROOT>/normalization_stats.npz, which BOTH training and
# inference load (via week_dir.parent/normalization_stats.npz). CPU-only.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_quic_norm_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_quic_norm_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Computing normalization stats over ${ROOT} ==="
$PYTHON scripts/data_prep/compute_normalization_stats.py \
    --dataset-root "$ROOT" \
    --sample-frac 0.05 \
    --num-workers 4 \
    --output "$ROOT/normalization_stats.npz"
STATUS=$?
echo "=== norm-stats exit ${STATUS} -> ${ROOT}/normalization_stats.npz ==="
exit $STATUS
