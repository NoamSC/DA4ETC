#!/bin/bash
#SBATCH --job-name=quic_prep
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=04:00:00
#SBATCH --partition=cpu-killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3

# Prepare CESNET-QUIC22 weeks into TLS-Year22-compatible layout (CPU-only).
# Array tasks 0-3 -> weeks 44, 45, 46, 47 (all weeks; the prep script is idempotent).
# See scripts/data_prep/prepare_quic_week.py / QUIC_W44_MULTIMODAL_HANDOFF.md.
# Edit the WEEKS array + --array range to (re)prepare a subset.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared

WEEKS=(44 45 46 47)
WEEK=${WEEKS[$SLURM_ARRAY_TASK_ID]}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_quic_prep_w${WEEK}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_quic_prep_w${WEEK}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Preparing CESNET-QUIC22 week ${WEEK} ==="
$PYTHON scripts/data_prep/prepare_quic_week.py --week "$WEEK" --out_root "$ROOT"
STATUS=$?
echo "=== prep exit ${STATUS} for week ${WEEK} ==="
exit $STATUS
