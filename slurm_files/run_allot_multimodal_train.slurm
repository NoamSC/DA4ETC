#!/bin/bash
#SBATCH --job-name=allot_mm_train
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=12:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1

# Train the multimodal model (Multimodal_CESNET) on two Allot time slices:
#   array task 0 -> --slice early    ("week 1" equivalent: start, partial chunk dropped)
#   array task 1 -> --slice quarter  ("week 16" equivalent: ~1/4 into the timeline)
# Each slice is ~2% of the global Allot timeline (see data_utils/allot_timeline.py).
# Normalization stats are auto-computed from each slice and cached.
# Checkpoints -> exps/allot_multimodal/<slice>_eq/weights/best_model.pth

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

SLICES=(early quarter)
SLICE=${SLICES[$SLURM_ARRAY_TASK_ID]}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_allot_mm_train_${SLICE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_allot_mm_train_${SLICE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Training Allot multimodal model, slice=${SLICE} ==="
$PYTHON scripts/train/train_allot_multimodal.py --slice "$SLICE" --override
STATUS=$?
echo "=== train exit ${STATUS}; checkpoint -> exps/allot_multimodal/${SLICE}_eq/weights/best_model.pth ==="
exit $STATUS
