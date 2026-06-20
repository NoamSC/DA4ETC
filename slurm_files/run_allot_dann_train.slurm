#!/bin/bash
#SBATCH --job-name=allot_dann_train
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

# DANN (domain-adversarial) training of the multimodal model on TWO closed-world
# (proprietary, anonymized) time slices:
#   array task 0 -> --slice early    ("week 1" equivalent slice; validation only)
#   array task 1 -> --slice quarter  (clean-regime "week 16" equivalent; source window 13)
# Transductive UDA: labeled SOURCE slice aligned to UNLABELED future windows.
# Checkpoints -> exps/allot_multimodal/<slice>_eq_dann/weights/best_model.pth

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

SLICES=(early quarter)
SLICE=${SLICES[$SLURM_ARRAY_TASK_ID]}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_allot_dann_train_${SLICE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_allot_dann_train_${SLICE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== DANN training Allot multimodal model, slice=${SLICE} ==="
$PYTHON scripts/train/train_allot_multimodal_dann.py --slice "$SLICE" \
    --lambda_dann 0.1 --lambda_grl_gamma 10.0 \
    --target-windows 20 27 34 41 --num-epochs 30 --override
STATUS=$?
echo "=== dann train exit ${STATUS}; checkpoint -> exps/allot_multimodal/${SLICE}_eq_dann/weights/best_model.pth ==="
exit $STATUS
