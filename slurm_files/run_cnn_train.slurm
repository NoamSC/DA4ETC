#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=12:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1

# Train a flowpic CNN (ConfigurableCNN, NOT multimodal) once on week 1 and once on
# week 16. Array task 0 -> week 1, task 1 -> week 16.
# Checkpoints -> exps/cesnet_cnn/WEEK-2022-{01,16}/weights/best_model.pth

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

WEEKS=(1 16)
WEEK=${WEEKS[$SLURM_ARRAY_TASK_ID]}
WEEK2=$(printf "%02d" "$WEEK")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_cnn_train_w${WEEK2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_cnn_train_w${WEEK2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

DATASET_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2

echo "=== Training flowpic CNN on week ${WEEK} (WEEK-2022-${WEEK2}) ==="
# --skip_sanity_check: the mock-batch sanity check builds a wrongly-shaped flowpic
# (1,256,256) that breaks the conv1d CNN; the real loader yields (256,256) correctly.
$PYTHON scripts/train/train_per_week_cesnet.py \
    --week "$WEEK" \
    --exp_name "cesnet_cnn/{}" \
    --dataset_root "$DATASET_ROOT" \
    --train_data_frac 1.0 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 30 \
    --batch_size 256 \
    --learning_rate 3e-3 \
    --skip_sanity_check
STATUS=$?

echo "=== train exit ${STATUS}; checkpoint -> exps/cesnet_cnn/WEEK-2022-${WEEK2}/weights/best_model.pth ==="
exit $STATUS
