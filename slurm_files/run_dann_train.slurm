#!/bin/bash
#SBATCH --job-name=dann_train
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
#SBATCH --array=0-52

# Train a DANN (domain-adversarial) multimodal model for EACH CESNET-TLS-Year22 week.
# Array task i -> WEEK-2022-i (weeks 0..52). DANN aligns the week's train split (source,
# domain=0) to its test split (target, domain=1) via a gradient-reversal domain head.
#
# Hyperparameters (per request): lambda_dann=1.0, GRL lambda_rgl=0.1, GRL gamma=10
# (gamma>0 is REQUIRED — gamma=0 makes the GRL coefficient identically 0, disabling DANN).
# Set via CLI overrides (NOT by editing config.py defaults, which would silently turn DANN
# on for every other training run). Recipe otherwise mirrors the existing dann_v01 run
# (frac 1.0, 50 epochs, batch 64). Partition: killable (non-H100). Checkpoints per epoch;
# --requeue resumes if preempted or if a week needs more than one 12h slot.
# Output: exps/cesnet_tls_dann_grl01_v01/WEEK-2022-NN/weights/best_model.pth

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
DATASET_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2

WEEK=$SLURM_ARRAY_TASK_ID
WEEK2=$(printf "%02d" "$WEEK")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_dann_train_w${WEEK2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_dann_train_w${WEEK2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== DANN training: WEEK-2022-${WEEK2} (lambda_dann=1.0, grl=0.1, gamma=10) ==="
$PYTHON scripts/train/train_per_week_cesnet.py \
    --multimodal \
    --week "$WEEK" \
    --exp_name "cesnet_tls_dann_grl01_v01/{}" \
    --dataset_root "$DATASET_ROOT" \
    --lambda_dann 1.0 \
    --lambda_rgl 0.1 \
    --lambda_grl_gamma 10 \
    --train_data_frac 1.0 \
    --val_data_frac 0.1 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 50 \
    --batch_size 64 \
    --learning_rate 3e-3
STATUS=$?
echo "=== DANN train exit ${STATUS} for WEEK-2022-${WEEK2} ==="
exit $STATUS
