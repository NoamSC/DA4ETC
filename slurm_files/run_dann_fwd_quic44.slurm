#!/bin/bash
#SBATCH --job-name=dann_fwd_quic44
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=12:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=45-47
#
# FORWARD-TRANSFER DANN for the UDA benchmark, CESNET-QUIC22 WEEK-44 SOURCE variant.
# Replica of run_dann_fwd_w16.slurm on the prepared QUIC dataset (weeks 44-47, 102
# classes). Array task i is the TARGET week (--val_week i in {45,46,47}); week 44 is
# the in-distribution source and is the degenerate no-gap case (skipped). The week-44
# labeled source is DANN-aligned to WEEK-2022-i's test split (transductive UDA), then
# frozen-evaluated on WEEK-2022-i (the DANN-row diagonal for the QUIC block).
#
# Hyperparameters mirror the TLS week-16 diagonal: lambda_dann=1.0, GRL lambda_rgl=0.1,
# gamma=10, 50 epochs, lr 3e-3. train_per_epoch_data_frac matches the QUIC source-model
# training data budget (the source model used train_data_frac 0.2); here we use the
# full QUIC week per epoch since each QUIC week is far smaller than a TLS week.
# Output: exps/cesnet_quic22_dann_fwd_w44_v01/WEEK-2022-44_val_WEEK-2022-NN/weights/best_model.pth

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
DATASET_ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared

SRC_WEEK=44
TGT_WEEK=$SLURM_ARRAY_TASK_ID
TGT2=$(printf "%02d" "$TGT_WEEK")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_dann_fwd_quic44_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_dann_fwd_quic44_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

if [ "$TGT_WEEK" -eq "$SRC_WEEK" ]; then
    echo "=== TGT week ${TGT2} == source ${SRC_WEEK}; degenerate (no domain gap), skipping ==="
    exit 0
fi

echo "=== DANN forward-transfer: SOURCE WEEK-2022-44 (QUIC) -> TARGET (--val_week) WEEK-2022-${TGT2} ==="
$PYTHON scripts/train/train_per_week_cesnet.py \
    --multimodal \
    --week "$SRC_WEEK" \
    --val_week "$TGT_WEEK" \
    --exp_name "cesnet_quic22_dann_fwd_w44_v01/{}" \
    --dataset_root "$DATASET_ROOT" \
    --lambda_dann 1.0 \
    --lambda_rgl 0.1 \
    --lambda_grl_gamma 10 \
    --train_data_frac 1.0 \
    --val_data_frac 0.1 \
    --train_per_epoch_data_frac 1.0 \
    --num_epochs 50 \
    --batch_size 64 \
    --num_workers 0 \
    --learning_rate 3e-3
STATUS=$?
echo "=== DANN forward-transfer exit ${STATUS} for SOURCE 44 -> TARGET ${TGT2} ==="
exit $STATUS
