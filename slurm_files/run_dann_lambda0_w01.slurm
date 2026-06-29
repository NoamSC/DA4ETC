#!/bin/bash
#SBATCH --job-name=dann_l0_w01
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
#
# ALIGNMENT-vs-RETRAINING CONFOUND CONTROL for the WEEK-1-source DANN forward sweep.
# Exact replica of run_dann_fwd_w01.slurm EXCEPT --lambda_dann 0.0 (was 1.0).
# With lambda_dann=0 the gradient-reversal domain-adversarial loss is NEVER added to
# the total loss (trainer.py L112 gate `lambda_dann > 0` is False; L127 multiplies by
# 0), so the domain head / GRL contribute no gradient -> a TRUE no-alignment control.
# Everything else is held identical to the lambda>0 run: fresh per-week model trained
# on the week-1 labeled source for 50 epochs, target week-N test split still loaded as
# the val/selection set and mixed into the combined BN forward. The ONLY difference is
# the adversarial alignment objective. If lambda=0 already recovers the +0.067 forward
# gain, the gain is retraining/re-init, not alignment.
#
# Array task i is the TARGET week (--val_week i). train_per_week auto-names the run
# WEEK-2022-01_val_WEEK-2022-NN under a DISTINCT exp dir so it never clobbers the
# lambda>0 run (cesnet_tls_dann_fwd_w01_v01).
# Output: exps/cesnet_tls_dann_lambda0_w01_v01/WEEK-2022-01_val_WEEK-2022-NN/weights/best_model.pth

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
DATASET_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2

SRC_WEEK=1
TGT_WEEK=$SLURM_ARRAY_TASK_ID
TGT2=$(printf "%02d" "$TGT_WEEK")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_dann_l0_w01_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_dann_l0_w01_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Target == source is a degenerate no-gap run; skip it (matches the lambda>0 sweep).
if [ "$TGT_WEEK" -eq "$SRC_WEEK" ]; then
    echo "=== TGT week ${TGT2} == source ${SRC_WEEK}; degenerate (no domain gap), skipping ==="
    exit 0
fi

echo "=== DANN-lambda0 CONTROL: SOURCE WEEK-2022-01 -> TARGET (--val_week) WEEK-2022-${TGT2} (lambda_dann=0) ==="
$PYTHON scripts/train/train_per_week_cesnet.py \
    --multimodal \
    --week "$SRC_WEEK" \
    --val_week "$TGT_WEEK" \
    --exp_name "cesnet_tls_dann_lambda0_w01_v01/{}" \
    --dataset_root "$DATASET_ROOT" \
    --lambda_dann 0.0 \
    --lambda_rgl 0.1 \
    --lambda_grl_gamma 10 \
    --train_data_frac 1.0 \
    --val_data_frac 0.1 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 50 \
    --batch_size 64 \
    --num_workers 0 \
    --learning_rate 3e-3
STATUS=$?
echo "=== DANN-lambda0 control exit ${STATUS} for SOURCE 01 -> TARGET ${TGT2} ==="
exit $STATUS
