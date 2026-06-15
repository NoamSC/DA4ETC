#!/bin/bash
#SBATCH --job-name=dann_fwd_w16
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
# FORWARD-TRANSFER DANN for the UDA benchmark (REPLACES run_dann_train.slurm, which was
# invalid: it passed no --val_week, so DANN aligned each week's train split to that SAME
# week's test split — no domain gap, domain classifier stuck at 50% / dann loss at 2*ln2
# for all 50 epochs, so DANN reduced to a source-only model with dead heads).
#
# The fix is a ONE-FLAG change: pass --val_week so the DANN target is a DIFFERENT week.
# This is the exact, already-proven path used for the past cross-week DANN runs (e.g.
# exps/cesnet_v11_dann_search_13_to_33, dirs named WEEK-2022-13_val_WEEK-2022-33, where
# the domain classifier reached ~96% — real alignment). No trainer code changes needed.
#
# SOURCE is fixed at WEEK-2022-16 (--week 16). Array task i is the TARGET week
# (--val_week i): the week-16 labeled source is aligned to WEEK-2022-i's test split via
# the gradient-reversal domain head, and that same split is the validation/selection set
# (transductive UDA, matching prior methodology). The resulting model is frozen-evaluated
# on WEEK-2022-i downstream — one model per target week (the DANN-row diagonal).
#
# Hyperparameters mirror the prior run: lambda_dann=1.0, GRL lambda_rgl=0.1, gamma=10
# (gamma>0 REQUIRED — gamma=0 makes the GRL coefficient identically 0, disabling DANN).
# train_per_week auto-names the experiment WEEK-2022-16_val_WEEK-2022-NN when val!=src.
# Output: exps/cesnet_tls_dann_fwd_w16_v01/WEEK-2022-16_val_WEEK-2022-NN/weights/best_model.pth

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
DATASET_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2

SRC_WEEK=16
TGT_WEEK=$SLURM_ARRAY_TASK_ID
TGT2=$(printf "%02d" "$TGT_WEEK")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_dann_fwd_w16_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_dann_fwd_w16_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Target == source is a degenerate no-gap run (this is exactly the old bug); skip it.
if [ "$TGT_WEEK" -eq "$SRC_WEEK" ]; then
    echo "=== TGT week ${TGT2} == source ${SRC_WEEK}; degenerate (no domain gap), skipping ==="
    exit 0
fi

echo "=== DANN forward-transfer: SOURCE WEEK-2022-16 -> TARGET (--val_week) WEEK-2022-${TGT2} ==="
$PYTHON scripts/train/train_per_week_cesnet.py \
    --multimodal \
    --week "$SRC_WEEK" \
    --val_week "$TGT_WEEK" \
    --exp_name "cesnet_tls_dann_fwd_w16_v01/{}" \
    --dataset_root "$DATASET_ROOT" \
    --lambda_dann 1.0 \
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
echo "=== DANN forward-transfer exit ${STATUS} for SOURCE 16 -> TARGET ${TGT2} ==="
exit $STATUS
