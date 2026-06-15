#!/bin/bash
#SBATCH --job-name=dann_normc
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=8:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0,10,25,33,44,52
#
# DANN NO-NORMALIZATION CONTRAST (diagnostic, NOT a benchmark). Identical to
# run_dann_fwd_w16.slurm (source week16 -> target week N via --val_week) EXCEPT it
#
# Purpose: confirm the "no global covariate gap" finding is real. The normalized sweep
# (cesnet_tls_dann_fwd_w16_v01) showed domain-classifier accuracy pinned at ~50% (weeks
# indistinguishable after shared normalization). If WITHOUT normalization the domain
# accuracy jumps back to ~90%+, that proves the old runs' separability was a raw
# input-scale artifact, not real drift -> supports the label-shift thesis.
#
# Representative 6-week target subset (early/mid/late) is enough for the contrast;
# kept small + only 10 epochs (the domain classifier separates by epoch ~1 when a gap
# exists) to minimize GPU and DISK. Rolling-checkpoint fix keeps ~2 files per model.
# Output: exps/cesnet_tls_dann_fwd_w16_normctrl_v01/WEEK-2022-16_val_WEEK-2022-NN/

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
DATASET_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2

SRC_WEEK=16
TGT_WEEK=$SLURM_ARRAY_TASK_ID
TGT2=$(printf "%02d" "$TGT_WEEK")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_dann_normc_w16_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_dann_normc_w16_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== DANN NORM control: SOURCE WEEK-2022-16 -> TARGET WEEK-2022-${TGT2} (inputs SHARED-normalized (matched control)) ==="
$PYTHON scripts/train/train_per_week_cesnet.py \
    --multimodal \
    --week "$SRC_WEEK" \
    --val_week "$TGT_WEEK" \
    --exp_name "cesnet_tls_dann_fwd_w16_normctrl_v01/{}" \
    --dataset_root "$DATASET_ROOT" \
    --lambda_dann 1.0 \
    --lambda_rgl 0.1 \
    --lambda_grl_gamma 10 \
    --train_data_frac 1.0 \
    --val_data_frac 0.1 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 10 \
    --batch_size 64 \
    --num_workers 0 \
    --learning_rate 3e-3
STATUS=$?
echo "=== DANN NORM-CTRL exit ${STATUS} for SOURCE 16 -> TARGET ${TGT2} ==="
exit $STATUS
