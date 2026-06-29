#!/bin/bash
#SBATCH --job-name=mmd_fwd_w16
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
#SBATCH --exclude=n-301
#
# FORWARD-TRANSFER MMD/DAN for the UDA benchmark — the MMD analogue of
# run_coral_fwd_w16.slurm. MMD (Long et al. 2015, DAN) aligns the marginal feature
# distributions of source and target via a multi-kernel RBF MMD penalty instead of an
# adversarial domain head, so there are no GRL/gamma knobs — only --lambda_mmd. Source+
# target are forwarded as ONE combined batch (the same BN-fix path DANN/CORAL use), so
# the covariate shift is preserved for the MMD term.
#
# SOURCE fixed at WEEK-2022-16 (--week 16); array task i is the TARGET week (--val_week i):
# the week-16 labeled source is aligned to WEEK-2022-i's unlabeled test split, which is
# also the validation/selection set (transductive UDA, matching the DANN/CORAL sweep). One
# model per target week -> the MMD-row diagonal.
#
# LAMBDA_MMD is calibrated (mmd_lcalib) so the MMD term is active without dominating the
# classification loss. compute_mmd_loss z-normalizes features, so the raw loss is O(1) and
# LAMBDA_MMD is SMALL. Output:
#   exps/cesnet_tls_mmd_fwd_w16_v01/WEEK-2022-16_val_WEEK-2022-NN/weights/best_model.pth
LAMBDA_MMD=5

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
DATASET_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2

SRC_WEEK=16
TGT_WEEK=$SLURM_ARRAY_TASK_ID
TGT2=$(printf "%02d" "$TGT_WEEK")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_mmd_fwd_w16_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_mmd_fwd_w16_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Target == source is degenerate (no domain gap); skip it.
if [ "$TGT_WEEK" -eq "$SRC_WEEK" ]; then
    echo "=== TGT week ${TGT2} == source ${SRC_WEEK}; degenerate, skipping ==="
    exit 0
fi

# GPU health guard: requeue off a node whose CUDA driver is dead.
if ! $PYTHON -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" 2>/dev/null; then
    echo "=== GPU dead on $(hostname); requeueing ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" 2>/dev/null
    sleep 30; exit 1
fi

echo "=== MMD forward-transfer: SOURCE WEEK-2022-16 -> TARGET (--val_week) WEEK-2022-${TGT2}, lambda_mmd=${LAMBDA_MMD} ==="
$PYTHON scripts/train/train_per_week_cesnet.py \
    --multimodal \
    --week "$SRC_WEEK" \
    --val_week "$TGT_WEEK" \
    --exp_name "cesnet_tls_mmd_fwd_w16_v01/{}" \
    --dataset_root "$DATASET_ROOT" \
    --lambda_mmd "$LAMBDA_MMD" \
    --train_data_frac 1.0 \
    --val_data_frac 0.1 \
    --train_per_epoch_data_frac 0.1 \
    --num_epochs 50 \
    --batch_size 64 \
    --num_workers 0 \
    --learning_rate 3e-3
STATUS=$?
echo "=== MMD forward-transfer exit ${STATUS} for SOURCE 16 -> TARGET ${TGT2} ==="
exit $STATUS
