#!/bin/bash
#SBATCH --job-name=quic_mm_train
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=12:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Train the multimodal model (Multimodal_CESNET: PPI conv1d + flowstats MLP) on the
# first week of CESNET-QUIC22 (W-2022-44), prepared into TLS-Year22-compatible layout
# by scripts/data_prep/prepare_quic_week.py.
# Checkpoints -> exps/cesnet_quic22_multimodal_v01/WEEK-2022-44/weights/best_model.pth
#
# --requeue + USR1 signal: the job checkpoints per epoch and resumes automatically if
# preempted on the killable partition (re-running resumes from the latest checkpoint).

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

DATASET_ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared
WEEK=44

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_quic_mm_train_w${WEEK}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_quic_mm_train_w${WEEK}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Training multimodal model on CESNET-QUIC22 W-2022-${WEEK} ==="
$PYTHON scripts/train/train_per_week_cesnet.py \
    --multimodal \
    --week "$WEEK" \
    --exp_name "cesnet_quic22_multimodal_v01/{}" \
    --dataset_root "$DATASET_ROOT" \
    --train_data_frac 0.2 \
    --val_data_frac 0.05 \
    --train_per_epoch_data_frac 1.0 \
    --num_epochs 30 \
    --batch_size 256 \
    --learning_rate 3e-3
STATUS=$?

echo "=== train exit ${STATUS}; checkpoint -> exps/cesnet_quic22_multimodal_v01/WEEK-2022-${WEEK}/weights/best_model.pth ==="
exit $STATUS
