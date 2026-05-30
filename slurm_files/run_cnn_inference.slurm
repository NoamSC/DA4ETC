#!/bin/bash
#SBATCH --job-name=cnn_infer
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=08:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1

# Run inference with the trained flowpic CNNs across all weeks.
# Array task 0 -> week-1 model, task 1 -> week-16 model.
#
# Results layout supports multiple datasets and models:
#   results/inference/<dataset>/<model>/train_week_<NN>/WEEK-2022-XX.npz

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

WEEKS=(1 16)
WEEK=${WEEKS[$SLURM_ARRAY_TASK_ID]}
WEEK2=$(printf "%02d" "$WEEK")

DATASET=cesnet-tls-year22
MODEL=cnn
DATASET_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2
OUT_DIR="results/inference/${DATASET}/${MODEL}/train_week_${WEEK2}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_cnn_infer_w${WEEK2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_cnn_infer_w${WEEK2}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Inference: CNN trained on WEEK-2022-${WEEK2} -> ${OUT_DIR} ==="
$PYTHON scripts/inference/run_week1_inference.py \
    --experiment_dir exps/cesnet_cnn \
    --train_week "WEEK-2022-${WEEK2}" \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUT_DIR" \
    --data_sample_frac 0.1 \
    --batch_size 64 \
    --num_workers 0 \
    --device cuda:0
STATUS=$?

echo "=== inference exit ${STATUS}; outputs -> ${OUT_DIR} ==="
exit $STATUS
