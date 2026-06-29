#!/bin/bash
#SBATCH --job-name=dann_diag_w01
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=04:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --exclude=n-301
#
# DIAGONAL inference for the WEEK-1-source DANN forward-transfer sweep
# (cesnet_tls_dann_fwd_w01_v01). For each target week N (!=1) it loads the model
# DANN-aligned source-week-1 -> week N (WEEK-2022-01_val_WEEK-2022-NN) and frozen-
# evaluates it on week N's test split -- the "DANN row" of the week-1 benchmark block.
# Vanilla eval (model.eval()), benchmark protocol (data_sample_frac=0.1, seed=42),
# skip-existing + atomic .npz writes, sharded 4 ways. Output:
#   results/inference/dann_fwd_w01_diagonal/WEEK-2022-NN.npz
NUM_JOBS=4

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="dann_diag_w01_${SLURM_ARRAY_TASK_ID}"
OUTFILE="logs/${TIMESTAMP}_${TAG}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${TAG}_${SLURM_ARRAY_JOB_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

if ! $PYTHON -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" 2>/dev/null; then
    echo "=== GPU dead on $(hostname); requeueing task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" 2>/dev/null
    sleep 30
    exit 1
fi

echo "=== DANN diagonal (week-1) inference shard ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS} ==="
$PYTHON scripts/inference/run_dann_diagonal_inference.py \
    --experiment_dir exps/cesnet_tls_dann_fwd_w01_v01 \
    --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22_v2 \
    --source_week 01 \
    --output_dir results/inference/dann_fwd_w01_diagonal \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --num_jobs $NUM_JOBS \
    --job_id $SLURM_ARRAY_TASK_ID
STATUS=$?
echo "=== DANN diagonal (week-1) shard ${SLURM_ARRAY_TASK_ID} exit ${STATUS} ==="
exit $STATUS
