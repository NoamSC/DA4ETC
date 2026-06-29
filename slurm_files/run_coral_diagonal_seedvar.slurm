#!/bin/bash
#SBATCH --job-name=coral_diag_seed
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
# SEED-VARIANT diagonal inference for the BN-fixed CORAL forward-transfer sweep
# (cesnet_tls_coral_fwd_w16_v01). Identical to run_coral_diagonal.slurm except the EVAL
# seed is taken from --export=ALL,SEED=$S (the diagonal models are fixed; only the 0.1
# data subsample changes). Re-running at seeds {1,2,3,4} gives the evaluation-noise band,
# so we can test whether the CORAL-vs-vanilla Delta=-0.001 is distinguishable from 0.
# Output: results/inference/coral_fwd_w16_diagonal_seed${SEED}/WEEK-2022-NN.npz
NUM_JOBS=4

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TAG="coral_diag_seed${SEED}_${SLURM_ARRAY_TASK_ID}"
OUTFILE="logs/${TIMESTAMP}_${TAG}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${TAG}_${SLURM_ARRAY_JOB_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# GPU health guard: some killable nodes accept GPU jobs but have a broken driver
# (torch._C._cuda_init() fails). Test CUDA before the real work; if it's dead,
# requeue this task onto a different node instead of dying with exit 1.
if ! $PYTHON -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" 2>/dev/null; then
    echo "=== GPU dead on $(hostname); requeueing task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ==="
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" 2>/dev/null
    sleep 30
    exit 1
fi

echo "=== CORAL diagonal inference (seed=${SEED}) shard ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS} ==="
$PYTHON scripts/inference/run_dann_diagonal_inference.py \
    --experiment_dir exps/cesnet_tls_coral_fwd_w16_v01 \
    --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22_v2 \
    --source_week 16 \
    --output_dir results/inference/coral_fwd_w16_diagonal_seed${SEED} \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed $SEED \
    --num_jobs $NUM_JOBS \
    --job_id $SLURM_ARRAY_TASK_ID
STATUS=$?
echo "=== CORAL diagonal (seed=${SEED}) shard ${SLURM_ARRAY_TASK_ID} exit ${STATUS} ==="
exit $STATUS
