#!/bin/bash
#SBATCH --job-name=allot_bnstats
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
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1

# AdaBN (BN test-batch statistics, NO gradient updates) for the Allot multimodal models,
# over all windows. Same protocol/seed/frac as the vanilla/TENT/CoTTA Allot runs ->
# directly comparable. AdaBN is fast (~vanilla speed), so no sharding.
#   task 0 = early slice, task 1 = quarter slice
# Output -> exps/allot_multimodal/<slice>_eq/inference_bnstats/window_*.npz
# Idempotent: run_allot_inference.py skips windows whose .npz already exists.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
export TQDM_DISABLE=1

IDX=$SLURM_ARRAY_TASK_ID
SLICES=(early quarter)
SLICE=${SLICES[$IDX]}
EXP_DIR="exps/allot_multimodal/${SLICE}_eq"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_allot_bnstats_${SLICE}_${SLURM_ARRAY_JOB_ID}_${IDX}.out"
ERRFILE="logs/${TIMESTAMP}_allot_bnstats_${SLICE}_${SLURM_ARRAY_JOB_ID}_${IDX}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${IDX}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${IDX}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Allot bnstats (AdaBN) on ${EXP_DIR} ==="
$PYTHON scripts/inference/run_allot_inference.py \
    --experiment_dir "$EXP_DIR" \
    --method bnstats \
    --shuffle \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device 'cuda:0'
STATUS=$?
echo "=== bnstats/${SLICE} exit ${STATUS} ==="
exit $STATUS
