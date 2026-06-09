#!/bin/bash
#SBATCH --job-name=allot_tta
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
#SBATCH --array=0-9

# Test-time adaptation (TENT / CoTTA) for the Allot multimodal models, over all
# windows of the Allot timeline. Reuses the TENT/CoTTA implementations from
# scripts/inference/run_inference.py via run_allot_inference.py --method.
#
# CoTTA is ~33 min/window (n_aug=32 fwd passes), so 52 windows wouldn't fit a 12h
# wall -> CoTTA is SHARDED 4 ways per model (each task does windows where
# idx %% 4 == shard). TENT (~8 min/window) runs all windows in one task.
# The exists() guard makes every task safe to re-run.
#
#   task 0  tent  early    (1 shard)        task 1  tent  quarter  (1 shard)
#   task 2-5 cotta early   (shards 0..3)     task 6-9 cotta quarter (shards 0..3)
#
# Output -> exps/allot_multimodal/<slice>_eq/{inference_tent,inference_cotta}/window_*.npz

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
# Silence tqdm: its progress-bar flushes are thousands of tiny writes to the .err
# log, and a single one failing on a transient shared-quota blip kills the whole
# job (it did, once). Disabling it leaves only the ~9MB-per-window npz writes.
export TQDM_DISABLE=1

IDX=$SLURM_ARRAY_TASK_ID
METHODS=(tent  tent    cotta cotta cotta cotta cotta  cotta   cotta   cotta)
SLICES=( early quarter early early early early quarter quarter quarter quarter)
SHARDS=( 0     0       0     1     2     3     0       1       2       3)
NSHARDS=(1     1       4     4     4     4     4       4       4       4)
METHOD=${METHODS[$IDX]}; SLICE=${SLICES[$IDX]}; SHARD=${SHARDS[$IDX]}; NSHARD=${NSHARDS[$IDX]}
EXP_DIR="exps/allot_multimodal/${SLICE}_eq"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_allot_tta_${METHOD}_${SLICE}_s${SHARD}of${NSHARD}_${SLURM_ARRAY_JOB_ID}_${IDX}.out"
ERRFILE="logs/${TIMESTAMP}_allot_tta_${METHOD}_${SLICE}_s${SHARD}of${NSHARD}_${SLURM_ARRAY_JOB_ID}_${IDX}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${IDX}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${IDX}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Allot ${METHOD} on ${EXP_DIR}, shard ${SHARD}/${NSHARD} ==="
$PYTHON scripts/inference/run_allot_inference.py \
    --experiment_dir "$EXP_DIR" \
    --method "$METHOD" \
    --num-shards "$NSHARD" --shard-id "$SHARD" \
    --shuffle \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device 'cuda:0'
STATUS=$?
echo "=== ${METHOD}/${SLICE} shard ${SHARD}/${NSHARD} exit ${STATUS} ==="
exit $STATUS
