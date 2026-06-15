#!/bin/bash
#SBATCH --job-name=infer_auditfix
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=08:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#
# Generic fixed-loader re-run of the UDA/TTA benchmark, after the 2026-06-14 validity
# audit (deterministic seeded eval loader + drop_last=False; CoTTA clean-prediction +
# BN-buffer reset). Writes ONLY under results/inference_auditfix/ so NOTHING under the
# existing results/inference/ tree is overridden — old (buggy) numbers stay for compare.
#
# Required exported vars: METHOD, TRAIN_WEEK, BATCH, EXP_DIR, DATA_ROOT
# 53 weeks are sharded across 4 array tasks (--num_jobs/--job_id), as in the prior runs.
NUM_JOBS=4
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

: "${METHOD:?set METHOD}"; : "${TRAIN_WEEK:?set TRAIN_WEEK}"; : "${BATCH:?set BATCH}"
: "${EXP_DIR:?set EXP_DIR}"; : "${DATA_ROOT:?set DATA_ROOT}"
OUTDIR="results/inference_auditfix/${TRAIN_WEEK}_${METHOD}_bs${BATCH}"

case "$METHOD" in
  tent)  EXTRA="--tent_lr 1e-3 --tent_steps 1" ;;
  cotta) EXTRA="--cotta_lr 1e-3 --cotta_steps 1" ;;
  *)     EXTRA="" ;;
esac

TAG="auditfix_${TRAIN_WEEK}_${METHOD}_bs${BATCH}_${SLURM_ARRAY_TASK_ID}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${TAG}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${TAG}_${SLURM_ARRAY_JOB_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== auditfix ${METHOD} ${TRAIN_WEEK} bs${BATCH} shard ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS} -> ${OUTDIR} ==="
$PYTHON scripts/inference/run_inference.py \
    --method "$METHOD" \
    --experiment_dir "$EXP_DIR" \
    --dataset_root "$DATA_ROOT" \
    --train_week "$TRAIN_WEEK" \
    --output_dir "$OUTDIR" \
    --batch_size "$BATCH" \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --embed_seed 0 \
    --device 'cuda:0' \
    --num_jobs $NUM_JOBS \
    --job_id $SLURM_ARRAY_TASK_ID \
    $EXTRA
echo "=== exit $? : ${OUTDIR} ==="
