#!/bin/bash
#SBATCH --job-name=infer_seedvar
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
# Multi-SEED significance study for the TTA controls (AdaBN/TENT) — closes audit
# red flags #7 (AdaBN deltas may be within run-to-run noise) and #8 (TENT bs64
# magnitude). Runs vanilla/bnstats/tent across several seeds; per seed the delta
# vs vanilla is computed on the SAME sampled population, then mean +/- std across
# seeds tells us if the effect is significant or noise.
#
# --acc_only => saves ONLY true+pred labels (~3MB/file vs 286MB) — disk-lean.
# Outputs to results/inference_seedvar/seed<S>_<week>_<method>_bs<B>/ (new tree).
# Required exported vars: SEEDV, METHOD, TRAIN_WEEK, BATCH, EXP_DIR, DATA_ROOT, NUM_JOBS
# Submit with: sbatch --array=0-3 (TLS, NUM_JOBS=4) or --array=0 (QUIC, NUM_JOBS=1).
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
: "${SEEDV:?}"; : "${METHOD:?}"; : "${TRAIN_WEEK:?}"; : "${BATCH:?}"
: "${EXP_DIR:?}"; : "${DATA_ROOT:?}"; : "${NUM_JOBS:?}"
OUTDIR="results/inference_seedvar/seed${SEEDV}_${TRAIN_WEEK}_${METHOD}_bs${BATCH}"

case "$METHOD" in
  tent)  EXTRA="--tent_lr 1e-3 --tent_steps 1" ;;
  cotta) EXTRA="--cotta_lr 1e-3 --cotta_steps 1" ;;
  *)     EXTRA="" ;;
esac

TAG="seedvar_s${SEEDV}_${TRAIN_WEEK}_${METHOD}_bs${BATCH}_${SLURM_ARRAY_TASK_ID}"
TS=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TS}_${TAG}_${SLURM_ARRAY_JOB_ID}.out"; ERRFILE="logs/${TS}_${TAG}_${SLURM_ARRAY_JOB_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"; exec 2>>"$ERRFILE"

echo "=== seedvar seed=${SEEDV} ${METHOD} ${TRAIN_WEEK} bs${BATCH} shard ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS} -> ${OUTDIR} ==="
$PYTHON scripts/inference/run_inference.py \
    --method "$METHOD" \
    --experiment_dir "$EXP_DIR" \
    --dataset_root "$DATA_ROOT" \
    --train_week "$TRAIN_WEEK" \
    --output_dir "$OUTDIR" \
    --batch_size "$BATCH" \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed "$SEEDV" \
    --device 'cuda:0' \
    --acc_only \
    --num_jobs "$NUM_JOBS" \
    --job_id "$SLURM_ARRAY_TASK_ID" \
    $EXTRA
echo "=== exit $? : ${OUTDIR} ==="
