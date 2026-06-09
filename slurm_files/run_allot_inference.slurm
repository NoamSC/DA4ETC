#!/bin/bash
#SBATCH --job-name=allot_mm_infer
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=12:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1

# Run each trained Allot model across ALL windows of the Allot timeline:
#   array task 0 -> early_eq model   (the "week 1" model)
#   array task 1 -> quarter_eq model (the "week 16" model)
# Per-window predictions -> exps/allot_multimodal/<slice>_eq/inference/window_<idx>.npz
#
# NOTE: this runs one model over every window in a single task. If it is too slow,
# split windows across more array tasks with --num-shards / --shard-id (see the
# handoff doc) -- the exists() guard makes that safe to re-run.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

SLICES=(early quarter)
SLICE=${SLICES[$SLURM_ARRAY_TASK_ID]}
EXP_DIR="exps/allot_multimodal/${SLICE}_eq"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_allot_mm_infer_${SLICE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_allot_mm_infer_${SLICE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Allot inference: model=${EXP_DIR} over all windows ==="
$PYTHON scripts/inference/run_allot_inference.py \
    --experiment_dir "$EXP_DIR" \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device 'cuda:0'
STATUS=$?
echo "=== inference exit ${STATUS}; npz -> ${EXP_DIR}/inference/ ==="
exit $STATUS
