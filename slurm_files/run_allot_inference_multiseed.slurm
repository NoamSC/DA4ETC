#!/bin/bash
#SBATCH --job-name=allot_infer_seed
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
#SBATCH --array=0-15

# Multi-seed inference for the headline comparison: quarter_eq (CLEAN source) slice,
# bs256, all 4 existing adapters, seeds {1,2,3,4} (seed 42 already done at bs64 and
# is being re-run at bs256 by run_allot_inference_bs256.slurm). The --seed sets the
# per-window 10% data subsample draw (matched-protocol seed semantics).
#   task = seed_idx * 4 + method_idx
#   seeds = (1 2 3 4); methods = (vanilla bnstats tent cotta)
# Output -> inference_<method>_bs256_seed<N>/  (or inference_bs256_seed<N>/ for vanilla)

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

SEEDS=(1 2 3 4)
METHODS=(vanilla bnstats tent cotta)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 4))]}
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID % 4))]}
EXP_DIR="exps/allot_multimodal/quarter_eq"
SUFFIX="inference"
[ "$METHOD" != "vanilla" ] && SUFFIX="inference_${METHOD}"
OUT_DIR="${EXP_DIR}/${SUFFIX}_bs256_seed${SEED}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_allot_infer_seed${SEED}_${METHOD}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_allot_infer_seed${SEED}_${METHOD}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== multiseed inference: ${EXP_DIR} method=${METHOD} seed=${SEED} -> ${OUT_DIR} ==="
$PYTHON scripts/inference/run_allot_inference.py \
    --experiment_dir "$EXP_DIR" \
    --method "$METHOD" \
    --output_dir "$OUT_DIR" \
    --batch_size 256 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed "$SEED" \
    --device 'cuda:0'
STATUS=$?
echo "=== multiseed inference exit ${STATUS}; npz -> ${OUT_DIR} ==="
exit $STATUS
