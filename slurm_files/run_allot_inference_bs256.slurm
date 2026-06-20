#!/bin/bash
#SBATCH --job-name=allot_infer_bs256
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
#SBATCH --array=0-7

# bs256 re-run of the four existing adapters on BOTH slices (matched protocol:
# CESNET tables report bs64 AND bs256). Output dirs get a _bs256 suffix so the
# already-computed bs64/seed42 outputs are NOT overwritten.
#   task = slice_idx * 4 + method_idx
#   slices  = (early quarter); methods = (vanilla bnstats tent cotta)

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

SLICES=(early quarter)
METHODS=(vanilla bnstats tent cotta)
SLICE=${SLICES[$((SLURM_ARRAY_TASK_ID / 4))]}
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID % 4))]}
EXP_DIR="exps/allot_multimodal/${SLICE}_eq"
SUFFIX="inference"
[ "$METHOD" != "vanilla" ] && SUFFIX="inference_${METHOD}"
OUT_DIR="${EXP_DIR}/${SUFFIX}_bs256"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_allot_infer_bs256_${SLICE}_${METHOD}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_allot_infer_bs256_${SLICE}_${METHOD}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== bs256 inference: ${EXP_DIR} method=${METHOD} -> ${OUT_DIR} ==="
$PYTHON scripts/inference/run_allot_inference.py \
    --experiment_dir "$EXP_DIR" \
    --method "$METHOD" \
    --output_dir "$OUT_DIR" \
    --batch_size 256 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device 'cuda:0'
STATUS=$?
echo "=== bs256 inference exit ${STATUS}; npz -> ${OUT_DIR} ==="
exit $STATUS
