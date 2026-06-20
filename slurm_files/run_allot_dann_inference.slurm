#!/bin/bash
#SBATCH --job-name=allot_dann_infer
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

# Forward-only inference of the DANN-trained models over all windows.
#   array task 0 -> early_eq_dann model
#   array task 1 -> quarter_eq_dann model (clean source, the headline)
# The DANN model is evaluated FROZEN with the vanilla forward pass (no test-time
# adaptation); only the trained weights differ from Source-only. Output goes to
# <exp>_dann/inference_dann/ so the table tool's SUFFIX["dann"] picks it up.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

SLICES=(early quarter)
SLICE=${SLICES[$SLURM_ARRAY_TASK_ID]}
EXP_DIR="exps/allot_multimodal/${SLICE}_eq_dann"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_allot_dann_infer_${SLICE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_allot_dann_infer_${SLICE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== DANN inference: model=${EXP_DIR} over all windows -> inference_dann/ ==="
$PYTHON scripts/inference/run_allot_inference.py \
    --experiment_dir "$EXP_DIR" \
    --method vanilla \
    --output_dir "${EXP_DIR}/inference_dann" \
    --batch_size 256 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device 'cuda:0'
STATUS=$?
echo "=== dann inference exit ${STATUS}; npz -> ${EXP_DIR}/inference_dann/ ==="
exit $STATUS
