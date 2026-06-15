#!/bin/bash
#SBATCH --job-name=emfix_w1
#SBATCH --output=logs/emfix_w1_%j.out
#SBATCH --error=logs/emfix_w1_%j.out
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --partition=killable
#SBATCH --exclude=t-806
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Native SLURM logging (stdout+stderr -> same file); no mv/exec tricks.
cd /home/anatbr/students/noamshakedc/da4etc
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

echo "### host=$(hostname) job=$SLURM_JOB_ID start=$(date) ###"

# --- Week-1 EM-fix rerun.  Non-destructive: writes only into figs/isolation_emfix/
#     (published figs/isolation/ untouched; base cache pre-seeded so Ours/Naive/
#     ground-truth stay bit-identical and only corrected em/em_bcts recompute). ---

echo "=== STEP 1: isolation estimator variants (w1) ==="
$PYTHON -u scripts/analysis/isolation_estimator_variants.py \
    --inference_dir results/inference/week_1_inference \
    --reference_week 1 \
    --output_dir figs/isolation_emfix \
    --contamination
rc=$?
echo "=== STEP 1 finished rc=$rc ==="
if [ $rc -ne 0 ]; then echo "ABORT: step 1 failed"; exit $rc; fi

$PYTHON -u -c "import numpy as np,sys; z=np.load('figs/isolation_emfix/estimator_variants_cache.npz'); print('cache OK', list(z.keys()))" || { echo "ABORT: variant cache invalid"; exit 3; }

echo "=== STEP 2: label-shift severity sweep (w1) ==="
$PYTHON -u scripts/analysis/labelshift_severity_sweep.py \
    --inference_dir results/inference/week_1_inference \
    --reference_week 1 \
    --output_dir figs/isolation_emfix \
    --no_mfwdd \
    --variant_estimators \
    --suffix _estimators
rc=$?
echo "=== STEP 2 finished rc=$rc ==="
if [ $rc -ne 0 ]; then echo "ABORT: step 2 failed"; exit $rc; fi

echo "### Completed emfix_w1 end=$(date) ###"
