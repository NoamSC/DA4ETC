#!/bin/bash
#SBATCH --job-name=week16_inference_rev
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=4:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Rename log files with timestamp for chronological sorting
EXPNAME="week16_inference_rev"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"

mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --reversed: process weeks 52 -> 00 so this job meets the forward job in the
# middle. Both write to the same figs/week_16_inference/ and skip any week the
# other has already produced.
$PYTHON scripts/inference/run_week1_inference.py \
    --experiment_dir 'exps/cesnet_multimodal_each_week_train_v01' \
    --dataset_root '/home/anatbr/dataset/CESNET-TLS-Year22_v2' \
    --train_week 'week_16' \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device 'cuda:0' \
    --reversed

echo "Completed week16 inference (reversed)"
