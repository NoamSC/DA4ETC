#!/bin/bash
#SBATCH --job-name=week1_inference_tent
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=4:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

NUM_JOBS=4

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc

# Create logs directory if it doesn't exist
mkdir -p logs

# Use the full path to the conda environment's Python
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Rename log files with timestamp for chronological sorting
EXPNAME="week1_inference_tent_${SLURM_ARRAY_TASK_ID}of${NUM_JOBS}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}.err"

mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true

exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

$PYTHON scripts/inference/run_week1_inference_tent.py \
    --experiment_dir 'exps/cesnet_multimodal_each_week_train_v01' \
    --dataset_root '/home/anatbr/dataset/CESNET-TLS-Year22_v2' \
    --train_week 'week_1' \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device 'cuda:0' \
    --tent_lr 1e-3 \
    --tent_steps 1 \
    --num_jobs $NUM_JOBS \
    --job_id $SLURM_ARRAY_TASK_ID

echo "Completed week1 TENT inference (task ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS})"
