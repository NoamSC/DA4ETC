#!/bin/bash
#SBATCH --job-name=inference_bnstats
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=4:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

# AdaBN (BN test-batch statistics, NO gradient updates) for the CESNET-TLS-Year22
# week-1 model across all 53 weeks. Same protocol/seed/frac as the vanilla/TENT/CoTTA
# week_1_inference runs -> directly comparable. The 53 weeks are split across 4 array
# tasks via --num_jobs/--job_id.

NUM_JOBS=4
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

EXPNAME="inference_bnstats_${SLURM_ARRAY_TASK_ID}of${NUM_JOBS}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

$PYTHON scripts/inference/run_inference.py \
    --method bnstats \
    --experiment_dir 'exps/cesnet_multimodal_each_week_train_v01' \
    --dataset_root '/home/anatbr/dataset/CESNET-TLS-Year22_v2' \
    --train_week 'week_1' \
    --output_dir 'results/inference/week_1_inference_bnstats' \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --embed_seed 0 \
    --device 'cuda:0' \
    --num_jobs $NUM_JOBS \
    --job_id $SLURM_ARRAY_TASK_ID

echo "Completed AdaBN (bnstats) inference (task ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS})"
