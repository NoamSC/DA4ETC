#!/bin/bash
#SBATCH --job-name=tls_tta
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=8:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

# Generic parametrized CESNET-TLS-Year22 TTA/AdaBN inference (vanilla/bnstats/tent/cotta).
# Source model = $EXP_DIR/$TRAIN_WEEK; forward-eval over all 53 weeks, split 4 ways.
# Idempotent: run_inference.py skips weeks whose .npz already exists (safe to requeue).
#
# Submit with --export, e.g.:
#   sbatch --export=ALL,METHOD=tent,TRAIN_WEEK=week_16,\
# OUTPUT_DIR=results/inference/week_16_inference_tent_fixed,BATCH_SIZE=64 \
#       slurm_files/run_tls_tta_param.slurm

NUM_JOBS=4
METHOD="${METHOD:?set METHOD=vanilla|bnstats|tent|cotta}"
TRAIN_WEEK="${TRAIN_WEEK:?set TRAIN_WEEK=week_1|week_16}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR=results/inference/...}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EXP_DIR="${EXP_DIR:-exps/cesnet_multimodal_each_week_train_v01}"

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

EXPNAME="tls_${METHOD}_${TRAIN_WEEK}_bs${BATCH_SIZE}_${SLURM_ARRAY_TASK_ID}of${NUM_JOBS}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== TLS $METHOD | $TRAIN_WEEK | bs$BATCH_SIZE | $EXP_DIR -> $OUTPUT_DIR ==="
$PYTHON scripts/inference/run_inference.py \
    --method "$METHOD" \
    --experiment_dir "$EXP_DIR" \
    --dataset_root '/home/anatbr/dataset/CESNET-TLS-Year22_v2' \
    --train_week "$TRAIN_WEEK" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --embed_seed 0 \
    --device 'cuda:0' \
    --tent_lr 1e-3 --tent_steps 1 \
    --cotta_lr 1e-3 --cotta_steps 1 --rst_m 0.01 --ap 0.9 --n_aug 32 --noise_std 0.02 \
    --num_jobs $NUM_JOBS \
    --job_id $SLURM_ARRAY_TASK_ID

echo "Completed $METHOD $TRAIN_WEEK bs$BATCH_SIZE (task ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS})"
