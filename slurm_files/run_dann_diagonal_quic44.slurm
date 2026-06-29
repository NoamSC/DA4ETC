#!/bin/bash
#SBATCH --job-name=dann_diag_quic44
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=02:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#
# DIAGONAL inference for the QUIC WEEK-44-source DANN forward-transfer sweep
# (cesnet_quic22_dann_fwd_w44_v01). For each target week N in {45,46,47} it loads the
# model DANN-aligned source-week-44 -> week N (WEEK-2022-44_val_WEEK-2022-NN) and
# frozen-evaluates it on week N's test split -- the "DANN row" of the QUIC block.
# Only 3 target weeks, so no sharding. Benchmark protocol (data_sample_frac=0.1,
# seed=42), skip-existing + atomic .npz writes. Output:
#   results/inference/dann_fwd_quic44_diagonal/WEEK-2022-NN.npz

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_dann_diag_quic44_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_dann_diag_quic44_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== DANN diagonal (QUIC week-44) inference ==="
$PYTHON scripts/inference/run_dann_diagonal_inference.py \
    --experiment_dir exps/cesnet_quic22_dann_fwd_w44_v01 \
    --dataset_root /home/anatbr/students/noamshakedc/cesnet-quic22-prepared \
    --source_week 44 \
    --output_dir results/inference/dann_fwd_quic44_diagonal \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42
STATUS=$?
echo "=== DANN diagonal (QUIC week-44) exit ${STATUS} ==="
exit $STATUS
