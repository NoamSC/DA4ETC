#!/bin/bash
#SBATCH --job-name=quic_bbse_monitor
#SBATCH --output=logs/slurm-%A.out.tmp
#SBATCH --error=logs/slurm-%A.err.tmp
#SBATCH --time=2:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# BBSE-corrected entropy-residual anomaly monitor (AUROC / FAR) on CESNET-QUIC22.
# Closes the one core gap: the monitor was already run on TLS (results/anomaly_mm/)
# and Allot, but NOT QUIC.
#
# Source = QUIC WEEK-2022-44 (102 classes); eval weeks 45/46/47 (only 4 weeks
# exist for QUIC; week 48 has no inference output). Inputs are the VANILLA
# frozen-source multimodal inference (102-class softmax confirmed):
#   results/inference_auditfix/WEEK-2022-44_vanilla_bs64/  (WEEK-2022-44..47.npz)
#
# Threshold note: QUIC degradation is mild -- vanilla macro-F1 is ~0.69/0.63/0.64
# for weeks 45/46/47, all above the script default --f1_threshold 0.60 (which would
# give ZERO anomaly-positive weeks and an undefined AUROC). We use 0.66, the natural
# gap between the healthy week 45 and the degraded weeks 46/47 (-> 1 healthy / 2
# degraded). With only 3 eval weeks this is an inherently small-sample result.
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
mkdir -p results/quic_monitor

echo "=== QUIC week-44 source, reference_week=44, MULTIMODAL vanilla ==="
$PY scripts/analysis/auroc_anomaly_detection.py \
    --reference_week 44 \
    --inference_dir results/inference_auditfix/WEEK-2022-44_vanilla_bs64 \
    --output_dir results/quic_monitor \
    --f1_threshold 0.66

echo "=== DONE quic_bbse_monitor ==="
