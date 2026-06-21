#!/bin/bash
#SBATCH --job-name=anomaly_mm
#SBATCH --output=logs/slurm-%A.out.tmp
#SBATCH --error=logs/slurm-%A.err.tmp
#SBATCH --time=4:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Controlled re-run of the viral anomaly-detection experiment on the MULTIMODAL
# backbone (results/inference_auditfix/*), the same 10%-sample set used by the
# negative-transfer tables and the stats-rigor table -- fixing the reviewer-flagged
# inconsistency that the published figures were on the FlowPic CNN backbone and
# (for Week-1) reference_week=0 (the biased easy week). This run is reference_week=1
# (Week-1 source, correct) and Week-16, both multimodal. Outputs -> results/anomaly_mm/.
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
mkdir -p results/anomaly_mm

echo "=== Week-1 source, reference_week=1, MULTIMODAL ==="
$PY scripts/analysis/auroc_anomaly_detection.py \
    --reference_week 1 \
    --inference_dir results/inference_auditfix/week_1_vanilla_bs64 \
    --output_dir results/anomaly_mm

echo "=== Week-16 forward replication, MULTIMODAL ==="
$PY scripts/analysis/auroc_anomaly_week16.py \
    --inference_dir results/inference_auditfix/week_16_vanilla_bs64 \
    --output_dir results/anomaly_mm --tag ref16_mm

echo "=== DONE anomaly_mm ==="
