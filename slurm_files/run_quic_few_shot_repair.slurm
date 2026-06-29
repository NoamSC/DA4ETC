#!/bin/bash
#SBATCH --job-name=quic_repair
#SBATCH --output=logs/slurm-%A.out.tmp
#SBATCH --error=logs/slurm-%A.err.tmp
#SBATCH --time=2:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#
# Few-shot per-class PROTOTYPE / k-NN REPAIR on CESNET-QUIC22 (week-44 source, 102
# classes, eval weeks 45-47) -- the QUIC replica of the TLS Week-16 repair experiment
# (scripts/analysis/few_shot_repair_loop.py + few_shot_repair_knn.py).
#
# Frozen QUIC week-44 source; forward-only eval; reuses the saved 10%-subsample
# embeddings under results/inference_auditfix/WEEK-2022-44_vanilla_bs64/ (NO recompute,
# no GPU). The repair recovers per-class recall on a flagged (teleported) class from
# K in {1,5,10,50} post-drift labels, and measures stable-class poaching / Macro-F1.
#
# CPU-only (numpy). The core scripts are unmodified-logic: dataset-specific paths,
# source week, output dirs and the target-class set are injected via env vars
# (defaults preserve the original TLS Week-16 behaviour). Target classes are the
# four largest frozen-source recall cliffs (week-44 ref -> week-47) plus one stable
# control:
#   62 google-www  (0.92->0.14)   19 dns-doh       (0.99->0.23)
#   49 google-fonts (0.92->0.17)  50 google-gstatic(0.89->0.26)   [all teleported]
#   17 discord      (0.95, flat)  [non-teleported control]
#
# Outputs:
#   results/repair/few_shot_repair_quic_v01/metrics.json   (NCM recovery vs stability)
#   results/repair/few_shot_knn_quic_v01/metrics.json      (k-NN/Parzen recovery vs poaching)
#   figs/repair/quic/*.png

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Dataset-specific config for the (unmodified-logic) repair harness.
export REPAIR_INF_DIR=results/inference_auditfix/WEEK-2022-44_vanilla_bs64
export REPAIR_LABEL_MAP=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared/label_mapping.json
export REPAIR_SOURCE_WEEK=44
export REPAIR_TARGETS_JSON=scripts/analysis/quic_repair_targets.json
export REPAIR_OUT_RES=results/repair/few_shot_repair_quic_v01
export REPAIR_KNN_OUT_RES=results/repair/few_shot_knn_quic_v01
export REPAIR_OUT_FIG=figs/repair/quic
export REPAIR_KNN_OUT_FIG=figs/repair/quic

echo "=== QUIC few-shot NCM prototype repair (recovery vs stability) ==="
$PY scripts/analysis/few_shot_repair_loop.py
echo "=== NCM repair exit $? ==="

echo "=== QUIC few-shot k-NN / Parzen repair (recovery vs poaching) ==="
$PY scripts/analysis/few_shot_repair_knn.py
echo "=== k-NN repair exit $? ==="

echo "=== DONE quic_repair ==="
