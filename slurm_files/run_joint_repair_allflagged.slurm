#!/bin/bash
#SBATCH --job-name=joint_repair_allflagged
#SBATCH --output=logs/slurm-%A.out.tmp
#SBATCH --error=logs/slurm-%A.err.tmp
#SBATCH --time=4:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ALL-FLAGGED joint few-shot repair -> GLOBAL macro-F1 over the forward year.
# Honest full diagnose-and-repair loop: on each forward week (17..52, frozen
# Week-16 source, 10% embeddings) repair EVERY class the label-free monitor
# flags (R_corr>0.21595 & present), INCLUDING false alarms, that has post-drift
# support (>=40 shots); recompute global macro-F1 (180) before vs after + stable
# (never-flagged) poaching. K=50, 5 seeds. Outputs ->
# results/repair/joint_repair_allflagged_v01/metrics.json
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
mkdir -p results/repair/joint_repair_allflagged_v01

echo "=== ALL-FLAGGED joint repair, forward year (Week-16 source, multimodal) ==="
$PY scripts/analysis/joint_repair_allflagged.py

echo "=== DONE joint_repair_allflagged ==="
