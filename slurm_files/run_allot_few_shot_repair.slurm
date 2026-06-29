#!/bin/bash
#SBATCH --job-name=allot_repair
#SBATCH --output=logs/slurm-%A.out.tmp
#SBATCH --error=logs/slurm-%A.err.tmp
#SBATCH --time=1:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Few-shot per-class REPAIR on the third, proprietary closed-world dataset
# (anonymized for review) -- multimodal backbone, `early` source slice.
# Allot analogue of the TLS Week-16 few-shot repair: per-class recall recovery
# from K in {1,5,10,50} labels on a flagged/degraded class, with stable-class
# poaching / Macro-F1 measured (negative-transfer guard). Reuses the TLS
# harnesses verbatim via the wrapper few_shot_repair_allot.py.
# Outputs -> results/repair/few_shot_repair_allot_v01/.
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
mkdir -p results/repair/few_shot_repair_allot_v01 figs/repair

echo "=== few-shot repair (early slice): k-NN/Parzen + NCM loop ==="
$PY scripts/analysis/few_shot_repair_allot.py --slice early_eq

echo "=== DONE allot_few_shot_repair ==="
