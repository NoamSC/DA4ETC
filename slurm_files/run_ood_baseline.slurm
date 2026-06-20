#!/bin/bash
#SBATCH --job-name=ood_baseline
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=03:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

EXPNAME="ood_baseline"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# Per-class OOD (energy / MSP) baseline vs BBSE-corrected residual.
# Week-16 source, forward eval weeks 17-52. Clean detection AUROC/FAR is
# seed-independent; the label-shift robustness loop is seeded, so we run all
# 5 seeds (1,2,3,4,42). Seed 42 builds the per-(class,week) score cache; the
# rest reuse it (only the cheap robustness loop is re-run per seed).

echo "===== SEED 42 (builds cache, primary) ====="
$PYTHON scripts/analysis/ood_baseline_detector.py \
    --inference_dir results/inference_auditfix/week_16_vanilla_bs64 \
    --checkpoint exps/cesnet_multimodal_each_week_train_v01/week_16/weights/best_model.pth \
    --reference_week 16 --forward \
    --seed 42 --suffix _w16fwd_seed42 --recompute

CACHE=results/ood_baseline/ood_scores_cache_w16fwd_seed42.npz
for S in 1 2 3 4; do
    echo "===== SEED $S (reuse seed-42 cache) ====="
    # share the seed-independent per-(class,week) score cache to avoid recompute
    cp -f "$CACHE" results/ood_baseline/ood_scores_cache_w16fwd_seed${S}.npz
    $PYTHON scripts/analysis/ood_baseline_detector.py \
        --inference_dir results/inference_auditfix/week_16_vanilla_bs64 \
        --checkpoint exps/cesnet_multimodal_each_week_train_v01/week_16/weights/best_model.pth \
        --reference_week 16 --forward \
        --seed $S --suffix _w16fwd_seed${S}
done

echo "===== DONE all seeds ====="
