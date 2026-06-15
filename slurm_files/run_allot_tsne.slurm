#!/bin/bash
#SBATCH --job-name=allot_tsne
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=08:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Unbuffered stdout/stderr so openTSNE's per-iteration KL trace + our progress
# prints land in the log live (block-buffering hid them last run).
export PYTHONUNBUFFERED=1

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="allot_tsne_teleportation"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- Fig-1-style t-SNE teleportation panels on Allot, both training slices.
#     FULL-data unbiased fit (all ~292k embeddings, openTSNE, 6000 iters — the
#     i6000 cache name differs from i3000/i1500, so this refits cleanly and
#     leaves the shorter-run caches intact for side-by-side comparison).
#     top 12 focal apps; grey background thinned to 30% (render-only). Final KL
#     divergence is printed to the .out log so convergence is checkable. ---
$PYTHON scripts/viz/plot_tsne_temporal_allot.py \
    --inference_dir exps/allot_multimodal/early_eq/inference \
    --top 12 --n_iter 6000 --bg_frac 0.3

$PYTHON scripts/viz/plot_tsne_temporal_allot.py \
    --inference_dir exps/allot_multimodal/quarter_eq/inference \
    --top 12 --n_iter 6000 --bg_frac 0.3

echo "Completed allot_tsne"
