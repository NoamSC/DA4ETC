#!/bin/bash
# Generate all drift figures for one inference, restricted to a forward week window.
#
# Default: the week-16-trained model, from week 16 into the future (weeks 16-52).
# Outputs land in figs/ with a w16_16to52 tag in the filename.
#
# Args (positional): DIR  MINWEEK  FOCAL  TOP  BG
#   BG = background samples per other class, pooled over weeks (denser grey map)
#
# Usage:
#   ./run_drift_figs.sh                                            # week_16_inference, weeks>=16
#   ./run_drift_figs.sh results/inference/week_16_inference 16     # explicit dir + min week
#   ./run_drift_figs.sh results/inference/week_9_inference  9 151  # different dir/min-week/focal
#   ./run_drift_figs.sh results/inference/week_16_inference 16 57 10 400  # denser background
#
# Assumes the ml2 conda env (set PYTHON to override the interpreter).
set -euo pipefail

DIR=${1:-results/inference/week_16_inference}
MINWEEK=${2:-16}
FOCAL=${3:-57}
TOP=${4:-10}
BG=${5:-200}                          # background samples per other class (pooled over weeks)
PY=${PYTHON:-/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python}

cd "$(dirname "$0")/../.."          # repo root (paths below are repo-root-relative)
V=scripts/viz

echo "=== inference dir : $DIR  (weeks >= $MINWEEK) ==="
echo "=== focal class   : $FOCAL ==="

echo; echo "[1/4] per-app density ridgelines (top 6) ..."
$PY $V/plot_density_drift.py        --inference_dir "$DIR" --min_week "$MINWEEK" --top 6

echo; echo "[2/4] multi-app concentration overlay (top $TOP) ..."
$PY $V/plot_density_drift_multi.py  --inference_dir "$DIR" --min_week "$MINWEEK" --top "$TOP"

echo; echo "[3/4] PCA temporal map (focal $FOCAL, bg/class $BG) ..."
$PY $V/plot_pca_temporal.py         --inference_dir "$DIR" --min_week "$MINWEEK" --focal_class "$FOCAL" --max_bg_per_class "$BG"

echo; echo "[4/4] t-SNE temporal map (focal $FOCAL, bg/class $BG) ..."
$PY $V/plot_tsne_temporal_static.py --inference_dir "$DIR" --min_week "$MINWEEK" --focal_class "$FOCAL" --max_bg_per_class "$BG"

echo; echo "=== done. figures in figs/ tagged w${MINWEEK}_* ==="
