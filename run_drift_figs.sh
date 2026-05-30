#!/bin/bash
# Generate all drift figures for one inference, restricted to a forward week window.
#
# Default: the week-16-trained model, from week 16 into the future (weeks 16-52).
# Outputs land in figs/ with a w16_16to52 tag in the filename.
#
# Usage:
#   ./run_drift_figs.sh                                  # week_16_inference, weeks>=16
#   ./run_drift_figs.sh figs/week_16_inference 16        # explicit dir + min week
#   ./run_drift_figs.sh figs/week_9_inference  9 151     # different dir/min-week/focal
#
# Assumes the ml2 conda env (set PYTHON to override the interpreter).
set -euo pipefail

DIR=${1:-figs/week_16_inference}
MINWEEK=${2:-16}
FOCAL=${3:-49}
TOP=${4:-10}
PY=${PYTHON:-/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python}

cd "$(dirname "$0")"

echo "=== inference dir : $DIR  (weeks >= $MINWEEK) ==="
echo "=== focal class   : $FOCAL ==="

echo; echo "[1/4] per-app density ridgelines (top 6) ..."
$PY plot_density_drift.py        --inference_dir "$DIR" --min_week "$MINWEEK" --top 6

echo; echo "[2/4] multi-app concentration overlay (top $TOP) ..."
$PY plot_density_drift_multi.py  --inference_dir "$DIR" --min_week "$MINWEEK" --top "$TOP"

echo; echo "[3/4] PCA temporal map (focal $FOCAL) ..."
$PY plot_pca_temporal.py         --inference_dir "$DIR" --min_week "$MINWEEK" --focal_class "$FOCAL"

echo; echo "[4/4] t-SNE temporal map (focal $FOCAL) ..."
$PY plot_tsne_temporal_static.py --inference_dir "$DIR" --min_week "$MINWEEK" --focal_class "$FOCAL"

echo; echo "=== done. figures in figs/ tagged w${MINWEEK}_* ==="
