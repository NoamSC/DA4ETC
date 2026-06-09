#!/bin/bash
#SBATCH --job-name=quic_bnstat
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=04:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

# DIAGNOSTIC #1: BN test-batch statistics, ZERO adaptation. Isolates how much of the
# TENT/CoTTA in-distribution degradation comes purely from swapping the trained BN
# running-stats for per-batch stats (vs the adaptation updates). Same model / seed /
# data_sample_frac / batch_size (64) as the original TENT/CoTTA runs so it's directly
# comparable. If week-44 acc here already ~ TENT/CoTTA, the BN swap is the cause.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared
OUTDIR=results/inference/quic_w44_inference_bnstats

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_quic_bnstats_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_quic_bnstats_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== bnstats (BN batch-stats, no adaptation): week-44 model -> weeks 44-47 ==="
$PYTHON scripts/inference/run_inference.py \
    --method bnstats \
    --experiment_dir exps/cesnet_quic22_multimodal_v01 \
    --train_week WEEK-2022-44 \
    --dataset_root "$ROOT" \
    --output_dir "$OUTDIR" \
    --batch_size 64 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device cuda:0
STATUS=$?
echo "=== bnstats exit ${STATUS} -> ${OUTDIR}/WEEK-2022-{44,45,46,47}.npz ==="

if [ $STATUS -eq 0 ]; then
  echo "=== bnstats per-week accuracy ==="
  $PYTHON - "$OUTDIR" <<'PYEOF'
import numpy as np, glob, os, re, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.npz'))):
    z = np.load(f, allow_pickle=True)
    if 'true_labels' not in z.files: continue
    t, p = z['true_labels'], z['pred_labels']
    w = re.search(r'WEEK-2022-(\d+)', f).group(1)
    print(f"week {w}: acc={(t==p).mean():.3f}  n={t.size}")
PYEOF
fi
exit $STATUS
