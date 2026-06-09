#!/bin/bash
#SBATCH --job-name=quic_infer
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

# Vanilla inference of the frozen week-44 multimodal model across ALL prepared
# CESNET-QUIC22 weeks (44 in-distribution, 45/46/47 drifted) for temporal-
# generalization analysis. Writes one .npz per week with true/pred labels, softmax,
# and subsampled embeddings. See QUIC_W44_MULTIMODAL_HANDOFF.md.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared
OUTDIR=results/inference/quic_w44_inference

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_quic_infer_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_quic_infer_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== Vanilla inference: week-44 model -> weeks 44-47 ==="
$PYTHON scripts/inference/run_inference.py \
    --method vanilla \
    --experiment_dir exps/cesnet_quic22_multimodal_v01 \
    --train_week WEEK-2022-44 \
    --dataset_root "$ROOT" \
    --output_dir "$OUTDIR" \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device cuda:0
STATUS=$?
echo "=== inference exit ${STATUS} -> ${OUTDIR}/WEEK-2022-{44,45,46,47}.npz ==="

if [ $STATUS -eq 0 ]; then
  echo "=== per-week accuracy ==="
  $PYTHON - "$OUTDIR" <<'PYEOF'
import numpy as np, glob, os, re, sys
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
    z = np.load(f, allow_pickle=True)
    if 'true_labels' not in z.files:
        continue
    t, p = z['true_labels'], z['pred_labels']
    w = re.search(r'WEEK-2022-(\d+)', f).group(1)
    print(f"week {w}: acc={(t==p).mean():.3f}  n={t.size}")
PYEOF
fi
exit $STATUS
