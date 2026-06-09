#!/bin/bash
#SBATCH --job-name=quic_tta256
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=08:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1

# DIAGNOSTIC #2: TENT / CoTTA at --batch_size 256 (matches the training batch size),
# vs the original runs at batch_size 64. Removes the small-batch confound: BN test-batch
# statistics are far less noisy at 256, so this should recover part of the in-distribution
# gap if small batches were the issue. Array 0 -> TENT, 1 -> CoTTA.

cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared

METHODS=(tent cotta)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
OUTDIR=results/inference/quic_w44_inference_${METHOD}_bs256

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_quic_${METHOD}_bs256_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRFILE="logs/${TIMESTAMP}_quic_${METHOD}_bs256_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== ${METHOD} @ batch_size 256: week-44 model -> weeks 44-47 ==="
$PYTHON scripts/inference/run_inference.py \
    --method "$METHOD" \
    --experiment_dir exps/cesnet_quic22_multimodal_v01 \
    --train_week WEEK-2022-44 \
    --dataset_root "$ROOT" \
    --output_dir "$OUTDIR" \
    --batch_size 256 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --device cuda:0
STATUS=$?
echo "=== ${METHOD} bs256 exit ${STATUS} -> ${OUTDIR}/WEEK-2022-{44,45,46,47}.npz ==="

if [ $STATUS -eq 0 ]; then
  echo "=== ${METHOD} bs256 per-week accuracy ==="
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
