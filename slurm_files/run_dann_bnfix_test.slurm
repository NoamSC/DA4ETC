#!/bin/bash
#SBATCH --job-name=dann_bnfix
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=2:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=44,52
# QUICK TEST of the BatchNorm combined-forward fix: source wk16 -> target wkN, 6 epochs,
# fresh (--override). If domain_classifier accuracy now rises above 50%, the fix works.
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
DATASET_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2
TGT=$SLURM_ARRAY_TASK_ID; TGT2=$(printf "%02d" "$TGT")
TS=$(date +%Y%m%d_%H%M%S)
OUT="logs/${TS}_dann_bnfix_w16_to_w${TGT2}_${SLURM_ARRAY_JOB_ID}.out"; ERR="${OUT%.out}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUT" 2>/dev/null||true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERR" 2>/dev/null||true
exec 1>>"$OUT"; exec 2>>"$ERR"
echo "=== BNFIX TEST: wk16 -> wk${TGT2} ==="
$PYTHON scripts/train/train_per_week_cesnet.py --multimodal --week 16 --val_week "$TGT" \
  --exp_name "cesnet_tls_dann_fwd_w16_bnfix_test/{}" --dataset_root "$DATASET_ROOT" \
  --lambda_dann 1.0 --lambda_rgl 0.1 --lambda_grl_gamma 10 \
  --train_data_frac 1.0 --val_data_frac 0.1 --train_per_epoch_data_frac 0.1 \
  --num_epochs 6 --batch_size 64 --num_workers 0 --learning_rate 3e-3 --override
echo "=== exit $? ==="
