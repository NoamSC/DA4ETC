#!/bin/bash
#SBATCH --job-name=coral_lcalib
#SBATCH --output=logs/coral_lcalib_%j.out
#SBATCH --error=logs/coral_lcalib_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=n-301
# Real-path lambda calibration for CORAL: train week16->week52 (far, max gap) for a few
# epochs at LAMBDA_CORAL=$LAM; inspect coral-loss decrease vs val-accuracy to choose lambda.
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
if ! $PY -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" 2>/dev/null; then
  echo "GPU dead on $(hostname); requeue"; scontrol requeue $SLURM_JOB_ID; sleep 30; exit 1; fi
echo "=== CORAL lambda-calib: w16->w52, lambda_coral=${LAM} ==="
$PY scripts/train/train_per_week_cesnet.py --multimodal \
    --week 16 --val_week 52 \
    --exp_name "cesnet_tls_coral_lcalib_v01/lam${LAM}/{}" \
    --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22_v2 \
    --lambda_coral ${LAM} \
    --train_data_frac 1.0 --val_data_frac 0.1 --train_per_epoch_data_frac 0.1 \
    --num_epochs 6 --batch_size 64 --num_workers 0 --learning_rate 3e-3
echo "=== done lambda=${LAM} ==="
