#!/bin/bash
#SBATCH --job-name=mmd_lcalib
#SBATCH --output=logs/mmd_lcalib_%j.out
#SBATCH --error=logs/mmd_lcalib_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=n-301
# Real-path lambda calibration for MMD: train week16->week52 (far, max gap) for a few
# epochs at LAMBDA_MMD=$LAM; inspect mmd-loss decrease vs val-accuracy to choose lambda.
# NOTE: compute_mmd_loss z-normalizes features per-dim before the RBF kernel, so the raw
# MMD loss is O(1) and LAMBDA_MMD must be SMALL (unlike CORAL's tiny-loss / lambda=100).
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
if ! $PY -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" 2>/dev/null; then
  echo "GPU dead on $(hostname); requeue"; scontrol requeue $SLURM_JOB_ID; sleep 30; exit 1; fi
echo "=== MMD lambda-calib: w16->w52, lambda_mmd=${LAM} ==="
$PY scripts/train/train_per_week_cesnet.py --multimodal \
    --week 16 --val_week 52 \
    --exp_name "cesnet_tls_mmd_lcalib_v01/lam${LAM}/{}" \
    --dataset_root /home/anatbr/dataset/CESNET-TLS-Year22_v2 \
    --lambda_mmd ${LAM} \
    --train_data_frac 1.0 --val_data_frac 0.1 --train_per_epoch_data_frac 0.1 \
    --num_epochs 6 --batch_size 64 --num_workers 0 --learning_rate 3e-3
echo "=== done lambda=${LAM} ==="
