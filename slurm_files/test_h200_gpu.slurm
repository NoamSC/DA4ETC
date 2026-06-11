#!/bin/bash
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --job-name=test_h200_gpu
#SBATCH --output=logs/test_h200_gpu_%j.log

PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

cd /home/anatbr/students/noamshakedc/da4etc

echo "=== Testing H200 GPU access ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Check GPU availability
echo "=== nvidia-smi output ==="
nvidia-smi

echo ""
echo "=== Python GPU test ==="
$PY -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    # Quick compute test
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f'Matrix multiply on GPU successful, result shape: {y.shape}')
"

echo ""
echo "=== Test complete ==="