#! /bin/sh

#SBATCH --job-name=simple_train
#SBATCH --output=/home/anatbr/students/noamshakedc/da4etc/test_slurm.out # redirect stdout
#SBATCH --error=/home/anatbr/students/noamshakedc/da4etc/test_slurm.err # redirect stderr
#SBATCH --partition=gpu-a100-killable # (see resources section)
#SBATCH --time=200 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=500000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

python train_model.py