#! /bin/sh

#SBATCH --job-name=bsearch_v1
#SBATCH --output=/home/anatbr/students/noamshakedc/da4etc/bsearch_v1.out  # Redirect stdout
#SBATCH --error=/home/anatbr/students/noamshakedc/da4etc/bsearch_v1.err   # Redirect stderr
#SBATCH --partition=gpu-h100-killable   # Ensure this is a valid partition
#SBATCH --time=16:30:00  # Reduced time limit to match cluster constraints
#SBATCH --signal=USR1@120  # Graceful exit signal
#SBATCH --nodes=1  # Number of machines
#SBATCH --ntasks=1  # Single process
#SBATCH --mem=8G  # Reduce memory request (32G may be too much)
#SBATCH --cpus-per-task=1  # CPU cores per process
#SBATCH --gres=gpu:1

# Define job name variable

# Run Python script
python bayesian_opt_search.py
