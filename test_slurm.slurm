#! /bin/sh
#SBATCH --job-name=simple_train
#SBATCH --output=/home/anatbr/students/noamshakedc/da4etc/test_slurm.out  # Redirect stdout
#SBATCH --error=/home/anatbr/students/noamshakedc/da4etc/test_slurm.err   # Redirect stderr
#SBATCH --partition=killable   # Ensure this is a valid partition
#SBATCH --time=16:30:00  # Reduced time limit to match cluster constraints
#SBATCH --signal=USR1@120  # Graceful exit signal
#SBATCH --nodes=1  # Number of machines
#SBATCH --ntasks=1  # Single process
#SBATCH --mem=8G  # Reduce memory request (32G may be too much)
#SBATCH --cpus-per-task=1  # CPU cores per process
#SBATCH --gres=gpu:1

# Print debugging information
echo "SLURM JOB STARTED"
echo "Running on node: $(hostname)"
echo "CUDA Devices Available:"
nvidia-smi  # Check GPU status
echo "Environment Variables:"
env  # Print environment variables

# Load modules if required (uncomment if your cluster uses module loading)
# module load python/3.x cuda/11.x  # Adjust versions as needed

# Activate virtual environment if applicable
# source /path/to/venv/bin/activate

# Ensure correct Python version
python --version
which python  # Check which Python is being used

# Run Python script
python /home/anatbr/students/noamshakedc/da4etc/train_model_on_different_locations.py

echo "SLURM JOB COMPLETED"
