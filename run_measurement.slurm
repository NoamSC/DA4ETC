#! /bin/bash

#SBATCH --job-name=train_allot_array
#SBATCH --output=/home/anatbr/students/noamshakedc/da4etc/logs/bsearch_%a.out
#SBATCH --error=/home/anatbr/students/noamshakedc/da4etc/logs/bsearch_%a.err
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=16:30:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-22  # Replace XX with actual number of test groups minus 1

python simple_model_train.py --test_index=${SLURM_ARRAY_TASK_ID} --use_cheat_to_match_target_label_distribution