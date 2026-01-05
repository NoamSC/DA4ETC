#! /bin/bash

#SBATCH --job-name=mirage_eda
#SBATCH --output=/home/anatbr/students/noamshakedc/da4etc/logs/mirage_eda_%j.out
#SBATCH --error=/home/anatbr/students/noamshakedc/da4etc/logs/mirage_eda_%j.err
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=16:30:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

python mirage_eda_complete.py
