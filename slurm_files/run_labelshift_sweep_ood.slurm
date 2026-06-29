#!/bin/bash
#SBATCH --job-name=ls_sweep_ood
#SBATCH --output=logs/slurm-%j.out.tmp
#SBATCH --error=logs/slurm-%j.err.tmp
#SBATCH --time=02:00:00
#SBATCH --partition=killable
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8

# Change to project directory
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs

# Conda env `ml2` python (no `conda activate`)
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

# Timestamped, chronologically-sortable log files
EXP_NAME_FULL="labelshift_sweep_fwd_ood"
EXPNAME=$(echo "$EXP_NAME_FULL" | sed 's/\//_/g' | sed 's/[{}]//g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_JOB_ID}.err"
mv "logs/slurm-${SLURM_JOB_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_JOB_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

# --- job command ---
# Adds energy/MSP per-class OOD detectors to the label-shift severity sweep,
# scored per-flow under the same resampling (reusing mfwdd's eix; no new RNG).
# clean_weeks 17-20 + n_trials 14 replicate the existing _fwd run so ours/naive
# reproduce. Severity grid default includes 20x.
$PYTHON scripts/analysis/labelshift_severity_sweep.py \
    --inference_dir results/inference_auditfix/week_16_vanilla_bs64 \
    --reference_week 16 --output_dir figs/isolation_w16 \
    --checkpoint exps/cesnet_multimodal_each_week_train_v01/week_16/weights/best_model.pth \
    --clean_weeks 17 20 --n_trials 14 \
    --suffix _fwd_ood --recompute --seed 42

echo "Completed labelshift_sweep_fwd_ood"
