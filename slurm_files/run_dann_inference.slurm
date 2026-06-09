#!/bin/bash
#SBATCH --job-name=dann_infer
#SBATCH --output=logs/slurm-%A_%a.out.tmp
#SBATCH --error=logs/slurm-%A_%a.err.tmp
#SBATCH --time=4:00:00
#SBATCH --partition=killable
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

# Forward-eval (frozen, classification head only) of the DANN SOURCE-week model across
# all 53 CESNET-TLS-Year22 weeks -> the DANN row of the SOTA benchmark, directly
# comparable to week_1_inference{,_bnstats,_tent_fixed,_cotta_fixed}. The DANN domain
# head is ignored at inference; --method vanilla runs the class head in eval().
# Submit with --dependency=afterok:<dann_train_jobid>_<SOURCE_WEEK> so the source model
# exists first. SOURCE_WEEK defaults to 1 (WEEK-2022-01); override via --export if the
# vanilla baseline's "week_1" maps to a different calendar week.

# SOURCE_WEEK = WEEK-2022-01: confirmed from run_cesnet_multimodal_array.slurm, which
# trained the vanilla benchmark with WEEK=ARRAY_TASK_ID, --week ${WEEK}, exp=week_${WEEK}.
# So the "week_1" baseline model = --week 1 = WEEK-2022-01. (Every model peaks on test
# week 00 because week 00 is an easy week, NOT because it is the train week.)
NUM_JOBS=4
SOURCE_WEEK="${SOURCE_WEEK:-WEEK-2022-01}"
OUTPUT_DIR="${OUTPUT_DIR:-results/inference/dann_w01src_inference}"
cd /home/anatbr/students/noamshakedc/da4etc
mkdir -p logs
PYTHON=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python

EXPNAME="dann_infer_${SLURM_ARRAY_TASK_ID}of${NUM_JOBS}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}.out"
ERRFILE="logs/${TIMESTAMP}_${EXPNAME}_${SLURM_ARRAY_JOB_ID}.err"
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out.tmp" "$OUTFILE" 2>/dev/null || true
mv "logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err.tmp" "$ERRFILE" 2>/dev/null || true
exec 1>>"$OUTFILE"
exec 2>>"$ERRFILE"

echo "=== DANN forward-eval: source ${SOURCE_WEEK} model -> all 53 weeks ==="
$PYTHON scripts/inference/run_inference.py \
    --method vanilla \
    --experiment_dir 'exps/cesnet_tls_dann_grl01_v01' \
    --dataset_root '/home/anatbr/dataset/CESNET-TLS-Year22_v2' \
    --train_week "$SOURCE_WEEK" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 64 \
    --num_workers 0 \
    --data_sample_frac 0.1 \
    --seed 42 \
    --embed_seed 0 \
    --device 'cuda:0' \
    --num_jobs $NUM_JOBS \
    --job_id $SLURM_ARRAY_TASK_ID

echo "Completed DANN forward-eval (task ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS})"
