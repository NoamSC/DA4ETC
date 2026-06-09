#!/bin/bash
# One-command reproduction of the full CESNET-QUIC22 week-44 experiment.
#
# Submits the entire SLURM DAG (prep -> norm-stats -> train -> {vanilla, TENT, CoTTA,
# bnstats, TTA@bs256} inference -> summary) with dependencies, so each stage starts
# only when its inputs are ready. This script just calls sbatch (fast) — the actual
# work runs on the cluster. See QUIC_W44_RESULTS.md.
#
# Usage:  bash slurm_files/submit_quic_pipeline.sh
# Watch:  squeue -u $USER | grep -i quic
# Result: results/inference/quic_w44_summary/  (printed table in the summary job .out)
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

echo "Submitting CESNET-QUIC22 week-44 pipeline ..."

PREP=$(sbatch --parsable slurm_files/run_quic_prepare.slurm)
echo "  prep (weeks 44-47)        : $PREP"

NORM=$(sbatch --parsable --dependency=afterok:$PREP slurm_files/run_quic_norm_stats.slurm)
echo "  normalization stats       : $NORM  (afterok:$PREP)"

TRAIN=$(sbatch --parsable --dependency=afterok:$NORM slurm_files/run_quic_multimodal_train.slurm)
echo "  train multimodal (wk44)   : $TRAIN  (afterok:$NORM)"

VAN=$(sbatch --parsable --dependency=afterok:$TRAIN slurm_files/run_quic_inference.slurm)
echo "  inference: vanilla        : $VAN  (afterok:$TRAIN)"

TTA=$(sbatch --parsable --dependency=afterok:$TRAIN slurm_files/run_quic_tta_inference.slurm)
echo "  inference: TENT + CoTTA   : $TTA  (afterok:$TRAIN)"

BN=$(sbatch --parsable --dependency=afterok:$TRAIN slurm_files/run_quic_bnstats.slurm)
echo "  inference: bnstats (AdaBN): $BN  (afterok:$TRAIN)"

BS=$(sbatch --parsable --dependency=afterok:$TRAIN slurm_files/run_quic_tta_bs256.slurm)
echo "  inference: TTA @ bs256    : $BS  (afterok:$TRAIN)"

SUM=$(sbatch --parsable --dependency=afterany:$VAN:$TTA:$BN:$BS slurm_files/run_quic_tta_summary.slurm)
echo "  summary (6-method table)  : $SUM  (afterany:$VAN:$TTA:$BN:$BS)"

echo
echo "All jobs submitted. Job IDs:"
printf "%s\n" "$PREP" "$NORM" "$TRAIN" "$VAN" "$TTA" "$BN" "$BS" "$SUM" | tee logs/quic_chain_jobids.txt
echo
echo "Watch:  squeue -u \$USER | grep -i quic"
echo "Result: tail -60 \$(ls -t logs/*quic_summary*.out | head -1)"
