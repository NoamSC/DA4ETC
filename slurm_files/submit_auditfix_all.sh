#!/bin/bash
# Submit the full post-audit re-run: DANN forward-transfer diagonal + the TTA grid
# (vanilla/AdaBN/TENT/CoTTA x bs64/bs256) for TLS week_1, week_16 and QUIC week_44.
# All TTA outputs go to results/inference_auditfix/ (new tree — overrides nothing).
set -euo pipefail
cd /home/anatbr/students/noamshakedc/da4etc

TLS_EXP=exps/cesnet_multimodal_each_week_train_v01
TLS_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2
QUIC_EXP=exps/cesnet_quic22_multimodal_v01
QUIC_ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared

submit() {  # method train_week batch exp root
  jid=$(sbatch --parsable \
    --export=ALL,METHOD="$1",TRAIN_WEEK="$2",BATCH="$3",EXP_DIR="$4",DATA_ROOT="$5" \
    slurm_files/run_inference_auditfix.slurm)
  echo "  ${jid}  ${1} ${2} bs${3}"
}

echo "### DANN forward-transfer diagonal (source week16 -> each target week) ###"
dann=$(sbatch --parsable slurm_files/run_dann_fwd_w16.slurm)
echo "  ${dann}  dann_fwd_w16 (array 0-52)"

echo "### TLS TTA grid ###"
for W in week_1 week_16; do
  submit vanilla "$W" 64  "$TLS_EXP" "$TLS_ROOT"
  submit bnstats "$W" 64  "$TLS_EXP" "$TLS_ROOT"
  submit tent    "$W" 64  "$TLS_EXP" "$TLS_ROOT"
  submit tent    "$W" 256 "$TLS_EXP" "$TLS_ROOT"
  submit cotta   "$W" 64  "$TLS_EXP" "$TLS_ROOT"
  submit cotta   "$W" 256 "$TLS_EXP" "$TLS_ROOT"
done

echo "### QUIC TTA grid (week44 source) ###"
for W in WEEK-2022-44; do
  submit vanilla "$W" 64  "$QUIC_EXP" "$QUIC_ROOT"
  submit bnstats "$W" 64  "$QUIC_EXP" "$QUIC_ROOT"
  submit tent    "$W" 64  "$QUIC_EXP" "$QUIC_ROOT"
  submit tent    "$W" 256 "$QUIC_EXP" "$QUIC_ROOT"
  submit cotta   "$W" 64  "$QUIC_EXP" "$QUIC_ROOT"
  submit cotta   "$W" 256 "$QUIC_EXP" "$QUIC_ROOT"
done
echo "### done submitting ###"
