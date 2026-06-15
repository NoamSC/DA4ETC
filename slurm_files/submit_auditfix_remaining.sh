#!/bin/bash
# Idempotent submitter for the auditfix groups that didn't fit under the QOS submit
# cap on the first pass. Re-runnable: records submitted groups in a state file and
# skips them next time; defers any that still hit the QOS limit. Prints REMAINING=N.
cd /home/anatbr/students/noamshakedc/da4etc
STATE="${CLAUDE_JOB_DIR:-/tmp}/tmp/auditfix_remaining_done.txt"
mkdir -p "$(dirname "$STATE")"; touch "$STATE"

TLS_EXP=exps/cesnet_multimodal_each_week_train_v01
TLS_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2
QUIC_EXP=exps/cesnet_quic22_multimodal_v01
QUIC_ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared

remaining=0
submit() {  # method week batch exp root
  local key="${1}_${2}_bs${3}"
  if grep -qxF "$key" "$STATE"; then return 0; fi
  local jid
  if jid=$(sbatch --parsable \
        --export=ALL,METHOD="$1",TRAIN_WEEK="$2",BATCH="$3",EXP_DIR="$4",DATA_ROOT="$5" \
        slurm_files/run_inference_auditfix.slurm 2>/dev/null); then
    echo "submitted ${key} -> ${jid}"
    echo "$key" >> "$STATE"
  else
    echo "deferred  ${key} (QOS cap)"
    remaining=$((remaining + 1))
  fi
}

submit cotta   week_16      256 "$TLS_EXP"  "$TLS_ROOT"
submit vanilla WEEK-2022-44 64  "$QUIC_EXP" "$QUIC_ROOT"
submit bnstats WEEK-2022-44 64  "$QUIC_EXP" "$QUIC_ROOT"
submit tent    WEEK-2022-44 64  "$QUIC_EXP" "$QUIC_ROOT"
submit tent    WEEK-2022-44 256 "$QUIC_EXP" "$QUIC_ROOT"
submit cotta   WEEK-2022-44 64  "$QUIC_EXP" "$QUIC_ROOT"
submit cotta   WEEK-2022-44 256 "$QUIC_EXP" "$QUIC_ROOT"

echo "REMAINING=${remaining}"
