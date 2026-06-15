#!/bin/bash
# Queue-aware, preemption-safe submitter for the multi-seed significance study.
# 5 seeds x {vanilla,bnstats,tent} x {TLS wk16, QUIC wk44} at bs64, --acc_only.
# Skips groups already complete (.npz count) or already queued (by job name);
# resubmits preempted ones. Prints INCOMPLETE=N. Safe to call in a drain loop.
cd /home/anatbr/students/noamshakedc/da4etc

# Disk safety backstop vs the user's 500GB netapp cap (NOT the mount free space).
# (seedvar is acc_only/tiny, but guard anyway.) Static siblings added as baseline.
DA4_MB=$(du -sm /home/anatbr/students/noamshakedc/da4etc 2>/dev/null | awk '{print $1}')
USED_MB=$(( ${DA4_MB:-0} + 69000 ))
if [ "$USED_MB" -gt $((470*1024)) ]; then
  echo "DISK_GUARD: tree ~$((USED_MB/1024))GB (>470GB of 500GB cap) -- NOT submitting"
  echo "INCOMPLETE=999"; exit 0
fi

TLS_EXP=exps/cesnet_multimodal_each_week_train_v01
TLS_ROOT=/home/anatbr/dataset/CESNET-TLS-Year22_v2
QUIC_EXP=exps/cesnet_quic22_multimodal_v01
QUIC_ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared
SEEDS="42 1 2 3 4"
QNAMES=$(squeue -u "$USER" -h -o "%j")
incomplete=0
sub() {  # seed method week batch exp root numjobs arrayspec want
  local key="sv_s${1}_${3}_${2}_bs${4}"
  local dir="results/inference_seedvar/seed${1}_${3}_${2}_bs${4}"
  local have; have=$(ls "$dir"/*.npz 2>/dev/null | wc -l)
  if [ "$have" -ge "$9" ]; then return 0; fi
  incomplete=$((incomplete + 1))
  if printf '%s\n' "$QNAMES" | grep -qxF "$key"; then echo "running   $key ($have/$9)"; return 0; fi
  local jid
  if jid=$(sbatch --parsable --job-name="$key" --array="$8" \
        --export=ALL,SEEDV="$1",METHOD="$2",TRAIN_WEEK="$3",BATCH="$4",EXP_DIR="$5",DATA_ROOT="$6",NUM_JOBS="$7" \
        slurm_files/run_inference_seedvar.slurm 2>/dev/null); then
    echo "submitted $key ($have/$9) -> $jid"
  else
    echo "deferred  $key ($have/$9) QOS-cap"
  fi
}
for s in $SEEDS; do
  sub "$s" vanilla week_16      64 "$TLS_EXP"  "$TLS_ROOT"  4 "0-3" 53
  sub "$s" bnstats week_16      64 "$TLS_EXP"  "$TLS_ROOT"  4 "0-3" 53
  sub "$s" tent    week_16      64 "$TLS_EXP"  "$TLS_ROOT"  4 "0-3" 53
  sub "$s" vanilla WEEK-2022-44 64 "$QUIC_EXP" "$QUIC_ROOT" 1 "0"   4
  sub "$s" bnstats WEEK-2022-44 64 "$QUIC_EXP" "$QUIC_ROOT" 1 "0"   4
  sub "$s" tent    WEEK-2022-44 64 "$QUIC_EXP" "$QUIC_ROOT" 1 "0"   4
done
echo "INCOMPLETE=${incomplete}"
