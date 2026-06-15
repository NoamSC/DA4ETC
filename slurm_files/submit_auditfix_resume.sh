#!/bin/bash
# Queue-aware resume submitter (preemption-safe, no state file): for each auditfix
# group, skip if its output dir is already complete (.npz >= expected) OR if a job
# named af_<key> is already queued/running; otherwise resubmit with that job name so
# run_inference fills the missing weeks (skip-existing). Safe to call repeatedly in a
# drain loop — preempted groups (no longer queued, still incomplete) get resubmitted.
# Prints INCOMPLETE=N (groups not yet at full .npz count).
cd /home/anatbr/students/noamshakedc/da4etc

# Disk safety backstop vs the user's 500GB netapp cap (NOT the mount free space).
# All of this job's writes land under da4etc; static siblings (env 54G, quic 14G,
# opt) are added as a measured baseline. Block new submissions at 470GB (30GB margin).
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

QNAMES=$(squeue -u "$USER" -h -o "%j")
incomplete=0
group() {  # method week batch exp root expected_npz
  local key="${2}_${1}_bs${3}"
  local dir="results/inference_auditfix/${key}"
  local have; have=$(ls "$dir"/*.npz 2>/dev/null | wc -l)
  if [ "$have" -ge "$6" ]; then return 0; fi          # complete
  incomplete=$((incomplete + 1))
  if printf '%s\n' "$QNAMES" | grep -qxF "af_${key}"; then
    echo "running   ${key} (${have}/${6})"; return 0   # already in queue
  fi
  local jid
  if jid=$(sbatch --parsable --job-name="af_${key}" \
        --export=ALL,METHOD="$1",TRAIN_WEEK="$2",BATCH="$3",EXP_DIR="$4",DATA_ROOT="$5" \
        slurm_files/run_inference_auditfix.slurm 2>/dev/null); then
    echo "submitted ${key} (${have}/${6}) -> ${jid}"
  else
    echo "deferred  ${key} (${have}/${6}) QOS-cap"
  fi
}

for W in week_1 week_16; do
  group vanilla "$W" 64  "$TLS_EXP" "$TLS_ROOT" 53
  group bnstats "$W" 64  "$TLS_EXP" "$TLS_ROOT" 53
  group tent    "$W" 64  "$TLS_EXP" "$TLS_ROOT" 53
  group tent    "$W" 256 "$TLS_EXP" "$TLS_ROOT" 53
  group cotta   "$W" 64  "$TLS_EXP" "$TLS_ROOT" 53
  group cotta   "$W" 256 "$TLS_EXP" "$TLS_ROOT" 53
done
group vanilla WEEK-2022-44 64  "$QUIC_EXP" "$QUIC_ROOT" 4
group bnstats WEEK-2022-44 64  "$QUIC_EXP" "$QUIC_ROOT" 4
group tent    WEEK-2022-44 64  "$QUIC_EXP" "$QUIC_ROOT" 4
group tent    WEEK-2022-44 256 "$QUIC_EXP" "$QUIC_ROOT" 4
group cotta   WEEK-2022-44 64  "$QUIC_EXP" "$QUIC_ROOT" 4
group cotta   WEEK-2022-44 256 "$QUIC_EXP" "$QUIC_ROOT" 4

echo "INCOMPLETE=${incomplete}"
