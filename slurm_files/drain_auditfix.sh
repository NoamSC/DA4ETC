#!/bin/bash
# Background drain loop: keep retrying the deferred auditfix submissions every 5 min
# as the QOS queue frees up, until all are in. Exits 0 when REMAINING=0.
cd /home/anatbr/students/noamshakedc/da4etc
for i in $(seq 1 288); do   # up to ~24h
  out=$(bash slurm_files/submit_auditfix_remaining.sh)
  rem=$(printf '%s\n' "$out" | sed -n 's/^REMAINING=//p')
  echo "[try $i] depth=$(squeue -u "$USER" -h -r | wc -l) remaining=$rem"
  printf '%s\n' "$out" | grep '^submitted' || true
  if [ "$rem" = "0" ]; then echo "ALL AUDITFIX GROUPS SUBMITTED"; exit 0; fi
  sleep 300
done
echo "TIMEOUT remaining=$rem"; exit 1
