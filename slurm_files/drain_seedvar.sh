#!/bin/bash
# Background drain for the multi-seed significance study: retry submit_seedvar.sh
# every 5 min until all seed x method x source groups are complete.
cd /home/anatbr/students/noamshakedc/da4etc
for i in $(seq 1 288); do
  out=$(bash slurm_files/submit_seedvar.sh)
  inc=$(printf '%s\n' "$out" | sed -n 's/^INCOMPLETE=//p')
  echo "[try $i] depth=$(squeue -u "$USER" -h -r | wc -l) incomplete=$inc"
  printf '%s\n' "$out" | grep -E '^(submitted|deferred)' || true
  if [ "$inc" = "0" ]; then echo "ALL SEEDVAR GROUPS COMPLETE"; exit 0; fi
  sleep 300
done
echo "TIMEOUT incomplete=$inc"; exit 1
