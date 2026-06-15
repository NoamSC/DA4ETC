#!/bin/bash
# Background drain for the resume pass: retry submit_auditfix_resume.sh every 5 min
# until all incomplete groups are (re)submitted. Exits 0 when REMAINING=0.
cd /home/anatbr/students/noamshakedc/da4etc
for i in $(seq 1 288); do
  out=$(bash slurm_files/submit_auditfix_resume.sh)
  inc=$(printf '%s\n' "$out" | sed -n 's/^INCOMPLETE=//p')
  echo "[try $i] depth=$(squeue -u "$USER" -h -r | wc -l) incomplete=$inc"
  printf '%s\n' "$out" | grep -E '^(submitted|deferred)' || true
  if [ "$inc" = "0" ]; then echo "ALL AUDITFIX GROUPS COMPLETE"; exit 0; fi
  sleep 300
done
echo "TIMEOUT incomplete=$inc"; exit 1
