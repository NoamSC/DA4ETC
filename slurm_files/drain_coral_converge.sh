#!/bin/bash
# Drive the CORAL sweep to convergence (50 epochs) despite killable
# preemption. Resubmits the array (resume from rolling checkpoints) whenever no
# coral_fwd_w16 job is queued and not all 52 targets have reached 50 epochs.
# Queue-aware (never double-submits a target -> no checkpoint races) + disk-guarded.
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
for i in $(seq 1 96); do   # ~48h at 30min cadence
  done=$($PY - <<'PYEOF'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
n=tot=0
for d in glob.glob("exps/cesnet_tls_coral_fwd_w16_v01/*/tensorboard"):
    tot+=1
    ea=EventAccumulator(d,size_guidance={'scalars':0}); ea.Reload()
    if 'Accuracy/validation' in ea.Tags().get('scalars',[]) and len(ea.Scalars('Accuracy/validation'))>=50:
        n+=1
print(f"{n}/{tot}")
PYEOF
)
  ndone=${done%%/*}; ntot=${done##*/}
  inq=$(squeue -u "$USER" -h -o "%j" | grep -c coral_fwd_w16)
  echo "[try $i] converged(>=50ep): $done ; dann in queue: $inq"
  if [ "${ndone:-0}" -ge "${ntot:-52}" ] && [ "${ntot:-0}" -ge 52 ]; then echo "ALL CORAL TARGETS CONVERGED"; exit 0; fi
  # disk guard (same 470GB cap logic)
  DA4_MB=$(du -sm /home/anatbr/students/noamshakedc/da4etc 2>/dev/null | awk '{print $1}')
  if [ $(( ${DA4_MB:-0} + 69000 )) -gt $((470*1024)) ]; then echo "  DISK_GUARD: >470GB, not resubmitting"; sleep 1800; continue; fi
  if [ "$inq" -eq 0 ]; then
    jid=$(sbatch --parsable slurm_files/run_coral_fwd_w16.slurm 2>/dev/null) && echo "  resubmitted CORAL array -> $jid" || echo "  resubmit deferred (QOS cap)"
  fi
  sleep 1800
done
echo "TIMEOUT: converged=$done"; exit 1
