#!/bin/bash
# Keep the DANN normalization-contrast arms alive until both reach enough epochs to
# judge (>=8 of 10). Queue-aware (no double-submit) + disk-guarded. 15-min cadence.
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
chk(){ $PY - "$1" <<'PYEOF'
import sys,glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
pre=sys.argv[1]; n=0
for tgt in ['00','10','25','33','44','52']:
  g=glob.glob(f"exps/{pre}/WEEK-2022-16_val_WEEK-2022-{tgt}/tensorboard")
  if not g: continue
  ea=EventAccumulator(g[0],size_guidance={'scalars':0}); ea.Reload()
  t='Accuracy/domain_classifier'
  if t in ea.Tags().get('scalars',[]) and len(ea.Scalars(t))>=8: n+=1
print(n)
PYEOF
}
for i in $(seq 1 96); do
  nn=$(chk cesnet_tls_dann_fwd_w16_nonorm_v01); nc=$(chk cesnet_tls_dann_fwd_w16_normctrl_v01)
  qn=$(squeue -u "$USER" -h -o "%j" | grep -c dann_nonorm)
  qc=$(squeue -u "$USER" -h -o "%j" | grep -c dann_normc)
  echo "[try $i] nonorm ${nn}/6 (q$qn)  normctrl ${nc}/6 (q$qc)"
  if [ "${nn:-0}" -ge 6 ] && [ "${nc:-0}" -ge 6 ]; then echo "CONTRAST COMPLETE"; exit 0; fi
  DA4=$(du -sm /home/anatbr/students/noamshakedc/da4etc 2>/dev/null | awk '{print $1}')
  if [ $(( ${DA4:-0} + 69000 )) -gt $((470*1024)) ]; then echo "  DISK_GUARD"; sleep 900; continue; fi
  if [ "${nn:-0}" -lt 6 ] && [ "$qn" -eq 0 ]; then sbatch slurm_files/run_dann_fwd_w16_nonorm.slurm 2>/dev/null && echo "  +nonorm"; fi
  if [ "${nc:-0}" -lt 6 ] && [ "$qc" -eq 0 ]; then sbatch slurm_files/run_dann_fwd_w16_normctrl.slurm 2>/dev/null && echo "  +normctrl"; fi
  sleep 900
done
echo "TIMEOUT nonorm=$nn normctrl=$nc"; exit 1
