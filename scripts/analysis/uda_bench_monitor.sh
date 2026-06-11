#!/bin/bash
# Persistent monitor for the UDA-benchmark inference jobs (TLS week-16/bs256/DANN + Allot).
# Exits (re-invoking the orchestrator) when EITHER all target outputs are complete OR
# the relevant inference jobs have drained from the queue (so gaps can be resubmitted).
# READ-ONLY: never deletes or modifies outputs. Safe to run repeatedly.
cd /home/anatbr/students/noamshakedc/da4etc
LOG=scripts/analysis/uda_bench_monitor.log
USER_=$(whoami)
START=$(date +%s)
MAX=43200   # 12h safety cap

tls_dirs=(week_16_inference_bnstats week_16_inference_tent_fixed week_16_inference_tent_bs256 \
          week_16_inference_cotta_fixed week_16_inference_cotta_bs256 \
          week_1_inference_tent_bs256 week_1_inference_cotta_bs256 \
          dann_w16src_inference dann_w01src_inference)

while true; do
  remaining=0
  report=""
  # TLS: each target 53
  for d in "${tls_dirs[@]}"; do
    n=$(ls results/inference/$d/WEEK-2022-*.npz 2>/dev/null | wc -l)
    report+=$(printf "  %2d/53  %s\n" "$n" "$d")
    [ "$n" -lt 53 ] && remaining=$((remaining + 53 - n))
  done
  # Allot: ceiling = vanilla count per slice
  for s in early quarter; do
    van=$(ls exps/allot_multimodal/${s}_eq/inference/window_*.npz 2>/dev/null | wc -l)
    for m in inference_bnstats inference_tent inference_cotta; do
      n=$(ls exps/allot_multimodal/${s}_eq/$m/window_*.npz 2>/dev/null | wc -l)
      report+=$(printf "  %2d/%2d  allot/%s/%s\n" "$n" "$van" "$s" "$m")
      [ "$van" -gt 0 ] && [ "$n" -lt "$van" ] && remaining=$((remaining + van - n))
    done
  done

  jobs=$(squeue -u "$USER_" -h -o "%j" 2>/dev/null | grep -Ec 'tls_tta|dann_infer|allot_tta|allot_bnstat')
  elapsed=$(( $(date +%s) - START ))

  {
    echo "===== $(date '+%F %T')  elapsed=${elapsed}s  remaining_npz=${remaining}  relevant_jobs=${jobs} ====="
    echo "$report"
  } > "$LOG"

  if [ "$remaining" -eq 0 ]; then
    echo "DONE: all targets complete at $(date '+%F %T')" >> "$LOG"; exit 0
  fi
  if [ "$jobs" -eq 0 ]; then
    echo "DRAINED: relevant jobs left the queue with ${remaining} npz still missing at $(date '+%F %T')" >> "$LOG"; exit 0
  fi
  if [ "$elapsed" -ge "$MAX" ]; then
    echo "TIMEOUT: 12h cap reached with ${remaining} npz missing" >> "$LOG"; exit 0
  fi
  sleep 300
done
