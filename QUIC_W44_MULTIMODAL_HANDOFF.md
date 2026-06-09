# CESNET-QUIC22 multimodal handoff (train on week 1 → infer on later weeks)

_Goal: train the multimodal model (`Multimodal_CESNET`: PPI conv1d + flowstats MLP)
on the **first** QUIC week, then use that frozen week-1 model for inference across the
remaining QUIC weeks for temporal-generalization analysis._

_CESNET-QUIC22 only has **4 weeks**: `W-2022-44, 45, 46, 47` (Oct–Nov 2022). So
"week 1" = `WEEK-2022-44`, and the inference targets are weeks 44 (in-distribution),
45, 46, 47._

---

## TL;DR — when you come back

```bash
cd /home/anatbr/students/noamshakedc/da4etc
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
ROOT=/home/anatbr/students/noamshakedc/cesnet-quic22-prepared

# 0. Is everything prepared? (need all 4 weeks + stats)
ls $ROOT/WEEK-2022-4{4,5,6,7}/train.parquet $ROOT/WEEK-2022-4{4,5,6,7}/test.parquet
ls $ROOT/normalization_stats.npz $ROOT/label_mapping.json

# 1. Did the training job finish? (checkpoint exists)
ls exps/cesnet_quic22_multimodal_v01/WEEK-2022-44/weights/best_model.pth
squeue -u $USER | grep quic_mm     # empty = done

# 2. Run vanilla inference of the week-44 model across ALL prepared weeks:
$PY scripts/inference/run_inference.py \
    --method vanilla \
    --experiment_dir exps/cesnet_quic22_multimodal_v01 \
    --train_week WEEK-2022-44 \
    --dataset_root $ROOT \
    --output_dir results/inference/quic_w44_inference \
    --data_sample_frac 0.1 --seed 42 --device cuda:0
# -> writes results/inference/quic_w44_inference/WEEK-2022-{44,45,46,47}.npz

# 3. Per-week accuracy (drift curve):
$PY - <<'PYEOF'
import numpy as np, glob, os, re
d='results/inference/quic_w44_inference'
for f in sorted(glob.glob(os.path.join(d,'*.npz'))):
    z=np.load(f, allow_pickle=True)
    if 'true_labels' not in z.files: continue
    t,p=z['true_labels'],z['pred_labels']
    w=re.search(r'WEEK-2022-(\d+)',f).group(1)
    print(f"week {w}: acc={ (t==p).mean():.3f }  n={t.size}")
PYEOF
```

---

## Background / what this thread set up

CESNET-QUIC22 is **not** a drop-in for the existing CESNET-TLS-Year22 pipeline. The
multimodal loader (`data_utils/cesnet_dataloader.create_parquet_loader`) and the
training/inference entrypoints assume the TLS-Year22 schema. Differences bridged:

| TLS-Year22 expects | QUIC has | Fix applied |
|---|---|---|
| split `PPI_IPT`/`PPI_DIRECTIONS`/`PPI_SIZES` | one combined `PPI` string `[[IPT];[DIR];[SIZE]]` (same channel order) | split in prep |
| `PHIST_*` as native list(8) | `PHIST_*` as string-encoded list | parsed to list(8) in prep |
| `FLOW_ENDREASON_END` (4th end reason → 44-dim flowstats) | only IDLE/ACTIVE/OTHER (3) | added `FLOW_ENDREASON_END=False` (constant 0; harmless after norm) |
| per-week `train.parquet`/`test.parquet` | only raw daily `W-2022-XX/<day>/flows-*.parquet` | 70/30 deterministic split in prep |
| `label_mapping.json` | none (102 apps + 3 background in `stats-dataset.json`) | built from the **102 apps** (background excluded) |
| `normalization_stats.npz` | none | computed via `compute_normalization_stats.py` |

Verified end-to-end: the prepared parquet feeds through the **real** `create_parquet_loader`
and yields `ppi (B,3,30)` + `flowstats (B,44)` — exactly what `Multimodal_CESNET`
consumes. **No model code change needed** (`flowstats_input_size=44` stays valid).

### Files created this thread (untracked)
- `scripts/data_prep/prepare_quic_week.py` — QUIC → TLS-compatible converter (one week at a time).
- `slurm_files/run_quic_multimodal_train.slurm` — multimodal training job for W-2022-44.
- `QUIC_W44_MULTIMODAL_HANDOFF.md` — this file.

### Prepared dataset root
`/home/anatbr/students/noamshakedc/cesnet-quic22-prepared/`  (lives on disk, not in repo;
`/home/anatbr/dataset/` is read-only, so it went under student home — 4.4 T free)
```
cesnet-quic22-prepared/
├── label_mapping.json            # 102 apps
├── normalization_stats.npz       # (3,30) ppi + (44,) flowstats
└── WEEK-2022-{44,45,46,47}/
    ├── train.parquet   (70%)
    ├── test.parquet    (30%)
    └── train_test_split.json
```

---

## STATE AT HANDOFF (2026-06-01)
Everything from prep → stats → training is now a **SLURM chain** (run anything
>a few minutes via sbatch, not login-node background — the earlier login-node driver
was killed mid-write and corrupted a partial parquet). Submitted chain:

| Job | Stage | Partition | Depends on |
|-----|-------|-----------|------------|
| `428848` (array 0-1) | prep weeks 46, 47 | cpu-killable | — |
| `428849` | normalization stats | cpu-killable | afterok:428848 |
| `428850` | multimodal training (week 44) | killable (gpu) | afterok:428849 |

Status (updated 2026-06-01, all stages COMPLETED unless noted):
- [x] weeks 44, 45, 46, 47 prepared — job `428848` ✅
- [x] `normalization_stats.npz` — job `428849` ✅
- [x] training (week 44) — job `428850` ✅ 30/30 epochs.
      **Best val acc 89.90%** (epoch 25). Checkpoint: `exps/cesnet_quic22_multimodal_v01/WEEK-2022-44/weights/best_model.pth`
- [~] inference across weeks 44–47 (week-44 model), 3 methods + a gated summary:
      | Job | What | Output dir |
      |-----|------|------------|
      | `437016` | vanilla | `results/inference/quic_w44_inference/` |
      | `437018_0` | TENT  | `results/inference/quic_w44_inference_tent/` |
      | `437018_1` | CoTTA | `results/inference/quic_w44_inference_cotta/` |
      | `437020` | summary (afterany:437016:437018) | `results/inference/quic_w44_summary/` |

SLURM scripts: `run_quic_inference.slurm` (vanilla), `run_quic_tta_inference.slurm`
(array 0=TENT/1=CoTTA), `run_quic_tta_summary.slurm` (compares all three).
All use the same `--seed 42 --data_sample_frac 0.1` and the dropout-fixed
`run_inference.py`, so vanilla/TENT/CoTTA are directly comparable (see TTA_RERUN_HANDOFF.md).

Summary job (`437020`, `scripts/analysis/quic_tta_summary.py`) auto-runs once all three
inference jobs finish; it prints per-week accuracy + macro-F1 with `method - vanilla`
deltas and writes `quic_w44_summary.json` + `quic_w44_accuracy_by_week.png`.

Check progress: `squeue -u $USER | grep -i quic`. Newest logs: `ls -t logs/*quic_*`.
Resubmit any stage with `sbatch slurm_files/<script>.slurm`.

> If any chain job fails, the SLURM scripts below are idempotent and re-submittable.
> Steps A/B/C document the manual equivalents (just `sbatch` the scripts).

---

## SLURM scripts (the whole pipeline) — re-submit any stage as needed
All multi-minute work runs via SLURM (NOT login-node background processes).
To run the full chain from scratch:
```bash
PREP=$(sbatch --parsable slurm_files/run_quic_prepare.slurm)              # weeks in WEEKS=() array
NORM=$(sbatch --parsable --dependency=afterok:$PREP slurm_files/run_quic_norm_stats.slurm)
sbatch --dependency=afterok:$NORM slurm_files/run_quic_multimodal_train.slurm
```

### Step A — prepare weeks — `slurm_files/run_quic_prepare.slurm` (cpu-killable, array)
Edit the `WEEKS=(46 47)` array in the script to choose weeks. Each writes
`WEEK-2022-<NN>/{train,test}.parquet` into the same root and rewrites the identical
`label_mapping.json` (label set fixed from `stats-dataset.json`). ~20–40 min/week
(per-row JSON parsing of ~3.5 M flows/day × 7 days). Weeks 44 & 45 already done.

### Step B — normalization stats — `slurm_files/run_quic_norm_stats.slurm` (cpu-killable)
The loader divides by these; **training and inference must use the same file**, at
`<ROOT>/normalization_stats.npz` (both entrypoints read `week_dir.parent/normalization_stats.npz`).
Sanity-check after:
`python -c "import numpy as np; z=np.load('$ROOT/normalization_stats.npz'); print(z['ppi_mean'].shape, z['flowstats_mean'].shape)"`
must print `(3, 30) (44,)`.

### Step C — training (week 44 only) — `slurm_files/run_quic_multimodal_train.slurm` (gpu)
```bash
sbatch slurm_files/run_quic_multimodal_train.slurm
```
This trains on `WEEK-2022-44` and writes
`exps/cesnet_quic22_multimodal_v01/WEEK-2022-44/weights/best_model.pth`.
Chosen params (in the slurm file): `--train_data_frac 0.2 --val_data_frac 0.05
--batch_size 256 --num_epochs 30 --learning_rate 3e-3`, `--mem 64G`.

**Why not the old multimodal config (`frac=1.0 / batch=64 / 100ep`)?** The multimodal
loader holds the entire sampled set in RAM as parsed Python lists; QUIC week 44 is
~24 M flows, so `frac=1.0` blows past memory and 100 epochs won't fit a 12 h slot.
The job `--requeue`s and resumes per-epoch checkpoints if preempted. To match the old
config exactly, bump `--mem` to ~200 G and lower `--train_data_frac`, or raise epochs.

---

## Step D — the analysis (what week-1→rest is for)
After inference (TL;DR step 2) you have `WEEK-2022-{44..47}.npz` each with
`true_labels, pred_labels, softmax, embeddings`. Expected pattern: highest accuracy on
week 44 (in-distribution), degrading on 45→47 as traffic drifts. The repo's
`scripts/analysis/` has temporal-generalization tooling (e.g.
`run_temporal_generalization.py`, `compute_confusion_matrices.py`) — point them at
`--dataset_root $ROOT` and the `cesnet_quic22_multimodal_v01` experiment dir if you want
the drift plots / confusion matrices, same as the TLS analysis.

To also test adaptation methods (TENT/CoTTA) on the drifted weeks, rerun step 2 with
`--method tent` / `--method cotta` (see `TTA_RERUN_HANDOFF.md` for the dropout-mode
gotcha and tuning knobs — same `run_inference.py`).

---

## Gotchas / sanity checks
- **`--train_week` is `WEEK-2022-44`**, not `week_1`. The checkpoint path is
  `experiment_dir/<train_week>/weights/best_model.pth`; the train script names the
  experiment subdir after the week (`cesnet_quic22_multimodal_v01/{}` → `WEEK-2022-44`).
- **All 4 weeks must be in `$ROOT`** before inference: `run_inference.py` calls
  `get_available_weeks(dataset_root)` and loops every `WEEK-2022-*` dir it finds. Missing
  a week just means no `.npz` for it.
- **Same normalization for train & infer.** Both read `<ROOT>/normalization_stats.npz`.
  If you regenerate it after training, retrain or accuracy will be wrong.
- **102 classes throughout.** `load_label_mapping($ROOT)` returns 102; the saved
  `config.json` rebuilds the model head to match. Don't mix in the background classes
  unless you re-prepare every week with `--include_background` and retrain.
- **Fair drift comparison:** keep `--data_sample_frac 0.1 --seed 42` across all weeks so
  each week's evaluated sample is deterministic.
- Inference uses `model.eval()` + `no_grad` for vanilla (correct, frozen source model).

## This file is untracked
It sits in your working tree. Delete once acted on, or commit if you want it kept.
