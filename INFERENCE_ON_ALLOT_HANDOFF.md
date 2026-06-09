# Allot multimodal: train + infer-on-the-rest handoff

_Goal: train the multi-modal CNN (`Multimodal_CESNET`, the RE of the QUIC-case
paper) on two ~2% time slices of the **Allot** dataset (an early one and one ~1/4
of the way in — the Allot equivalents of CESNET "week 1" and "week 16"), then run
each trained model across the **whole** Allot timeline to measure how it
generalizes over time (data drift)._

This note is self-contained — you do not need the chat that produced it.

---

## TL;DR — what to run

```bash
cd /home/anatbr/students/noamshakedc/da4etc
# 1. Train both slices (array task 0=early, 1=quarter). Auto-computes norm stats.
sbatch slurm_files/run_allot_multimodal_train.slurm
# 2. After BOTH finish, infer each model over all 52 windows.
sbatch slurm_files/run_allot_inference.slurm
# 3. Analyze (see "Step 4"): per-window accuracy curve per model.
```
Outputs land in `exps/allot_multimodal/{early,quarter}_eq/` (weights + `inference/window_*.npz`).

---

## Background: why this is a custom pipeline (read once)

The Allot chunks only contain **packet sequences** (`ppi-ps/pd/pdt`, `ppiLen`,
`appId`) — they have **no flowstats** (no BYTES/PACKETS/PHIST/FLOW_ENDREASON that
the CESNET multimodal pipeline feeds as a 44-dim vector). So the CESNET loader and
the CESNET inference script (`run_week1_inference.py`, which hardcodes
`flowstats_input_size=44` and iterates `WEEK-*` dirs) **do not work on Allot**.

To run the multimodal model on Allot we **derive a reduced 12-dim flowstats
vector from the packet sequences** and built an Allot-specific loader, training
entry, and inference script. The model is constructed with
`flowstats_input_size=FLOWSTATS_DIM (=12)`, not 44.

> Decision on record: when asked "how to feed flowstats since Allot has none", the
> choice was **derive reduced flowstats from PPI** (not PPI-only). The 12 features
> are window-level aggregates (bytes/packets fwd+rev, duration, ppiLen, roundtrips,
> size & IPT mean/std/max). They overlap somewhat with what the CNN already sees in
> the PPI sequence, and `FLOW_ENDREASON` is unrecoverable — see "Caveats".

## The Allot time axis (how slices/windows are defined)

`data_utils/allot_timeline.py` is the single source of truth, imported by **both**
training and inference so they agree.

- All chunks from `domain_0..3` are concatenated and sorted by filename timestamp.
  The 4 domains don't overlap in time, so this is one chronological timeline:
  domain_0 (2024-09-05→09), domain_1 (09-13→17), domain_2 (2025-02-07→13),
  domain_3 (2025-03-09→15).
- **467 chunks total**, windowed into groups of `round(0.02*467)=9` →
  **52 windows** (~2% each; deliberately close to CESNET's ~53 weeks).
- Training slices (verified):
  - `early`   = window **0** minus its first (partial) chunk → 8 chunks,
    **2024-09-05 11:00 → 18:00** (domain_0). The "week 1" equivalent.
  - `quarter` = window **13** (exactly 25% of the timeline) → 9 chunks,
    **2024-09-14 07:00 → 15:00** (domain_1). The "week 16" equivalent.

Recompute/inspect any time:
```bash
PY=/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python
$PY -c "from data_utils.allot_timeline import *; print('windows', num_windows(), 'quarter idx', quarter_window_index()); \
print('early', [p.name for p in get_training_slice('early')[1]]); \
print('quarter', [p.name for p in get_training_slice('quarter')[1]])"
```

---

## What was built (files)

| File | Role |
|------|------|
| `data_utils/allot_timeline.py` | Global chunk ordering + 2% windows + named training slices |
| `data_utils/allot_multimodal_dataloader.py` | `AllotMultimodalDataset` → `((ppi[3,30], flowstats[12]), label)`; derives flowstats; `appId`→index mapping |
| `scripts/data_prep/compute_allot_normalization_stats.py` | z-score stats per slice (`compute_and_save_norm_stats`, called automatically by training) |
| `scripts/train/train_allot_multimodal.py` | Train one slice → `exps/allot_multimodal/<slice>_eq/` |
| `scripts/inference/run_allot_inference.py` | Run a trained model over all windows → per-window `.npz` |
| `slurm_files/run_allot_multimodal_train.slurm` | array 0=early, 1=quarter |
| `slurm_files/run_allot_inference.slurm` | array 0=early model, 1=quarter model |

Env / python: `/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python`.
Status as of writing: all scripts import + argparse-OK; the loader+model forward was
smoke-tested on real domain_0 chunks (ppi `(B,3,30)`, flowstats `(B,12)`,
`class_preds (B, n_classes)`). **Training/inference have NOT been run yet** — no GPU
in the setup session.

---

## Step 1 — train both slices
```bash
sbatch slurm_files/run_allot_multimodal_train.slurm     # array 0-1
squeue -u $USER | grep allot_mm_train                   # empty = done
```
Each task: builds the slice, holds out the last 20% of its chunks as validation,
builds the `appId`→index label mapping from the **train** chunks, auto-computes
`normalization_stats.npz`, trains 30 epochs, saves best-val checkpoint.

Per-slice outputs in `exps/allot_multimodal/<slice>_eq/`:
- `weights/best_model.pth`  (dict with `model_state_dict`)
- `label_mapping.json`      (appId → class index; defines the model's output space)
- `normalization_stats.npz` (`ppi_mean/std [3,30]`, `flowstats_mean/std [12]`)
- `slice_manifest.json`     (exact train/val chunk paths + window index)
- `plots/`, `tensorboard/`

Manual single-slice run (e.g. to tune): 
```bash
$PY scripts/train/train_allot_multimodal.py --slice early --override \
    --num-epochs 30 --batch-size 256 --lr 3e-3
```
Budget flags (`--num-epochs/--batch-size/--lr/--train-data-frac/--train-per-epoch-frac/--val-data-frac`)
are set on the CLI **on purpose** — `config.py` may be in debug mode (tiny fracs);
the script overrides those so the run is self-contained.

**Verify before inference:**
```bash
ls exps/allot_multimodal/early_eq/weights/best_model.pth \
   exps/allot_multimodal/quarter_eq/weights/best_model.pth
# check final val accuracy isn't garbage:
grep -i "best model" logs/*allot_mm_train_early* | tail -1
grep -i "best model" logs/*allot_mm_train_quarter* | tail -1
```

## Step 2 — infer each model over the whole timeline
```bash
sbatch slurm_files/run_allot_inference.slurm            # array 0-1
squeue -u $USER | grep allot_mm_infer
```
Writes `exps/allot_multimodal/<slice>_eq/inference/window_<00..51>.npz`, each with
`true_labels, pred_labels, softmax, embeddings(+indices), window_index, start_ts,
end_ts, n_total`. The `exists()` guard makes re-runs safe (resumes).

**Verify:** `ls exps/allot_multimodal/early_eq/inference/*.npz | wc -l` → up to 52
(a window with no rows of known classes is skipped — that's expected, not all 52
are guaranteed).

If one task is too slow, shard the windows across more array tasks:
```bash
$PY scripts/inference/run_allot_inference.py --experiment_dir exps/allot_multimodal/early_eq \
    --num-shards 4 --shard-id $SLURM_ARRAY_TASK_ID    # run array 0-3
```

## Step 3 — sanity checks ("no easy bugs")
- **In-distribution check:** for the `early` model, accuracy on its own training
  windows (0; and `quarter` model on window 13) should be the **highest** of the
  curve. If a far window beats the training window, something's off.
- **Same `--window-frac` everywhere.** Training and inference both default to
  `0.02`. If you change it for training, pass the same to inference, or windows
  won't line up.
- **Closed-world:** inference drops rows whose `appId` was not in the training
  slice's `label_mapping.json` (model can't predict unseen classes). Two models
  trained on different slices have **different** label spaces, so don't compare
  their raw class indices — compare accuracy curves.
- Fair sampling: inference uses `seed=42`, `data_sample_frac=0.1` (in the slurm).

## Step 4 — analyze (per-window accuracy / drift curve)
```bash
$PY - <<'PY'
import numpy as np, glob, os, json
for slice_name in ['early','quarter']:
    d=f'exps/allot_multimodal/{slice_name}_eq/inference'
    man=json.load(open(f'exps/allot_multimodal/{slice_name}_eq/slice_manifest.json'))
    rows=[]
    for f in sorted(glob.glob(os.path.join(d,'window_*.npz'))):
        z=np.load(f, allow_pickle=True)
        t,p=z['true_labels'],z['pred_labels']
        if t.size: rows.append((int(z['window_index']), str(z['start_ts']), (t==p).mean(), t.size))
    print(f"\n=== {slice_name} model (trained on window {man['window_index']}) ===")
    print(f"{'win':>3} {'start':>16} {'acc':>6} {'n':>8}")
    for w,ts,acc,n in rows:
        flag=' <-- train' if w==man['window_index'] else ''
        print(f"{w:>3} {ts:>16} {acc:>6.3f} {n:>8}{flag}")
PY
```
This is the headline result: each model's accuracy across all 52 windows. Expect
high accuracy near the training window and degradation as you move away (drift) —
the Allot analogue of the QUIC-paper week-over-week drop.

---

## Output format (slim) + gotchas learned
- **npz is slim** (~12 MB/window, not 225 MB). Per window: `true_labels`,
  `pred_labels` (int16), `topk_idx` (N×5 int16), `topk_prob` (N×5 float16), a
  `embeddings` subsample (N/100, float16) + `embedding_indices`, and
  `window_index/start_ts/end_ts/n_total`. **Full softmax is NOT saved** — use
  `topk_idx[:,0]` for top-1 and the top-5 columns for top-5 acc / confidence.
- **Disk quota:** the first run wrote full softmax+embeddings (225 MB/window → ~90 GB)
  and hit the shared `Netapp5_anatbr` group quota mid-run (TTA died with
  `OSError: Errno 122`). Slim format avoids it. If quota bites again, also clear old
  `results/` (~99 GB) / stale `exps/`.
- **Walls:** vanilla inference ~8h (52 windows × ~9 min, data-loading bound) → 12h wall.
  CoTTA is ~33 min/window (n_aug=32), so `run_allot_tta.slurm` **shards CoTTA 4-ways
  per model** (`--num-shards 4 --shard-id K`); TENT (~8 min/win) runs unsharded.

## Caveats / honest limitations
1. **Derived flowstats are window-level, not whole-flow.** Allot exposes only the
   first ~30 packets, so bytes/packets/duration are over that window and partly
   duplicate the PPI sequence the CNN already sees. Less independent signal than
   CESNET's whole-flow stats + histograms. `FLOW_ENDREASON` is unrecoverable.
2. **Direction convention assumed `ppi-pd==1` = forward.** If `Allot/Readme.docx`
   says `1` = server→client, the fwd/rev flowstats are just name-swapped — harmless
   for training, only matters for feature interpretation. (30-sec check if you care.)
3. **Two models = two label spaces** (built from each slice's appIds). Don't compare
   class indices across models; compare accuracy curves.
4. **Not yet executed** — see Step 1/2 verification before trusting any numbers.

## TODO — still owed
- [ ] **DANN on Allot (NOT done).** DANN is a *training* variant (gradient-reversal
      domain head), not an inference-time method, so it needs its own training run —
      deferred for now. To do it: train with `Multimodal_CESNET(lambda_rgl>0, ...)`
      using source = a training slice and target = unlabeled later windows; the model
      already supports `lambda_rgl`/`lambda_grl_gamma` and has a `domain_classifier`.
      The CESNET DANN slurms (`run_cesnet_*dann*.slurm`, `bayesian_dann_search.py`) are
      the reference. Wire an Allot DANN training entry analogous to
      `train_allot_multimodal.py`.

## Possible next steps
- Add a third slice for a cleaner drift picture, or train on early+quarter combined.
- TTA on the Allot drift curve: **TENT/CoTTA are wired** — `run_allot_inference.py
  --method {tent,cotta}` + `slurm_files/run_allot_tta.slurm` (reuses the TENT/CoTTA
  implementations from `scripts/inference/run_inference.py`). Outputs land in
  `exps/allot_multimodal/<slice>_eq/{inference_tent,inference_cotta}/`. See
  `TTA_RERUN_HANDOFF.md` for how to read those results vs vanilla.

---
_This file is untracked; commit it if you want it kept, or delete after acting on it._
