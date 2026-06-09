Done. Wrote a self-contained handoff to TTA_RERUN_HANDOFF.md. The short version:

When you come back
1. Check the reruns finished (jobs 417790 CoTTA, 417794 TENT):


squeue -u $USER | grep -iE 'cotta|tent'        # empty = done
ls results/inference/week_1_inference_cotta_fixed/*.npz | wc -l   # expect 53
ls results/inference/week_1_inference_tent_fixed/*.npz  | wc -l   # expect 53
Tail the newest logs/*inference_{cotta,tent}*.err for errors.

2. Measure vs vanilla — run the Python block in the handoff (Step 2). It prints per-week accuracy for vanilla / tent_fixed / cotta_fixed plus method − vanilla. The doc also shows how to add the old buggy dirs to see the before/after delta.

3. How to judge "good":

TTA is only worth it if method − vanilla ≥ 0, especially on far/drifted weeks.
Key sanity check that the fix worked: in-distribution week 1 should now be ~vanilla. The dropout bug had dragged CoTTA week-1 to 0.768 vs vanilla's 0.845 — if it's still well below, something's wrong.
TENT may still collapse mid-year (that's real entropy-minimization instability under label shift, not the bug).
4. "No easy bugs" checklist (in the doc): same seed=42/data_sample_frac=0.1 across dirs → identical samples; predictions come from a dropout-off forward; runs are episodic (reset per week) — use --num_jobs 1 --no_reset if you want true continual CoTTA.

Output dirs: fixed → ..._cotta_fixed/ & ..._tent_fixed/; buggy originals kept at ..._cotta/ & ..._tent/; baseline week_1_inference/.

One note: TTA_RERUN_HANDOFF.md is untracked — it'll sit in your working tree for you; delete it once you've acted on it, or commit it if you want it kept.




# TTA rerun handoff (CoTTA / TENT vs vanilla)

_Context: the multimodal week-1 model's TENT & CoTTA inference had a dropout bug
(predictions were read from a model left in `train()` mode, p=0.3 dropout active).
Fixed and resubmitted. This note is what to do when you come back._

## What was changed
- **CoTTA** ([scripts/inference/run_inference.py](scripts/inference/run_inference.py),
  dup [scripts/inference/run_week1_inference_cotta.py](scripts/inference/run_week1_inference_cotta.py)):
  - teacher now `.eval()` (dropout off; BN still on batch stats) — it produces the saved predictions.
  - anchor now copied BEFORE BN running-stats are nulled, so it uses source BN stats.
- **TENT** ([scripts/inference/run_inference.py](scripts/inference/run_inference.py),
  dup [scripts/inference/run_week1_inference_tent.py](scripts/inference/run_week1_inference_tent.py)):
  - `nn.Dropout` modules forced to `.eval()` in the config fn (BN stays on batch stats).

## Jobs submitted (2026-05-31)
| Job | Method | Output dir |
|-----|--------|------------|
| `417790` (array 0-3) | CoTTA fixed | `results/inference/week_1_inference_cotta_fixed/` |
| `417794` (array 0-3) | TENT fixed  | `results/inference/week_1_inference_tent_fixed/` |

Originals (buggy, kept for comparison): `results/inference/week_1_inference_{cotta,tent}/`
Vanilla baseline (correct, model.eval): `results/inference/week_1_inference/`

## Step 1 — confirm the reruns finished cleanly
```bash
squeue -u $USER | grep -iE 'cotta|tent'           # should be empty when done
ls results/inference/week_1_inference_cotta_fixed/*.npz | wc -l   # expect 53
ls results/inference/week_1_inference_tent_fixed/*.npz  | wc -l   # expect 53
# check logs for errors:
ls -t logs/*inference_cotta* logs/*inference_tent* | head; # tail the newest .err
```

## Step 2 — measure accuracy vs vanilla (per-week + summary)
Run from repo root. (Skips non-label cache files like `metrics_cache_*.npz`.)
```bash
/home/anatbr/students/noamshakedc/env/anaconda3/envs/ml2/bin/python - <<'PY'
import numpy as np, glob, os, re
dirs = {
  'vanilla':    'results/inference/week_1_inference',
  'tent_fixed': 'results/inference/week_1_inference_tent_fixed',
  'cotta_fixed':'results/inference/week_1_inference_cotta_fixed',
  # add the old buggy dirs here to see the before/after delta:
  # 'tent_old': 'results/inference/week_1_inference_tent',
  # 'cotta_old':'results/inference/week_1_inference_cotta',
}
def wk(f):
    m=re.search(r'WEEK-2022-(\d+)',f); return int(m.group(1)) if m else None
res={}
for name,d in dirs.items():
    acc={}
    for f in sorted(glob.glob(os.path.join(d,'*.npz'))):
        w=wk(f)
        if w is None: continue
        z=np.load(f, allow_pickle=True)
        if 'true_labels' not in z.files or 'pred_labels' not in z.files: continue
        t,p=z['true_labels'],z['pred_labels']
        if t.size: acc[w]=(t==p).mean()
    res[name]=acc
common=sorted(set.intersection(*[set(a) for a in res.values()]))
print(f"{'wk':>3} "+" ".join(f"{n:>11}" for n in res))
for w in common:
    print(f"{w:>3} "+" ".join(f"{res[n][w]:>11.3f}" for n in res))
print("-"*60)
print(f"{'avg':>3} "+" ".join(f"{np.mean([res[n][w] for w in common]):>11.3f}" for n in res))
base=res['vanilla']
for n in res:
    if n=='vanilla': continue
    d=np.mean([res[n][w]-base[w] for w in common])
    print(f"  {n} - vanilla = {d:+.3f}  (negative => worse than no adaptation)")
PY
```

### How to judge "are they good?"
- **Goal of TTA:** beat the vanilla (frozen week-1 model) baseline, especially on
  far weeks (drift). `method - vanilla` should be **>= 0** to be worth anything.
- Sanity check the fix worked: in-distribution **week 1** accuracy for the fixed
  runs should now be ~vanilla (the dropout bug had dragged CoTTA week-1 down to
  0.768 vs vanilla 0.845). If week-1 is still well below vanilla, something's off.
- TENT may still collapse in mid weeks (0.18-0.34 before) — that's a real
  entropy-minimization instability under label shift, not only the dropout bug.
  The fix reduces noise but might not fully fix collapse.

## Step 3 — quick "no easy bugs" checklist
- Fair comparison: all dirs must use the **same** `seed=42` and
  `data_sample_frac=0.1` (set in the slurm files) → identical samples per week.
- Vanilla uses `model.eval()`+`no_grad` (correct). ✔ already verified.
- Fixed TENT/CoTTA: predictions must come from a **dropout-off** forward
  (teacher.eval for CoTTA; Dropout.eval for TENT). ✔ that's what the fix did.
- Episodic vs continual: both run **episodic** (model reset to source each week;
  no `--no_reset`). The 4-way `--num_jobs` split chunks weeks across array tasks,
  so continual mode wouldn't carry state across jobs anyway. If you want true
  continual CoTTA, run single-job (`--num_jobs 1`) with `--no_reset`.
- Augmentation is plain Gaussian noise std=0.02 on both modalities — a tuning
  knob, not a bug; revisit if CoTTA underperforms.

## If results still look bad after the fix
Likely levers (not bugs): adaptation LR (`--cotta_lr/--tent_lr 1e-3`),
`--rst_m 0.01` (CoTTA restore), `--ap 0.9` (aug gate), `--noise_std 0.02`,
batch size 64 (small batches hurt batch-stat BN under label shift).
