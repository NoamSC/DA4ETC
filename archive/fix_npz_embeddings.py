"""
Fix all existing NPZ files after over-sampling accident.

What happened: three cleanup jobs ran in series, each taking 10% of the
already-subsampled embeddings.  true_labels / pred_labels / softmax are
intact at full N.  Embeddings ended up at ~0.1% of N instead of 10%.

This script:
  1. Deletes corrupt files.
  2. For valid files: drops the wrong embedding_indices key, keeps embeddings
     as-is (small random sample — still valid for distance/drift analysis).
  3. Prints a disk-usage summary.

Future inference runs use the fixed scripts that sample exactly 10% of N
and write correct embedding_indices.
"""

import numpy as np
from pathlib import Path

dirs = [
    'figs/week_1_inference',
    'figs/week_1_inference_tent',
    'figs/week_1_inference_cotta',
]

total_before = total_after = 0
ok = fixed = deleted = 0

for d in dirs:
    for f in sorted(Path(d).glob('WEEK-*.npz')):
        size_before = f.stat().st_size
        total_before += size_before

        try:
            data = np.load(f)
            _ = data['true_labels']
        except Exception as e:
            print(f"DELETE corrupt: {f.parent.name}/{f.name}: {e}")
            f.unlink()
            deleted += 1
            continue

        N    = len(data['true_labels'])
        E    = len(data['embeddings'])
        acc  = (data['true_labels'] == data['pred_labels']).mean()

        # rewrite without embedding_indices (it's wrong in all existing files)
        np.savez_compressed(
            f,
            true_labels=data['true_labels'],
            pred_labels=data['pred_labels'],
            softmax=data['softmax'],
            embeddings=data['embeddings'],   # keep whatever is there
        )

        size_after = f.stat().st_size
        total_after += size_after
        fixed += 1
        print(f"  {f.parent.name}/{f.name}: N={N:>7}  emb={E:>5} ({E/N*100:.2f}%)"
              f"  acc={acc:.3f}  {size_before/1e6:.1f}→{size_after/1e6:.1f} MB",
              flush=True)

print(f"\nProcessed: {fixed} fixed, {deleted} deleted")
print(f"Total disk: {total_before/1e9:.2f} GB → {total_after/1e9:.2f} GB")
print(f"\nNOTE: existing embeddings are {'':.0f}smaller than the target 10% of N.")
print("      Run fresh inference to get properly sampled embeddings for drift analysis.")
