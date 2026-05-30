#!/usr/bin/env python
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

inference_dir = Path('figs/week_1_inference')
output_path   = inference_dir / 'macro_f1_over_time.png'

files = sorted(
    inference_dir.glob('WEEK-2022-*.npz'),
    key=lambda p: int(re.search(r'(\d+)$', p.stem).group(1))
)

week_nums, macro_f1s = [], []
for f in tqdm(files):
    wn = int(re.search(r'(\d+)$', f.stem).group(1))
    d  = np.load(f)
    num_classes = d['softmax'].shape[1]
    score = f1_score(d['true_labels'], d['pred_labels'],
                     labels=list(range(num_classes)),
                     average='macro', zero_division=0)
    week_nums.append(wn)
    macro_f1s.append(score)

week_nums = np.array(week_nums)
macro_f1s = np.array(macro_f1s)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(week_nums, macro_f1s, color='steelblue', linewidth=2,
        marker='o', markersize=4)
ax.axhline(macro_f1s[0], color='grey', linewidth=0.8, linestyle='--',
           label=f'Week-0 baseline ({macro_f1s[0]:.3f})')
ax.set_xlabel('Week Number', fontsize=11)
ax.set_ylabel('Macro F1', fontsize=11)
ax.set_title('Macro F1 over time — model trained on week 1', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_xticks(week_nums)
ax.tick_params(axis='x', labelsize=7, rotation=45)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved -> {output_path}")
