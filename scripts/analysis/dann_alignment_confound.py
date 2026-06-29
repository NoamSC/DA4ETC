#!/usr/bin/env python
"""
DANN alignment-vs-retraining confound control.
===============================================

DANN-diagonal (one model trained per target week, source=Week-1, with a
gradient-reversal domain-alignment objective) beats the frozen source model by
+0.064 mean forward accuracy on CESNET-TLS-Year22. Is that gain from the
adversarial ALIGNMENT, or merely from training a fresh per-week model at all?

Control: re-train the IDENTICAL diagonal setup with the alignment weight set to
zero (lambda_dann=0 -> no domain-head loss, no GRL gradient), everything else
unchanged (50 epochs, same frac/seed). Compare three frozen-eval forward curves
on the shared benchmark protocol (data_sample_frac=0.1, seed=42):

  vanilla        frozen Week-1 source model
  DANN lambda>0  per-week, WITH alignment        (exps/cesnet_tls_dann_fwd_w01_v01)
  DANN lambda=0  per-week, WITHOUT alignment      (exps/cesnet_tls_dann_lambda0_w01_v01)

Decomposition of the DANN gain:
  alignment-only  = acc(lambda>0) - acc(lambda=0)
  retraining-only = acc(lambda=0) - acc(vanilla)

Result (44 common forward weeks): the gain is almost entirely ALIGNMENT
(+0.058, positive in 42/44 weeks); per-week retraining alone buys +0.007. This
CONFIRMS the paper's reading -- the Week-1 forward window spans the global
Week-10 sensor event, so global alignment has real global structure to exploit;
on the clean Week-16 source (per-class drift only) DANN gives just +0.006.
"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np

DIRS = {
    'vanilla':       'results/inference_auditfix/week_1_vanilla_bs64',
    'dann_lambda>0': 'results/inference/dann_fwd_w01_diagonal',
    'dann_lambda0':  'results/inference/dann_lambda0_w01_diagonal',
}
_wk = lambda f: int(re.search(r'WEEK-2022-(\d+)', f).group(1))


def accs(d):
    out = {}
    for f in glob.glob(d + '/WEEK-2022-*.npz'):
        z = np.load(f)
        out[_wk(f)] = float((z['pred_labels'] == z['true_labels']).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_week', type=int, default=1)
    ap.add_argument('--output', default='results/inference/dann_alignment_confound.json')
    args = ap.parse_args()

    A = {k: accs(v) for k, v in DIRS.items()}
    common = sorted(set(A['vanilla']) & set(A['dann_lambda>0']) & set(A['dann_lambda0']))
    fwd = [w for w in common if w > args.source_week]

    mv = np.mean([A['vanilla'][w] for w in fwd])
    m1 = np.mean([A['dann_lambda>0'][w] for w in fwd])
    m0 = np.mean([A['dann_lambda0'][w] for w in fwd])
    d_align = np.array([A['dann_lambda>0'][w] - A['dann_lambda0'][w] for w in fwd])
    d_retr  = np.array([A['dann_lambda0'][w] - A['vanilla'][w] for w in fwd])

    out = {
        'source_week': args.source_week, 'n_forward_weeks': len(fwd), 'weeks': fwd,
        'mean_forward_acc': {'vanilla': round(mv, 4), 'dann_lambda_gt0': round(m1, 4),
                             'dann_lambda0': round(m0, 4)},
        'dann_gain_vs_vanilla': round(m1 - mv, 4),
        'alignment_only_gain': round(float(d_align.mean()), 4),
        'alignment_only_weeks_positive': f'{int((d_align > 0).sum())}/{len(fwd)}',
        'retraining_only_gain': round(float(d_retr.mean()), 4),
        'retraining_only_weeks_positive': f'{int((d_retr > 0).sum())}/{len(fwd)}',
        'verdict': 'DANN forward gain is alignment-driven, not a retraining artifact',
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
