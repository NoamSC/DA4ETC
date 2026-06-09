#!/usr/bin/env python
"""
Task 1 -- Raw-data feature profiling for docker-registry (class 49) concept drift.

Compares the raw network-feature distributions of the PRE-jump window
(weeks 25-27, 2022) against the POST-jump window (weeks 28-30, 2022):

  * descriptive stats (mean / median / variance) for every numeric feature;
  * Welch t-test and Mann-Whitney U + percent delta + Cohen's d, ranked;
  * categorical TLS-handshake profiling (JA3 fingerprint, SNI) -- a proxy for
    protocol/version changes;
  * per-week trajectory plots to localize the jump at the 27 -> 28 boundary.

Outputs:
  results/docker_drift/task1_feature_stats.csv
  results/docker_drift/task1_summary.txt
  figs/docker_drift/task1_distributions.png
  figs/docker_drift/task1_weekly_trajectory.png
  figs/docker_drift/task1_tls_ja3_shift.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_features import load_all  # noqa: E402

FIG_DIR = Path('figs/docker_drift')
RES_DIR = Path('results/docker_drift')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

META = {'week', 'period', 'tls_ja3', 'tls_sni', 'dst_port'}


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / max(na + nb - 2, 1))
    return (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0


def main():
    print('Loading docker-registry flows for weeks 25-30 ...')
    df, _, _ = load_all(weeks=range(25, 31), max_flows=4000)

    pre = df[df.period == 'pre']
    post = df[df.period == 'post']
    num_cols = [c for c in df.columns if c not in META]

    # ---- per-feature statistics & tests -------------------------------------
    rows = []
    for c in num_cols:
        a = pre[c].to_numpy(float)
        b = post[c].to_numpy(float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        mu_a, mu_b = a.mean(), b.mean()
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        try:
            u, p_u = stats.mannwhitneyu(a, b, alternative='two-sided')
        except ValueError:
            u, p_u = np.nan, 1.0
        pct = 100 * (mu_b - mu_a) / abs(mu_a) if mu_a != 0 else np.nan
        rows.append({
            'feature': c,
            'pre_mean': mu_a, 'post_mean': mu_b,
            'pre_median': np.median(a), 'post_median': np.median(b),
            'pre_var': a.var(ddof=1), 'post_var': b.var(ddof=1),
            'pct_delta': pct,
            'cohens_d': cohens_d(a, b),
            't_stat': t, 'p_ttest': p_t,
            'mannwhitney_p': p_u,
        })
    stat = pd.DataFrame(rows)
    stat['abs_d'] = stat['cohens_d'].abs()
    stat = stat.sort_values('abs_d', ascending=False).reset_index(drop=True)
    stat.to_csv(RES_DIR / 'task1_feature_stats.csv', index=False)

    # ---- text summary --------------------------------------------------------
    lines = []
    lines.append('=' * 78)
    lines.append('TASK 1 -- docker-registry (class 49) raw feature drift: PRE(W25-27) vs POST(W28-30)')
    lines.append('=' * 78)
    lines.append(f'n_pre = {len(pre)}   n_post = {len(post)}\n')
    lines.append('Top features by |Cohen\'s d| (effect size of the pre->post shift):')
    lines.append('-' * 78)
    hdr = f"{'feature':24s} {'pre_mean':>12s} {'post_mean':>12s} {'%delta':>9s} {'cohen_d':>8s} {'p(t)':>10s}"
    lines.append(hdr)
    for _, r in stat.head(15).iterrows():
        lines.append(f"{r['feature']:24s} {r['pre_mean']:12.3f} {r['post_mean']:12.3f} "
                     f"{r['pct_delta']:8.1f}% {r['cohens_d']:8.3f} {r['p_ttest']:10.2e}")

    # ---- TLS / JA3 categorical profiling ------------------------------------
    lines.append('\n' + '=' * 78)
    lines.append('TLS handshake metadata (proxy for protocol/version change)')
    lines.append('-' * 78)
    for col, label in [('tls_ja3', 'JA3 fingerprint'), ('tls_sni', 'SNI')]:
        pre_v = pre[col].fillna('<none>')
        post_v = post[col].fillna('<none>')
        lines.append(f'\n{label}:')
        lines.append(f'  unique pre={pre_v.nunique()}  post={post_v.nunique()}')
        top_pre = pre_v.value_counts(normalize=True).head(3)
        top_post = post_v.value_counts(normalize=True).head(3)
        lines.append('  top-3 PRE share : ' +
                     ', '.join(f'{str(k)[:24]}={v:.2%}' for k, v in top_pre.items()))
        lines.append('  top-3 POST share: ' +
                     ', '.join(f'{str(k)[:24]}={v:.2%}' for k, v in top_post.items()))
        # shift in the dominant pre fingerprint's share
        dom = top_pre.index[0]
        lines.append(f'  dominant-PRE value share: pre={pre_v.eq(dom).mean():.2%} '
                     f'-> post={post_v.eq(dom).mean():.2%}')

    summary = '\n'.join(lines)
    (RES_DIR / 'task1_summary.txt').write_text(summary + '\n')
    print('\n' + summary)

    # ---- distribution plots for top-9 shifted features ----------------------
    top9 = stat.head(9)['feature'].tolist()
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, c in zip(axes.ravel(), top9):
        a = pre[c].to_numpy(float)
        b = post[c].to_numpy(float)
        lo, hi = np.nanpercentile(np.concatenate([a, b]), [1, 99])
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 50)
        ax.hist(a, bins=bins, alpha=0.55, density=True, label='PRE (W25-27)', color='#1f77b4')
        ax.hist(b, bins=bins, alpha=0.55, density=True, label='POST (W28-30)', color='#d62728')
        d = stat.loc[stat.feature == c, 'cohens_d'].iloc[0]
        ax.set_title(f'{c}\n(Cohen d={d:.2f})', fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle('docker-registry raw feature distributions: PRE vs POST jump', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / 'task1_distributions.png', dpi=130)
    plt.close(fig)

    # ---- per-week trajectory (localize the jump) ----------------------------
    traj_feats = top9[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    wk = sorted(df.week.unique())
    for ax, c in zip(axes.ravel(), traj_feats):
        g = df.groupby('week')[c]
        med = g.median()
        q1 = g.quantile(0.25)
        q3 = g.quantile(0.75)
        ax.plot(wk, med.loc[wk], 'o-', color='#2c3e50')
        ax.fill_between(wk, q1.loc[wk], q3.loc[wk], alpha=0.2, color='#2c3e50')
        ax.axvspan(27.5, 30.5, color='#d62728', alpha=0.08)
        ax.axvline(27.5, color='#d62728', ls='--', lw=1.5)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel('week (2022)')
    fig.suptitle('Per-week median (IQR band); red line = 27->28 hypothesized jump', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / 'task1_weekly_trajectory.png', dpi=130)
    plt.close(fig)

    # ---- JA3 fingerprint share over weeks -----------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    top_ja3 = df['tls_ja3'].fillna('<none>').value_counts().head(6).index.tolist()
    for ja3 in top_ja3:
        share = (df.assign(j=df['tls_ja3'].fillna('<none>'))
                   .groupby('week')['j'].apply(lambda s: (s == ja3).mean()))
        ax.plot(wk, share.loc[wk], 'o-', label=str(ja3)[:20])
    ax.axvline(27.5, color='#d62728', ls='--', lw=1.5, label='27->28 jump')
    ax.set_xlabel('week (2022)')
    ax.set_ylabel('share of docker-registry flows')
    ax.set_title('TLS JA3 fingerprint composition over weeks (top-6)')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'task1_tls_ja3_shift.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    print('\nSaved:')
    for f in ['task1_feature_stats.csv', 'task1_summary.txt']:
        print('  ', RES_DIR / f)
    for f in ['task1_distributions.png', 'task1_weekly_trajectory.png', 'task1_tls_ja3_shift.png']:
        print('  ', FIG_DIR / f)


if __name__ == '__main__':
    main()
