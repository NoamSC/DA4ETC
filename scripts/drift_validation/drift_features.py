#!/usr/bin/env python
"""
Shared feature extraction for the docker-registry (class 49) concept-drift study.

Reads CESNET-TLS-Year22 weekly parquet files, isolates the 'docker-registry'
application, and produces, per flow:

  * an interpretable scalar feature table (pandas DataFrame) used for raw-data
    profiling (Task 1) and the pre-vs-post binary domain classifier (Task 2);
  * the model's *exact* normalized inputs -- PPI (3, 30) and flowstats (44,) --
    so the trained Week-1 encoder can be run faithfully for latent-drift SHAP.

The 44 flowstats follow the same order as
data_utils/cesnet_dataloader.py::ParquetCESNETDataset.__getitem__.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATASET_ROOT = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v2')
APP = 'docker-registry'
CLASS_ID = 49
MAX_PKTS = 30

# Raw columns we need (model inputs + TLS metadata for profiling)
_PARQUET_COLS = [
    'APP', 'TLS_JA3', 'TLS_SNI', 'PROTOCOL', 'DST_PORT',
    'PPI_IPT', 'PPI_DIRECTIONS', 'PPI_SIZES',
    'BYTES', 'BYTES_REV', 'PACKETS', 'PACKETS_REV',
    'DURATION', 'PPI_LEN', 'PPI_DURATION', 'PPI_ROUNDTRIPS',
    'PHIST_SRC_SIZES', 'PHIST_DST_SIZES', 'PHIST_SRC_IPT', 'PHIST_DST_IPT',
    'FLOW_ENDREASON_IDLE', 'FLOW_ENDREASON_ACTIVE',
    'FLOW_ENDREASON_END', 'FLOW_ENDREASON_OTHER',
]

# Exact 44-dim flowstats names, in dataloader order.
FLOWSTATS_NAMES = (
    ['BYTES', 'BYTES_REV', 'PACKETS', 'PACKETS_REV',
     'DURATION', 'PPI_LEN', 'PPI_DURATION', 'PPI_ROUNDTRIPS']
    + [f'PHIST_SRC_SIZES_{i}' for i in range(8)]
    + [f'PHIST_DST_SIZES_{i}' for i in range(8)]
    + [f'PHIST_SRC_IPT_{i}' for i in range(8)]
    + [f'PHIST_DST_IPT_{i}' for i in range(8)]
    + ['FLOW_ENDREASON_IDLE', 'FLOW_ENDREASON_ACTIVE',
       'FLOW_ENDREASON_END', 'FLOW_ENDREASON_OTHER']
)
assert len(FLOWSTATS_NAMES) == 44


def _as_array(v):
    """Parquet list / np.ndarray / string-encoded list -> float np.array."""
    if v is None:
        return np.array([], dtype=float)
    if isinstance(v, str):
        import ast
        v = ast.literal_eval(v)
    return np.asarray(v, dtype=float)


def _pad(arr, n=MAX_PKTS):
    arr = arr[:n]
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    return arr


def _seq_stats(sizes, ipt, dirs):
    """Per-flow derived statistics from the PPI packet sequences.

    Only the valid (non-padded) prefix is used: valid length = len(sizes).
    Directions: +1 = forward (client->server), -1 = reverse.
    """
    L = len(sizes)
    out = {}
    if L == 0:
        return None
    out['ppi_size_mean'] = float(np.mean(sizes))
    out['ppi_size_std'] = float(np.std(sizes))
    out['ppi_size_max'] = float(np.max(sizes))
    out['ppi_size_min'] = float(np.min(sizes))
    out['ppi_size_first'] = float(sizes[0])
    out['ppi_size_last'] = float(sizes[-1])
    out['ppi_size_sum'] = float(np.sum(sizes))

    fwd_mask = dirs > 0
    rev_mask = dirs < 0
    out['ppi_n_fwd'] = int(np.sum(fwd_mask))
    out['ppi_n_rev'] = int(np.sum(rev_mask))
    out['ppi_fwd_size_mean'] = float(np.mean(sizes[fwd_mask])) if fwd_mask.any() else 0.0
    out['ppi_rev_size_mean'] = float(np.mean(sizes[rev_mask])) if rev_mask.any() else 0.0
    # number of direction flips along the sequence
    out['ppi_dir_changes'] = int(np.sum(np.abs(np.diff(np.sign(dirs))) > 0)) if L > 1 else 0

    # inter-packet times (IAT). PPI_IPT[0] is conventionally 0 (first pkt).
    iat = ipt[1:] if L > 1 else ipt
    if len(iat) > 0:
        out['ppi_iat_mean'] = float(np.mean(iat))
        out['ppi_iat_std'] = float(np.std(iat))
        out['ppi_iat_max'] = float(np.max(iat))
        out['ppi_iat_first'] = float(iat[0])
    else:
        out['ppi_iat_mean'] = out['ppi_iat_std'] = 0.0
        out['ppi_iat_max'] = out['ppi_iat_first'] = 0.0
    return out


def week_to_period(week):
    if week in (25, 26, 27):
        return 'pre'
    if week in (28, 29, 30):
        return 'post'
    return 'other'


def load_norm_stats():
    s = np.load(DATASET_ROOT / 'normalization_stats.npz')
    return (s['ppi_mean'].astype(np.float32), s['ppi_std'].astype(np.float32),
            s['flowstats_mean'].astype(np.float32), s['flowstats_std'].astype(np.float32))


def extract_week(week, max_flows=None, splits=('train', 'test'), seed=42):
    """Return (df, X_ppi, X_flow) for docker-registry flows of one week.

    df      : interpretable features + 'week','period','tls_ja3','tls_sni'
    X_ppi   : (N, 3, 30) RAW (un-normalized) [IPT, DIR, SIZE]
    X_flow  : (N, 44)    RAW (un-normalized) flowstats, FLOWSTATS_NAMES order
    Rows are aligned across the three returns.
    """
    frames = []
    for split in splits:
        p = DATASET_ROOT / f'WEEK-2022-{week:02d}' / f'{split}.parquet'
        if not p.exists():
            continue
        t = pq.read_table(p, columns=_PARQUET_COLS, filters=[('APP', '=', APP)])
        frames.append(t.to_pandas())
    if not frames:
        return None, None, None
    df_raw = pd.concat(frames, ignore_index=True)
    if max_flows is not None and len(df_raw) > max_flows:
        df_raw = df_raw.sample(n=max_flows, random_state=seed).reset_index(drop=True)

    rows, ppi_list, flow_list = [], [], []
    for _, r in df_raw.iterrows():
        sizes = _as_array(r['PPI_SIZES'])
        ipt = _as_array(r['PPI_IPT'])
        dirs = _as_array(r['PPI_DIRECTIONS'])
        n = min(len(sizes), len(ipt), len(dirs))
        sizes, ipt, dirs = sizes[:n], ipt[:n], dirs[:n]
        sstat = _seq_stats(sizes, ipt, dirs)
        if sstat is None:
            continue

        BYTES, BYTES_REV = float(r['BYTES']), float(r['BYTES_REV'])
        PACKETS, PACKETS_REV = float(r['PACKETS']), float(r['PACKETS_REV'])
        tot_b = BYTES + BYTES_REV
        tot_p = PACKETS + PACKETS_REV

        feat = {
            'week': week,
            'period': week_to_period(week),
            'tls_ja3': r['TLS_JA3'],
            'tls_sni': r['TLS_SNI'],
            'dst_port': int(r['DST_PORT']) if r['DST_PORT'] is not None else -1,
            # raw scalars
            'BYTES': BYTES, 'BYTES_REV': BYTES_REV,
            'PACKETS': PACKETS, 'PACKETS_REV': PACKETS_REV,
            'DURATION': float(r['DURATION']),
            'PPI_LEN': float(r['PPI_LEN']),
            'PPI_DURATION': float(r['PPI_DURATION']),
            'PPI_ROUNDTRIPS': float(r['PPI_ROUNDTRIPS']),
            # derived directional / payload
            'total_bytes': tot_b,
            'total_packets': tot_p,
            'bytes_per_pkt_fwd': BYTES / max(PACKETS, 1.0),
            'bytes_per_pkt_rev': BYTES_REV / max(PACKETS_REV, 1.0),
            'byte_download_ratio': BYTES_REV / tot_b if tot_b > 0 else 0.0,
            'pkt_download_ratio': PACKETS_REV / tot_p if tot_p > 0 else 0.0,
            'flow_end_idle': float(bool(r['FLOW_ENDREASON_IDLE'])),
            'flow_end_active': float(bool(r['FLOW_ENDREASON_ACTIVE'])),
            'flow_end_end': float(bool(r['FLOW_ENDREASON_END'])),
            'flow_end_other': float(bool(r['FLOW_ENDREASON_OTHER'])),
        }
        feat.update(sstat)
        rows.append(feat)

        # model-exact RAW tensors
        ppi = np.stack([_pad(ipt), _pad(dirs), _pad(sizes)]).astype(np.float32)  # (3,30)
        ppi_list.append(ppi)
        flow_vec = np.array([
            BYTES, BYTES_REV, PACKETS, PACKETS_REV,
            float(r['DURATION']), float(r['PPI_LEN']),
            float(r['PPI_DURATION']), float(r['PPI_ROUNDTRIPS']),
            *_as_array(r['PHIST_SRC_SIZES']), *_as_array(r['PHIST_DST_SIZES']),
            *_as_array(r['PHIST_SRC_IPT']), *_as_array(r['PHIST_DST_IPT']),
            float(bool(r['FLOW_ENDREASON_IDLE'])), float(bool(r['FLOW_ENDREASON_ACTIVE'])),
            float(bool(r['FLOW_ENDREASON_END'])), float(bool(r['FLOW_ENDREASON_OTHER'])),
        ], dtype=np.float32)
        assert flow_vec.shape == (44,), flow_vec.shape
        flow_list.append(flow_vec)

    df = pd.DataFrame(rows)
    X_ppi = np.stack(ppi_list)
    X_flow = np.stack(flow_list)
    return df, X_ppi, X_flow


def load_all(weeks=range(25, 31), max_flows=4000):
    dfs, ppis, flows = [], [], []
    for w in weeks:
        df, xp, xf = extract_week(w, max_flows=max_flows)
        if df is None or len(df) == 0:
            print(f'  week {w}: no data')
            continue
        print(f'  week {w}: {len(df)} docker-registry flows')
        dfs.append(df)
        ppis.append(xp)
        flows.append(xf)
    return (pd.concat(dfs, ignore_index=True),
            np.concatenate(ppis), np.concatenate(flows))


if __name__ == '__main__':
    df, xp, xf = load_all()
    print('\nTotal flows:', len(df))
    print('Periods:\n', df['period'].value_counts())
    print('Feature columns:', [c for c in df.columns if c not in
                               ('week', 'period', 'tls_ja3', 'tls_sni', 'dst_port')])
    print('X_ppi', xp.shape, 'X_flow', xf.shape)
