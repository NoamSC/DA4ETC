"""
Multimodal dataloader for the Allot dataset.

The Allot hourly chunks (CSV or parquet) only contain per-packet sequence
information (PPI) plus an ``appId`` label -- they do NOT carry the rich
flow-statistics that the CESNET pipeline provides. To feed the multi-modal
CNN (``models.multimodal_cesnet.Multimodal_CESNET``), which expects a PPI
tensor of shape ``(3, 30)`` AND a flowstats vector, we DERIVE a reduced
flowstats vector from the packet sequences themselves.

Per-flow Allot columns used:
  - ``ppi-ps``  : per-packet sizes        (space-separated int list in a string)
  - ``ppi-pd``  : per-packet directions   (0/1 list; converted to -1/+1)
  - ``ppi-pdt`` : cumulative packet times (ms; converted to per-packet IPT)
  - ``ppiLen``  : number of packets in the flow
  - ``appId``   : service label

PPI tensor channels (matching CESNET ordering ``[IPT, DIRECTIONS, SIZES]``):
  - channel 0: inter-packet time (diff of cumulative ``ppi-pdt``, first = 0)
  - channel 1: direction (+1 for ``ppi-pd``==1, -1 otherwise)
  - channel 2: size (``ppi-ps``)

Derived flowstats (``FLOWSTATS_DIM`` features, all computed over the PPI window;
note these are window-level aggregates, not whole-flow totals -- Allot does not
expose whole-flow stats):
  0: bytes_fwd     sum of sizes where direction is forward (pd==1)
  1: bytes_rev     sum of sizes where direction is reverse (pd==0)
  2: packets_fwd   count of forward packets
  3: packets_rev   count of reverse packets
  4: duration_ms   last cumulative time minus first
  5: ppi_len       the ppiLen field
  6: roundtrips    number of direction changes in the sequence
  7: mean_size     mean packet size
  8: std_size      std of packet size
  9: mean_ipt      mean inter-packet time
 10: std_ipt       std of inter-packet time
 11: max_ipt       max inter-packet time
"""

import re
import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PPI_LEN = 30
PPI_CHANNELS = 3

FLOWSTATS_NAMES = [
    "bytes_fwd", "bytes_rev", "packets_fwd", "packets_rev",
    "duration_ms", "ppi_len", "roundtrips",
    "mean_size", "std_size", "mean_ipt", "std_ipt", "max_ipt",
]
FLOWSTATS_DIM = len(FLOWSTATS_NAMES)

# Raw Allot column names
COL_SIZES = "ppi-ps"
COL_DIRS = "ppi-pd"
COL_TIMES = "ppi-pdt"
COL_PPI_LEN = "ppiLen"
COL_APP = "appId"


def _parse_int_list(x):
    """Parse a value that may be a python list/array or a string like '[ 1  2 3]'."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]
    if isinstance(x, str):
        return [int(v) for v in re.findall(r"-?\d+", x)]
    return []


def derive_flowstats(sizes, dirs01, times_cumulative, ppi_len_field):
    """Compute the reduced flowstats vector (FLOWSTATS_DIM,) from PPI sequences."""
    sizes = np.asarray(sizes, dtype=np.float64)
    dirs = np.asarray(dirs01, dtype=np.float64)
    pdt = np.asarray(times_cumulative, dtype=np.float64)
    n = len(sizes)

    if n == 0:
        return np.zeros(FLOWSTATS_DIM, dtype=np.float32)

    fwd = dirs == 1
    rev = ~fwd
    ipt = np.diff(pdt) if n > 1 else np.array([0.0])

    flowstats = np.array([
        sizes[fwd].sum(),                       # bytes_fwd
        sizes[rev].sum(),                       # bytes_rev
        float(fwd.sum()),                       # packets_fwd
        float(rev.sum()),                       # packets_rev
        float(pdt[-1] - pdt[0]),                # duration_ms
        float(ppi_len_field),                   # ppi_len
        float(np.sum(dirs[1:] != dirs[:-1])),   # roundtrips
        float(sizes.mean()),                    # mean_size
        float(sizes.std()),                     # std_size
        float(ipt.mean()),                      # mean_ipt
        float(ipt.std()),                       # std_ipt
        float(ipt.max()),                       # max_ipt
    ], dtype=np.float32)
    return flowstats


def build_ppi(sizes, dirs01, times_cumulative, max_packets=PPI_LEN):
    """Build the (3, max_packets) PPI tensor: [IPT, DIRECTIONS, SIZES], zero-padded."""
    n = min(len(sizes), len(dirs01), len(times_cumulative), max_packets)
    sizes = [float(s) for s in sizes[:n]]
    dirs = [1.0 if d == 1 else -1.0 for d in dirs01[:n]]
    pdt = [float(t) for t in times_cumulative[:n]]
    ipt = [0.0] + [pdt[i] - pdt[i - 1] for i in range(1, n)]

    def pad(a):
        return a + [0.0] * (max_packets - len(a))

    ppi = np.stack([pad(ipt), pad(dirs), pad(sizes)]).astype(np.float32)
    assert ppi.shape == (PPI_CHANNELS, max_packets)
    return ppi


def build_allot_label_mapping(files):
    """Scan ``appId`` across files and return a {appId: contiguous_index} dict (sorted)."""
    app_ids = set()
    for f in files:
        df = _read_allot_file(f, columns=[COL_APP])
        app_ids.update(int(a) for a in df[COL_APP].unique())
    return {app_id: i for i, app_id in enumerate(sorted(app_ids))}


def _read_allot_file(path, columns=None):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


class AllotMultimodalDataset(Dataset):
    def __init__(
        self,
        files,
        label_mapping,
        max_packets=PPI_LEN,
        data_sample_frac=None,
        seed=42,
        normalization_stats=None,
    ):
        assert isinstance(files, (list, tuple)) and len(files) > 0, \
            "files must be a non-empty list"
        assert isinstance(label_mapping, dict) and len(label_mapping) > 0, \
            "label_mapping must be a non-empty dict"
        assert max_packets > 0, "max_packets must be > 0"
        if data_sample_frac is not None:
            assert 0 < data_sample_frac <= 1, "data_sample_frac must be in (0, 1]"

        self.files = list(files)
        # Normalize keys to int so lookups work regardless of how the mapping was built
        self.label_mapping = {int(k): int(v) for k, v in label_mapping.items()}
        self.max_packets = max_packets

        # ---------- normalization ----------
        self.normalize = normalization_stats is not None
        if self.normalize:
            stats = (np.load(normalization_stats)
                     if isinstance(normalization_stats, (str, Path))
                     else normalization_stats)
            self.ppi_mean = torch.tensor(stats["ppi_mean"], dtype=torch.float32)
            self.ppi_std = torch.tensor(stats["ppi_std"], dtype=torch.float32)
            self.flowstats_mean = torch.tensor(stats["flowstats_mean"], dtype=torch.float32)
            self.flowstats_std = torch.tensor(stats["flowstats_std"], dtype=torch.float32)
            assert self.ppi_mean.shape == (PPI_CHANNELS, max_packets)
            assert self.ppi_std.shape == (PPI_CHANNELS, max_packets)
            assert self.flowstats_mean.shape == (FLOWSTATS_DIM,)
            assert self.flowstats_std.shape == (FLOWSTATS_DIM,)

        self.dfs = []
        self.cum_lens = [0]
        rng = np.random.RandomState(seed)
        use_cols = [COL_SIZES, COL_DIRS, COL_TIMES, COL_PPI_LEN, COL_APP]

        for f in self.files:
            df = _read_allot_file(f, columns=use_cols)

            missing = set(use_cols) - set(df.columns)
            assert not missing, f"Missing columns in {f}: {missing}"

            # Parse the string-encoded list columns once, up front.
            for col in (COL_SIZES, COL_DIRS, COL_TIMES):
                df[col] = df[col].apply(_parse_int_list)

            df = df[df[COL_APP].isin(self.label_mapping)]
            assert len(df) > 0, f"No rows with known appId left after filtering in {f}"

            if data_sample_frac is not None:
                file_seed = int(rng.randint(0, 2**31 - 1))
                df = df.sample(frac=data_sample_frac, random_state=file_seed)
                assert len(df) > 0, f"Sampling removed all rows in {f}"

            df = df.reset_index(drop=True)
            self.dfs.append(df)
            self.cum_lens.append(self.cum_lens[-1] + len(df))

        self.total_len = self.cum_lens[-1]
        assert self.total_len > 0, "Dataset is empty after processing all files"

    def __len__(self):
        return self.total_len

    def get_class_counts(self):
        """Return {label_index: count} across the whole dataset."""
        counts = {}
        for df in self.dfs:
            for app_id, n in df[COL_APP].value_counts().items():
                label = self.label_mapping[int(app_id)]
                counts[label] = counts.get(label, 0) + int(n)
        return counts

    def __getitem__(self, idx):
        assert 0 <= idx < self.total_len, "Index out of bounds"
        file_i = bisect.bisect_right(self.cum_lens, idx) - 1
        local_i = idx - self.cum_lens[file_i]
        row = self.dfs[file_i].iloc[local_i]

        sizes = row[COL_SIZES]
        dirs = row[COL_DIRS]
        times = row[COL_TIMES]

        ppi = torch.from_numpy(build_ppi(sizes, dirs, times, self.max_packets))
        flowstats = torch.from_numpy(
            derive_flowstats(sizes, dirs, times, row[COL_PPI_LEN])
        )

        if self.normalize:
            ppi = (ppi - self.ppi_mean) / (self.ppi_std + 1e-8)
            flowstats = (flowstats - self.flowstats_mean) / (self.flowstats_std + 1e-8)

        label = self.label_mapping[int(row[COL_APP])]
        return (ppi, flowstats), torch.tensor(label, dtype=torch.long)


def create_allot_multimodal_loader(
    files,
    label_mapping,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    max_packets=PPI_LEN,
    data_sample_frac=None,
    seed=42,
    normalization_stats=None,
    drop_last=True,
):
    assert batch_size > 0, "batch_size must be > 0"
    assert num_workers >= 0, "num_workers must be >= 0"

    dataset = AllotMultimodalDataset(
        files=files,
        label_mapping=label_mapping,
        max_packets=max_packets,
        data_sample_frac=data_sample_frac,
        seed=seed,
        normalization_stats=normalization_stats,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
