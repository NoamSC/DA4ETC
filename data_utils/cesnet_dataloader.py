import torch
from torch.utils.data import Dataset, DataLoader
import bisect
import numpy as np
import ast
import pyarrow.parquet as pq
from pathlib import Path

def _parse_list_column(series):
    """Parse a column that may contain string-encoded lists."""
    if series.dtype == object and len(series) > 0:
        first_val = series.iloc[0]
        if isinstance(first_val, str):
            return series.apply(ast.literal_eval)
    return series

class ParquetCESNETDataset(Dataset):
    def __init__(
        self,
        parquet_files,
        label_mapping,
        max_packets=30,
        data_sample_frac=None,
        seed=42,
        columns=None,
        normalization_stats=None,
    ):
        # ---------- basic sanity ----------
        assert isinstance(parquet_files, (list, tuple)) and len(parquet_files) > 0, \
            "parquet_files must be a non-empty list"
        assert isinstance(label_mapping, dict) and len(label_mapping) > 0, \
            "label_mapping must be a non-empty dict"
        assert max_packets > 0, "max_packets must be > 0"
        if data_sample_frac is not None:
            assert 0 < data_sample_frac <= 1, "data_sample_frac must be in (0, 1]"

        self.files = list(parquet_files)
        self.label_mapping = label_mapping
        self.max_packets = max_packets
        self.data_sample_frac = data_sample_frac
        self.seed = seed

        if columns is None:
            columns = [
                "APP",
                "PPI_IPT", "PPI_DIRECTIONS", "PPI_SIZES",
                "BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV",
                "DURATION", "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS",
                "PHIST_SRC_SIZES", "PHIST_DST_SIZES",
                "PHIST_SRC_IPT", "PHIST_DST_IPT", "FLOW_ENDREASON_IDLE",
                "FLOW_ENDREASON_ACTIVE", "FLOW_ENDREASON_END",
                "FLOW_ENDREASON_OTHER",
            ]
        self.columns = columns

        # Load normalization statistics
        self.normalize = normalization_stats is not None
        if self.normalize:
            if isinstance(normalization_stats, (str, Path)):
                stats = np.load(normalization_stats)
            else:
                stats = normalization_stats
            self.ppi_mean = torch.tensor(stats['ppi_mean'], dtype=torch.float32)
            self.ppi_std = torch.tensor(stats['ppi_std'], dtype=torch.float32)
            self.flowstats_mean = torch.tensor(stats['flowstats_mean'], dtype=torch.float32)
            self.flowstats_std = torch.tensor(stats['flowstats_std'], dtype=torch.float32)
            assert self.ppi_mean.shape == (3, 30)
            assert self.ppi_std.shape == (3, 30)
            assert self.flowstats_mean.shape == (44,)
            assert self.flowstats_std.shape == (44,)

        self.dfs = []
        self.cum_lens = [0]

        rng = np.random.RandomState(self.seed)

        # Columns that may be stored as string-encoded lists
        list_columns = [
            "PPI_IPT", "PPI_DIRECTIONS", "PPI_SIZES",
            "PHIST_SRC_SIZES", "PHIST_DST_SIZES",
            "PHIST_SRC_IPT", "PHIST_DST_IPT",
        ]

        for f in self.files:
            table = pq.read_table(f, columns=self.columns)
            # table = table.slice(0, 1000)
            df = table.to_pandas()

            # ---------- column sanity ----------
            missing = set(self.columns) - set(df.columns)
            assert not missing, f"Missing columns in {f}: {missing}"

            # ---------- parse string-encoded list columns ----------
            for col in list_columns:
                if col in df.columns:
                    df[col] = _parse_list_column(df[col])

            # ---------- label sanity ----------
            df = df[df["APP"].isin(self.label_mapping)]
            assert len(df) > 0, f"No valid labels left after filtering in {f}"

            # ---------- optional sampling ----------
            if self.data_sample_frac is not None:
                file_seed = int(rng.randint(0, 2**31 - 1))
                df = df.sample(frac=self.data_sample_frac, random_state=file_seed)
                assert len(df) > 0, f"Sampling removed all rows in {f}"

            df = df.reset_index(drop=True)

            self.dfs.append(df)
            self.cum_lens.append(self.cum_lens[-1] + len(df))

        self.total_len = self.cum_lens[-1]
        assert self.total_len > 0, "Dataset is empty after processing all files"

    def __len__(self):
        return self.total_len

    def _pad(self, x, value=0.0):
        assert hasattr(x, "__len__"), "PPI element must be list-like"
        x = list(x)[: self.max_packets]
        if len(x) < self.max_packets:
            x = x + [value] * (self.max_packets - len(x))
        return x

    def __getitem__(self, idx):
        assert 0 <= idx < self.total_len, "Index out of bounds"

        file_i = bisect.bisect_right(self.cum_lens, idx) - 1
        local_i = idx - self.cum_lens[file_i]
        row = self.dfs[file_i].iloc[local_i]

        # ---------- PPI sanity ----------
        assert row["PPI_IPT"] is not None
        assert row["PPI_DIRECTIONS"] is not None
        assert row["PPI_SIZES"] is not None

        ipt = self._pad(row["PPI_IPT"], 0.0)
        directions = self._pad(row["PPI_DIRECTIONS"], 0.0)
        sizes = self._pad(row["PPI_SIZES"], 0.0)

        # lengths must match
        assert len(ipt) == len(directions) == len(sizes) == self.max_packets

        # Output shape: (C, T) = (3, max_packets) for Multimodal_CESNET compatibility
        pstats = torch.tensor(
            [ipt, directions, sizes],
            dtype=torch.float32
        )  # (3, max_packets)

        assert pstats.shape == (3, self.max_packets)

        # ---------- FLOWSTATS sanity ----------
        for col in (
            "PHIST_SRC_SIZES", "PHIST_DST_SIZES",
            "PHIST_SRC_IPT", "PHIST_DST_IPT"
        ):
            assert len(row[col]) == 8, f"{col} must have 8 bins, it has {len(row[col])}"

        flowstats = torch.tensor([
            row["BYTES"],
            row["BYTES_REV"],
            row["PACKETS"],
            row["PACKETS_REV"],
            row["DURATION"],
            row["PPI_LEN"],
            row["PPI_DURATION"],
            row["PPI_ROUNDTRIPS"],
            *row["PHIST_SRC_SIZES"],
            *row["PHIST_DST_SIZES"],
            *row["PHIST_SRC_IPT"],
            *row["PHIST_DST_IPT"],
            row["FLOW_ENDREASON_IDLE"],
            row["FLOW_ENDREASON_ACTIVE"],
            row["FLOW_ENDREASON_END"],
            row["FLOW_ENDREASON_OTHER"],

        ], dtype=torch.float32)

        assert flowstats.numel() == 44, f"FLOWSTATS must have 44 features, and not {flowstats.numel()}"

        # ---------- NORMALIZATION ----------
        if self.normalize:
            pstats = (pstats - self.ppi_mean) / (self.ppi_std + 1e-8)
            flowstats = (flowstats - self.flowstats_mean) / (self.flowstats_std + 1e-8)

        # ---------- LABEL ----------
        label = self.label_mapping[row["APP"]]
        assert isinstance(label, int)

        # Return ((ppi, flowstats), label) to match trainer expected format
        return (pstats, flowstats), torch.tensor(label, dtype=torch.long)


def create_parquet_loader(
    parquet_files,
    label_mapping,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    max_packets=30,
    data_sample_frac=None,
    seed=42,
    columns=None,
    normalization_stats=None,
    drop_last=True,
):
    assert batch_size > 0, "batch_size must be > 0"
    assert num_workers >= 0, "num_workers must be >= 0"

    dataset = ParquetCESNETDataset(
        parquet_files=parquet_files,
        label_mapping=label_mapping,
        max_packets=max_packets,
        data_sample_frac=data_sample_frac,
        seed=seed,
        columns=columns,
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
