import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


def session_2d_histogram(ts, sizes, resolution=256, max_delta_time=None, min_packet_count=0):
    if max_delta_time is None:
        max_delta_time = ts[-1] - ts[0] if len(ts) > 1 else 1e-3
    if max_delta_time == 0:
        max_delta_time = 1e-3  # prevent divide-by-zero

    ts_norm = ((ts - ts[0]) / max_delta_time) * resolution
    bin_edges = np.linspace(0, resolution, resolution + 1)
    H, _, _ = np.histogram2d(sizes, ts_norm, bins=(bin_edges, bin_edges))
    return H.astype(np.uint16)


class FlowPicCSVDataset(Dataset):
    def __init__(self, csv_path, resolution=256, label_mapping=None, max_rows=None):
        self.csv_path = Path(csv_path)
        self.resolution = resolution
        self.line_offsets = []
        self.labels = []
        self.label_mapping = label_mapping or {}
        self.headers = []
        label_counter = len(self.label_mapping)
        if max_rows is not None:
            self.max_rows = max_rows
        else:
            self.max_rows = float('inf')
            
        with open(self.csv_path, 'r') as f:
            self.headers = f.readline().strip().split(',')
            offset = f.tell()
            buffer = ""
            while True:
                line = f.readline()
                if not line:
                    break
                buffer += line
                try:
                    reader = csv.reader([buffer])
                    parsed = next(reader)
                    row = dict(zip(self.headers, parsed))
                    app_id = int(row['appId'])
                    if app_id not in self.label_mapping:
                        self.label_mapping[app_id] = label_counter
                        label_counter += 1
                    self.labels.append(self.label_mapping[app_id])
                    self.line_offsets.append(offset)
                    offset = f.tell()
                    buffer = ""
                    row_count = len(self.line_offsets)
                    if row_count >= self.max_rows:
                        break
                except Exception:
                    continue  # keep reading until we get a complete row

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        try:
            with open(self.csv_path, 'r') as f:
                f.seek(self.line_offsets[idx])
                buffer = ""
                while True:
                    line = f.readline()
                    if not line:
                        break
                    buffer += line
                    try:
                        reader = csv.reader([buffer])
                        parsed = next(reader)
                        break
                    except Exception:
                        continue  # keep reading until complete

            row = dict(zip(self.headers, parsed))

            ts_str = row['ppi-pdt'].strip('[]')
            sizes_str = row['ppi-ps'].strip('[]')

            ts = np.fromstring(ts_str, sep=' ') if ts_str else np.array([])
            sizes = np.fromstring(sizes_str, sep=' ') if sizes_str else np.array([])

            if len(ts) < 2 or len(sizes) != len(ts):
                raise ValueError("Invalid flow data")

            flowpic = session_2d_histogram(ts, sizes, resolution=self.resolution)
            label = self.labels[idx]

            return torch.tensor(flowpic, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error reading index {idx}: {e}")
            dummy = np.zeros((self.resolution, self.resolution), dtype=np.float32)
            return torch.tensor(dummy), torch.tensor(-1)
        