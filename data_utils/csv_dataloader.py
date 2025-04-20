import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

MTU = 1500  # Maximum Transmission Unit in bytes

def session_2d_histogram(ts, sizes, resolution=MTU, max_delta_time=10):
    ts_norm = ((np.array(ts) - ts[0]) / max_delta_time) * MTU
    bin_edges = np.linspace(0, MTU, resolution + 1)
    H, _, _ = np.histogram2d(sizes, ts_norm, bins=(bin_edges, bin_edges))
    return H.astype(np.uint16)
    
class CSVFlowPicDataset(Dataset):
    def __init__(self, csv_paths, resolution=MTU, max_dt_ms=30000):
        self.csv_paths = csv_paths
        self.resolution = resolution
        self.max_delta_time = max_dt_ms
        
        # Index all sessions at initialization for efficiency
        self.sessions = []
        self.labels = []
        self.rows = []
        self._prepare_index()
        
    def _prepare_index(self):
        for csv_file in self.csv_paths:
            df = pd.read_csv(csv_file)
            df['ppi-pdt'] = df['ppi-pdt'].apply(lambda x: list(map(int, re.findall(r'\d+', x))))
            df['ppi-pd'] = df['ppi-pd'].apply(lambda x: list(map(int, re.findall(r'\d+', x))))
            df['ppi-ps'] = df['ppi-ps'].apply(lambda x: list(map(int, re.findall(r'\d+', x))))
            df['ppi-paux'] = df['ppi-paux'].apply(lambda x: list(map(int, re.findall(r'\d+', x))))
            
            for _, row in df.iterrows():
                ts = row['ppi-pdt']
                if len(ts) >= 1:
                    self.sessions.append((ts, row['ppi-ps']))
                    self.labels.append(row['appId'])
                    self.rows.append(row)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        ts, sizes = self.sessions[idx]
        flowpic = session_2d_histogram(ts, sizes, self.resolution, self.max_delta_time)

        flowpic_tensor = torch.tensor(flowpic, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # can get other features from self.rows[idx] if needed

        return flowpic_tensor, label_tensor


def create_csv_flowpic_loader(csv_paths, batch_size=64, shuffle=True, num_workers=4, resolution=MTU, max_dt_ms=30000):
    dataset = CSVFlowPicDataset(csv_paths, resolution=resolution, max_dt_ms=max_dt_ms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
