import re
import warnings


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def extract_numbers(x):
    if isinstance(x, str):
        # If x is a string, extract numbers
        return list(map(int, re.findall(r'\d+', x)))
    elif isinstance(x, list):
        return x
    else:
        return []    

MTU = 1500  # Maximum Transmission Unit in bytes

def session_2d_histogram(ts, sizes, resolution=MTU, max_delta_time=10, log_t_axis=False):
    ts_array = np.array(ts)
    sizes_array = np.array(sizes)
    
    # Normalize timestamps
    ts_norm = ((ts_array - ts_array[0]) / max_delta_time) * MTU
    
    if log_t_axis:
        ts_norm = np.log1p(ts_norm)  # log(1 + x)
        log_max_time = np.log1p(MTU)
        bin_edges_x = np.linspace(0, log_max_time, resolution + 1)
    else:
        bin_edges_x = np.linspace(0, MTU, resolution + 1)
    
    bin_edges_y = np.linspace(0, MTU, resolution + 1)

    H, _, _ = np.histogram2d(sizes_array, ts_norm, bins=(bin_edges_y, bin_edges_x))
    return H.astype(np.uint16)
    
class CSVFlowPicDataset(Dataset):
    def __init__(self, csv_paths, resolution=MTU, max_dt_ms=30000, label_mapping=None, log_t_axis=False): 
        self.csv_paths = csv_paths
        self.resolution = resolution
        self.max_delta_time = max_dt_ms
        self.label_mapping = label_mapping
        self.log_t_axis = log_t_axis
        
        # Index all sessions at initialization for efficiency
        self.sessions = []
        self.labels = []
        self.rows = []
        self._prepare_index()
        
    def _prepare_index(self):
        for csv_file in tqdm(self.csv_paths):
            if isinstance(csv_file, pd.DataFrame):
                # If csv_file is already a DataFrame, use it directly
                df = csv_file.copy()
            else:
                df = pd.read_csv(csv_file)
                
            df['ppi-pdt'] = df['ppi-pdt'].transform(extract_numbers)
            df['ppi-pd'] = df['ppi-pd'].transform(extract_numbers)
            df['ppi-ps'] = df['ppi-ps'].transform(extract_numbers)
            df['ppi-paux'] = df['ppi-paux'].transform(extract_numbers)
            
            for _, row in df.iterrows():
                ts = row['ppi-pdt']
                if len(ts) >= 1:
                    if self.label_mapping is not None:
                        app_id = row['appId']
                        if app_id in self.label_mapping:
                            self.labels.append(self.label_mapping[app_id])
                        else:
                            warnings.warn(f"App ID {app_id} not found in label mapping.")
                            continue
                    else:     
                        self.labels.append(row['appId'])
                    self.sessions.append((ts, row['ppi-ps']))
                    self.rows.append(row)

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        ts, sizes = self.sessions[idx]
        flowpic = session_2d_histogram(ts, sizes, self.resolution, self.max_delta_time, log_t_axis=self.log_t_axis)

        flowpic_tensor = torch.tensor(flowpic, dtype=torch.float32) # .unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # can get other features from self.rows[idx] if needed

        return flowpic_tensor, label_tensor


def create_csv_flowpic_loader(csv_paths, batch_size=64, shuffle=True, num_workers=4,
                              resolution=MTU, max_dt_ms=30000, label_mapping=None, log_t_axis=False):
    dataset = CSVFlowPicDataset(csv_paths, resolution=resolution, max_dt_ms=max_dt_ms,
                                label_mapping=label_mapping, log_t_axis=log_t_axis)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
