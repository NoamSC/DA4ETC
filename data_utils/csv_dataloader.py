import re
import warnings
import ast


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

def safe_parse_list(x):
    """Safely parse a string representation of a list."""
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []

    # Check if it looks like a list structure
    x = x.strip()
    if not (x.startswith('[') and x.endswith(']')):
        return []

    try:
        result = ast.literal_eval(x)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass

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
    def __init__(self, csv_paths, resolution=MTU, max_dt_ms=30000, label_mapping=None,
                 log_t_axis=False, verbose=False, data_sample_frac=None, seed=42,
                 dataset_format='auto'):
        self.csv_paths = csv_paths
        self.resolution = resolution
        self.max_delta_time = max_dt_ms
        self.label_mapping = label_mapping
        self.log_t_axis = log_t_axis
        self.verbose = verbose
        self.data_sample_frac = data_sample_frac
        self.seed = seed
        self.dataset_format = dataset_format  # 'auto', 'mirage', or 'cesnet'

        # Index all sessions at initialization for efficiency
        self.sessions = []
        self.labels = []
        self.rows = []
        self._prepare_index()

    def _prepare_index(self):
        files = self.csv_paths
        if self.verbose:
            files = tqdm(self.csv_paths, desc="Loading files")
        for file_path in files:
            if isinstance(file_path, pd.DataFrame):
                # If file_path is already a DataFrame, use it directly
                df = file_path.copy()
            else:
                # Read file based on extension
                file_str = str(file_path)
                if file_str.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_str.endswith('.csv.xz'):
                    df = pd.read_csv(file_path, compression='xz')
                elif file_str.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    # Infer compression
                    df = pd.read_csv(file_path, compression='infer')

            if self.data_sample_frac is not None:
                df = df.sample(frac=self.data_sample_frac, random_state=self.seed)

            # Auto-detect format based on columns
            format_type = self.dataset_format
            if format_type == 'auto':
                if 'PPI_IPT' in df.columns and 'APP' in df.columns:
                    format_type = 'cesnet_parquet'
                elif 'PPI' in df.columns and 'APP' in df.columns:
                    format_type = 'cesnet'
                elif 'ppi-pdt' in df.columns and 'appId' in df.columns:
                    format_type = 'mirage'
                else:
                    raise ValueError(f"Cannot auto-detect dataset format. Columns: {df.columns.tolist()}")

            if format_type == 'cesnet':
                self._process_cesnet_format(df)
            elif format_type == 'cesnet_parquet':
                self._process_cesnet_parquet_format(df)
            elif format_type == 'mirage':
                self._process_mirage_format(df)
            else:
                raise ValueError(f"Unknown dataset format: {format_type}")

            # Update progress bar with total samples loaded
            if self.verbose and isinstance(files, tqdm):
                files.set_postfix({'total_samples': len(self.sessions)})

    def _process_mirage_format(self, df):
        """Process Mirage dataset format (original format)."""
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
                        continue
                else:
                    self.labels.append(row['appId'])
                self.sessions.append((ts, row['ppi-ps']))
                self.rows.append(row)

    def _process_cesnet_format(self, df):
        """Process CESNET dataset format (CSV with unparsed PPI)."""
        assert self.label_mapping is not None, "label_mapping must be provided for CESNET dataset"

        # Parse PPI column which contains [ipt, directions, sizes, aux]
        df['PPI'] = df['PPI'].transform(safe_parse_list)

        for _, row in df.iterrows():
            ppi = row['PPI']

            # PPI format: [ipt_list, direction_list, size_list, aux_list]
            if not isinstance(ppi, list) or len(ppi) < 3:
                continue

            ts = ppi[0]  # inter-packet times
            sizes = ppi[2]  # packet sizes

            # Skip if empty
            if not isinstance(ts, list) or not isinstance(sizes, list) or len(ts) < 1:
                continue

            # Get label
            app_id = row['APP']
            if app_id in self.label_mapping:
                self.labels.append(self.label_mapping[app_id])
            else:
                continue

            self.sessions.append((ts, sizes))
            self.rows.append(row)

    def _process_cesnet_parquet_format(self, df):
        """Process CESNET dataset format (Parquet with pre-parsed PPI)."""
        assert self.label_mapping is not None, "label_mapping must be provided for CESNET dataset"

        for _, row in df.iterrows():
            ts = row['PPI_IPT']
            sizes = row['PPI_SIZES']

            # Accept both lists and numpy arrays
            # Skip if empty or invalid (check for None, empty, or wrong type)
            if ts is None or sizes is None:
                continue
            if not hasattr(ts, '__len__') or not hasattr(sizes, '__len__'):
                continue
            if len(ts) < 1:
                continue

            # Get label
            app_id = row['APP']
            if app_id in self.label_mapping:
                self.labels.append(self.label_mapping[app_id])
            else:
                continue

            self.sessions.append((ts, sizes))
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
    
    def get_class_counts(self):
        """Return a dictionary with class_id as key and count as value."""
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts


def create_csv_flowpic_loader(csv_paths, batch_size=64, shuffle=True, num_workers=4,
                              data_sample_frac=None, seed=42,
                              resolution=MTU, max_dt_ms=30000, label_mapping=None, log_t_axis=False,
                              dataset_format='auto', verbose=False):
    dataset = CSVFlowPicDataset(csv_paths, resolution=resolution, max_dt_ms=max_dt_ms,
                                label_mapping=label_mapping, log_t_axis=log_t_axis,
                                data_sample_frac=data_sample_frac, seed=seed,
                                dataset_format=dataset_format,
                                verbose=verbose)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
