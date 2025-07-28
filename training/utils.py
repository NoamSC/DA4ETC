import random
import torch
import json
from pathlib import Path
import inspect
import os
from datetime import datetime, timedelta
from collections import defaultdict
import re

import pandas as pd
import numpy as np
from tqdm import tqdm

def group_chunks_by_interval(parent_dir, interval, start_date=None):
    """
    Groups chunk_*.csv files by time intervals.

    Args:
        parent_dir: Path to the parent directory (contains domain_* subdirs)
        interval: A datetime.timedelta object (e.g., timedelta(days=1))
        start_date: Optional datetime to align grouping intervals. Defaults to datetime.min.

    Returns:
        A dict mapping interval start datetime -> list of Path objects
    """
    chunk_pattern = re.compile(r"chunk_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.csv")
    result = defaultdict(list)

    anchor = start_date if start_date else datetime.min
    total_seconds = interval.total_seconds()

    for subdir in parent_dir.glob("domain_*"):
        if subdir.is_dir():
            for file in subdir.glob("chunk_*.csv"):
                match = chunk_pattern.match(file.name)
                if match:
                    dt = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M")
                    delta = (dt - anchor).total_seconds()
                    aligned_seconds = (delta // total_seconds) * total_seconds
                    aligned_dt = anchor + timedelta(seconds=aligned_seconds)
                    result[aligned_dt].append(file)

    return dict(result)



# def get_df_from_csvs(domain_idx, start_time_str, end_time_str, label_whitelist, sample_frac=None, verbose=False):
#     """
#     Load and concatenate CSV chunks from a specific domain and time range.

#     Parameters:
#         domain_idx (int): Index of the domain folder.
#         start_time_str (str): Start time, e.g. '2024-09-05 10:00'.
#         end_time_str (str): End time, e.g. '2024-09-06 10:00'.
#         label_whitelist (list): List of allowed appId values.
#         sample_frac (float, optional): If set (e.g., 0.1), sample that fraction from each individual CSV.
#         verbose (bool): If True, shows progress bar.

#     Returns:
#         pd.DataFrame: Concatenated and optionally subsampled DataFrame.
#     """
#     start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
#     end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M")
#     domain_path = f"../../../dataset/Allot/allot_hourly_chunks/domain_{domain_idx}"
    
#     dfs = []
#     filenames = [f for f in os.listdir(domain_path) if f.endswith('.csv')]

#     iterator = tqdm(filenames, desc="Reading chunks") if verbose else filenames

#     for filename in iterator:
#         try:
#             timestamp_str = filename.split("chunk_")[1].replace(".csv", "")
#             file_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M")
#         except Exception:
#             continue

#         if start_time <= file_time < end_time:
#             file_path = os.path.join(domain_path, filename)
#             df = pd.read_csv(file_path)
#             df = df[df['appId'].isin(label_whitelist)]
#             if sample_frac is not None and 0 < sample_frac < 1:
#                 df = df.sample(frac=sample_frac)
#             dfs.append(df)
    
#     if dfs:
#         df = pd.concat(dfs, ignore_index=True)
#         df = df.sample(frac=1).reset_index(drop=True)
#     else:
#         df = pd.DataFrame()
    
#     return df



def set_seed(seed):
    """
    Set random seed for reproducibility across libraries.
    
    Args:
    - seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_config_to_json(config_module, output_file_path):
    """
    Save all attributes of a config module to a JSON file.
    
    Parameters:
        config_module (module): The module containing configuration variables.
        output_file (str): The path to the output JSON file.
    """
    # Helper function to serialize non-serializable objects to strings
    def serialize_value(value):
        if isinstance(value, (Path, torch.device, type)):
            return str(value)  # Convert Path and device objects to strings
        return value  # Leave other types unchanged

    # Create a dictionary from the module's attributes
    config_dict = {}
    for name, value in inspect.getmembers(config_module):
        if not name.startswith("__") and not inspect.ismodule(value) and not inspect.isfunction(value):
            config_dict[name] = serialize_value(value)

    # Save to a JSON file
    with open(output_file_path, "w") as f:
        json.dump(config_dict, f, indent=4)

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a checkpoint file.
    
    Args:
    - model (torch.nn.Module): The model to load weights into.
    - checkpoint_path (str or Path): The path to the checkpoint file.
    - device (str or torch.device): The device to load the model on.
    
    Returns:
    - model (torch.nn.Module): The model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model

def count_model_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
    - model (torch.nn.Module): The model to count parameters for.
    
    Returns:
    - int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
