import os
import tempfile
import torch
from pathlib import Path
from csv import writer
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from data_utils.csv_dataloader import create_csv_flowpic_loader
from models.configurable_cnn import ConfigurableCNN
from training.trainer import train_model
from training.utils import set_seed, save_config_to_json
from config import Config

label_whitelist = [386,  497,  998,  171,  485, 2613,  340,  373,  561,  967,  436, 1088,
        961,  682,  521,  964, 1450, 1448,  965, 42]
apps_id_df = pd.read_csv('/home/anatbr/dataset/Allot/ver02 - Location 01 - 2024.09.12-17_AppIds G Z.csv', index_col=0)
# apps_id_df = apps_id_df[apps_id_df.index.isin(label_whitelist)]
label_mapping = {row['names']: i for i, row in apps_id_df.reset_index().iterrows()}
label_indices_mapping = {label_index: new_label_index for
                         new_label_index, (label_index, row)
                         in enumerate(apps_id_df.iterrows())
                         if label_index in label_whitelist}

num_classes = len(label_mapping)
cfg = Config(RESOLUTION=64)



chunk_path = os.path.join('data', 'allot_small_csvs', 'chunks', 'chunk_000.csv')


for i in range(0, 201, 25):
    train_loader = create_csv_flowpic_loader([chunk_path], batch_size=64, num_workers=0,
                                         shuffle=False, resolution=64,
                                         label_mapping=label_indices_mapping, log_t_axis=False) 
    
    cfg.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()


    experiment_path = cfg.EXPERIMENT_PATH / f"000_to_{i:003d}"
    experiment_path.mkdir(parents=True, exist_ok=True)

    weights_save_dir = experiment_path / 'weights'
    plots_save_dir = experiment_path / 'plots'
    weights_save_dir.mkdir(parents=True, exist_ok=True)
    plots_save_dir.mkdir(parents=True, exist_ok=True)

    save_config_to_json(config_module=cfg, output_file_path=experiment_path / "config.json")
    
    chunk_path = os.path.join('data', 'allot_small_csvs', 'chunks', f'chunk_{i:03d}.csv')
    test_loader = create_csv_flowpic_loader([chunk_path], batch_size=64, num_workers=0,
                                            shuffle=False, resolution=64,
                                            label_mapping=label_indices_mapping, log_t_axis=False)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.NUM_EPOCHS,
        device=cfg.DEVICE,
        weights_save_dir=weights_save_dir,
        plots_save_dir=plots_save_dir,
        label_mapping=label_mapping,
        lambda_mmd=cfg.LAMBDA_MMD,
        mmd_bandwidths=cfg.MMD_BANDWIDTHS,
        lambda_dann=cfg.LAMBDA_DANN,
    )

    final_model_path = weights_save_dir / f"model_final_001_to_{i:03d}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model for train domain train_domain to test domain {i:03d} saved to {final_model_path}")
