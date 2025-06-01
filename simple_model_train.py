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
from training.utils import set_seed, save_config_to_json, get_df_from_csvs
from config import Config

def run_full_exp(cfg):
    """
    Run a full experiment with the given configuration.
    
    Parameters:
        cfg (Config): Configuration object containing all parameters for the experiment.
        
    Returns:
        float: The maximum validation accuracy achieved during training.
    """
    
    # Set random seed for reproducibility
        

    set_seed(42)

    label_whitelist = [386,  497,  998,  171,  485, 2613,  340,  373,  561,  967,  436, 1088,
            961,  682,  521,  964, 1450, 1448,  965, 42][:4]
    apps_id_df = pd.read_csv('/home/anatbr/students/noamshakedc/da4etc/data/app_id_mapping.csv', index_col=0)
    apps_id_df = apps_id_df[apps_id_df.index.isin(label_whitelist)]
    label_mapping = {row['names']: i for i, row in apps_id_df.reset_index().iterrows()}
    label_indices_mapping = {label_index: new_label_index for
                            new_label_index, (label_index, row)
                            in enumerate(apps_id_df.iterrows())
                            if label_index in label_whitelist}

    num_classes = len(label_mapping)

    # train_df_domain_1 = get_df_from_csvs(1, 0, 70, label_whitelist)
    # train_df_domain_2 = get_df_from_csvs(2, 0, 70, label_whitelist)
    # train_df_domain_3 = get_df_from_csvs(3, 0, 70, label_whitelist)
    train_df_domain_4 = get_df_from_csvs(4, 0, 70, label_whitelist)
    # test_df_domain_1 = get_df_from_csvs(1, 70, 100, label_whitelist)
    # test_df_domain_2 = get_df_from_csvs(2, 70, 100, label_whitelist)
    # test_df_domain_3 = get_df_from_csvs(3, 70, 100, label_whitelist)
    test_df_domain_4 = get_df_from_csvs(4, 70, 100, label_whitelist)


    train_df, test_df = train_df_domain_4, test_df_domain_4
        
    train_loader = create_csv_flowpic_loader([train_df], batch_size=cfg.BATCH_SIZE, num_workers=0,
                                            shuffle=True, resolution=cfg.RESOLUTION,
                                            label_mapping=label_indices_mapping, log_t_axis=False) 

    cfg.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()


    experiment_path = cfg.EXPERIMENT_PATH
    experiment_path.mkdir(parents=True, exist_ok=True)

    weights_save_dir = experiment_path / 'weights'
    plots_save_dir = experiment_path / 'plots'
    weights_save_dir.mkdir(parents=True, exist_ok=True)
    plots_save_dir.mkdir(parents=True, exist_ok=True)

    save_config_to_json(config_module=cfg, output_file_path=experiment_path / "config.json")

    test_loader = create_csv_flowpic_loader([test_df], batch_size=cfg.BATCH_SIZE, num_workers=0,
                                            shuffle=False, resolution=cfg.RESOLUTION,
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
        # adapt_batch_norm=cfg.ADAPT_BATCH_NORM,
    )

    final_model_path = weights_save_dir / f"model_final_000train_to_000test.pth"
    torch.save(model.state_dict(), final_model_path)
    
    training_history_dict = torch.load(experiment_path / 'plots' / 'training_history.pth', weights_only=False)
    max_val_accuracy = np.max(training_history_dict['val_accuracies'])
    return max_val_accuracy

if __name__ == "__main__":
    base_experiment_path = Path("exps") / "measureing_domain_shift"
    cfg = Config(BASE_EXPERIMENTS_PATH=base_experiment_path, EXPERIMENT_NAME='4_to_4', SEED=1234)
    max_val_accuracy = run_full_exp(cfg)
    print(f"Max validation accuracy: {max_val_accuracy:.4f}")
