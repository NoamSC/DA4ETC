import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

import config as cfg
from data_utils.pcap_dataloader import PcapDataLoader
from data_utils.data_utils import extract_pcap_info
from models.configurable_cnn import ConfigurableCNN
from training.trainer import train_and_validate
from training.utils import set_seed, save_config_to_json

def load_and_prepare_data(location):
    """
    Load PCAP paths, extract metadata, and create train/validation DataLoaders.

    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - label_mapping: Dictionary mapping labels to indices.
    """
    # Load PCAP file paths
    # pcaps = list(cfg.DATA_PATH.glob('**/*.pcap'))
    pcaps = pd.read_csv('pcap_paths.csv').values[:, 1]
    
    # Extract metadata from PCAP paths
    df = pd.DataFrame(
        [(str(p), *extract_pcap_info(p)) for p in pcaps],
        columns=['pcap_path', 'location', 'date', 'app', 'vpn_type']
    )
    
    # Filter and sample data
    df = df[df.location == location].sample(frac=cfg.SAMPLE_FRAC).sample(1500).sort_values(by='date')
    
    # Split into train/validation
    split_index = int(len(df) * cfg.TRAIN_SPLIT_RATIO)
    df_train, df_val = df[:split_index], df[split_index:]
    
    # Create label mapping
    unique_labels = df['app'].unique()
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # Create DataLoaders
    train_loader = PcapDataLoader(
        df_train['pcap_path'], df_train['app'],
        batch_size=cfg.BATCH_SIZE, shuffle=True, min_flow_length=cfg.MIN_FLOW_LENGTH,
        resolution=cfg.RESOLUTION, label_mapping=label_mapping
    )
    val_loader = PcapDataLoader(
        df_val['pcap_path'], df_val['app'],
        batch_size=cfg.BATCH_SIZE, shuffle=False, min_flow_length=cfg.MIN_FLOW_LENGTH,
        resolution=cfg.RESOLUTION, label_mapping=label_mapping
    )
    
    return train_loader, val_loader, label_mapping

if __name__ == "__main__":
    # Initialize directories for experiment
    cfg.EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)
    cfg.EXPERIMENT_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    cfg.EXPERIMENT_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config_to_json(config_module=cfg, output_file_path=cfg.EXPERIMENT_PATH / "config.json")

    # Set random seed
    set_seed(cfg.SEED)

    # Load and prepare data
    print("Loading and preparing data...")
    train_loader, val_loader, label_mapping = load_and_prepare_data(location=cfg.LOCATION)
    num_classes = len(label_mapping)
    cfg.MODEL_PARAMS['num_classes'] = num_classes

    # Initialize the model
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)

    # Set up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Start training
    print("Starting training...")
    train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.NUM_EPOCHS,
        device=cfg.DEVICE,
        weights_save_dir=cfg.EXPERIMENT_PLOTS_PATH,
        plots_save_dir=cfg.EXPERIMENT_PLOTS_PATH,
        num_classes=num_classes
    )

    # Save final model checkpoint
    if cfg.SAVE_MODEL_CHECKPOINT:
        torch.save(model.state_dict(), cfg.SAVE_MODEL_CHECKPOINT)
        print(f"Model checkpoint saved to {cfg.SAVE_MODEL_CHECKPOINT}")