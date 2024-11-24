import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

import config as cfg
from data_utils.pcap_dataloader import PcapDataLoader
from models.configurable_cnn import ConfigurableCNN
from training.trainer import train_and_validate
from training.utils import set_seed, save_config_to_json

def load_and_prepare_data():
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
    df = df[df.location == 'TLVunContainer1'].sample(frac=cfg.SAMPLE_FRAC).sample(1500).sort_values(by='date')
    
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


def extract_pcap_info(path):
    """
    Extract metadata (location, date, app, vpn type) from PCAP file path.

    Args:
    - path (str): Path to the PCAP file.

    Returns:
    - Tuple (location, date, app, vpn_type).
    """
    parts = Path(path).parts
    location, date, app, vpn_type = parts[3], pd.to_datetime(parts[4], format='%Y%m%d_%H%M%S'), parts[5], parts[6]
    return location, date, app, vpn_type


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
    train_loader, val_loader, label_mapping = load_and_prepare_data()
    num_classes = len(label_mapping)
    cfg.MODEL_PARAMS['num_classes'] = num_classes

    # Initialize the model
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
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
        save_dir=cfg.EXPERIMENT_PATH,
        num_classes=num_classes
    )

    # Save final model checkpoint
    if cfg.SAVE_MODEL_CHECKPOINT:
        torch.save(model.state_dict(), cfg.SAVE_MODEL_CHECKPOINT)
        print(f"Model checkpoint saved to {cfg.SAVE_MODEL_CHECKPOINT}")