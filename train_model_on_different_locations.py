import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pickle
import config as cfg
from torch.utils.data import DataLoader
from models.configurable_cnn import ConfigurableCNN
from training.trainer import train_and_validate
from training.utils import set_seed, save_config_to_json

def load_cached_dataset(location, path_format="cached_datasets/datasets_{location}_256.pkl"):
    """
    Load a cached dataset for a specific location.

    Args:
    - location (str): The location whose dataset should be loaded.

    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - label_mapping: Dictionary mapping labels to indices.
    """
    dataset_path = cfg.DATA_PATH / path_format.format(location=location)
    print(f"Loading cached dataset from: {dataset_path}")
    
    with open(dataset_path, "rb") as f:
        train_dataset, val_dataset = pickle.load(f)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

def run_experiment(location):
    """
    Train and validate a model for a specific location.

    Args:
    - location (str): The location to use for training and validation.
    """
    
    # Load cached dataset and prepare DataLoaders
    train_loader, val_loader = load_cached_dataset(location)
    label_mapping = cfg.LABEL_MAPPING
    num_classes = len(label_mapping)
    cfg.MODEL_PARAMS['num_classes'] = num_classes

    # Initialize the model
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Define experiment-specific paths
    experiment_path = cfg.EXPERIMENT_PATH / location
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    weights_save_dir = experiment_path / 'weights'
    plots_save_dir = experiment_path / 'plots'
    weights_save_dir.mkdir(parents=True, exist_ok=True)
    plots_save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for this experiment
    save_config_to_json(config_module=cfg, output_file_path=experiment_path / "config.json")

    # Train and validate

    train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.NUM_EPOCHS,
        device=cfg.DEVICE,
        weights_save_dir=weights_save_dir,
        plots_save_dir=plots_save_dir,
        label_mapping=label_mapping,
    )

    # Save final model checkpoint
    final_model_path = weights_save_dir / f"model_final_{location}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model for location {location} saved to {final_model_path}")

if __name__ == "__main__":
    # Initialize experiment directory
    cfg.EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    set_seed(cfg.SEED)

    # Define the locations to test
    locations = [
        'AwsCont', 'BenContainer', 'CabSpicy1',
        'GCP-Iowa', 'HujiPC', 'TLVunContainer1', 'TLVunContainer2'
    ]

    # Run experiments for each location sequentially
    for location in locations:
        print(f"Running experiment for location: {location}")
        run_experiment(location)
