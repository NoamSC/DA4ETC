import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pickle
import config as cfg
from torch.utils.data import DataLoader
from models.configurable_cnn import ConfigurableCNN
from training.trainer import train_model
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

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def run_experiment_with_mmd(train_domain, test_domain):
    """
    Train and validate a model for a specific train-test domain pair using MMD.

    Args:
    - train_domain (str): The domain to use for training.
    - test_domain (str): The domain to use for testing (MMD computation).
    """
    train_loader, _ = load_cached_dataset(train_domain)
    _, test_loader = load_cached_dataset(test_domain)
    label_mapping = cfg.LABEL_MAPPING
    num_classes = len(label_mapping)
    cfg.MODEL_PARAMS['num_classes'] = num_classes

    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    experiment_path = cfg.EXPERIMENT_PATH / f"{train_domain}_to_{test_domain}"
    experiment_path.mkdir(parents=True, exist_ok=True)

    weights_save_dir = experiment_path / 'weights'
    plots_save_dir = experiment_path / 'plots'
    weights_save_dir.mkdir(parents=True, exist_ok=True)
    plots_save_dir.mkdir(parents=True, exist_ok=True)

    save_config_to_json(config_module=cfg, output_file_path=experiment_path / "config.json")

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
        lambda_mmd=1e2,
    )

    final_model_path = weights_save_dir / f"model_final_{train_domain}_to_{test_domain}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model for train domain {train_domain} to test domain {test_domain} saved to {final_model_path}")

if __name__ == "__main__":
    cfg.EXPERIMENT_PATH.mkdir(parents=True, exist_ok=False)
    set_seed(cfg.SEED)

    domains = [
        'AwsCont', 'BenContainer', 'CabSpicy1',
        'GCP-Iowa', 'HujiPC', 'TLVunContainer1', 'TLVunContainer2'
    ]
    

    for train_domain in domains:
        for test_domain in domains:
            # if train_domain != test_domain:
            print(f"Running MMD experiment for train domain: {train_domain} and test domain: {test_domain}")
            run_experiment_with_mmd(train_domain, test_domain)
