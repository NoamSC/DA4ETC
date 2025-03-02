from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

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
        lambda_mmd=cfg.LAMBDA_MMD,
    )

    final_model_path = weights_save_dir / f"model_final_{train_domain}_to_{test_domain}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Model for train domain {train_domain} to test domain {test_domain} saved to {final_model_path}")

def adapt_batch_norm_statistics(model, loader, device):
    """
    Update the batch normalization statistics of a model using a given DataLoader.
    """
    model.train()  # Enable BatchNorm stats updating
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            model(inputs)  # Forward pass updates BatchNorm stats
    model.eval()


def load_cached_val_loader(location):
    """
    Load a cached validation dataset for a specific location.
    """
    dataset_path = cfg.DATA_PATH / f"cached_datasets/datasets_{location}_256.pkl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Validation dataset not found at: {dataset_path}")
    print(f"Loading validation dataset from: {dataset_path}")
    
    with open(dataset_path, "rb") as f:
        _, val_dataset = pickle.load(f)

    return DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)


def find_latest_model_path(train_loc, test_loc, epoch='*'):
    """
    Find the latest model for the given train-test pair.
    """
    train_dir = cfg.EXPERIMENT_PATH / f"{train_loc}_to_{test_loc}"
    weights_dir = train_dir / 'weights'
    train_metrics_path = train_dir / 'plots' / 'training_history.pth'
    train_metrics = torch.load(train_metrics_path)

    if not weights_dir.exists():
        return None

    if epoch == 'latest':
        # choosing latest
        model_files = sorted(weights_dir.glob(f"model_weights_epoch_{epoch}.pth"))
        return model_files[-1] if model_files else None
    elif epoch == 'best':
        # epochs we count from 1
        best_epoch = np.argmax(train_metrics['val_accuracies']) + 1
        return weights_dir / f"model_weights_epoch_{best_epoch}.pth"
    else:
        model_files = sorted(weights_dir.glob(f"model_weights_epoch_{epoch}.pth"))
        assert len(model_files) > 0, "No such epoch"
        return model_files[-1] if model_files else None


def evaluate_model_on_loader(model, val_loader, device):
    """
    Evaluate a model on a given DataLoader and return the accuracy.
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0

if __name__ == "__main__":
    cfg.EXPERIMENT_PATH.mkdir(parents=True, exist_ok=False)
    set_seed(cfg.SEED)

    locations = [
        'AwsCont', 'BenContainer',  'CabSpicy1',
        'HujiPC', 'TLVunContainer1', 'TLVunContainer2'
    ]
    
    # Train models
    for train_domain in locations:
        for test_domain in locations:
            # if train_domain != test_domain:
            print(f"Running MMD experiment for train domain: {train_domain} and test domain: {test_domain}")
            run_experiment_with_mmd(train_domain, test_domain)

    # Evaluate performance
    num_locations = len(locations)
    accuracy_matrix = np.zeros((num_locations, num_locations))

    # Load validation DataLoaders
    print("Loading validation loaders for all locations...")
    val_loaders = {loc: load_cached_val_loader(loc) for loc in locations}

    # Iterate over train-test pairs
    for i, train_loc in enumerate(locations):
        for j, test_loc in enumerate(locations):
            print(f"Evaluating train domain: {train_loc}, test domain: {test_loc}")
            
            # Find latest model for train-test pair
            model_path = find_latest_model_path(train_loc, test_loc, epoch='best')
            if model_path is None:
                print(f"  No model found for {train_loc} -> {test_loc}. Setting accuracy to 0.")
                accuracy_matrix[i, j] = 0
                continue
            
            print(f"  Loading model weights from: {model_path}")
            model = ConfigurableCNN(cfg.MODEL_PARAMS)
            model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
            model.to(cfg.DEVICE)

            # Adapt to test domain
            # print(f"  Adapting BatchNorm statistics for test domain: {test_loc}")
            # adapt_batch_norm_statistics(model, val_loaders[test_loc], cfg.DEVICE)
            
            # Evaluate model
            accuracy = evaluate_model_on_loader(model, val_loaders[test_loc], cfg.DEVICE)
            accuracy_matrix[i, j] = accuracy
            print(f"  Accuracy: {accuracy:.4f}")

    # Save accuracy matrix
    print("\nCross-Domain Accuracy Matrix:")
    print(accuracy_matrix)
    np.save(cfg.EXPERIMENT_PATH / "cross_domain_accuracy_matrix.npy", accuracy_matrix)

    # Plot accuracy matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(accuracy_matrix, cmap='viridis', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(label='Accuracy')
    for i in range(num_locations):
        for j in range(num_locations):
            plt.text(j, i, f'{accuracy_matrix[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
    plt.xticks(ticks=np.arange(num_locations), labels=locations, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(num_locations), labels=locations)
    plt.title('Cross-Domain Accuracy Matrix')
    plt.xlabel('Test Domain')
    plt.ylabel('Train Domain')
    plt.tight_layout()
    plt.savefig(cfg.EXPERIMENT_PATH / 'cross_domain_accuracy_matrix.png', dpi=300)