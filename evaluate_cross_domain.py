import torch
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.configurable_cnn import ConfigurableCNN
from training.utils import set_seed
import config as cfg


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
    weights_dir = cfg.EXPERIMENT_PATH / f"{train_loc}_to_{test_loc}" / "weights"
    if not weights_dir.exists():
        return None

    model_files = sorted(weights_dir.glob(f"model_weights_epoch_{epoch}.pth"))
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


def main():
    # Set random seed for reproducibility
    set_seed(cfg.SEED)

    # Define train and test locations
    locations = [
        'AwsCont', 'BenContainer', 'CabSpicy1',
        'HujiPC', 'TLVunContainer1', 'TLVunContainer2'
    ]

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
            model_path = find_latest_model_path(train_loc, test_loc, epoch=5)
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


if __name__ == "__main__":
    main()
