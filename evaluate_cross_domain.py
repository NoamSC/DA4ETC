import torch
from pathlib import Path
import pickle
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from models.configurable_cnn import ConfigurableCNN
from training.utils import set_seed
import config as cfg

def load_cached_val_loader(location):
    """
    Load a cached validation dataset for a specific location.

    Args:
    - location (str): The location whose validation dataset should be loaded.

    Returns:
    - val_loader: DataLoader for validation data.
    """
    dataset_path = cfg.DATA_PATH / f"cached_datasets/datasets_{location}_256.pkl"
    print(f"Loading validation dataset from: {dataset_path}")
    
    with open(dataset_path, "rb") as f:
        _, val_dataset = pickle.load(f)

    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    return val_loader

def evaluate_model_on_loader(model, val_loader, device):
    """
    Evaluate a model on a given DataLoader and return the accuracy.

    Args:
    - model (torch.nn.Module): Trained model to evaluate.
    - val_loader (DataLoader): Validation DataLoader to test the model on.
    - device (torch.device): Device to run the evaluation.

    Returns:
    - accuracy (float): Accuracy of the model on the validation dataset.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def main():
    # Set random seed for reproducibility
    set_seed(cfg.SEED)

    final_models_epochs = {'AwsCont': 4, 'BenContainer': 4, 'CabSpicy1': 4,
        'GCP-Iowa': 4, 'HujiPC': 4, 'TLVunContainer1': 4, 'TLVunContainer2': 4

    }

    # Define locations and initialize accuracy matrix
    locations = [
        'AwsCont', 'BenContainer', 'CabSpicy1',
        'GCP-Iowa', 'HujiPC', 'TLVunContainer1', 'TLVunContainer2'
    ]
    num_locations = len(locations)
    accuracy_matrix = np.zeros((num_locations, num_locations))

    # Load all validation DataLoaders
    print("Loading validation loaders for all locations...")
    val_loaders = {loc: load_cached_val_loader(loc) for loc in locations}

    # Iterate through each model and evaluate on all validation datasets
    for i, train_loc in enumerate(locations):
        print(f"\nEvaluating model trained on {train_loc}")
        
        # Load the model weights for the current training location
        chosen_epoch = final_models_epochs[train_loc]
        model_path = cfg.EXPERIMENT_PATH / train_loc / "weights" / f"model_weights_epoch_{chosen_epoch}.pth"
        print(f"Loading model weights from: {model_path}")
        
        # Initialize model
        model = ConfigurableCNN(cfg.MODEL_PARAMS)
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        model.to(cfg.DEVICE)

        # Evaluate on each validation dataset
        for j, eval_loc in enumerate(locations):
            print(f"  Testing on validation dataset from {eval_loc}")
            val_loader = val_loaders[eval_loc]
            accuracy = evaluate_model_on_loader(model, val_loader, cfg.DEVICE)
            accuracy_matrix[i, j] = accuracy
            print(f"    Accuracy: {accuracy:.4f}")

    # Save and print the accuracy matrix
    print("\nCross-Domain Accuracy Matrix:")
    print(accuracy_matrix)

    # Optionally save the matrix to a file
    np.save(cfg.EXPERIMENT_PATH / "cross_domain_accuracy_matrix.npy", accuracy_matrix)
    print(f"Accuracy matrix saved to: {cfg.EXPERIMENT_PATH / 'cross_domain_accuracy_matrix.npy'}")

    # Create a plot
    plt.figure(figsize=(8, 8))
    plt.imshow(accuracy_matrix, cmap='viridis', interpolation='none', vmin=0, vmax=1)

    # Add color bar
    plt.colorbar(label='Value')

    # Add values in each cell
    rows, cols = accuracy_matrix.shape
    for i in range(rows):
        for j in range(cols):
            plt.text(j, i, f'{accuracy_matrix[i, j]:.2f}',  # Format to 2 decimal places
                    ha='center', va='center', color='white', fontsize=8)

    # Set the tick labels to the names
    plt.xticks(ticks=np.arange(cols), labels=locations, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(rows), labels=locations)

    # Add labels (optional)
    plt.title('Matrix with Values and Named Locations')
    plt.xlabel('Locations')
    plt.ylabel('Locations')

    # Save the plot as an image
    plt.tight_layout()
    plt.savefig(cfg.EXPERIMENT_PATH / 'cross_domain_accuracy_matrix.png', dpi=300)


if __name__ == "__main__":
    main()