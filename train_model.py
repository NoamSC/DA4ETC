import os
import random
import json
from pathlib import Path
import inspect

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from simple_dataloader import create_dataloader
import matplotlib.pyplot as plt

# Importing parameters from config.py
import config as cfg

# ---------- Helper Functions ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.SEED)

def load_and_prepare_data():
    # pcaps = list(cfg.DATA_PATH.glob('**/*.pcap'))
    pcaps = pd.read_csv('pcap_paths.csv').values[:, 1]
    df = pd.DataFrame([(str(p), *extract_pcap_info(p)) for p in pcaps], columns=['pcap_path', 'location', 'date', 'app', 'vpn_type'])
    df = df[df.location == 'TLVunContainer1'].sample(frac=cfg.SAMPLE_FRAC).sort_values(by='date').sample(10)
    
    split_index = int(len(df) * cfg.TRAIN_SPLIT_RATIO)
    df_train, df_val = df[:split_index], df[split_index:]
    
    train_loader = create_dataloader(df_train['pcap_path'], df_train['app'], cfg.BATCH_SIZE, True, cfg.MIN_FLOW_LENGTH, cfg.RESOLUTION)
    val_loader = create_dataloader(df_val['pcap_path'], df_val['app'], cfg.BATCH_SIZE, False, cfg.MIN_FLOW_LENGTH, cfg.RESOLUTION)
    
    return train_loader, val_loader, len(df_train['app'].unique())

def extract_pcap_info(path):
    parts = Path(path).parts
    location, date, app, vpn_type = parts[3], pd.to_datetime(parts[4], format='%Y%m%d_%H%M%S'), parts[5], parts[6]
    return location, date, app, vpn_type

# ---------- Model Definition ----------
class ConfigurableCNN(nn.Module):
    def __init__(self, params):
        super(ConfigurableCNN, self).__init__()
        self.params = params
        self.conv_type = params['conv_type']  # '1d' or '2d' determined in config file

        layers = []
        in_channels = params['input_shape']
        
        # Create convolutional layers based on specified type in each layer config
        for conv_layer in params['conv_layers']:
            layer_type = conv_layer.get('type', self.conv_type)  # Use specified layer type or default to conv_type

            if layer_type == '1d':
                layers.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=conv_layer['out_channels'],
                        kernel_size=conv_layer['kernel_size'],
                        stride=conv_layer['stride'],
                        padding=conv_layer['padding']
                    )
                )
                layers.append(nn.ReLU())
                in_channels = conv_layer['out_channels']
            elif layer_type == '2d':
                layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_layer['out_channels'],
                        kernel_size=(conv_layer['kernel_size'], conv_layer['kernel_size']),
                        stride=(conv_layer['stride'], conv_layer['stride']),
                        padding=(conv_layer['padding'], conv_layer['padding'])
                    )
                )
                layers.append(nn.ReLU())
                in_channels = conv_layer['out_channels']
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        # Sequential container for conv layers
        self.conv_layers = nn.Sequential(*layers)
        
        # Pooling layer (type determined by conv_type)
        PoolLayer = nn.MaxPool1d if self.conv_type == '1d' else nn.MaxPool2d
        self.pool = PoolLayer(kernel_size=params['pool_kernel_size'], stride=params['pool_stride'])

        # Calculate the flattened size after convolutions and pooling
        self.flattened_size = self._get_flattened_size(params['input_shape'])

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, params['fc1_out_features'])
        self.fc2 = nn.Linear(params['fc1_out_features'], params['num_classes'])
        self.dropout = nn.Dropout(params['dropout_prob'])

    def _get_flattened_size(self, input_shape):
        # Determine input size dynamically based on conv_type ('1d' or '2d')
        if self.conv_type == '1d':
            x = torch.randn(1, self.params['input_shape'], input_shape)  # Assuming single channel 1D input
        else:
            x = torch.randn(1, 1, input_shape, input_shape)  # Assuming single channel 2D input

        x = self.conv_layers(x)
        x = self.pool(x)
        return x.numel()

    def forward(self, x):
        x = x.transpose(1, 2) # apply 1d on cols and not on rows
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------- Training Function ----------
def train_and_validate(model, train_loader, val_loader, num_epochs=10, device='cpu', weights_save_path_format=None):
    train_batch_losses = []
    train_batch_accuracies = []
    val_batch_losses = []
    val_batch_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        # Training loop with progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)", leave=False) as pbar:
            for i, (flowpics, labels) in enumerate(pbar):
                # Move inputs and labels to the device (GPU or CPU)
                # flowpics = flowpics.unsqueeze(1)  # add a channels dimension
                flowpics, labels = flowpics.to(device), labels.to(device).long()

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(flowpics)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate training loss and accuracy
                running_train_loss += loss.item() * flowpics.size(0)
                total_train_samples += flowpics.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_train_predictions += (predicted == labels).sum().item()

                # Store batch-level training metrics
                train_batch_losses.append(loss.item())
                train_batch_accuracies.append((correct_train_predictions / total_train_samples) * 100)

        # Calculate epoch-level training loss and accuracy
        mean_train_loss = running_train_loss / total_train_samples
        mean_train_accuracy = (correct_train_predictions / total_train_samples) * 100

        # Validation loop with progress bar
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0

        with torch.no_grad():  # Disable gradient computation for validation
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False) as pbar:
                for i, (flowpics, labels) in enumerate(pbar):
                    # Move inputs and labels to the device (GPU or CPU)
                    # flowpics = flowpics.unsqueeze(1)  # add a channels dimension
                    flowpics, labels = flowpics.to(device), labels.to(device).long()

                    # Forward pass
                    outputs = model(flowpics)
                    loss = criterion(outputs, labels)

                    # Accumulate validation loss and accuracy
                    running_val_loss += loss.item() * flowpics.size(0)
                    total_val_samples += flowpics.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct_val_predictions += (predicted == labels).sum().item()

                    # Store batch-level validation metrics
                    val_batch_losses.append(loss.item())
                    val_batch_accuracies.append((correct_val_predictions / total_val_samples) * 100)

        # Calculate epoch-level validation loss and accuracy
        mean_val_loss = running_val_loss / total_val_samples
        mean_val_accuracy = (correct_val_predictions / total_val_samples) * 100

        if weights_save_path_format:
            torch.save(model.state_dict(), weights_save_path_format.format(epoch=epoch))

        # Print mean statistics for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Training - Mean Loss: {mean_train_loss:.4f}, Mean Accuracy: {mean_train_accuracy:.2f}%")
        print(f"  Validation - Mean Loss: {mean_val_loss:.4f}, Mean Accuracy: {mean_val_accuracy:.2f}%")

        # Plot and save the figure after each epoch
        plt.figure(figsize=(10, 5))
        
        # Plot training and validation losses per batch
        plt.subplot(1, 2, 1)
        plt.plot(train_batch_losses, label='Training Loss per Batch')
        plt.plot(val_batch_losses, label='Validation Loss per Batch')
        plt.xlabel('Batch Count')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Batch')
        plt.legend()

        # Plot training and validation accuracies per batch
        plt.subplot(1, 2, 2)
        plt.plot(train_batch_accuracies, label='Training Accuracy per Batch')
        plt.plot(val_batch_accuracies, label='Validation Accuracy per Batch')
        plt.xlabel('Batch Count')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy per Batch')
        plt.legend()

        plt.suptitle(f'Epoch {epoch+1}/{num_epochs}')
        plt.savefig(cfg.EXPERIMENT_PATH / f'training_progress_epoch_{epoch+1}.png')
        plt.close()

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


def init_exp(cfg):
    cfg.EXPERIMENT_PATH.mkdir(exist_ok=True)
    cfg.EXPERIMENT_PLOTS_PATH.mkdir(exist_ok=True)
    cfg.EXPERIMENT_WEIGHTS_PATH.mkdir(exist_ok=True)

    save_config_to_json(cfg, cfg.EXPERIMENT_PATH / "config.json")


# ---------- Main Execution ----------
if __name__ == "__main__":
    init_exp(cfg)

    print("preparing data")
    train_loader, val_loader, num_classes = load_and_prepare_data()
    cfg.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    print("starting training")
    train_and_validate(model, train_loader, val_loader,
                       num_epochs=cfg.NUM_EPOCHS, device=cfg.DEVICE)

    if cfg.SAVE_MODEL_CHECKPOINT:
        torch.save(model.state_dict(), cfg.SAVE_MODEL_CHECKPOINT)
        print(f"Model saved to {cfg.SAVE_MODEL_CHECKPOINT}")
