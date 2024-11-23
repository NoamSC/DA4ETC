import os
import random
import json
from pathlib import Path
import inspect
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from simple_dataloader import create_dataloader
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
    pcaps = list(cfg.DATA_PATH.glob('**/*.pcap'))
    pcaps = pd.read_csv('pcap_paths.csv').values[:, 1]
    df = pd.DataFrame([(str(p), *extract_pcap_info(p)) for p in pcaps], columns=['pcap_path', 'location', 'date', 'app', 'vpn_type'])
    df = df[df.location == 'TLVunContainer1'].sample(frac=cfg.SAMPLE_FRAC).sample(5000).sort_values(by='date')
    
    split_index = int(len(df) * cfg.TRAIN_SPLIT_RATIO)
    df_train, df_val = df[:split_index], df[split_index:]

    # # Shuffle the DataFrame
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # # Calculate the split index based on the ratio
    # split_index = int(len(df) * cfg.TRAIN_SPLIT_RATIO)

    # # Split into training and validation sets
    # df_train = df.iloc[:split_index]
    # df_val = df.iloc[split_index:]

    # with open('cached_dataloaders_64.pkl', 'rb') as f:
    #     train_loader, val_loader = pickle.load(f)
    #     label_mapping = {'Google Search': 0, 'Amazon': 1, 'Twitch': 2, 'Youtube': 3}
    
    unique_labels = df['app'].unique()
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    print(label_mapping)

    train_loader = create_dataloader(df_train['pcap_path'], df_train['app'], cfg.BATCH_SIZE, True, cfg.MIN_FLOW_LENGTH, cfg.RESOLUTION, label_mapping=label_mapping)
    val_loader = create_dataloader(df_val['pcap_path'], df_val['app'], cfg.BATCH_SIZE, False, cfg.MIN_FLOW_LENGTH, cfg.RESOLUTION, label_mapping=label_mapping)

    return train_loader, val_loader, label_mapping

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

        # Use nn.ModuleList instead of a regular list to store layers
        self.layers = nn.ModuleList()
        in_channels = params['input_shape']

        # Create convolutional layers based on specified type in each layer config
        for conv_layer in params['conv_layers']:
            layer_type = conv_layer.get('type', self.conv_type)  # Use specified layer type or default to conv_type

            if layer_type == '1d':
                self.layers.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=conv_layer['out_channels'],
                        kernel_size=conv_layer['kernel_size'],
                        stride=conv_layer['stride'],
                        padding=conv_layer['padding']
                    )
                )
                self.layers.append(nn.GELU())
                self.layers.append(nn.MaxPool1d(kernel_size=params['pool_kernel_size'], stride=params['pool_stride']))
                in_channels = conv_layer['out_channels']
            elif layer_type == '2d':
                self.layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_layer['out_channels'],
                        kernel_size=(conv_layer['kernel_size'], conv_layer['kernel_size']),
                        stride=(conv_layer['stride'], conv_layer['stride']),
                        padding=(conv_layer['padding'], conv_layer['padding'])
                    )
                )
                self.layers.append(nn.ReLU())
                self.layers.append(nn.MaxPool2d(kernel_size=params['pool_kernel_size'], stride=params['pool_stride']))
                in_channels = conv_layer['out_channels']
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        # Calculate the flattened size after convolutions and pooling
        self.flattened_size = self._get_flattened_size(params['input_shape'])

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, params['fc1_out_features'])
        self.fc2 = nn.Linear(params['fc1_out_features'], params['num_classes'])
        self.dropout = nn.Dropout(params['dropout_prob'])

    def _get_flattened_size(self, input_shape):
        # Determine input size dynamically based on conv_type ('1d' or '2d')
        if self.conv_type == '1d':
            x = torch.randn(1, self.params['input_shape'], input_shape).to(next(self.parameters()).device)
        else:
            x = torch.randn(1, 1, input_shape, input_shape).to(next(self.parameters()).device)

        for layer in self.layers:
            x = layer(x)  # Pass through each layer in the list
        return x.numel()

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)  # Explicitly apply each layer

        x = x.view(x.size(0), -1)  # Flatten
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------- Training Function ----------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
                flowpics, labels = flowpics.to(device), labels.to(device).long()

                optimizer.zero_grad()
                outputs = model(flowpics)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * flowpics.size(0)
                total_train_samples += flowpics.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_train_predictions += (predicted == labels).sum().item()

                train_batch_losses.append(loss.item())
                train_batch_accuracies.append((correct_train_predictions / total_train_samples) * 100)

        mean_train_loss = running_train_loss / total_train_samples
        mean_train_accuracy = (correct_train_predictions / total_train_samples) * 100

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val_predictions = 0
        total_val_samples = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)", leave=False) as pbar:
                for i, (flowpics, labels) in enumerate(pbar):
                    flowpics, labels = flowpics.to(device), labels.to(device).long()
                    outputs = model(flowpics)
                    loss = criterion(outputs, labels)

                    running_val_loss += loss.item() * flowpics.size(0)
                    total_val_samples += flowpics.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct_val_predictions += (predicted == labels).sum().item()

                    val_batch_losses.append(loss.item())
                    val_batch_accuracies.append((correct_val_predictions / total_val_samples) * 100)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

        mean_val_loss = running_val_loss / total_val_samples
        mean_val_accuracy = (correct_val_predictions / total_val_samples) * 100

        if weights_save_path_format:
            torch.save(model.state_dict(), str(weights_save_path_format).format(epoch=epoch))

        # Save confusion matrix for validation
        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title(f"Confusion Matrix - Epoch {epoch+1}")
        plt.savefig(cfg.EXPERIMENT_PLOTS_PATH / f'confusion_matrix_epoch_{epoch+1}.png')
        plt.close()

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Training - Mean Loss: {mean_train_loss:.4f}, Mean Accuracy: {mean_train_accuracy:.2f}%")
        print(f"  Validation - Mean Loss: {mean_val_loss:.4f}, Mean Accuracy: {mean_val_accuracy:.2f}%")

        # Plot training and validation metrics
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_batch_losses, label='Training Loss per Batch')
        plt.plot(val_batch_losses, label='Validation Loss per Batch')
        plt.xlabel('Batch Count')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Batch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_batch_accuracies, label='Training Accuracy per Batch')
        plt.plot(val_batch_accuracies, label='Validation Accuracy per Batch')
        plt.xlabel('Batch Count')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy per Batch')
        plt.legend()

        plt.suptitle(f'Epoch {epoch+1}/{num_epochs}')
        plt.savefig(cfg.EXPERIMENT_PLOTS_PATH / f'training_progress_epoch_{epoch+1}.png')
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
    train_loader, val_loader, label_mapping = load_and_prepare_data()
    num_classes = len(label_mapping)
    cfg.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    print("starting training")
    train_and_validate(model, train_loader, val_loader,
                       num_epochs=cfg.NUM_EPOCHS, device=cfg.DEVICE,
                       weights_save_path_format=cfg.EXPERIMENT_WEIGHTS_PATH / 'model_weights_epoch_{epoch}.pth')

    if cfg.SAVE_MODEL_CHECKPOINT:
        torch.save(model.state_dict(), cfg.SAVE_MODEL_CHECKPOINT)
        print(f"Model saved to {cfg.SAVE_MODEL_CHECKPOINT}")
