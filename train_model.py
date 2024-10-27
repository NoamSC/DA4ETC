import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
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
    pcaps = list(cfg.DATA_PATH.glob('**/*.pcap'))
    df = pd.DataFrame([(str(p), *extract_pcap_info(p)) for p in pcaps], columns=['pcap_path', 'location', 'date', 'app', 'vpn_type'])
    df = df[df.location == 'HujiPC'].sample(frac=cfg.SAMPLE_FRAC).sort_values(by='date')
    
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
        layers = [nn.Conv2d(1, layer['out_channels'], layer['kernel_size'], layer['stride'], layer['padding']) for layer in params['conv_layers']]
        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.pool = nn.MaxPool2d(params['pool_kernel_size'], params['pool_stride'])
        self.fc1 = nn.Linear(self._get_flattened_size(params['image_size']), params['fc1_out_features'])
        self.fc2 = nn.Linear(params['fc1_out_features'], params['num_classes'])
        self.dropout = nn.Dropout(params['dropout_prob'])
        
    def _get_flattened_size(self, size):
        x = torch.randn(1, 1, size, size)
        return self.pool(self.conv_layers(x)).numel()
        
    def forward(self, x):
        x = self.pool(self.conv_layers(x))
        x = x.view(x.size(0), -1)
        x = self.fc2(F.relu(self.dropout(self.fc1(x))))
        return x

# ---------- Training Function ----------
def train_and_validate(model, train_loader, val_loader):
    model.train()
    for epoch in range(cfg.NUM_EPOCHS):
        # Training
        train_loss, correct_train, total_train = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} (Train)"):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train += labels.size(0)
        
        train_accuracy = correct_train / total_train * 100
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Train Loss: {train_loss/total_train:.4f}, Accuracy: {train_accuracy:.2f}%")
        
        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} (Val)"):
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val += labels.size(0)
        
        val_accuracy = correct_val / total_val * 100
        print(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS} - Val Loss: {val_loss/total_val:.4f}, Accuracy: {val_accuracy:.2f}%")

# ---------- Main Execution ----------
if __name__ == "__main__":
    train_loader, val_loader, num_classes = load_and_prepare_data()
    cfg.MODEL_PARAMS['num_classes'] = num_classes
    model = ConfigurableCNN(cfg.MODEL_PARAMS).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    train_and_validate(model, train_loader, val_loader)

    if cfg.SAVE_MODEL_CHECKPOINT:
        torch.save(model.state_dict(), cfg.SAVE_MODEL_CHECKPOINT)
        print(f"Model saved to {cfg.SAVE_MODEL_CHECKPOINT}")
