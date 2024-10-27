# config.py

import torch
from pathlib import Path

# ---------- Environment and reproducibility ----------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Dataset Parameters ----------
DATA_PATH = Path('data/ben_bucket')
SAMPLE_FRAC = 0.3  # Fraction of the HujiPC data to sample
TRAIN_SPLIT_RATIO = 0.7
BATCH_SIZE = 8

# ---------- FlowPic Generation Parameters ----------
MIN_FLOW_LENGTH = 100
RESOLUTION = 1500

# ---------- Model Parameters ----------
MODEL_PARAMS = {
    'num_classes': None,  # Will be set after dataset loading
    'image_size': 1500,
    'conv_layers': [
        {'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    ],
    'in_channels': 1,
    'pool_kernel_size': 2,
    'pool_stride': 2,
    'fc1_out_features': 16,
    'dropout_prob': 0.4
}

# ---------- Training Parameters ----------
LEARNING_RATE = 1e-3
NUM_EPOCHS = 7

# ---------- Logging and Checkpoint Parameters ----------
SAVE_PLOTS = True
SAVE_MODEL_CHECKPOINT = 'model_checkpoint.pth'
