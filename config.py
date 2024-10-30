# config.py

import torch
from pathlib import Path

# ---------- Experiment details ----------

EXPERIMENT_NAME = '1d'
EXPERIMENT_PATH = Path(f'exps/{EXPERIMENT_NAME}')

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
RESOLUTION = 2048

# ---------- Model Parameters ----------
MODEL_PARAMS = {
    'num_classes': None,  # Will be set after dataset loading
    'conv_type': '1d',  # Options: '1d' for 1D convolutions, '2d' for 2D convolutions
    'input_shape': 2048,  # Changed from 'image_size' to a more general 'input_shape'
    'conv_layers': [
        {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        # Additional layers can be defined as needed
    ],
    'in_channels': 2048,
    'pool_kernel_size': 1,
    'pool_stride': 1,
    'fc1_out_features': 16,
    'dropout_prob': 0.4
}

# ---------- Training Parameters ----------
LEARNING_RATE = 1e-3
NUM_EPOCHS = 7

# ---------- Logging and Checkpoint Parameters ----------
SAVE_PLOTS = True
SAVE_MODEL_CHECKPOINT = EXPERIMENT_PATH / 'model_checkpoint.pth'
