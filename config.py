# config.py

import torch
from pathlib import Path

# ---------- Experiment details ----------
# EXPERIMENT_NAME = '1d_temporal_split_net_v2_no_transpose_gelu_fixed_label_mapping'
EXPERIMENT_NAME = 'tmp'
EXPERIMENT_PATH = Path(f'exps/{EXPERIMENT_NAME}')

# ---------- Environment and reproducibility ----------
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- Dataset Parameters ----------
DATA_PATH = Path('data/ben_bucket')
SAMPLE_FRAC = 1.0 
TRAIN_SPLIT_RATIO = 0.7
BATCH_SIZE = 8

# ---------- FlowPic Generation Parameters ----------
MIN_FLOW_LENGTH = 100
RESOLUTION = 64

# ---------- Model Parameters ----------
MODEL_PARAMS = {
    'num_classes': None,  # Will be set after dataset loading
    'conv_type': '1d',  # Options: '1d' for 1D convolutions, '2d' for 2D convolutions
    'input_shape': RESOLUTION,  # Changed from 'image_size' to a more general 'input_shape'
    'conv_layers': [
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        # {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        # Additional layers can be defined as needed
    ],
    'pool_kernel_size': 2,
    'pool_stride': 2,
    'fc1_out_features': 64,
    'dropout_prob': 0.4
}

# ---------- Training Parameters ----------
LEARNING_RATE = 1e-4
NUM_EPOCHS = 7

# ---------- Logging and Checkpoint Parameters ----------
SAVE_PLOTS = True
EXPERIMENT_PLOTS_PATH = EXPERIMENT_PATH / 'plots'

EXPERIMENT_WEIGHTS_PATH = EXPERIMENT_PATH / 'weights'
SAVE_MODEL_CHECKPOINT = EXPERIMENT_WEIGHTS_PATH / 'model_checkpoint_epoch_{epoch}.pth'
