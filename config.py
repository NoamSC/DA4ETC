import torch
from pathlib import Path

# ---------- Experiment details ----------
EXPERIMENT_NAME = 'fast_CNN_advanced_mmd_on_features_mmd_1e0'
EXPERIMENT_PATH = Path(f'exps/{EXPERIMENT_NAME}')

# ---------- Environment and reproducibility ----------
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- Dataset Parameters ----------
DATA_PATH = Path('data/ben_bucket')
LOCATION = 'TLVunContainer1'
SAMPLE_FRAC = 1.0 
TRAIN_SPLIT_RATIO = 0.7
BATCH_SIZE = 16
LABEL_MAPPING = {'Amazon': 0, 'Google Search': 1, 'Twitch': 2, 'Youtube': 3}


# ---------- FlowPic Generation Parameters ----------
MIN_FLOW_LENGTH = 100
RESOLUTION = 256

# ---------- Model Parameters ----------
MODEL_PARAMS = {
    'num_classes': len(LABEL_MAPPING),  # Will be set after dataset loading
    'conv_type': '1d',  # Options: '1d' for 1D convolutions, '2d' for 2D convolutions
    'input_shape': RESOLUTION,
    'conv_layers': [
        {'out_channels':  16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels':  32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels':  64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
        {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'type': '1d'},
    ],
    'pool_kernel_size': 2,
    'pool_stride': 2,
    'fc1_out_features': 64,
    'dropout_prob': 0.25,
    'use_batch_norm': True,
}

# ---------- Training Parameters ----------
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20

# ---------- DA Parameters ----------
LAMBDA_MMD = 1

# ---------- Logging and Checkpoint Parameters ----------
SAVE_PLOTS = True
EXPERIMENT_PLOTS_PATH = EXPERIMENT_PATH / 'plots'

EXPERIMENT_WEIGHTS_PATH = EXPERIMENT_PATH / 'weights'
SAVE_MODEL_CHECKPOINT = EXPERIMENT_WEIGHTS_PATH / 'model_checkpoint_epoch_{epoch}.pth'
