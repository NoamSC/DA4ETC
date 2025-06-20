from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class Config:
    # Experiment Details
    EXPERIMENT_NAME: str = "allot_domain_change_v10"
    EXPERIMENT_PATH: Path = field(init=False)

    # Environment and Reproducibility
    SEED: int = 42
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Dataset Parameters
    DATA_PATH: Path = Path("data/ben_bucket")
    SAMPLE_FRAC: float = 1.0
    TRAIN_SPLIT_RATIO: float = 0.7
    BATCH_SIZE: int = 64
    # LABEL_MAPPING: dict = field(default_factory=lambda: {'Amazon': 0, 'Google Search': 1, 'Twitch': 2, 'Youtube': 3})
    # LOCATIONS: list = field(default_factory=lambda: [
    #     'AwsCont', 'BenContainer', # 'CabSpicy1',
    #     # 'HujiPC', 'TLVunContainer1', 'TLVunContainer2'
    # ])

    # FlowPic Generation Parameters
    MIN_FLOW_LENGTH: int = 100
    RESOLUTION: int = 256

    # Model Parameters
    MODEL_PARAMS: dict = field(default_factory=lambda: {
        'conv_type': '1d',
        # 'input_shape': 256, # This will be set in __post_init__
        'conv_layers': [
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        ],
        'pool_kernel_size': 2,
        'pool_stride': 2,
        'fc1_out_features': 64,
        'dropout_prob': 0.3,
        'use_batch_norm': True,
        
        # Domain Adaptation Model Parameters
        'lambda_rgl': 1e-2,
        'dann_fc_out_features': 64,
        'lambda_grl_gamma': 10,
    })

    # Training Parameters
    LEARNING_RATE: float = 3e-3
    NUM_EPOCHS: int = 50
    WEIGHT_DECAY: float = 1e-4

    # Domain Adaptation Parameters
    LAMBDA_MMD: float = 0
    MMD_BANDWIDTHS: list = field(default_factory=lambda: [1e-1, 1e0, 1e1])
    
    LAMBDA_DANN: float = 0 # 1e0

    # Logging and Checkpoints
    SAVE_PLOTS: bool = True
    EXPERIMENT_PLOTS_PATH: Path = field(init=False)
    EXPERIMENT_WEIGHTS_PATH: Path = field(init=False)
    SAVE_MODEL_CHECKPOINT: Path = field(init=False)

    def __post_init__(self):
        self.EXPERIMENT_PATH = Path(f"exps/{self.EXPERIMENT_NAME}")
        self.EXPERIMENT_PLOTS_PATH = self.EXPERIMENT_PATH / "plots"
        self.EXPERIMENT_WEIGHTS_PATH = self.EXPERIMENT_PATH / "weights"
        self.SAVE_MODEL_CHECKPOINT = self.EXPERIMENT_WEIGHTS_PATH / "model_checkpoint_epoch_{epoch}.pth"
        self.MODEL_PARAMS['input_shape'] = self.RESOLUTION
