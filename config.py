from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class Config:
    # Experiment Details
    # EXPERIMENT_NAME: str = "allot_daily_degradation_v14_tensorboard/{}"
    # EXPERIMENT_NAME: str = "debug/{}"
    EXPERIMENT_NAME: str = "cesnet_v4_dann/{}"
    DESCRIPTION: str = "same as v3 with dann"
    # EXPERIMENT_PATH: Path = field(init=False)
    BASE_EXPERIMENTS_PATH: Path = Path("exps/")

    # Environment and Reproducibility
    SEED: int = 42
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    LABEL_WHITELIST: list = field(default_factory=lambda: [
        386, 497, 998, 171, 485, 2613, 340, 373, 561, 967, 436, 1088,
        961, 682, 521, 964, 1450, 1448, 965, 42
    ])
    TRAIN_DATA_FRAC: float = 1.0             # Load 100% from train.parquet
    VAL_DATA_FRAC: float = 0.001              # Load 1% from test.parquet
    TRAIN_PER_EPOCH_DATA_FRAC: float = 0.001 # Use 0.1% of loaded training data per epoch

    # Dataset Parameters
    # DATA_PATH: Path = Path("data/ben_bucket")
    # SAMPLE_FRAC: float = 1.0
    # TRAIN_SPLIT_RATIO: float = 0.7
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 8  # Number of workers for data loading
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
        'lambda_rgl': 0.002168,
        'dann_fc_out_features': 64,
        'lambda_grl_gamma': 10,
        
    })

    # Training Parameters
    LEARNING_RATE: float = 3e-3
    NUM_EPOCHS: int = 15
    WEIGHT_DECAY: float = 1e-4

    # Domain Adaptation Parameters
    LAMBDA_MMD: float = 0
    MMD_BANDWIDTHS: list = field(default_factory=lambda: [1e-1, 1e0, 1e1])
    
    LAMBDA_DANN: float = 0.020626 # 1e0 # 1e0

    ADAPT_BATCH_NORM = False

    # MUST Domain Adaptation Parameters
    MUST_PARAMS: dict = field(default_factory=lambda: {
        'iterations': 10000,            # Number of training iterations
        'alpha': 0.5,                  # Ping-pong loss weight (coupling teacher-student)
        'pseudo_threshold': 0.75,      # Confidence threshold for pseudo-labels
        'warm_start_epochs': 5,        # Pre-training epochs on source
        'eval_every': 500,              # Evaluation frequency (iterations)
        'target_batches_per_iter': 2,  # How many target batches to process per source batch
        'optimizer': 'adamw',          # Optimizer: 'sgd' or 'adamw'
        'momentum': 0.9,               # SGD momentum (only used if optimizer='sgd')
        'betas': (0.9, 0.999),         # AdamW betas (only used if optimizer='adamw')
        'eps': 1e-8,                   # AdamW epsilon (only used if optimizer='adamw')
    })

    # Logging and Checkpoints
    SAVE_PLOTS: bool = True
    # EXPERIMENT_PLOTS_PATH: Path = field(init=False)
    # EXPERIMENT_WEIGHTS_PATH: Path = field(init=False)
    # SAVE_MODEL_CHECKPOINT: Path = field(init=False)
        
    @property
    def EXPERIMENT_PATH(self) -> Path:
        return self.BASE_EXPERIMENTS_PATH / self.EXPERIMENT_NAME

    @property
    def EXPERIMENT_PLOTS_PATH(self) -> Path:
        return self.EXPERIMENT_PATH / "plots"

    @property
    def EXPERIMENT_WEIGHTS_PATH(self) -> Path:
        return self.EXPERIMENT_PATH / "weights"

    @property
    def EXPERIMENT_TENSORBOARD_PATH(self) -> Path:
        return self.EXPERIMENT_PATH / "tensorboard"

    @property
    def SAVE_MODEL_CHECKPOINT(self) -> Path:
        return self.EXPERIMENT_WEIGHTS_PATH / "model_checkpoint_epoch_{epoch}.pth"


    def __post_init__(self):
        # Set input shape based on resolution
        self.MODEL_PARAMS['input_shape'] = self.RESOLUTION
        
