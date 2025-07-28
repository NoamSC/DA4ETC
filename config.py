from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class Config:
    # Experiment Details
    # EXPERIMENT_NAME: str = "allot_dann_bsearch_v7"
    EXPERIMENT_NAME: str = "allot_daily_degradation_v6_label_fix_dann/{}"
    # EXPERIMENT_PATH: Path = field(init=False)
    BASE_EXPERIMENTS_PATH: Path = Path("exps/")

    # Environment and Reproducibility
    SEED: int = 42
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    LABEL_WHITELIST: list = field(default_factory=lambda: [
        386, 497, 998, 171, 485, 2613, 340, 373, 561, 967, 436, 1088,
        961, 682, 521, 964, 1450, 1448, 965, 42
    ])
    SAMPLE_FRAC_FROM_CSVS: float = 1e-3

    # Dataset Parameters
    # DATA_PATH: Path = Path("data/ben_bucket")
    # SAMPLE_FRAC: float = 1.0
    # TRAIN_SPLIT_RATIO: float = 0.7
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
        'lambda_rgl': 0, #1e-2,
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
    
    LAMBDA_DANN: float = 1e0 # 1e0
    
    ADAPT_BATCH_NORM = False

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
    def SAVE_MODEL_CHECKPOINT(self) -> Path:
        return self.EXPERIMENT_WEIGHTS_PATH / "model_checkpoint_epoch_{epoch}.pth"


    def __post_init__(self):
        # Set input shape based on resolution
        self.MODEL_PARAMS['input_shape'] = self.RESOLUTION
        
