from dataclasses import dataclass, field
from pathlib import Path
import torch

'''
{'Samsung Galaxy Store': 0,
 'WhatsApp Transfer': 1,
 'Xiaomi Services': 2,
 'Vungle': 3,
 'HTTP': 4,
 'FaceTime': 5,
 'Google RCS': 6,
 'NTP': 7,
 'Xiaomi Cloud': 8,
 'Skype': 9,
 'Temu': 10,
 'iTunes': 11,
 'WeChat': 12,
 'BitTorrent': 13,
 'Bybit': 14,
 'Instagram': 15,
 'TLS with Encrypted Client Hello': 16,
 'DNS over HTTPS': 17,
 'DNS over TLS': 18,
 'Google Search': 19,
 'Office 365': 20,
 'GoogleHangout': 21,
 'Waze': 22,
 'Akamai': 23,
 'Quic Obfuscated': 24,
 'Snapchat': 25,
 'YouTube': 26,
 'SamsungServices': 27,
 'Roblox': 28,
 'TCP Port 443': 29,
 'STUN': 30,
 'Telegram': 31,
 'Facebook': 32,
 'Facebook Calls': 33,
 'BitTorrent Tracker': 34,
 'FacebookContent': 35,
 'NetFlix_Browsing': 36,
 'Office 365 OutLook': 37,
 'MS OutLook': 38,
 'Samsung Cloud': 39,
 'HTTP_Browsing': 40,
 'BitTorrentDHT': 41,
 'NetFlix': 42,
 'Gmail': 43,
 'Twitter': 44,
 'Imo': 45,
 'Amazon Services': 46,
 'Liftoff': 47,
 'WindowsUpdate': 48,
 'Facebook Video': 49,
 'Spotify': 50,
 'Truecaller': 51,
 'Mobile Apple Store': 52,
 'MS Services': 53,
 'Crashlytics': 54,
 'Google Push Notifications': 55,
 'byteoversea': 56,
 'TLS with no SNI': 57,
 'Viber': 58,
 'WhatsApp': 59,
 'Yahoo Management': 60,
 'Google Calendar': 61,
 'Game sites': 62,
 'Google Maps': 63,
 'Online radio and online TV sites': 64,
 'Pay per surf sites': 65,
 'AmazonCloud': 66,
 'iCloud': 67,
 'Facebook Chat': 68,
 'Apple Push Notification': 69,
 'Supercell': 70,
 'GoogleServices': 71,
 'Linkedin': 72,
 'Mobile phone sites': 73,
 'Shopping sites': 74,
 'Travel sites': 75,
 'Ynet': 76,
 'Business sites': 77,
 'Press sites': 78,
 'Computing sites': 79,
 'Alibaba': 80,
 'Banners': 81,
 'AliExpress': 82,
 'Search engine sites': 83,
 'yad2': 84,
 'Pinterest': 85,
 'Google Play': 86,
 'Google UserContent': 87,
 '365Scores': 88,
 'Directory and street map sites': 89,
 'PUBG': 90,
 'YouTube Browsing': 91,
 'Advertisements': 92,
 'Generic CDN': 93,
 'HTTPS_Streaming': 94,
 'Analytics': 95,
 'iCloud Private Relay': 96,
 'TikTok Live': 97,
 'Unity 3D': 98,
 'Wolt': 99,
 'Moovit': 100,
 'Google Cloud': 101,
 'Google Meet': 102,
 'HTTPS': 103,
 'Tik Tok': 104,
 'Siri': 105,
 'Weather.com': 106,
 'AppleServices': 107,
 'AppleCalendar': 108,
 'Cloudflare': 109,
 'WhatsApp Calls': 110,
 'Yandex': 111,
 'Adult Streaming': 112}
'''

@dataclass
class Config:
    # Experiment Details
    EXPERIMENT_NAME: str = "allot_simple_v9_more_data_split"
    EXPERIMENT_PATH: Path = field(init=False)

    # Environment and Reproducibility
    SEED: int = 42
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Dataset Parameters
    DATA_PATH: Path = Path("data/ben_bucket")
    SAMPLE_FRAC: float = 1.0
    TRAIN_SPLIT_RATIO: float = 0.7
    BATCH_SIZE: int = 16
    LABEL_MAPPING: dict = field(default_factory=lambda: {'Amazon': 0, 'Google Search': 1, 'Twitch': 2, 'Youtube': 3})
    LOCATIONS: list = field(default_factory=lambda: [
        'AwsCont', 'BenContainer', # 'CabSpicy1',
        # 'HujiPC', 'TLVunContainer1', 'TLVunContainer2'
    ])

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
    NUM_EPOCHS: int = 20
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
