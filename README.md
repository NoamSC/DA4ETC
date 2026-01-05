# Domain Adaptation for Encrypted Traffic Classification (DA4ETC)

This repository contains code for training and evaluating deep learning models on encrypted network traffic classification tasks, with a focus on domain adaptation techniques and temporal drift analysis using the CESNET-TLS-Year22 dataset.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset Structure](#dataset-structure)
- [Main Scripts](#main-scripts)
- [Configuration](#configuration)
- [Data Sampling Parameters](#data-sampling-parameters)
- [Training Workflow](#training-workflow)
- [Key Components](#key-components)

## Project Structure

```
da4etc/
├── config.py                          # Global configuration and hyperparameters
├── train_per_week_cesnet.py          # Main training script for weekly models
├── split_cesnet_train_test.py        # Split CESNET data into train/test sets
├── preprocess_cesnet_to_parquet.py   # Convert raw CESNET data to parquet format
├── simple_model_train.py             # Simple baseline training script
│
├── data_utils/                        # Data loading and preprocessing utilities
│   ├── csv_dataloader.py             # FlowPic dataset and dataloader creation
│   └── ...
│
├── models/                            # Neural network architectures
│   ├── configurable_cnn.py           # Configurable CNN for FlowPic classification
│   └── ...
│
├── training/                          # Training utilities and domain adaptation
│   ├── trainer.py                    # Main training loop with MMD/DANN support
│   ├── visualization.py              # Plotting and TensorBoard logging
│   ├── utils.py                      # Helper functions (set_seed, save_config, etc.)
│   └── domain_adaptation_methods/    # Domain adaptation implementations
│       ├── bbse_resampling.py        # Black Box Shift Estimation
│       └── ...
│
├── exps/                              # Experiment results (auto-generated)
│   └── debug/                        # Experiment outputs organized by name
│       └── WEEK-2022-XX/             # Per-week experiment directories
│           ├── weights/              # Model checkpoints
│           ├── plots/                # Training visualizations
│           ├── tensorboard/          # TensorBoard logs
│           └── config.json           # Experiment configuration
│
├── logs/                              # SLURM job logs (auto-generated)
├── figs/                              # Analysis figures and visualizations
│   ├── cesnet_figs/                  # CESNET dataset analysis
│   └── mirage_analysis_results/      # MIRAGE dataset analysis
│
├── run_weekly_training.slurm         # SLURM script for parallel weekly training
├── CESNET_PREPROCESSING_README.md    # CESNET data preprocessing guide
├── WEEKLY_TRAINING_README.md         # Weekly training workflow guide
└── README.md                          # This file
```

## Dataset Structure

### CESNET-TLS-Year22 Dataset
Located at: `/home/anatbr/dataset/CESNET-TLS-Year22/`

```
CESNET-TLS-Year22/
├── label_mapping.json                 # Maps app names to integer IDs
├── stats-dataset.json                 # Dataset-wide statistics
│
└── WEEK-2022-XX/                      # One directory per week (00-52)
    ├── train.parquet                  # Training data (70% of week data)
    ├── test.parquet                   # Test data (30% of week data)
    ├── train_test_split.json         # Split metadata (seed, ratios, counts)
    ├── stats-week.json               # Weekly statistics
    │
    └── YYYY-MM-DD/                    # Raw daily data directories
        └── *.csv.xz                   # Compressed CSV files (original format + parquets)
```

#### Data Split Details
- **Train/Test Split**: 70% training, 30% testing per week
- **Deterministic Splitting**: Each week uses seed = 42 + week_index
- **Temporal Structure**: Data organized by calendar weeks (2022)
- **Classes**: 180 application types (see `label_mapping.json`)

#### Parquet File Schema
Each parquet file contains the following columns:
- `APP`: Application name (string)
- `PPI_IPT`: Packet size and inter-packet time sequences
- Additional metadata columns (see CESNET_PREPROCESSING_README.md)

## Main Scripts

### Training Scripts

#### `train_per_week_cesnet.py`
Main script for training weekly models with FlowPic representation.

**Usage:**
```bash
python train_per_week_cesnet.py \
    --week 18 \
    --dataset_root /path/to/CESNET-TLS-Year22 \
    --train_data_frac 0.01 \
    --val_data_frac 0.01 \
    --train_per_epoch_data_frac 1.0 \
    --batch_size 64 \
    --num_epochs 15 \
    --learning_rate 3e-3
```

**Key Arguments:**
- `--week`: Train on specific week (0-52)
- `--start_week/--end_week`: Train on range of weeks
- `--train_data_frac`: Fraction of training data to load (default: 0.01 = 1%)
- `--val_data_frac`: Fraction of validation data to load (default: auto-calculated)
- `--train_per_epoch_data_frac`: Fraction of loaded training data to use per epoch (default: 1.0)
- `--override`: Delete existing experiment and start fresh

#### `run_weekly_training.slurm`
SLURM array job for parallel training across all weeks.

**Usage:**
```bash
sbatch run_weekly_training.slurm
```

**Configuration:**
- Array size: 0-52 (53 weeks)
- Partition: gpu-h100-killable
- Memory: 32GB
- Time limit: 16.5 hours

### Data Preprocessing Scripts

#### `split_cesnet_train_test.py`
Splits weekly data into train/test parquet files.

**Features:**
- 70/30 train/test split
- Deterministic with week-dependent seeds
- Saves split metadata in `train_test_split.json`

#### `preprocess_cesnet_to_parquet.py`
Converts raw CESNET CSV files to parquet format.

**Features:**
- Processes compressed CSV.xz files
- Generates weekly aggregations
- Creates label mappings

## Configuration

### `config.py`
Global configuration using Python dataclasses.

**Key Parameters:**

```python
# Experiment
EXPERIMENT_NAME: str = "debug/{}"        # Experiment directory template
BASE_EXPERIMENTS_PATH: Path = Path("exps/")

# Reproducibility
SEED: int = 42
DEVICE: torch.device                     # Auto-detected (cuda:0 or cpu)

# Data Sampling (NEW in this version)
TRAIN_DATA_FRAC: float = 1e-2           # Load 1% from train.parquet
VAL_DATA_FRAC: float = None              # Auto: train_data_frac * train_per_epoch_data_frac
TRAIN_PER_EPOCH_DATA_FRAC: float = 1.0  # Use 100% of loaded data per epoch

# Dataset
BATCH_SIZE: int = 64
LABEL_WHITELIST: list                   # Subset of classes to use (180 total)

# FlowPic Parameters
MIN_FLOW_LENGTH: int = 100
RESOLUTION: int = 256                    # FlowPic image size (256x256)

# Model Architecture
MODEL_PARAMS: dict                       # CNN configuration (see config.py)

# Training
LEARNING_RATE: float = 3e-3
NUM_EPOCHS: int = 15
WEIGHT_DECAY: float = 1e-4

# Domain Adaptation
LAMBDA_MMD: float = 0.0                  # MMD loss weight
LAMBDA_DANN: float = 0.0                 # DANN loss weight
MMD_BANDWIDTHS: list = [1e-1, 1e0, 1e1]
```

## Data Sampling Parameters

The training system supports three-level sampling control for flexible experimentation:

### 1. `train_data_frac` - Training Data Loading
Controls how much data to load from `train.parquet` into memory.

**Default:** 0.01 (1% of training data)

**Example:** With 10M training samples:
- `train_data_frac=0.01` → Loads 100K samples
- `train_data_frac=0.1` → Loads 1M samples

**When to use:** Adjust based on available memory and dataset size.

### 2. `val_data_frac` - Validation Data Loading
Controls how much data to load from `test.parquet` into memory.

**Default:** `train_data_frac * train_per_epoch_data_frac` (auto-calculated)

**Example:**
- Explicit: `--val_data_frac 0.1` → Always load 10% of test data
- Auto: With `train_data_frac=0.01` and `train_per_epoch_data_frac=1.0` → `val_data_frac=0.01`

**Key Feature:** Validation sets are **reproducible** across experiments when using the same `val_data_frac` value (deterministic sampling with `seed=42`).

### 3. `train_per_epoch_data_frac` - Per-Epoch Training Sampling
Controls what fraction of the **loaded** training data to use each epoch.

**Default:** 1.0 (use all loaded data)

**Example:** With 100K loaded training samples:
- `train_per_epoch_data_frac=1.0` → Train on all 100K samples per epoch
- `train_per_epoch_data_frac=0.5` → Train on 50K random samples per epoch
- **Different samples each epoch** (epoch-dependent seed: `seed + epoch`)

**When to use:**
- `< 1.0`: Faster epochs for rapid prototyping or hyperparameter search
- `= 1.0`: Standard training on full loaded dataset

### Sampling Examples

```bash
# Example 1: Standard training (1% data, full epochs)
python train_per_week_cesnet.py \
    --train_data_frac 0.01 \
    --train_per_epoch_data_frac 1.0
# Loads: 1% train, 1% val
# Per epoch: Uses 100% of loaded data

# Example 2: Per-epoch sampling (1% data, 50% per epoch)
python train_per_week_cesnet.py \
    --train_data_frac 0.01 \
    --train_per_epoch_data_frac 0.5
# Loads: 1% train, 0.5% val (auto-calculated)
# Per epoch: Uses 50% of loaded data (different samples each epoch)

# Example 3: Custom validation size (1% train, 10% val)
python train_per_week_cesnet.py \
    --train_data_frac 0.01 \
    --val_data_frac 0.1 \
    --train_per_epoch_data_frac 1.0
# Loads: 1% train, 10% val (explicit)
# Per epoch: Uses 100% of loaded data

# Example 4: Large data experiment (10% train, 2% val, 20% per epoch)
python train_per_week_cesnet.py \
    --train_data_frac 0.1 \
    --val_data_frac 0.02 \
    --train_per_epoch_data_frac 0.2
# Loads: 10% train, 2% val
# Per epoch: Uses 20% of loaded data
```

### Validation Set Reproducibility

**Question:** Will the validation set be the same across different experiments?

**Answer:** YES! The validation set is deterministic and reproducible when:
1. Same `val_data_frac` value is used
2. Same seed (`cfg.SEED = 42`)
3. Same `test.parquet` file

**Technical Details:**
- Data loading uses `pandas.sample(frac=val_data_frac, random_state=seed)`
- Validation is NEVER affected by `train_per_epoch_data_frac`
- Same validation samples → Fair comparison across experiments

## Training Workflow

### Single Week Training

```bash
# Train on week 18 with default settings
python train_per_week_cesnet.py --week 18

# Train with custom sampling
python train_per_week_cesnet.py \
    --week 18 \
    --train_data_frac 0.05 \
    --val_data_frac 0.1 \
    --train_per_epoch_data_frac 0.8 \
    --num_epochs 20
```

### Batch Training (All Weeks)

```bash
# Submit SLURM array job
sbatch run_weekly_training.slurm

# Monitor jobs
squeue -u $USER

# Check specific job log
cat logs/weekly_train_JOBID_18.out
```

### Resume Training

The system automatically detects and resumes from checkpoints:

```bash
# Run same command - will resume from last checkpoint
python train_per_week_cesnet.py --week 18

# Force restart (delete existing experiment)
python train_per_week_cesnet.py --week 18 --override
```

### Experiment Outputs

Each experiment creates:
```
exps/debug/WEEK-2022-18/
├── config.json                    # Saved configuration
├── weights/
│   ├── model_weights_epoch_1.pth  # Per-epoch checkpoints
│   ├── model_weights_epoch_2.pth
│   └── best_model.pth             # Best model by validation accuracy
├── plots/
│   └── training_history.pth       # Loss and accuracy curves
└── tensorboard/
    └── events.out.tfevents.*      # TensorBoard logs
```

### View TensorBoard

```bash
tensorboard --logdir exps/debug/WEEK-2022-18/tensorboard
```

## Key Components

### Data Loading (`data_utils/csv_dataloader.py`)

**CSVFlowPicDataset:**
- Loads parquet/CSV files with FlowPic representation
- Supports CESNET and MIRAGE dataset formats
- Configurable data sampling (`data_sample_frac`, `seed`)
- On-the-fly FlowPic generation from PPI-IPT sequences

**create_csv_flowpic_loader:**
```python
loader = create_csv_flowpic_loader(
    csv_paths=['train.parquet'],
    batch_size=64,
    shuffle=True,
    data_sample_frac=0.01,      # Load 1% of data
    seed=42,                     # Reproducibility
    resolution=256,              # 256x256 FlowPic
    max_dt_ms=4000,             # Max time window
    label_mapping=mapping,
    dataset_format='cesnet_parquet'
)
```

### Model Architecture (`models/configurable_cnn.py`)

**ConfigurableCNN:**
- Flexible CNN architecture for FlowPic classification
- Configurable conv layers, pooling, dropout, batch norm
- Support for domain adaptation (MMD, DANN)
- Input: 256x256 FlowPic images
- Output: Class predictions (180 classes)

### Training Loop (`training/trainer.py`)

**train_model:**
- Supports vanilla training + domain adaptation (MMD, DANN)
- TensorBoard logging (loss, accuracy, confusion matrices)
- Automatic checkpointing and best model saving
- Resume from checkpoint support
- **NEW:** Per-epoch sampling support (`train_per_epoch_data_frac`, `seed`)

**Features:**
- Per-epoch sampler creation for variable training data
- Epoch-dependent seed for reproducible sampling variation
- Validation always uses 100% of loaded data

### Domain Adaptation Methods

**Maximum Mean Discrepancy (MMD):**
```python
# Enable MMD loss
cfg.LAMBDA_MMD = 1.0
cfg.MMD_BANDWIDTHS = [0.1, 1.0, 10.0]
```

**Domain Adversarial Neural Network (DANN):**
```python
# Enable DANN
cfg.LAMBDA_DANN = 1.0
```

**Black Box Shift Estimation (BBSE):**
```python
from training.domain_adaptation_methods.bbse_resampling import create_bbse_resampled_dataloader

resampled_loader = create_bbse_resampled_dataloader(
    source_loader=train_loader,
    target_loader=test_loader,
    device='cuda'
)
```

## Additional Resources

- **CESNET Preprocessing:** See [CESNET_PREPROCESSING_README.md](CESNET_PREPROCESSING_README.md)
- **Weekly Training Guide:** See [WEEKLY_TRAINING_README.md](WEEKLY_TRAINING_README.md)
- **Plan Files:** See `.claude/plans/` for implementation plans

## Citation

If you use this code, please cite the CESNET-TLS-Year22 dataset:
```
@inproceedings{cesnet-tls-year22,
  title={CESNET-TLS-Year22: A Year-Long TLS Network Traffic Dataset},
  author={...},
  booktitle={...},
  year={2024}
}
```

## License

[Add your license information here]

## Contact

[Add contact information or links to related papers/repos]