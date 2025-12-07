# TensorBoard Logging Guide

## Overview

The training pipeline now includes **TensorBoard logging** for real-time monitoring and **automatic best model checkpointing**. All visualizations are logged to TensorBoard - no more PNG files scattered around!

## What's New

### 1. TensorBoard Logging

All training metrics and visualizations are now logged to TensorBoard:

#### Standard Training (`trainer.py`)
- **Scalars**:
  - Loss metrics: train loss, validation loss, regular loss, MMD loss, DANN loss
  - Accuracy metrics: train accuracy, validation accuracy
  - Per-epoch tracking: All metrics tracked at each epoch
- **Images**:
  - Confusion matrix: Logged every epoch to track model predictions
  - Training metrics summary: Final plot showing loss/accuracy curves

#### MUST Training (`must_trainer.py`)
- **Scalars**:
  - Warm Start Phase: train/val loss and accuracy during teacher pre-training
  - MUST Training Phase:
    - Teacher and Student accuracy on source and target domains
    - Teacher and Student loss on source and target domains
    - Pseudo-label usage percentage
    - All metrics tracked at each evaluation interval
- **Images**:
  - Confusion matrices (4 total): teacher/student on source/target domains
  - MUST metrics summary: 4-panel visualization showing teacher/student performance and pseudo-label usage

### 2. Best Model Checkpointing

#### Standard Training
- Saves `best_model.pth` based on highest validation accuracy
- Checkpoint includes:
  - Model state dict
  - Optimizer state dict
  - Best validation accuracy and loss
  - Epoch number

#### MUST Training
- Saves `best_teacher.pth` and `best_student.pth` separately
- Each checkpoint includes:
  - Model state dict
  - Domain-specific batch normalization parameters (source and target)
  - Best target accuracy and source accuracy
  - Iteration number

### 3. Experiment Structure

Each experiment now has a **clean, self-contained directory** with everything you need:

```
exps/
└── experiment_name/
    └── test_split_name/
        ├── config.json              # Saved configuration
        ├── weights/
        │   ├── best_model.pth       # Best model checkpoint (standard training)
        │   ├── best_teacher.pth     # Best teacher checkpoint (MUST)
        │   ├── best_student.pth     # Best student checkpoint (MUST)
        │   └── model_weights_epoch_*.pth  # Per-epoch checkpoints (optional)
        └── tensorboard/
            └── events.out.tfevents.*  # TensorBoard logs (scalars + images)
```

**Note**: No more `plots/` directory! All visualizations (confusion matrices, metric plots) are now in TensorBoard.

## How to Use

### 1. Run Training

Training scripts now support **automatic resume** from the last checkpoint:

```bash
# Start training (or resume from last checkpoint if exists)
python simple_model_train.py --test_index 0

# Override existing experiment and start fresh
python simple_model_train.py --test_index 0 --override

# With resampling
python simple_model_train.py --test_index 0 --use_cheat_to_match_target_label_distribution

# MUST training
python simple_model_train.py --test_index 0 --use_must

# Resume MUST training (or override to start fresh)
python simple_model_train.py --test_index 0 --use_must [--override]
```

**Resume Behavior:**
- **Without `--override`**: Automatically detects and resumes from the latest checkpoint if the experiment directory exists
- **With `--override`**: Deletes existing experiment directory and starts fresh training
- Standard training resumes from the last epoch checkpoint
- MUST training resumes from the last iteration checkpoint
- Training history and metrics are preserved when resuming

### 2. View TensorBoard

After training starts, launch TensorBoard to monitor progress:

```bash
# For a specific experiment
tensorboard --logdir exps/experiment_name/test_split_name/tensorboard

# For all experiments (compare multiple runs)
tensorboard --logdir exps/

# Specify port if default is occupied
tensorboard --logdir exps/ --port 6007
```

Then open your browser to `http://localhost:6006` (or the specified port).

### 3. Load Best Model

To load the best model for evaluation:

```python
import torch
from models.configurable_cnn import ConfigurableCNN

# Standard training
checkpoint = torch.load('exps/experiment_name/test_split/weights/best_model.pth')
model = ConfigurableCNN(model_params)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
print(f"Achieved at epoch: {checkpoint['epoch']}")

# MUST training
checkpoint = torch.load('exps/experiment_name/test_split/weights/best_student.pth')
model = ConfigurableCNN(model_params)
model.load_state_dict(checkpoint['model_state_dict'])
# For target domain, also load target BN parameters
from training.must_trainer import inject_bn_params
inject_bn_params(model, checkpoint['bn_target'])
print(f"Best target accuracy: {checkpoint['target_acc']:.2f}%")
print(f"Achieved at iteration: {checkpoint['iteration']}")
```

## TensorBoard Features

### Scalars Tab
- View loss and accuracy trends over time
- Compare multiple experiments side-by-side
- Smooth curves with adjustable smoothing factor
- Example useful tags:
  - `Loss/*` - All loss metrics
  - `Accuracy/*` - All accuracy metrics
  - `MUST/*` - MUST-specific metrics (if using MUST training)

### Images Tab
- **Confusion matrices**: View model predictions on validation/test sets
  - Standard training: `Confusion_Matrix/validation`
  - MUST training: `Confusion_Matrix/teacher_source`, `teacher_target`, `student_source`, `student_target`
- **Training curves**: Summary plots at the end of training
  - Standard training: `Training_Metrics/summary`
  - MUST training: `MUST_Metrics/summary` (4-panel visualization)

### Tips
- **Compare runs**: Select multiple experiments in the left sidebar to overlay curves
- **Download data**: Export metrics as CSV for further analysis
- **Zoom**: Click and drag on plots to zoom into specific regions
- **Toggle smoothing**: Adjust the smoothing slider for cleaner trend lines

## Configuration Changes

The experiment name in `config.py` has been updated to:
```python
EXPERIMENT_NAME: str = "allot_daily_degradation_v14_tensorboard/{}"
```

You can customize this as needed for your experiments.

## Benefits

1. **Clean Experiment Folders**: Everything you need in one place - config, best weights, and TensorBoard logs
2. **Real-time Monitoring**: View training progress without waiting for completion
3. **Better Comparison**: Easily compare multiple experiments side-by-side
4. **Automatic Best Model**: No need to manually track which epoch performed best
5. **Automatic Resume**: Training can be interrupted and resumed without losing progress
6. **Reproducibility**: Config saved with each experiment for full reproducibility
7. **Rich Visualizations**: Interactive plots and confusion matrices in TensorBoard
8. **Remote Access**: Can tunnel TensorBoard to view training on remote machines
9. **No PNG Clutter**: All visualizations in TensorBoard, no scattered image files

## Notes

- TensorBoard logs (scalars + images) are saved automatically during training
- Best model selection is based on validation/target accuracy
- All checkpoints include full state for resuming training if needed
- Config is automatically saved to `config.json` in each experiment directory
- The `plots/` directory is no longer created - everything is in TensorBoard!
- Resume functionality works automatically - just rerun the same command to continue training
- Use `--override` flag to force restart an experiment from scratch
