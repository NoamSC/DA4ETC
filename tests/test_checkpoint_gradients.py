"""
Load a trained checkpoint and examine gradients before/after the GRL layer.
This will show us if the GRL is actually affecting gradients during actual training.
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
from config import Config
from models.configurable_cnn import ConfigurableCNN
from data_utils.csv_dataloader import create_csv_flowpic_loader

print("=" * 120)
print("TESTING GRADIENTS BEFORE/AFTER GRL IN TRAINED MODEL")
print("=" * 120)

# Load the crazy_rgl experiment checkpoint
exp_path = Path('exps/cesnet_v4_dann_sanity/normal_v13_grl_effect_test_crazy_rgl_WEEK-2022-33')
checkpoint_path = exp_path / 'weights' / 'model_weights_epoch_10.pth'

# Load config from that experiment
with open(exp_path / 'config.json', 'r') as f:
    saved_config = json.load(f)

print(f"\nLoaded experiment config:")
print(f"  lambda_rgl = {saved_config['MODEL_PARAMS']['lambda_rgl']}")
print(f"  lambda_grl_gamma = {saved_config['MODEL_PARAMS']['lambda_grl_gamma']}")
print(f"  LAMBDA_DANN = {saved_config['LAMBDA_DANN']}")

# Create model with same config
cfg = Config()
cfg.MODEL_PARAMS = saved_config['MODEL_PARAMS']
cfg.MODEL_PARAMS['num_classes'] = 180
cfg.RESOLUTION = saved_config['RESOLUTION']
cfg.SEED = saved_config['SEED']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create model
model = ConfigurableCNN(cfg.MODEL_PARAMS).to(device)

# Load checkpoint
print(f"\nLoading checkpoint: {checkpoint_path}")
if not checkpoint_path.exists():
    print(f"ERROR: Checkpoint not found!")
    print(f"Available checkpoints:")
    for ckpt in sorted((exp_path / 'weights').glob('*.pth')):
        print(f"  {ckpt.name}")
    exit(1)

state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)

# Set epoch to match epoch 10 (normalized epoch = 10/50 = 0.2)
model.set_epoch(10 / 50)
print(f"\nSet model epoch to 10/50 = 0.2")
print(f"GRL lambda = {model.grl.lambda_}")

# Debug: Check model architecture
print(f"\n" + "=" * 80)
print("MODEL ARCHITECTURE CHECK")
print("=" * 80)
print(f"\nDoes model have GRL? {hasattr(model, 'grl')}")
print(f"GRL type: {type(model.grl)}")
print(f"GRL lambda_: {model.grl.lambda_}")

# Check if domain_classifier exists and is connected through GRL
print(f"\nModel forward method uses:")
print(f"  - feature_extractor: {hasattr(model, 'feature_extractor')}")
print(f"  - grl: {hasattr(model, 'grl')}")
print(f"  - domain_classifier: {hasattr(model, 'domain_classifier_convs') or hasattr(model, 'domain_classifier')}")

# Let's trace through a forward pass manually
print(f"\nTracing forward pass for domain prediction:")
with torch.no_grad():
    dummy_input = source_inputs[:2]  # Just 2 samples
    print(f"  1. Input shape: {dummy_input.shape}")

    # Through feature extractor
    features = model.feature_extractor(dummy_input)
    features = features.view(features.size(0), -1)
    print(f"  2. After feature_extractor: {features.shape}, requires_grad={features.requires_grad}")

    # Through GRL
    grl_output = model.grl(features)
    print(f"  3. After GRL: {grl_output.shape}, requires_grad={grl_output.requires_grad}")
    print(f"     Is same tensor as features? {grl_output is features}")

    # Check if grl_output is in computation graph
    print(f"     Has grad_fn? {grl_output.grad_fn is not None}")

# Load data
dataset_root = Path('/home/anatbr/dataset/CESNET-TLS-Year22_v1')
with open(dataset_root / 'label_mapping.json', 'r') as f:
    label_mapping = json.load(f)

print(f"\nLoading data...")
train_loader = create_csv_flowpic_loader(
    [dataset_root / 'WEEK-2022-33' / 'train.parquet'],
    batch_size=64,
    num_workers=0,
    shuffle=True,
    resolution=cfg.RESOLUTION,
    data_sample_frac=0.001,
    seed=cfg.SEED,
    label_mapping=label_mapping,
    log_t_axis=False,
    max_dt_ms=4000,
    dataset_format='cesnet_parquet',
    verbose=False
)

test_loader = create_csv_flowpic_loader(
    [dataset_root / 'WEEK-2022-40' / 'test.parquet'],
    batch_size=64,
    num_workers=0,
    shuffle=False,
    resolution=cfg.RESOLUTION,
    data_sample_frac=0.001,
    seed=cfg.SEED,
    label_mapping=label_mapping,
    log_t_axis=False,
    max_dt_ms=4000,
    dataset_format='cesnet_parquet',
    verbose=False
)

# Get batches
source_batch = next(iter(train_loader))
target_batch = next(iter(test_loader))

source_inputs, source_labels = source_batch
target_inputs, _ = target_batch

source_inputs = source_inputs.to(device)
target_inputs = target_inputs.to(device)
source_labels = source_labels.to(device).long()

print(f"Batch sizes: source={len(source_inputs)}, target={len(target_inputs)}")

# ============================================================================
# TEST 1: Examine gradients from domain loss through the GRL
# ============================================================================
print("\n" + "=" * 120)
print("TEST 1: Gradients from DOMAIN LOSS through GRL")
print("=" * 120)

model.train()

# Forward pass
source_outputs = model(source_inputs)
target_outputs = model(target_inputs)

# Create domain labels
criterion = nn.CrossEntropyLoss()
source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

# Compute domain loss
dann_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
dann_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

lambda_dann = saved_config['LAMBDA_DANN']
domain_loss_weighted = lambda_dann * dann_loss

print(f"\nDomain loss: {dann_loss.item():.4f}")
print(f"Weighted domain loss (lambda_dann={lambda_dann}): {domain_loss_weighted.item():.4f}")

# Domain classifier accuracy
with torch.no_grad():
    _, source_pred = source_outputs['domain_preds'].max(1)
    _, target_pred = target_outputs['domain_preds'].max(1)
    source_correct = source_pred.eq(source_domain_labels).sum().item()
    target_correct = target_pred.eq(target_domain_labels).sum().item()
    domain_acc = (source_correct + target_correct) / (len(source_domain_labels) + len(target_domain_labels)) * 100
    print(f"Domain classifier accuracy: {domain_acc:.2f}%")

# Backward through domain loss
model.zero_grad()
domain_loss_weighted.backward()

# Check gradients at different points
print(f"\nGradient norms at different network layers:")

# 1. Domain classifier (after domain loss, before GRL)
domain_classifier_grad = 0
for name, param in model.named_parameters():
    if 'domain_classifier' in name and param.grad is not None:
        domain_classifier_grad += param.grad.norm().item() ** 2
domain_classifier_grad = domain_classifier_grad ** 0.5
print(f"  1. Domain classifier (before GRL):  {domain_classifier_grad:.6f}")

# 2. Feature extractor (after GRL)
feature_extractor_grad = 0
for name, param in model.named_parameters():
    if 'feature_extractor' in name and param.grad is not None:
        feature_extractor_grad += param.grad.norm().item() ** 2
feature_extractor_grad = feature_extractor_grad ** 0.5
print(f"  2. Feature extractor (after GRL):   {feature_extractor_grad:.6f}")

# 3. Check if gradient was reversed
# The gradient should be LARGER in feature extractor if GRL is amplifying (lambda > 1)
# or SMALLER if GRL is attenuating (lambda < 1)
print(f"\nGRL lambda: {model.grl.lambda_}")
print(f"Expected gradient ratio (feature/domain): ~{model.grl.lambda_:.2f}")
if domain_classifier_grad > 0:
    actual_ratio = feature_extractor_grad / domain_classifier_grad
    print(f"Actual gradient ratio: {actual_ratio:.2f}")

    if abs(actual_ratio - model.grl.lambda_) / model.grl.lambda_ < 0.5:
        print(f"✓ Gradient scaling matches GRL lambda (within 50%)")
    else:
        print(f"⚠️  Gradient scaling DOES NOT match GRL lambda!")

# ============================================================================
# TEST 2: Compare gradients from classification vs domain loss
# ============================================================================
print("\n" + "=" * 120)
print("TEST 2: Gradient magnitude comparison")
print("=" * 120)

# Need fresh forward pass
model.zero_grad()
source_outputs_fresh = model(source_inputs)
target_outputs_fresh = model(target_inputs)

# Classification loss gradients
classification_loss = criterion(source_outputs_fresh['class_preds'], source_labels)
classification_loss.backward()

class_grad_feature = 0
for name, param in model.named_parameters():
    if 'feature_extractor' in name and param.grad is not None:
        class_grad_feature += param.grad.norm().item() ** 2
class_grad_feature = class_grad_feature ** 0.5

print(f"\nFeature extractor gradient norms:")
print(f"  From classification loss: {class_grad_feature:.6f}")
print(f"  From domain loss:        {feature_extractor_grad:.6f}")
print(f"  Ratio (domain/class):    {feature_extractor_grad / class_grad_feature:.4f}")

# ============================================================================
# TEST 3: Check gradients from domain loss AGAIN (debug)
# ============================================================================
print("\n" + "=" * 120)
print("TEST 3: Re-test domain loss gradients with fresh forward pass")
print("=" * 120)

model.zero_grad()

# Fresh forward pass
source_outputs_test3 = model(source_inputs)
target_outputs_test3 = model(target_inputs)

# Recompute domain loss
source_domain_labels_test3 = torch.zeros(len(source_outputs_test3['domain_preds'])).long().to(device)
target_domain_labels_test3 = torch.ones(len(target_outputs_test3['domain_preds'])).long().to(device)

dann_loss_test3 = criterion(source_outputs_test3['domain_preds'], source_domain_labels_test3)
dann_loss_test3 += criterion(target_outputs_test3['domain_preds'], target_domain_labels_test3)

print(f"Domain loss: {dann_loss_test3.item():.4f}")

# Backward
dann_loss_test3.backward()

# Check gradients
domain_grad_test3 = 0
feature_grad_test3 = 0

for name, param in model.named_parameters():
    if param.grad is not None:
        if 'domain_classifier' in name:
            domain_grad_test3 += param.grad.norm().item() ** 2
        if 'feature_extractor' in name:
            feature_grad_test3 += param.grad.norm().item() ** 2

domain_grad_test3 = domain_grad_test3 ** 0.5
feature_grad_test3 = feature_grad_test3 ** 0.5

print(f"\nGradient norms (re-test):")
print(f"  Domain classifier:  {domain_grad_test3:.6f}")
print(f"  Feature extractor:  {feature_grad_test3:.6f}")

if feature_grad_test3 == 0:
    print(f"\n❌ CRITICAL BUG CONFIRMED!")
    print(f"   Feature extractor gets ZERO gradients from domain loss!")
    print(f"   The GRL is NOT working in the trained model!")

# ============================================================================
# TEST 4: Check what happens with TOTAL loss (as in actual training)
# ============================================================================
print("\n" + "=" * 120)
print("TEST 4: Total loss gradients (as in actual training)")
print("=" * 120)

model.zero_grad()

# Fresh forward pass
source_outputs = model(source_inputs)
target_outputs = model(target_inputs)

classification_loss = criterion(source_outputs['class_preds'], source_labels)

dann_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
dann_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

total_loss = classification_loss + lambda_dann * dann_loss

print(f"\nLoss components:")
print(f"  Classification loss: {classification_loss.item():.4f}")
print(f"  DANN loss:          {dann_loss.item():.4f}")
print(f"  Total loss:         {total_loss.item():.4f}")

total_loss.backward()

total_grad_feature = 0
for name, param in model.named_parameters():
    if 'feature_extractor' in name and param.grad is not None:
        total_grad_feature += param.grad.norm().item() ** 2
total_grad_feature = total_grad_feature ** 0.5

print(f"\nFeature extractor gradient norm from total loss: {total_grad_feature:.6f}")
print(f"Sum of individual gradients: {class_grad_feature + feature_extractor_grad:.6f}")

# Check if gradients are additive (they should be approximately)
diff_pct = abs(total_grad_feature - (class_grad_feature + feature_extractor_grad)) / total_grad_feature * 100
if diff_pct < 10:
    print(f"✓ Gradients are approximately additive (diff: {diff_pct:.1f}%)")
else:
    print(f"⚠️  Gradients are NOT additive (diff: {diff_pct:.1f}%)")

print("\n" + "=" * 120)
print("SUMMARY")
print("=" * 120)

print(f"\nKey findings:")
print(f"1. Domain classifier is at {domain_acc:.1f}% accuracy")
if abs(domain_acc - 50) < 2:
    print(f"   → Domain classifier is NOT learning (stuck at random)")
    print(f"   → This explains why all experiments produce identical results!")
else:
    print(f"   → Domain classifier IS learning")

print(f"\n2. GRL lambda = {model.grl.lambda_:.2f}")
print(f"   Domain loss gradients are {feature_extractor_grad/class_grad_feature*100:.1f}% of classification gradients")

if feature_extractor_grad / class_grad_feature < 0.01:
    print(f"   → Domain gradients are NEGLIGIBLE compared to classification")
    print(f"   → This explains why lambda_rgl has no effect!")
elif feature_extractor_grad / class_grad_feature > 1.0:
    print(f"   → Domain gradients DOMINATE classification gradients")
    print(f"   → Different lambda_rgl should produce very different results")
else:
    print(f"   → Domain and classification gradients are comparable")
