"""
Debug script to trace the complete flow from config → model → GRL lambda.
This will help us understand if lambda_rgl is being used correctly.
"""
import torch
from config import Config
from models.configurable_cnn import ConfigurableCNN

print("=" * 80)
print("DEBUGGING CONFIG FLOW")
print("=" * 80)

# Create config
cfg = Config()

print("\n1. Initial cfg.MODEL_PARAMS['lambda_rgl']:")
print(f"   {cfg.MODEL_PARAMS.get('lambda_rgl', 'NOT SET')}")

# Simulate command-line override
print("\n2. Simulating command-line override:")
lambda_rgl_override = 10000.0
cfg.MODEL_PARAMS['lambda_rgl'] = lambda_rgl_override
print(f"   cfg.MODEL_PARAMS['lambda_rgl'] = {cfg.MODEL_PARAMS['lambda_rgl']}")

# Add required params
cfg.MODEL_PARAMS['num_classes'] = 180
if 'lambda_grl_gamma' not in cfg.MODEL_PARAMS:
    cfg.MODEL_PARAMS['lambda_grl_gamma'] = 10

# Create model
print("\n3. Creating model with cfg.MODEL_PARAMS:")
print(f"   lambda_rgl = {cfg.MODEL_PARAMS['lambda_rgl']}")
print(f"   lambda_grl_gamma = {cfg.MODEL_PARAMS['lambda_grl_gamma']}")

model = ConfigurableCNN(cfg.MODEL_PARAMS)

print("\n4. After model creation:")
print(f"   model.params['lambda_rgl'] = {model.params['lambda_rgl']}")
print(f"   model.params['lambda_grl_gamma'] = {model.params['lambda_grl_gamma']}")
print(f"   model.grl.lambda_ = {model.grl.lambda_}")

# Now modify cfg.MODEL_PARAMS to see if model.params is a reference
print("\n5. Testing if model.params is a reference to cfg.MODEL_PARAMS:")
cfg.MODEL_PARAMS['lambda_rgl'] = 5.0
print(f"   Changed cfg.MODEL_PARAMS['lambda_rgl'] to 5.0")
print(f"   model.params['lambda_rgl'] = {model.params['lambda_rgl']}")
print(f"   Are they the same object? {cfg.MODEL_PARAMS is model.params}")

# Test set_epoch
print("\n6. Testing set_epoch at different epochs:")
epochs_to_test = [0.0, 0.02, 0.1, 0.5, 1.0]
print(f"   {'Epoch':>6s}  {'Factor':>8s}  {'GRL Lambda':>12s}")
print(f"   {'-'*6}  {'-'*8}  {'-'*12}")

for epoch in epochs_to_test:
    model.set_epoch(epoch)
    # Calculate factor manually to verify
    gamma = model.params['lambda_grl_gamma']
    factor = (2 / (1 + torch.exp(torch.tensor(-gamma * epoch))) - 1).item()
    expected_lambda = model.params['lambda_rgl'] * factor

    print(f"   {epoch:6.1f}  {factor:8.4f}  {model.grl.lambda_:12.6f} (expected: {expected_lambda:.6f})")

# Verify model.params is not shared
print(f"\n7. Checking if cfg.MODEL_PARAMS is shared with model.params:")
print(f"   cfg.MODEL_PARAMS is model.params: {cfg.MODEL_PARAMS is model.params}")
print(f"   id(cfg.MODEL_PARAMS) = {id(cfg.MODEL_PARAMS)}")
print(f"   id(model.params) = {id(model.params)}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if cfg.MODEL_PARAMS is model.params:
    print("\n❌ WARNING: cfg.MODEL_PARAMS and model.params are THE SAME OBJECT!")
    print("   This means changes to cfg.MODEL_PARAMS will affect model.params")
    print("   If multiple models share cfg.MODEL_PARAMS, they'll interfere with each other!")
else:
    print("\n✓ cfg.MODEL_PARAMS and model.params are different objects")
    print("  This is correct - models won't interfere with each other")
