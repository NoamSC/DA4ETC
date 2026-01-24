"""
Test if model.set_epoch() actually reaches the GRL layer when wrapped in DataParallel.
This is a common cause for "bit-exact" runs despite config changes.
"""
import torch
import torch.nn as nn
from config import Config
from models.configurable_cnn import ConfigurableCNN

print("=" * 80)
print("TESTING WRAPPED MODEL UPDATE PROPAGATION")
print("=" * 80)

# 1. Setup
cfg = Config()
cfg.MODEL_PARAMS['num_classes'] = 10
# Set a HUGE lambda to make sure we see the difference clearly
cfg.MODEL_PARAMS['lambda_rgl'] = 10000.0 
cfg.MODEL_PARAMS['lambda_grl_gamma'] = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 2. Initialize Model
base_model = ConfigurableCNN(cfg.MODEL_PARAMS).to(device)

# 3. Wrap it (Simulating actual training conditions)
# NOTE: Even with 1 GPU, DataParallel wraps the object structure
if torch.cuda.device_count() >= 1:
    print("Wrapping model in DataParallel...")
    model = nn.DataParallel(base_model)
else:
    print("WARNING: No GPU found. Simulating wrapper manually.")
    class MockWrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x): return self.module(x)
    model = MockWrapper(base_model)

print(f"\nModel type: {type(model)}")
print(f"Inner model type: {type(model.module)}")

# 4. Check Initial State
print(f"\nInitial GRL Lambda (via inner module): {model.module.grl.lambda_}")

# 5. Attempt to update epoch
print("\nAttempting to call set_epoch(0.5)...")

success = False
try:
    # Scenario A: Calling directly on wrapper (Standard mistake)
    if hasattr(model, 'set_epoch'):
        model.set_epoch(0.5)
        print("  ✓ Called model.set_epoch() directly")
        success = True
    else:
        print("  ⚠️  Wrapper does NOT have 'set_epoch' method!")
        
        # Scenario B: How it should be done
        if hasattr(model.module, 'set_epoch'):
            print("  ℹ️  Calling model.module.set_epoch() instead...")
            model.module.set_epoch(0.5)
            success = True
        else:
             print("  ❌ Inner module also missing set_epoch!")

except Exception as e:
    print(f"  ❌ Exception during call: {e}")

# 6. Verify if the update actually happened
actual_lambda = model.module.grl.lambda_

# Calculate expected value
# Standard schedule: 2 / (1 + exp(-gamma * p)) - 1
# Multiplied by lambda_rgl
import math
p = 0.5
gamma = 10.0
standard_sched = 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
expected_lambda = standard_sched * cfg.MODEL_PARAMS['lambda_rgl']

print("\n" + "-" * 40)
print(f"RESULTS Check:")
print(f"  Config lambda_rgl: {cfg.MODEL_PARAMS['lambda_rgl']}")
print(f"  Expected GRL lambda: {expected_lambda:.4f}")
print(f"  Actual GRL lambda:   {actual_lambda:.4f}")
print("-" * 40)

if abs(actual_lambda - expected_lambda) < 1e-3:
    print("\n✓ SUCCESS: Update propagated correctly.")
    print("  This means the wrapper is NOT the issue (or you are handling it correctly).")
else:
    print("\n❌ FAILURE: Lambda did not update!")
    print("  ROOT CAUSE FOUND: The training loop is likely calling set_epoch on the")
    print("  DataParallel wrapper, which is ignoring it or failing silently.")
    print("\n  FIX: In your training loop, change:")
    print("    model.set_epoch(epoch)")
    print("  To:")
    print("    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):")
    print("        model.module.set_epoch(epoch)")
    print("    else:")
    print("        model.set_epoch(epoch)")