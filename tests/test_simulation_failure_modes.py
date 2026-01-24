"""
Test to reproduce the 'Double Fault' theory:
1. Late configuration update (Model gets default lambda)
2. Wrapper blocking set_epoch (Lambda never changes)
"""
import torch
import torch.nn as nn
from config import Config
from models.configurable_cnn import ConfigurableCNN

print("=" * 80)
print("TESTING TRAIN LOOP FAILURE MODES")
print("=" * 80)

# 1. Inspect Defaults
cfg = Config()
default_lambda = cfg.MODEL_PARAMS.get('lambda_rgl', 'NOT_SET')
print(f"Default lambda_rgl in Config: {default_lambda}")

# 2. Simulate the "Late Config" Bug
# Many users do:
#   model = Model(cfg)
#   cfg.lambda_rgl = args.lambda_rgl  <-- TOO LATE!
print("\n[Scenario A] Simulating Late Config Update:")
cfg.MODEL_PARAMS['num_classes'] = 180
model = ConfigurableCNN(cfg.MODEL_PARAMS) # Init with default

# "User" tries to set it to 10000
target_lambda = 10000.0
cfg.MODEL_PARAMS['lambda_rgl'] = target_lambda 
print(f"  > User updated config to: {target_lambda}")
print(f"  > Actual Model lambda_:   {model.grl.lambda_}")

if model.grl.lambda_ != target_lambda:
    print("  ❌ FAIL: Model ignored the config update (Init happened too early)")
    current_lambda = model.grl.lambda_
else:
    print("  ✓ PASS: Config update worked")
    current_lambda = model.grl.lambda_

# 3. Simulate the "Wrapper" Bug
if torch.cuda.is_available():
    model = nn.DataParallel(model).cuda()
    print(f"\n[Scenario B] Wrapped model in DataParallel")

print("\n[Scenario C] Simulating Training Loop (set_epoch):")
try:
    # Typical training loop check
    if hasattr(model, 'set_epoch'):
        model.set_epoch(0.5)
        print("  ✓ model.set_epoch() called")
    else:
        print("  ⚠️  model.set_epoch() SKIPPED (Wrapper hides method)")
except Exception as e:
    print(f"  ❌ Error calling set_epoch: {e}")

# 4. Final Verdict
# Access internal lambda to see what we trained with
if isinstance(model, nn.DataParallel):
    final_lambda = model.module.grl.lambda_
else:
    final_lambda = model.grl.lambda_

print("-" * 40)
print(f"FINAL EFFECTIVE LAMBDA: {final_lambda}")
print("-" * 40)

if final_lambda < 100 and target_lambda == 10000:
    print("\n🕵️‍♂️ MYSTERY SOLVED:")
    print("   The model trained with lambda ~0 (Default) instead of 10000.")
    print("   1. 'Late Config' kept init value low.")
    print("   2. 'Wrapper Bug' prevented schedule from increasing it.")
    print("   -> Result: Bit-exact runs, no divergence, ignored settings.")