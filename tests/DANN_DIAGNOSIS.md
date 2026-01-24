# DANN Gradient Flow Test Results & Recommendations

## Test Results Summary

**Test Run**: Week 33 → Week 40 (source → target)
**Model**: Untrained (random initialization)
**GRL Lambda**: 0.00053 ⚠️ **EXTREMELY SMALL**

### All Tests Failed - Root Cause Identified

| Test | Result | Status |
|------|--------|--------|
| 1. Gradient Flow | 0.00054 | ⚠️ Too small |
| 2. Domain Logits Std | 0.042 | ❌ Not learning |
| 3. Frozen Features | 54.59% | ⚠️ Expected (untrained) |
| 4. GRL Sign | 0.0 cosine | ❌ Gradients too small to measure |

## Critical Finding: GRL Lambda Too Small

### Current Configuration (config.py)

```python
'lambda_rgl': 0.002168,        # Line 84
'lambda_grl_gamma': 1,         # Line 85
```

### The Problem

At epoch 0.5, the GRL lambda is computed as:
```
factor = (2 / (1 + exp(-gamma * epoch)) - 1)
       = (2 / (1 + exp(-1 * 0.5)) - 1)
       ≈ 0.245

GRL lambda = lambda_rgl * factor
           = 0.002168 * 0.245
           ≈ 0.00053  ← WAY TOO SMALL!
```

This tiny value means:
- Gradients are reversed but multiplied by ~0.0005
- Effectively no gradient signal reaches the feature extractor
- Domain adaptation cannot work

## Recommended Fixes

### Option 1: Increase `lambda_rgl` (Recommended)

```python
# In config.py line 84
'lambda_rgl': 0.1,  # Increase from 0.002168
```

**Typical values**: 0.01 to 1.0

### Option 2: Adjust Scheduling (`lambda_grl_gamma`)

```python
# In config.py line 85
'lambda_grl_gamma': 10,  # Increase from 1 for faster ramp-up
```

Higher gamma → faster growth → larger lambda earlier in training

### Option 3: Start with Fixed Lambda (Debugging)

Temporarily disable scheduling to test:

```python
# In models/configurable_cnn.py line 148
# Comment out the scheduling:
# self.grl.lambda_ = self.params['lambda_rgl'] * factor

# Use fixed value instead:
self.grl.lambda_ = 1.0  # Or 0.1, 0.5, etc.
```

## GRL Lambda Scheduling Explained

The current schedule: `lambda = lambda_rgl * (2 / (1 + exp(-gamma * p)) - 1)`

Where `p = epoch / num_epochs` goes from 0 → 1

| Epoch Progress | gamma=1 | gamma=10 | gamma=100 |
|----------------|---------|----------|-----------|
| p=0.0 (start)  | 0.000   | 0.000    | 0.000     |
| p=0.1          | 0.099   | 0.761    | 1.000     |
| p=0.5 (middle) | 0.462   | 1.000    | 1.000     |
| p=1.0 (end)    | 0.761   | 1.000    | 1.000     |

Multiplied by `lambda_rgl` to get final GRL lambda.

**Current issue**: `lambda_rgl` is so small that even at p=1.0, GRL lambda is only 0.002168 * 0.761 ≈ 0.00165

## Next Steps

### 1. Fix Configuration

Edit [config.py](config.py):

```python
# Line 84-85
'lambda_rgl': 0.1,        # Increase from 0.002168
'lambda_grl_gamma': 10,   # Increase from 1
```

### 2. Re-run Tests

```bash
python test_dann_gradient_flow.py \
    --source_week 33 \
    --target_week 40 \
    --data_frac 0.01
```

**Expected after fix**:
- ✅ Test 1: Gradient norm > 0.01
- ✅ Test 4: GRL sign correctly reversed
- ⚠️ Test 2 & 3: Still may fail (untrained model)

### 3. Test with Trained Model

To properly validate Test 3 (domain signal exists):

```bash
python test_dann_gradient_flow.py \
    --source_week 33 \
    --target_week 40 \
    --checkpoint exps/your_experiment/weights/best_model.pth
```

**Expected with trained model**:
- ✅ Test 1: Strong gradients
- ✅ Test 2: Diverse logits
- ✅ Test 3: Accuracy > 70% (if domain shift exists)
- ✅ Test 4: Correct sign reversal

## Understanding Test 3 Results

### Untrained Model (Current)
- Random features contain no domain information
- Domain classifier learns ~50% (random)
- **This is expected and OK**

### Trained Model (After Training)
- Features contain task-specific + domain-specific information
- Domain classifier can easily distinguish domains
- If accuracy > 70%: **Strong domain signal exists**
- If accuracy < 60%: **Insufficient domain shift** (try different week pairs)

## Typical DANN Hyperparameters

Based on common DANN implementations:

| Parameter | Typical Range | Your Current | Recommendation |
|-----------|--------------|--------------|----------------|
| `lambda_rgl` | 0.01 - 1.0 | 0.002168 | **0.1** |
| `lambda_grl_gamma` | 10 - 100 | 1 | **10** |
| `lambda_dann` | 0.01 - 1.0 | 0.020626 | 0.02 (OK) |

## Additional Diagnostics

### Check Training Logs

After training, verify in TensorBoard or logs:
1. **Domain classifier accuracy** should be ~50% (good adaptation)
   - Too high (>70%) → domain classifier winning (increase `lambda_rgl`)
   - Too low (<40%) → something wrong
2. **DANN loss** should be non-zero and stable
3. **Target domain accuracy** should improve over epochs

### Monitor During Training

Add to your training loop ([trainer.py](trainer.py)):

```python
# After line 339 (domain_acc printing)
print(f"         GRL Lambda={model.grl.lambda_:.6f}")
print(f"         Domain Grad Norm={domain_grad_norm:.6f}")  # Add grad norm tracking
```

## Reference: DANN Paper Values

From "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016):
- `lambda_p = 2 / (1 + exp(-10 * p)) - 1`
- This corresponds to `lambda_rgl=1.0`, `gamma=10`

Your current settings are ~500x smaller than the paper's recommendation!

## Summary

**The Good News**: Your implementation appears architecturally correct!

**The Issue**: Hyperparameters are severely misconfigured
- GRL lambda 500x too small
- Gradients cannot flow properly

**The Fix**:
1. Change `lambda_rgl` to 0.1
2. Change `lambda_grl_gamma` to 10
3. Re-train and re-test

**Expected Outcome**: DANN should work properly after these changes.
