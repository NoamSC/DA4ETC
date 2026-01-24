# DANN Gradient Flow Test Suite

This test suite helps diagnose common silent bugs in Domain-Adversarial Neural Network (DANN) implementations by verifying proper gradient flow and GRL functionality.

## What This Tests

The test suite performs 4 critical tests:

### 1. **Gradient Flow Verification**
- Computes domain loss and backpropagates
- Measures gradient norms in feature extractor layers
- **Pass criteria**: Gradient norm > 0.01
- **Failure indicates**: GRL not connected, computation graph broken, or domain classifier not using feature extractor outputs

### 2. **Domain Logits Statistics**
- Monitors mean and standard deviation of domain predictions
- **Pass criteria**: Logits std > 0.1
- **Failure indicates**: Domain classifier not learning, disconnected graph, or initialization issues

### 3. **Frozen Features Domain Classifier**
- Freezes feature extractor and trains domain classifier alone
- Tests if domain signal exists in the data
- **Pass criteria**:
  - Accuracy > 70% → Strong domain signal (if DANN fails, issue is GRL/loss weights/optimizer)
  - Accuracy 55-70% → Weak domain signal
  - Accuracy < 55% → No domain signal (data/labels/architecture issue)

### 4. **GRL Sign Verification**
- Compares gradients with and without GRL
- Verifies gradient reversal (cosine similarity should be < -0.5)
- **Pass criteria**: All feature extractor layers show negative cosine similarity
- **Failure indicates**: GRL implementation has wrong sign or is not applied

## Usage

### Basic Usage

```bash
python test_dann_gradient_flow.py
```

This uses default parameters:
- Dataset: `/home/anatbr/dataset/CESNET-TLS-Year22_v1`
- Source week: 33 (WEEK-2022-33)
- Target week: 40 (WEEK-2022-40)
- Data fraction: 1% (for fast testing)

### Custom Parameters

```bash
python test_dann_gradient_flow.py \
    --dataset_root /path/to/CESNET-TLS-Year22_v1 \
    --source_week 33 \
    --target_week 40 \
    --data_frac 0.01 \
    --checkpoint /path/to/model_weights.pth \
    --num_frozen_batches 100
```

### Arguments

- `--dataset_root`: Path to CESNET-TLS-Year22 dataset root (default: `/home/anatbr/dataset/CESNET-TLS-Year22_v1`)
- `--source_week`: Source week number (default: 33)
- `--target_week`: Target week number (default: 40)
- `--data_frac`: Fraction of data to use (default: 0.01 = 1%)
- `--checkpoint`: Path to trained model checkpoint (optional, **recommended for accurate Test 3 results**)
- `--num_frozen_batches`: Number of batches for frozen features test (default: 100)

**Important**: For Test 3 (Frozen Features) to be meaningful, you should load a **trained** model checkpoint. With random initialization, features contain no domain information, so the test will show ~50% accuracy regardless of domain shift.

## Output

The script generates:

1. **Console output**: Detailed test results with pass/fail indicators
2. **Visualization**: `test_results/WEEK-2022-XX_to_WEEK-2022-YY/dann_gradient_flow_tests.png`
   - Domain classifier accuracy during frozen features training
   - Domain loss curve

## Example Output

```
================================================================================
DANN GRADIENT FLOW TEST SUITE
================================================================================

Model: ConfigurableCNN
Device: cuda:0
GRL Lambda: 0.761594

================================================================================
TEST 1: Gradient Flow Verification
================================================================================

Gradient norms in feature extractor:
  feature_extractor.0.weight                    :     0.012345
  feature_extractor.0.bias                      :     0.001234
  ...

  TOTAL FEATURE EXTRACTOR GRADIENT NORM         :     0.123456

  ✓ PASS: Gradients are flowing to feature extractor

================================================================================
TEST 2: Domain Logits Statistics
================================================================================

Source domain logits:
  Mean:     0.1234
  Std:      0.5678

Target domain logits:
  Mean:     0.2345
  Std:      0.6789

  ✓ PASS: Logits show reasonable variance

================================================================================
TEST 3: Domain Classifier with Frozen Features
================================================================================

Training domain classifier with frozen feature extractor...
Using 50 batches, learning rate = 0.001

Final domain classifier accuracy (frozen features): 85.23%

  ✓ PASS: Domain classifier can distinguish domains!
  If DANN is not working, the issue is likely:
    - GRL scheduling (lambda too small/large)
    - Loss weight balance (lambda_dann)
    - Optimizer dynamics (learning rate, momentum)

================================================================================
TEST 4: GRL Sign Verification
================================================================================

Checking gradient sign reversal:
  ✓ feature_extractor.0.weight                    : cosine_sim =  -0.9876
  ✓ feature_extractor.0.bias                      : cosine_sim =  -0.9543
  ...

  ✓ PASS: GRL correctly reverses gradients

================================================================================
TEST SUMMARY
================================================================================

1. Gradient Flow:
   - Feature extractor gradient norm: 0.123456
   - Status: ✓ PASS

2. Domain Logits Statistics:
   - Logits std: 0.5678
   - Status: ✓ PASS

3. Frozen Features Domain Classifier:
   - Final accuracy: 85.23%
   - Status: ✓ PASS (strong domain signal)

4. GRL Sign Verification:
   - Status: ✓ PASS
```

## Interpreting Results

### Important Note About Your Current Results

Your test run showed:
- **GRL Lambda = 0.00053** (extremely small!)
- All tests failed due to tiny gradients

**Root Cause**: The GRL lambda scheduling starts too small. At epoch 0.5:
- `lambda_rgl` = 0.002168 (from config)
- Scheduling factor ≈ 0.245
- Result: 0.002168 × 0.245 ≈ **0.00053**

**Recommendations**:
1. **Increase `lambda_rgl`** in [config.py:84](config.py#L84) to at least 0.01-0.1
2. **Adjust `lambda_grl_gamma`** in [config.py:85](config.py#L85) (lower values = faster ramp-up)
3. **Test with a trained checkpoint** to verify Test 3 properly

### All Tests Pass
Your DANN implementation is correct! If training still doesn't work:
- Tune `lambda_dann` (domain loss weight in [config.py:98](config.py#L98))
- Tune `lambda_rgl` (GRL strength in [config.py:84](config.py#L84))
- Adjust `lambda_grl_gamma` (GRL scheduling parameter in [config.py:85](config.py#L85))
- Check optimizer learning rate

### Test 1 Fails (No Gradient Flow)
**Problem**: Domain loss not connected to feature extractor
**Solutions**:
- Verify GRL is applied to feature extractor output
- Check that domain classifier uses GRL-transformed features
- Ensure no `.detach()` calls breaking the computation graph

### Test 2 Fails (Low Logits Std)
**Problem**: Domain classifier not learning
**Solutions**:
- Check learning rate (might be too low)
- Verify initialization
- Ensure domain labels are correct

### Test 3 Fails (Low Frozen Accuracy)
**Problem**: Insufficient domain signal
**Solutions**:
- Verify source/target data are actually different domains
- Check data loading (labels might be wrong)
- Increase domain classifier capacity
- Use different source/target week pairs with more domain shift

### Test 4 Fails (Wrong Sign)
**Problem**: GRL not reversing gradients
**Solutions**:
- Check `GradientReversalFunction.backward()` implementation
- Ensure gradient is multiplied by `-lambda_` not `+lambda_`
- Verify GRL is properly registered as an autograd function

## Files

- `test_dann_gradient_flow.py`: Main test script
- `models/configurable_cnn.py`: Contains GRL and DANN model
- `training/trainer.py`: Training loop with DANN support
- `config.py`: Configuration including `lambda_rgl`, `lambda_dann`, etc.
