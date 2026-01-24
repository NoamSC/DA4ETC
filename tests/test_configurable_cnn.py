"""Comprehensive tests for the refactored ConfigurableCNN."""

import torch
import torch.nn as nn
from models.configurable_cnn import ConfigurableCNN, GradientReversalLayer
from config import Config


def test_basic_forward_pass():
    """Test basic forward pass with DANN enabled."""
    print("\n" + "="*80)
    print("TEST 1: Basic Forward Pass")
    print("="*80)

    cfg = Config()
    model_params = cfg.MODEL_PARAMS.copy()
    model_params['num_classes'] = 20

    model = ConfigurableCNN(model_params)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, cfg.RESOLUTION, cfg.RESOLUTION)

    with torch.no_grad():
        output = model(x)

    assert 'class_preds' in output, "Missing class_preds in output"
    assert 'features' in output, "Missing features in output"
    assert 'domain_preds' in output, "Missing domain_preds in output (DANN enabled)"

    assert output['class_preds'].shape == (batch_size, 20), f"Wrong class_preds shape: {output['class_preds'].shape}"
    assert output['domain_preds'].shape == (batch_size, 2), f"Wrong domain_preds shape: {output['domain_preds'].shape}"

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Class predictions shape: {output['class_preds'].shape}")
    print(f"✅ Domain predictions shape: {output['domain_preds'].shape}")
    print(f"✅ Features shape: {output['features'].shape}")


def test_without_dann():
    """Test model without DANN (lambda_rgl = 0)."""
    print("\n" + "="*80)
    print("TEST 2: Model Without DANN")
    print("="*80)

    cfg = Config()
    model_params = cfg.MODEL_PARAMS.copy()
    model_params['num_classes'] = 20
    model_params['lambda_rgl'] = 0  # Disable DANN

    model = ConfigurableCNN(model_params)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, cfg.RESOLUTION, cfg.RESOLUTION)

    with torch.no_grad():
        output = model(x)

    assert 'class_preds' in output, "Missing class_preds in output"
    assert 'features' in output, "Missing features in output"
    assert 'domain_preds' not in output, "domain_preds should not be present when DANN is disabled"

    print(f"✅ DANN disabled successfully")
    print(f"✅ Output keys: {output.keys()}")
    print(f"✅ No domain classifier created")


def test_gradient_reversal_layer():
    """Test that gradient reversal layer works correctly."""
    print("\n" + "="*80)
    print("TEST 3: Gradient Reversal Layer")
    print("="*80)

    grl = GradientReversalLayer(lambda_=1.0)

    # Forward pass
    x = torch.randn(4, 10, requires_grad=True)
    y = grl(x)

    # Check forward is identity
    assert torch.allclose(x, y), "GRL forward should be identity"
    print(f"✅ Forward pass is identity")

    # Backward pass
    loss = y.sum()
    loss.backward()

    # Check gradient is reversed
    expected_grad = -torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad), "GRL should reverse gradients"
    print(f"✅ Backward pass reverses gradients")

    # Test with different lambda
    x2 = torch.randn(4, 10, requires_grad=True)
    grl2 = GradientReversalLayer(lambda_=0.5)
    y2 = grl2(x2)
    loss2 = y2.sum()
    loss2.backward()

    expected_grad2 = -0.5 * torch.ones_like(x2)
    assert torch.allclose(x2.grad, expected_grad2), "GRL should scale reversed gradients by lambda"
    print(f"✅ Lambda scaling works correctly")


def test_epoch_based_lambda_adjustment():
    """Test that GRL lambda adjusts correctly over epochs."""
    print("\n" + "="*80)
    print("TEST 4: Epoch-Based Lambda Adjustment")
    print("="*80)

    cfg = Config()
    model_params = cfg.MODEL_PARAMS.copy()
    model_params['num_classes'] = 20
    model_params['lambda_rgl'] = 1.0
    model_params['lambda_grl_gamma'] = 1.0

    model = ConfigurableCNN(model_params)

    # Test lambda progression
    epochs = [0, 5, 10, 20, 50]
    lambdas = []

    for epoch in epochs:
        model.set_epoch(epoch / 50)
        lambdas.append(model.grl.lambda_)

    print(f"✅ Lambda progression:")
    for epoch, lam in zip(epochs, lambdas):
        print(f"   Epoch {epoch:2d}: λ = {lam:.6f}")

    # Lambda should increase over epochs
    assert all(lambdas[i] <= lambdas[i+1] for i in range(len(lambdas)-1)), \
        "Lambda should increase (or stay same) over epochs"
    print(f"✅ Lambda increases monotonically over epochs")


def test_different_architectures():
    """Test model with different architecture configurations."""
    print("\n" + "="*80)
    print("TEST 5: Different Architectures")
    print("="*80)

    cfg = Config()

    # Test 1: Deeper feature extractor
    params1 = cfg.MODEL_PARAMS.copy()
    params1['num_classes'] = 10
    params1['feature_extractor']['conv_layers'] = [
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    ]

    model1 = ConfigurableCNN(params1)
    x1 = torch.randn(2, cfg.RESOLUTION, cfg.RESOLUTION)
    with torch.no_grad():
        out1 = model1(x1)
    assert out1['class_preds'].shape == (2, 10), "Deep feature extractor failed"
    print(f"✅ Deeper feature extractor (3 layers) works")

    # Test 2: Multiple FC layers in label predictor
    params2 = cfg.MODEL_PARAMS.copy()
    params2['num_classes'] = 5
    params2['label_predictor']['fc_layers'] = [128, 64, 32]

    model2 = ConfigurableCNN(params2)
    x2 = torch.randn(2, cfg.RESOLUTION, cfg.RESOLUTION)
    with torch.no_grad():
        out2 = model2(x2)
    assert out2['class_preds'].shape == (2, 5), "Multiple FC layers failed"
    print(f"✅ Multiple FC layers in label predictor (3 layers) works")

    # Test 3: No conv layers in label predictor
    params3 = cfg.MODEL_PARAMS.copy()
    params3['num_classes'] = 15
    params3['label_predictor']['conv_layers'] = []
    params3['label_predictor']['fc_layers'] = [256, 128]

    model3 = ConfigurableCNN(params3)
    x3 = torch.randn(2, cfg.RESOLUTION, cfg.RESOLUTION)
    with torch.no_grad():
        out3 = model3(x3)
    assert out3['class_preds'].shape == (2, 15), "No conv in label predictor failed"
    print(f"✅ Label predictor with no conv layers works")


def test_get_features():
    """Test the get_features method."""
    print("\n" + "="*80)
    print("TEST 6: Get Features Method")
    print("="*80)

    cfg = Config()
    model_params = cfg.MODEL_PARAMS.copy()
    model_params['num_classes'] = 20

    model = ConfigurableCNN(model_params)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, cfg.RESOLUTION, cfg.RESOLUTION)

    with torch.no_grad():
        features = model.get_features(x)

    # Features should be 3D (batch, channels, spatial)
    assert len(features.shape) == 3, f"Features should be 3D, got shape {features.shape}"
    assert features.shape[0] == batch_size, "Batch dimension mismatch"

    # The spatial dimensions should be reduced after pooling
    original_spatial = cfg.RESOLUTION
    expected_channels = model_params['feature_extractor']['conv_layers'][-1]['out_channels']
    assert features.shape[1] == expected_channels, \
        f"Feature channels should be {expected_channels}, got {features.shape[1]}"

    print(f"✅ Feature shape: {features.shape}")
    print(f"✅ Feature extractor output channels: {expected_channels}")
    print(f"✅ Spatial dimension reduced from {original_spatial} to {features.shape[2]}")


def test_parameter_count():
    """Test that parameter counts are reasonable."""
    print("\n" + "="*80)
    print("TEST 7: Parameter Count")
    print("="*80)

    cfg = Config()
    model_params = cfg.MODEL_PARAMS.copy()
    model_params['num_classes'] = 20

    model = ConfigurableCNN(model_params)

    # Count parameters by component
    feature_params = sum(p.numel() for p in model.feature_extractor.parameters())
    label_conv_params = sum(p.numel() for p in model.label_predictor_convs.parameters())
    label_fc_params = sum(p.numel() for p in model.label_predictor_fcs.parameters())
    label_out_params = sum(p.numel() for p in model.label_output.parameters())

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Feature Extractor:        {feature_params:,} parameters")
    print(f"Label Predictor Convs:    {label_conv_params:,} parameters")
    print(f"Label Predictor FCs:      {label_fc_params:,} parameters")
    print(f"Label Output Layer:       {label_out_params:,} parameters")

    if model_params['lambda_rgl'] > 0:
        domain_conv_params = sum(p.numel() for p in model.domain_classifier_convs.parameters())
        domain_fc_params = sum(p.numel() for p in model.domain_classifier_fcs.parameters())
        domain_out_params = sum(p.numel() for p in model.domain_output.parameters())

        print(f"Domain Classifier Convs:  {domain_conv_params:,} parameters")
        print(f"Domain Classifier FCs:    {domain_fc_params:,} parameters")
        print(f"Domain Output Layer:      {domain_out_params:,} parameters")

    print(f"\nTotal Parameters:         {total_params:,}")

    assert total_params > 0, "Model should have parameters"
    print(f"✅ Model has {total_params:,} trainable parameters")


def test_backward_pass():
    """Test that gradients flow correctly through all components."""
    print("\n" + "="*80)
    print("TEST 8: Backward Pass and Gradient Flow")
    print("="*80)

    cfg = Config()
    model_params = cfg.MODEL_PARAMS.copy()
    model_params['num_classes'] = 20

    model = ConfigurableCNN(model_params)
    model.train()

    batch_size = 4
    x = torch.randn(batch_size, cfg.RESOLUTION, cfg.RESOLUTION)

    output = model(x)

    # Test class loss
    class_loss = output['class_preds'].sum()
    class_loss.backward(retain_graph=True)

    # Check that gradients exist
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients computed for class loss"
    print(f"✅ Gradients flow through label predictor")

    # Zero gradients
    model.zero_grad()

    # Test domain loss
    if 'domain_preds' in output:
        domain_loss = output['domain_preds'].sum()
        domain_loss.backward()

        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "No gradients computed for domain loss"
        print(f"✅ Gradients flow through domain classifier")

        # Check that feature extractor gets gradients from both branches
        feature_has_grads = any(p.grad is not None for p in model.feature_extractor.parameters())
        assert feature_has_grads, "Feature extractor should receive gradients"
        print(f"✅ Feature extractor receives gradients")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*80)
    print("#" + " "*30 + "CONFIGURABLE CNN TESTS" + " "*28 + "#")
    print("#"*80)

    try:
        test_basic_forward_pass()
        test_without_dann()
        test_gradient_reversal_layer()
        test_epoch_based_lambda_adjustment()
        test_different_architectures()
        test_get_features()
        test_parameter_count()
        test_backward_pass()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
