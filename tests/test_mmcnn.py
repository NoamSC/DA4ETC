"""
Short tests for ConfigurableMMCNN to verify it behaves similarly to ConfigurableCNN.
"""
import torch
import sys
sys.path.append('/home/anatbr/students/noamshakedc/da4etc')

from models.configurable_mmcnn import ConfigurableMMCNN


def test_basic_forward():
    """Test basic forward pass without DANN."""
    print("Test 1: Basic forward pass (no DANN)")

    params = {
        'num_classes': 102,
        'pstats_format': 'btc',
        'flowstats_dim': 44,
        'lambda_rgl': 0,
    }

    model = ConfigurableMMCNN(params)
    model.eval()

    # Create dummy inputs
    batch_size = 4
    pstats = torch.randn(batch_size, 30, 3)  # (B, T, C) format
    flowstats = torch.randn(batch_size, 44)

    # Forward pass
    output = model((pstats, flowstats))

    # Check output structure
    assert 'class_preds' in output, "Missing class_preds in output"
    assert 'features' in output, "Missing features in output"
    assert 'domain_preds' not in output, "domain_preds should not be present without DANN"

    # Check shapes
    assert output['class_preds'].shape == (batch_size, 102), f"Wrong class_preds shape: {output['class_preds'].shape}"
    assert output['features'].shape == (batch_size, 600), f"Wrong features shape: {output['features'].shape}"

    print("✓ Passed: Output dict has correct keys and shapes")
    print(f"  - class_preds: {output['class_preds'].shape}")
    print(f"  - features: {output['features'].shape}")
    print()


def test_bct_format():
    """Test with BCT format (B, C, T)."""
    print("Test 2: BCT format input")

    params = {
        'num_classes': 10,
        'pstats_format': 'bct',  # (B, C, T) format
        'flowstats_dim': 44,
        'lambda_rgl': 0,
    }

    model = ConfigurableMMCNN(params)
    model.eval()

    batch_size = 2
    pstats = torch.randn(batch_size, 3, 30)  # (B, C, T) format
    flowstats = torch.randn(batch_size, 44)

    output = model((pstats, flowstats))

    assert output['class_preds'].shape == (batch_size, 10), f"Wrong shape: {output['class_preds'].shape}"
    assert output['features'].shape == (batch_size, 600), f"Wrong features shape: {output['features'].shape}"

    print("✓ Passed: BCT format works correctly")
    print()


def test_dann_mode():
    """Test DANN mode with domain classifier."""
    print("Test 3: DANN mode (with domain classifier)")

    params = {
        'num_classes': 50,
        'pstats_format': 'btc',
        'flowstats_dim': 44,
        'lambda_rgl': 1.0,
        'dann_fc_out_features': 128,
        'lambda_grl_gamma': 10.0,
    }

    model = ConfigurableMMCNN(params)
    model.eval()

    batch_size = 3
    pstats = torch.randn(batch_size, 30, 3)
    flowstats = torch.randn(batch_size, 44)

    output = model((pstats, flowstats))

    # Check output structure
    assert 'class_preds' in output, "Missing class_preds"
    assert 'features' in output, "Missing features"
    assert 'domain_preds' in output, "Missing domain_preds in DANN mode"

    # Check shapes
    assert output['class_preds'].shape == (batch_size, 50), f"Wrong class_preds shape: {output['class_preds'].shape}"
    assert output['features'].shape == (batch_size, 600), f"Wrong features shape: {output['features'].shape}"
    assert output['domain_preds'].shape == (batch_size, 2), f"Wrong domain_preds shape: {output['domain_preds'].shape}"

    print("✓ Passed: DANN mode works, domain_preds present")
    print(f"  - class_preds: {output['class_preds'].shape}")
    print(f"  - features: {output['features'].shape}")
    print(f"  - domain_preds: {output['domain_preds'].shape}")
    print()


def test_set_epoch():
    """Test set_epoch method for DANN lambda scheduling."""
    print("Test 4: set_epoch and GRL lambda scheduling")

    params = {
        'num_classes': 20,
        'lambda_rgl': 1.0,
        'dann_fc_out_features': 64,
        'lambda_grl_gamma': 0.1,  # Use realistic gamma value
    }

    model = ConfigurableMMCNN(params)

    # Test epoch updates
    initial_lambda = model.domain_classifier[0].lambda_
    print(f"  Initial lambda: {initial_lambda}")

    model.set_epoch(5)
    lambda_epoch_5 = model.domain_classifier[0].lambda_

    model.set_epoch(10)
    lambda_epoch_10 = model.domain_classifier[0].lambda_

    # Lambda should increase with epochs (factor goes from ~0.24 to ~0.46)
    assert lambda_epoch_10 > lambda_epoch_5, f"Lambda should increase with epoch (got {lambda_epoch_5:.4f} -> {lambda_epoch_10:.4f})"

    print(f"✓ Passed: Lambda scheduling works (increases with epochs)")
    print()


def test_custom_dimensions():
    """Test with custom flowstats dimension and dropout rates."""
    print("Test 5: Custom dimensions and dropout rates")

    params = {
        'num_classes': 15,
        'flowstats_dim': 60,  # Non-default
        'dropout_cnn': 0.2,
        'dropout_flow': 0.3,
        'dropout_shared': 0.4,
        'gem_p_init': 4.0,
        'lambda_rgl': 0,
    }

    model = ConfigurableMMCNN(params)
    model.eval()

    batch_size = 2
    pstats = torch.randn(batch_size, 30, 3)
    flowstats = torch.randn(batch_size, 60)  # Custom dimension

    output = model((pstats, flowstats))

    assert output['class_preds'].shape == (batch_size, 15), f"Wrong shape: {output['class_preds'].shape}"

    print("✓ Passed: Custom dimensions work correctly")
    print()


def test_backward_pass():
    """Test backward pass to ensure gradients flow properly."""
    print("Test 6: Backward pass (gradient flow)")

    params = {
        'num_classes': 10,
        'lambda_rgl': 0,
    }

    model = ConfigurableMMCNN(params)
    model.train()

    batch_size = 2
    pstats = torch.randn(batch_size, 30, 3, requires_grad=True)
    flowstats = torch.randn(batch_size, 44, requires_grad=True)

    output = model((pstats, flowstats))
    loss = output['class_preds'].sum()
    loss.backward()

    # Check gradients exist
    assert pstats.grad is not None, "No gradient for pstats"
    assert flowstats.grad is not None, "No gradient for flowstats"

    # Check model parameters have gradients
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "No gradients in model parameters"

    print("✓ Passed: Gradients flow correctly through the model")
    print()


def test_get_features():
    """Test get_features method."""
    print("Test 7: get_features method")

    params = {
        'num_classes': 5,
        'lambda_rgl': 0,
    }

    model = ConfigurableMMCNN(params)
    model.eval()

    batch_size = 2
    pstats = torch.randn(batch_size, 30, 3)
    flowstats = torch.randn(batch_size, 44)

    features = model.get_features(pstats, flowstats)

    assert features.shape == (batch_size, 600), f"Wrong features shape: {features.shape}"

    # Features from get_features should match those in forward()
    output = model((pstats, flowstats))
    assert torch.allclose(features, output['features']), "get_features output doesn't match forward()"

    print("✓ Passed: get_features method works correctly")
    print()


if __name__ == '__main__':
    print("="*60)
    print("Testing ConfigurableMMCNN")
    print("="*60)
    print()

    test_basic_forward()
    test_bct_format()
    test_dann_mode()
    test_set_epoch()
    test_custom_dimensions()
    test_backward_pass()
    test_get_features()

    print("="*60)
    print("All tests passed! ✓")
    print("="*60)
