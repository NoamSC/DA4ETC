"""
Debug script to check architecture dimensions and gradient flow.
"""

import torch
from models.configurable_cnn import ConfigurableCNN
from config import Config


def check_architecture_dimensions():
    """Check spatial dimensions through the network."""
    config = Config()
    config.MODEL_PARAMS['num_classes'] = 180

    model = ConfigurableCNN(config.MODEL_PARAMS)
    model.eval()

    # Create dummy input
    batch_size = 4
    input_shape = config.MODEL_PARAMS['input_shape']
    x = torch.randn(batch_size, input_shape, input_shape)

    print("="*80)
    print("ARCHITECTURE DIMENSION FLOW")
    print("="*80)

    print(f"\nInput shape: {x.shape}")

    # Track through feature extractor
    print("\n--- FEATURE EXTRACTOR ---")
    x_transposed = x.transpose(1, 2)
    print(f"After transpose: {x_transposed.shape}")

    temp = x_transposed
    for i, layer in enumerate(model.feature_extractor):
        temp = layer(temp)
        print(f"After layer {i} ({layer.__class__.__name__}): {temp.shape}")

    features = temp
    print(f"\nFeature extractor output: {features.shape}")

    # Track through domain classifier
    print("\n--- DOMAIN CLASSIFIER ---")
    print(f"Input (from features): {features.shape}")

    # Apply GRL (identity in forward)
    domain_features = model.grl(features)
    print(f"After GRL: {domain_features.shape}")

    for i, layer in enumerate(model.domain_classifier_convs):
        domain_features = layer(domain_features)
        print(f"After conv layer {i} ({layer.__class__.__name__}): {domain_features.shape}")

    print(f"\nBefore flatten: {domain_features.shape}")
    domain_features_flat = domain_features.view(domain_features.size(0), -1)
    print(f"After flatten: {domain_features_flat.shape}")

    if domain_features_flat.shape[1] == 0:
        print("\n⚠️  CRITICAL ERROR: Flattened features have size 0!")
        print("The spatial dimensions were reduced to zero by the conv/pooling layers.")
        print("This means the domain classifier cannot process any information.")
        return

    for i, fc_layer in enumerate(model.domain_classifier_fcs):
        domain_features_flat = fc_layer(domain_features_flat)
        print(f"After FC layer {i}: {domain_features_flat.shape}")

    domain_preds = model.domain_output(domain_features_flat)
    print(f"Final domain predictions: {domain_preds.shape}")

    # Check if predictions are meaningful
    print(f"\nDomain predictions:")
    print(f"  Mean: {domain_preds.mean().item():.4f}")
    print(f"  Std: {domain_preds.std().item():.4f}")
    print(f"  Min: {domain_preds.min().item():.4f}")
    print(f"  Max: {domain_preds.max().item():.4f}")


def test_backward_pass():
    """Test if gradients can flow backward."""
    config = Config()
    config.MODEL_PARAMS['num_classes'] = 180

    model = ConfigurableCNN(config.MODEL_PARAMS)
    model.train()

    # Create dummy input and target
    batch_size = 4
    input_shape = config.MODEL_PARAMS['input_shape']
    x = torch.randn(batch_size, input_shape, input_shape)
    domain_labels = torch.zeros(batch_size).long()

    print("\n" + "="*80)
    print("GRADIENT FLOW TEST")
    print("="*80)

    # Temporarily set GRL lambda to 1.0
    original_lambda = model.grl.lambda_
    model.grl.lambda_ = 1.0
    print(f"\nGRL lambda set to: {model.grl.lambda_}")

    # Forward pass
    model.zero_grad()
    outputs = model(x)

    if 'domain_preds' not in outputs:
        print("\n⚠️  ERROR: Model did not produce domain predictions!")
        return

    domain_preds = outputs['domain_preds']
    print(f"\nDomain predictions shape: {domain_preds.shape}")
    print(f"Domain predictions stats:")
    print(f"  Mean: {domain_preds.mean().item():.4f}")
    print(f"  Std: {domain_preds.std().item():.4f}")

    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(domain_preds, domain_labels)
    print(f"\nDomain loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients
    print("\nGradient statistics in feature extractor:")
    has_gradients = False
    for name, param in model.named_parameters():
        if 'feature_extractor' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            print(f"  {name:50s}: norm={grad_norm:12.8f}, mean={grad_mean:12.8f}, std={grad_std:12.8f}")
            if grad_norm > 1e-10:
                has_gradients = True

    if has_gradients:
        print("\n✓ Gradients are flowing to feature extractor")
    else:
        print("\n✗ No gradients in feature extractor (all are zero or None)")

    # Restore lambda
    model.grl.lambda_ = original_lambda


def check_domain_classifier_capacity():
    """Check if domain classifier has enough capacity."""
    config = Config()
    config.MODEL_PARAMS['num_classes'] = 180

    model = ConfigurableCNN(config.MODEL_PARAMS)

    print("\n" + "="*80)
    print("MODEL PARAMETER COUNT")
    print("="*80)

    total_params = 0
    feature_params = 0
    label_params = 0
    domain_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if 'feature_extractor' in name:
            feature_params += num_params
        elif 'label_predictor' in name or 'label_output' in name:
            label_params += num_params
        elif 'domain_classifier' in name or 'domain_output' in name:
            domain_params += num_params

    print(f"\nTotal parameters: {total_params:,}")
    print(f"  Feature extractor: {feature_params:,} ({100*feature_params/total_params:.1f}%)")
    print(f"  Label predictor: {label_params:,} ({100*label_params/total_params:.1f}%)")
    print(f"  Domain classifier: {domain_params:,} ({100*domain_params/total_params:.1f}%)")

    if domain_params < 1000:
        print(f"\n⚠️  WARNING: Domain classifier has very few parameters ({domain_params})")
        print("This might indicate architectural issues.")


if __name__ == "__main__":
    check_architecture_dimensions()
    test_backward_pass()
    check_domain_classifier_capacity()
