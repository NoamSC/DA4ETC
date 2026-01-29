"""
Integration tests between cesnet_dataloader.py and multimodal_cesnet.py.

This test verifies that data produced by ParquetCESNETDataset can be consumed
directly by the Multimodal_CESNET model and trainer.py without any transformation.

Data flow:
- cesnet_dataloader produces: ((ppi, flowstats), labels) - trainer compatible format
  - ppi: (B, C, T) = (B, 3, 30)
  - flowstats: (B, 44)
- Multimodal_CESNET expects: (ppi, flowstats) tuple, returns dict with class_preds, features
- trainer.py expects: (inputs, labels) where inputs can be a tuple
"""
import json
import sys
import torch

sys.path.append('/home/anatbr/students/noamshakedc/da4etc')

from data_utils.cesnet_dataloader import ParquetCESNETDataset, create_parquet_loader
from models.multimodal_cesnet import Multimodal_CESNET

# Paths to CESNET dataset
DATASET_ROOT = "/home/anatbr/dataset/CESNET-TLS-Year22_v2"
PARQUET_PATH = "sampled_data.parquet"
# f"{DATASET_ROOT}/WEEK-2022-30/2022-07-28/flows-20220728.parquet"
LABEL_MAPPING_PATH = f"{DATASET_ROOT}/label_mapping.json"


def load_label_mapping():
    """Load label mapping from CESNET dataset's label_mapping.json."""
    with open(LABEL_MAPPING_PATH, 'r') as f:
        label_mapping = json.load(f)
    # label_mapping.json already maps app_name -> int index
    return label_mapping


def test_dataloader_output_shapes():
    """Test that the dataloader produces expected output shapes."""
    print("Test 1: Dataloader output shapes")

    label_mapping = load_label_mapping()
    print(f"  Loaded {len(label_mapping)} classes from label_mapping.json")

    dataset = ParquetCESNETDataset(
        parquet_files=[PARQUET_PATH],
        label_mapping=label_mapping,
        max_packets=30,
        data_sample_frac=0.001,  # Small sample for testing
    )

    (ppi, flowstats), label = dataset[0]

    assert ppi.shape == (3, 30), f"Expected ppi shape (3, 30), got {ppi.shape}"
    assert flowstats.shape == (44,), f"Expected flowstats shape (44,), got {flowstats.shape}"
    assert label.shape == (), f"Expected scalar label, got shape {label.shape}"

    print(f"  ✓ ppi shape: {ppi.shape} (C, T) format")
    print(f"  ✓ flowstats shape: {flowstats.shape}")
    print(f"  ✓ label shape: {label.shape}")
    print()


def test_dataloader_batched_shapes():
    """Test batched output shapes from the dataloader."""
    print("Test 2: Batched dataloader output shapes")

    label_mapping = load_label_mapping()
    batch_size = 16

    loader = create_parquet_loader(
        parquet_files=[PARQUET_PATH],
        label_mapping=label_mapping,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=0,
        # data_sample_frac=0.001,
    )

    (ppi_batch, flowstats_batch), labels_batch = next(iter(loader))

    assert ppi_batch.shape == (batch_size, 3, 30), f"Expected {(batch_size, 3, 30)}, got {ppi_batch.shape}"
    assert flowstats_batch.shape == (batch_size, 44), f"Expected (B, 44), got {flowstats_batch.shape}"
    assert labels_batch.shape == (batch_size,), f"Expected (B,), got {labels_batch.shape}"

    print(f"  ✓ Batched ppi shape: {ppi_batch.shape} (B, C, T) format")
    print(f"  ✓ Batched flowstats shape: {flowstats_batch.shape}")
    print(f"  ✓ Batched labels shape: {labels_batch.shape}")
    print()
    return loader


def test_model_input_requirements():
    """Test Multimodal_CESNET input requirements."""
    print("Test 3: Model input requirements")

    num_classes = 5
    model = Multimodal_CESNET(
        num_classes=num_classes,
        flowstats_input_size=44,
        ppi_input_channels=3,
    )
    model.eval()

    batch_size = 4
    # Model expects (B, C, T) format for Conv1d
    ppi = torch.randn(batch_size, 3, 30)
    flowstats = torch.randn(batch_size, 44)

    output = model(ppi, flowstats)

    assert isinstance(output, dict), f"Expected dict output, got {type(output)}"
    assert output['class_preds'].shape == (batch_size, num_classes), f"Expected ({batch_size}, {num_classes}), got {output['class_preds'].shape}"
    assert 'features' in output, "Expected 'features' key in output"
    assert 'domain_preds' in output, "Expected 'domain_preds' key in output"

    print(f"  ✓ Model expects ppi shape: (B, C, T) = (B, 3, 30)")
    print(f"  ✓ Model expects flowstats shape: (B, 44)")
    print(f"  ✓ Model output: dict with class_preds {output['class_preds'].shape}, features {output['features'].shape}")
    print()


def test_direct_model_compatibility():
    """Demonstrate that dataloader output is directly compatible with model (no transformation needed)."""
    print("Test 4: Direct model compatibility (trainer format)")

    label_mapping = load_label_mapping()
    num_classes = len(label_mapping)
    batch_size = 8

    loader = create_parquet_loader(
        parquet_files=[PARQUET_PATH],
        label_mapping=label_mapping,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        data_sample_frac=0.001,
    )

    model = Multimodal_CESNET(
        num_classes=num_classes,
        flowstats_input_size=44,
        ppi_input_channels=3,
    )
    model.eval()

    (ppi_batch, flowstats_batch), labels_batch = next(iter(loader))

    # Dataloader produces ((ppi, flowstats), labels) format for trainer compatibility
    assert ppi_batch.shape == (batch_size, 3, 30), "Unexpected ppi shape"
    assert flowstats_batch.shape == (batch_size, 44), "Unexpected flowstats shape"

    # Forward pass works with tuple input (trainer format)
    with torch.no_grad():
        output = model((ppi_batch, flowstats_batch))

    assert output['class_preds'].shape == (batch_size, num_classes), f"Unexpected output shape: {output['class_preds'].shape}"

    print(f"  ✓ Dataloader output: ((ppi {ppi_batch.shape}, flowstats {flowstats_batch.shape}), labels)")
    print(f"  ✓ Model output: class_preds {output['class_preds'].shape}")
    print("  ✓ Direct integration works with trainer format!")
    print()


def test_full_integration_forward_pass():
    """Full integration test: dataloader -> transform -> model -> loss."""
    print("Test 5: Full integration (dataloader -> model -> loss)")

    label_mapping = load_label_mapping()
    num_classes = len(label_mapping)
    batch_size = 16

    loader = create_parquet_loader(
        parquet_files=[PARQUET_PATH],
        label_mapping=label_mapping,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        data_sample_frac=0.001,
    )

    model = Multimodal_CESNET(
        num_classes=num_classes,
        flowstats_input_size=44,
        ppi_input_channels=3,
    )
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for inputs, labels_batch in loader:
        # Trainer format: inputs = (ppi, flowstats), labels_batch = labels
        # Forward pass
        outputs = model(inputs)

        # Compute loss using class_preds from output dict
        loss = criterion(outputs['class_preds'], labels_batch)
        total_loss += loss.item()
        num_batches += 1

        if num_batches >= 5:  # Limit for faster testing
            break

    avg_loss = total_loss / num_batches

    print(f"  ✓ Processed {num_batches} batches")
    print(f"  ✓ Average loss: {avg_loss:.4f}")
    print("  ✓ Full forward pass integration successful")
    print()


def test_full_integration_backward_pass():
    """Test that gradients flow correctly through the integrated pipeline."""
    print("Test 6: Full integration with backward pass (gradient flow)")

    label_mapping = load_label_mapping()
    num_classes = len(label_mapping)
    batch_size = 8

    loader = create_parquet_loader(
        parquet_files=[PARQUET_PATH],
        label_mapping=label_mapping,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        data_sample_frac=0.001,
    )

    model = Multimodal_CESNET(
        num_classes=num_classes,
        flowstats_input_size=44,
        ppi_input_channels=3,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    inputs, labels_batch = next(iter(loader))

    # Forward pass - trainer format with tuple input
    outputs = model(inputs)
    loss = criterion(outputs['class_preds'], labels_batch)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist for model parameters
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    param_count = sum(1 for p in model.parameters() if p.requires_grad)

    assert grad_count == param_count, f"Not all parameters have gradients: {grad_count}/{param_count}"

    # Optimizer step
    optimizer.step()

    print(f"  ✓ Backward pass completed")
    print(f"  ✓ All {param_count} trainable parameters received gradients")
    print(f"  ✓ Optimizer step successful")
    print()


if __name__ == '__main__':
    print("=" * 70)
    print("Integration Tests: cesnet_dataloader.py <-> multimodal_cesnet.py")
    print("=" * 70)
    print()

    test_dataloader_output_shapes()
    test_dataloader_batched_shapes()
    test_model_input_requirements()
    test_direct_model_compatibility()
    test_full_integration_forward_pass()
    test_full_integration_backward_pass()

    print("=" * 70)
    print("All integration tests passed!")
    print("=" * 70)
    print()
    print("cesnet_dataloader outputs ((ppi, flowstats), labels) format,")
    print("which is directly compatible with trainer.py and Multimodal_CESNET!")
