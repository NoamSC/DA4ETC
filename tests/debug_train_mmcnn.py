"""
Debug training script for ConfigurableMMCNN with synthetic data.
Verifies the model can overfit on small synthetic data with DANN.

Usage:
    python debug_train_mmcnn.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from models.configurable_mmcnn import ConfigurableMMCNN


def create_synthetic_data(num_samples=512, num_classes=10, flowstats_dim=44, seed=42):
    """Create synthetic pstats/flowstats data with class-correlated patterns."""
    torch.manual_seed(seed)

    # Create class-correlated patterns so the model can learn something
    pstats = torch.randn(num_samples, 30, 3)
    flowstats = torch.randn(num_samples, flowstats_dim)
    labels = torch.randint(0, num_classes, (num_samples,))

    # Inject class signal into the data
    for i in range(num_samples):
        c = labels[i].item()
        pstats[i, :5, 0] += c * 0.5  # class signal in first 5 packet sizes
        flowstats[i, :4] += c * 0.3  # class signal in first 4 flow features

    return pstats, flowstats, labels


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model config
    num_classes = 10
    params = {
        'num_classes': num_classes,
        'flowstats_dim': 44,
        'pstats_format': 'btc',
        'dropout_cnn': 0.1,
        'dropout_flow': 0.1,
        'dropout_shared': 0.2,
        # DANN
        'lambda_rgl': 0.1,
        'dann_fc_out_features': 128,
        'lambda_grl_gamma': 10.0,
    }

    model = ConfigurableMMCNN(params).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Synthetic data
    pstats_src, flowstats_src, labels_src = create_synthetic_data(
        num_samples=512, num_classes=num_classes
    )
    pstats_tgt, flowstats_tgt, _ = create_synthetic_data(
        num_samples=256, num_classes=num_classes, seed=99
    )

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    num_epochs = 20

    # Output dir
    exp_dir = Path("exps/debug/mmcnn_synthetic")
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {num_epochs} epochs (batch_size={batch_size})...")
    print(f"Source samples: {len(labels_src)}, Target samples: {len(pstats_tgt)}")
    print(f"Experiment dir: {exp_dir}\n")

    for epoch in range(num_epochs):
        model.train()
        model.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_dann_loss = 0.0
        correct = 0
        total = 0

        # Mini-batch training
        indices = torch.randperm(len(labels_src))
        for start in range(0, len(labels_src), batch_size):
            idx = indices[start:start + batch_size]
            ps = pstats_src[idx].to(device)
            fs = flowstats_src[idx].to(device)
            lbl = labels_src[idx].to(device)

            optimizer.zero_grad()
            out = model((ps, fs))

            # Classification loss
            cls_loss = criterion(out['class_preds'], lbl)

            # DANN loss
            dann_loss = torch.tensor(0.0, device=device)
            if 'domain_preds' in out:
                # Source domain labels = 0
                src_domain_labels = torch.zeros(len(idx), dtype=torch.long, device=device)
                dann_loss = criterion(out['domain_preds'], src_domain_labels)

                # Target forward
                tgt_idx = torch.randint(0, len(pstats_tgt), (len(idx),))
                tgt_out = model((pstats_tgt[tgt_idx].to(device), flowstats_tgt[tgt_idx].to(device)))
                tgt_domain_labels = torch.ones(len(idx), dtype=torch.long, device=device)
                dann_loss += criterion(tgt_out['domain_preds'], tgt_domain_labels)

            loss = cls_loss + params['lambda_rgl'] * dann_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_dann_loss += dann_loss.item()
            correct += (out['class_preds'].argmax(1) == lbl).sum().item()
            total += len(lbl)

        n_batches = (len(labels_src) + batch_size - 1) // batch_size
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"loss={epoch_loss/n_batches:.4f} "
              f"cls={epoch_cls_loss/n_batches:.4f} "
              f"dann={epoch_dann_loss/n_batches:.4f} | "
              f"acc={acc:.1f}%")

    # Save checkpoint
    ckpt_path = exp_dir / "model_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")

    # Final eval
    model.eval()
    with torch.no_grad():
        out = model((pstats_src.to(device), flowstats_src.to(device)))
        final_acc = (out['class_preds'].argmax(1) == labels_src.to(device)).float().mean().item()
    print(f"Final train accuracy: {100*final_acc:.1f}%")

    if final_acc > 0.9:
        print("SUCCESS: Model can overfit synthetic data.")
    else:
        print("WARNING: Model did not fully overfit. Check architecture/training.")


if __name__ == "__main__":
    main()
