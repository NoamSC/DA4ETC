from itertools import cycle
import time

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

from training.visualization import (plot_confusion_matrix, plot_metrics,
                                     compute_topk_accuracy, compute_per_class_precision,
                                     create_tsne_visualization)

def compute_mmd_loss(source_features, target_features, kernel='rbf', bandwidths=[0.1, 1, 10]):
    def rbf_kernel(x, y, bandwidth):
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        dist = x_norm + y_norm.T - 2 * torch.mm(x, y.T)
        return torch.exp(-dist / (2 * bandwidth ** 2 + 1e-6))
    
    if kernel != 'rbf':
        raise ValueError("Unsupported kernel type")

    # normalize features for mmd
    normalized_target_features = (target_features - target_features.mean(dim=0)) / (target_features.std(dim=0, unbiased=False) + 1e-6)
    normalized_source_features = (source_features - source_features.mean(dim=0)) / (source_features.std(dim=0, unbiased=False) + 1e-6)

    loss = 0
    for bw in bandwidths:
        xx = rbf_kernel(normalized_source_features, normalized_source_features, bw).mean()
        yy = rbf_kernel(normalized_target_features, normalized_target_features, bw).mean()
        xy = rbf_kernel(normalized_source_features, normalized_target_features, bw).mean()
        loss += xx + yy - 2 * xy
    return loss / len(bandwidths)


def train_one_epoch(model, train_loader, criterion, optimizer, device, lambda_mmd=0.0, test_loader=None, mmd_bandwidths=[0.1, 1, 10], lambda_dann=0.0):
    """Train model for one epoch. Supports optional MMD and DANN losses."""
    model.train()
    metrics = {
        'train_loss': 0, 'train_correct': 0, 'train_total': 0,
        'classification_loss': 0, 'mmd_loss': 0, 'dann_loss': 0,
        'domain_correct': 0, 'domain_total': 0
    }

    test_iter = cycle(test_loader) if test_loader is not None else None

    for train_inputs, train_labels in tqdm(train_loader, desc="Training", leave=False, ncols=100, mininterval=10.0):
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device).long()
        optimizer.zero_grad()

        # Source domain forward pass
        source_outputs = model(train_inputs)

        # Target domain forward pass
        target_outputs = None
        if test_iter is not None:
            test_inputs, _ = next(test_iter)
            target_outputs = model(test_inputs.to(device))

        # Loss computation
        classification_loss = criterion(source_outputs['class_preds'], train_labels)

        mmd_loss = 0
        if target_outputs is not None and lambda_mmd > 0:
            mmd_loss = compute_mmd_loss(source_outputs['features'], target_outputs['features'], bandwidths=mmd_bandwidths)

        dann_loss = 0
        if target_outputs is not None and lambda_dann > 0:
            source_domain_labels = torch.zeros(len(source_outputs['domain_preds'])).long().to(device)
            target_domain_labels = torch.ones(len(target_outputs['domain_preds'])).long().to(device)

            dann_loss = criterion(source_outputs['domain_preds'], source_domain_labels)
            dann_loss += criterion(target_outputs['domain_preds'], target_domain_labels)

            # Track domain classifier accuracy
            with torch.no_grad():
                _, source_pred = source_outputs['domain_preds'].max(1)
                _, target_pred = target_outputs['domain_preds'].max(1)
                metrics['domain_correct'] += source_pred.eq(source_domain_labels).sum().item()
                metrics['domain_correct'] += target_pred.eq(target_domain_labels).sum().item()
                metrics['domain_total'] += len(source_domain_labels) + len(target_domain_labels)

        total_loss = classification_loss + lambda_mmd * mmd_loss + lambda_dann * dann_loss
        total_loss.backward()
        optimizer.step()

        # Track metrics
        metrics['train_loss'] += total_loss.item() * train_inputs.size(0)
        metrics['classification_loss'] += classification_loss.item()
        metrics['mmd_loss'] += mmd_loss.item() if isinstance(mmd_loss, torch.Tensor) else 0
        metrics['dann_loss'] += dann_loss.item() if isinstance(dann_loss, torch.Tensor) else 0

        _, predicted = source_outputs['class_preds'].max(1)
        metrics['train_correct'] += predicted.eq(train_labels).sum().item()
        metrics['train_total'] += train_labels.size(0)

    # Return dict
    return_dict = {
        'train_loss': metrics['train_loss'] / metrics['train_total'],
        'train_acc': 100.0 * metrics['train_correct'] / metrics['train_total'],
        'classification_loss': metrics['classification_loss'] / len(train_loader),
        'mmd_loss': metrics['mmd_loss'] / len(train_loader) * lambda_mmd if lambda_mmd > 0 else 0,
        'dann_loss': metrics['dann_loss'] / len(train_loader) * lambda_dann if lambda_dann > 0 else 0,
        'domain_acc': 100.0 * metrics['domain_correct'] / metrics['domain_total'] if metrics['domain_total'] > 0 else 0
    }

    return return_dict


def validate(model, val_loader, criterion, device, return_features=False):
    """Evaluate the model on the validation set.

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
        return_features: If True, also return logits and features for advanced metrics

    Returns:
        tuple: (val_loss, val_acc, all_labels, all_predictions, [all_logits, all_features])
    """
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_labels, all_predictions = [], []
    all_logits, all_features = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False, ncols=100, mininterval=5.0):
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            class_preds = outputs['class_preds']
            loss = criterion(class_preds, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = class_preds.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if return_features:
                all_logits.append(class_preds.cpu())
                all_features.append(outputs['features'].cpu())

    if return_features:
        all_logits = torch.cat(all_logits, dim=0)
        all_features = torch.cat(all_features, dim=0)
        return (
            val_loss / val_total,
            100.0 * val_correct / val_total,
            all_labels,
            all_predictions,
            all_logits,
            all_features.numpy()
        )
    else:
        return (
            val_loss / val_total,
            100.0 * val_correct / val_total,
            all_labels,
            all_predictions
        )


def batch_norm_adaptation(model, target_loader, device):
    """Adapt batch norm layers to the target distribution."""
    model.train()  # Ensure running stats get updated
    with torch.no_grad():
        for inputs, _ in tqdm(target_loader, desc="Batch Norm Adaptation", leave=False, ncols=100, mininterval=5.0):
            inputs = inputs.to(device)
            _ = model(inputs)  # Forward pass to update batch norm statistics


def train_model(model, train_loader, criterion, optimizer, num_epochs, device,
                weights_save_dir, plots_save_dir, label_mapping, lambda_mmd=0.0,
                test_loader=None, adapt_batch_norm=False, mmd_bandwidths=[1], lambda_dann=0.0,
                resume_checkpoint_path=None, resume_from_epoch=0,
                train_per_epoch_data_frac=1.0, seed=42, enable_profiler=False):
    """
    General training function with support for MMD, DANN, and batch norm adaptation.
    Logs to TensorBoard and saves the best model based on validation accuracy.
    Supports resuming from checkpoint.

    Args:
        enable_profiler: If True, enable PyTorch profiler for the first 2 epochs
    """
    # Initialize TensorBoard writer
    tensorboard_dir = plots_save_dir.parent / 'tensorboard'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # Track best model
    best_val_acc = 0.0
    best_epoch = 0

    # other losses
    mmd_loss_name = 'MMD Loss'
    dann_loss_name = 'DANN Loss'
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_regular_losses, train_other_losses = [], {mmd_loss_name: [], dann_loss_name: []}

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint_path is not None and resume_from_epoch > 0:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")

        # Load model state
        checkpoint = torch.load(resume_checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Loaded model from epoch {resume_from_epoch}")

        # Load training history if available
        history_path = plots_save_dir / "training_history.pth"
        if history_path.exists():
            history = torch.load(history_path, weights_only=False)
            train_losses = history.get('train_losses', [])
            val_losses = history.get('val_losses', [])
            train_accuracies = history.get('train_accuracies', [])
            val_accuracies = history.get('val_accuracies', [])
            train_regular_losses = history.get('train_regular_losses', [])
            train_other_losses = history.get('train_other_losses', {mmd_loss_name: [], dann_loss_name: []})
            print(f"Loaded training history up to epoch {len(train_losses)}")

            # Find best epoch so far
            if val_accuracies:
                best_val_acc = max(val_accuracies)
                best_epoch = val_accuracies.index(best_val_acc) + 1
                print(f"Best validation accuracy so far: {best_val_acc:.2f}% at epoch {best_epoch}")

        start_epoch = resume_from_epoch
        print(f"Resuming training from epoch {start_epoch + 1}")
        print(f"{'='*60}\n")

    # Keep reference to original train_loader
    original_train_loader = train_loader

    # Setup profiler if enabled
    profiler_context = None
    if enable_profiler and start_epoch < 2:
        print("\n" + "="*60)
        print("PROFILER ENABLED - Will profile first 2 epochs")
        print("="*60 + "\n")
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler_context.__enter__()

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        # Create per-epoch sampler if needed
        if train_per_epoch_data_frac < 1.0:
            import random
            from torch.utils.data import SubsetRandomSampler, DataLoader

            # Generate epoch-dependent subset indices
            n_total = len(original_train_loader.dataset)
            n_samples = int(n_total * train_per_epoch_data_frac)
            g = random.Random(seed + epoch)
            indices = list(range(n_total))
            g.shuffle(indices)
            subset_indices = indices[:n_samples]

            # Create new sampler and DataLoader for this epoch
            sampler = SubsetRandomSampler(subset_indices)
            train_loader = DataLoader(
                original_train_loader.dataset,
                batch_size=original_train_loader.batch_size,
                sampler=sampler,
                num_workers=original_train_loader.num_workers
            )

        model.set_epoch(epoch / num_epochs)

        # Training phase with timing
        train_start_time = time.time()
        with record_function("train_epoch"):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, lambda_mmd, test_loader, mmd_bandwidths, lambda_dann
            )
        train_time = time.time() - train_start_time

        train_losses.append(train_metrics['train_loss'])
        train_accuracies.append(train_metrics['train_acc'])
        train_regular_losses.append(train_metrics['classification_loss'])
        train_other_losses[mmd_loss_name].append(train_metrics['mmd_loss'])
        train_other_losses[dann_loss_name].append(train_metrics['dann_loss'])

        # Validation with enhanced metrics
        val_start_time = time.time()
        with record_function("validation"):
            val_result = validate(model, test_loader, criterion, device, return_features=True)
        val_time = time.time() - val_start_time
        val_loss, val_acc, all_labels, all_predictions, all_logits, all_features = val_result
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Compute advanced metrics
        metrics_start_time = time.time()
        with record_function("compute_metrics"):
            topk_metrics = compute_topk_accuracy(all_logits, torch.tensor(all_labels), k_values=[3, 5, 10])
            per_class_metrics = compute_per_class_precision(np.array(all_labels), np.array(all_predictions), num_classes=len(label_mapping))
        metrics_time = time.time() - metrics_start_time

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_metrics['train_acc'], epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        writer.add_scalar('Loss/train_regular', train_metrics['classification_loss'], epoch)

        # Log advanced metrics
        writer.add_scalar('Accuracy/top_3', topk_metrics['top_3'], epoch)
        writer.add_scalar('Accuracy/top_5', topk_metrics['top_5'], epoch)
        writer.add_scalar('Accuracy/top_10', topk_metrics['top_10'], epoch)
        writer.add_scalar('Accuracy/mean_per_class_precision', per_class_metrics['mean_per_class_precision'], epoch)

        if lambda_mmd > 0:
            writer.add_scalar('Loss/mmd', train_metrics['mmd_loss'], epoch)
        if lambda_dann > 0:
            writer.add_scalar('Loss/dann', train_metrics['dann_loss'], epoch)
            writer.add_scalar('Accuracy/domain_classifier', train_metrics['domain_acc'], epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_metrics['train_loss']:.4f}, Accuracy={train_metrics['train_acc']:.2f}%")
        print(f"         Val Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
        print(f"         Top-3: {topk_metrics['top_3']:.2f}%, Top-5: {topk_metrics['top_5']:.2f}%, Top-10: {topk_metrics['top_10']:.2f}%")
        print(f"         Mean Per-Class Precision: {per_class_metrics['mean_per_class_precision']:.2f}%")
        for loss_name, loss_values in train_other_losses.items():
            print(f"         {loss_name}={loss_values[-1]:.4f}, Regular Loss={train_metrics['classification_loss']:.4f}")
        if lambda_dann > 0:
            print(f"         Domain Classifier Accuracy={train_metrics['domain_acc']:.2f}%")

        # Save epoch checkpoint
        save_start_time = time.time()
        with record_function("save_checkpoint"):
            model_save_path = weights_save_dir / f"model_weights_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_save_path)
        save_time = time.time() - save_start_time

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = weights_save_dir / "best_model.pth"
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"         *** New best model saved! Val Acc: {best_val_acc:.2f}% ***")

        # Log confusion matrix to TensorBoard (normalized for better visualization)
        viz_start_time = time.time()
        with record_function("visualizations"):
            class_names = sorted(label_mapping, key=label_mapping.get)
            cm_fig = plot_confusion_matrix(all_labels, all_predictions, class_names=class_names,
                                           epoch=epoch+1, normalize='true')
            if cm_fig is not None:
                writer.add_figure('Confusion_Matrix/validation', cm_fig, epoch)
                plt.close(cm_fig)

            # Create and log t-SNE visualization each validation
            try:
                tsne_fig = create_tsne_visualization(
                    all_features, np.array(all_labels), class_names,
                    epoch=epoch+1, save_dir=plots_save_dir,
                    n_samples=min(5000, len(all_features))
                )
                print(f"         t-SNE visualization saved to plots/tsne_epoch_{epoch+1}.html")
            except Exception as e:
                print(f"         Warning: Failed to create t-SNE visualization: {e}")
        viz_time = time.time() - viz_start_time

        # Log metrics plot to TensorBoard (only at the end to show full history)
        if epoch == num_epochs - 1:
            metrics_fig = plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, additional_train_metrics=train_other_losses)
            if metrics_fig is not None:
                writer.add_figure('Training_Metrics/summary', metrics_fig, epoch)
                plt.close(metrics_fig)

        training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "train_regular_losses": train_regular_losses,
            "train_other_losses": train_other_losses,
        }

        history_save_path = plots_save_dir / "training_history.pth"
        torch.save(training_history, history_save_path)

        # Print timing breakdown
        epoch_total_time = time.time() - epoch_start_time
        print(f"\n         Timing breakdown:")
        print(f"           Training:       {train_time:6.2f}s ({100*train_time/epoch_total_time:5.1f}%)")
        print(f"           Validation:     {val_time:6.2f}s ({100*val_time/epoch_total_time:5.1f}%)")
        print(f"           Metrics:        {metrics_time:6.2f}s ({100*metrics_time/epoch_total_time:5.1f}%)")
        print(f"           Visualizations: {viz_time:6.2f}s ({100*viz_time/epoch_total_time:5.1f}%)")
        print(f"           Checkpointing:  {save_time:6.2f}s ({100*save_time/epoch_total_time:5.1f}%)")
        print(f"           Total:          {epoch_total_time:6.2f}s")

        # Stop profiler after first 2 epochs
        if enable_profiler and epoch == 1:
            print("\n" + "="*60)
            print("STOPPING PROFILER AND GENERATING REPORT")
            print("="*60 + "\n")
            profiler_context.__exit__(None, None, None)

            # Save profiler results
            profiler_dir = plots_save_dir.parent / 'profiler'
            profiler_dir.mkdir(parents=True, exist_ok=True)

            # Print profiler summary
            print("\n" + "="*60)
            print("PROFILER SUMMARY - Top operations by time")
            print("="*60)
            print(profiler_context.key_averages().table(sort_by="cuda_time_total", row_limit=20))

            # Export trace for visualization
            trace_path = profiler_dir / "trace.json"
            profiler_context.export_chrome_trace(str(trace_path))
            print(f"\nProfiler trace saved to: {trace_path}")
            print(f"View in Chrome at: chrome://tracing")
            print("="*60 + "\n")

            profiler_context = None

    # Close TensorBoard writer
    writer.close()

    print("Training completed.")
    print(f"\nBest model saved at epoch {best_epoch} with validation accuracy: {best_val_acc:.2f}%")
    print(f"TensorBoard logs saved to: {tensorboard_dir}")
    print(f"To view logs, run: tensorboard --logdir {tensorboard_dir}")

