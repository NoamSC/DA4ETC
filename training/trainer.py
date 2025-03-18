from itertools import cycle

import torch
from tqdm import tqdm

from training.visualization import plot_confusion_matrix, plot_metrics

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
    train_loss, train_correct, train_total = 0, 0, 0
    classification_loss_total, mmd_loss_total, dann_loss_total = 0, 0, 0

    test_iter = cycle(test_loader) if test_loader is not None else None

    for train_inputs, train_labels in tqdm(train_loader, desc="Training", leave=False):
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device).long()
        
        if test_iter is not None:
            # be careful not to use test labels
            test_inputs, _ = next(test_iter)
            test_inputs = test_inputs.to(device)
            model_test_outputs = model(test_inputs)
            test_features = model_test_outputs['features']
            test_class_preds = model_test_outputs['class_preds']
            target_domain_preds = model_test_outputs.get('domain_preds', None)
        
        optimizer.zero_grad()
        
        # forward pass
        total_loss, mmd_loss, dann_loss = 0, 0, 0
        model_train_outputs = model(train_inputs)
        train_class_preds = model_train_outputs['class_preds']
        train_features = model_train_outputs['features']
        source_domain_preds = model_train_outputs.get('domain_preds', None)
        
        # compute normal classification score
        classification_loss = criterion(train_class_preds, train_labels)

        # compute MMD loss
        if test_loader is not None and lambda_mmd > 0:
            mmd_loss = compute_mmd_loss(train_features, test_features, bandwidths=mmd_bandwidths)
            mmd_loss_total += mmd_loss.item()

        # compute DANN loss
        if test_loader is not None and lambda_dann > 0:
            source_domain_labels = torch.zeros(len(source_domain_preds)).to(device).long()
            target_domain_labels = torch.ones(len(target_domain_preds)).to(device).long()
            
            dann_loss = criterion(source_domain_preds, source_domain_labels)
            dann_loss += criterion(target_domain_preds, target_domain_labels)
            dann_loss_total += dann_loss.item()

        total_loss = classification_loss + mmd_loss * lambda_mmd + dann_loss * lambda_dann
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item() * train_inputs.size(0)
        _, predicted = train_class_preds.max(1)
        train_correct += predicted.eq(train_labels).sum().item()
        train_total += train_labels.size(0)
        classification_loss_total += classification_loss.item()

    return (
        train_loss / train_total, 
        100.0 * train_correct / train_total,
        classification_loss_total / len(train_loader), 
        mmd_loss_total / len(train_loader) * lambda_mmd if lambda_mmd > 0 else 0,
        dann_loss_total / len(train_loader) * lambda_dann if lambda_dann > 0 else 0
    )


def validate(model, val_loader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).long()
            class_preds = model(inputs)['class_preds']
            loss = criterion(class_preds, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = class_preds.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

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
        for inputs, _ in tqdm(target_loader, desc="Batch Norm Adaptation", leave=False):
            inputs = inputs.to(device)
            _ = model(inputs)  # Forward pass to update batch norm statistics


def train_model(model, train_loader, criterion, optimizer, num_epochs, device, 
                weights_save_dir, plots_save_dir, label_mapping, lambda_mmd=0.0, 
                test_loader=None, adapt_batch_norm=False, mmd_bandwidths=[1], lambda_dann=0.0):
    """
    General training function with support for MMD, DANN, and batch norm adaptation.
    """
    # other losses
    mmd_loss_name = 'MMD Loss'
    dann_loss_name = 'DANN Loss'
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_regular_losses, train_other_losses = [], {mmd_loss_name: [], dann_loss_name: []}

    for epoch in range(num_epochs):
        train_loss, train_acc, regular_loss, mmd_loss, dann_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, lambda_mmd, test_loader, mmd_bandwidths, lambda_dann
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_regular_losses.append(regular_loss)
        train_other_losses[mmd_loss_name].append(mmd_loss)
        train_other_losses[dann_loss_name].append(dann_loss)

        val_loss, val_acc, all_labels, all_predictions = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Accuracy={train_acc:.2f}%")
        print(f"         Val Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
        for loss_name, loss_values in train_other_losses.items():
            print(f"         {loss_name}={loss_values[-1]:.4f}, Regular Loss={regular_loss:.4f}")

        # save weights
        model_save_path = weights_save_dir / f"model_weights_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_save_path)

        # Save confusion matrix
        class_names = sorted(label_mapping, key=label_mapping.get)
        plot_confusion_matrix(all_labels, all_predictions, class_names=class_names, epoch=epoch+1, save_dir=plots_save_dir)

        # Save metrics plot
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch=None, save_dir=plots_save_dir, additional_train_metrics=train_other_losses)

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

    print("Training completed.")

