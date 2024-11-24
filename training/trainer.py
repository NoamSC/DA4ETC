import torch
from tqdm import tqdm
from training.visualization import plot_confusion_matrix, plot_metrics

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir, num_classes):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device).long()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total += labels.size(0)

        train_losses.append(train_loss / train_total)
        train_accuracies.append(100.0 * train_correct / train_total)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_labels, all_predictions = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_losses.append(val_loss / val_total)
        val_accuracies.append(100.0 * val_correct / val_total)

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Accuracy={train_accuracies[-1]:.2f}%")
        print(f"              Val Loss={val_losses[-1]:.4f}, Accuracy={val_accuracies[-1]:.2f}%")

        # Save confusion matrix
        plot_confusion_matrix(all_labels, all_predictions, class_names=range(num_classes), epoch=epoch+1, save_dir=save_dir)

        # Save metrics plot
        plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch=epoch+1, save_dir=save_dir)

        # Save model weights
        model_save_path = save_dir / f"model_weights_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_save_path)
