import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(labels, predictions, class_names, epoch, save_dir):
    """
    Plot and save a confusion matrix for the given labels and predictions.

    Args:
    - labels (array-like): True labels.
    - predictions (array-like): Predicted labels.
    - class_names (list): List of class names corresponding to label indices.
    - epoch (int): Current epoch number.
    - save_dir (Path): Directory to save the plot.
    """
    cm = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

    save_path = save_dir / f"confusion_matrix_epoch_{epoch}.png"
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.savefig(save_path)
    plt.close()

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir, additional_train_metrics=None, epoch=None):
    """
    Extended plot_metrics function to include additional training metrics.
    
    Args:
    - train_losses: List of training losses.
    - val_losses: List of validation losses.
    - train_accuracies: List of training accuracies.
    - val_accuracies: List of validation accuracies.
    - epoch: Current epoch.
    - save_dir: Directory to save the plot.
    - additional_train_metrics: Dictionary of additional metrics to plot.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    if additional_train_metrics:
        for name, values in additional_train_metrics.items():
            plt.plot(values, label=name)
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    save_path = save_dir / f"metrics{'_epoch_' + str(epoch) if epoch is not None else ''}.png"
    plt.savefig(save_path)
    plt.close()

