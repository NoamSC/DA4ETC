import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(labels, predictions, class_names, epoch=None, save_dir=None):
    """
    Create a confusion matrix figure.

    Args:
    - labels (array-like): True labels.
    - predictions (array-like): Predicted labels.
    - class_names (list): List of class names corresponding to label indices.
    - epoch (int, optional): Current epoch number for title.
    - save_dir (Path, optional): Directory to save the plot (deprecated, for backward compatibility).

    Returns:
    - matplotlib.figure.Figure: The confusion matrix figure.
    """
    cm = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)

    if epoch is not None:
        ax.set_title(f"Confusion Matrix - Epoch {epoch}")
    else:
        ax.set_title("Confusion Matrix")

    plt.tight_layout()

    # Backward compatibility: save if save_dir is provided
    if save_dir is not None:
        save_path = save_dir / f"confusion_matrix_epoch_{epoch}.png"
        plt.savefig(save_path)
        plt.close()
        return None

    return fig

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir=None, additional_train_metrics=None, epoch=None):
    """
    Create a metrics figure with loss and accuracy plots.

    Args:
    - train_losses: List of training losses.
    - val_losses: List of validation losses.
    - train_accuracies: List of training accuracies.
    - val_accuracies: List of validation accuracies.
    - save_dir (Path, optional): Directory to save the plot (deprecated, for backward compatibility).
    - additional_train_metrics: Dictionary of additional metrics to plot.
    - epoch: Current epoch.

    Returns:
    - matplotlib.figure.Figure: The metrics figure.
    """
    fig = plt.figure(figsize=(12, 6))

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

    # Backward compatibility: save if save_dir is provided
    if save_dir is not None:
        save_path = save_dir / f"metrics{'_epoch_' + str(epoch) if epoch is not None else ''}.png"
        plt.savefig(save_path)
        plt.close()
        return None

    return fig

