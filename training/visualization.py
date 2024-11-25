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

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch, save_dir):
    """
    Plot and save training/validation losses and accuracies over epochs.

    Args:
    - train_losses (list): Training loss per batch.
    - val_losses (list): Validation loss per batch.
    - train_accuracies (list): Training accuracy per batch.
    - val_accuracies (list): Validation accuracy per batch.
    - epoch (int): Current epoch number.
    - save_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(10, 5))
    
    # Losses plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracies plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', alpha=0.7)
    plt.plot(val_accuracies, label='Validation Accuracy', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Save the figure
    save_path = save_dir / f"training_progress.png"
    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(save_path)
    plt.close()
