import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import torch

def plot_confusion_matrix(labels, predictions, class_names, epoch=None, save_dir=None, normalize='true'):
    """
    Create a confusion matrix figure with improved readability for many classes.

    Args:
    - labels (array-like): True labels.
    - predictions (array-like): Predicted labels.
    - class_names (list): List of class names corresponding to label indices.
    - epoch (int, optional): Current epoch number for title.
    - save_dir (Path, optional): Directory to save the plot (deprecated, for backward compatibility).
    - normalize (str): 'true' (row-wise), 'pred' (column-wise), 'all', or None.

    Returns:
    - matplotlib.figure.Figure: The confusion matrix figure.
    """
    cm = confusion_matrix(labels, predictions, labels=range(len(class_names)), normalize=normalize)

    # For many classes, use a heatmap without text annotations
    n_classes = len(class_names)

    if n_classes > 50:
        # Large number of classes: simple heatmap without annotations
        fig, ax = plt.subplots(figsize=(16, 14))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
        ax.figure.colorbar(im, ax=ax)

        # Set ticks sparingly
        tick_marks = np.arange(0, n_classes, max(1, n_classes // 20))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([class_names[i] if i < len(class_names) else '' for i in tick_marks],
                          rotation=90, ha='center', fontsize=6)
        ax.set_yticklabels([class_names[i] if i < len(class_names) else '' for i in tick_marks],
                          fontsize=6)

        ax.set_ylabel('True label', fontsize=10)
        ax.set_xlabel('Predicted label', fontsize=10)
    else:
        # Moderate number of classes: use ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.5), max(10, n_classes * 0.5)))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax,
                 include_values=(n_classes <= 30))  # Only show values if not too many classes

    norm_str = f" (Normalized by {normalize})" if normalize else ""
    if epoch is not None:
        ax.set_title(f"Confusion Matrix{norm_str} - Epoch {epoch}", fontsize=12)
    else:
        ax.set_title(f"Confusion Matrix{norm_str}", fontsize=12)

    plt.tight_layout()

    # Backward compatibility: save if save_dir is provided
    if save_dir is not None:
        save_path = save_dir / f"confusion_matrix_epoch_{epoch}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None

    return fig

def compute_topk_accuracy(logits, labels, k_values=[3, 5, 10]):
    """
    Compute top-K accuracy metrics.

    Args:
    - logits (torch.Tensor or np.ndarray): Model output logits or probabilities, shape (N, num_classes)
    - labels (torch.Tensor or np.ndarray): True labels, shape (N,)
    - k_values (list): List of K values to compute top-K accuracy for

    Returns:
    - dict: Dictionary mapping 'top_k' to accuracy percentage
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    if logits.dim() == 1:
        # Already predictions, not logits
        raise ValueError("Expected logits/probabilities, got 1D predictions")

    results = {}
    for k in k_values:
        _, topk_preds = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
        topk_acc = correct.any(dim=1).float().mean().item() * 100
        results[f'top_{k}'] = topk_acc

    return results


def compute_per_class_precision(labels, predictions, num_classes):
    """
    Compute per-class precision and return mean.

    Args:
    - labels (array-like): True labels
    - predictions (array-like): Predicted labels
    - num_classes (int): Total number of classes

    Returns:
    - dict: Dictionary with 'mean_per_class_precision' and 'per_class_precision' array
    """
    # Compute precision for each class
    precision_per_class = precision_score(
        labels, predictions,
        labels=range(num_classes),
        average=None,
        zero_division=0
    )

    mean_precision = np.nanmean(precision_per_class)

    return {
        'mean_per_class_precision': mean_precision * 100,  # Convert to percentage
        'per_class_precision': precision_per_class
    }


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
            if 'loss' in name.lower():
                plt.plot(values, label=name)
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    if additional_train_metrics:
        for name, values in additional_train_metrics.items():
            if 'accuracy' in name.lower() or 'top' in name.lower():
                plt.plot(values, label=name)
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


def create_tsne_visualization(features, labels, class_names, epoch=None, save_dir=None,
                              n_samples=5000, perplexity=30, random_state=42):
    """
    Create an interactive t-SNE visualization using plotly.

    Args:
    - features (np.ndarray): Feature vectors, shape (N, feature_dim)
    - labels (np.ndarray): True labels, shape (N,)
    - class_names (list): List of class names
    - epoch (int, optional): Current epoch number
    - save_dir (Path, optional): Directory to save the HTML plot
    - n_samples (int): Maximum number of samples to visualize (for performance)
    - perplexity (int): t-SNE perplexity parameter
    - random_state (int): Random seed for reproducibility

    Returns:
    - plotly.graph_objects.Figure: Interactive t-SNE plot
    """
    # Subsample if needed
    if len(features) > n_samples:
        indices = np.random.RandomState(random_state).choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    # Run t-SNE
    print(f"Computing t-SNE with {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(features) - 1),
                random_state=random_state, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    # Create hover text with class names
    hover_text = [f"Class: {class_names[label]}<br>Label ID: {label}" for label in labels]

    # Create plotly scatter plot
    fig = go.Figure(data=go.Scattergl(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        mode='markers',
        marker=dict(
            size=4,
            color=labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Class ID"),
            line=dict(width=0.5, color='white'),
            opacity=0.7
        ),
        text=hover_text,
        hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
    ))

    title = f"t-SNE Visualization - Epoch {epoch}" if epoch is not None else "t-SNE Visualization"
    fig.update_layout(
        title=title,
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        width=1000,
        height=800,
        hovermode='closest'
    )

    # Save if directory provided
    if save_dir is not None:
        save_path = save_dir / f"tsne_epoch_{epoch}.html"
        fig.write_html(str(save_path))
        print(f"t-SNE visualization saved to {save_path}")

    return fig

