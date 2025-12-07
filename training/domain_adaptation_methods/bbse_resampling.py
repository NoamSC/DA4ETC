import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
import copy
from collections import defaultdict, Counter


def create_bbse_resampled_dataloader(source_loader, target_loader, device='cuda', 
                                   num_epochs=10, learning_rate=0.001, 
                                   model_architecture=None, verbose=True):
    """
    Create a resampled DataLoader using Black Box Shift Estimation (BBSE) to estimate
    target label distribution without using target labels.
    
    Paper: https://arxiv.org/abs/1802.03916
    
    Args:
        source_loader: DataLoader with labeled source data
        target_loader: DataLoader with target data (labels will be ignored)
        device: Device to train the estimation model on
        num_epochs: Number of epochs to train the estimation model
        learning_rate: Learning rate for the estimation model
        model_architecture: Optional model architecture (if None, uses simple CNN)
        verbose: Whether to print progress information
        
    Returns:
        DataLoader: Resampled source dataloader based on estimated target distribution
    """
    
    if verbose:
        print("=" * 60)
        print("BBSE: BLACK BOX SHIFT ESTIMATION")
        print("=" * 60)
        print("Step 1: Training classifier on source domain...")
    
    # Step 1: Create and train a classifier on source data
    estimation_model = _create_estimation_model(source_loader, model_architecture, device)
    _train_estimation_model(estimation_model, source_loader, device, num_epochs, learning_rate, verbose)
    
    if verbose:
        print("\nStep 2: Computing confusion matrix on source domain...")
    
    # Step 2: Get confusion matrix on source data
    confusion_mat, source_class_counts = _get_confusion_matrix(estimation_model, source_loader, device)
    
    if verbose:
        print(f"Confusion matrix shape: {confusion_mat.shape}")
        print("Step 3: Predicting on target domain (without using target labels)...")
    
    # Step 3: Get predicted label distribution on target data (without using true labels)
    target_predicted_dist = _get_predicted_distribution(estimation_model, target_loader, device)
    
    if verbose:
        print("Step 4: Solving BBSE optimization problem...")
    
    # Step 4: Solve BBSE optimization to estimate true target distribution
    estimated_target_dist = _solve_bbse_optimization(confusion_mat, target_predicted_dist, verbose)
    
    if verbose:
        print("Step 5: Creating resampled dataloader...")
        _print_bbse_comparison(source_class_counts, target_predicted_dist, estimated_target_dist)
    
    # Step 5: Create resampled dataloader based on estimated target distribution
    resampled_loader = _create_resampled_loader_from_distribution(
        source_loader, estimated_target_dist, len(target_loader.dataset)
    )
    
    if verbose:
        print("BBSE resampling completed!")
        print("=" * 60)
    
    return resampled_loader


def _create_estimation_model(source_loader, model_architecture, device):
    """Create a simple model for label shift estimation."""
    
    # Get input shape and number of classes from source data
    sample_batch = next(iter(source_loader))
    input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
    num_classes = len(source_loader.dataset.get_class_counts())
    
    if model_architecture is not None:
        model = model_architecture
    else:
        # Simple CNN for estimation (you can customize this)
        if len(input_shape) == 3:  # Image data (C, H, W)
            model = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        else:
            # Fallback for other data types
            input_size = np.prod(input_shape)
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
    
    return model.to(device)


def _train_estimation_model(model, source_loader, device, num_epochs, learning_rate, verbose):
    """Train the estimation model on source data."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in source_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        if verbose and (epoch + 1) % max(1, num_epochs // 5) == 0:
            acc = 100. * correct / total
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss: {total_loss/len(source_loader):.4f}, "
                  f"Accuracy: {acc:.2f}%")


def _get_confusion_matrix(model, source_loader, device):
    """Get confusion matrix and class counts on source data."""
    
    model.eval()
    all_preds = []
    all_labels = []
    class_counts = defaultdict(int)
    
    with torch.no_grad():
        for inputs, labels in source_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for label in labels.cpu().numpy():
                class_counts[float(label)] += 1
    
    # Create confusion matrix (rows = true labels, cols = predicted labels)
    num_classes = len(class_counts)
    confusion_mat = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    # Normalize by row (convert to conditional probabilities P(predicted | true))
    confusion_mat = confusion_mat.astype(np.float64)
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_mat = confusion_mat / row_sums
    
    return confusion_mat, dict(class_counts)


def _get_predicted_distribution(model, target_loader, device):
    """Get predicted label distribution on target data (without using true labels)."""
    
    model.eval()
    predicted_counts = defaultdict(int)
    total_samples = 0
    
    with torch.no_grad():
        for inputs, _ in target_loader:  # Ignore true labels!
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for pred in predicted.cpu().numpy():
                predicted_counts[float(pred)] += 1
                total_samples += 1
    
    # Convert to probability distribution
    predicted_dist = {}
    for class_id in predicted_counts:
        predicted_dist[class_id] = predicted_counts[class_id] / total_samples
    
    return predicted_dist


def _solve_bbse_optimization(confusion_matrix, target_predicted_dist, verbose=True):
    """
    Solve the BBSE optimization problem:
    Find true target distribution y such that predicted_dist = confusion_matrix @ y
    
    This is formulated as a constrained optimization problem.
    """
    
    num_classes = confusion_matrix.shape[0]
    
    # Convert target_predicted_dist to numpy array
    y_hat = np.zeros(num_classes)
    for class_id, prob in target_predicted_dist.items():
        y_hat[int(class_id)] = prob
    
    # BBSE optimization: minimize ||C @ y - y_hat||^2 subject to y >= 0, sum(y) = 1
    def objective(y):
        return np.sum((confusion_matrix @ y - y_hat) ** 2)
    
    # Constraints: y >= 0 and sum(y) = 1
    constraints = [
        {'type': 'eq', 'fun': lambda y: np.sum(y) - 1.0}  # Sum to 1
    ]
    bounds = [(0, 1) for _ in range(num_classes)]  # Each probability >= 0
    
    # Initial guess: uniform distribution
    y0 = np.ones(num_classes) / num_classes
    
    # Solve optimization
    result = minimize(objective, y0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success and verbose:
        print(f"  Warning: BBSE optimization may not have converged: {result.message}")
    
    estimated_y = result.x
    
    # Convert back to dictionary format
    estimated_target_dist = {}
    for i, prob in enumerate(estimated_y):
        if prob > 1e-10:  # Filter out very small probabilities
            estimated_target_dist[float(i)] = prob
    
    if verbose:
        print(f"  Optimization converged: {result.success}")
        print(f"  Final objective value: {result.fun:.6f}")
    
    return estimated_target_dist


def _print_bbse_comparison(source_dist, target_predicted_dist, estimated_target_dist):
    """Print comparison of distributions."""
    
    print("\nBBSE Distribution Estimation Results:")
    print("-" * 70)
    print(f"{'Class':<8} {'Source':<15} {'Target (Pred)':<15} {'Target (Est)':<15}")
    print("-" * 70)
    
    all_classes = set(source_dist.keys()) | set(target_predicted_dist.keys()) | set(estimated_target_dist.keys())
    
    source_total = sum(source_dist.values())
    
    for class_id in sorted(all_classes):
        source_prob = source_dist.get(class_id, 0) / source_total
        pred_prob = target_predicted_dist.get(class_id, 0)
        est_prob = estimated_target_dist.get(class_id, 0)
        
        print(f"{int(class_id):<8} {source_prob:>8.3f}({source_prob*100:>4.1f}%) "
              f"{pred_prob:>8.3f}({pred_prob*100:>4.1f}%) "
              f"{est_prob:>8.3f}({est_prob*100:>4.1f}%)")
    
    print("-" * 70)


def _create_resampled_loader_from_distribution(source_loader, target_dist, target_size):
    """Create resampled dataloader based on estimated target distribution."""
    
    # Get source distribution
    source_class_counts = source_loader.dataset.get_class_counts()
    source_total = sum(source_class_counts.values())
    
    # Calculate sampling weights
    sampling_weights = {}
    for class_id in source_class_counts:
        source_prob = source_class_counts[class_id] / source_total
        target_prob = target_dist.get(class_id, 0)
        
        if source_prob > 0:
            sampling_weights[class_id] = target_prob / source_prob
        else:
            sampling_weights[class_id] = 0.0
    
    # Create sample weights for each item in source dataset
    sample_weights = []
    for i in range(len(source_loader.dataset)):
        _, label = source_loader.dataset[i]
        sample_weights.append(sampling_weights[float(label)])
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    # Create the resampled DataLoader
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=target_size,
        replacement=True
    )
    
    resampled_loader = DataLoader(
        source_loader.dataset,
        sampler=sampler,
        batch_size=source_loader.batch_size,
        num_workers=source_loader.num_workers
    )
    
    return resampled_loader


# Example usage function
def compare_resampling_methods(source_loader, target_loader, device='cuda'):
    """
    Compare the standard resampling (using target labels) vs BBSE resampling.
    This is for analysis purposes only.
    """
    
    print("COMPARING RESAMPLING METHODS")
    print("=" * 80)
    
    # Method 1: Standard resampling (uses target labels - cheating)
    print("Method 1: Standard resampling (using true target labels)")
    standard_resampled = create_resampled_dataloader(source_loader, target_loader)
    
    print("\n" + "=" * 80)
    
    # Method 2: BBSE resampling (no target labels)
    print("Method 2: BBSE resampling (no target labels used)")
    bbse_resampled = create_bbse_resampled_dataloader(source_loader, target_loader, device=device)
    
    return standard_resampled, bbse_resampled


# Import the standard resampling function for comparison
def create_resampled_dataloader(source_loader, target_loader):
    """Standard resampling using true target labels (for comparison)."""
    source_class_counts = source_loader.dataset.get_class_counts()
    target_class_counts = target_loader.dataset.get_class_counts()
    
    source_total = sum(source_class_counts.values())
    target_total = sum(target_class_counts.values())
    
    sampling_weights = {}
    for class_id in source_class_counts:
        if class_id in target_class_counts and source_class_counts[class_id] > 0:
            target_prop = target_class_counts[class_id] / target_total
            source_prop = source_class_counts[class_id] / source_total
            sampling_weights[class_id] = target_prop / source_prop
        else:
            sampling_weights[class_id] = 0.0
    
    sample_weights = []
    for i in range(len(source_loader.dataset)):
        _, label = source_loader.dataset[i]
        sample_weights.append(sampling_weights[float(label)])
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(target_loader.dataset),
        replacement=True
    )
    
    return DataLoader(
        source_loader.dataset,
        sampler=sampler,
        batch_size=target_loader.batch_size,
        num_workers=source_loader.num_workers
    )