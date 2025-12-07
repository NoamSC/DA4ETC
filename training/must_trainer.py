"""
MUST (Multi-Source Unsupervised-Supervised Transfer) Domain Adaptation Trainer

This module implements the MUST algorithm for domain adaptation, featuring:
- Teacher-Student architecture with domain-specific batch normalization
- Pseudo-label generation with confidence thresholding
- Ping-pong regularization for teacher-student consistency
- Warm start pre-training on source domain

Reference: Based on MUST domain adaptation methodology
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from training.trainer import validate
from training.visualization import plot_confusion_matrix
import matplotlib.pyplot as plt


# ============================================================================
# Batch Normalization Utilities
# ============================================================================

def get_bn_layer_params(bn_layer):
    """Extract parameters from a single BatchNorm layer."""
    return {
        'running_mean': bn_layer.running_mean.clone(),
        'running_var': bn_layer.running_var.clone(),
        'weight': bn_layer.weight.clone() if bn_layer.weight is not None else None,
        'bias': bn_layer.bias.clone() if bn_layer.bias is not None else None,
        'num_batches_tracked': bn_layer.num_batches_tracked.clone()
    }


def set_bn_layer_params(bn_layer, params):
    """Set parameters for a single BatchNorm layer."""
    bn_layer.running_mean.copy_(params['running_mean'])
    bn_layer.running_var.copy_(params['running_var'])
    if params['weight'] is not None:
        bn_layer.weight.data.copy_(params['weight'])
    if params['bias'] is not None:
        bn_layer.bias.data.copy_(params['bias'])
    bn_layer.num_batches_tracked.copy_(params['num_batches_tracked'])


def get_bn_params(model):
    """Extract all BatchNorm parameters from a model."""
    bn_params = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_params[name] = get_bn_layer_params(module)
    return bn_params


def inject_bn_params(model, bn_params):
    """Inject BatchNorm parameters into a model."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if name in bn_params:
                set_bn_layer_params(module, bn_params[name])


# ============================================================================
# MetaModel Class
# ============================================================================

class MetaModel:
    """
    Wrapper for a model with domain-specific batch normalization.

    Manages separate BN statistics for source and target domains, allowing
    the same model to adapt to different data distributions.
    """

    def __init__(self, base_model, criterion, device):
        """
        Initialize MetaModel.

        Args:
            base_model: The underlying neural network model
            criterion: Loss function
            device: torch device (cuda/cpu)
        """
        self.model = base_model
        self.criterion = criterion
        self.device = device
        self.optimizer = None  # Set externally after initialization
        self.domain2bn = {}  # Store BN params per domain
        self.current_domain = None

    def init_bn_for_domains(self, domain_names):
        """
        Initialize BN storage for each domain with current model state.

        Args:
            domain_names: List of domain names (e.g., ['source', 'target'])
        """
        current_bn = get_bn_params(self.model)
        for domain in domain_names:
            # Manually clone the BN parameters to avoid deepcopy issues
            self.domain2bn[domain] = {}
            for layer_name, layer_params in current_bn.items():
                self.domain2bn[domain][layer_name] = {
                    'running_mean': layer_params['running_mean'].clone(),
                    'running_var': layer_params['running_var'].clone(),
                    'weight': layer_params['weight'].clone() if layer_params['weight'] is not None else None,
                    'bias': layer_params['bias'].clone() if layer_params['bias'] is not None else None,
                    'num_batches_tracked': layer_params['num_batches_tracked'].clone()
                }
        print(f"Initialized BN for domains: {domain_names}")

    def load_bn(self, domain):
        """
        Load domain-specific BN parameters before forward pass.

        Args:
            domain: Domain name to load BN for
        """
        if domain not in self.domain2bn:
            raise ValueError(f"Domain '{domain}' not initialized. Call init_bn_for_domains() first.")
        inject_bn_params(self.model, self.domain2bn[domain])
        self.current_domain = domain

    def save_bn(self, domain=None):
        """
        Save current BN state after training.

        Args:
            domain: Domain name to save BN for. Uses current_domain if not specified.
        """
        if domain is None:
            domain = self.current_domain
        if domain is None:
            raise ValueError("No domain specified and no current domain set.")
        self.domain2bn[domain] = get_bn_params(self.model)

    def copy_weights_from(self, other_metamodel):
        """
        Copy model weights and BN parameters from another MetaModel.

        Args:
            other_metamodel: Source MetaModel to copy from
        """
        self.model.load_state_dict(other_metamodel.model.state_dict())
        # Manually clone BN parameters
        self.domain2bn = {}
        for domain, bn_params in other_metamodel.domain2bn.items():
            self.domain2bn[domain] = {}
            for layer_name, layer_params in bn_params.items():
                self.domain2bn[domain][layer_name] = {
                    'running_mean': layer_params['running_mean'].clone(),
                    'running_var': layer_params['running_var'].clone(),
                    'weight': layer_params['weight'].clone() if layer_params['weight'] is not None else None,
                    'bias': layer_params['bias'].clone() if layer_params['bias'] is not None else None,
                    'num_batches_tracked': layer_params['num_batches_tracked'].clone()
                }


# ============================================================================
# Warm Start Training
# ============================================================================

def warm_start(teacher, source_loader, target_loader, num_epochs, device, writer=None):
    """
    Pre-train teacher model on source domain with BN adaptation.

    Args:
        teacher: MetaModel instance for teacher
        source_loader: DataLoader for source domain
        target_loader: DataLoader for target domain (for BN adaptation)
        num_epochs: Number of warm start epochs
        device: torch device
        writer: TensorBoard SummaryWriter (optional)

    Returns:
        tuple: (train_losses, train_accuracies, val_losses, val_accuracies)
    """
    print("=" * 60)
    print("WARM START: Pre-training teacher on source domain")
    print("=" * 60)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        teacher.model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_x, batch_y in tqdm(source_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()

            # Load source BN
            teacher.load_bn('source')

            # Zero gradients
            teacher.optimizer.zero_grad()

            # Forward pass
            model_outputs = teacher.model(batch_x)
            train_class_preds = model_outputs['class_preds']
            loss = teacher.criterion(train_class_preds, batch_y)

            # Backward pass
            loss.backward()
            teacher.optimizer.step()

            # Save updated source BN
            teacher.save_bn('source')

            # Track metrics
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = train_class_preds.max(1)
            train_correct += predicted.eq(batch_y).sum().item()
            train_total += batch_y.size(0)

        # Calculate epoch metrics
        avg_train_loss = train_loss / train_total
        avg_train_acc = 100.0 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

        # Adapt target BN before validation (important!)
        teacher.load_bn('target')
        teacher.model.train()
        with torch.no_grad():
            for batch_x, _ in tqdm(target_loader, desc="Adapting target BN", leave=False):
                batch_x = batch_x.to(device)
                _ = teacher.model(batch_x)
        teacher.save_bn('target')

        # Validation phase - evaluate on target with adapted target BN
        teacher.model.eval()
        val_loss, val_acc, _, _ = validate(
            teacher.model,
            target_loader,
            teacher.criterion,
            device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('WarmStart/train_loss', avg_train_loss, epoch)
            writer.add_scalar('WarmStart/train_acc', avg_train_acc, epoch)
            writer.add_scalar('WarmStart/val_loss', val_loss, epoch)
            writer.add_scalar('WarmStart/val_acc', val_acc, epoch)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    print("\nWarm start completed!")
    print(f"Final training accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies


# ============================================================================
# MUST Training - Single Iteration
# ============================================================================

def train_must_one_iteration(teacher, student, source_batch, target_iter,
                              target_batches_per_iter, alpha, pseudo_threshold, device):
    """
    Execute one MUST training iteration with 4 steps.

    Steps:
        1. Train teacher on source batch
        2. Generate pseudo-labels on target
        3. Train student on high-confidence pseudo-labels
        4. Ping-pong regularization (teacher ← student)

    Args:
        teacher: MetaModel for teacher network
        student: MetaModel for student network
        source_batch: Tuple of (inputs, labels) from source domain
        target_iter: Iterator for target domain batches
        target_batches_per_iter: Number of target batches to process
        alpha: Ping-pong loss weight
        pseudo_threshold: Confidence threshold for pseudo-labels
        device: torch device

    Returns:
        float: Pseudo-label usage percentage
    """
    source_batch_x, source_batch_y = source_batch
    source_batch_x = source_batch_x.to(device)
    source_batch_y = source_batch_y.to(device).long()

    # ========== STEP 1: Train teacher on source ==========
    teacher.load_bn('source')
    teacher.optimizer.zero_grad()

    # Forward pass (don't step yet - will step after ping-pong)
    teacher_outputs = teacher.model(source_batch_x)
    teacher_preds = teacher_outputs['class_preds']
    teacher_loss = teacher.criterion(teacher_preds, source_batch_y)
    teacher_loss.backward()

    # Save updated source BN
    teacher.save_bn('source')

    # ========== STEPS 2 & 3: Generate pseudo-labels and train student ==========
    teacher.load_bn('target')
    student.load_bn('target')

    teacher.model.eval()  # Teacher in eval mode for pseudo-label generation
    student.model.train()  # Student in train mode

    total_target_samples = 0
    high_conf_samples = 0

    for _ in range(target_batches_per_iter):
        try:
            target_batch_x, _ = next(target_iter)  # Ignore labels (unsupervised)
        except StopIteration:
            break

        target_batch_x = target_batch_x.to(device)
        total_target_samples += target_batch_x.size(0)

        # Generate pseudo-labels with teacher
        with torch.no_grad():
            teacher_target_outputs = teacher.model(target_batch_x)
            teacher_target_logits = teacher_target_outputs['class_preds']
            teacher_target_probs = torch.softmax(teacher_target_logits, dim=1)

        # Threshold filtering
        max_probs, pseudo_labels = torch.max(teacher_target_probs, dim=1)
        high_conf_mask = max_probs > pseudo_threshold
        high_conf_samples += high_conf_mask.sum().item()

        if high_conf_mask.sum() > 0:
            # Train student on high-confidence samples
            student.optimizer.zero_grad()

            student_outputs = student.model(target_batch_x[high_conf_mask])
            student_preds = student_outputs['class_preds']

            # Soft loss: match teacher's probability distribution
            # KL divergence expects log probabilities as input
            student_log_probs = torch.log_softmax(student_preds, dim=1)
            teacher_probs_filtered = teacher_target_probs[high_conf_mask].detach()

            student_loss = student.criterion(student_log_probs, teacher_probs_filtered)
            student_loss.backward()
            student.optimizer.step()

    student.save_bn('target')

    # Track pseudo-label usage
    pseudo_usage = 100.0 * high_conf_samples / total_target_samples if total_target_samples > 0 else 0

    # ========== STEP 4: Ping-pong regularization ==========
    # Make student predict on source batch to regularize teacher
    student.load_bn('source')
    student.model.eval()

    student_source_outputs = student.model(source_batch_x)
    student_source_preds = student_source_outputs['class_preds']

    # Ping-pong loss: teacher should be consistent with student on source
    ping_pong_loss = alpha * nn.CrossEntropyLoss()(student_source_preds, source_batch_y)
    ping_pong_loss.backward()

    # Now step the teacher optimizer (accumulated gradients from Step 1 + Step 4)
    teacher.optimizer.step()
    teacher.save_bn('target')  # Update target BN for teacher

    return pseudo_usage


# ============================================================================
# MUST Visualization
# ============================================================================

def create_must_metrics_figure(metrics, pseudo_threshold):
    """
    Create comprehensive 4-panel MUST visualization figure.

    Args:
        metrics: Dictionary containing training metrics
        pseudo_threshold: Threshold value for reference line

    Returns:
        matplotlib.figure.Figure: The MUST metrics figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Teacher Performance
    axes[0, 0].plot(metrics['iterations'], metrics['teacher_source_acc'],
                    marker='o', label='Source', linewidth=2)
    axes[0, 0].plot(metrics['iterations'], metrics['teacher_target_acc'],
                    marker='s', label='Target', linewidth=2)
    axes[0, 0].set_title('Teacher Network Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Student Performance
    axes[0, 1].plot(metrics['iterations'], metrics['student_source_acc'],
                    marker='o', label='Source', linewidth=2, color='orange')
    axes[0, 1].plot(metrics['iterations'], metrics['student_target_acc'],
                    marker='s', label='Target', linewidth=2, color='red')
    axes[0, 1].set_title('Student Network Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Target Domain Comparison
    axes[1, 0].plot(metrics['iterations'], metrics['teacher_target_acc'],
                    marker='o', label='Teacher', linewidth=2)
    axes[1, 0].plot(metrics['iterations'], metrics['student_target_acc'],
                    marker='s', label='Student', linewidth=2)
    axes[1, 0].set_title('Target Domain: Teacher vs Student', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Pseudo-label Usage
    axes[1, 1].plot(metrics['iterations'], metrics['pseudo_label_usage'],
                    marker='D', linewidth=2, color='green')
    axes[1, 1].set_title('Pseudo-Label Usage Over Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('High-Confidence Samples (%)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=pseudo_threshold*100, color='r', linestyle='--',
                        label=f'Threshold={pseudo_threshold}')
    axes[1, 1].legend(fontsize=11)

    plt.tight_layout()
    return fig


# ============================================================================
# Main MUST Training Loop
# ============================================================================

def train_must(teacher_model, student_model, source_loader, target_loader,
               criterion, teacher_optimizer, student_optimizer, must_params,
               device, weights_save_dir, plots_save_dir, label_mapping,
               resume_checkpoint=None, resume_from_iteration=0):
    """
    Main MUST training loop with TensorBoard logging and best model tracking.
    Supports resuming from checkpoint.

    Args:
        teacher_model: Teacher network (ConfigurableCNN)
        student_model: Student network (ConfigurableCNN)
        source_loader: DataLoader for source (labeled) domain
        target_loader: DataLoader for target (unlabeled) domain
        criterion: Loss function for classification
        teacher_optimizer: Optimizer for teacher
        student_optimizer: Optimizer for student
        must_params: Dictionary of MUST hyperparameters
        device: torch device
        weights_save_dir: Directory to save model checkpoints
        plots_save_dir: Directory to save plots
        label_mapping: Dictionary mapping labels to indices
        resume_checkpoint: Checkpoint dictionary to resume from (optional)
        resume_from_iteration: Iteration number to resume from (optional)

    Returns:
        dict: Training metrics and final results
    """
    # Initialize TensorBoard writer
    tensorboard_dir = plots_save_dir.parent / 'tensorboard'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # Track best models
    best_teacher_target_acc = 0.0
    best_student_target_acc = 0.0
    best_teacher_iter = 0
    best_student_iter = 0

    # Extract MUST parameters
    iterations = must_params['iterations']
    alpha = must_params['alpha']
    pseudo_threshold = must_params['pseudo_threshold']
    warm_start_epochs = must_params['warm_start_epochs']
    eval_every = must_params['eval_every']
    target_batches_per_iter = must_params['target_batches_per_iter']

    # Wrap models in MetaModel
    teacher = MetaModel(teacher_model, criterion, device)
    teacher.optimizer = teacher_optimizer

    student_criterion = nn.KLDivLoss(reduction='batchmean')  # Use KL divergence for soft labels
    student = MetaModel(student_model, student_criterion, device)
    student.optimizer = student_optimizer

    # Initialize domain-specific batch normalization
    teacher.init_bn_for_domains(['source', 'target'])
    student.init_bn_for_domains(['source', 'target'])

    # Initialize metrics tracking
    metrics = {
        'teacher_source_acc': [],
        'teacher_target_acc': [],
        'student_source_acc': [],
        'student_target_acc': [],
        'pseudo_label_usage': [],
        'iterations': []
    }

    # Resume from checkpoint if provided
    start_iteration = 0
    if resume_checkpoint is not None and resume_from_iteration > 0:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*60}")

        # Load teacher state
        teacher.model.load_state_dict(resume_checkpoint['teacher_state_dict'])
        teacher.domain2bn['source'] = resume_checkpoint['teacher_bn_source']
        teacher.domain2bn['target'] = resume_checkpoint['teacher_bn_target']
        print(f"Loaded teacher from iteration {resume_checkpoint['iteration']}")

        # Load student state
        student.model.load_state_dict(resume_checkpoint['student_state_dict'])
        student.domain2bn['source'] = resume_checkpoint['student_bn_source']
        student.domain2bn['target'] = resume_checkpoint['student_bn_target']
        print(f"Loaded student from iteration {resume_checkpoint['iteration']}")

        # Load metrics if available
        if 'metrics' in resume_checkpoint:
            metrics = resume_checkpoint['metrics']
            print(f"Loaded training history up to iteration {metrics['iterations'][-1]}")

            # Find best models so far
            if metrics['teacher_target_acc']:
                best_teacher_target_acc = max(metrics['teacher_target_acc'])
                best_teacher_iter = metrics['iterations'][metrics['teacher_target_acc'].index(best_teacher_target_acc)]
                print(f"Best teacher target accuracy so far: {best_teacher_target_acc:.2f}% at iteration {best_teacher_iter}")

            if metrics['student_target_acc']:
                best_student_target_acc = max(metrics['student_target_acc'])
                best_student_iter = metrics['iterations'][metrics['student_target_acc'].index(best_student_target_acc)]
                print(f"Best student target accuracy so far: {best_student_target_acc:.2f}% at iteration {best_student_iter}")

        start_iteration = resume_from_iteration
        print(f"Resuming training from iteration {start_iteration}")
        print(f"{'='*60}\n")
    else:
        # Warm start: Pre-train teacher on source (only if not resuming)
        warm_start(teacher, source_loader, target_loader, warm_start_epochs, device, writer=writer)

        # Copy teacher weights to student
        print("\nCopying teacher weights to student...")
        student.copy_weights_from(teacher)

    print("\n" + "=" * 60)
    print("MUST TRAINING")
    print("=" * 60)

    # Create iterators
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for iteration in tqdm(range(start_iteration, iterations), desc="MUST Training", initial=start_iteration, total=iterations):
        # Get source batch
        try:
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_batch = next(source_iter)

        # Execute one MUST iteration
        pseudo_usage = train_must_one_iteration(
            teacher, student, source_batch, target_iter,
            target_batches_per_iter, alpha, pseudo_threshold, device
        )

        # Periodic evaluation
        if iteration % eval_every == 0 or iteration == iterations - 1:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration} / {iterations}")
            print(f"{'='*60}")

            # Teacher on source
            teacher.load_bn('source')
            t_src_loss, t_src_acc, t_src_labels, t_src_preds = validate(
                teacher.model, source_loader, teacher.criterion, device
            )

            # Teacher on target
            teacher.load_bn('target')
            t_tgt_loss, t_tgt_acc, t_tgt_labels, t_tgt_preds = validate(
                teacher.model, target_loader, teacher.criterion, device
            )

            # Student on source
            student.load_bn('source')
            s_src_loss, s_src_acc, s_src_labels, s_src_preds = validate(
                student.model, source_loader, nn.CrossEntropyLoss(), device
            )

            # Student on target
            student.load_bn('target')
            s_tgt_loss, s_tgt_acc, s_tgt_labels, s_tgt_preds = validate(
                student.model, target_loader, nn.CrossEntropyLoss(), device
            )

            # Store metrics
            metrics['teacher_source_acc'].append(t_src_acc)
            metrics['teacher_target_acc'].append(t_tgt_acc)
            metrics['student_source_acc'].append(s_src_acc)
            metrics['student_target_acc'].append(s_tgt_acc)
            metrics['pseudo_label_usage'].append(pseudo_usage)
            metrics['iterations'].append(iteration)

            # Log to TensorBoard
            writer.add_scalar('MUST/teacher_source_acc', t_src_acc, iteration)
            writer.add_scalar('MUST/teacher_target_acc', t_tgt_acc, iteration)
            writer.add_scalar('MUST/student_source_acc', s_src_acc, iteration)
            writer.add_scalar('MUST/student_target_acc', s_tgt_acc, iteration)
            writer.add_scalar('MUST/pseudo_label_usage', pseudo_usage, iteration)
            writer.add_scalar('MUST/teacher_source_loss', t_src_loss, iteration)
            writer.add_scalar('MUST/teacher_target_loss', t_tgt_loss, iteration)
            writer.add_scalar('MUST/student_source_loss', s_src_loss, iteration)
            writer.add_scalar('MUST/student_target_loss', s_tgt_loss, iteration)

            print(f"Teacher: Source={t_src_acc:.2f}% | Target={t_tgt_acc:.2f}%")
            print(f"Student: Source={s_src_acc:.2f}% | Target={s_tgt_acc:.2f}%")
            print(f"Pseudo-label usage: {pseudo_usage:.1f}%")

            # Save best teacher model
            if t_tgt_acc > best_teacher_target_acc:
                best_teacher_target_acc = t_tgt_acc
                best_teacher_iter = iteration
                best_teacher_path = weights_save_dir / "best_teacher.pth"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': teacher.model.state_dict(),
                    'bn_source': teacher.domain2bn['source'],
                    'bn_target': teacher.domain2bn['target'],
                    'target_acc': t_tgt_acc,
                    'source_acc': t_src_acc,
                }, best_teacher_path)
                print(f"*** New best teacher model saved! Target Acc: {t_tgt_acc:.2f}% ***")

            # Save best student model
            if s_tgt_acc > best_student_target_acc:
                best_student_target_acc = s_tgt_acc
                best_student_iter = iteration
                best_student_path = weights_save_dir / "best_student.pth"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': student.model.state_dict(),
                    'bn_source': student.domain2bn['source'],
                    'bn_target': student.domain2bn['target'],
                    'target_acc': s_tgt_acc,
                    'source_acc': s_src_acc,
                }, best_student_path)
                print(f"*** New best student model saved! Target Acc: {s_tgt_acc:.2f}% ***")

            # Log confusion matrices to TensorBoard
            class_names = sorted(label_mapping, key=label_mapping.get)

            t_src_fig = plot_confusion_matrix(t_src_labels, t_src_preds, class_names)
            if t_src_fig is not None:
                writer.add_figure('Confusion_Matrix/teacher_source', t_src_fig, iteration)
                plt.close(t_src_fig)

            t_tgt_fig = plot_confusion_matrix(t_tgt_labels, t_tgt_preds, class_names)
            if t_tgt_fig is not None:
                writer.add_figure('Confusion_Matrix/teacher_target', t_tgt_fig, iteration)
                plt.close(t_tgt_fig)

            s_src_fig = plot_confusion_matrix(s_src_labels, s_src_preds, class_names)
            if s_src_fig is not None:
                writer.add_figure('Confusion_Matrix/student_source', s_src_fig, iteration)
                plt.close(s_src_fig)

            s_tgt_fig = plot_confusion_matrix(s_tgt_labels, s_tgt_preds, class_names)
            if s_tgt_fig is not None:
                writer.add_figure('Confusion_Matrix/student_target', s_tgt_fig, iteration)
                plt.close(s_tgt_fig)

            # Log MUST metrics plot to TensorBoard
            must_fig = create_must_metrics_figure(metrics, pseudo_threshold)
            if must_fig is not None:
                writer.add_figure('MUST_Metrics/summary', must_fig, iteration)
                plt.close(must_fig)

            # Save checkpoint
            checkpoint = {
                'iteration': iteration,
                'teacher_state_dict': teacher.model.state_dict(),
                'teacher_bn_source': teacher.domain2bn['source'],
                'teacher_bn_target': teacher.domain2bn['target'],
                'student_state_dict': student.model.state_dict(),
                'student_bn_source': student.domain2bn['source'],
                'student_bn_target': student.domain2bn['target'],
                'metrics': metrics,
                'must_params': must_params,
            }

            checkpoint_path = weights_save_dir / f'must_checkpoint_iter_{iteration}.pth'
            torch.save(checkpoint, checkpoint_path)

            # Save training history
            history_path = plots_save_dir / 'must_training_history.pth'
            torch.save(metrics, history_path)

    # Close TensorBoard writer
    writer.close()

    print("\n" + "=" * 60)
    print("MUST training completed!")
    print("=" * 60)

    # Final results
    final_results = {
        'student_target_acc': metrics['student_target_acc'][-1],
        'teacher_target_acc': metrics['teacher_target_acc'][-1],
        'improvement': metrics['student_target_acc'][-1] - metrics['teacher_target_acc'][-1],
        'best_student_target_acc': best_student_target_acc,
        'best_teacher_target_acc': best_teacher_target_acc,
        'best_student_iter': best_student_iter,
        'best_teacher_iter': best_teacher_iter,
    }

    print(f"\nFinal Student Target Accuracy: {final_results['student_target_acc']:.2f}%")
    print(f"Final Teacher Target Accuracy: {final_results['teacher_target_acc']:.2f}%")
    print(f"Student vs Teacher Improvement: {final_results['improvement']:+.2f}%")
    print(f"\nBest Student Target Accuracy: {best_student_target_acc:.2f}% (iteration {best_student_iter})")
    print(f"Best Teacher Target Accuracy: {best_teacher_target_acc:.2f}% (iteration {best_teacher_iter})")
    print(f"\nTensorBoard logs saved to: {tensorboard_dir}")
    print(f"To view logs, run: tensorboard --logdir {tensorboard_dir}")

    return {
        'metrics': metrics,
        'final_results': final_results,
        'teacher': teacher,
        'student': student
    }
