"""
Visualize GRL lambda scheduling to help tune hyperparameters.

This script plots how GRL lambda changes over training for different
hyperparameter settings.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_grl_schedule(num_epochs, lambda_rgl, gamma):
    """
    Compute GRL lambda schedule over training.

    Args:
        num_epochs: Total number of training epochs
        lambda_rgl: Base GRL strength
        gamma: Schedule steepness parameter

    Returns:
        epochs: Array of epoch numbers
        lambdas: Array of GRL lambda values
    """
    epochs = np.linspace(0, num_epochs, 1000)
    p = epochs / num_epochs  # Progress from 0 to 1

    # DANN schedule: lambda = lambda_rgl * (2 / (1 + exp(-gamma * p)) - 1)
    factor = 2 / (1 + np.exp(-gamma * p)) - 1
    lambdas = lambda_rgl * factor

    return epochs, lambdas


def plot_grl_schedules(num_epochs=15, output_dir='test_results'):
    """
    Plot GRL schedules for different hyperparameter configurations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Effect of lambda_rgl (fixed gamma=10)
    ax = axes[0, 0]
    gamma = 10
    for lambda_rgl in [0.01, 0.05, 0.1, 0.5, 1.0]:
        epochs, lambdas = compute_grl_schedule(num_epochs, lambda_rgl, gamma)
        ax.plot(epochs, lambdas, label=f'λ_rgl={lambda_rgl}', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('GRL Lambda', fontsize=12)
    ax.set_title(f'Effect of λ_rgl (γ={gamma})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Effect of gamma (fixed lambda_rgl=0.1)
    ax = axes[0, 1]
    lambda_rgl = 0.1
    for gamma in [1, 5, 10, 50, 100]:
        epochs, lambdas = compute_grl_schedule(num_epochs, lambda_rgl, gamma)
        ax.plot(epochs, lambdas, label=f'γ={gamma}', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('GRL Lambda', fontsize=12)
    ax.set_title(f'Effect of γ (λ_rgl={lambda_rgl})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Current config vs recommended
    ax = axes[1, 0]

    # Current config
    current_lambda_rgl = 0.002168
    current_gamma = 1
    epochs, lambdas = compute_grl_schedule(num_epochs, current_lambda_rgl, current_gamma)
    ax.plot(epochs, lambdas, label=f'Current (λ_rgl={current_lambda_rgl:.4f}, γ={current_gamma})',
            linewidth=3, linestyle='--', color='red')

    # Recommended config
    rec_lambda_rgl = 0.1
    rec_gamma = 10
    epochs, lambdas = compute_grl_schedule(num_epochs, rec_lambda_rgl, rec_gamma)
    ax.plot(epochs, lambdas, label=f'Recommended (λ_rgl={rec_lambda_rgl}, γ={rec_gamma})',
            linewidth=3, color='green')

    # DANN paper config
    paper_lambda_rgl = 1.0
    paper_gamma = 10
    epochs, lambdas = compute_grl_schedule(num_epochs, paper_lambda_rgl, paper_gamma)
    ax.plot(epochs, lambdas, label=f'DANN Paper (λ_rgl={paper_lambda_rgl}, γ={paper_gamma})',
            linewidth=3, linestyle='-.', color='blue')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('GRL Lambda', fontsize=12)
    ax.set_title('Current vs Recommended vs Paper', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 4: Schedule factor only (2 / (1 + exp(-gamma * p)) - 1)
    ax = axes[1, 1]
    p = np.linspace(0, 1, 1000)

    for gamma in [1, 5, 10, 50, 100]:
        factor = 2 / (1 + np.exp(-gamma * p)) - 1
        ax.plot(p, factor, label=f'γ={gamma}', linewidth=2)

    ax.set_xlabel('Training Progress (p)', fontsize=12)
    ax.set_ylabel('Schedule Factor', fontsize=12)
    ax.set_title('Schedule Factor: 2/(1+exp(-γp)) - 1', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    # Save figure
    save_path = output_dir / 'grl_lambda_schedules.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nGRL schedule visualization saved to: {save_path}")

    # Also print numerical values at key epochs
    print("\n" + "="*80)
    print("GRL Lambda Values at Key Epochs")
    print("="*80)

    configs = [
        ("Current Config", 0.002168, 1),
        ("Recommended", 0.1, 10),
        ("DANN Paper", 1.0, 10),
    ]

    for name, lambda_rgl, gamma in configs:
        print(f"\n{name} (λ_rgl={lambda_rgl}, γ={gamma}):")
        print(f"  {'Epoch':>6s}  {'Progress':>10s}  {'Factor':>10s}  {'GRL Lambda':>12s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")

        for epoch in [0, 1, 5, 10, 15]:
            p = epoch / num_epochs
            factor = 2 / (1 + np.exp(-gamma * p)) - 1
            grl_lambda = lambda_rgl * factor
            print(f"  {epoch:6d}  {p:10.2f}  {factor:10.4f}  {grl_lambda:12.6f}")

    print("\n" + "="*80)
    print("\nKey Insights:")
    print("- Current config: GRL lambda stays < 0.002 throughout training (TOO SMALL)")
    print("- Recommended: GRL lambda reaches ~0.1 by epoch 5 (GOOD)")
    print("- DANN Paper: GRL lambda reaches ~1.0 by epoch 5 (AGGRESSIVE)")
    print("\nRule of thumb: GRL lambda should be comparable to loss magnitudes (0.01-1.0)")
    print("="*80)

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize GRL lambda scheduling")
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=15,
        help='Number of training epochs to visualize (default: 15)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_results',
        help='Directory to save visualization (default: test_results)'
    )
    args = parser.parse_args()

    plot_grl_schedules(num_epochs=args.num_epochs, output_dir=args.output_dir)
