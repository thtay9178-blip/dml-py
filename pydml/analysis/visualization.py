"""
Visualization utilities for DML-PY.

This module provides tools for visualizing training progress, model behavior,
and collaborative learning dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import torch


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history with loss and accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    history: Dict[str, List[float]],
    num_models: int,
    metric: str = 'val_acc',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot comparison of individual model performances.
    
    Args:
        history: Training history
        num_models: Number of models
        metric: Metric to plot ('val_acc' or 'train_acc')
        save_path: Path to save figure
        show: Whether to display figure
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    
    for i in range(num_models):
        key = f'{metric}_model_{i}'
        if key in history and history[key]:
            plt.plot(history[key], label=f'Model {i}', 
                    linewidth=2, color=colors[i], marker='o', markersize=3)
    
    # Plot average
    if metric in history and history[metric]:
        plt.plot(history[metric], label='Average', 
                linewidth=3, color='black', linestyle='--', alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Model-wise {metric.replace("_", " ").title()}', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_entropy_distribution(
    entropies: torch.Tensor,
    title: str = 'Prediction Entropy Distribution',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot distribution of prediction entropies.
    
    Args:
        entropies: Tensor of entropy values
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    plt.figure(figsize=(10, 6))
    
    entropies_np = entropies.cpu().numpy() if torch.is_tensor(entropies) else entropies
    
    plt.hist(entropies_np, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(entropies_np.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {entropies_np.mean():.3f}')
    
    plt.xlabel('Entropy', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_agreement_matrix(
    models: List[torch.nn.Module],
    data_loader,
    device: str = 'cuda',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot agreement matrix showing how often models agree with each other.
    
    Args:
        models: List of models
        data_loader: Data loader
        device: Device to use
        save_path: Path to save figure
        show: Whether to display figure
    """
    n_models = len(models)
    agreement_matrix = np.zeros((n_models, n_models))
    
    # Set models to eval
    for model in models:
        model.eval()
    
    # Collect predictions
    all_predictions = [[] for _ in range(n_models)]
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            
            for i, model in enumerate(models):
                outputs = model(inputs)
                _, preds = outputs.max(1)
                all_predictions[i].extend(preds.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = [np.array(preds) for preds in all_predictions]
    
    # Compute agreement rates
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                agreement_matrix[i, j] = 100.0
            else:
                agreement = (all_predictions[i] == all_predictions[j]).mean() * 100
                agreement_matrix[i, j] = agreement
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(agreement_matrix, cmap='YlOrRd', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Agreement Rate (%)', fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels([f'Model {i}' for i in range(n_models)])
    ax.set_yticklabels([f'Model {i}' for i in range(n_models)])
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            text = ax.text(j, i, f'{agreement_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Model Agreement Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return agreement_matrix


def plot_robustness_comparison(
    robustness_results: Dict[int, Dict[float, float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot robustness comparison across models.
    
    Args:
        robustness_results: Dict mapping model_id -> {noise_level: accuracy}
        save_path: Path to save figure
        show: Whether to display figure
    """
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robustness_results)))
    
    for i, (model_id, results) in enumerate(robustness_results.items()):
        noise_levels = sorted(results.keys())
        accuracies = [results[noise] for noise in noise_levels]
        
        plt.plot(noise_levels, accuracies, 
                label=f'Model {model_id}',
                marker='o', linewidth=2, markersize=6,
                color=colors[i])
    
    plt.xlabel('Noise Level (σ)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Robustness to Parameter Noise', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_training_dashboard(
    history: Dict[str, List[float]],
    num_models: int,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive training dashboard with multiple subplots.
    
    Args:
        history: Training history
        num_models: Number of models
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(history['val_loss'], label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train', linewidth=2)
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(history['val_acc'], label='Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Per-model accuracy
    ax3 = fig.add_subplot(gs[1, :])
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    for i in range(num_models):
        key = f'val_acc_model_{i}'
        if key in history and history[key]:
            ax3.plot(history[key], label=f'Model {i}', 
                    linewidth=2, color=colors[i], marker='o', markersize=3)
    if 'val_acc' in history and history['val_acc']:
        ax3.plot(history['val_acc'], label='Average', 
                linewidth=3, color='black', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title('Per-Model Performance', fontweight='bold')
    ax3.legend(ncol=num_models+1, loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Final accuracies bar chart
    ax4 = fig.add_subplot(gs[2, 0])
    final_accs = []
    for i in range(num_models):
        key = f'val_acc_model_{i}'
        if key in history and history[key]:
            final_accs.append(history[key][-1])
    
    if final_accs:
        x = np.arange(len(final_accs))
        bars = ax4.bar(x, final_accs, color=colors[:len(final_accs)], alpha=0.7, edgecolor='black')
        ax4.axhline(np.mean(final_accs), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(final_accs):.2f}%')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Final Val Accuracy (%)')
        ax4.set_title('Final Model Performance', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'M{i}' for i in range(len(final_accs))])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Learning progress
    ax5 = fig.add_subplot(gs[2, 1])
    if 'train_acc' in history and len(history['train_acc']) > 0:
        epochs = np.arange(len(history['train_acc']))
        improvement = np.array(history['train_acc']) - history['train_acc'][0]
        ax5.fill_between(epochs, 0, improvement, alpha=0.5, color='green')
        ax5.plot(epochs, improvement, linewidth=2, color='darkgreen')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Accuracy Improvement (%)')
        ax5.set_title('Learning Progress', fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    plt.show()
