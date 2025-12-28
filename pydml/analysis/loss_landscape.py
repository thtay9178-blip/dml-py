"""
Loss Landscape Visualization for DML-PY.

Provides tools to visualize the loss landscape of trained models, helping
understand optimization trajectories and model convergence properties.

Features:
- 1D loss curves along random directions
- 2D loss surface plots
- 3D interactive visualizations
- Mode connectivity analysis
- Trajectory plotting during training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
import matplotlib.pyplot as plt
from pathlib import Path


class LossLandscape:
    """
    Visualizer for neural network loss landscapes.
    
    Args:
        model: PyTorch model to analyze
        criterion: Loss function
        dataloader: DataLoader for loss computation
        device: Device for computation
        
    Example:
        >>> landscape = LossLandscape(model, criterion, val_loader)
        >>> landscape.plot_1d('loss_curve.png')
        >>> landscape.plot_2d('loss_surface.png')
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        dataloader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        
        # Store original parameters
        self.original_params = [p.clone() for p in model.parameters()]
    
    def compute_loss(self) -> float:
        """Compute loss on full dataset."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        return total_loss / total_samples
    
    def perturb_parameters(self, direction: List[torch.Tensor], alpha: float):
        """
        Perturb model parameters along a direction.
        
        Args:
            direction: List of perturbation tensors
            alpha: Step size
        """
        for param, orig, d in zip(self.model.parameters(), self.original_params, direction):
            param.data = orig + alpha * d
    
    def restore_parameters(self):
        """Restore model to original parameters."""
        for param, orig in zip(self.model.parameters(), self.original_params):
            param.data = orig.clone()
    
    def random_direction(self, normalize: bool = True) -> List[torch.Tensor]:
        """
        Generate random direction in parameter space.
        
        Args:
            normalize: Whether to normalize direction
            
        Returns:
            List of random perturbation tensors
        """
        direction = [torch.randn_like(p) for p in self.model.parameters()]
        
        if normalize:
            # Normalize to unit norm
            norm = torch.sqrt(sum((d ** 2).sum() for d in direction))
            direction = [d / norm for d in direction]
        
        return direction
    
    def plot_1d(
        self,
        output_path: str,
        direction: Optional[List[torch.Tensor]] = None,
        alpha_min: float = -1.0,
        alpha_max: float = 1.0,
        num_points: int = 51,
        title: str = "1D Loss Landscape"
    ):
        """
        Plot 1D loss curve along a direction.
        
        Args:
            output_path: Path to save plot
            direction: Direction to plot (random if None)
            alpha_min: Minimum perturbation value
            alpha_max: Maximum perturbation value
            num_points: Number of points to sample
            title: Plot title
        """
        if direction is None:
            direction = self.random_direction()
        
        alphas = np.linspace(alpha_min, alpha_max, num_points)
        losses = []
        
        print("Computing 1D loss landscape...")
        for alpha in alphas:
            self.perturb_parameters(direction, alpha)
            loss = self.compute_loss()
            losses.append(loss)
            self.restore_parameters()
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, losses, 'b-', linewidth=2)
        plt.axvline(x=0, color='r', linestyle='--', label='Original model')
        plt.xlabel('Step size (α)', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"✓ 1D loss landscape saved to {output_path}")
    
    def plot_2d(
        self,
        output_path: str,
        direction1: Optional[List[torch.Tensor]] = None,
        direction2: Optional[List[torch.Tensor]] = None,
        alpha_min: float = -1.0,
        alpha_max: float = 1.0,
        beta_min: float = -1.0,
        beta_max: float = 1.0,
        num_points: int = 25,
        title: str = "2D Loss Landscape",
        plot_3d: bool = False
    ):
        """
        Plot 2D loss surface.
        
        Args:
            output_path: Path to save plot
            direction1: First direction (random if None)
            direction2: Second direction (random if None)
            alpha_min: Minimum value for first direction
            alpha_max: Maximum value for first direction
            beta_min: Minimum value for second direction
            beta_max: Maximum value for second direction
            num_points: Number of points per axis
            title: Plot title
            plot_3d: Whether to create 3D plot
        """
        if direction1 is None:
            direction1 = self.random_direction()
        if direction2 is None:
            direction2 = self.random_direction()
        
        alphas = np.linspace(alpha_min, alpha_max, num_points)
        betas = np.linspace(beta_min, beta_max, num_points)
        
        losses = np.zeros((num_points, num_points))
        
        print(f"Computing 2D loss landscape ({num_points}x{num_points} grid)...")
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # Combine directions
                combined_dir = [d1 * alpha + d2 * beta 
                               for d1, d2 in zip(direction1, direction2)]
                
                self.perturb_parameters(combined_dir, 1.0)
                losses[i, j] = self.compute_loss()
                self.restore_parameters()
            
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{num_points}")
        
        # Plot
        if plot_3d:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            X, Y = np.meshgrid(betas, alphas)
            surf = ax.plot_surface(X, Y, losses, cmap='viridis', alpha=0.8)
            
            ax.set_xlabel('Direction 2 (β)', fontsize=11)
            ax.set_ylabel('Direction 1 (α)', fontsize=11)
            ax.set_zlabel('Loss', fontsize=11)
            ax.set_title(title, fontsize=14)
            fig.colorbar(surf, shrink=0.5, aspect=5)
        else:
            plt.figure(figsize=(10, 8))
            plt.contourf(betas, alphas, losses, levels=20, cmap='viridis')
            plt.colorbar(label='Loss')
            plt.scatter([0], [0], color='red', s=100, marker='*', 
                       label='Original model', zorder=5)
            plt.xlabel('Direction 2 (β)', fontsize=12)
            plt.ylabel('Direction 1 (α)', fontsize=12)
            plt.title(title, fontsize=14)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"✓ 2D loss landscape saved to {output_path}")
    
    def plot_trajectory(
        self,
        checkpoints: List[Dict],
        output_path: str,
        direction1: Optional[List[torch.Tensor]] = None,
        direction2: Optional[List[torch.Tensor]] = None,
        title: str = "Training Trajectory"
    ):
        """
        Plot training trajectory on loss landscape.
        
        Args:
            checkpoints: List of model state dicts from training
            output_path: Path to save plot
            direction1: First projection direction
            direction2: Second projection direction
            title: Plot title
        """
        if direction1 is None:
            direction1 = self.random_direction()
        if direction2 is None:
            direction2 = self.random_direction()
        
        # Project checkpoints onto 2D plane
        projections = []
        
        for checkpoint in checkpoints:
            self.model.load_state_dict(checkpoint)
            
            # Compute projection
            delta = [p - orig for p, orig in zip(self.model.parameters(), self.original_params)]
            alpha = sum((d * d1).sum() for d, d1 in zip(delta, direction1)).item()
            beta = sum((d * d2).sum() for d, d2 in zip(delta, direction2)).item()
            
            projections.append((alpha, beta))
        
        self.restore_parameters()
        
        # Plot trajectory
        projections = np.array(projections)
        
        plt.figure(figsize=(10, 8))
        plt.plot(projections[:, 1], projections[:, 0], 'b-', alpha=0.5, linewidth=2)
        plt.scatter(projections[:, 1], projections[:, 0], c=range(len(projections)), 
                   cmap='viridis', s=50, zorder=5)
        plt.scatter([0], [0], color='red', s=200, marker='*', 
                   label='Start', zorder=10)
        plt.scatter([projections[-1, 1]], [projections[-1, 0]], 
                   color='green', s=200, marker='*', label='End', zorder=10)
        
        plt.xlabel('Direction 2 (β)', fontsize=12)
        plt.ylabel('Direction 1 (α)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.colorbar(label='Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"✓ Training trajectory saved to {output_path}")


def quick_landscape_analysis(
    model: nn.Module,
    criterion: Callable,
    dataloader,
    output_dir: str = 'landscape_plots',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Quick loss landscape analysis with standard plots.
    
    Args:
        model: PyTorch model
        criterion: Loss function
        dataloader: DataLoader for evaluation
        output_dir: Directory to save plots
        device: Device for computation
        
    Example:
        >>> quick_landscape_analysis(model, nn.CrossEntropyLoss(), val_loader)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    landscape = LossLandscape(model, criterion, dataloader, device)
    
    # Generate random directions
    dir1 = landscape.random_direction()
    dir2 = landscape.random_direction()
    
    # Create plots
    landscape.plot_1d(output_dir / '1d_landscape.png', direction=dir1)
    landscape.plot_2d(output_dir / '2d_landscape_contour.png', 
                     direction1=dir1, direction2=dir2, num_points=20)
    landscape.plot_2d(output_dir / '2d_landscape_3d.png',
                     direction1=dir1, direction2=dir2, num_points=20, plot_3d=True)
    
    print(f"\n✓ Loss landscape analysis complete! Plots saved to {output_dir}")


__all__ = [
    'LossLandscape',
    'quick_landscape_analysis'
]
