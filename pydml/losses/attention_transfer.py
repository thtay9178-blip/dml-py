"""
Attention Transfer Loss.

Implementation of attention transfer for knowledge distillation and mutual learning.
Based on "Paying More Attention to Attention" (Zagoruyko & Komodakis, ICLR 2017)

Attention transfer encourages student networks to have similar spatial attention
maps as teacher networks, leading to better knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer Loss.
    
    Computes the L2 distance between attention maps of two networks.
    Attention maps are computed as the sum of squared activations across channels.
    
    For a feature map F of shape (B, C, H, W):
        Attention(F) = sum(F^2, dim=1) -> (B, H, W)
    
    Loss = ||Attention(F_student) - Attention(F_teacher)||_2
    
    Args:
        normalize: Whether to normalize attention maps
        attention_type: Type of attention ('sum_squares', 'mean_squares', 'mean_abs')
    
    Reference:
        Zagoruyko & Komodakis. "Paying More Attention to Attention:
        Improving the Performance of CNNs via Attention Transfer." ICLR 2017.
    """
    
    def __init__(
        self, 
        normalize: bool = True,
        attention_type: str = 'sum_squares'
    ):
        super().__init__()
        self.normalize = normalize
        self.attention_type = attention_type
    
    def compute_attention_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention map from feature map.
        
        Args:
            feature_map: Feature tensor of shape (B, C, H, W) or (B, C)
        
        Returns:
            Attention map of shape (B, H, W) or (B,)
        """
        if self.attention_type == 'sum_squares':
            # Sum of squared activations across channels
            attention = torch.sum(feature_map ** 2, dim=1)
        elif self.attention_type == 'mean_squares':
            # Mean of squared activations
            attention = torch.mean(feature_map ** 2, dim=1)
        elif self.attention_type == 'mean_abs':
            # Mean of absolute activations
            attention = torch.mean(torch.abs(feature_map), dim=1)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        
        # Normalize if requested
        if self.normalize:
            # Flatten spatial dimensions
            attention_flat = attention.view(attention.size(0), -1)
            # L2 normalize
            attention_flat = F.normalize(attention_flat, p=2, dim=1)
            # Reshape back
            attention = attention_flat.view_as(attention)
        
        return attention
    
    def forward(
        self, 
        student_features: torch.Tensor, 
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention transfer loss.
        
        Args:
            student_features: Student feature map (B, C, H, W)
            teacher_features: Teacher feature map (B, C, H, W)
        
        Returns:
            Attention transfer loss
        """
        # Compute attention maps
        student_attention = self.compute_attention_map(student_features)
        teacher_attention = self.compute_attention_map(teacher_features)
        
        # Match spatial dimensions if needed
        if student_attention.shape != teacher_attention.shape:
            # Use adaptive pooling to match dimensions
            target_size = min(student_attention.size(-1), teacher_attention.size(-1))
            student_attention = F.adaptive_avg_pool2d(
                student_attention.unsqueeze(1), (target_size, target_size)
            ).squeeze(1)
            teacher_attention = F.adaptive_avg_pool2d(
                teacher_attention.unsqueeze(1), (target_size, target_size)
            ).squeeze(1)
        
        # Compute L2 loss
        loss = F.mse_loss(student_attention, teacher_attention)
        
        return loss


class MultiLayerAttentionTransferLoss(nn.Module):
    """
    Multi-layer attention transfer loss.
    
    Computes attention transfer loss across multiple layers and aggregates them.
    
    Args:
        layer_weights: Optional weights for each layer. If None, uses equal weights.
        normalize: Whether to normalize attention maps
        attention_type: Type of attention computation
    """
    
    def __init__(
        self,
        layer_weights: Optional[List[float]] = None,
        normalize: bool = True,
        attention_type: str = 'sum_squares'
    ):
        super().__init__()
        self.layer_weights = layer_weights
        self.at_loss = AttentionTransferLoss(normalize, attention_type)
    
    def forward(
        self,
        student_features_list: List[torch.Tensor],
        teacher_features_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-layer attention transfer loss.
        
        Args:
            student_features_list: List of student feature maps
            teacher_features_list: List of teacher feature maps
        
        Returns:
            Aggregated attention transfer loss
        """
        if len(student_features_list) != len(teacher_features_list):
            raise ValueError("Number of student and teacher feature maps must match")
        
        num_layers = len(student_features_list)
        
        # Use equal weights if not specified
        if self.layer_weights is None:
            weights = [1.0 / num_layers] * num_layers
        else:
            weights = self.layer_weights
            if len(weights) != num_layers:
                raise ValueError("Number of weights must match number of layers")
        
        # Compute weighted sum of attention losses
        total_loss = 0.0
        for student_feat, teacher_feat, weight in zip(
            student_features_list, teacher_features_list, weights
        ):
            layer_loss = self.at_loss(student_feat, teacher_feat)
            total_loss += weight * layer_loss
        
        return total_loss


class AttentionMatchingLoss(nn.Module):
    """
    Attention matching for mutual learning.
    
    Computes pairwise attention matching between multiple models.
    Useful for DML where all models should have similar attention patterns.
    
    Args:
        normalize: Whether to normalize attention maps
        attention_type: Type of attention computation
        aggregation: How to aggregate pairwise losses ('mean', 'sum')
    """
    
    def __init__(
        self,
        normalize: bool = True,
        attention_type: str = 'sum_squares',
        aggregation: str = 'mean'
    ):
        super().__init__()
        self.at_loss = AttentionTransferLoss(normalize, attention_type)
        self.aggregation = aggregation
    
    def forward(
        self,
        features_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute pairwise attention matching loss.
        
        Args:
            features_list: List of feature maps from different models
        
        Returns:
            Attention matching loss
        """
        num_models = len(features_list)
        
        if num_models < 2:
            return torch.tensor(0.0, device=features_list[0].device)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Compute all pairwise losses
        for i in range(num_models):
            for j in range(i + 1, num_models):
                pairwise_loss = self.at_loss(features_list[i], features_list[j])
                total_loss += pairwise_loss
                num_pairs += 1
        
        # Aggregate
        if self.aggregation == 'mean':
            return total_loss / num_pairs if num_pairs > 0 else total_loss
        elif self.aggregation == 'sum':
            return total_loss
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


class SpatialAttentionVisualization:
    """
    Utility class for visualizing attention maps.
    
    Helps debug and understand what spatial regions models are focusing on.
    """
    
    @staticmethod
    def get_attention_map(
        feature_map: torch.Tensor,
        attention_type: str = 'sum_squares',
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract attention map from feature map for visualization.
        
        Args:
            feature_map: Feature tensor (B, C, H, W)
            attention_type: Type of attention computation
            normalize: Whether to normalize
        
        Returns:
            Attention map (B, H, W) ready for visualization
        """
        loss_fn = AttentionTransferLoss(normalize, attention_type)
        attention = loss_fn.compute_attention_map(feature_map)
        return attention
    
    @staticmethod
    def visualize_attention(
        attention_map: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention map using matplotlib.
        
        Args:
            attention_map: Attention tensor (H, W) or (B, H, W)
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        # Handle batch dimension
        if len(attention_map.shape) == 3:
            attention_map = attention_map[0]  # Take first in batch
        
        # Convert to numpy
        attention_np = attention_map.detach().cpu().numpy()
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.imshow(attention_np, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Attention Intensity')
        plt.title('Spatial Attention Map')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Attention map saved to {save_path}")
        
        plt.show()


# Convenience function for common use case
def attention_transfer_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Convenience function to compute attention transfer loss.
    
    Args:
        student_features: Student feature map
        teacher_features: Teacher feature map
        normalize: Whether to normalize attention maps
    
    Returns:
        Attention transfer loss
    """
    loss_fn = AttentionTransferLoss(normalize=normalize)
    return loss_fn(student_features, teacher_features)
