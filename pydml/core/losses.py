"""
Loss functions for collaborative learning.

This module provides base classes and common loss functions used in collaborative training.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(ABC, nn.Module):
    """
    Abstract base class for loss functions.
    
    All loss functions should inherit from this class and implement the forward method.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute the loss."""
        pass


class CrossEntropyLoss(BaseLoss):
    """
    Standard cross-entropy loss for classification.
    
    Args:
        reduction: Specifies the reduction to apply ('none', 'mean', 'sum')
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            logits: Model predictions (logits), shape (batch_size, num_classes)
            targets: Ground truth labels, shape (batch_size,)
        
        Returns:
            Cross-entropy loss
        """
        return self.ce_loss(logits, targets)


class KLDivergenceLoss(BaseLoss):
    """
    KL Divergence loss for knowledge distillation.
    
    This loss measures how different two probability distributions are.
    Used in Deep Mutual Learning for peer learning.
    
    Args:
        temperature: Temperature for softening probability distributions
        reduction: Specifies the reduction to apply ('none', 'batchmean', 'sum', 'mean')
    """
    
    def __init__(self, temperature: float = 1.0, reduction: str = 'batchmean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between student and teacher predictions.
        
        Args:
            student_logits: Student model predictions (logits), shape (batch_size, num_classes)
            teacher_logits: Teacher model predictions (logits), shape (batch_size, num_classes)
        
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Compute KL divergence
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction=self.reduction
        )
        
        # Scale by temperature^2 as per distillation literature
        return kl_loss * (self.temperature ** 2)


class DMLLoss(BaseLoss):
    """
    Deep Mutual Learning loss combining supervised and mimicry losses.
    
    This is the complete loss function used in the DML paper:
    L = L_CE + λ * Σ L_KL
    
    Args:
        temperature: Temperature for KL divergence
        supervised_weight: Weight for supervised (cross-entropy) loss
        mimicry_weight: Weight for mimicry (KL divergence) losses
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        supervised_weight: float = 1.0,
        mimicry_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.supervised_weight = supervised_weight
        self.mimicry_weight = mimicry_weight
        
        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = KLDivergenceLoss(temperature=temperature)
    
    def forward(
        self,
        logits: torch.Tensor,
        peer_logits_list: list,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DML loss for one model.
        
        Args:
            logits: Current model's predictions (logits)
            peer_logits_list: List of peer models' predictions (logits)
            targets: Ground truth labels
        
        Returns:
            Total DML loss
        """
        # Supervised loss
        supervised_loss = self.ce_loss(logits, targets)
        
        # Mimicry loss (average KL divergence from all peers)
        mimicry_loss = 0.0
        for peer_logits in peer_logits_list:
            mimicry_loss += self.kl_loss(logits, peer_logits.detach())
        
        if len(peer_logits_list) > 0:
            mimicry_loss /= len(peer_logits_list)
        
        # Total loss
        total_loss = (
            self.supervised_weight * supervised_loss +
            self.mimicry_weight * mimicry_loss
        )
        
        return total_loss


class LossRegistry:
    """
    Registry for managing loss functions.
    
    Allows easy registration and retrieval of custom loss functions.
    """
    
    _registry = {
        'cross_entropy': CrossEntropyLoss,
        'kl_divergence': KLDivergenceLoss,
        'dml': DMLLoss,
    }
    
    @classmethod
    def register(cls, name: str, loss_class: type):
        """Register a new loss function."""
        cls._registry[name] = loss_class
    
    @classmethod
    def get(cls, name: str, **kwargs):
        """Get a loss function by name."""
        if name not in cls._registry:
            raise ValueError(f"Loss '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_available(cls):
        """List all available loss functions."""
        return list(cls._registry.keys())
