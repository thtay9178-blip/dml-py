"""
Confidence-Weighted Deep Mutual Learning.

A novel extension of DML where peer learning weights are adaptively adjusted
based on prediction confidence, allowing models to learn more from confident
peers and less from uncertain ones.

Key Innovation:
- Traditional DML treats all peer predictions equally
- Confidence-Weighted DML adjusts learning based on prediction confidence
- Models learn more from peers that are confident and correct
- Reduces negative transfer from uncertain predictions

Research Contribution:
This addresses a key limitation in standard DML where models can learn
incorrect information from uncertain peer predictions, especially early
in training or on difficult examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from dataclasses import dataclass

from pydml.core import BaseCollaborativeTrainer


@dataclass
class ConfidenceWeightedConfig:
    """
    Configuration for Confidence-Weighted DML.
    
    Args:
        confidence_threshold: Minimum confidence to consider peer prediction (0-1)
        weighting_strategy: Strategy for computing weights ('softmax', 'linear', 'exp')
        temperature: Temperature for softmax weighting
        min_weight: Minimum weight for any peer (prevents complete ignoring)
        max_weight: Maximum weight for any peer (prevents over-reliance)
        adaptive: Whether to adapt thresholds during training
        
    Example:
        >>> config = ConfidenceWeightedConfig(
        ...     confidence_threshold=0.5,
        ...     weighting_strategy='softmax',
        ...     temperature=2.0
        ... )
    """
    confidence_threshold: float = 0.5
    weighting_strategy: str = 'softmax'
    temperature: float = 2.0
    min_weight: float = 0.1
    max_weight: float = 2.0
    adaptive: bool = True


class ConfidenceWeightedDML(BaseCollaborativeTrainer):
    """
    Confidence-Weighted Deep Mutual Learning Trainer.
    
    Extends standard DML by weighting peer contributions based on
    prediction confidence, allowing more effective knowledge transfer.
    
    Args:
        models: List of neural network models
        config: Confidence weighting configuration
        learning_rate: Learning rate for optimization
        device: Device for training ('cuda' or 'cpu')
        
    Example:
        >>> models = [resnet32(10), wrn_28_10(10)]
        >>> config = ConfidenceWeightedConfig(confidence_threshold=0.6)
        >>> trainer = ConfidenceWeightedDML(models, config)
        >>> results = trainer.fit(train_loader, val_loader, epochs=50)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[ConfidenceWeightedConfig] = None,
        learning_rate: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        # Create optimizers manually before calling super().__init__
        import torch.optim as optim
        optimizers = [optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) 
                     for model in models]
        
        super().__init__(models, optimizers=optimizers, device=device)
        self.config = config or ConfidenceWeightedConfig()
        
        # Track confidence statistics
        self.confidence_history = {i: [] for i in range(len(models))}
        self.weight_history = {i: [] for i in range(len(models))}
        
    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction confidence from logits.
        
        Args:
            logits: Model output logits [batch_size, num_classes]
            
        Returns:
            Confidence scores [batch_size]
        """
        probs = F.softmax(logits, dim=1)
        confidence, _ = probs.max(dim=1)
        return confidence
    
    def compute_peer_weights(
        self,
        confidences: List[torch.Tensor],
        correctness: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Compute weights for each peer based on confidence.
        
        Args:
            confidences: List of confidence tensors for each model
            correctness: Optional list of correctness indicators
            
        Returns:
            List of weight tensors for each peer
        """
        weights = []
        
        for i, conf in enumerate(confidences):
            if self.config.weighting_strategy == 'softmax':
                # Softmax over confidence scores
                weight = F.softmax(conf / self.config.temperature, dim=0)
                
            elif self.config.weighting_strategy == 'linear':
                # Linear scaling of confidence
                weight = conf.clone()
                
            elif self.config.weighting_strategy == 'exp':
                # Exponential scaling
                weight = torch.exp(conf * self.config.temperature)
            
            else:
                raise ValueError(f"Unknown weighting strategy: {self.config.weighting_strategy}")
            
            # Apply threshold (create new tensor, don't use in-place)
            weight = torch.where(
                conf > self.config.confidence_threshold,
                weight,
                torch.zeros_like(weight)
            )
            
            # Incorporate correctness if available
            if correctness is not None:
                weight = weight * correctness[i].float()
            
            # Clamp weights
            weight = weight.clamp(self.config.min_weight, self.config.max_weight)
            
            weights.append(weight.detach())  # Detach to avoid gradient issues
        
        return weights
    
    def compute_collaborative_loss(
        self,
        outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence-weighted mutual learning loss.
        
        Args:
            outputs: List of model outputs
            targets: Ground truth labels
            
        Returns:
            Dictionary mapping 'model_{i}' to its total loss
        """
        num_models = len(outputs)
        
        # Compute confidences for all models
        confidences = [self.compute_confidence(out) for out in outputs]
        
        # Compute correctness (if targets available)
        predictions = [out.argmax(dim=1) for out in outputs]
        correctness = [(pred == targets).float() for pred in predictions]
        
        # Compute peer weights
        weights = self.compute_peer_weights(confidences, correctness)
        
        # Store statistics
        for i, (conf, weight) in enumerate(zip(confidences, weights)):
            self.confidence_history[i].append(conf.mean().item())
            self.weight_history[i].append(weight.mean().item())
        
        # Compute weighted KL divergence for each model
        losses = {}
        
        for i in range(num_models):
            # Supervised cross-entropy loss
            ce_loss = F.cross_entropy(outputs[i], targets)
            
            # Peer learning loss (weighted KL divergence)
            peer_loss = 0.0
            
            for j in range(num_models):
                if i == j:
                    continue
                
                # KL divergence from model i to model j
                kl_div = F.kl_div(
                    F.log_softmax(outputs[i] / self.config.temperature, dim=1),
                    F.softmax(outputs[j].detach() / self.config.temperature, dim=1),
                    reduction='none'
                ).sum(dim=1)
                
                # Weight by peer confidence (use mean weight across batch)
                weighted_kl = (kl_div * weights[j]).mean()
                peer_loss += weighted_kl
            
            # Average peer loss
            peer_loss = peer_loss / (num_models - 1) if num_models > 1 else 0.0
            
            # Total loss = supervised + peer learning
            losses[f'model_{i}'] = ce_loss + peer_loss
        
        return losses
    
    def adapt_threshold(self, epoch: int, total_epochs: int):
        """
        Adaptively adjust confidence threshold during training.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
        """
        if not self.config.adaptive:
            return
        
        # Start with high threshold, gradually decrease
        # Early training: Only learn from very confident peers
        # Later training: Accept more peer predictions
        progress = epoch / total_epochs
        initial_threshold = 0.8
        final_threshold = 0.4
        
        self.config.confidence_threshold = (
            initial_threshold - (initial_threshold - final_threshold) * progress
        )
    
    def get_confidence_stats(self) -> Dict:
        """
        Get statistics about confidence and weights during training.
        
        Returns:
            Dictionary with confidence and weight statistics
        """
        stats = {
            'confidence_history': self.confidence_history,
            'weight_history': self.weight_history,
            'final_threshold': self.config.confidence_threshold
        }
        
        # Compute averages
        for i in range(len(self.models)):
            stats[f'model_{i}_avg_confidence'] = sum(self.confidence_history[i]) / len(self.confidence_history[i])
            stats[f'model_{i}_avg_weight'] = sum(self.weight_history[i]) / len(self.weight_history[i])
        
        return stats


def compare_standard_vs_confidence_weighted(
    models_standard: List[nn.Module],
    models_weighted: List[nn.Module],
    train_loader,
    val_loader,
    epochs: int = 50
) -> Dict:
    """
    Compare standard DML with confidence-weighted DML.
    
    Args:
        models_standard: Models for standard DML
        models_weighted: Models for confidence-weighted DML
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        
    Returns:
        Comparison results
        
    Example:
        >>> from pydml.trainers import DMLTrainer
        >>> models_std = [resnet20(10), wrn_16_2(10)]
        >>> models_cw = [resnet20(10), wrn_16_2(10)]
        >>> results = compare_standard_vs_confidence_weighted(
        ...     models_std, models_cw, train_loader, val_loader
        ... )
    """
    from pydml.trainers import DMLTrainer
    
    print("Training Standard DML...")
    trainer_standard = DMLTrainer(models_standard)
    results_standard = trainer_standard.fit(train_loader, val_loader, epochs=epochs)
    
    print("\nTraining Confidence-Weighted DML...")
    config = ConfidenceWeightedConfig(
        confidence_threshold=0.6,
        weighting_strategy='softmax',
        adaptive=True
    )
    trainer_weighted = ConfidenceWeightedDML(models_weighted, config)
    results_weighted = trainer_weighted.fit(train_loader, val_loader, epochs=epochs)
    
    # Compare results
    comparison = {
        'standard_final_acc': results_standard['avg_acc'][-1],
        'weighted_final_acc': results_weighted['avg_acc'][-1],
        'improvement': results_weighted['avg_acc'][-1] - results_standard['avg_acc'][-1],
        'standard_history': results_standard,
        'weighted_history': results_weighted,
        'confidence_stats': trainer_weighted.get_confidence_stats()
    }
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Standard DML:          {comparison['standard_final_acc']:.2f}%")
    print(f"Confidence-Weighted:   {comparison['weighted_final_acc']:.2f}%")
    print(f"Improvement:           +{comparison['improvement']:.2f}%")
    print(f"{'='*60}")
    
    return comparison


__all__ = [
    'ConfidenceWeightedConfig',
    'ConfidenceWeightedDML',
    'compare_standard_vs_confidence_weighted'
]
