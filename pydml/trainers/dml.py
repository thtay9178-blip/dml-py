"""
Deep Mutual Learning (DML) Trainer.

Implementation of the DML algorithm from the paper:
"Deep Mutual Learning" (CVPR 2018)

Reference: https://arxiv.org/abs/1706.00384
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import torch.nn as nn

from pydml.core.base_trainer import BaseCollaborativeTrainer
from pydml.core.losses import CrossEntropyLoss, KLDivergenceLoss


@dataclass
class DMLConfig:
    """
    Configuration for Deep Mutual Learning.
    
    Args:
        temperature: Temperature for softening probability distributions (default: 3.0)
        supervised_weight: Weight for supervised (cross-entropy) loss (default: 1.0)
        mimicry_weight: Weight for mimicry (KL divergence) losses (default: 1.0)
        peer_selection: Strategy for selecting peers ('all', 'best', 'random', 'dynamic')
    """
    temperature: float = 3.0
    supervised_weight: float = 1.0
    mimicry_weight: float = 1.0
    peer_selection: str = 'all'


class DMLTrainer(BaseCollaborativeTrainer):
    """
    Deep Mutual Learning Trainer.
    
    Implements Algorithm 1 from the DML paper. Trains multiple neural networks
    collaboratively by having them learn from each other's predictions.
    
    Key idea: Each network learns from both:
    1. The ground truth labels (supervised loss)
    2. The predictions of its peer networks (mimicry loss via KL divergence)
    
    Example:
        >>> from pydml import DMLTrainer
        >>> from torchvision import models
        >>> 
        >>> # Create multiple models
        >>> models = [models.resnet18(num_classes=100), models.resnet18(num_classes=100)]
        >>> 
        >>> # Train collaboratively
        >>> trainer = DMLTrainer(models, device='cuda')
        >>> trainer.fit(train_loader, val_loader, epochs=100)
    
    Args:
        models: List of PyTorch models to train collaboratively
        config: DMLConfig instance with hyperparameters
        device: Device to train on ('cuda' or 'cpu')
        optimizers: Optional list of optimizers (one per model)
        schedulers: Optional list of learning rate schedulers
        callbacks: Optional list of callbacks for training hooks
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[DMLConfig] = None,
        device: str = 'cuda',
        optimizers: Optional[List[torch.optim.Optimizer]] = None,
        schedulers: Optional[List] = None,
        callbacks: Optional[List] = None,
    ):
        super().__init__(
            models=models,
            device=device,
            optimizers=optimizers,
            schedulers=schedulers,
            callbacks=callbacks,
        )
        
        # Handle dict or DMLConfig object
        if isinstance(config, dict):
            self.config = DMLConfig(**config)
        else:
            self.config = config or DMLConfig()
        
        # Initialize loss functions
        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = KLDivergenceLoss(temperature=self.config.temperature)
        
        print(f"DML Trainer initialized with {self.num_models} models")
        print(f"Config: temperature={self.config.temperature}, "
              f"supervised_weight={self.config.supervised_weight}, "
              f"mimicry_weight={self.config.mimicry_weight}")
    
    def compute_collaborative_loss(
        self,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DML loss for all models.
        
        For each model k:
            L_k = L_CE(y_k, y_true) + (1/K-1) * Σ_{j≠k} L_KL(y_k, y_j)
        
        Where:
            - L_CE is the supervised cross-entropy loss
            - L_KL is the KL divergence mimicry loss
            - K is the number of models
            - y_k is the prediction of model k
            - y_true is the ground truth label
        
        Args:
            outputs: List of model outputs (logits), one per model
            targets: Ground truth labels
        
        Returns:
            Dictionary mapping 'model_{i}' to its total loss
        """
        losses = {}
        
        for i in range(self.num_models):
            # Supervised loss: Learn from ground truth
            supervised_loss = self.ce_loss(outputs[i], targets)
            
            # Mimicry loss: Learn from peer predictions
            mimicry_loss = 0.0
            peer_count = 0
            
            for j in range(self.num_models):
                if i != j:  # Don't learn from yourself
                    # KL divergence between model i and peer j
                    # Use .detach() on peer to prevent gradient flow
                    kl_div = self.kl_loss(outputs[i], outputs[j].detach())
                    mimicry_loss += kl_div
                    peer_count += 1
            
            # Average mimicry loss over all peers
            if peer_count > 0:
                mimicry_loss /= peer_count
            
            # Total loss with configurable weights
            total_loss = (
                self.config.supervised_weight * supervised_loss +
                self.config.mimicry_weight * mimicry_loss
            )
            
            losses[f'model_{i}'] = total_loss
        
        return losses
    
    def get_ensemble_predictions(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble predictions by averaging all model outputs.
        
        Args:
            inputs: Input batch
        
        Returns:
            Ensemble predictions (averaged logits)
        """
        self.eval()
        with torch.no_grad():
            outputs = []
            for model in self.models:
                output = model(inputs)
                outputs.append(output)
            
            # Average predictions
            ensemble_output = torch.stack(outputs).mean(dim=0)
        
        return ensemble_output
    
    def eval(self):
        """Set all models to evaluation mode."""
        for model in self.models:
            model.eval()
    
    def train(self):
        """Set all models to training mode."""
        for model in self.models:
            model.train()


# Alias for backward compatibility
DeepMutualLearningTrainer = DMLTrainer
