"""
Feature-Based Deep Mutual Learning.

Extends DML by matching intermediate layer features in addition to output logits.
This provides richer knowledge transfer through intermediate representations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pydml.core.base_trainer import BaseCollaborativeTrainer
from pydml.core.losses import CrossEntropyLoss, KLDivergenceLoss


@dataclass
class FeatureDMLConfig:
    """
    Configuration for Feature-Based DML.
    
    Args:
        temperature: Temperature for KL divergence
        supervised_weight: Weight for supervised loss
        logit_mimicry_weight: Weight for output logit matching
        feature_mimicry_weight: Weight for feature matching
        feature_loss_type: Type of feature loss ('mse', 'cosine', 'l1')
    """
    temperature: float = 3.0
    supervised_weight: float = 1.0
    logit_mimicry_weight: float = 1.0
    feature_mimicry_weight: float = 0.5
    feature_loss_type: str = 'mse'


class FeatureExtractor(nn.Module):
    """
    Wrapper to extract intermediate features from a model.
    
    Args:
        model: The model to extract features from
        layer_names: Names or indices of layers to extract features from
    """
    
    def __init__(self, model: nn.Module, layer_names: Optional[List] = None):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        
        if layer_names is not None:
            self._register_hooks()
    
    def _get_hook(self, name):
        """Create a hook function that stores the feature map."""
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    def _register_hooks(self):
        """Register forward hooks to extract features."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)
    
    def forward(self, x):
        """Forward pass that returns both output and intermediate features."""
        self.features = {}
        output = self.model(x)
        return output, self.features
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class FeatureDMLTrainer(BaseCollaborativeTrainer):
    """
    Feature-Based Deep Mutual Learning Trainer.
    
    Extends standard DML by matching intermediate layer features in addition
    to output logits. This encourages models to learn similar internal
    representations, potentially improving knowledge transfer.
    
    Loss = L_supervised + L_logit_mimicry + L_feature_mimicry
    
    Where:
        L_supervised: Standard cross-entropy with ground truth
        L_logit_mimicry: KL divergence between output distributions
        L_feature_mimicry: MSE/Cosine similarity between intermediate features
    
    Example:
        >>> models = [resnet32(), mobilenet_v2()]
        >>> feature_layers = ['layer2', 'layer3']  # Extract from these layers
        >>> trainer = FeatureDMLTrainer(
        ...     models=models,
        ...     feature_layers_list=[feature_layers, feature_layers],
        ...     config=FeatureDMLConfig(feature_mimicry_weight=0.5)
        ... )
        >>> trainer.fit(train_loader, epochs=100)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        feature_layers_list: Optional[List[List[str]]] = None,
        config: Optional[FeatureDMLConfig] = None,
        device: str = 'cuda',
        optimizers: Optional[List] = None,
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
        
        # Handle dict or config object
        if isinstance(config, dict):
            self.config = FeatureDMLConfig(**config)
        else:
            self.config = config or FeatureDMLConfig()
        
        # Setup feature extractors
        self.feature_layers_list = feature_layers_list
        self.feature_extractors = []
        
        if feature_layers_list is not None:
            for i, (model, layer_names) in enumerate(zip(self.models, feature_layers_list)):
                extractor = FeatureExtractor(model, layer_names)
                self.feature_extractors.append(extractor)
        else:
            # No feature extraction, fall back to logit-only matching
            self.feature_extractors = None
        
        # Loss functions
        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = KLDivergenceLoss(temperature=self.config.temperature)
        
        # Feature loss
        if self.config.feature_loss_type == 'mse':
            self.feature_loss_fn = nn.MSELoss()
        elif self.config.feature_loss_type == 'l1':
            self.feature_loss_fn = nn.L1Loss()
        elif self.config.feature_loss_type == 'cosine':
            self.feature_loss_fn = self._cosine_similarity_loss
        else:
            raise ValueError(f"Unknown feature loss type: {self.config.feature_loss_type}")
        
        print(f"Feature-Based DML Trainer initialized with {self.num_models} models")
        print(f"Config: temp={self.config.temperature}, "
              f"feature_weight={self.config.feature_mimicry_weight}, "
              f"feature_loss={self.config.feature_loss_type}")
    
    def _cosine_similarity_loss(self, features1, features2):
        """Compute cosine similarity loss (1 - cosine_similarity)."""
        features1_flat = features1.view(features1.size(0), -1)
        features2_flat = features2.view(features2.size(0), -1)
        
        # Normalize
        features1_norm = nn.functional.normalize(features1_flat, p=2, dim=1)
        features2_norm = nn.functional.normalize(features2_flat, p=2, dim=1)
        
        # Cosine similarity
        cosine_sim = (features1_norm * features2_norm).sum(dim=1).mean()
        
        # Return loss (1 - similarity)
        return 1 - cosine_sim
    
    def _match_feature_dimensions(self, features1, features2):
        """
        Match feature dimensions if they differ.
        
        Uses adaptive pooling to match spatial dimensions and
        1x1 convolution to match channel dimensions if needed.
        """
        if features1.shape == features2.shape:
            return features1, features2
        
        # Match spatial dimensions
        if len(features1.shape) == 4:  # Conv features (B, C, H, W)
            target_size = min(features1.size(2), features2.size(2))
            if features1.size(2) != target_size or features1.size(3) != target_size:
                features1 = nn.functional.adaptive_avg_pool2d(features1, (target_size, target_size))
            if features2.size(2) != target_size or features2.size(3) != target_size:
                features2 = nn.functional.adaptive_avg_pool2d(features2, (target_size, target_size))
        
        # For channel mismatch, flatten or average
        if features1.size(1) != features2.size(1):
            # Simple approach: global average pooling to match dimensions
            features1 = features1.view(features1.size(0), -1)
            features2 = features2.view(features2.size(0), -1)
            
            # Pad or truncate to match
            min_dim = min(features1.size(1), features2.size(1))
            features1 = features1[:, :min_dim]
            features2 = features2[:, :min_dim]
        
        return features1, features2
    
    def compute_feature_matching_loss(
        self,
        features_dict_i: Dict[str, torch.Tensor],
        features_dict_j: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute feature matching loss between two models' intermediate features.
        
        Args:
            features_dict_i: Dictionary of feature maps from model i
            features_dict_j: Dictionary of feature maps from model j
        
        Returns:
            Feature matching loss
        """
        total_loss = 0.0
        num_layers = 0
        
        # Match features from common layers
        common_layers = set(features_dict_i.keys()) & set(features_dict_j.keys())
        
        for layer_name in common_layers:
            feat_i = features_dict_i[layer_name]
            feat_j = features_dict_j[layer_name]
            
            # Match dimensions if needed
            feat_i, feat_j = self._match_feature_dimensions(feat_i, feat_j)
            
            # Compute loss
            loss = self.feature_loss_fn(feat_i, feat_j)
            total_loss += loss
            num_layers += 1
        
        if num_layers > 0:
            total_loss = total_loss / num_layers
        
        return total_loss
    
    def compute_collaborative_loss(
        self,
        outputs_and_features: List[Tuple[torch.Tensor, Dict]],
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute collaborative loss with both logit and feature matching.
        
        Args:
            outputs_and_features: List of (output, features_dict) tuples
            targets: Ground truth labels
        
        Returns:
            Dictionary of losses for each model
        """
        losses = {}
        
        # Extract outputs and features
        outputs = [item[0] for item in outputs_and_features]
        features_list = [item[1] for item in outputs_and_features]
        
        for i in range(self.num_models):
            # 1. Supervised loss
            supervised_loss = self.ce_loss(outputs[i], targets)
            
            # 2. Logit mimicry loss (KL divergence with peers)
            logit_mimicry_loss = 0.0
            for j in range(self.num_models):
                if i != j:
                    kl_loss = self.kl_loss(outputs[i], outputs[j])
                    logit_mimicry_loss += kl_loss
            
            if self.num_models > 1:
                logit_mimicry_loss = logit_mimicry_loss / (self.num_models - 1)
            
            # 3. Feature mimicry loss
            feature_mimicry_loss = 0.0
            if self.feature_extractors is not None and features_list[0]:
                for j in range(self.num_models):
                    if i != j:
                        feat_loss = self.compute_feature_matching_loss(
                            features_list[i], features_list[j]
                        )
                        feature_mimicry_loss += feat_loss
                
                if self.num_models > 1:
                    feature_mimicry_loss = feature_mimicry_loss / (self.num_models - 1)
            
            # Combine losses
            total_loss = (
                self.config.supervised_weight * supervised_loss +
                self.config.logit_mimicry_weight * logit_mimicry_loss +
                self.config.feature_mimicry_weight * feature_mimicry_loss
            )
            
            losses[f'model_{i}'] = total_loss
        
        return losses
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with feature matching."""
        for model in self.models:
            model.train()
        
        epoch_losses = {f'model_{i}': 0.0 for i in range(self.num_models)}
        total_correct = {f'model_{i}': 0 for i in range(self.num_models)}
        total_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass with feature extraction
            if self.feature_extractors is not None:
                outputs_and_features = []
                for extractor in self.feature_extractors:
                    output, features = extractor(inputs)
                    outputs_and_features.append((output, features))
            else:
                # No feature extraction
                outputs_and_features = [(model(inputs), {}) for model in self.models]
            
            # Compute losses
            losses = self.compute_collaborative_loss(outputs_and_features, targets)
            
            # Backward and optimize each model
            # Zero all gradients first
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            
            # Compute all gradients
            for i in range(self.num_models):
                losses[f'model_{i}'].backward(retain_graph=True)
            
            # Step all optimizers
            for optimizer in self.optimizers:
                optimizer.step()
            
            # Track metrics
            for i in range(self.num_models):
                epoch_losses[f'model_{i}'] += losses[f'model_{i}'].item()
                
                output = outputs_and_features[i][0]
                _, predicted = output.max(1)
                total_correct[f'model_{i}'] += predicted.eq(targets).sum().item()
            
            total_samples += targets.size(0)
        
        # Compute average metrics
        metrics = {
            'train_loss': sum(epoch_losses.values()) / (self.num_models * len(train_loader)),
            'train_acc': sum(total_correct.values()) / (self.num_models * total_samples) * 100,
        }
        
        for i in range(self.num_models):
            metrics[f'train_loss_model_{i}'] = epoch_losses[f'model_{i}'] / len(train_loader)
            metrics[f'train_acc_model_{i}'] = total_correct[f'model_{i}'] / total_samples * 100
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate models on validation set."""
        for model in self.models:
            model.eval()
        
        total_losses = [0.0] * self.num_models
        correct = [0] * self.num_models
        total = 0
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass (no feature extraction needed for eval)
            outputs = []
            for model in self.models:
                output = model(inputs)
                outputs.append(output)
            
            # Simple loss computation (no features)
            for i in range(self.num_models):
                loss = self.ce_loss(outputs[i], targets)
                total_losses[i] += loss.item()
                
                _, predicted = outputs[i].max(1)
                correct[i] += predicted.eq(targets).sum().item()
            
            total += targets.size(0)
        
        # Compute metrics
        metrics = {
            'val_loss': sum(total_losses) / (self.num_models * len(val_loader)),
            'val_acc': sum(correct) / (self.num_models * total) * 100,
        }
        
        for i in range(self.num_models):
            metrics[f'val_loss_model_{i}'] = total_losses[i] / len(val_loader)
            metrics[f'val_acc_model_{i}'] = correct[i] / total * 100
        
        return metrics
    
    def __del__(self):
        """Cleanup: remove feature extraction hooks."""
        if hasattr(self, 'feature_extractors') and self.feature_extractors is not None:
            for extractor in self.feature_extractors:
                extractor.remove_hooks()
