"""
Curriculum Learning Strategy for DML-PY.

Implements curriculum learning where training starts with easier examples
and gradually progresses to harder ones.

Reference: "Curriculum Learning" (Bengio et al., 2009)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Optional
from torch.utils.data import Sampler, Dataset


class CurriculumSampler(Sampler):
    """
    Sampler that orders samples by difficulty.
    
    Args:
        dataset: Dataset to sample from
        difficulty_scores: Difficulty score for each sample (higher = harder)
        start_easy: Whether to start with easy samples
        curriculum_schedule: Function that takes epoch and returns difficulty threshold
    """
    
    def __init__(
        self,
        dataset: Dataset,
        difficulty_scores: np.ndarray,
        start_easy: bool = True,
        curriculum_schedule: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.start_easy = start_easy
        self.curriculum_schedule = curriculum_schedule or self._default_schedule
        
        # Sort indices by difficulty
        self.sorted_indices = np.argsort(difficulty_scores)
        if not start_easy:
            self.sorted_indices = self.sorted_indices[::-1]
        
        self.current_epoch = 0
    
    def _default_schedule(self, epoch: int) -> float:
        """
        Default curriculum schedule.
        
        Returns the fraction of dataset to include based on epoch.
        """
        # Start with 30% of data, reach 100% by epoch 100
        return min(0.3 + (epoch * 0.007), 1.0)
    
    def set_epoch(self, epoch: int):
        """Set the current epoch for curriculum scheduling."""
        self.current_epoch = epoch
    
    def __iter__(self):
        # Determine how much of the dataset to include
        fraction = self.curriculum_schedule(self.current_epoch)
        n_samples = int(len(self.dataset) * fraction)
        
        # Select samples based on curriculum
        selected_indices = self.sorted_indices[:n_samples]
        
        # Shuffle the selected samples
        shuffled_indices = selected_indices[np.random.permutation(len(selected_indices))]
        
        return iter(shuffled_indices.tolist())
    
    def __len__(self):
        fraction = self.curriculum_schedule(self.current_epoch)
        return int(len(self.dataset) * fraction)


class DifficultyEstimator:
    """
    Estimate sample difficulty for curriculum learning.
    
    Different strategies for estimating difficulty:
    1. Prediction confidence
    2. Loss magnitude
    3. Model agreement
    4. Manual labels
    """
    
    @staticmethod
    def by_prediction_confidence(
        model: nn.Module,
        dataset: Dataset,
        device: str = 'cuda'
    ) -> np.ndarray:
        """
        Estimate difficulty by prediction confidence.
        
        Higher confidence = easier sample.
        """
        model.eval()
        difficulties = []
        
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                # Max probability as confidence
                confidences, _ = probs.max(dim=1)
                
                # Difficulty = 1 - confidence
                batch_difficulties = 1 - confidences.cpu().numpy()
                difficulties.extend(batch_difficulties)
        
        return np.array(difficulties)
    
    @staticmethod
    def by_loss_magnitude(
        model: nn.Module,
        dataset: Dataset,
        device: str = 'cuda'
    ) -> np.ndarray:
        """
        Estimate difficulty by loss magnitude.
        
        Higher loss = harder sample.
        """
        model.eval()
        difficulties = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                losses = criterion(outputs, targets)
                difficulties.extend(losses.cpu().numpy())
        
        return np.array(difficulties)
    
    @staticmethod
    def by_ensemble_agreement(
        models: List[nn.Module],
        dataset: Dataset,
        device: str = 'cuda'
    ) -> np.ndarray:
        """
        Estimate difficulty by model agreement.
        
        Low agreement = harder sample (ambiguous).
        """
        for model in models:
            model.eval()
        
        difficulties = []
        
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                
                # Get predictions from all models
                predictions = []
                for model in models:
                    outputs = model(inputs)
                    _, preds = outputs.max(dim=1)
                    predictions.append(preds)
                
                # Compute agreement rate for each sample
                predictions = torch.stack(predictions)  # (n_models, batch_size)
                
                for i in range(inputs.size(0)):
                    sample_preds = predictions[:, i]
                    # Agreement = fraction of models that agree with majority
                    mode_pred = torch.mode(sample_preds).values
                    agreement = (sample_preds == mode_pred).float().mean().item()
                    
                    # Difficulty = 1 - agreement
                    difficulties.append(1 - agreement)
        
        return np.array(difficulties)


class CurriculumStrategy:
    """
    High-level curriculum learning strategy.
    
    Args:
        strategy: Difficulty estimation strategy ('confidence', 'loss', 'agreement')
        start_easy: Whether to start with easy samples
        warmup_epochs: Number of epochs for warmup before full curriculum
    """
    
    def __init__(
        self,
        strategy: str = 'confidence',
        start_easy: bool = True,
        warmup_epochs: int = 10
    ):
        self.strategy = strategy
        self.start_easy = start_easy
        self.warmup_epochs = warmup_epochs
    
    def create_curriculum_loader(
        self,
        dataset: Dataset,
        model_or_models,
        batch_size: int = 128,
        device: str = 'cuda'
    ):
        """
        Create a data loader with curriculum sampling.
        
        Args:
            dataset: Dataset to sample from
            model_or_models: Model or list of models for difficulty estimation
            batch_size: Batch size
            device: Device to use
        
        Returns:
            DataLoader with curriculum sampler
        """
        # Estimate difficulty
        if self.strategy == 'confidence':
            models = [model_or_models] if not isinstance(model_or_models, list) else model_or_models
            difficulties = DifficultyEstimator.by_prediction_confidence(
                models[0], dataset, device
            )
        elif self.strategy == 'loss':
            models = [model_or_models] if not isinstance(model_or_models, list) else model_or_models
            difficulties = DifficultyEstimator.by_loss_magnitude(
                models[0], dataset, device
            )
        elif self.strategy == 'agreement':
            models = model_or_models if isinstance(model_or_models, list) else [model_or_models]
            difficulties = DifficultyEstimator.by_ensemble_agreement(
                models, dataset, device
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Create curriculum sampler
        def schedule(epoch):
            if epoch < self.warmup_epochs:
                return 1.0  # Use all data during warmup
            else:
                # Gradual curriculum after warmup
                progress = (epoch - self.warmup_epochs) / 100.0
                return min(0.3 + progress * 0.7, 1.0)
        
        sampler = CurriculumSampler(
            dataset=dataset,
            difficulty_scores=difficulties,
            start_easy=self.start_easy,
            curriculum_schedule=schedule
        )
        
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    def __repr__(self):
        return f"CurriculumStrategy(strategy='{self.strategy}', start_easy={self.start_easy})"
