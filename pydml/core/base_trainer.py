"""
Base trainer class for collaborative learning.

This module provides the abstract base class that all collaborative trainers inherit from.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


class BaseCollaborativeTrainer(ABC):
    """
    Abstract base class for all collaborative learning trainers.
    
    This class defines the interface that all collaborative trainers must implement.
    It handles common functionality like training loops, evaluation, checkpointing, etc.
    
    Args:
        models: List of PyTorch models to train collaboratively
        device: Device to train on ('cuda' or 'cpu')
        optimizers: Optional list of optimizers (one per model). If None, creates Adam optimizers.
        schedulers: Optional list of learning rate schedulers
        callbacks: Optional list of callbacks for training hooks
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = 'cuda',
        optimizers: Optional[List[torch.optim.Optimizer]] = None,
        schedulers: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        self.models = models
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_models = len(models)
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
        
        # Setup optimizers (default: Adam with lr=0.001)
        if optimizers is None:
            self.optimizers = [
                torch.optim.Adam(model.parameters(), lr=0.001)
                for model in self.models
            ]
        else:
            assert len(optimizers) == self.num_models, \
                f"Number of optimizers ({len(optimizers)}) must match number of models ({self.num_models})"
            self.optimizers = optimizers
        
        # Setup schedulers
        self.schedulers = schedulers or []
        
        # Setup callbacks
        self.callbacks = callbacks or []
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    @abstractmethod
    def compute_collaborative_loss(
        self,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the collaborative loss for all models.
        
        This method must be implemented by subclasses to define how models learn from each other.
        
        Args:
            outputs: List of model outputs (logits), one per model
            targets: Ground truth labels
        
        Returns:
            Dictionary mapping model index to its total loss
        """
        pass
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        # Set models to training mode
        for model in self.models:
            model.train()
        
        total_losses = [0.0] * self.num_models
        correct = [0] * self.num_models
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass for all models
            outputs = []
            for model in self.models:
                output = model(inputs)
                outputs.append(output)
            
            # Compute collaborative loss
            losses = self.compute_collaborative_loss(outputs, targets)
            
            # Backward pass and optimization for each model
            for i, (optimizer, model) in enumerate(zip(self.optimizers, self.models)):
                optimizer.zero_grad()
                loss = losses[f'model_{i}']
                loss.backward(retain_graph=(i < self.num_models - 1))
                optimizer.step()
                
                total_losses[i] += loss.item()
            
            # Compute accuracy
            for i, output in enumerate(outputs):
                _, predicted = output.max(1)
                correct[i] += predicted.eq(targets).sum().item()
            
            total += targets.size(0)
            self.global_step += 1
            
            # Update progress bar
            avg_loss = sum(total_losses) / (self.num_models * (batch_idx + 1))
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Compute epoch metrics
        metrics = {
            'train_loss': sum(total_losses) / (self.num_models * len(train_loader)),
            'train_acc': sum(correct) / (self.num_models * total) * 100,
        }
        
        # Per-model metrics
        for i in range(self.num_models):
            metrics[f'train_loss_model_{i}'] = total_losses[i] / len(train_loader)
            metrics[f'train_acc_model_{i}'] = correct[i] / total * 100
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate all models on validation set.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Dictionary of validation metrics
        """
        # Set models to evaluation mode
        for model in self.models:
            model.eval()
        
        total_losses = [0.0] * self.num_models
        correct = [0] * self.num_models
        total = 0
        
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass for all models
            outputs = []
            for model in self.models:
                output = model(inputs)
                outputs.append(output)
            
            # Compute loss
            losses = self.compute_collaborative_loss(outputs, targets)
            
            for i in range(self.num_models):
                total_losses[i] += losses[f'model_{i}'].item()
            
            # Compute accuracy
            for i, output in enumerate(outputs):
                _, predicted = output.max(1)
                correct[i] += predicted.eq(targets).sum().item()
            
            total += targets.size(0)
        
        # Compute metrics
        metrics = {
            'val_loss': sum(total_losses) / (self.num_models * len(val_loader)),
            'val_acc': sum(correct) / (self.num_models * total) * 100,
        }
        
        # Per-model metrics
        for i in range(self.num_models):
            metrics[f'val_loss_model_{i}'] = total_losses[i] / len(val_loader)
            metrics[f'val_acc_model_{i}'] = correct[i] / total * 100
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the models for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            verbose: Whether to print training progress
        
        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Training {self.num_models} models collaboratively for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate if validation loader provided
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            else:
                val_metrics = {}
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            if val_loader is not None:
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_acc'].append(val_metrics['val_acc'])
            
            # Step schedulers
            for scheduler in self.schedulers:
                scheduler.step()
            
            # Print epoch summary
            if verbose:
                epoch_time = time.time() - start_time
                print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.2f}s")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']:.2f}%")
                if val_loader is not None:
                    print(f"  Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.2f}%")
                
                # Per-model stats
                for i in range(self.num_models):
                    print(f"  Model {i}: Train Acc: {train_metrics[f'train_acc_model_{i}']:.2f}%", end="")
                    if val_loader is not None:
                        print(f" | Val Acc: {val_metrics[f'val_acc_model_{i}']:.2f}%")
                    else:
                        print()
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save checkpoint of all models and optimizers."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'history': self.history,
            'models': [model.state_dict() for model in self.models],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint of all models and optimizers."""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        
        for model, state_dict in zip(self.models, checkpoint['models']):
            model.load_state_dict(state_dict)
        
        for opt, state_dict in zip(self.optimizers, checkpoint['optimizers']):
            opt.load_state_dict(state_dict)
        
        print(f"Checkpoint loaded from {path} (epoch {self.current_epoch})")
