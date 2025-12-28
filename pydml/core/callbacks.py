"""
Callback system for training hooks.

This module provides a callback system for extending trainer functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Callback(ABC):
    """
    Abstract base class for callbacks.
    
    Callbacks can be used to execute custom code at different points during training.
    """
    
    def on_train_begin(self, trainer: Any):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: Any):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: Any, batch: int):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer: Any, batch: int, loss: float):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when validation metric stops improving.
    
    Args:
        monitor: Metric to monitor (e.g., 'val_loss', 'val_acc')
        patience: Number of epochs to wait before stopping
        mode: 'min' for metrics to minimize, 'max' for metrics to maximize
        min_delta: Minimum change to qualify as improvement
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Check if training should stop."""
        if self.monitor not in metrics:
            return
        
        current_value = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best {self.monitor}: {self.best_value:.4f}")
                # Note: Actual stopping would require trainer support
                # This is a simplified version


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Args:
        filepath: Path template for checkpoint files (can include {epoch}, {val_loss}, etc.)
        monitor: Metric to monitor for saving best model
        mode: 'min' for metrics to minimize, 'max' for metrics to maximize
        save_best_only: If True, only save when monitored metric improves
        save_freq: Save every N epochs (if save_best_only is False)
    """
    
    def __init__(
        self,
        filepath: str = 'checkpoint_epoch_{epoch}.pt',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_freq: int = 1,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint if conditions are met."""
        if self.save_best_only:
            if self.monitor not in metrics:
                return
            
            current_value = metrics[self.monitor]
            
            if self.mode == 'min':
                improved = current_value < self.best_value
            else:
                improved = current_value > self.best_value
            
            if improved:
                self.best_value = current_value
                filepath = self.filepath.format(epoch=epoch, **metrics)
                trainer.save_checkpoint(filepath)
                print(f"Saved best model to {filepath}")
        else:
            if epoch % self.save_freq == 0:
                filepath = self.filepath.format(epoch=epoch, **metrics)
                trainer.save_checkpoint(filepath)
                print(f"Saved checkpoint to {filepath}")


class LearningRateLogger(Callback):
    """
    Log learning rates during training.
    """
    
    def on_epoch_begin(self, trainer: Any, epoch: int):
        """Log current learning rates."""
        lrs = [opt.param_groups[0]['lr'] for opt in trainer.optimizers]
        print(f"Learning rates: {lrs}")


class TensorBoardLogger(Callback):
    """
    Log metrics to TensorBoard.
    
    Args:
        log_dir: Directory to save TensorBoard logs
    """
    
    def __init__(self, log_dir: str = 'runs'):
        self.log_dir = log_dir
        self.writer = None
    
    def on_train_begin(self, trainer: Any):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            print(f"TensorBoard logging to {self.log_dir}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
        
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
    
    def on_train_end(self, trainer: Any):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
