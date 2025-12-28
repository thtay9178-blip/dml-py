"""
Mixed Precision Training Support for DML-PY.

Provides automatic mixed precision (AMP) training utilities for faster training
with lower memory usage on compatible GPUs.

Supports:
- PyTorch native AMP (torch.cuda.amp)
- Automatic gradient scaling
- Compatible with all DML-PY trainers
- Optional fallback for CPU training
"""

import torch
from typing import Optional, Dict, List
from contextlib import nullcontext


class AMPConfig:
    """
    Configuration for Automatic Mixed Precision training.
    
    Args:
        enabled: Whether to use AMP (auto-detected if None)
        dtype: Data type for mixed precision (default: float16)
        init_scale: Initial scale factor for gradient scaler
        growth_factor: Factor to multiply scale by if no inf/NaN gradients
        backoff_factor: Factor to multiply scale by if inf/NaN gradients found
        growth_interval: Number of iterations before attempting to increase scale
        
    Example:
        >>> config = AMPConfig(enabled=True, dtype=torch.float16)
        >>> trainer = DMLTrainer(models, amp_config=config)
    """
    
    def __init__(
        self,
        enabled: Optional[bool] = None,
        dtype: torch.dtype = torch.float16,
        init_scale: float = 2.**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        # Auto-detect if not specified
        if enabled is None:
            enabled = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        self.enabled = enabled
        self.dtype = dtype
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval


class AMPManager:
    """
    Manager for Automatic Mixed Precision training.
    
    Handles gradient scaling and autocast contexts for mixed precision training.
    
    Args:
        config: AMP configuration
        device: Device for training
        
    Example:
        >>> amp_manager = AMPManager(AMPConfig(enabled=True))
        >>> 
        >>> with amp_manager.autocast():
        >>>     outputs = model(inputs)
        >>>     loss = criterion(outputs, targets)
        >>> 
        >>> amp_manager.scale_loss(loss).backward()
        >>> amp_manager.step(optimizer)
        >>> amp_manager.update()
    """
    
    def __init__(self, config: AMPConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize gradient scaler if enabled
        if self.config.enabled and device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=config.init_scale,
                growth_factor=config.growth_factor,
                backoff_factor=config.backoff_factor,
                growth_interval=config.growth_interval
            )
        else:
            self.scaler = None
    
    def autocast(self):
        """
        Return autocast context manager.
        
        Returns:
            Context manager for mixed precision forward pass
        """
        if self.config.enabled and self.device == 'cuda':
            return torch.cuda.amp.autocast(dtype=self.config.dtype)
        else:
            return nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for gradient computation.
        
        Args:
            loss: Unscaled loss tensor
            
        Returns:
            Scaled loss if AMP enabled, otherwise original loss
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Perform optimizer step with gradient unscaling.
        
        Args:
            optimizer: Optimizer to step
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update(self):
        """Update gradient scaler state."""
        if self.scaler is not None:
            self.scaler.update()
    
    def unscale_(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients before gradient clipping or inspection.
        
        Args:
            optimizer: Optimizer containing gradients to unscale
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        if self.scaler is not None:
            return {'scaler': self.scaler.state_dict()}
        return {}
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint."""
        if self.scaler is not None and 'scaler' in state_dict:
            self.scaler.load_state_dict(state_dict['scaler'])


def apply_amp_to_trainer(trainer, amp_config: Optional[AMPConfig] = None):
    """
    Apply AMP to an existing trainer.
    
    Args:
        trainer: DML-PY trainer instance
        amp_config: AMP configuration (auto-configured if None)
        
    Returns:
        Modified trainer with AMP support
        
    Example:
        >>> trainer = DMLTrainer(models)
        >>> trainer = apply_amp_to_trainer(trainer, AMPConfig(enabled=True))
    """
    if amp_config is None:
        amp_config = AMPConfig()
    
    trainer.amp_manager = AMPManager(amp_config, trainer.device)
    
    # Store original training step
    original_train_step = trainer.train_step
    
    def amp_train_step(self, batch, batch_idx):
        """Training step with AMP support."""
        # Use autocast for forward pass
        with self.amp_manager.autocast():
            result = original_train_step(batch, batch_idx)
        return result
    
    # Replace training step
    import types
    trainer.train_step = types.MethodType(amp_train_step, trainer)
    
    return trainer


__all__ = [
    'AMPConfig',
    'AMPManager',
    'apply_amp_to_trainer'
]
