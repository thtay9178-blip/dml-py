"""
Distributed Training Support for DML-PY.

Provides utilities for multi-GPU and multi-node distributed training using
PyTorch DistributedDataParallel (DDP).

Features:
- Automatic DDP setup and teardown
- Gradient synchronization across processes
- Compatible with all DML-PY trainers
- Support for multiple backend (nccl, gloo, mpi)
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, List, Callable
import os


class DistributedConfig:
    """
    Configuration for distributed training.
    
    Args:
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        init_method: URL for process group initialization
        world_size: Total number of processes
        rank: Rank of current process
        local_rank: Rank within the node
        find_unused_parameters: Whether to find unused parameters in DDP
        
    Example:
        >>> config = DistributedConfig(backend='nccl', world_size=4)
        >>> trainer = DMLTrainer(models, distributed_config=config)
    """
    
    def __init__(
        self,
        backend: str = 'nccl',
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        find_unused_parameters: bool = False
    ):
        self.backend = backend
        self.init_method = init_method or 'env://'
        
        # Auto-detect from environment if not provided
        self.world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        self.rank = rank or int(os.environ.get('RANK', 0))
        self.local_rank = local_rank or int(os.environ.get('LOCAL_RANK', 0))
        
        self.find_unused_parameters = find_unused_parameters


class DistributedManager:
    """
    Manager for distributed training.
    
    Handles process group initialization, model wrapping with DDP,
    and proper cleanup.
    
    Args:
        config: Distributed configuration
        
    Example:
        >>> dist_manager = DistributedManager(DistributedConfig())
        >>> dist_manager.setup()
        >>> 
        >>> # Wrap models
        >>> models = [dist_manager.wrap_model(m) for m in models]
        >>> 
        >>> # Training...
        >>> 
        >>> dist_manager.cleanup()
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
    
    def setup(self):
        """Initialize distributed process group."""
        if self.config.world_size > 1 and not self.is_initialized:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            self.is_initialized = True
            
            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
    
    def cleanup(self):
        """Destroy distributed process group."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
    
    def wrap_model(self, model: torch.nn.Module, device_ids: Optional[List[int]] = None) -> torch.nn.Module:
        """
        Wrap model with DistributedDataParallel.
        
        Args:
            model: Model to wrap
            device_ids: GPU IDs to use (auto-detected if None)
            
        Returns:
            DDP-wrapped model
        """
        if self.config.world_size <= 1:
            return model
        
        if device_ids is None and torch.cuda.is_available():
            device_ids = [self.config.local_rank]
        
        return DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=self.config.find_unused_parameters
        )
    
    def create_distributed_sampler(self, dataset, shuffle: bool = True):
        """
        Create DistributedSampler for dataset.
        
        Args:
            dataset: Dataset to sample from
            shuffle: Whether to shuffle data
            
        Returns:
            DistributedSampler if multi-process, None otherwise
        """
        if self.config.world_size <= 1:
            return None
        
        return DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle
        )
    
    def is_main_process(self) -> bool:
        """Check if current process is the main process."""
        return self.config.rank == 0
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """
        All-reduce operation across processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            
        Returns:
            Reduced tensor
        """
        if self.is_initialized:
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather tensors from all processes.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            List of tensors from all processes
        """
        if not self.is_initialized:
            return [tensor]
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list


def launch_distributed(
    train_fn: Callable,
    nprocs: int,
    config: Optional[DistributedConfig] = None,
    **kwargs
):
    """
    Launch distributed training across multiple processes.
    
    Args:
        train_fn: Training function to run on each process
        nprocs: Number of processes to spawn
        config: Distributed configuration
        **kwargs: Additional arguments passed to train_fn
        
    Example:
        >>> def train(rank, world_size):
        >>>     config = DistributedConfig(rank=rank, world_size=world_size)
        >>>     dist_manager = DistributedManager(config)
        >>>     dist_manager.setup()
        >>>     # Training code...
        >>>     dist_manager.cleanup()
        >>> 
        >>> launch_distributed(train, nprocs=4)
    """
    if config is None:
        config = DistributedConfig(world_size=nprocs)
    
    mp.spawn(
        train_fn,
        args=(config,) + tuple(kwargs.values()),
        nprocs=nprocs,
        join=True
    )


def apply_distributed_to_trainer(trainer, dist_config: Optional[DistributedConfig] = None):
    """
    Apply distributed training to an existing trainer.
    
    Args:
        trainer: DML-PY trainer instance
        dist_config: Distributed configuration
        
    Returns:
        Modified trainer with distributed support
        
    Example:
        >>> trainer = DMLTrainer(models)
        >>> trainer = apply_distributed_to_trainer(trainer, DistributedConfig())
    """
    if dist_config is None:
        dist_config = DistributedConfig()
    
    dist_manager = DistributedManager(dist_config)
    dist_manager.setup()
    
    # Wrap all models with DDP
    trainer.models = [dist_manager.wrap_model(m) for m in trainer.models]
    trainer.dist_manager = dist_manager
    
    return trainer


__all__ = [
    'DistributedConfig',
    'DistributedManager',
    'launch_distributed',
    'apply_distributed_to_trainer'
]
