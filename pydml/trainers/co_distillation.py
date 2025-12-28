"""
Co-Distillation Trainer.

Combines traditional teacher-student knowledge distillation with 
deep mutual learning among multiple student networks.

This hybrid approach allows:
1. Students to learn from a pre-trained teacher
2. Students to learn from each other (mutual learning)
3. Students to maintain diversity while benefiting from teacher guidance

Reference: Extends ideas from:
- "Deep Mutual Learning" (Zhang et al., CVPR 2018)
- "Knowledge Distillation" (Hinton et al., 2015)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pydml.core.base_trainer import BaseCollaborativeTrainer
from pydml.core.losses import CrossEntropyLoss, KLDivergenceLoss


@dataclass
class CoDistillationConfig:
    """
    Configuration for Co-Distillation.
    
    Args:
        temperature: Temperature for KL divergence
        teacher_weight: Weight for teacher distillation loss
        peer_weight: Weight for peer mutual learning loss
        supervised_weight: Weight for supervised loss
        teacher_temp: Separate temperature for teacher (if None, uses temperature)
    """
    temperature: float = 3.0
    teacher_weight: float = 0.5
    peer_weight: float = 0.3
    supervised_weight: float = 0.2
    teacher_temp: Optional[float] = None


class CoDistillationTrainer(BaseCollaborativeTrainer):
    """
    Co-Distillation Trainer.
    
    Trains multiple student networks collaboratively while learning from
    a pre-trained teacher network.
    
    Loss for each student i:
        L_i = w_sup * L_CE(y_i, y_true)                    [supervised]
            + w_teacher * L_KL(y_i, y_teacher)             [teacher distillation]
            + w_peer * (1/K-1) * Σ_{j≠i} L_KL(y_i, y_j)   [peer mutual learning]
    
    Benefits:
    - Students get strong guidance from teacher
    - Students maintain diversity through mutual learning
    - Better than single student distillation
    - Better than pure mutual learning without teacher
    
    Example:
        >>> teacher = resnet50(pretrained=True)
        >>> students = [mobilenet_v2(), efficientnet_b0(), resnet18()]
        >>> 
        >>> trainer = CoDistillationTrainer(
        ...     teacher=teacher,
        ...     students=students,
        ...     config=CoDistillationConfig(teacher_weight=0.5, peer_weight=0.3)
        ... )
        >>> trainer.fit(train_loader, epochs=100)
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        students: List[nn.Module],
        config: Optional[CoDistillationConfig] = None,
        device: str = 'cuda',
        optimizers: Optional[List] = None,
        schedulers: Optional[List] = None,
        callbacks: Optional[List] = None,
    ):
        # Initialize base trainer with students as models
        super().__init__(
            models=students,
            device=device,
            optimizers=optimizers,
            schedulers=schedulers,
            callbacks=callbacks,
        )
        
        # Handle dict or config object
        if isinstance(config, dict):
            self.config = CoDistillationConfig(**config)
        else:
            self.config = config or CoDistillationConfig()
        
        # Setup teacher
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Loss functions
        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = KLDivergenceLoss(temperature=self.config.temperature)
        
        # Separate teacher KL loss if different temperature
        teacher_temp = self.config.teacher_temp or self.config.temperature
        self.teacher_kl_loss = KLDivergenceLoss(temperature=teacher_temp)
        
        print(f"Co-Distillation Trainer initialized")
        print(f"  Teacher: {teacher.__class__.__name__}")
        print(f"  Students: {self.num_models} models")
        print(f"  Loss weights: supervised={self.config.supervised_weight}, "
              f"teacher={self.config.teacher_weight}, peer={self.config.peer_weight}")
        print(f"  Temperature: {self.config.temperature}")
    
    def compute_collaborative_loss(
        self,
        student_outputs: List[torch.Tensor],
        targets: torch.Tensor,
        teacher_output: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute co-distillation loss combining teacher guidance and peer learning.
        
        Args:
            student_outputs: List of student output logits
            targets: Ground truth labels
            teacher_output: Teacher output logits
        
        Returns:
            Dictionary of losses for each student
        """
        losses = {}
        
        for i in range(self.num_models):
            # 1. Supervised loss (cross-entropy with ground truth)
            supervised_loss = self.ce_loss(student_outputs[i], targets)
            
            # 2. Teacher distillation loss
            teacher_distill_loss = self.teacher_kl_loss(
                student_outputs[i], teacher_output
            )
            
            # 3. Peer mutual learning loss
            peer_loss = 0.0
            for j in range(self.num_models):
                if i != j:
                    kl_loss = self.kl_loss(student_outputs[i], student_outputs[j])
                    peer_loss += kl_loss
            
            if self.num_models > 1:
                peer_loss = peer_loss / (self.num_models - 1)
            
            # Combine all losses
            total_loss = (
                self.config.supervised_weight * supervised_loss +
                self.config.teacher_weight * teacher_distill_loss +
                self.config.peer_weight * peer_loss
            )
            
            losses[f'model_{i}'] = total_loss
        
        return losses
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with co-distillation."""
        # Set students to train mode
        for model in self.models:
            model.train()
        
        # Teacher always in eval
        self.teacher.eval()
        
        epoch_losses = {f'model_{i}': 0.0 for i in range(self.num_models)}
        total_correct = {f'model_{i}': 0 for i in range(self.num_models)}
        total_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_output = self.teacher(inputs)
            
            # Get student predictions
            student_outputs = [model(inputs) for model in self.models]
            
            # Compute losses
            losses = self.compute_collaborative_loss(
                student_outputs, targets, teacher_output
            )
            
            # Backward and optimize each student
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            
            for i in range(self.num_models):
                losses[f'model_{i}'].backward(retain_graph=True)
            
            for optimizer in self.optimizers:
                optimizer.step()
            
            # Track metrics
            for i in range(self.num_models):
                epoch_losses[f'model_{i}'] += losses[f'model_{i}'].item()
                
                _, predicted = student_outputs[i].max(1)
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
    def evaluate_with_teacher(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate both students and teacher on validation set.
        
        Returns metrics including teacher performance for comparison.
        """
        # Set to eval mode
        for model in self.models:
            model.eval()
        self.teacher.eval()
        
        total_losses = [0.0] * self.num_models
        correct = [0] * self.num_models
        teacher_correct = 0
        total = 0
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Teacher prediction
            teacher_output = self.teacher(inputs)
            _, teacher_pred = teacher_output.max(1)
            teacher_correct += teacher_pred.eq(targets).sum().item()
            
            # Student predictions
            for i, model in enumerate(self.models):
                output = model(inputs)
                loss = self.ce_loss(output, targets)
                total_losses[i] += loss.item()
                
                _, predicted = output.max(1)
                correct[i] += predicted.eq(targets).sum().item()
            
            total += targets.size(0)
        
        # Compute metrics
        metrics = {
            'val_loss': sum(total_losses) / (self.num_models * len(val_loader)),
            'val_acc': sum(correct) / (self.num_models * total) * 100,
            'teacher_acc': teacher_correct / total * 100,
        }
        
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
        Train students with co-distillation.
        
        Overrides base fit to include teacher evaluation.
        """
        if verbose:
            print(f"Starting co-distillation training on {self.device}")
            print(f"Training {self.num_models} student models for {epochs} epochs")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'teacher_acc': [],
        }
        
        for i in range(self.num_models):
            history[f'train_loss_model_{i}'] = []
            history[f'train_acc_model_{i}'] = []
            history[f'val_loss_model_{i}'] = []
            history[f'val_acc_model_{i}'] = []
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            if val_loader is not None:
                val_metrics = self.evaluate_with_teacher(val_loader)
            else:
                val_metrics = {}
            
            # Record history
            for key in train_metrics:
                if key in history:
                    history[key].append(train_metrics[key])
            
            for key in val_metrics:
                if key in history:
                    history[key].append(val_metrics[key])
            
            # Print progress
            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"\nEpoch {epoch}/{epochs}:")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f} | "
                      f"Train Acc: {train_metrics['train_acc']:.2f}%")
                
                if val_loader is not None:
                    print(f"  Val Loss: {val_metrics['val_loss']:.4f} | "
                          f"Val Acc: {val_metrics['val_acc']:.2f}%")
                    print(f"  Teacher Acc: {val_metrics['teacher_acc']:.2f}%")
                    
                    # Show per-student performance
                    for i in range(self.num_models):
                        student_acc = val_metrics[f'val_acc_model_{i}']
                        print(f"    Student {i}: {student_acc:.2f}%")
            
            # Callbacks
            if self.callbacks:
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, train_metrics, val_metrics)
            
            # Schedulers
            if self.schedulers:
                for scheduler in self.schedulers:
                    scheduler.step()
        
        if verbose:
            print("\nTraining completed!")
        
        return history
