"""
Knowledge Distillation Trainer.

Implementation of traditional knowledge distillation where a student network
learns from a pre-trained teacher network.

Reference: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pydml.core.losses import CrossEntropyLoss, KLDivergenceLoss


@dataclass
class DistillationConfig:
    """
    Configuration for Knowledge Distillation.
    
    Args:
        temperature: Temperature for softening probability distributions
        alpha: Weight for distillation loss (1-alpha for student loss)
    """
    temperature: float = 3.0
    alpha: float = 0.5


class DistillationTrainer:
    """
    Knowledge Distillation Trainer.
    
    Trains a student network to mimic a pre-trained teacher network.
    
    Loss = alpha * KL(student, teacher) + (1-alpha) * CE(student, labels)
    
    Args:
        teacher: Pre-trained teacher model
        student: Student model to train
        config: DistillationConfig instance
        device: Device to train on
        optimizer: Optimizer for student (if None, uses Adam)
        scheduler: Learning rate scheduler
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None,
        device: str = 'cuda',
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[any] = None,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config or DistillationConfig()
        
        # Teacher is always in eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Loss functions
        self.ce_loss = CrossEntropyLoss()
        self.kl_loss = KLDivergenceLoss(temperature=self.config.temperature)
        
        print(f"Distillation Trainer initialized")
        print(f"Temperature: {self.config.temperature}, Alpha: {self.config.alpha}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            # Get student predictions
            student_logits = self.student(inputs)
            
            # Distillation loss
            distill_loss = self.kl_loss(student_logits, teacher_logits)
            
            # Student loss (hard labels)
            student_loss = self.ce_loss(student_logits, targets)
            
            # Combined loss
            loss = self.config.alpha * distill_loss + (1 - self.config.alpha) * student_loss
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict:
        """Evaluate student on validation set."""
        self.student.eval()
        
        correct = 0
        total = 0
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.student(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return {'val_acc': accuracy}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
    ) -> dict:
        """Train the student model."""
        print(f"Starting distillation training for {epochs} epochs")
        
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Evaluate
            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                history['val_acc'].append(metrics['val_acc'])
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} | Val Acc: {metrics['val_acc']:.2f}%")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f}")
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        return history
