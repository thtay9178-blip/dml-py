"""
Attention Transfer Example for DML-PY.

This example demonstrates how attention transfer improves knowledge distillation
by matching spatial attention patterns between teacher and student.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from pydml.models.cifar import resnet32, mobilenet_v2
from pydml.losses import AttentionTransferLoss, attention_transfer_loss


class AttentionDistillationTrainer:
    """
    Simple trainer for distillation with attention transfer.
    
    Combines:
    - Standard cross-entropy loss
    - KL divergence between logits
    - Attention transfer at intermediate layers
    """
    
    def __init__(
        self,
        teacher,
        student,
        device='cpu',
        alpha=0.5,
        beta=0.3,
        temperature=4.0
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device
        self.alpha = alpha  # Weight for KL loss
        self.beta = beta    # Weight for attention loss
        self.temperature = temperature
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.at_loss = AttentionTransferLoss(normalize=True)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
        
        # Hooks to extract intermediate features
        self.teacher_features = []
        self.student_features = []
        self._register_hooks()
    
    def _get_hook(self, feature_list):
        def hook(module, input, output):
            feature_list.append(output)
        return hook
    
    def _register_hooks(self):
        """Register hooks to extract features from layer2 and layer3."""
        # For ResNet, extract from layer2 and layer3
        if hasattr(self.teacher, 'layer2'):
            self.teacher.layer2.register_forward_hook(
                self._get_hook(self.teacher_features)
            )
        if hasattr(self.teacher, 'layer3'):
            self.teacher.layer3.register_forward_hook(
                self._get_hook(self.teacher_features)
            )
        
        if hasattr(self.student, 'layer2'):
            self.student.layer2.register_forward_hook(
                self._get_hook(self.student_features)
            )
        if hasattr(self.student, 'layer3'):
            self.student.layer3.register_forward_hook(
                self._get_hook(self.student_features)
            )
    
    def train_epoch(self, train_loader):
        self.student.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Clear feature lists
            self.teacher_features.clear()
            self.student_features.clear()
            
            # Forward pass
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            
            student_logits = self.student(inputs)
            
            # 1. Cross-entropy loss
            ce_loss = self.ce_loss(student_logits, targets)
            
            # 2. KL divergence loss
            teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)
            student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=1)
            kl_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
            
            # 3. Attention transfer loss
            at_loss = 0.0
            if len(self.teacher_features) > 0 and len(self.student_features) > 0:
                for teacher_feat, student_feat in zip(
                    self.teacher_features, self.student_features
                ):
                    at_loss += self.at_loss(student_feat, teacher_feat)
                at_loss /= len(self.teacher_features)
            
            # Combined loss
            loss = (1 - self.alpha - self.beta) * ce_loss + \
                   self.alpha * kl_loss + \
                   self.beta * at_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, test_loader):
        self.student.eval()
        correct = 0
        total = 0
        
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Clear features
            self.student_features.clear()
            
            outputs = self.student(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total


def main():
    print("=" * 60)
    print("DML-PY - Attention Transfer Demo")
    print("=" * 60)
    
    # Config
    device = 'cpu'
    num_epochs = 10
    batch_size = 32
    num_samples = 200
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples: {num_samples}")
    
    # Load data
    print("\n1. Loading data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_indices = torch.randperm(len(train_dataset))[:num_samples]
    test_indices = torch.randperm(len(test_dataset))[:num_samples // 2]
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, 
                            shuffle=False, num_workers=0)
    
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Test samples: {len(test_subset)}")
    
    # Train teacher
    print("\n2. Training teacher model (ResNet32)...")
    print("-" * 60)
    
    teacher = resnet32(num_classes=10).to(device)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, num_epochs + 1):
        teacher.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            
            teacher_optimizer.zero_grad()
            loss.backward()
            teacher_optimizer.step()
        
        if epoch % 5 == 0 or epoch == num_epochs:
            teacher.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = teacher(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            acc = 100. * correct / total
            print(f"  Epoch {epoch}/{num_epochs} - Accuracy: {acc:.2f}%")
    
    teacher_acc = acc
    print(f"\n✓ Teacher training complete! Final accuracy: {teacher_acc:.2f}%")
    print("-" * 60)
    
    # Train student WITH attention transfer
    print("\n3. Training student WITH attention transfer...")
    print("-" * 60)
    
    student_with_at = resnet32(num_classes=10)
    
    trainer_with_at = AttentionDistillationTrainer(
        teacher=teacher,
        student=student_with_at,
        device=device,
        alpha=0.5,  # KL loss weight
        beta=0.3,   # Attention loss weight
        temperature=4.0
    )
    
    print(f"  Loss weights: CE=0.2, KL=0.5, Attention=0.3")
    
    for epoch in range(1, num_epochs + 1):
        loss = trainer_with_at.train_epoch(train_loader)
        
        if epoch % 5 == 0 or epoch == num_epochs:
            acc = trainer_with_at.evaluate(test_loader)
            print(f"  Epoch {epoch}/{num_epochs} - Loss: {loss:.4f} | Accuracy: {acc:.2f}%")
    
    acc_with_at = trainer_with_at.evaluate(test_loader)
    print(f"\n✓ Training complete! Final accuracy: {acc_with_at:.2f}%")
    print("-" * 60)
    
    # Train student WITHOUT attention transfer (standard distillation)
    print("\n4. Training student WITHOUT attention transfer (standard KD)...")
    print("-" * 60)
    
    student_no_at = resnet32(num_classes=10)
    
    trainer_no_at = AttentionDistillationTrainer(
        teacher=teacher,
        student=student_no_at,
        device=device,
        alpha=0.7,  # More KL loss
        beta=0.0,   # No attention loss
        temperature=4.0
    )
    
    print(f"  Loss weights: CE=0.3, KL=0.7, Attention=0.0")
    
    for epoch in range(1, num_epochs + 1):
        loss = trainer_no_at.train_epoch(train_loader)
        
        if epoch % 5 == 0 or epoch == num_epochs:
            acc = trainer_no_at.evaluate(test_loader)
            print(f"  Epoch {epoch}/{num_epochs} - Loss: {loss:.4f} | Accuracy: {acc:.2f}%")
    
    acc_no_at = trainer_no_at.evaluate(test_loader)
    print(f"\n✓ Training complete! Final accuracy: {acc_no_at:.2f}%")
    print("-" * 60)
    
    # Compare results
    print("\n5. Results Summary")
    print("=" * 60)
    print(f"  Teacher (ResNet32):              {teacher_acc:.2f}%")
    print(f"  Student w/ Attention Transfer:   {acc_with_at:.2f}%")
    print(f"  Student w/o Attention Transfer:  {acc_no_at:.2f}%")
    print("-" * 60)
    
    improvement = acc_with_at - acc_no_at
    gap_with = teacher_acc - acc_with_at
    gap_without = teacher_acc - acc_no_at
    
    print(f"\n  Improvement from Attention:      {improvement:+.2f}%")
    print(f"  Gap to Teacher (with AT):        {gap_with:.2f}%")
    print(f"  Gap to Teacher (without AT):     {gap_without:.2f}%")
    
    if improvement > 0:
        print("\n  ✓ Attention transfer helped!")
        print("    Student learned better spatial attention patterns")
    else:
        print("\n  Note: Results may vary with small datasets")
    
    print("\n" + "=" * 60)
    print("Attention Transfer demo completed!")
    print("\nKey Insights:")
    print("  - Attention transfer matches spatial activation patterns")
    print("  - Encourages similar 'where to look' behavior")
    print("  - Complements logit-based knowledge transfer")
    print("  - Particularly effective for similar architectures")
    print("=" * 60)


if __name__ == '__main__':
    main()
