"""
Knowledge Distillation Example for DML-PY.

This example demonstrates traditional teacher-student distillation:
1. Train a teacher model
2. Use it to train a smaller student model
3. Compare student performance with and without distillation
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from pydml.trainers import DistillationTrainer, DistillationConfig
from pydml.models.cifar import resnet32, mobilenet_v2


def train_baseline_student(student, train_loader, test_loader, device, epochs=5):
    """Train student from scratch without teacher."""
    student = student.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Training student from scratch (baseline)...")
    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = student(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    
    return acc


def main():
    print("=" * 60)
    print("DML-PY - Knowledge Distillation Demo")
    print("=" * 60)
    
    # Config
    device = 'cpu'
    num_epochs = 10
    batch_size = 32
    num_samples = 200  # Small dataset for demo
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples: {num_samples}")
    
    # Prepare data
    print("\n1. Loading CIFAR-10 data...")
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
    
    # Use subsets for speed
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
    
    # Step 1: Train teacher model
    print("\n2. Training teacher model (ResNet32)...")
    print("-" * 60)
    
    teacher = resnet32(num_classes=10).to(device)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, num_epochs + 1):
        teacher.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            
            teacher_optimizer.zero_grad()
            loss.backward()
            teacher_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if epoch % 5 == 0 or epoch == num_epochs:
            # Evaluate
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
            
            teacher_acc = 100. * correct / total
            print(f"  Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} | Acc: {teacher_acc:.2f}%")
    
    print(f"\n✓ Teacher training complete! Final accuracy: {teacher_acc:.2f}%")
    print("-" * 60)
    
    # Step 2: Train student with distillation
    print("\n3. Training student (MobileNetV2) WITH distillation...")
    print("-" * 60)
    
    student_distilled = mobilenet_v2(num_classes=10)
    
    distillation_config = DistillationConfig(
        temperature=4.0,
        alpha=0.7  # 70% distillation loss, 30% hard label loss
    )
    
    distill_trainer = DistillationTrainer(
        teacher=teacher,
        student=student_distilled,
        config=distillation_config,
        device=device
    )
    
    history_distilled = distill_trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=num_epochs
    )
    
    distilled_acc = history_distilled['val_acc'][-1]
    print(f"\n✓ Distillation complete! Final accuracy: {distilled_acc:.2f}%")
    print("-" * 60)
    
    # Step 3: Train student baseline (without distillation)
    print("\n4. Training student WITHOUT distillation (baseline)...")
    print("-" * 60)
    
    student_baseline = mobilenet_v2(num_classes=10)
    baseline_acc = train_baseline_student(
        student_baseline, train_loader, test_loader, device, epochs=num_epochs
    )
    
    print(f"\n✓ Baseline training complete! Final accuracy: {baseline_acc:.2f}%")
    print("-" * 60)
    
    # Step 4: Compare results
    print("\n5. Results Summary")
    print("=" * 60)
    print(f"  Teacher (ResNet32):           {teacher_acc:.2f}%")
    print(f"  Student w/ Distillation:      {distilled_acc:.2f}%")
    print(f"  Student w/o Distillation:     {baseline_acc:.2f}%")
    print("-" * 60)
    
    improvement = distilled_acc - baseline_acc
    teacher_gap = teacher_acc - distilled_acc
    
    print(f"\n  Distillation Improvement:     {improvement:+.2f}%")
    print(f"  Gap to Teacher:               {teacher_gap:.2f}%")
    
    if improvement > 0:
        print("\n  ✓ Knowledge distillation helped!")
        print(f"    Student learned {improvement:.1f}% better with teacher guidance")
    else:
        print("\n  Note: Results may vary with small datasets")
    
    # Model size comparison
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student_distilled.parameters())
    compression = 100 * (1 - student_params / teacher_params)
    
    print(f"\n  Model Compression:")
    print(f"    Teacher params:  {teacher_params:,}")
    print(f"    Student params:  {student_params:,}")
    print(f"    Reduction:       {compression:.1f}%")
    
    print("\n" + "=" * 60)
    print("Knowledge distillation demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
