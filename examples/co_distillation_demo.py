"""
Co-Distillation Example for DML-PY.

This example demonstrates the hybrid approach of co-distillation:
combining teacher-student distillation with peer mutual learning.

We compare three approaches:
1. Single student learning from teacher (standard distillation)
2. Multiple students with mutual learning (DML)
3. Multiple students with teacher + mutual learning (Co-Distillation)
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from pydml.trainers import (
    DistillationTrainer, 
    DMLTrainer,
    CoDistillationTrainer,
    CoDistillationConfig
)
from pydml.models.cifar import resnet32, mobilenet_v2


def train_teacher(device, train_loader, test_loader, epochs=10):
    """Train a teacher model."""
    print("Training teacher model...")
    teacher = resnet32(num_classes=10).to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        teacher.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0 or epoch == epochs:
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
            print(f"  Epoch {epoch}/{epochs} - Accuracy: {acc:.2f}%")
    
    return teacher


def main():
    print("=" * 70)
    print("DML-PY - Co-Distillation Demo")
    print("=" * 70)
    
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
    
    # Step 1: Train teacher
    print("\n2. Training teacher model...")
    print("-" * 70)
    teacher = train_teacher(device, train_loader, test_loader, epochs=num_epochs)
    
    # Evaluate teacher
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
    print(f"\n✓ Teacher training complete! Accuracy: {teacher_acc:.2f}%")
    print("-" * 70)
    
    # Approach 1: Single Student with Standard Distillation
    print("\n3. Approach 1: Single student with standard distillation...")
    print("-" * 70)
    
    single_student = mobilenet_v2(num_classes=10)
    
    single_trainer = DistillationTrainer(
        teacher=teacher,
        student=single_student,
        device=device
    )
    
    single_history = single_trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=num_epochs
    )
    
    single_acc = single_history['val_acc'][-1]
    print(f"\n✓ Single student accuracy: {single_acc:.2f}%")
    print("-" * 70)
    
    # Approach 2: Multiple Students with DML (no teacher)
    print("\n4. Approach 2: Multiple students with mutual learning (DML)...")
    print("-" * 70)
    
    dml_students = [
        mobilenet_v2(num_classes=10),
        mobilenet_v2(num_classes=10),
    ]
    
    dml_trainer = DMLTrainer(
        models=dml_students,
        device=device
    )
    
    dml_history = dml_trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=num_epochs,
        verbose=False
    )
    
    dml_acc = dml_history['val_acc'][-1]
    print(f"\n✓ DML average accuracy: {dml_acc:.2f}%")
    print("-" * 70)
    
    # Approach 3: Co-Distillation (teacher + mutual learning)
    print("\n5. Approach 3: Co-distillation (teacher + mutual learning)...")
    print("-" * 70)
    
    codistill_students = [
        mobilenet_v2(num_classes=10),
        mobilenet_v2(num_classes=10),
    ]
    
    config = CoDistillationConfig(
        temperature=4.0,
        teacher_weight=0.5,   # 50% teacher guidance
        peer_weight=0.3,      # 30% peer learning
        supervised_weight=0.2  # 20% ground truth
    )
    
    codistill_trainer = CoDistillationTrainer(
        teacher=teacher,
        students=codistill_students,
        config=config,
        device=device
    )
    
    codistill_history = codistill_trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=num_epochs,
        verbose=True
    )
    
    codistill_acc = codistill_history['val_acc'][-1]
    print(f"\n✓ Co-distillation average accuracy: {codistill_acc:.2f}%")
    print("-" * 70)
    
    # Compare all approaches
    print("\n6. Results Comparison")
    print("=" * 70)
    print(f"  Teacher (ResNet32):                      {teacher_acc:.2f}%")
    print("-" * 70)
    print(f"  Approach 1 - Single Student:             {single_acc:.2f}%")
    print(f"  Approach 2 - DML (no teacher):           {dml_acc:.2f}%")
    print(f"  Approach 3 - Co-Distillation:            {codistill_acc:.2f}%")
    print("-" * 70)
    
    # Improvements
    codistill_vs_single = codistill_acc - single_acc
    codistill_vs_dml = codistill_acc - dml_acc
    
    print(f"\n  Co-Distillation vs Single Student:       {codistill_vs_single:+.2f}%")
    print(f"  Co-Distillation vs DML:                  {codistill_vs_dml:+.2f}%")
    
    # Best approach
    best_acc = max(single_acc, dml_acc, codistill_acc)
    
    if best_acc == codistill_acc and codistill_acc > single_acc:
        print("\n  ✓ Co-distillation is the winner!")
        print("    Best of both worlds: teacher guidance + peer diversity")
    elif best_acc == single_acc:
        print("\n  Single student performs best")
        print("    May need more students or different weights for co-distillation")
    else:
        print("\n  DML performs best")
        print("    Mutual learning alone is effective")
    
    # Show per-student performance in co-distillation
    print("\n7. Individual Student Performance (Co-Distillation)")
    print("-" * 70)
    for i in range(len(codistill_students)):
        student_acc = codistill_history[f'val_acc_model_{i}'][-1]
        gap_to_teacher = teacher_acc - student_acc
        print(f"  Student {i}: {student_acc:.2f}% (gap to teacher: {gap_to_teacher:.2f}%)")
    
    print("\n" + "=" * 70)
    print("Co-Distillation demo completed!")
    print("\nKey Insights:")
    print("  1. Co-distillation combines teacher guidance with peer learning")
    print("  2. Students benefit from both expert knowledge and peer diversity")
    print("  3. Can achieve better performance than either approach alone")
    print("  4. Particularly useful when training multiple compact models")
    print("  5. Balancing weights (teacher/peer/supervised) is important")
    print("=" * 70)


if __name__ == '__main__':
    main()
