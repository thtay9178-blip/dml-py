"""
Curriculum Learning Example for DML-PY.

This example demonstrates how to use curriculum learning
to progressively train on harder examples.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from pydml.trainers import DMLTrainer
from pydml.models.cifar import resnet32
from pydml.strategies.curriculum import DifficultyEstimator, CurriculumStrategy


def main():
    print("=" * 60)
    print("DML-PY - Curriculum Learning Demo")
    print("=" * 60)
    
    # Config
    device = 'cpu'
    num_epochs = 3
    batch_size = 16
    num_samples = 100
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples: {num_samples}")
    
    # Prepare data
    print("\n1. Loading data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Use subset
    indices = torch.randperm(len(train_dataset))[:num_samples]
    train_subset = Subset(train_dataset, indices)
    
    print(f"  Train samples: {len(train_subset)}")
    
    # Create models for difficulty estimation
    print("\n2. Creating models for difficulty estimation...")
    models = [
        resnet32(num_classes=10),
        resnet32(num_classes=10),
    ]
    
    # Train briefly to get model predictions for difficulty estimation
    print("\n3. Brief initial training for difficulty estimation...")
    trainer = DMLTrainer(models=models, device=device)
    
    # Create initial data loader
    init_loader = DataLoader(train_subset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    
    # Train for 1 epoch to get initial predictions
    print("  Training for 1 epoch to estimate difficulties...")
    history = trainer.fit(init_loader, epochs=1, verbose=False)
    print(f"  Initial accuracy: {history['train_acc'][-1]:.2f}%")
    
    # Estimate difficulty using ensemble agreement
    print("\n4. Estimating sample difficulties...")
    difficulties = DifficultyEstimator.by_ensemble_agreement(
        models=models,
        dataset=train_subset,
        device=device
    )
    
    print(f"  Difficulty range: [{difficulties.min():.3f}, {difficulties.max():.3f}]")
    print(f"  Mean difficulty: {difficulties.mean():.3f}")
    
    # Train with curriculum - manually implement easy-to-hard
    print("\n5. Training with curriculum learning...")
    print("  Using easy-to-hard pacing strategy")
    print("-" * 60)
    
    # Sort samples by difficulty (easy first)
    sorted_indices = torch.argsort(torch.tensor(difficulties))
    
    for epoch in range(1, num_epochs + 1):
        # Progressively include more samples
        progress = epoch / num_epochs
        n_samples_to_use = int(num_samples * (0.3 + 0.7 * progress))
        
        # Select easiest n_samples_to_use
        selected_indices = indices[sorted_indices[:n_samples_to_use]]
        
        # Create subset and loader
        curriculum_subset = Subset(train_dataset, selected_indices)
        curriculum_loader = DataLoader(curriculum_subset, batch_size=batch_size,
                                      shuffle=True, num_workers=0)
        
        print(f"\nEpoch {epoch}/{num_epochs} - Training on {n_samples_to_use} samples "
              f"({n_samples_to_use/num_samples*100:.1f}% of data)")
        
        # Train for one epoch
        history = trainer.fit(curriculum_loader, epochs=1, verbose=False)
        
        # Print results
        train_acc = history['train_acc'][-1]
        train_loss = history['train_loss'][-1]
        print(f"  Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
    
    print("-" * 60)
    
    # Compare with regular training
    print("\n6. For comparison: training without curriculum...")
    
    # Create fresh models
    models_baseline = [
        resnet32(num_classes=10),
        resnet32(num_classes=10),
    ]
    
    trainer_baseline = DMLTrainer(models=models_baseline, device=device)
    
    # Train on full dataset
    full_loader = DataLoader(train_subset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    
    history_baseline = trainer_baseline.fit(full_loader, epochs=num_epochs, 
                                           verbose=False)
    
    final_acc_baseline = history_baseline['train_acc'][-1]
    print(f"  Baseline final accuracy: {final_acc_baseline:.2f}%")
    
    # Compare
    final_acc_curriculum = history['train_acc'][-1]
    print(f"  Curriculum final accuracy: {final_acc_curriculum:.2f}%")
    
    improvement = final_acc_curriculum - final_acc_baseline
    print(f"\n  Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("  ✓ Curriculum learning helped!")
    else:
        print("  Note: Results may vary with small datasets/short training")
    
    print("\n" + "=" * 60)
    print("Curriculum learning demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
