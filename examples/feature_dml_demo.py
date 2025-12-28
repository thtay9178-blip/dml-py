"""
Feature-Based Deep Mutual Learning Example.

This example demonstrates how feature-level matching improves
knowledge transfer compared to logit-only matching.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from pydml.trainers import DMLTrainer, FeatureDMLTrainer, FeatureDMLConfig
from pydml.models.cifar import resnet32


def get_resnet_feature_layers():
    """Get layer names for feature extraction from ResNet."""
    # Extract from intermediate layers
    return ['layer1', 'layer2', 'layer3']


def main():
    print("=" * 60)
    print("DML-PY - Feature-Based Mutual Learning Demo")
    print("=" * 60)
    
    # Config
    device = 'cpu'
    num_epochs = 5
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
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use subset
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
    
    # Experiment 1: Standard DML (logit-only)
    print("\n2. Training with Standard DML (logit-only matching)...")
    print("-" * 60)
    
    models_standard = [
        resnet32(num_classes=10),
        resnet32(num_classes=10),
    ]
    
    trainer_standard = DMLTrainer(
        models=models_standard,
        device=device
    )
    
    history_standard = trainer_standard.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=num_epochs,
        verbose=False
    )
    
    final_acc_standard = history_standard['val_acc'][-1]
    print(f"  Final validation accuracy: {final_acc_standard:.2f}%")
    print("-" * 60)
    
    # Experiment 2: Feature-Based DML
    print("\n3. Training with Feature-Based DML (logit + feature matching)...")
    print("-" * 60)
    
    models_feature = [
        resnet32(num_classes=10),
        resnet32(num_classes=10),
    ]
    
    # Define which layers to extract features from
    feature_layers = get_resnet_feature_layers()
    print(f"  Extracting features from layers: {feature_layers}")
    
    config = FeatureDMLConfig(
        temperature=3.0,
        supervised_weight=1.0,
        logit_mimicry_weight=1.0,
        feature_mimicry_weight=0.5,  # Additional feature matching
        feature_loss_type='mse'
    )
    
    trainer_feature = FeatureDMLTrainer(
        models=models_feature,
        feature_layers_list=[feature_layers, feature_layers],
        config=config,
        device=device
    )
    
    history_feature = trainer_feature.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=num_epochs,
        verbose=False
    )
    
    final_acc_feature = history_feature['val_acc'][-1]
    print(f"\n  Final validation accuracy: {final_acc_feature:.2f}%")
    print("-" * 60)
    
    # Compare results
    print("\n4. Results Comparison")
    print("=" * 60)
    print(f"  Standard DML (logit-only):    {final_acc_standard:.2f}%")
    print(f"  Feature-Based DML:            {final_acc_feature:.2f}%")
    print("-" * 60)
    
    improvement = final_acc_feature - final_acc_standard
    print(f"\n  Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("\n  ✓ Feature-level matching helped!")
        print("    Models learned better internal representations")
    elif improvement < 0:
        print("\n  Note: Feature matching added complexity")
        print("    May need more data or tuning for benefits")
    else:
        print("\n  Results are similar - more training may help")
    
    # Show training curves
    print("\n5. Training Curves")
    print("-" * 60)
    print("  Standard DML:")
    for i, (loss, acc) in enumerate(zip(history_standard['train_loss'], 
                                       history_standard['train_acc'])):
        val_acc = history_standard['val_acc'][i] if history_standard['val_acc'] else 0
        print(f"    Epoch {i+1}: Loss={loss:.4f}, Train Acc={acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    print("\n  Feature-Based DML:")
    for i, (loss, acc) in enumerate(zip(history_feature['train_loss'], 
                                       history_feature['train_acc'])):
        val_acc = history_feature['val_acc'][i] if history_feature['val_acc'] else 0
        print(f"    Epoch {i+1}: Loss={loss:.4f}, Train Acc={acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Feature-Based DML demo completed!")
    print("\nKey Insights:")
    print("  - Feature matching encourages similar internal representations")
    print("  - Can improve knowledge transfer beyond logit matching")
    print("  - Particularly useful when models have similar architectures")
    print("=" * 60)


if __name__ == '__main__':
    main()
