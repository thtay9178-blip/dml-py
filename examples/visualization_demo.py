"""
Complete demo with visualization for DML-PY.

This example demonstrates:
1. Training multiple models with DML
2. Visualizing training progress
3. Analyzing model agreement
4. Creating comprehensive dashboards
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from pydml.trainers import DMLTrainer
from pydml.models.cifar import resnet32, mobilenet_v2
from pydml.analysis import (
    plot_training_history,
    plot_model_comparison,
    create_training_dashboard,
    plot_agreement_matrix
)


def main():
    print("=" * 60)
    print("DML-PY - Visualization Demo")
    print("=" * 60)
    
    # Config
    device = 'cpu'  # Use CPU for safety
    num_epochs = 3
    batch_size = 16
    num_samples = 100  # Small subset for demo
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Samples: {num_samples}")
    
    # Prepare tiny dataset
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
    
    # Use subset for speed
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
    
    # Create models
    print("\n2. Creating models...")
    models = [
        resnet32(num_classes=10),
        mobilenet_v2(num_classes=10),
    ]
    
    for i, model in enumerate(models):
        print(f"  Model {i}: {model.__class__.__name__}")
    
    # Create trainer
    print("\n3. Setting up trainer...")
    trainer = DMLTrainer(
        models=models,
        device=device,
        config={
            'temperature': 3.0,
            'supervised_weight': 1.0,
            'mimicry_weight': 0.5,
        }
    )
    
    # Train
    print("\n4. Training...")
    print("-" * 60)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=num_epochs,
        verbose=True
    )
    print("-" * 60)
    
    # Final evaluation
    print("\n5. Final evaluation...")
    results = trainer.evaluate(test_loader)
    print(f"  Average accuracy: {results['val_acc']:.2f}%")
    
    for i in range(len(models)):
        model_acc = results[f'val_acc_model_{i}']
        print(f"  Model {i} accuracy: {model_acc:.2f}%")
    
    # Visualizations
    print("\n6. Creating visualizations...")
    
    # Basic training history
    print("  a) Plotting training history...")
    plot_training_history(history, save_path='training_history.png', show=False)
    
    # Model comparison
    print("  b) Plotting model comparison...")
    plot_model_comparison(history, len(models), metric='val_acc',
                         save_path='model_comparison.png', show=False)
    
    # Comprehensive dashboard
    print("  c) Creating training dashboard...")
    create_training_dashboard(history, len(models), 
                             save_path='training_dashboard.png')
    
    # Agreement matrix
    print("  d) Computing agreement matrix...")
    agreement = plot_agreement_matrix(models, test_loader, device=device,
                                     save_path='agreement_matrix.png', show=False)
    
    avg_agreement = (agreement.sum() - agreement.trace()) / (len(models) * (len(models) - 1))
    print(f"      Average inter-model agreement: {avg_agreement:.2f}%")
    
    print("\n" + "=" * 60)
    print("Visualization demo completed!")
    print("Generated files:")
    print("  - training_history.png")
    print("  - model_comparison.png")
    print("  - training_dashboard.png")
    print("  - agreement_matrix.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
