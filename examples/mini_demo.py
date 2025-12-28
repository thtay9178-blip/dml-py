"""
Mini Demo - DML-PY with Minimal Memory Usage

This demo trains on a tiny subset of data to show the workflow
without consuming much memory.
"""

import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from torch.utils.data import DataLoader, TensorDataset

def create_tiny_dataset(num_samples=50, num_classes=10):
    """Create a tiny synthetic dataset."""
    X = torch.randn(num_samples, 3, 32, 32)
    Y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, Y)

def main():
    print("=" * 60)
    print("DML-PY Mini Demo (Memory-Efficient)")
    print("=" * 60)
    
    # Configuration - very light
    device = 'cpu'  # Use CPU to avoid GPU memory
    num_epochs = 5  # Just 5 epochs for demo
    batch_size = 8  # Small batch
    num_classes = 10
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    
    # Create tiny datasets
    print(f"\nCreating tiny synthetic dataset...")
    train_dataset = create_tiny_dataset(50, num_classes)
    val_dataset = create_tiny_dataset(20, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create 2 small models
    print(f"\nCreating 2 ResNet32 models...")
    models = [
        resnet32(num_classes=num_classes),
        resnet32(num_classes=num_classes)
    ]
    
    # Configure DML
    config = DMLConfig(
        temperature=3.0,
        supervised_weight=1.0,
        mimicry_weight=1.0
    )
    
    # Setup optimizers
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for model in models
    ]
    
    # Create trainer
    print(f"\nInitializing DML trainer...")
    trainer = DMLTrainer(
        models=models,
        config=config,
        device=device,
        optimizers=optimizers
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting Training")
    print('='*60)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        verbose=True
    )
    
    # Results
    print(f"\n{'='*60}")
    print("Results")
    print('='*60)
    
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    
    print(f"\nPer-Model Performance:")
    val_metrics = trainer.evaluate(val_loader)
    for i in range(len(models)):
        acc = val_metrics[f'val_acc_model_{i}']
        print(f"  Model {i}: {acc:.2f}%")
    
    print(f"\n{'='*60}")
    print("✅ Demo Complete!")
    print('='*60)
    print("\nKey Observations:")
    print("  ✓ Both models trained collaboratively")
    print("  ✓ Learning from each other via mimicry loss")
    print("  ✓ Memory usage kept minimal")
    print(f"  ✓ Training completed in {num_epochs} epochs")
    
    print("\nNext Steps:")
    print("  1. Try with real data: python examples/quick_start.py")
    print("  2. Run full benchmark: python examples/cifar100_benchmark.py")
    print("  3. Analyze robustness: see examples/advanced_training.py")

if __name__ == '__main__':
    main()
