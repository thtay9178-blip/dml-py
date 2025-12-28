"""
Quick Start Example for DML-PY.

This example demonstrates the basic usage of DML-PY for training
multiple neural networks collaboratively using Deep Mutual Learning.
"""

import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32, mobilenet_v2
from pydml.utils.data import get_cifar100_loaders


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 200
    batch_size = 128
    num_classes = 100
    
    print("=" * 60)
    print("DML-PY Quick Start Example - CIFAR-100")
    print("=" * 60)
    
    # Load CIFAR-100 data
    print("\nLoading CIFAR-100 dataset...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size,
        num_workers=4,
        val_split=0.1,
        download=True
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create multiple models to train collaboratively
    print("\nCreating models...")
    models = [
        resnet32(num_classes=num_classes),
        resnet32(num_classes=num_classes),
    ]
    print(f"Created {len(models)} models for collaborative training")
    
    # Configure DML
    config = DMLConfig(
        temperature=3.0,
        supervised_weight=1.0,
        mimicry_weight=1.0,
    )
    
    # Create optimizers (one per model)
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        for model in models
    ]
    
    # Create learning rate schedulers
    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)
        for opt in optimizers
    ]
    
    # Create DML trainer
    print("\nInitializing DML trainer...")
    trainer = DMLTrainer(
        models=models,
        config=config,
        device=device,
        optimizers=optimizers,
        schedulers=schedulers,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Training for {num_epochs} epochs on {device}")
    print("=" * 60)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        verbose=True
    )
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nTest Accuracy (Average): {test_metrics['val_acc']:.2f}%")
    for i in range(len(models)):
        print(f"Model {i} Test Accuracy: {test_metrics[f'val_acc_model_{i}']:.2f}%")
    
    # Save final checkpoint
    checkpoint_path = 'dml_cifar100_final.pt'
    trainer.save_checkpoint(checkpoint_path)
    print(f"\nFinal checkpoint saved to {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
