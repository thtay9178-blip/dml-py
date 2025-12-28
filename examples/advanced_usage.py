"""
Advanced DML Example with Custom Configuration.

This example demonstrates:
1. Training heterogeneous networks (different architectures)
2. Using callbacks for monitoring
3. Custom learning rate schedules
4. Ensemble evaluation
"""

import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32, mobilenet_v2, wrn_28_10
from pydml.core.callbacks import ModelCheckpoint, TensorBoardLogger
from pydml.utils.data import get_cifar100_loaders
from pydml.utils.metrics import ensemble_accuracy, diversity_score, agreement_matrix
import numpy as np


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 200
    batch_size = 128
    num_classes = 100
    
    print("="*60)
    print("Advanced DML Example - Heterogeneous Networks")
    print("="*60)
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size,
        num_workers=2,
        val_split=0.1,
        download=True
    )
    
    # Create heterogeneous models (different architectures)
    print("\nCreating heterogeneous models...")
    models = [
        resnet32(num_classes=num_classes),
        mobilenet_v2(num_classes=num_classes, width_mult=1.0),
        resnet32(num_classes=num_classes),  # Another ResNet for comparison
    ]
    
    model_names = ['ResNet-32', 'MobileNetV2', 'ResNet-32-B']
    print(f"Models: {', '.join(model_names)}")
    
    # Count parameters
    for name, model in zip(model_names, models):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {num_params:,} parameters")
    
    # Configure DML
    config = DMLConfig(
        temperature=3.0,
        supervised_weight=1.0,
        mimicry_weight=1.0,
    )
    
    # Create optimizers with different learning rates for different architectures
    optimizers = [
        torch.optim.SGD(models[0].parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
        torch.optim.SGD(models[1].parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),  # Lower LR for MobileNet
        torch.optim.SGD(models[2].parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
    ]
    
    # Create schedulers
    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)
        for opt in optimizers
    ]
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            filepath='checkpoints/dml_heterogeneous_epoch_{epoch}.pt',
            monitor='val_acc',
            mode='max',
            save_best_only=True,
        ),
        TensorBoardLogger(log_dir='runs/dml_heterogeneous'),
    ]
    
    # Create trainer
    print("\nInitializing DML trainer...")
    trainer = DMLTrainer(
        models=models,
        config=config,
        device=device,
        optimizers=optimizers,
        schedulers=schedulers,
        callbacks=callbacks,
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        verbose=True
    )
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nIndividual Model Performance:")
    for i, name in enumerate(model_names):
        acc = test_metrics[f'val_acc_model_{i}']
        print(f"  {name}: {acc:.2f}%")
    
    # Ensemble evaluation
    print("\nEvaluating ensemble...")
    trainer.eval()
    
    all_outputs = [[] for _ in range(len(models))]
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            for i, model in enumerate(models):
                outputs = model(inputs)
                all_outputs[i].append(outputs.cpu())
            all_targets.append(targets)
    
    # Concatenate all batches
    all_outputs = [torch.cat(outputs) for outputs in all_outputs]
    all_targets = torch.cat(all_targets)
    
    # Compute ensemble metrics
    ensemble_acc = ensemble_accuracy(all_outputs, all_targets)
    diversity = diversity_score(all_outputs)
    agreement_mat = agreement_matrix(all_outputs)
    
    print(f"\nEnsemble Performance:")
    print(f"  Ensemble Accuracy: {ensemble_acc:.2f}%")
    print(f"  Diversity Score: {diversity:.4f}")
    print(f"\nAgreement Matrix:")
    print("          ", end="")
    for name in model_names:
        print(f"{name[:12]:>12}", end=" ")
    print()
    for i, name in enumerate(model_names):
        print(f"{name[:12]:>12}", end=" ")
        for j in range(len(models)):
            print(f"{agreement_mat[i, j]:12.4f}", end=" ")
        print()
    
    # Improvement analysis
    avg_individual = sum([test_metrics[f'val_acc_model_{i}'] for i in range(len(models))]) / len(models)
    improvement = ensemble_acc - avg_individual
    
    print(f"\nAnalysis:")
    print(f"  Average Individual Accuracy: {avg_individual:.2f}%")
    print(f"  Ensemble Accuracy: {ensemble_acc:.2f}%")
    print(f"  Improvement: +{improvement:.2f}%")
    
    # Save final models
    trainer.save_checkpoint('dml_heterogeneous_final.pt')
    print(f"\nFinal checkpoint saved")


if __name__ == '__main__':
    main()
