"""
Advanced Example: Custom DML Training with Analysis.

This example demonstrates:
1. Custom training configurations
2. Advanced callbacks
3. Robustness analysis
4. Visualization
"""

import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from pydml.utils.data import get_cifar10_loaders
from pydml.utils.logging import ExperimentLogger, print_model_summary
from pydml.analysis.robustness import compare_model_robustness
from pydml.core.callbacks import ModelCheckpoint, TensorBoardLogger
import matplotlib.pyplot as plt


def plot_training_history(history, save_path='training_curves.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 100  # Reduced for demonstration
    batch_size = 128
    num_models = 3
    num_classes = 10
    
    # Initialize experiment logger
    logger = ExperimentLogger('dml_advanced_cifar10')
    
    # Log configuration
    config_dict = {
        'dataset': 'CIFAR-10',
        'num_models': num_models,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'device': device,
    }
    logger.log_config(config_dict)
    
    print("=" * 60)
    print("Advanced DML Training Example")
    print("=" * 60)
    
    # Load data
    print("\nLoading CIFAR-10...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=2,
        val_split=0.1,
        download=True
    )
    
    # Create models
    print(f"\nCreating {num_models} ResNet32 models...")
    models = [resnet32(num_classes=num_classes) for _ in range(num_models)]
    
    # Print model summary
    print_model_summary(models[0], input_size=(1, 3, 32, 32))
    
    # Configure DML
    dml_config = DMLConfig(
        temperature=3.0,
        supervised_weight=1.0,
        mimicry_weight=1.0,
    )
    
    # Create optimizers with weight decay
    optimizers = [
        torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        for model in models
    ]
    
    # Create learning rate schedulers
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        for opt in optimizers
    ]
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=f'{logger.exp_dir}/best_model_{{epoch}}.pt',
            monitor='val_acc',
            mode='max',
            save_best_only=True
        ),
        TensorBoardLogger(log_dir=f'{logger.exp_dir}/tensorboard'),
    ]
    
    # Create trainer
    print("\nInitializing DML trainer...")
    trainer = DMLTrainer(
        models=models,
        config=dml_config,
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
    
    # Log metrics
    for epoch in range(num_epochs):
        if epoch < len(history['train_loss']):
            metrics = {
                'train_loss': history['train_loss'][epoch],
                'train_acc': history['train_acc'][epoch],
            }
            if history['val_loss']:
                metrics['val_loss'] = history['val_loss'][epoch]
                metrics['val_acc'] = history['val_acc'][epoch]
            logger.log_metrics(epoch, metrics)
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_history(history, save_path=f'{logger.exp_dir}/training_curves.png')
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest Accuracy (Average): {test_metrics['val_acc']:.2f}%")
    
    for i in range(num_models):
        acc = test_metrics[f'val_acc_model_{i}']
        print(f"  Model {i}: {acc:.2f}%")
        logger.log_model(trainer.models[i], name=f'model_{i}_final')
    
    # Robustness Analysis
    print("\n" + "=" * 60)
    print("Robustness Analysis")
    print("=" * 60)
    
    noise_levels = [0.001, 0.005, 0.01]
    robustness_results = compare_model_robustness(
        models=trainer.models,
        test_loader=test_loader,
        noise_levels=noise_levels,
        device=device
    )
    
    # Log final notes
    logger.log_text(f"Final test accuracy: {test_metrics['val_acc']:.2f}%")
    logger.log_text(f"Training completed successfully with {num_models} models")
    
    # Finalize experiment
    logger.finalize()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {logger.exp_dir}")


if __name__ == '__main__':
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        main()
    except ImportError:
        print("Warning: matplotlib not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")
        main()
