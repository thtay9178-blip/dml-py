"""
Temperature Scaling Demo - Adaptive Temperature Strategies
==========================================================

This demo compares different temperature scheduling strategies for
knowledge distillation and their impact on model performance.

Temperature controls the softness of probability distributions:
- High T: Soft, smooth distributions (exploration)
- Low T: Sharp, peaked distributions (exploitation)

We compare 6 strategies:
1. Constant: Fixed temperature (baseline)
2. Linear: Linear annealing from high to low
3. Exponential: Exponential decay
4. Cosine: Smooth cosine annealing
5. Cyclical: Oscillating temperature
6. Adaptive: Metric-based adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time

from pydml.models.cifar import resnet32
from pydml.core.losses import KLDivergenceLoss
from pydml.strategies.temperature_scaling import (
    TemperatureSchedulerConfig,
    create_temperature_scheduler,
    TemperatureAnalyzer,
)


def get_cifar10_loaders(batch_size=128, subset_size=5000):
    """Load CIFAR-10 dataset with subset for quick testing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Use subset for faster training
    train_indices = np.random.choice(len(trainset), subset_size, replace=False)
    test_indices = np.random.choice(len(testset), min(1000, len(testset)), replace=False)
    
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_with_temperature(
    model,
    teacher,
    train_loader,
    test_loader,
    scheduler,
    epochs=20,
    device='cpu'
):
    """
    Train model with knowledge distillation using adaptive temperature.
    
    Args:
        model: Student model to train
        teacher: Pre-trained teacher model
        train_loader: Training data loader
        test_loader: Test data loader
        scheduler: Temperature scheduler
        epochs: Number of epochs
        device: Device to train on
    
    Returns:
        Dictionary with training history
    """
    model.to(device)
    teacher.to(device)
    teacher.eval()  # Teacher in eval mode
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = KLDivergenceLoss(reduction='batchmean')
    
    history = {
        'train_loss': [],
        'test_acc': [],
        'temperatures': [],
        'train_time': []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        # Compute current temperature
        metrics = {}
        if epoch > 0:
            # Use previous epoch metrics
            metrics = {
                'loss': history['train_loss'][-1] if history['train_loss'] else 1.0,
                'accuracy': history['test_acc'][-1] if history['test_acc'] else 50.0,
            }
        
        current_temp = scheduler.step(epoch, metrics)
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Student predictions
            student_logits = model(inputs)
            
            # Teacher predictions (no grad)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            # Combined loss: CE + KD with temperature
            loss_ce = ce_loss(student_logits, targets)
            loss_kd = kd_loss(student_logits, teacher_logits, temperature=current_temp)
            
            # Weight KD loss
            alpha = 0.5
            loss = alpha * loss_ce + (1 - alpha) * loss_kd
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluate
        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['test_acc'].append(test_acc)
        history['temperatures'].append(current_temp)
        history['train_time'].append(epoch_time)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: Loss={epoch_loss/len(train_loader):.4f}, "
                  f"Acc={test_acc:.2f}%, T={current_temp:.2f}")
    
    return history


def evaluate(model, test_loader, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total


def compare_temperature_strategies():
    """Compare different temperature scheduling strategies."""
    print("=" * 80)
    print("Temperature Scaling Comparison Demo")
    print("=" * 80)
    
    device = 'cpu'  # Use CPU for memory safety
    epochs = 20
    
    # Load data
    print("\n1. Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, subset_size=5000)
    print("   ✓ Data loaded (5000 train samples, 1000 test samples)")
    
    # Create teacher model (pre-trained)
    print("\n2. Training teacher model...")
    teacher = resnet32(num_classes=10)
    teacher.to(device)
    
    # Quick teacher training (simplified)
    optimizer = optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):  # Quick teacher training
        teacher.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    teacher_acc = evaluate(teacher, test_loader, device)
    print(f"   ✓ Teacher trained: {teacher_acc:.2f}% accuracy")
    
    # Define temperature strategies
    strategies = {
        'Constant (T=4)': TemperatureSchedulerConfig(
            strategy='constant',
            initial_temp=4.0
        ),
        'Linear Annealing': TemperatureSchedulerConfig(
            strategy='linear',
            initial_temp=8.0,
            final_temp=1.0,
            warmup_epochs=3,
            total_epochs=epochs
        ),
        'Exponential Decay': TemperatureSchedulerConfig(
            strategy='exponential',
            initial_temp=8.0,
            final_temp=1.0,
            warmup_epochs=3,
            total_epochs=epochs
        ),
        'Cosine Annealing': TemperatureSchedulerConfig(
            strategy='cosine',
            initial_temp=8.0,
            final_temp=1.0,
            warmup_epochs=3,
            total_epochs=epochs
        ),
        'Cyclical (2↔8)': TemperatureSchedulerConfig(
            strategy='cyclical',
            min_temp=2.0,
            max_temp=8.0,
            cycle_length=5
        ),
        'Adaptive': TemperatureSchedulerConfig(
            strategy='adaptive',
            initial_temp=4.0,
            min_temp=1.0,
            max_temp=10.0,
            adaptation_rate=0.2
        ),
    }
    
    # Train with each strategy
    print(f"\n3. Training students with different temperature strategies ({epochs} epochs each)...")
    print("=" * 80)
    
    results = {}
    
    for name, config in strategies.items():
        print(f"\n[{name}]")
        
        # Create fresh student model
        student = resnet32(num_classes=10)
        scheduler = create_temperature_scheduler(config)
        
        # Train
        history = train_with_temperature(
            student, teacher, train_loader, test_loader,
            scheduler, epochs, device
        )
        
        results[name] = {
            'history': history,
            'final_acc': history['test_acc'][-1],
            'best_acc': max(history['test_acc']),
            'mean_temp': np.mean(history['temperatures']),
            'total_time': sum(history['train_time'])
        }
        
        print(f"  Final: Acc={results[name]['final_acc']:.2f}%, "
              f"Best={results[name]['best_acc']:.2f}%, "
              f"Avg T={results[name]['mean_temp']:.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Final Acc':<12} {'Best Acc':<12} {'Avg Temp':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for name, data in results.items():
        print(f"{name:<25} {data['final_acc']:>10.2f}%  {data['best_acc']:>10.2f}%  "
              f"{data['mean_temp']:>10.2f}  {data['total_time']:>8.1f}")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['best_acc'])
    print("-" * 80)
    print(f"🏆 Best Strategy: {best_strategy[0]} ({best_strategy[1]['best_acc']:.2f}% accuracy)")
    print("=" * 80)
    
    # Visualize temperature schedules
    print("\n4. Generating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Plot 1: Temperature schedules
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (name, data) in enumerate(results.items()):
            ax = axes[idx]
            
            # Plot temperature over time
            temps = data['history']['temperatures']
            ax.plot(temps, 'b-', linewidth=2, label='Temperature')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Temperature', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            ax.set_title(name, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Plot accuracy on secondary axis
            ax2 = ax.twinx()
            accs = data['history']['test_acc']
            ax2.plot(accs, 'r--', linewidth=2, label='Accuracy')
            ax2.set_ylabel('Test Accuracy (%)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('temperature_strategies_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved: temperature_strategies_comparison.png")
        plt.close()
        
        # Plot 2: Final accuracy comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(results.keys())
        final_accs = [results[name]['final_acc'] for name in names]
        best_accs = [results[name]['best_acc'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, final_accs, width, label='Final Accuracy', alpha=0.8)
        ax.bar(x + width/2, best_accs, width, label='Best Accuracy', alpha=0.8)
        
        ax.set_xlabel('Temperature Strategy')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Temperature Strategy Performance Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temperature_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved: temperature_accuracy_comparison.png")
        plt.close()
        
    except ImportError:
        print("   ⚠ Matplotlib not available for visualization")
    
    print("\n" + "=" * 80)
    print("✓ Temperature scaling demo completed successfully!")
    print("=" * 80)
    
    # Key insights
    print("\n📊 KEY INSIGHTS:")
    print("-" * 80)
    
    improvements = {name: data['best_acc'] - results['Constant (T=4)']['best_acc'] 
                   for name, data in results.items() if name != 'Constant (T=4)'}
    
    if improvements:
        best_improvement = max(improvements.items(), key=lambda x: x[1])
        print(f"• Best improvement: {best_improvement[0]} (+{best_improvement[1]:.2f}% over constant)")
    
    print(f"• Teacher accuracy: {teacher_acc:.2f}%")
    print(f"• Adaptive temperature responds to training dynamics")
    print(f"• Cyclical temperature alternates exploration/exploitation")
    print(f"• Annealing schedules progressively sharpen distributions")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comparison
    compare_temperature_strategies()
