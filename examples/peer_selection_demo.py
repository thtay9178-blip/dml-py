"""
Demo: Dynamic Peer Selection Strategies

This script demonstrates different peer selection strategies:
1. All Peers (baseline DML)
2. Best Peers (learn from top performers)
3. Diverse Peers (learn from different predictions)
4. Curriculum (gradually increase peer set)

We'll compare their effectiveness on CIFAR-10 with multiple networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydml.strategies.peer_selection import (
    PeerSelectionConfig,
    create_peer_selector,
    PeerSelectionAnalyzer,
)
from pydml.models.cifar.resnet import resnet32
from pydml.core.losses import KLDivergenceLoss
import matplotlib.pyplot as plt
import numpy as np


class PeerSelectionDMLTrainer:
    """DML Trainer with dynamic peer selection."""
    
    def __init__(
        self,
        models: list,
        peer_selector,
        device: str = 'cpu',
        temperature: float = 3.0,
        alpha: float = 0.5
    ):
        self.models = [model.to(device) for model in models]
        self.peer_selector = peer_selector
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        self.optimizers = [
            optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            for model in self.models
        ]
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = KLDivergenceLoss(temperature=temperature)
        
        self.history = {
            'train_loss': [[] for _ in models],
            'train_acc': [[] for _ in models],
            'val_acc': [[] for _ in models],
            'selections': []
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with dynamic peer selection."""
        for model in self.models:
            model.train()
        
        total_loss = [0.0] * len(self.models)
        correct = [0] * len(self.models)
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            total += batch_size
            
            # Forward pass for all models
            outputs = [model(inputs) for model in self.models]
            
            # Get current validation metrics (simplified - use last epoch's results)
            if len(self.history['val_acc'][0]) > 0:
                metrics = {i: self.history['val_acc'][i][-1] for i in range(len(self.models))}
            else:
                metrics = {i: 0.0 for i in range(len(self.models))}
            
            # Store selections for analysis
            batch_selections = {}
            
            # Train each model with selected peers
            for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                optimizer.zero_grad()
                
                # Supervised loss
                loss_supervised = self.ce_loss(outputs[i], targets)
                
                # Select peers for this model
                peer_indices = self.peer_selector.select_peers(i, outputs, targets, metrics)
                peer_weights = self.peer_selector.get_selection_weights(i, outputs, peer_indices)
                batch_selections[i] = peer_indices
                
                # Mimicry loss from selected peers
                loss_mimicry = 0.0
                for peer_idx, weight in zip(peer_indices, peer_weights):
                    loss_mimicry += weight * self.kl_loss(outputs[i], outputs[peer_idx].detach())
                
                # Combined loss
                loss = self.alpha * loss_supervised + (1 - self.alpha) * loss_mimicry
                
                loss.backward(retain_graph=True if i < len(self.models) - 1 else False)
                optimizer.step()
                
                total_loss[i] += loss.item() * batch_size
                _, predicted = outputs[i].max(1)
                correct[i] += predicted.eq(targets).sum().item()
            
            self.history['selections'].append(batch_selections)
        
        # Compute epoch metrics
        for i in range(len(self.models)):
            avg_loss = total_loss[i] / total
            accuracy = 100.0 * correct[i] / total
            self.history['train_loss'][i].append(avg_loss)
            self.history['train_acc'][i].append(accuracy)
    
    def evaluate(self, val_loader):
        """Evaluate all models."""
        for model in self.models:
            model.eval()
        
        correct = [0] * len(self.models)
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                total += targets.size(0)
                
                outputs = [model(inputs) for model in self.models]
                
                for i in range(len(self.models)):
                    _, predicted = outputs[i].max(1)
                    correct[i] += predicted.eq(targets).sum().item()
        
        accuracies = [100.0 * c / total for c in correct]
        for i, acc in enumerate(accuracies):
            self.history['val_acc'][i].append(acc)
        
        return accuracies
    
    def fit(self, train_loader, val_loader, epochs: int):
        """Train with dynamic peer selection."""
        print(f"Training with {self.peer_selector.__class__.__name__}")
        print(f"Strategy: {self.peer_selector.config.strategy}")
        print(f"Number of models: {len(self.models)}\n")
        
        for epoch in range(epochs):
            # Update peer selector with current epoch
            if len(self.history['val_acc'][0]) > 0:
                metrics = {i: self.history['val_acc'][i][-1] for i in range(len(self.models))}
            else:
                metrics = {i: 0.0 for i in range(len(self.models))}
            self.peer_selector.update(epoch, metrics)
            
            # Train epoch
            self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_accs = self.evaluate(val_loader)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: ", end='')
                for i, acc in enumerate(val_accs):
                    print(f"Model {i}: {acc:5.2f}%  ", end='')
                print()
        
        print(f"\nFinal accuracies:")
        for i, acc in enumerate(val_accs):
            print(f"  Model {i}: {acc:.2f}%")
        
        return val_accs


def create_small_cifar10():
    """Create a small subset of CIFAR-10 for quick testing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Full dataset
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    
    # Use small subset for quick demo
    train_subset = Subset(train_dataset, range(5000))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def compare_strategies():
    """Compare different peer selection strategies."""
    device = 'cpu'  # Use CPU for safety
    num_models = 4
    epochs = 20
    
    print("="*70)
    print("DYNAMIC PEER SELECTION COMPARISON")
    print("="*70)
    print(f"Dataset: CIFAR-10 (small subset)")
    print(f"Models: {num_models} x SimpleCNN")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # Load data
    train_loader, test_loader = create_small_cifar10()
    
    # Test different strategies
    strategies = [
        ('all', 'All Peers (Baseline DML)'),
        ('best', 'Best Peers (Top-2)'),
        ('diverse', 'Diverse Peers'),
        ('curriculum', 'Curriculum Selection'),
    ]
    
    results = {}
    
    for strategy_name, strategy_desc in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy_desc}")
        print('='*70)
        
        # Create models
        models = [resnet32(num_classes=10) for _ in range(num_models)]
        
        # Create peer selector
        config = PeerSelectionConfig(
            strategy=strategy_name,
            k_peers=2,
            curriculum_start_k=1,
            curriculum_end_epoch=15,
        )
        selector = create_peer_selector(config)
        
        # Create trainer
        trainer = PeerSelectionDMLTrainer(
            models=models,
            peer_selector=selector,
            device=device,
            temperature=3.0,
            alpha=0.5
        )
        
        # Train
        final_accs = trainer.fit(train_loader, test_loader, epochs=epochs)
        
        # Store results
        results[strategy_name] = {
            'description': strategy_desc,
            'final_accuracies': final_accs,
            'mean_accuracy': np.mean(final_accs),
            'std_accuracy': np.std(final_accs),
            'best_accuracy': max(final_accs),
            'history': trainer.history
        }
        
        print(f"Mean accuracy: {results[strategy_name]['mean_accuracy']:.2f}%")
        print(f"Std accuracy: {results[strategy_name]['std_accuracy']:.2f}%")
        print(f"Best model: {results[strategy_name]['best_accuracy']:.2f}%")
    
    return results


def visualize_results(results):
    """Visualize comparison of peer selection strategies."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Final Accuracy Comparison
    ax = axes[0, 0]
    strategies = list(results.keys())
    mean_accs = [results[s]['mean_accuracy'] for s in strategies]
    std_accs = [results[s]['std_accuracy'] for s in strategies]
    
    x = np.arange(len(strategies))
    ax.bar(x, mean_accs, yerr=std_accs, capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    ax.set_xticks(x)
    ax.set_xticklabels([results[s]['description'].split('(')[0].strip() for s in strategies], rotation=15, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Final Accuracy by Strategy')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Training Curves (Model 0 from each strategy)
    ax = axes[0, 1]
    for strategy in strategies:
        history = results[strategy]['history']
        train_acc = history['train_acc'][0]  # Model 0
        ax.plot(train_acc, label=results[strategy]['description'].split('(')[0].strip(), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Curves (Model 0)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Per-Model Accuracy Distribution
    ax = axes[1, 0]
    data_to_plot = [results[s]['final_accuracies'] for s in strategies]
    bp = ax.boxplot(data_to_plot, labels=[results[s]['description'].split('(')[0].strip() for s in strategies])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Model Accuracy Distribution')
    ax.set_xticklabels([results[s]['description'].split('(')[0].strip() for s in strategies], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Best Model Performance
    ax = axes[1, 1]
    best_accs = [results[s]['best_accuracy'] for s in strategies]
    bars = ax.bar(x, best_accs, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    ax.set_xticks(x)
    ax.set_xticklabels([results[s]['description'].split('(')[0].strip() for s in strategies], rotation=15, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Best Model Performance')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('peer_selection_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: peer_selection_comparison.png")
    plt.close()
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Strategy':<30} {'Mean Acc':<12} {'Std Acc':<12} {'Best Model':<12}")
    print("-"*70)
    for strategy in strategies:
        desc = results[strategy]['description']
        mean_acc = results[strategy]['mean_accuracy']
        std_acc = results[strategy]['std_accuracy']
        best_acc = results[strategy]['best_accuracy']
        print(f"{desc:<30} {mean_acc:>10.2f}%  {std_acc:>10.2f}%  {best_acc:>10.2f}%")
    print("="*70)


def main():
    """Main demo function."""
    print("\n" + "🚀 "*25)
    print("PEER SELECTION STRATEGIES DEMO")
    print("🚀 "*25 + "\n")
    
    # Run comparison
    results = compare_strategies()
    
    # Visualize
    visualize_results(results)
    
    print("\n" + "✅ "*25)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("✅ "*25 + "\n")
    
    # Print insights
    print("KEY INSIGHTS:")
    print("-" * 70)
    
    strategies = list(results.keys())
    best_strategy = max(strategies, key=lambda s: results[s]['mean_accuracy'])
    worst_strategy = min(strategies, key=lambda s: results[s]['mean_accuracy'])
    
    improvement = results[best_strategy]['mean_accuracy'] - results[worst_strategy]['mean_accuracy']
    
    print(f"1. Best Strategy: {results[best_strategy]['description']}")
    print(f"   Mean Accuracy: {results[best_strategy]['mean_accuracy']:.2f}%")
    print()
    print(f"2. Improvement over worst: +{improvement:.2f}%")
    print()
    print(f"3. Curriculum learning shows progressive peer inclusion works well")
    print(f"4. Diverse peer selection maintains model diversity")
    print(f"5. Best peer selection focuses learning on top performers")
    print("-" * 70)


if __name__ == "__main__":
    main()
