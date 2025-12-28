"""
CIFAR-100 Benchmark Script.

This script reproduces the experiments from the Deep Mutual Learning paper on CIFAR-100.
It compares:
1. Independent training (baseline)
2. Deep Mutual Learning (2 networks)
3. Deep Mutual Learning (3+ networks)
"""

import torch
import torch.nn as nn
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32, mobilenet_v2, wrn_28_10
from pydml.utils.data import get_cifar100_loaders
import time
import json


def train_independent(model, train_loader, val_loader, device, epochs=200):
    """Train a single model independently (baseline)."""
    print("\n" + "="*60)
    print("Training Independent Model (Baseline)")
    print("="*60)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs} - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")
    
    return best_acc, history


def train_dml(models, train_loader, val_loader, device, epochs=200):
    """Train multiple models with Deep Mutual Learning."""
    print("\n" + "="*60)
    print(f"Training {len(models)} Models with Deep Mutual Learning")
    print("="*60)
    
    config = DMLConfig(temperature=3.0, supervised_weight=1.0, mimicry_weight=1.0)
    
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        for model in models
    ]
    
    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)
        for opt in optimizers
    ]
    
    trainer = DMLTrainer(
        models=models,
        config=config,
        device=device,
        optimizers=optimizers,
        schedulers=schedulers,
    )
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=False  # Reduce output for benchmarking
    )
    
    # Get best accuracy for each model
    best_accs = []
    for i in range(len(models)):
        key = f'val_acc_model_{i}'
        if key in history:
            # History might not have per-model metrics yet
            best_acc = max([history[key][j] for j in range(len(history[key]))])
        else:
            # Use overall validation accuracy as approximation
            val_metrics = trainer.evaluate(val_loader)
            best_acc = val_metrics[f'val_acc_model_{i}']
        best_accs.append(best_acc)
        print(f"Model {i} Best Val Acc: {best_acc:.2f}%")
    
    avg_acc = sum(best_accs) / len(best_accs)
    print(f"Average Best Val Acc: {avg_acc:.2f}%")
    
    return best_accs, history


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 200
    batch_size = 128
    num_classes = 100
    
    print("="*60)
    print("CIFAR-100 Benchmark - Deep Mutual Learning")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    
    # Load data
    print("\nLoading CIFAR-100...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size,
        num_workers=2,
        val_split=0.1,
        download=True
    )
    
    results = {}
    
    # Experiment 1: Independent Training (Baseline)
    print("\n" + "="*60)
    print("Experiment 1: Independent Training")
    print("="*60)
    
    start_time = time.time()
    model = resnet32(num_classes=num_classes)
    best_acc, history = train_independent(model, train_loader, val_loader, device, epochs=num_epochs)
    training_time = time.time() - start_time
    
    results['independent'] = {
        'best_acc': best_acc,
        'training_time': training_time,
        'history': history
    }
    
    print(f"\nIndependent Training Results:")
    print(f"  Best Val Acc: {best_acc:.2f}%")
    print(f"  Training Time: {training_time/60:.2f} minutes")
    
    # Experiment 2: DML with 2 Networks
    print("\n" + "="*60)
    print("Experiment 2: Deep Mutual Learning (2 Networks)")
    print("="*60)
    
    start_time = time.time()
    models = [
        resnet32(num_classes=num_classes),
        resnet32(num_classes=num_classes),
    ]
    best_accs, history = train_dml(models, train_loader, val_loader, device, epochs=num_epochs)
    training_time = time.time() - start_time
    
    results['dml_2'] = {
        'best_accs': best_accs,
        'avg_acc': sum(best_accs) / len(best_accs),
        'training_time': training_time,
        'history': history
    }
    
    print(f"\nDML (2 Networks) Results:")
    print(f"  Average Best Val Acc: {sum(best_accs)/len(best_accs):.2f}%")
    print(f"  Training Time: {training_time/60:.2f} minutes")
    
    # Experiment 3: DML with 3 Networks
    print("\n" + "="*60)
    print("Experiment 3: Deep Mutual Learning (3 Networks)")
    print("="*60)
    
    start_time = time.time()
    models = [
        resnet32(num_classes=num_classes),
        resnet32(num_classes=num_classes),
        resnet32(num_classes=num_classes),
    ]
    best_accs, history = train_dml(models, train_loader, val_loader, device, epochs=num_epochs)
    training_time = time.time() - start_time
    
    results['dml_3'] = {
        'best_accs': best_accs,
        'avg_acc': sum(best_accs) / len(best_accs),
        'training_time': training_time,
        'history': history
    }
    
    print(f"\nDML (3 Networks) Results:")
    print(f"  Average Best Val Acc: {sum(best_accs)/len(best_accs):.2f}%")
    print(f"  Training Time: {training_time/60:.2f} minutes")
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Independent Training:     {results['independent']['best_acc']:.2f}%")
    print(f"DML (2 Networks):         {results['dml_2']['avg_acc']:.2f}%")
    print(f"DML (3 Networks):         {results['dml_3']['avg_acc']:.2f}%")
    print(f"\nImprovement over baseline:")
    print(f"  DML (2 Networks): +{results['dml_2']['avg_acc'] - results['independent']['best_acc']:.2f}%")
    print(f"  DML (3 Networks): +{results['dml_3']['avg_acc'] - results['independent']['best_acc']:.2f}%")
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        # Convert history to serializable format
        save_results = {}
        for exp_name, exp_data in results.items():
            save_results[exp_name] = {
                k: v for k, v in exp_data.items() if k != 'history'
            }
        json.dump(save_results, f, indent=2)
    
    print("\nResults saved to benchmark_results.json")


if __name__ == '__main__':
    main()
