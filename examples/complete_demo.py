"""
Complete Workflow Demo - DML-PY

This script demonstrates a complete workflow:
1. Data loading
2. Model creation
3. Training
4. Evaluation
5. Analysis
6. Comparison with baseline

Run with: python examples/complete_demo.py
"""

import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32
from pydml.utils.data import get_cifar10_loaders
from pydml.analysis.robustness import evaluate_model
import time


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)


def train_baseline(train_loader, val_loader, test_loader, device, epochs=50):
    """Train a single model (baseline)."""
    print_section("BASELINE: Single Model Training")
    
    model = resnet32(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    
    print(f"Training for {epochs} epochs...")
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            val_acc = evaluate_model(model, val_loader, device)
            best_val_acc = max(best_val_acc, val_acc)
            print(f"Epoch {epoch}/{epochs} - Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")
        
        scheduler.step()
    
    train_time = time.time() - start_time
    test_acc = evaluate_model(model, test_loader, device)
    
    print(f"\nBaseline Results:")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Training Time: {train_time:.2f}s ({train_time/60:.2f} min)")
    
    return model, test_acc, train_time


def train_dml(train_loader, val_loader, test_loader, device, num_models=2, epochs=50):
    """Train with Deep Mutual Learning."""
    print_section(f"DML: {num_models} Models Collaborative Training")
    
    # Create models
    models = [resnet32(num_classes=10) for _ in range(num_models)]
    print(f"Created {num_models} ResNet32 models")
    
    # Configure DML
    config = DMLConfig(temperature=3.0, supervised_weight=1.0, mimicry_weight=1.0)
    
    # Setup optimizers and schedulers
    optimizers = [
        torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        for model in models
    ]
    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 40], gamma=0.1)
        for opt in optimizers
    ]
    
    # Create trainer
    trainer = DMLTrainer(
        models=models,
        config=config,
        device=device,
        optimizers=optimizers,
        schedulers=schedulers
    )
    
    print(f"Training for {epochs} epochs...")
    start_time = time.time()
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=False  # Reduce output
    )
    
    train_time = time.time() - start_time
    
    # Evaluate each model
    print("\nPer-Model Results:")
    test_accs = []
    for i, model in enumerate(trainer.models):
        test_acc = evaluate_model(model, test_loader, device)
        test_accs.append(test_acc)
        print(f"  Model {i}: {test_acc:.2f}%")
    
    avg_acc = sum(test_accs) / len(test_accs)
    
    print(f"\nDML Results:")
    print(f"  Average Test Acc: {avg_acc:.2f}%")
    print(f"  Best Model: {max(test_accs):.2f}%")
    print(f"  Training Time: {train_time:.2f}s ({train_time/60:.2f} min)")
    
    return trainer, avg_acc, train_time


def main():
    print_section("DML-PY Complete Workflow Demo")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 50  # Reduced for demo (use 200 for paper results)
    batch_size = 128
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Dataset: CIFAR-10")
    
    # Load data
    print_section("Loading CIFAR-10 Dataset")
    
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=2,
        val_split=0.1,
        download=True
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Experiment 1: Baseline
    baseline_model, baseline_acc, baseline_time = train_baseline(
        train_loader, val_loader, test_loader, device, epochs=num_epochs
    )
    
    # Experiment 2: DML with 2 models
    dml_trainer, dml_acc, dml_time = train_dml(
        train_loader, val_loader, test_loader, device, num_models=2, epochs=num_epochs
    )
    
    # Comparison
    print_section("COMPARISON")
    
    print(f"\nAccuracy:")
    print(f"  Baseline (1 model):      {baseline_acc:.2f}%")
    print(f"  DML (2 models average):  {dml_acc:.2f}%")
    print(f"  Improvement:             +{dml_acc - baseline_acc:.2f}%")
    
    print(f"\nTraining Time:")
    print(f"  Baseline:  {baseline_time:.2f}s ({baseline_time/60:.2f} min)")
    print(f"  DML:       {dml_time:.2f}s ({dml_time/60:.2f} min)")
    print(f"  Overhead:  +{(dml_time/baseline_time - 1)*100:.1f}%")
    
    # Analysis
    print_section("ANALYSIS")
    
    print("\nKey Insights:")
    if dml_acc > baseline_acc:
        print(f"  ✓ DML improved accuracy by {dml_acc - baseline_acc:.2f}%")
    else:
        print(f"  ⚠ DML accuracy similar to baseline (may need more epochs)")
    
    time_ratio = dml_time / baseline_time
    print(f"  ✓ Training time overhead: {(time_ratio - 1)*100:.1f}%")
    print(f"  ✓ You get {len(dml_trainer.models)} trained models instead of 1")
    
    print("\nNext Steps:")
    print("  1. Run with more epochs (200) for better results")
    print("  2. Try different architectures (MobileNet, WRN)")
    print("  3. Experiment with 3+ models")
    print("  4. Analyze robustness with pydml.analysis.robustness")
    print("  5. Run full benchmark: python examples/cifar100_benchmark.py")
    
    print_section("Demo Complete!")
    print("\n✅ Successfully demonstrated DML-PY workflow")
    print("📊 Results show collaborative learning benefits")
    print("🚀 Ready for production use!")


if __name__ == '__main__':
    main()
