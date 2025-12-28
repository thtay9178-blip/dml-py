"""
Lightweight memory-efficient test for DML-PY.

This test uses CPU, small models, and tiny batches to avoid memory issues.
"""

import torch
import torch.nn as nn
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32

print("=" * 60)
print("DML-PY Lightweight Test (Memory-Efficient)")
print("=" * 60)

# Force CPU to avoid GPU memory issues
device = 'cpu'
print(f"\n✓ Using device: {device}")

# Test 1: Basic model creation
print("\n[Test 1] Creating models...")
try:
    models = [resnet32(num_classes=10) for _ in range(2)]
    print(f"✓ Created {len(models)} ResNet32 models")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 2: DML trainer creation
print("\n[Test 2] Creating DML trainer...")
try:
    config = DMLConfig(temperature=3.0)
    trainer = DMLTrainer(models, config=config, device=device)
    print(f"✓ DML Trainer initialized")
    print(f"  - Temperature: {config.temperature}")
    print(f"  - Models: {trainer.num_models}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 3: Forward pass with tiny batch
print("\n[Test 3] Testing forward pass...")
try:
    batch_size = 4  # Very small to save memory
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))
    
    for i, model in enumerate(trainer.models):
        model.eval()
        with torch.no_grad():
            output = model(x)
            assert output.shape == (batch_size, 10), f"Wrong shape: {output.shape}"
    
    print(f"✓ Forward pass works (batch_size={batch_size})")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 4: Loss computation
print("\n[Test 4] Testing loss computation...")
try:
    outputs = [model(x) for model in trainer.models]
    losses = trainer.compute_collaborative_loss(outputs, y)
    
    assert len(losses) == len(models), f"Expected {len(models)} losses"
    for i in range(len(models)):
        assert f'model_{i}' in losses, f"Missing loss for model_{i}"
        assert not torch.isnan(losses[f'model_{i}']), "Loss is NaN"
    
    print(f"✓ Loss computation works")
    print(f"  - Model 0 loss: {losses['model_0'].item():.4f}")
    print(f"  - Model 1 loss: {losses['model_1'].item():.4f}")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 5: Single training step (very small)
print("\n[Test 5] Testing single training step...")
try:
    # Create tiny dataset
    train_data = [(torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()) for _ in range(8)]
    from torch.utils.data import DataLoader, TensorDataset
    
    X = torch.stack([x for x, _ in train_data])
    Y = torch.tensor([y for _, y in train_data])
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Train for just 1 epoch
    trainer.train()
    for inputs, targets in train_loader:
        outputs = [model(inputs) for model in trainer.models]
        losses = trainer.compute_collaborative_loss(outputs, targets)
        
        # Backward pass for first model only (to save memory)
        trainer.optimizers[0].zero_grad()
        losses['model_0'].backward()
        trainer.optimizers[0].step()
        break  # Only one batch
    
    print(f"✓ Training step works")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 6: Model evaluation
print("\n[Test 6] Testing evaluation...")
try:
    trainer.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in train_loader:
            outputs = trainer.models[0](inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100.0 * correct / total
    print(f"✓ Evaluation works (accuracy: {acc:.2f}%)")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 7: Checkpoint save/load
print("\n[Test 7] Testing checkpoint save/load...")
try:
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
        trainer.save_checkpoint(checkpoint_path)
        assert os.path.exists(checkpoint_path), "Checkpoint file not created"
        
        # Load it back
        trainer.load_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint save/load works")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n✓ DML-PY is working correctly")
print("✓ All core functionality verified")
print("✓ Memory usage kept minimal (CPU only, small batches)")
print("\nYou can now:")
print("  1. Run examples with: python examples/quick_start.py")
print("  2. Run full tests with: pytest tests/")
print("  3. Try the demo: python examples/complete_demo.py")
