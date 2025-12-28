# Getting Started with DML-PY

This guide will help you get up and running with DML-PY in minutes.

## Installation

### Quick Install

```bash
cd DML-PY
pip install -e .
```

### With UV (Recommended - Fast!)

```bash
cd DML-PY
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Verify Installation

Run the quick test:

```bash
python examples/test_installation.py
```

You should see:

```
✅ All tests passed!
DML-PY is ready to use!
```

## Your First DML Training

### 1. Simple Example (CIFAR-10)

```python
import torch
from dml-py import DMLTrainer, DMLConfig
from dml-py.models.cifar import resnet32
from dml-py.utils.data import get_cifar10_loaders

# Load data
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=128,
    download=True
)

# Create 2 models
models = [
    resnet32(num_classes=10),
    resnet32(num_classes=10)
]

# Train with DML
trainer = DMLTrainer(models, device='cuda')
trainer.fit(train_loader, val_loader, epochs=50)

# Evaluate
metrics = trainer.evaluate(test_loader)
print(f"Test Accuracy: {metrics['val_acc']:.2f}%")
```

### 2. Run Existing Examples

#### Quick Start

```bash
python examples/quick_start.py
```

#### Full Benchmark (Warning: Takes 4-6 hours on GPU)

```bash
python examples/cifar100_benchmark.py
```

#### Advanced Training with Analysis

```bash
python examples/advanced_training.py
```

## Common Use Cases

### Train with Different Architectures

```python
from dml-py.models.cifar import resnet32, mobilenet_v2, wrn_28_10

# Mix different architectures
models = [
    resnet32(num_classes=100),
    mobilenet_v2(num_classes=100),
    wrn_28_10(num_classes=100)
]

trainer = DMLTrainer(models, device='cuda')
```

### Custom Training Configuration

```python
from dml-py import DMLConfig

# Configure DML parameters
config = DMLConfig(
    temperature=5.0,           # Higher = softer targets
    supervised_weight=1.0,     # Weight for ground truth loss
    mimicry_weight=0.5,        # Weight for peer learning
)

# Custom optimizers
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

# Learning rate schedulers
schedulers = [
    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    for opt in optimizers
]

trainer = DMLTrainer(
    models=models,
    config=config,
    device='cuda',
    optimizers=optimizers,
    schedulers=schedulers
)
```

### Use Callbacks

```python
from dml-py.core.callbacks import ModelCheckpoint, TensorBoardLogger

callbacks = [
    # Save best model
    ModelCheckpoint(
        filepath='checkpoints/best_model_{epoch}.pt',
        monitor='val_acc',
        mode='max',
        save_best_only=True
    ),

    # Log to TensorBoard
    TensorBoardLogger(log_dir='runs/my_experiment'),
]

trainer = DMLTrainer(models, callbacks=callbacks)
```

### Analyze Model Robustness

```python
from dml-py.analysis.robustness import compare_model_robustness

# After training
results = compare_model_robustness(
    models=trainer.models,
    test_loader=test_loader,
    noise_levels=[0.001, 0.005, 0.01, 0.02],
    device='cuda'
)

# Results show accuracy with different noise levels
```

## Understanding DML

### How It Works

1. **Multiple Networks**: Train K networks simultaneously
2. **Two Loss Components**:

   - **Supervised Loss**: Learn from ground truth labels (cross-entropy)
   - **Mimicry Loss**: Learn from peer predictions (KL divergence)

3. **Key Formula**:
   ```
   L_k = L_CE(y_k, y_true) + λ * (1/K-1) * Σ L_KL(y_k, y_j)
   ```

### Key Parameters

- **temperature**: Controls softness of probability distributions

  - Higher (e.g., 5.0) = softer, more uncertain targets
  - Lower (e.g., 1.0) = harder, more confident targets
  - Default: 3.0

- **supervised_weight**: Weight for ground truth loss

  - Default: 1.0

- **mimicry_weight**: Weight for peer learning
  - Default: 1.0

### Why Use DML?

1. **Better Accuracy**: Each model learns from peers
2. **Robust Models**: Flatter minima, better generalization
3. **Ensemble Benefits**: Multiple trained models for free
4. **No Pre-training**: All networks train from scratch together

## Tips & Best Practices

### 1. Start Small

- Test with CIFAR-10 (10 classes) before CIFAR-100
- Use fewer epochs initially (50 instead of 200)
- Start with 2 models before trying 3+

### 2. GPU Recommendations

- 2 ResNet32 models: ~4GB VRAM
- 3 ResNet32 models: ~6GB VRAM
- Reduce batch size if OOM

### 3. Training Time

- CIFAR-10, 2 models, 50 epochs: ~30 min on RTX 3090
- CIFAR-100, 2 models, 200 epochs: ~4 hours on RTX 3090

### 4. Hyperparameter Tuning

- Temperature: Try 1.0, 3.0, 5.0, 10.0
- Learning rate: 0.1 with SGD works well
- Weight decay: 5e-4 for CIFAR

### 5. Monitoring Training

```python
# Enable verbose output
history = trainer.fit(train_loader, val_loader, epochs=100, verbose=True)

# Check history
print(f"Best val acc: {max(history['val_acc']):.2f}%")
```

## Troubleshooting

### ImportError: No module named 'dml-py'

```bash
# Make sure you installed in editable mode
pip install -e .
```

### CUDA Out of Memory

```python
# Reduce batch size
train_loader, val_loader, test_loader = get_cifar10_loaders(
    batch_size=64  # instead of 128
)

# Or train fewer models
models = [resnet32(num_classes=10)]  # instead of 2+
```

### Slow Training

```python
# Reduce number of workers if CPU bottleneck
train_loader, _, _ = get_cifar10_loaders(num_workers=2)

# Use mixed precision (if available)
# Coming in Phase 3
```

## Next Steps

1. **Run Examples**: Try all examples in `examples/`
2. **Read Documentation**: Check `PLAN.md` and `IMPLEMENTATION_SUMMARY.md`
3. **Run Tests**: `pytest tests/ -v`
4. **Experiment**: Try different architectures and configurations
5. **Contribute**: Add new features or models!

## Getting Help

- **Issues**: Check existing examples first
- **Questions**: Read `PLAN.md` for design decisions
- **Bugs**: Run tests to isolate the problem
- **Feature Requests**: See roadmap in `PLAN.md`

## Quick Command Reference

```bash
# Installation
pip install -e .

# Testing
python examples/test_installation.py
pytest tests/ -v

# Examples
python examples/quick_start.py
python examples/cifar100_benchmark.py
python examples/advanced_training.py

# Statistics
python scripts/project_stats.py

# Project Structure
tree -L 2 -I '__pycache__|*.pyc|.venv'
```

## What's Next?

Check out:

- `examples/advanced_training.py` for full-featured training
- `dml-py/analysis/robustness.py` for model analysis
- `PLAN.md` for the complete roadmap

Happy training! 🚀
