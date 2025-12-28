# Contributing to DML-PY

Thank you for your interest in contributing to DML-PY! This guide will help you get started.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Code Style](#code-style)

## Code of Conduct

Be respectful, inclusive, and professional in all interactions. We're building a welcoming community.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+
- Git
- Virtual environment tool (venv, conda, etc.)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/DML-PY.git
   cd DML-PY
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/DML-PY.git
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install requirements separately
pip install -r requirements.txt
pip install pytest black flake8 mypy isort
```

### 3. Verify Installation

```bash
python -m pytest tests/
```

All tests should pass!

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add docstrings to all functions/classes
- Include type hints where possible
- Keep commits atomic and focused

### 3. Commit Guidelines

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed (wrap at 72 chars).
Explain WHAT and WHY, not HOW.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")
- Reference issues: "Fixes #123"
```

Examples:

```
Add temperature scaling strategy

Implement dynamic temperature adjustment during training
to improve knowledge distillation quality.

Fixes #42
```

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_dml.py

# Run with coverage
pytest --cov=dml-py --cov-report=html

# Run specific test
pytest tests/test_dml.py::test_dml_training
```

### Write Tests

All new features must include tests:

```python
# tests/test_new_feature.py
import pytest
import torch
from dml-py.your_module import YourClass

def test_your_feature():
    """Test description."""
    # Arrange
    input_data = torch.randn(10, 3, 32, 32)

    # Act
    result = YourClass().process(input_data)

    # Assert
    assert result.shape == (10, 10)
    assert result.sum() > 0
```

### Test Checklist

- [ ] All tests pass
- [ ] New features have tests
- [ ] Edge cases are covered
- [ ] Test coverage > 80%

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def train_model(model, data_loader, epochs=10):
    """
    Train a neural network model.

    Args:
        model (nn.Module): Model to train
        data_loader (DataLoader): Training data
        epochs (int, optional): Number of training epochs. Default: 10

    Returns:
        Dict[str, List[float]]: Training history with keys:
            - 'train_loss': Training loss per epoch
            - 'train_acc': Training accuracy per epoch

    Raises:
        ValueError: If epochs < 1

    Example:
        >>> model = ResNet20(num_classes=10)
        >>> history = train_model(model, train_loader, epochs=50)
        >>> print(f"Final accuracy: {history['train_acc'][-1]:.2f}%")
    """
    pass
```

### Update Documentation

- Update README.md if adding user-facing features
- Add examples to `examples/` directory
- Create/update Jupyter notebooks if appropriate
- Update PLAN.md status if completing planned features

## Pull Request Process

### 1. Update Your Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Push Changes

```bash
git push origin your-branch-name
```

### 3. Create Pull Request

1. Go to GitHub and click "New Pull Request"
2. Fill in the PR template:

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing

- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist

- [ ] Code follows style guidelines
- [ ] Docstrings added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

## Related Issues

Fixes #123
```

### 4. Code Review

- Address reviewer feedback promptly
- Push new commits to the same branch
- Request re-review when ready

### 5. Merge

Once approved, a maintainer will merge your PR!

## Code Style

### Python Style

Follow PEP 8 with these specifics:

```python
# Imports
import os
import sys

import torch
import torch.nn as nn

from dml-py.core import BaseTrainer
from dml-py.models import resnet20

# Constants
MAX_EPOCHS = 100
DEFAULT_LR = 0.1

# Functions
def compute_loss(outputs, targets, temperature=3.0):
    """Compute KL divergence loss."""
    pass

# Classes
class DMLTrainer:
    """Deep Mutual Learning trainer."""

    def __init__(self, models, learning_rate=0.1):
        self.models = models
        self.lr = learning_rate
```

### Formatting Tools

Use these tools to ensure consistent formatting:

```bash
# Auto-format with black
black dml-py/ tests/

# Sort imports
isort dml-py/ tests/

# Check style
flake8 dml-py/ tests/

# Type checking
mypy dml-py/
```

### Pre-commit Hook (Optional)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
black --check dml-py/ tests/
isort --check dml-py/ tests/
flake8 dml-py/ tests/
pytest
```

Make it executable:

```bash
chmod +x .git/hooks/pre-commit
```

## Project Structure

```
DML-PY/
├── dml-py/              # Main package
│   ├── core/           # Base classes and core functionality
│   ├── trainers/       # DML trainer implementations
│   ├── models/         # Model architectures
│   ├── losses/         # Loss functions
│   ├── strategies/     # Training strategies
│   ├── analysis/       # Analysis tools
│   └── utils/          # Utility functions
├── tests/              # Test suite
├── examples/           # Example scripts
├── notebooks/          # Jupyter tutorials
└── docs/               # Documentation
```

## Adding New Features

### 1. New Trainer

```python
# dml-py/trainers/my_trainer.py
from dml-py.core import BaseCollaborativeTrainer

class MyTrainer(BaseCollaborativeTrainer):
    """My custom DML trainer."""

    def compute_collaborative_loss(self, outputs, targets):
        # Implement your logic
        pass
```

### 2. New Strategy

```python
# dml-py/strategies/my_strategy.py
class MyStrategy:
    """My training strategy."""

    def apply(self, trainer):
        # Modify trainer behavior
        pass
```

### 3. Add Tests

```python
# tests/test_my_feature.py
def test_my_trainer():
    trainer = MyTrainer(models)
    assert trainer is not None
```

### 4. Add Example

```python
# examples/my_example.py
"""Example of using MyTrainer."""
from dml-py.trainers import MyTrainer

# Usage example...
```

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Features**: Open a GitHub Issue with use case description
- **Security**: Email maintainers directly (see README)

## Recognition

Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Forever appreciated! 🎉

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making DML-PY better! 🚀
