"""
Unit tests for DML trainer.
"""

import pytest
import torch
import torch.nn as nn
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def test_dml_trainer_initialization():
    """Test DML trainer initialization."""
    models = [SimpleModel() for _ in range(2)]
    config = DMLConfig()
    trainer = DMLTrainer(models, config=config, device='cpu')
    
    assert trainer.num_models == 2
    assert trainer.config.temperature == 3.0
    assert len(trainer.optimizers) == 2


def test_dml_loss_computation():
    """Test DML loss computation."""
    models = [SimpleModel() for _ in range(2)]
    trainer = DMLTrainer(models, device='cpu')
    
    # Create dummy data
    batch_size = 16
    num_classes = 10
    outputs = [torch.randn(batch_size, num_classes) for _ in range(2)]
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Compute losses
    losses = trainer.compute_collaborative_loss(outputs, targets)
    
    assert len(losses) == 2
    assert 'model_0' in losses
    assert 'model_1' in losses
    assert losses['model_0'].numel() == 1
    assert losses['model_1'].numel() == 1


def test_dml_forward_pass():
    """Test forward pass through models."""
    models = [resnet32(num_classes=10) for _ in range(2)]
    trainer = DMLTrainer(models, device='cpu')
    
    # Create dummy input
    x = torch.randn(4, 3, 32, 32)
    
    # Forward pass
    trainer.eval()
    with torch.no_grad():
        for model in trainer.models:
            output = model(x)
            assert output.shape == (4, 10)


def test_ensemble_predictions():
    """Test ensemble predictions."""
    models = [SimpleModel() for _ in range(3)]
    trainer = DMLTrainer(models, device='cpu')
    
    x = torch.randn(8, 10)
    ensemble_output = trainer.get_ensemble_predictions(x)
    
    assert ensemble_output.shape == (8, 10)


def test_config_validation():
    """Test configuration validation."""
    config = DMLConfig(
        temperature=5.0,
        supervised_weight=0.5,
        mimicry_weight=2.0
    )
    
    assert config.temperature == 5.0
    assert config.supervised_weight == 0.5
    assert config.mimicry_weight == 2.0


def test_multiple_models():
    """Test with different numbers of models."""
    for num_models in [2, 3, 5]:
        models = [SimpleModel() for _ in range(num_models)]
        trainer = DMLTrainer(models, device='cpu')
        assert trainer.num_models == num_models


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
