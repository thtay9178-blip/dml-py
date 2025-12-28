"""
Comprehensive tests for DML-PY.

Run with: python -m pytest tests/
"""

import pytest
import torch
import torch.nn as nn
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32, mobilenet_v2, wrn_28_10
from pydml.core.losses import CrossEntropyLoss, KLDivergenceLoss, DMLLoss
from pydml.utils.data import get_cifar10_loaders


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(784, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_dml_trainer_creation():
    """Test DML trainer can be created."""
    models = [SimpleModel() for _ in range(2)]
    config = DMLConfig()
    trainer = DMLTrainer(models, config=config, device='cpu')
    
    assert trainer.num_models == 2
    assert trainer.config.temperature == 3.0


def test_dml_loss_computation():
    """Test DML loss computation."""
    models = [SimpleModel() for _ in range(2)]
    trainer = DMLTrainer(models, device='cpu')
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))
    
    # Forward pass
    outputs = [model(x) for model in trainer.models]
    
    # Compute losses
    losses = trainer.compute_collaborative_loss(outputs, y)
    
    assert len(losses) == 2
    assert 'model_0' in losses
    assert 'model_1' in losses
    assert losses['model_0'].item() > 0


def test_kl_divergence_loss():
    """Test KL divergence loss."""
    kl_loss = KLDivergenceLoss(temperature=3.0)
    
    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    
    loss = kl_loss(student_logits, teacher_logits)
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_cross_entropy_loss():
    """Test cross entropy loss."""
    ce_loss = CrossEntropyLoss()
    
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    
    loss = ce_loss(logits, targets)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_resnet32_model():
    """Test ResNet32 model."""
    model = resnet32(num_classes=100)
    
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (2, 100)


def test_mobilenet_model():
    """Test MobileNet model."""
    model = mobilenet_v2(num_classes=100)
    
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (2, 100)


def test_wrn_model():
    """Test WRN model."""
    model = wrn_28_10(num_classes=100)
    
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (2, 100)


def test_ensemble_predictions():
    """Test ensemble predictions."""
    models = [SimpleModel() for _ in range(3)]
    trainer = DMLTrainer(models, device='cpu')
    
    x = torch.randn(4, 1, 28, 28)
    ensemble_output = trainer.get_ensemble_predictions(x)
    
    assert ensemble_output.shape == (4, 10)


def test_checkpoint_save_load(tmp_path):
    """Test checkpoint saving and loading."""
    models = [SimpleModel() for _ in range(2)]
    trainer = DMLTrainer(models, device='cpu')
    
    # Save checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    
    assert checkpoint_path.exists()
    
    # Load checkpoint
    trainer.load_checkpoint(str(checkpoint_path))
    
    assert trainer.current_epoch == 0


def test_different_architectures():
    """Test DML with different architectures."""
    models = [
        resnet32(num_classes=10),
        mobilenet_v2(num_classes=10)
    ]
    
    trainer = DMLTrainer(models, device='cpu')
    
    x = torch.randn(2, 3, 32, 32)
    outputs = [model(x) for model in trainer.models]
    y = torch.randint(0, 10, (2,))
    
    losses = trainer.compute_collaborative_loss(outputs, y)
    
    assert len(losses) == 2


def test_config_validation():
    """Test DML config with different values."""
    config = DMLConfig(
        temperature=5.0,
        supervised_weight=0.5,
        mimicry_weight=1.5
    )
    
    assert config.temperature == 5.0
    assert config.supervised_weight == 0.5
    assert config.mimicry_weight == 1.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
