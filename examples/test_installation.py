"""
Simple test to verify DML-PY installation and basic functionality.
"""

import torch
from pydml import DMLTrainer, DMLConfig
from pydml.models.cifar import resnet32


def test_dml_basic():
    """Test basic DML functionality with dummy data."""
    print("Testing DML-PY basic functionality...")
    
    # Create dummy data
    batch_size = 16
    num_classes = 10
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # Create models
    models = [resnet32(num_classes=num_classes) for _ in range(2)]
    
    # Create trainer
    config = DMLConfig(temperature=3.0)
    trainer = DMLTrainer(models, config=config, device='cpu')
    
    # Test forward pass
    print("Testing forward pass...")
    for model in trainer.models:
        model.eval()
        with torch.no_grad():
            output = model(x)
            assert output.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {output.shape}"
    
    print("✓ Forward pass works")
    
    # Test loss computation
    print("Testing loss computation...")
    outputs = [model(x) for model in trainer.models]
    losses = trainer.compute_collaborative_loss(outputs, y)
    
    assert len(losses) == len(models), \
        f"Expected {len(models)} losses, got {len(losses)}"
    
    for i in range(len(models)):
        assert f'model_{i}' in losses, f"Missing loss for model_{i}"
        assert losses[f'model_{i}'].numel() == 1, "Loss should be a scalar"
    
    print("✓ Loss computation works")
    
    print("\n✅ All tests passed!")
    print("DML-PY is ready to use!")


if __name__ == '__main__':
    test_dml_basic()
