"""
Unit tests for loss functions.
"""

import pytest
import torch
from pydml.losses import CrossEntropyLoss, KLDivergenceLoss, DMLLoss


def test_cross_entropy_loss():
    """Test cross-entropy loss."""
    loss_fn = CrossEntropyLoss()
    
    logits = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))
    
    loss = loss_fn(logits, targets)
    assert loss.numel() == 1
    assert loss.item() >= 0


def test_kl_divergence_loss():
    """Test KL divergence loss."""
    loss_fn = KLDivergenceLoss(temperature=3.0)
    
    student_logits = torch.randn(10, 5)
    teacher_logits = torch.randn(10, 5)
    
    loss = loss_fn(student_logits, teacher_logits)
    assert loss.numel() == 1
    assert loss.item() >= 0


def test_dml_loss():
    """Test DML loss."""
    loss_fn = DMLLoss(temperature=3.0, supervised_weight=1.0, mimicry_weight=1.0)
    
    logits = torch.randn(10, 5)
    peer_logits_list = [torch.randn(10, 5), torch.randn(10, 5)]
    targets = torch.randint(0, 5, (10,))
    
    loss = loss_fn(logits, peer_logits_list, targets)
    assert loss.numel() == 1
    assert loss.item() >= 0


def test_temperature_scaling():
    """Test that temperature affects KL divergence."""
    student_logits = torch.randn(10, 5)
    teacher_logits = torch.randn(10, 5)
    
    loss_low_temp = KLDivergenceLoss(temperature=1.0)(student_logits, teacher_logits)
    loss_high_temp = KLDivergenceLoss(temperature=5.0)(student_logits, teacher_logits)
    
    # Higher temperature should produce larger loss values (scaled by T^2)
    assert loss_high_temp.item() > loss_low_temp.item()


def test_loss_weights():
    """Test DML loss with different weights."""
    logits = torch.randn(10, 5)
    peer_logits_list = [torch.randn(10, 5)]
    targets = torch.randint(0, 5, (10,))
    
    loss_fn_1 = DMLLoss(supervised_weight=1.0, mimicry_weight=0.0)
    loss_fn_2 = DMLLoss(supervised_weight=0.0, mimicry_weight=1.0)
    
    loss_1 = loss_fn_1(logits, peer_logits_list, targets)
    loss_2 = loss_fn_2(logits, peer_logits_list, targets)
    
    # Losses should be different
    assert not torch.allclose(loss_1, loss_2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
