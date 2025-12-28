"""
Robustness analysis utilities for DML-PY.

This module provides tools for analyzing the robustness of trained models,
including noise injection tests and minima flatness analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader


def add_noise_to_model(model: nn.Module, noise_std: float) -> nn.Module:
    """
    Add Gaussian noise to model parameters.
    
    Args:
        model: PyTorch model
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Model with noisy parameters
    """
    noisy_model = type(model)(**{})  # Create new instance
    noisy_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for param in noisy_model.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)
    
    return noisy_model


def test_robustness_to_noise(
    model: nn.Module,
    test_loader: DataLoader,
    noise_levels: List[float],
    device: str = 'cuda'
) -> Dict[float, float]:
    """
    Test model robustness to parameter noise.
    
    Args:
        model: Trained model to test
        test_loader: Test data loader
        noise_levels: List of noise standard deviations to test
        device: Device to run on
    
    Returns:
        Dictionary mapping noise level to accuracy
    """
    model = model.to(device)
    results = {}
    
    # Test original model
    original_acc = evaluate_model(model, test_loader, device)
    results[0.0] = original_acc
    
    print(f"Original accuracy: {original_acc:.2f}%")
    
    # Test with different noise levels
    for noise_std in noise_levels:
        noisy_model = add_noise_to_model(model, noise_std)
        noisy_model = noisy_model.to(device)
        noisy_acc = evaluate_model(noisy_model, test_loader, device)
        results[noise_std] = noisy_acc
        
        print(f"Noise σ={noise_std:.4f}: {noisy_acc:.2f}% (drop: {original_acc - noisy_acc:.2f}%)")
    
    return results


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str) -> float:
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to run on
    
    Returns:
        Accuracy (0-100)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total


def compute_flatness_metric(
    model: nn.Module,
    test_loader: DataLoader,
    noise_std: float = 0.01,
    num_samples: int = 10,
    device: str = 'cuda'
) -> float:
    """
    Compute flatness metric for a model's loss landscape.
    
    Flatter minima are generally more robust and generalize better.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        noise_std: Standard deviation of noise to add
        num_samples: Number of noise samples to average over
        device: Device to run on
    
    Returns:
        Flatness metric (lower is flatter/better)
    """
    model = model.to(device)
    original_acc = evaluate_model(model, test_loader, device)
    
    acc_drops = []
    for _ in range(num_samples):
        noisy_model = add_noise_to_model(model, noise_std)
        noisy_model = noisy_model.to(device)
        noisy_acc = evaluate_model(noisy_model, test_loader, device)
        acc_drop = original_acc - noisy_acc
        acc_drops.append(acc_drop)
    
    # Average accuracy drop as flatness metric
    flatness = np.mean(acc_drops)
    return flatness


def compare_model_robustness(
    models: List[nn.Module],
    test_loader: DataLoader,
    noise_levels: List[float] = [0.001, 0.005, 0.01, 0.02],
    device: str = 'cuda'
) -> Dict[int, Dict[float, float]]:
    """
    Compare robustness of multiple models.
    
    Args:
        models: List of models to compare
        test_loader: Test data loader
        noise_levels: List of noise levels to test
        device: Device to run on
    
    Returns:
        Dictionary mapping model index to robustness results
    """
    results = {}
    
    print("Comparing model robustness...")
    print("=" * 60)
    
    for i, model in enumerate(models):
        print(f"\nModel {i}:")
        results[i] = test_robustness_to_noise(model, test_loader, noise_levels, device)
    
    # Summary
    print("\n" + "=" * 60)
    print("Robustness Comparison Summary")
    print("=" * 60)
    
    for noise_std in [0.0] + noise_levels:
        print(f"\nNoise σ={noise_std:.4f}:")
        for i in range(len(models)):
            acc = results[i][noise_std]
            print(f"  Model {i}: {acc:.2f}%")
    
    # Compute flatness scores
    print("\n" + "=" * 60)
    print("Flatness Scores (lower is better):")
    print("=" * 60)
    
    for i, model in enumerate(models):
        flatness = compute_flatness_metric(model, test_loader, noise_std=0.01, device=device)
        print(f"Model {i}: {flatness:.4f}")
    
    return results


def adversarial_robustness_test(
    model: nn.Module,
    test_loader: DataLoader,
    epsilon: float = 0.03,
    device: str = 'cuda'
) -> float:
    """
    Test model robustness to adversarial examples (FGSM attack).
    
    Args:
        model: Model to test
        test_loader: Test data loader
        epsilon: Perturbation magnitude
        device: Device to run on
    
    Returns:
        Adversarial accuracy (0-100)
    """
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Compute gradients
        model.zero_grad()
        loss.backward()
        
        # FGSM attack
        with torch.no_grad():
            perturbed_inputs = inputs + epsilon * inputs.grad.sign()
            perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
            
            # Evaluate on perturbed inputs
            outputs = model(perturbed_inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total
