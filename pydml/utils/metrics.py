"""
Metrics computation utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> List[float]:
    """
    Compute top-k accuracy.
    
    Args:
        output: Model predictions (logits), shape (batch_size, num_classes)
        target: Ground truth labels, shape (batch_size,)
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy.
    
    Args:
        logits: Model predictions (logits)
        labels: Ground truth labels
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def entropy(probs: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Compute entropy of probability distributions.
    
    Args:
        probs: Probability distributions
        dim: Dimension along which to compute entropy
    
    Returns:
        Entropy values
    """
    return -(probs * torch.log(probs + 1e-10)).sum(dim=dim)


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute prediction entropy.
    
    Args:
        logits: Model predictions (logits)
    
    Returns:
        Entropy values
    """
    probs = F.softmax(logits, dim=1)
    return entropy(probs, dim=1)


def ensemble_accuracy(
    outputs_list: List[torch.Tensor],
    targets: torch.Tensor
) -> float:
    """
    Compute ensemble accuracy by averaging predictions.
    
    Args:
        outputs_list: List of model outputs (logits)
        targets: Ground truth labels
    
    Returns:
        Ensemble accuracy
    """
    # Average logits
    avg_logits = torch.stack(outputs_list).mean(dim=0)
    _, predicted = avg_logits.max(1)
    correct = predicted.eq(targets).sum().item()
    accuracy = 100. * correct / targets.size(0)
    
    return accuracy


def diversity_score(
    outputs_list: List[torch.Tensor]
) -> float:
    """
    Compute diversity score between models.
    
    Measures how different the predictions are across models.
    Higher diversity means models make different predictions.
    
    Args:
        outputs_list: List of model outputs (logits)
    
    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    predictions = [output.argmax(dim=1) for output in outputs_list]
    
    # Compute pairwise disagreement
    disagreements = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            disagreement = (predictions[i] != predictions[j]).float().mean().item()
            disagreements.append(disagreement)
    
    if len(disagreements) == 0:
        return 0.0
    
    return np.mean(disagreements)


def agreement_matrix(
    outputs_list: List[torch.Tensor]
) -> np.ndarray:
    """
    Compute agreement matrix between models.
    
    Args:
        outputs_list: List of model outputs (logits)
    
    Returns:
        Agreement matrix (num_models x num_models)
    """
    num_models = len(outputs_list)
    predictions = [output.argmax(dim=1) for output in outputs_list]
    
    matrix = np.zeros((num_models, num_models))
    
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                matrix[i, j] = 1.0
            else:
                agreement = (predictions[i] == predictions[j]).float().mean().item()
                matrix[i, j] = agreement
    
    return matrix
