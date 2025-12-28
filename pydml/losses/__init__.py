"""Losses module for DML-PY."""

from pydml.core.losses import (
    BaseLoss,
    CrossEntropyLoss,
    KLDivergenceLoss,
    DMLLoss,
    LossRegistry,
)

from pydml.losses.attention_transfer import (
    AttentionTransferLoss,
    MultiLayerAttentionTransferLoss,
    AttentionMatchingLoss,
    SpatialAttentionVisualization,
    attention_transfer_loss,
)

__all__ = [
    "BaseLoss",
    "CrossEntropyLoss",
    "KLDivergenceLoss",
    "DMLLoss",
    "LossRegistry",
    "AttentionTransferLoss",
    "MultiLayerAttentionTransferLoss",
    "AttentionMatchingLoss",
    "SpatialAttentionVisualization",
    "attention_transfer_loss",
]
