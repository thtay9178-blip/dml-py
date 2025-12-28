"""
Strategies module for advanced training techniques.

This module provides strategies for curriculum learning, temperature adaptation,
peer selection, and other advanced training methods.
"""

from .peer_selection import (
    PeerSelector,
    PeerSelectionConfig,
    AllPeersSelector,
    BestPeersSelector,
    DiversePeersSelector,
    CurriculumPeersSelector,
    ConfidenceWeightedSelector,
    TournamentSelector,
    create_peer_selector,
    PeerSelectionAnalyzer,
)

from .temperature_scaling import (
    TemperatureScheduler,
    TemperatureSchedulerConfig,
    ConstantTemperature,
    LinearAnnealing,
    ExponentialAnnealing,
    CosineAnnealing,
    AdaptiveTemperature,
    CyclicalTemperature,
    PerformanceBasedTemperature,
    create_temperature_scheduler,
    TemperatureAnalyzer,
)

__all__ = [
    'PeerSelector',
    'PeerSelectionConfig',
    'AllPeersSelector',
    'BestPeersSelector',
    'DiversePeersSelector',
    'CurriculumPeersSelector',
    'ConfidenceWeightedSelector',
    'TournamentSelector',
    'create_peer_selector',
    'PeerSelectionAnalyzer',
    'TemperatureScheduler',
    'TemperatureSchedulerConfig',
    'ConstantTemperature',
    'LinearAnnealing',
    'ExponentialAnnealing',
    'CosineAnnealing',
    'AdaptiveTemperature',
    'CyclicalTemperature',
    'PerformanceBasedTemperature',
    'create_temperature_scheduler',
    'TemperatureAnalyzer',
]
