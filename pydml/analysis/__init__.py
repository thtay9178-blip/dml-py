"""Analysis module for DML-PY."""

from .robustness import (
    test_robustness_to_noise,
    compute_flatness_metric,
    adversarial_robustness_test
)

from .visualization import (
    plot_training_history,
    plot_model_comparison,
    plot_entropy_distribution,
    plot_agreement_matrix,
    plot_robustness_comparison,
    create_training_dashboard
)

from .loss_landscape import LossLandscape, quick_landscape_analysis

__all__ = [
    'test_robustness_to_noise',
    'compute_flatness_metric',
    'adversarial_robustness_test',
    'plot_training_history',
    'plot_model_comparison',
    'plot_entropy_distribution',
    'plot_agreement_matrix',
    'plot_robustness_comparison',
    'create_training_dashboard',
    'LossLandscape',
    'quick_landscape_analysis',
]
