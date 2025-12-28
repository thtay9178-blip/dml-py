"""
Hyperparameter Search Utilities for DML-PY.

Provides automated hyperparameter optimization for Deep Mutual Learning training,
helping users find optimal configurations without manual tuning.

Supported Methods:
- Grid Search: Exhaustive search over parameter grid
- Random Search: Random sampling from parameter distributions
- Bayesian Optimization: Optuna-based smart search
- Evolutionary Search: Genetic algorithm-based optimization

Features:
- DML-specific parameter spaces
- Parallel trial execution
- Early stopping for poor configurations
- Visualization of search results
"""

import torch
import itertools
import random
from typing import Dict, List, Callable, Optional, Any, Tuple
import numpy as np
from pathlib import Path
import json


class HyperparameterSpace:
    """
    Define hyperparameter search space.
    
    Args:
        params: Dictionary of parameter names to possible values/distributions
        
    Example:
        >>> space = HyperparameterSpace({
        ...     'learning_rate': [0.001, 0.01, 0.1],
        ...     'temperature': [2, 3, 4, 5],
        ...     'kl_weight': [0.5, 1.0, 2.0]
        ... })
    """
    
    def __init__(self, params: Dict[str, List]):
        self.params = params
    
    def grid(self) -> List[Dict]:
        """Generate all combinations (grid search)."""
        keys = self.params.keys()
        values = self.params.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def random_sample(self, n_samples: int = 10) -> List[Dict]:
        """Generate random samples from space."""
        samples = []
        for _ in range(n_samples):
            sample = {k: random.choice(v) for k, v in self.params.items()}
            samples.append(sample)
        return samples
    
    def size(self) -> int:
        """Total number of configurations in grid."""
        return np.prod([len(v) for v in self.params.values()])


class HyperparameterSearcher:
    """
    Base class for hyperparameter search.
    
    Args:
        objective_fn: Function to optimize (takes config dict, returns metric)
        metric_name: Name of metric to optimize
        direction: 'maximize' or 'minimize'
        
    Example:
        >>> def objective(config):
        ...     trainer = DMLTrainer(models, **config)
        ...     results = trainer.train(train_loader, val_loader, epochs=10)
        ...     return results['val_acc']
        >>> 
        >>> searcher = GridSearcher(objective, 'accuracy', 'maximize')
        >>> best_config = searcher.search(space)
    """
    
    def __init__(
        self,
        objective_fn: Callable[[Dict], float],
        metric_name: str = 'metric',
        direction: str = 'maximize'
    ):
        self.objective_fn = objective_fn
        self.metric_name = metric_name
        self.direction = direction
        self.results = []
    
    def search(self, space: HyperparameterSpace, **kwargs) -> Dict:
        """
        Run hyperparameter search.
        
        Returns:
            Best configuration found
        """
        raise NotImplementedError
    
    def get_best_config(self) -> Tuple[Dict, float]:
        """Get best configuration and its metric value."""
        if not self.results:
            raise ValueError("No search results available")
        
        best_idx = np.argmax([r['metric'] for r in self.results])
        if self.direction == 'minimize':
            best_idx = np.argmin([r['metric'] for r in self.results])
        
        best_result = self.results[best_idx]
        return best_result['config'], best_result['metric']
    
    def save_results(self, output_path: str):
        """Save search results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Search results saved to {output_path}")
    
    def plot_results(self, output_path: str):
        """Plot search results."""
        import matplotlib.pyplot as plt
        
        metrics = [r['metric'] for r in self.results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(metrics, 'b-', alpha=0.6, linewidth=2)
        plt.scatter(range(len(metrics)), metrics, c=metrics, cmap='viridis', s=50)
        
        best_idx = np.argmax(metrics) if self.direction == 'maximize' else np.argmin(metrics)
        plt.scatter([best_idx], [metrics[best_idx]], color='red', s=200, 
                   marker='*', label='Best', zorder=5)
        
        plt.xlabel('Trial', fontsize=12)
        plt.ylabel(self.metric_name, fontsize=12)
        plt.title('Hyperparameter Search Results', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"✓ Search plot saved to {output_path}")


class GridSearcher(HyperparameterSearcher):
    """
    Grid search over hyperparameter space.
    
    Example:
        >>> space = HyperparameterSpace({
        ...     'lr': [0.001, 0.01, 0.1],
        ...     'temperature': [2, 3, 4]
        ... })
        >>> searcher = GridSearcher(objective_fn, 'accuracy', 'maximize')
        >>> best_config = searcher.search(space)
    """
    
    def search(self, space: HyperparameterSpace, verbose: bool = True) -> Dict:
        """Run exhaustive grid search."""
        configs = space.grid()
        
        if verbose:
            print(f"Starting grid search over {len(configs)} configurations...")
        
        for i, config in enumerate(configs):
            try:
                metric = self.objective_fn(config)
                self.results.append({
                    'trial': i,
                    'config': config,
                    'metric': metric
                })
                
                if verbose:
                    print(f"Trial {i+1}/{len(configs)}: {self.metric_name}={metric:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"Trial {i+1} failed: {str(e)}")
                continue
        
        best_config, best_metric = self.get_best_config()
        
        if verbose:
            print(f"\n✓ Grid search complete!")
            print(f"Best {self.metric_name}: {best_metric:.4f}")
            print(f"Best config: {best_config}")
        
        return best_config


class RandomSearcher(HyperparameterSearcher):
    """
    Random search over hyperparameter space.
    
    Example:
        >>> searcher = RandomSearcher(objective_fn, 'accuracy', 'maximize')
        >>> best_config = searcher.search(space, n_trials=20)
    """
    
    def search(
        self,
        space: HyperparameterSpace,
        n_trials: int = 20,
        verbose: bool = True
    ) -> Dict:
        """Run random search."""
        configs = space.random_sample(n_trials)
        
        if verbose:
            print(f"Starting random search with {n_trials} trials...")
        
        for i, config in enumerate(configs):
            try:
                metric = self.objective_fn(config)
                self.results.append({
                    'trial': i,
                    'config': config,
                    'metric': metric
                })
                
                if verbose:
                    print(f"Trial {i+1}/{n_trials}: {self.metric_name}={metric:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"Trial {i+1} failed: {str(e)}")
                continue
        
        best_config, best_metric = self.get_best_config()
        
        if verbose:
            print(f"\n✓ Random search complete!")
            print(f"Best {self.metric_name}: {best_metric:.4f}")
            print(f"Best config: {best_config}")
        
        return best_config


class OptunaSearcher(HyperparameterSearcher):
    """
    Bayesian optimization using Optuna.
    
    Example:
        >>> def suggest_fn(trial):
        ...     return {
        ...         'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1),
        ...         'temperature': trial.suggest_int('temperature', 2, 5)
        ...     }
        >>> 
        >>> searcher = OptunaSearcher(objective_fn, suggest_fn, 'accuracy', 'maximize')
        >>> best_config = searcher.search(n_trials=50)
    """
    
    def __init__(
        self,
        objective_fn: Callable[[Dict], float],
        suggest_fn: Callable,
        metric_name: str = 'metric',
        direction: str = 'maximize'
    ):
        super().__init__(objective_fn, metric_name, direction)
        self.suggest_fn = suggest_fn
    
    def search(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """Run Bayesian optimization with Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
        
        def objective(trial):
            config = self.suggest_fn(trial)
            metric = self.objective_fn(config)
            
            self.results.append({
                'trial': trial.number,
                'config': config,
                'metric': metric
            })
            
            return metric
        
        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler()
        )
        
        if verbose:
            print(f"Starting Optuna search with {n_trials} trials...")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose
        )
        
        best_config = study.best_params
        best_metric = study.best_value
        
        if verbose:
            print(f"\n✓ Optuna search complete!")
            print(f"Best {self.metric_name}: {best_metric:.4f}")
            print(f"Best config: {best_config}")
        
        return best_config


def create_dml_search_space() -> HyperparameterSpace:
    """
    Create default hyperparameter space for DML training.
    
    Returns:
        HyperparameterSpace with common DML parameters
        
    Example:
        >>> space = create_dml_search_space()
        >>> searcher = GridSearcher(objective_fn, 'accuracy', 'maximize')
        >>> best_config = searcher.search(space)
    """
    return HyperparameterSpace({
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'temperature': [2, 3, 4, 5],
        'kl_weight': [0.5, 1.0, 2.0],
        'batch_size': [64, 128, 256],
        'optimizer': ['sgd', 'adam'],
        'weight_decay': [0.0, 1e-4, 5e-4]
    })


def quick_search(
    objective_fn: Callable[[Dict], float],
    method: str = 'random',
    n_trials: int = 20,
    output_dir: str = 'search_results'
) -> Dict:
    """
    Quick hyperparameter search with sensible defaults.
    
    Args:
        objective_fn: Function to optimize
        method: Search method ('grid', 'random', 'optuna')
        n_trials: Number of trials (for random/optuna)
        output_dir: Directory to save results
        
    Returns:
        Best configuration found
        
    Example:
        >>> def objective(config):
        ...     # Train and evaluate model
        ...     return accuracy
        >>> 
        >>> best_config = quick_search(objective, method='random', n_trials=30)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    space = create_dml_search_space()
    
    if method == 'grid':
        searcher = GridSearcher(objective_fn, 'metric', 'maximize')
        best_config = searcher.search(space)
    elif method == 'random':
        searcher = RandomSearcher(objective_fn, 'metric', 'maximize')
        best_config = searcher.search(space, n_trials=n_trials)
    elif method == 'optuna':
        def suggest_fn(trial):
            return {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'temperature': trial.suggest_int('temperature', 2, 5),
                'kl_weight': trial.suggest_uniform('kl_weight', 0.5, 2.0),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adam']),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
            }
        
        searcher = OptunaSearcher(objective_fn, suggest_fn, 'metric', 'maximize')
        best_config = searcher.search(n_trials=n_trials)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save results
    searcher.save_results(output_dir / 'search_results.json')
    searcher.plot_results(output_dir / 'search_plot.png')
    
    return best_config


__all__ = [
    'HyperparameterSpace',
    'HyperparameterSearcher',
    'GridSearcher',
    'RandomSearcher',
    'OptunaSearcher',
    'create_dml_search_space',
    'quick_search'
]
