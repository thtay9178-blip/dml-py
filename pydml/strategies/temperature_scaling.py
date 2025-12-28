"""
Adaptive Temperature Scaling for Knowledge Distillation.

This module implements dynamic temperature adjustment strategies that
automatically tune the softmax temperature during training based on
training progress, model confidence, and performance metrics.

Temperature controls the "softness" of probability distributions in KD:
- Low T (→1): Sharp, peaked distributions (hard labels)
- High T (→∞): Smooth, flat distributions (soft labels)

Adaptive strategies:
- Annealing: Start high, gradually decrease (curriculum-style)
- Performance-based: Adjust based on validation metrics
- Confidence-based: Adjust based on prediction confidence
- Loss-based: Adjust based on loss magnitude
- Cyclical: Oscillate between high and low temperatures

Reference: Extends ideas from knowledge distillation and curriculum learning.
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


@dataclass
class TemperatureSchedulerConfig:
    """Configuration for temperature scheduling."""
    strategy: str = "constant"  # constant, linear, exponential, cosine, adaptive, cyclical
    initial_temp: float = 5.0
    final_temp: float = 1.0
    warmup_epochs: int = 10
    total_epochs: int = 100
    min_temp: float = 1.0
    max_temp: float = 10.0
    cycle_length: int = 20  # For cyclical strategy
    adaptation_rate: float = 0.1  # For adaptive strategy


class TemperatureScheduler(ABC):
    """Abstract base class for temperature schedulers."""
    
    def __init__(self, config: TemperatureSchedulerConfig):
        self.config = config
        self.current_epoch = 0
        self.current_temp = config.initial_temp
        self.history = {
            'temperatures': [],
            'metrics': [],
            'adjustments': []
        }
    
    @abstractmethod
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """
        Update temperature for the current epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Optional dictionary of metrics (loss, accuracy, confidence, etc.)
        
        Returns:
            Updated temperature value
        """
        pass
    
    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.current_temp
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_epoch = 0
        self.current_temp = self.config.initial_temp
        self.history = {
            'temperatures': [],
            'metrics': [],
            'adjustments': []
        }


class ConstantTemperature(TemperatureScheduler):
    """Fixed temperature throughout training."""
    
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Return constant temperature."""
        self.current_epoch = epoch
        self.current_temp = self.config.initial_temp
        self.history['temperatures'].append(self.current_temp)
        if metrics:
            self.history['metrics'].append(metrics.copy())
        return self.current_temp


class LinearAnnealing(TemperatureScheduler):
    """Linear temperature annealing from high to low."""
    
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Linearly decrease temperature."""
        self.current_epoch = epoch
        
        if epoch < self.config.warmup_epochs:
            # Warmup phase: stay at initial temperature
            self.current_temp = self.config.initial_temp
        else:
            # Linear annealing
            progress = (epoch - self.config.warmup_epochs) / (self.config.total_epochs - self.config.warmup_epochs)
            progress = min(1.0, progress)
            self.current_temp = self.config.initial_temp + progress * (self.config.final_temp - self.config.initial_temp)
        
        # Clamp to bounds
        self.current_temp = max(self.config.min_temp, min(self.config.max_temp, self.current_temp))
        
        self.history['temperatures'].append(self.current_temp)
        if metrics:
            self.history['metrics'].append(metrics.copy())
        
        return self.current_temp


class ExponentialAnnealing(TemperatureScheduler):
    """Exponential temperature decay."""
    
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Exponentially decrease temperature."""
        self.current_epoch = epoch
        
        if epoch < self.config.warmup_epochs:
            self.current_temp = self.config.initial_temp
        else:
            # Exponential decay: T = T_init * exp(-rate * progress)
            progress = (epoch - self.config.warmup_epochs) / (self.config.total_epochs - self.config.warmup_epochs)
            progress = min(1.0, progress)
            
            # Compute decay rate to reach final_temp at end
            if self.config.final_temp > 0:
                decay_rate = -math.log(self.config.final_temp / self.config.initial_temp)
                self.current_temp = self.config.initial_temp * math.exp(-decay_rate * progress)
            else:
                self.current_temp = self.config.initial_temp * (1 - progress)
        
        self.current_temp = max(self.config.min_temp, min(self.config.max_temp, self.current_temp))
        
        self.history['temperatures'].append(self.current_temp)
        if metrics:
            self.history['metrics'].append(metrics.copy())
        
        return self.current_temp


class CosineAnnealing(TemperatureScheduler):
    """Cosine annealing schedule (smooth decay)."""
    
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Apply cosine annealing to temperature."""
        self.current_epoch = epoch
        
        if epoch < self.config.warmup_epochs:
            self.current_temp = self.config.initial_temp
        else:
            # Cosine annealing
            progress = (epoch - self.config.warmup_epochs) / (self.config.total_epochs - self.config.warmup_epochs)
            progress = min(1.0, progress)
            
            # Cosine schedule: (1 + cos(π * progress)) / 2
            cosine_factor = (1 + math.cos(math.pi * progress)) / 2
            temp_range = self.config.initial_temp - self.config.final_temp
            self.current_temp = self.config.final_temp + temp_range * cosine_factor
        
        self.current_temp = max(self.config.min_temp, min(self.config.max_temp, self.current_temp))
        
        self.history['temperatures'].append(self.current_temp)
        if metrics:
            self.history['metrics'].append(metrics.copy())
        
        return self.current_temp


class AdaptiveTemperature(TemperatureScheduler):
    """
    Adaptive temperature based on training metrics.
    
    Adjusts temperature based on:
    - Loss: Higher loss → higher temperature (more smoothing)
    - Confidence: Lower confidence → higher temperature
    - Accuracy: Lower accuracy → higher temperature
    """
    
    def __init__(self, config: TemperatureSchedulerConfig):
        super().__init__(config)
        self.baseline_loss = None
        self.baseline_confidence = None
    
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Adaptively adjust temperature based on metrics."""
        self.current_epoch = epoch
        
        if metrics is None:
            # No metrics, use default schedule
            return ConstantTemperature(self.config).step(epoch)
        
        # Initialize baselines
        if self.baseline_loss is None and 'loss' in metrics:
            self.baseline_loss = metrics['loss']
        if self.baseline_confidence is None and 'confidence' in metrics:
            self.baseline_confidence = metrics['confidence']
        
        # Compute adaptation signal
        adaptation = 0.0
        num_signals = 0
        
        # Loss-based adaptation
        if 'loss' in metrics and self.baseline_loss is not None:
            loss_ratio = metrics['loss'] / (self.baseline_loss + 1e-8)
            if loss_ratio > 1.0:
                # Loss increasing: increase temperature (more smoothing)
                adaptation += (loss_ratio - 1.0)
            else:
                # Loss decreasing: decrease temperature (sharper)
                adaptation -= (1.0 - loss_ratio)
            num_signals += 1
        
        # Confidence-based adaptation
        if 'confidence' in metrics:
            if self.baseline_confidence is not None:
                conf_ratio = metrics['confidence'] / (self.baseline_confidence + 1e-8)
                if conf_ratio < 1.0:
                    # Confidence decreasing: increase temperature
                    adaptation += (1.0 - conf_ratio)
                else:
                    # Confidence increasing: decrease temperature
                    adaptation -= (conf_ratio - 1.0)
            num_signals += 1
        
        # Accuracy-based adaptation (if available)
        if 'accuracy' in metrics:
            if metrics['accuracy'] < 50.0:  # Low accuracy
                adaptation += 0.5  # Increase temperature
            elif metrics['accuracy'] > 80.0:  # High accuracy
                adaptation -= 0.5  # Decrease temperature
            num_signals += 1
        
        # Average adaptation signal
        if num_signals > 0:
            adaptation /= num_signals
        
        # Apply adaptation with learning rate
        delta_temp = self.config.adaptation_rate * adaptation
        self.current_temp += delta_temp
        
        # Clamp to bounds
        self.current_temp = max(self.config.min_temp, min(self.config.max_temp, self.current_temp))
        
        self.history['temperatures'].append(self.current_temp)
        self.history['metrics'].append(metrics.copy())
        self.history['adjustments'].append(delta_temp)
        
        return self.current_temp


class CyclicalTemperature(TemperatureScheduler):
    """
    Cyclical temperature schedule.
    
    Oscillates between high and low temperatures to alternate between
    exploration (high T) and exploitation (low T).
    """
    
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Apply cyclical temperature schedule."""
        self.current_epoch = epoch
        
        # Compute position in cycle
        cycle_position = (epoch % self.config.cycle_length) / self.config.cycle_length
        
        # Triangular wave: 0 → 1 → 0
        if cycle_position < 0.5:
            # Increasing phase
            factor = 2 * cycle_position
        else:
            # Decreasing phase
            factor = 2 * (1 - cycle_position)
        
        # Interpolate between min and max temperature
        temp_range = self.config.max_temp - self.config.min_temp
        self.current_temp = self.config.min_temp + factor * temp_range
        
        self.history['temperatures'].append(self.current_temp)
        if metrics:
            self.history['metrics'].append(metrics.copy())
        
        return self.current_temp


class PerformanceBasedTemperature(TemperatureScheduler):
    """
    Adjust temperature based on validation performance.
    
    - If performance improving: decrease temperature (sharper predictions)
    - If performance plateauing/degrading: increase temperature (more exploration)
    """
    
    def __init__(self, config: TemperatureSchedulerConfig):
        super().__init__(config)
        self.best_metric = None
        self.patience_counter = 0
        self.patience = 5
    
    def step(self, epoch: int, metrics: Optional[Dict] = None) -> float:
        """Adjust temperature based on performance."""
        self.current_epoch = epoch
        
        if metrics is None or 'accuracy' not in metrics:
            return self.current_temp
        
        current_accuracy = metrics['accuracy']
        
        # Initialize best metric
        if self.best_metric is None:
            self.best_metric = current_accuracy
            self.history['temperatures'].append(self.current_temp)
            self.history['metrics'].append(metrics.copy())
            return self.current_temp
        
        # Check if performance improved
        if current_accuracy > self.best_metric + 0.5:  # Improvement threshold
            # Performance improving: decrease temperature
            self.best_metric = current_accuracy
            self.patience_counter = 0
            self.current_temp = max(self.config.min_temp, self.current_temp * 0.95)
        else:
            # Performance not improving: increase patience counter
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                # Plateau detected: increase temperature
                self.current_temp = min(self.config.max_temp, self.current_temp * 1.1)
                self.patience_counter = 0
        
        self.history['temperatures'].append(self.current_temp)
        self.history['metrics'].append(metrics.copy())
        
        return self.current_temp


def create_temperature_scheduler(config: TemperatureSchedulerConfig) -> TemperatureScheduler:
    """Factory function to create temperature scheduler."""
    scheduler_map = {
        'constant': ConstantTemperature,
        'linear': LinearAnnealing,
        'exponential': ExponentialAnnealing,
        'cosine': CosineAnnealing,
        'adaptive': AdaptiveTemperature,
        'cyclical': CyclicalTemperature,
        'performance': PerformanceBasedTemperature,
    }
    
    if config.strategy not in scheduler_map:
        raise ValueError(f"Unknown temperature strategy: {config.strategy}")
    
    return scheduler_map[config.strategy](config)


class TemperatureAnalyzer:
    """Tools for analyzing temperature schedules."""
    
    @staticmethod
    def visualize_schedule(
        schedulers: Dict[str, TemperatureScheduler],
        total_epochs: int,
        save_path: Optional[str] = None
    ):
        """Visualize different temperature schedules."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for visualization")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for name, scheduler in schedulers.items():
            scheduler.reset()
            temperatures = []
            
            for epoch in range(total_epochs):
                temp = scheduler.step(epoch)
                temperatures.append(temp)
            
            ax.plot(temperatures, label=name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Temperature', fontsize=12)
        ax.set_title('Temperature Scheduling Strategies', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved temperature schedule visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def analyze_effectiveness(history: Dict) -> Dict:
        """Analyze the effectiveness of temperature scheduling."""
        stats = {
            'mean_temperature': 0.0,
            'std_temperature': 0.0,
            'min_temperature': float('inf'),
            'max_temperature': float('-inf'),
            'temperature_range': 0.0,
        }
        
        if not history.get('temperatures'):
            return stats
        
        temps = np.array(history['temperatures'])
        
        stats['mean_temperature'] = float(np.mean(temps))
        stats['std_temperature'] = float(np.std(temps))
        stats['min_temperature'] = float(np.min(temps))
        stats['max_temperature'] = float(np.max(temps))
        stats['temperature_range'] = stats['max_temperature'] - stats['min_temperature']
        
        # Correlate with metrics if available
        if history.get('metrics') and len(history['metrics']) > 0:
            # Check if accuracy improves as temperature changes
            if all('accuracy' in m for m in history['metrics']):
                accuracies = [m['accuracy'] for m in history['metrics']]
                correlation = np.corrcoef(temps[:len(accuracies)], accuracies)[0, 1]
                stats['temp_accuracy_correlation'] = float(correlation)
        
        return stats


# Example usage
if __name__ == "__main__":
    print("Testing Temperature Scheduling Strategies...")
    print("=" * 70)
    
    total_epochs = 100
    
    # Create different schedulers
    configs = {
        'Constant': TemperatureSchedulerConfig(strategy='constant', initial_temp=4.0),
        'Linear': TemperatureSchedulerConfig(strategy='linear', initial_temp=8.0, final_temp=1.0, warmup_epochs=10, total_epochs=total_epochs),
        'Exponential': TemperatureSchedulerConfig(strategy='exponential', initial_temp=8.0, final_temp=1.0, warmup_epochs=10, total_epochs=total_epochs),
        'Cosine': TemperatureSchedulerConfig(strategy='cosine', initial_temp=8.0, final_temp=1.0, warmup_epochs=10, total_epochs=total_epochs),
        'Cyclical': TemperatureSchedulerConfig(strategy='cyclical', min_temp=2.0, max_temp=8.0, cycle_length=20),
    }
    
    schedulers = {name: create_temperature_scheduler(config) for name, config in configs.items()}
    
    # Test each scheduler
    for name, scheduler in schedulers.items():
        print(f"\n{name} Scheduler:")
        print(f"  Epoch 0:   T = {scheduler.step(0):.2f}")
        print(f"  Epoch 25:  T = {scheduler.step(25):.2f}")
        print(f"  Epoch 50:  T = {scheduler.step(50):.2f}")
        print(f"  Epoch 99:  T = {scheduler.step(99):.2f}")
    
    # Visualize all schedules
    print("\n" + "=" * 70)
    print("Generating visualization...")
    TemperatureAnalyzer.visualize_schedule(schedulers, total_epochs, 'temperature_schedules.png')
    
    print("\n" + "=" * 70)
    print("All temperature strategies tested successfully!")
    print("=" * 70)
