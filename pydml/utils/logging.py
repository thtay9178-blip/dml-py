"""
Logging utilities for DML-PY.

This module provides logging and experiment tracking utilities.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class ExperimentLogger:
    """
    Logger for tracking experiments and their results.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs (default: 'experiments')
    """
    
    def __init__(self, experiment_name: str, log_dir: str = 'experiments'):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.start_time = time.time()
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.metrics_history = {}
        self.config = {}
        
        print(f"Experiment logger initialized: {self.exp_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config = config
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        if 'epochs' not in self.metrics_history:
            self.metrics_history['epochs'] = []
        
        self.metrics_history['epochs'].append(epoch)
        
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def log_model(self, model: torch.nn.Module, name: str = 'model'):
        """Save model checkpoint."""
        model_path = os.path.join(self.exp_dir, f'{name}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
    
    def log_text(self, text: str, filename: str = 'notes.txt'):
        """Log text to a file."""
        text_path = os.path.join(self.exp_dir, filename)
        with open(text_path, 'a') as f:
            f.write(f"[{datetime.now()}] {text}\n")
    
    def save_metrics(self):
        """Save all metrics to a JSON file."""
        metrics_path = os.path.join(self.exp_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def finalize(self):
        """Finalize the experiment and save all data."""
        elapsed_time = time.time() - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'elapsed_time': elapsed_time,
            'config': self.config,
            'final_metrics': {
                key: values[-1] if values else None
                for key, values in self.metrics_history.items()
                if key != 'epochs'
            }
        }
        
        summary_path = os.path.join(self.exp_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.save_metrics()
        print(f"Experiment finalized. Total time: {elapsed_time/60:.2f} minutes")
        print(f"Results saved to: {self.exp_dir}")


class ConsoleLogger:
    """Simple console logger with formatting."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def info(self, message: str):
        """Log info message."""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def warning(self, message: str):
        """Log warning message."""
        if self.verbose:
            print(f"[WARNING] {message}")
    
    def error(self, message: str):
        """Log error message."""
        print(f"[ERROR] {message}")
    
    def success(self, message: str):
        """Log success message."""
        if self.verbose:
            print(f"[SUCCESS] {message}")
    
    def section(self, title: str, width: int = 60):
        """Print a section header."""
        if self.verbose:
            print("\n" + "=" * width)
            print(title.center(width))
            print("=" * width)
    
    def subsection(self, title: str, width: int = 60):
        """Print a subsection header."""
        if self.verbose:
            print("\n" + "-" * width)
            print(title)
            print("-" * width)


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 32, 32)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input size for the model
    """
    print("\nModel Summary:")
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size in MB
    param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"Model size: {param_size:.2f} MB")
    
    print("=" * 60)
