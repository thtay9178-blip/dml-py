"""
Model Export Utilities for DML-PY.

Provides utilities to export trained DML-PY models to various formats
for deployment and inference in production environments.

Supported Formats:
- ONNX: Cross-platform neural network format
- TorchScript: PyTorch's JIT-compiled format
- State Dict: PyTorch native checkpoint format

Features:
- Automatic input shape inference
- Optimization for inference
- Ensemble export support
- Quantization support
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path
import warnings


class ExportConfig:
    """
    Configuration for model export.
    
    Args:
        format: Export format ('onnx', 'torchscript', 'state_dict')
        opset_version: ONNX opset version (default: 11)
        dynamic_axes: Dynamic axes for ONNX export
        optimize_for_inference: Whether to optimize for inference
        quantize: Whether to apply quantization
        input_names: Names for input tensors
        output_names: Names for output tensors
        
    Example:
        >>> config = ExportConfig(
        ...     format='onnx',
        ...     opset_version=13,
        ...     optimize_for_inference=True
        ... )
    """
    
    def __init__(
        self,
        format: str = 'onnx',
        opset_version: int = 11,
        dynamic_axes: Optional[Dict] = None,
        optimize_for_inference: bool = True,
        quantize: bool = False,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None
    ):
        self.format = format.lower()
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes or {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        self.optimize_for_inference = optimize_for_inference
        self.quantize = quantize
        self.input_names = input_names or ['input']
        self.output_names = output_names or ['output']


class ModelExporter:
    """
    Exporter for DML-PY trained models.
    
    Args:
        model: PyTorch model to export
        config: Export configuration
        
    Example:
        >>> model = trainer.models[0]
        >>> exporter = ModelExporter(model, ExportConfig(format='onnx'))
        >>> exporter.export('model.onnx', input_shape=(1, 3, 32, 32))
    """
    
    def __init__(self, model: nn.Module, config: ExportConfig):
        self.model = model
        self.config = config
        
        # Set model to eval mode
        self.model.eval()
    
    def export(
        self,
        output_path: Union[str, Path],
        input_shape: Tuple[int, ...],
        device: str = 'cpu'
    ):
        """
        Export model to specified format.
        
        Args:
            output_path: Path to save exported model
            input_shape: Shape of input tensor (including batch dimension)
            device: Device for export ('cpu' or 'cuda')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        if self.config.format == 'onnx':
            self._export_onnx(output_path, dummy_input)
        elif self.config.format == 'torchscript':
            self._export_torchscript(output_path, dummy_input)
        elif self.config.format == 'state_dict':
            self._export_state_dict(output_path)
        else:
            raise ValueError(f"Unsupported export format: {self.config.format}")
        
        print(f"✓ Model exported successfully to {output_path}")
    
    def _export_onnx(self, output_path: Path, dummy_input: torch.Tensor):
        """Export model to ONNX format."""
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=self.config.opset_version,
                do_constant_folding=self.config.optimize_for_inference,
                input_names=self.config.input_names,
                output_names=self.config.output_names,
                dynamic_axes=self.config.dynamic_axes
            )
            
            # Verify exported model
            self._verify_onnx(output_path, dummy_input)
            
        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {str(e)}")
    
    def _verify_onnx(self, output_path: Path, dummy_input: torch.Tensor):
        """Verify ONNX model can be loaded and produces correct output."""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Run inference with ONNX Runtime
            ort_session = ort.InferenceSession(str(output_path))
            ort_inputs = {self.config.input_names[0]: dummy_input.cpu().numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare with PyTorch output
            with torch.no_grad():
                torch_output = self.model(dummy_input).cpu().numpy()
            
            # Check if outputs match
            import numpy as np
            if not np.allclose(ort_outputs[0], torch_output, rtol=1e-3, atol=1e-5):
                warnings.warn("ONNX output differs from PyTorch output")
            else:
                print("✓ ONNX export verified successfully")
                
        except ImportError:
            warnings.warn("onnx/onnxruntime not installed, skipping verification")
        except Exception as e:
            warnings.warn(f"ONNX verification failed: {str(e)}")
    
    def _export_torchscript(self, output_path: Path, dummy_input: torch.Tensor):
        """Export model to TorchScript format."""
        try:
            # Use tracing for export
            traced_model = torch.jit.trace(self.model, dummy_input)
            
            # Optimize if requested
            if self.config.optimize_for_inference:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Apply quantization if requested
            if self.config.quantize:
                traced_model = torch.quantization.quantize_dynamic(
                    traced_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
            
            # Save model
            traced_model.save(str(output_path))
            
            # Verify
            self._verify_torchscript(output_path, dummy_input)
            
        except Exception as e:
            raise RuntimeError(f"TorchScript export failed: {str(e)}")
    
    def _verify_torchscript(self, output_path: Path, dummy_input: torch.Tensor):
        """Verify TorchScript model can be loaded and produces correct output."""
        try:
            # Load traced model
            loaded_model = torch.jit.load(str(output_path))
            
            # Compare outputs
            with torch.no_grad():
                original_output = self.model(dummy_input)
                loaded_output = loaded_model(dummy_input)
            
            if not torch.allclose(original_output, loaded_output, rtol=1e-3, atol=1e-5):
                warnings.warn("TorchScript output differs from original PyTorch output")
            else:
                print("✓ TorchScript export verified successfully")
                
        except Exception as e:
            warnings.warn(f"TorchScript verification failed: {str(e)}")
    
    def _export_state_dict(self, output_path: Path):
        """Export model state dict (PyTorch native format)."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__
        }, output_path)


def export_ensemble(
    models: List[nn.Module],
    output_dir: Union[str, Path],
    config: ExportConfig,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
):
    """
    Export multiple models from an ensemble.
    
    Args:
        models: List of PyTorch models
        output_dir: Directory to save exported models
        config: Export configuration
        input_shape: Shape of input tensor
        device: Device for export
        
    Example:
        >>> models = trainer.models
        >>> export_ensemble(
        ...     models,
        ...     'exported_models/',
        ...     ExportConfig(format='onnx'),
        ...     input_shape=(1, 3, 32, 32)
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, model in enumerate(models):
        exporter = ModelExporter(model, config)
        
        # Determine file extension
        if config.format == 'onnx':
            ext = '.onnx'
        elif config.format == 'torchscript':
            ext = '.pt'
        else:
            ext = '.pth'
        
        output_path = output_dir / f"model_{i}{ext}"
        exporter.export(output_path, input_shape, device)
    
    print(f"\n✓ Exported {len(models)} models to {output_dir}")


def quick_export(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...],
    format: str = 'onnx'
):
    """
    Quick export with default configuration.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save exported model
        input_shape: Shape of input tensor
        format: Export format ('onnx', 'torchscript', 'state_dict')
        
    Example:
        >>> quick_export(model, 'model.onnx', (1, 3, 32, 32))
    """
    config = ExportConfig(format=format)
    exporter = ModelExporter(model, config)
    exporter.export(output_path, input_shape)


__all__ = [
    'ExportConfig',
    'ModelExporter',
    'export_ensemble',
    'quick_export'
]
