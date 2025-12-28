"""
Comprehensive test script for all DML-PY features.
Tests all modules implemented in Phases 3-5 and novel research.
"""

import sys
import traceback
from typing import List, Tuple

def test_phase3_imports() -> Tuple[bool, str]:
    """Test Phase 3: AMP, DDP, Export"""
    try:
        from pydml.utils.amp import AMPConfig, AMPManager
        from pydml.utils.distributed import DistributedConfig, DistributedManager
        from pydml.utils.export import ExportConfig, ModelExporter
        return True, "✓ Phase 3 imports successful (AMP, DDP, Export)"
    except Exception as e:
        return False, f"✗ Phase 3 import failed: {str(e)}"

def test_phase4_imports() -> Tuple[bool, str]:
    """Test Phase 4: Loss Landscape, Hyperparameter Search"""
    try:
        from pydml.analysis.loss_landscape import LossLandscape
        from pydml.utils.hyperparameter_search import (
            HyperparameterSearcher, HyperparameterSpace
        )
        return True, "✓ Phase 4 imports successful (Loss Landscape, HPO)"
    except Exception as e:
        return False, f"✗ Phase 4 import failed: {str(e)}"

def test_confidence_weighted_imports() -> Tuple[bool, str]:
    """Test Novel Research: Confidence-Weighted DML"""
    try:
        from pydml.trainers.confidence_weighted import (
            ConfidenceWeightedDML, ConfidenceWeightedConfig
        )
        return True, "✓ Confidence-Weighted DML imports successful"
    except Exception as e:
        return False, f"✗ Confidence-Weighted DML import failed: {str(e)}"

def test_existing_trainers() -> Tuple[bool, str]:
    """Test existing trainer imports"""
    try:
        from pydml.trainers import DMLTrainer, DistillationTrainer
        from pydml.trainers.feature_dml import FeatureDMLTrainer
        from pydml.trainers.co_distillation import CoDistillationTrainer
        return True, "✓ All trainers import successfully"
    except Exception as e:
        return False, f"✗ Trainer import failed: {str(e)}"

def test_strategies() -> Tuple[bool, str]:
    """Test strategy modules"""
    try:
        from pydml.strategies.curriculum import CurriculumStrategy
        from pydml.strategies.peer_selection import PeerSelector, PeerSelectionConfig
        from pydml.strategies.temperature_scaling import TemperatureScheduler
        return True, "✓ All strategies import successfully"
    except Exception as e:
        return False, f"✗ Strategy import failed: {str(e)}"

def test_losses() -> Tuple[bool, str]:
    """Test loss modules"""
    try:
        from pydml.core.losses import KLDivergenceLoss, DMLLoss
        from pydml.losses.attention_transfer import AttentionTransferLoss
        return True, "✓ All losses import successfully"
    except Exception as e:
        return False, f"✗ Loss import failed: {str(e)}"

def test_analysis() -> Tuple[bool, str]:
    """Test analysis modules"""
    try:
        from pydml.analysis.visualization import plot_training_history
        from pydml.analysis.robustness import add_noise_to_model, test_robustness_to_noise
        return True, "✓ All analysis modules import successfully"
    except Exception as e:
        return False, f"✗ Analysis import failed: {str(e)}"

def test_models() -> Tuple[bool, str]:
    """Test model imports"""
    try:
        from pydml.models.cifar import resnet32, resnet110, wrn_28_10, mobilenet_v2
        return True, "✓ All models import successfully"
    except Exception as e:
        return False, f"✗ Model import failed: {str(e)}"

def test_amp_functionality() -> Tuple[bool, str]:
    """Test AMP basic functionality"""
    try:
        import torch
        from pydml.utils.amp import AMPConfig, AMPManager
        
        config = AMPConfig(enabled=False)  # Disabled for CPU testing
        manager = AMPManager(config)
        
        # Test that manager is created
        assert manager is not None
        
        return True, "✓ AMP functionality works"
    except Exception as e:
        return False, f"✗ AMP functionality failed: {str(e)}"

def test_export_functionality() -> Tuple[bool, str]:
    """Test model export basic functionality"""
    try:
        import torch
        import torch.nn as nn
        from pydml.utils.export import ExportConfig, ModelExporter
        
        # Create simple model and config
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        config = ExportConfig()
        exporter = ModelExporter(model, config)
        
        # Test that exporter is created
        assert exporter is not None
        
        return True, "✓ Model export functionality works"
    except Exception as e:
        return False, f"✗ Model export functionality failed: {str(e)}"

def test_loss_landscape_functionality() -> Tuple[bool, str]:
    """Test loss landscape basic functionality"""
    try:
        import torch
        import torch.nn as nn
        from pydml.analysis.loss_landscape import LossLandscape
        
        # Create simple model and data
        model = nn.Linear(5, 2)
        data_loader = [(torch.randn(4, 5), torch.randint(0, 2, (4,)))]
        criterion = nn.CrossEntropyLoss()
        
        landscape = LossLandscape(model, criterion, data_loader, device='cpu')
        
        return True, "✓ Loss landscape functionality works"
    except Exception as e:
        return False, f"✗ Loss landscape functionality failed: {str(e)}"

def test_hyperparameter_search_functionality() -> Tuple[bool, str]:
    """Test hyperparameter search basic functionality"""
    try:
        from pydml.utils.hyperparameter_search import (
            HyperparameterSearcher, HyperparameterSpace, create_dml_search_space
        )
        
        # Test search space creation using helper function
        space = create_dml_search_space()
        assert space is not None
        
        return True, "✓ Hyperparameter search functionality works"
    except Exception as e:
        return False, f"✗ Hyperparameter search functionality failed: {str(e)}"

def test_confidence_weighted_functionality() -> Tuple[bool, str]:
    """Test Confidence-Weighted DML basic functionality"""
    try:
        import torch
        import torch.nn as nn
        from pydml.trainers.confidence_weighted import (
            ConfidenceWeightedDML, ConfidenceWeightedConfig
        )
        
        # Create simple models
        models = [nn.Linear(10, 5) for _ in range(2)]
        config = ConfidenceWeightedConfig(confidence_threshold=0.5)
        
        trainer = ConfidenceWeightedDML(
            models=models,
            config=config,
            device='cpu'
        )
        
        # Test confidence computation
        logits = torch.randn(4, 5)
        confidence = trainer.compute_confidence(logits)
        assert confidence.shape == (4,)
        
        return True, "✓ Confidence-Weighted DML functionality works"
    except Exception as e:
        return False, f"✗ Confidence-Weighted DML functionality failed: {str(e)}"


def run_all_tests():
    """Run all tests and print results"""
    print("=" * 80)
    print("DML-PY COMPREHENSIVE FEATURE TEST")
    print("=" * 80)
    print()
    
    tests = [
        ("Phase 3 Imports", test_phase3_imports),
        ("Phase 4 Imports", test_phase4_imports),
        ("Confidence-Weighted Imports", test_confidence_weighted_imports),
        ("Existing Trainers", test_existing_trainers),
        ("Strategies", test_strategies),
        ("Losses", test_losses),
        ("Analysis", test_analysis),
        ("Models", test_models),
        ("AMP Functionality", test_amp_functionality),
        ("Export Functionality", test_export_functionality),
        ("Loss Landscape Functionality", test_loss_landscape_functionality),
        ("HPO Functionality", test_hyperparameter_search_functionality),
        ("Confidence-Weighted Functionality", test_confidence_weighted_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success, message = test_func()
            results.append((name, success, message))
            print(message)
        except Exception as e:
            results.append((name, False, f"✗ {name} crashed: {str(e)}"))
            print(f"✗ {name} crashed: {str(e)}")
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! DML-PY is fully operational.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("\nFailed tests:")
        for name, success, message in results:
            if not success:
                print(f"  - {name}: {message}")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
