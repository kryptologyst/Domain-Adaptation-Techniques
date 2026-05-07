#!/usr/bin/env python3
"""Quick test script to verify the domain adaptation package is working."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from domain_adaptation.models import SourceOnlyModel, AdversarialDomainAdaptationModel
        print("✅ Models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import models: {e}")
        return False
    
    try:
        from domain_adaptation.data import create_synthetic_domain_data
        print("✅ Data utilities imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data utilities: {e}")
        return False
    
    try:
        from domain_adaptation.losses import DomainAdaptationLoss
        print("✅ Loss functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import loss functions: {e}")
        return False
    
    try:
        from domain_adaptation.utils import get_device, set_seed
        print("✅ Utilities imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utilities: {e}")
        return False
    
    return True


def test_models():
    """Test model creation and forward pass."""
    print("\nTesting models...")
    
    try:
        import torch
        from domain_adaptation.models import SourceOnlyModel
        
        # Create model
        model = SourceOnlyModel()
        print("✅ SourceOnlyModel created successfully")
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
        print("✅ Forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_data():
    """Test data generation."""
    print("\nTesting data generation...")
    
    try:
        from domain_adaptation.data import create_synthetic_domain_data
        
        # Generate synthetic data
        source_data, source_targets, target_data, target_targets = create_synthetic_domain_data(
            num_samples=100,
            num_classes=10,
            input_dim=784
        )
        
        assert source_data.shape == (100, 784), f"Expected source shape (100, 784), got {source_data.shape}"
        assert target_data.shape == (100, 784), f"Expected target shape (100, 784), got {target_data.shape}"
        assert len(source_targets) == 100, f"Expected 100 source targets, got {len(source_targets)}"
        assert len(target_targets) == 100, f"Expected 100 target targets, got {len(target_targets)}"
        
        print("✅ Synthetic data generation successful")
        return True
        
    except Exception as e:
        print(f"❌ Data test failed: {e}")
        return False


def test_safety():
    """Test safety utilities."""
    print("\nTesting safety utilities...")
    
    try:
        from domain_adaptation.safety import SafetyChecker, EthicsChecker
        
        # Test safety checker
        checker = SafetyChecker()
        print("✅ SafetyChecker created successfully")
        
        # Test ethics checker
        ethics_checker = EthicsChecker()
        print("✅ EthicsChecker created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Safety test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Domain Adaptation Techniques - Package Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_models,
        test_data,
        test_safety,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is working correctly.")
        print("\nNext steps:")
        print("1. Run the interactive demo: python run_demo.py")
        print("2. Try the Jupyter notebook: notebooks/quick_start.ipynb")
        print("3. Run training: python src/train.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
