"""Test suite for domain adaptation models."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from domain_adaptation.models import (
    AdversarialDomainAdaptationModel,
    CORALModel,
    SourceOnlyModel,
    TargetOnlyModel,
)
from domain_adaptation.losses import DomainAdaptationLoss, CORALLoss, MMDLoss
from domain_adaptation.data import DomainDataset, PairedDomainDataset
from domain_adaptation.metrics import DomainAdaptationMetrics
from domain_adaptation.safety import SafetyChecker, EthicsChecker, ComplianceManager


class TestModels:
    """Test model architectures."""
    
    def test_source_only_model(self):
        """Test source-only model."""
        model = SourceOnlyModel()
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_adversarial_model(self):
        """Test adversarial domain adaptation model."""
        model = AdversarialDomainAdaptationModel()
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        label_logits, domain_logits = model(x)
        
        assert label_logits.shape == (2, 10)
        assert domain_logits.shape == (2, 1)
        assert not torch.isnan(label_logits).any()
        assert not torch.isnan(domain_logits).any()
    
    def test_coral_model(self):
        """Test CORAL model."""
        model = CORALModel()
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        
        assert output.shape == (2, 10)
        
        # Test CORAL loss
        source_features = torch.randn(10, 128)
        target_features = torch.randn(10, 128)
        coral_loss = model.coral_loss(source_features, target_features)
        
        assert coral_loss >= 0
        assert not torch.isnan(coral_loss)


class TestLosses:
    """Test loss functions."""
    
    def test_coral_loss(self):
        """Test CORAL loss."""
        loss_fn = CORALLoss()
        
        source_features = torch.randn(10, 128)
        target_features = torch.randn(10, 128)
        
        loss = loss_fn(source_features, target_features)
        
        assert loss >= 0
        assert not torch.isnan(loss)
    
    def test_mmd_loss(self):
        """Test MMD loss."""
        loss_fn = MMDLoss()
        
        source_features = torch.randn(10, 128)
        target_features = torch.randn(10, 128)
        
        loss = loss_fn(source_features, target_features)
        
        assert loss >= 0
        assert not torch.isnan(loss)
    
    def test_domain_adaptation_loss(self):
        """Test combined domain adaptation loss."""
        loss_fn = DomainAdaptationLoss()
        
        label_logits = torch.randn(10, 10)
        domain_logits = torch.randn(10, 1)
        labels = torch.randint(0, 10, (10,))
        domain_labels = torch.randint(0, 2, (10, 1))
        
        loss = loss_fn(label_logits, domain_logits, labels, domain_labels)
        
        assert loss >= 0
        assert not torch.isnan(loss)


class TestData:
    """Test data utilities."""
    
    def test_domain_dataset(self):
        """Test domain dataset."""
        data = torch.randn(10, 1, 28, 28)
        targets = torch.randint(0, 10, (10,))
        domain_labels = torch.randint(0, 2, (10,))
        
        dataset = DomainDataset(data, targets, domain_labels)
        
        assert len(dataset) == 10
        
        # Test getitem
        sample_data, sample_target, sample_domain = dataset[0]
        assert sample_data.shape == (1, 28, 28)
        assert isinstance(sample_target, torch.Tensor)
        assert isinstance(sample_domain, torch.Tensor)
    
    def test_paired_domain_dataset(self):
        """Test paired domain dataset."""
        source_data = torch.randn(10, 1, 28, 28)
        source_targets = torch.randint(0, 10, (10,))
        target_data = torch.randn(8, 1, 28, 28)  # Different size
        target_targets = torch.randint(0, 10, (8,))
        
        dataset = PairedDomainDataset(
            source_data, source_targets, target_data, target_targets
        )
        
        assert len(dataset) == 8  # Minimum size
        
        # Test getitem
        src_data, src_target, tgt_data, tgt_target = dataset[0]
        assert src_data.shape == (1, 28, 28)
        assert tgt_data.shape == (1, 28, 28)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_domain_adaptation_metrics(self):
        """Test domain adaptation metrics."""
        metrics = DomainAdaptationMetrics()
        
        # Add some predictions
        predictions = torch.randint(0, 10, (100,))
        targets = torch.randint(0, 10, (100,))
        probabilities = torch.randn(100, 10)
        
        metrics.update(predictions, targets, probabilities)
        
        # Compute metrics
        results = metrics.compute()
        
        assert "accuracy" in results
        assert "f1_macro" in results
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["f1_macro"] <= 1


class TestSafety:
    """Test safety utilities."""
    
    def test_safety_checker(self):
        """Test safety checker."""
        checker = SafetyChecker()
        
        # Create a simple model
        model = nn.Linear(10, 5)
        
        # Check safety
        results = checker.check_model_safety(model)
        
        assert "is_safe" in results
        assert "warnings" in results
        assert "errors" in results
        assert "recommendations" in results
    
    def test_ethics_checker(self):
        """Test ethics checker."""
        checker = EthicsChecker()
        
        # Check general domain
        results = checker.check_application_ethics("general", "research")
        assert results["is_ethical"] is True
        assert results["risk_level"] == "low"
        
        # Check sensitive domain
        results = checker.check_application_ethics("medical", "diagnosis")
        assert results["is_ethical"] is False
        assert results["risk_level"] == "high"
    
    def test_compliance_manager(self):
        """Test compliance manager."""
        manager = ComplianceManager()
        
        # Create a simple model
        model = nn.Linear(10, 5)
        
        # Validate project
        results = manager.validate_project(model, "general", "research")
        
        assert "overall_compliant" in results
        assert "safety" in results
        assert "ethics" in results
        assert "recommendations" in results


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training simulation."""
        # Create model
        model = SourceOnlyModel()
        
        # Create dummy data
        data = torch.randn(100, 1, 28, 28)
        targets = torch.randint(0, 10, (100,))
        
        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        
        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for batch_data, batch_targets in dataloader:
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_targets)
            loss.backward()
            optimizer.step()
            
            # Check for NaN
            assert not torch.isnan(loss).any()
            break  # Just test one iteration
    
    def test_model_comparison(self):
        """Test model comparison."""
        models = {
            "source_only": SourceOnlyModel(),
            "target_only": TargetOnlyModel(),
            "adversarial": AdversarialDomainAdaptationModel(),
            "coral": CORALModel(),
        }
        
        # Test all models can be instantiated and run
        x = torch.randn(2, 1, 28, 28)
        
        for name, model in models.items():
            if name == "adversarial":
                output = model(x)
                assert len(output) == 2  # label_logits, domain_logits
            else:
                output = model(x)
                assert output.shape == (2, 10)


if __name__ == "__main__":
    pytest.main([__file__])
