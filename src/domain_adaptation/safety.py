"""Safety and compliance utilities for domain adaptation."""

import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class SafetyChecker:
    """Safety checker for domain adaptation models."""
    
    def __init__(self) -> None:
        """Initialize safety checker."""
        self.warnings_log: List[str] = []
        self.errors_log: List[str] = []
    
    def check_model_safety(self, model: nn.Module) -> Dict[str, Any]:
        """Check model for safety issues.
        
        Args:
            model: Model to check.
            
        Returns:
            Dictionary of safety check results.
        """
        results = {
            "is_safe": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check for NaN parameters
        nan_params = self._check_nan_parameters(model)
        if nan_params:
            results["errors"].extend(nan_params)
            results["is_safe"] = False
        
        # Check for infinite parameters
        inf_params = self._check_infinite_parameters(model)
        if inf_params:
            results["errors"].extend(inf_params)
            results["is_safe"] = False
        
        # Check parameter ranges
        range_warnings = self._check_parameter_ranges(model)
        results["warnings"].extend(range_warnings)
        
        # Check for potential overfitting
        overfitting_warnings = self._check_overfitting_indicators(model)
        results["warnings"].extend(overfitting_warnings)
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        return results
    
    def _check_nan_parameters(self, model: nn.Module) -> List[str]:
        """Check for NaN parameters."""
        errors = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                errors.append(f"NaN detected in parameter: {name}")
        return errors
    
    def _check_infinite_parameters(self, model: nn.Module) -> List[str]:
        """Check for infinite parameters."""
        errors = []
        for name, param in model.named_parameters():
            if torch.isinf(param).any():
                errors.append(f"Infinite value detected in parameter: {name}")
        return errors
    
    def _check_parameter_ranges(self, model: nn.Module) -> List[str]:
        """Check parameter value ranges."""
        warnings = []
        for name, param in model.named_parameters():
            if param.abs().max() > 100:
                warnings.append(f"Large parameter values in {name}: max={param.abs().max():.2f}")
            if param.std() > 10:
                warnings.append(f"High parameter variance in {name}: std={param.std():.2f}")
        return warnings
    
    def _check_overfitting_indicators(self, model: nn.Module) -> List[str]:
        """Check for overfitting indicators."""
        warnings = []
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 1_000_000:
            warnings.append(f"Large model with {total_params:,} parameters - risk of overfitting")
        
        return warnings
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if results["errors"]:
            recommendations.append("Fix all errors before using the model")
        
        if results["warnings"]:
            recommendations.append("Review warnings and consider model validation")
        
        recommendations.extend([
            "Validate model on held-out test data",
            "Monitor model performance in production",
            "Implement proper error handling",
            "Use uncertainty quantification when possible"
        ])
        
        return recommendations


class EthicsChecker:
    """Ethics checker for domain adaptation applications."""
    
    def __init__(self) -> None:
        """Initialize ethics checker."""
        self.sensitive_domains = [
            "medical", "healthcare", "diagnosis", "treatment",
            "financial", "banking", "trading", "credit",
            "biometric", "surveillance", "recognition",
            "legal", "judicial", "criminal",
            "social", "demographic", "personal"
        ]
    
    def check_application_ethics(self, domain: str, use_case: str) -> Dict[str, Any]:
        """Check ethics of domain adaptation application.
        
        Args:
            domain: Application domain.
            use_case: Specific use case.
            
        Returns:
            Dictionary of ethics check results.
        """
        results = {
            "is_ethical": True,
            "concerns": [],
            "recommendations": [],
            "risk_level": "low"
        }
        
        domain_lower = domain.lower()
        use_case_lower = use_case.lower()
        
        # Check for sensitive domains
        for sensitive in self.sensitive_domains:
            if sensitive in domain_lower or sensitive in use_case_lower:
                results["concerns"].append(f"Sensitive domain detected: {sensitive}")
                results["risk_level"] = "high"
                results["is_ethical"] = False
        
        # Generate recommendations based on risk level
        if results["risk_level"] == "high":
            results["recommendations"].extend([
                "Require human oversight and validation",
                "Implement bias detection and mitigation",
                "Ensure proper consent and privacy protection",
                "Conduct thorough impact assessment",
                "Consider alternative approaches"
            ])
        else:
            results["recommendations"].extend([
                "Monitor for bias and fairness",
                "Validate on diverse test sets",
                "Document limitations and assumptions",
                "Regular performance monitoring"
            ])
        
        return results


class ComplianceManager:
    """Compliance manager for domain adaptation projects."""
    
    def __init__(self) -> None:
        """Initialize compliance manager."""
        self.safety_checker = SafetyChecker()
        self.ethics_checker = EthicsChecker()
    
    def validate_project(
        self,
        model: nn.Module,
        domain: str = "general",
        use_case: str = "research",
    ) -> Dict[str, Any]:
        """Validate project for safety and compliance.
        
        Args:
            model: Model to validate.
            domain: Application domain.
            use_case: Specific use case.
            
        Returns:
            Comprehensive validation results.
        """
        results = {
            "overall_compliant": True,
            "safety": self.safety_checker.check_model_safety(model),
            "ethics": self.ethics_checker.check_application_ethics(domain, use_case),
            "recommendations": []
        }
        
        # Overall compliance
        if not results["safety"]["is_safe"] or not results["ethics"]["is_ethical"]:
            results["overall_compliant"] = False
        
        # Combine recommendations
        results["recommendations"].extend(results["safety"]["recommendations"])
        results["recommendations"].extend(results["ethics"]["recommendations"])
        
        return results
    
    def generate_compliance_report(
        self,
        validation_results: Dict[str, Any],
        project_name: str = "Domain Adaptation Project",
    ) -> str:
        """Generate compliance report.
        
        Args:
            validation_results: Results from validate_project.
            project_name: Name of the project.
            
        Returns:
            Formatted compliance report.
        """
        report = f"""
COMPLIANCE REPORT: {project_name}
{'=' * 50}

OVERALL STATUS: {'✅ COMPLIANT' if validation_results['overall_compliant'] else '❌ NON-COMPLIANT'}

SAFETY CHECK:
- Status: {'✅ PASS' if validation_results['safety']['is_safe'] else '❌ FAIL'}
- Warnings: {len(validation_results['safety']['warnings'])}
- Errors: {len(validation_results['safety']['errors'])}

ETHICS CHECK:
- Status: {'✅ PASS' if validation_results['ethics']['is_ethical'] else '❌ FAIL'}
- Risk Level: {validation_results['ethics']['risk_level'].upper()}
- Concerns: {len(validation_results['ethics']['concerns'])}

RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(validation_results['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
DISCLAIMER:
This is a research/educational project. Not intended for production use.
Human oversight required for any critical applications.
Use at your own risk.

Generated by Domain Adaptation Safety Framework
"""
        
        return report


def add_safety_warnings(func):
    """Decorator to add safety warnings to functions."""
    def wrapper(*args, **kwargs):
        warnings.warn(
            "This function is for research/educational purposes only. "
            "Not intended for production use without proper validation.",
            UserWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper


def validate_input_data(data: torch.Tensor, expected_shape: Optional[tuple] = None) -> bool:
    """Validate input data for safety.
    
    Args:
        data: Input data tensor.
        expected_shape: Expected data shape.
        
    Returns:
        True if data is valid.
    """
    # Check for NaN or infinite values
    if torch.isnan(data).any() or torch.isinf(data).any():
        raise ValueError("Input data contains NaN or infinite values")
    
    # Check shape if provided
    if expected_shape and data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")
    
    return True


def sanitize_predictions(predictions: torch.Tensor, confidence_threshold: float = 0.5) -> torch.Tensor:
    """Sanitize predictions for safety.
    
    Args:
        predictions: Model predictions.
        confidence_threshold: Minimum confidence threshold.
        
    Returns:
        Sanitized predictions.
    """
    # Ensure predictions are valid
    if torch.isnan(predictions).any() or torch.isinf(predictions).any():
        warnings.warn("Invalid predictions detected, returning zeros")
        return torch.zeros_like(predictions)
    
    # Apply confidence threshold for probabilities
    if predictions.dim() > 1 and predictions.size(1) > 1:
        probs = torch.softmax(predictions, dim=1)
        max_probs = probs.max(dim=1)[0]
        
        # Mask low confidence predictions
        low_confidence = max_probs < confidence_threshold
        if low_confidence.any():
            warnings.warn(f"Low confidence predictions detected: {low_confidence.sum().item()} samples")
    
    return predictions
