"""Evaluation metrics for domain adaptation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class DomainAdaptationMetrics:
    """Metrics for domain adaptation evaluation."""
    
    def __init__(self, num_classes: int = 10) -> None:
        """Initialize metrics.
        
        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions: List[int] = []
        self.targets: List[int] = []
        self.probabilities: List[np.ndarray] = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
    ) -> None:
        """Update metrics with new predictions.
        
        Args:
            predictions: Predicted labels.
            targets: True labels.
            probabilities: Prediction probabilities.
        """
        self.predictions.extend(predictions.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of computed metrics.
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(targets, predictions)
        metrics["f1_macro"] = f1_score(targets, predictions, average="macro")
        metrics["f1_weighted"] = f1_score(targets, predictions, average="weighted")
        
        # Per-class F1 scores
        f1_per_class = f1_score(targets, predictions, average=None)
        for i, f1 in enumerate(f1_per_class):
            metrics[f"f1_class_{i}"] = f1
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics["confusion_matrix"] = cm.tolist()
        
        # ROC AUC (if probabilities available)
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            
            if self.num_classes == 2:
                # Binary classification
                metrics["roc_auc"] = roc_auc_score(targets, probabilities[:, 1])
            else:
                # Multi-class classification
                try:
                    metrics["roc_auc_ovr"] = roc_auc_score(
                        targets, probabilities, multi_class="ovr", average="macro"
                    )
                    metrics["roc_auc_ovo"] = roc_auc_score(
                        targets, probabilities, multi_class="ovo", average="macro"
                    )
                except ValueError:
                    # Handle case where some classes are missing
                    pass
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Get detailed classification report.
        
        Returns:
            Classification report string.
        """
        if not self.predictions:
            return "No predictions available."
        
        return classification_report(
            self.targets,
            self.predictions,
            target_names=[f"Class_{i}" for i in range(self.num_classes)],
        )


class DomainShiftMetrics:
    """Metrics for measuring domain shift."""
    
    @staticmethod
    def compute_mmd(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
    ) -> float:
        """Compute Maximum Mean Discrepancy (MMD).
        
        Args:
            source_features: Source domain features.
            target_features: Target domain features.
            kernel_type: Type of kernel.
            gamma: Kernel parameter.
            
        Returns:
            MMD value.
        """
        if kernel_type == "rbf":
            # RBF kernel
            pairwise_dist_ss = torch.cdist(source_features, source_features, p=2) ** 2
            pairwise_dist_tt = torch.cdist(target_features, target_features, p=2) ** 2
            pairwise_dist_st = torch.cdist(source_features, target_features, p=2) ** 2
            
            k_ss = torch.exp(-gamma * pairwise_dist_ss)
            k_tt = torch.exp(-gamma * pairwise_dist_tt)
            k_st = torch.exp(-gamma * pairwise_dist_st)
        else:
            # Linear kernel
            k_ss = torch.mm(source_features, source_features.t())
            k_tt = torch.mm(target_features, target_features.t())
            k_st = torch.mm(source_features, target_features.t())
        
        n_s = source_features.size(0)
        n_t = target_features.size(0)
        
        # Remove diagonal terms
        k_ss = k_ss - torch.diag(torch.diag(k_ss))
        k_tt = k_tt - torch.diag(torch.diag(k_tt))
        
        mmd = (k_ss.sum() / (n_s * (n_s - 1)) + 
               k_tt.sum() / (n_t * (n_t - 1)) - 
               2 * k_st.sum() / (n_s * n_t))
        
        return mmd.item()
    
    @staticmethod
    def compute_coral_distance(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> float:
        """Compute CORAL distance.
        
        Args:
            source_features: Source domain features.
            target_features: Target domain features.
            
        Returns:
            CORAL distance.
        """
        # Compute covariance matrices
        source_cov = torch.mm(source_features.t(), source_features) / (source_features.size(0) - 1)
        target_cov = torch.mm(target_features.t(), target_features) / (target_features.size(0) - 1)
        
        # Compute Frobenius norm
        distance = torch.norm(source_cov - target_cov, p='fro') ** 2
        distance = distance / (4 * source_features.size(1) ** 2)
        
        return distance.item()
    
    @staticmethod
    def compute_wasserstein_distance(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> float:
        """Compute Wasserstein distance (approximation).
        
        Args:
            source_features: Source domain features.
            target_features: Target domain features.
            
        Returns:
            Wasserstein distance approximation.
        """
        # Compute mean and covariance
        source_mean = source_features.mean(dim=0)
        target_mean = target_features.mean(dim=0)
        
        source_cov = torch.mm((source_features - source_mean).t(), 
                              (source_features - source_mean)) / (source_features.size(0) - 1)
        target_cov = torch.mm((target_features - target_mean).t(), 
                              (target_features - target_mean)) / (target_features.size(0) - 1)
        
        # Mean difference
        mean_diff = torch.norm(source_mean - target_mean, p=2) ** 2
        
        # Covariance difference (trace)
        cov_diff = torch.trace(source_cov + target_cov - 2 * torch.sqrtm(source_cov @ target_cov))
        
        return (mean_diff + cov_diff).item()


class AdversarialMetrics:
    """Metrics for adversarial training evaluation."""
    
    @staticmethod
    def compute_domain_classification_accuracy(
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> float:
        """Compute domain classification accuracy.
        
        Args:
            domain_logits: Domain classification logits.
            domain_labels: True domain labels.
            
        Returns:
            Domain classification accuracy.
        """
        predictions = (torch.sigmoid(domain_logits) > 0.5).float()
        accuracy = (predictions == domain_labels.float()).float().mean()
        return accuracy.item()
    
    @staticmethod
    def compute_domain_confusion(
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute domain confusion metrics.
        
        Args:
            domain_logits: Domain classification logits.
            domain_labels: True domain labels.
            
        Returns:
            Dictionary of confusion metrics.
        """
        predictions = (torch.sigmoid(domain_logits) > 0.5).float()
        
        # True positives, false positives, etc.
        tp = ((predictions == 1) & (domain_labels == 1)).float().sum()
        tn = ((predictions == 0) & (domain_labels == 0)).float().sum()
        fp = ((predictions == 1) & (domain_labels == 0)).float().sum()
        fn = ((predictions == 0) & (domain_labels == 1)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "accuracy": ((tp + tn) / (tp + tn + fp + fn)).item(),
        }


def compute_all_metrics(
    model: torch.nn.Module,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    device: torch.device,
    extract_features: bool = True,
) -> Dict[str, float]:
    """Compute comprehensive metrics for domain adaptation.
    
    Args:
        model: Trained model.
        source_loader: Source domain data loader.
        target_loader: Target domain data loader.
        device: Device to run on.
        extract_features: Whether to extract features for domain shift metrics.
        
    Returns:
        Dictionary of all computed metrics.
    """
    model.eval()
    
    all_metrics = {}
    
    # Source domain metrics
    source_metrics = DomainAdaptationMetrics()
    source_features_list = []
    
    with torch.no_grad():
        for data, targets, _ in source_loader:
            data, targets = data.to(device), targets.to(device)
            
            if extract_features:
                features = model.extract_features(data)
                source_features_list.append(features)
            
            if hasattr(model, 'forward') and len(model.forward(data)) == 2:
                # Adversarial model
                logits, _ = model(data)
            else:
                # Simple model
                logits = model(data)
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            source_metrics.update(predictions, targets, probabilities)
    
    source_results = source_metrics.compute()
    for key, value in source_results.items():
        all_metrics[f"source_{key}"] = value
    
    # Target domain metrics
    target_metrics = DomainAdaptationMetrics()
    target_features_list = []
    
    with torch.no_grad():
        for data, targets, _ in target_loader:
            data, targets = data.to(device), targets.to(device)
            
            if extract_features:
                features = model.extract_features(data)
                target_features_list.append(features)
            
            if hasattr(model, 'forward') and len(model.forward(data)) == 2:
                # Adversarial model
                logits, _ = model(data)
            else:
                # Simple model
                logits = model(data)
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            target_metrics.update(predictions, targets, probabilities)
    
    target_results = target_metrics.compute()
    for key, value in target_results.items():
        all_metrics[f"target_{key}"] = value
    
    # Domain shift metrics
    if extract_features and source_features_list and target_features_list:
        source_features = torch.cat(source_features_list, dim=0)
        target_features = torch.cat(target_features_list, dim=0)
        
        all_metrics["mmd_distance"] = DomainShiftMetrics.compute_mmd(
            source_features, target_features
        )
        all_metrics["coral_distance"] = DomainShiftMetrics.compute_coral_distance(
            source_features, target_features
        )
        all_metrics["wasserstein_distance"] = DomainShiftMetrics.compute_wasserstein_distance(
            source_features, target_features
        )
    
    return all_metrics
