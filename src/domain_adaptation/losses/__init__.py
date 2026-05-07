"""Loss functions for domain adaptation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """Adversarial loss for domain adaptation."""
    
    def __init__(self, reduction: str = "mean") -> None:
        """Initialize adversarial loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(
        self,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adversarial loss.
        
        Args:
            domain_logits: Domain classification logits.
            domain_labels: Domain labels (0 for source, 1 for target).
            
        Returns:
            Adversarial loss.
        """
        return self.bce_loss(domain_logits, domain_labels.float())


class CORALLoss(nn.Module):
    """CORAL (CORrelation ALignment) loss."""
    
    def __init__(self) -> None:
        """Initialize CORAL loss."""
        super().__init__()
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CORAL loss.
        
        Args:
            source_features: Source domain features.
            target_features: Target domain features.
            
        Returns:
            CORAL loss.
        """
        # Compute covariance matrices
        source_cov = torch.mm(source_features.t(), source_features) / (source_features.size(0) - 1)
        target_cov = torch.mm(target_features.t(), target_features) / (target_features.size(0) - 1)
        
        # Compute Frobenius norm
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        loss = loss / (4 * source_features.size(1) ** 2)
        
        return loss


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy (MMD) loss."""
    
    def __init__(self, kernel_type: str = "rbf", gamma: Optional[float] = None) -> None:
        """Initialize MMD loss.
        
        Args:
            kernel_type: Type of kernel ('rbf', 'linear', 'poly').
            gamma: Kernel parameter for RBF kernel.
        """
        super().__init__()
        self.kernel_type = kernel_type
        self.gamma = gamma or 1.0
    
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel.
        
        Args:
            x: First tensor.
            y: Second tensor.
            
        Returns:
            Kernel matrix.
        """
        pairwise_dist = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-self.gamma * pairwise_dist)
    
    def linear_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute linear kernel.
        
        Args:
            x: First tensor.
            y: Second tensor.
            
        Returns:
            Kernel matrix.
        """
        return torch.mm(x, y.t())
    
    def poly_kernel(self, x: torch.Tensor, y: torch.Tensor, degree: int = 3) -> torch.Tensor:
        """Compute polynomial kernel.
        
        Args:
            x: First tensor.
            y: Second tensor.
            degree: Polynomial degree.
            
        Returns:
            Kernel matrix.
        """
        return torch.pow(torch.mm(x, y.t()) + 1, degree)
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MMD loss.
        
        Args:
            source_features: Source domain features.
            target_features: Target domain features.
            
        Returns:
            MMD loss.
        """
        # Compute kernel matrices
        if self.kernel_type == "rbf":
            k_ss = self.rbf_kernel(source_features, source_features)
            k_tt = self.rbf_kernel(target_features, target_features)
            k_st = self.rbf_kernel(source_features, target_features)
        elif self.kernel_type == "linear":
            k_ss = self.linear_kernel(source_features, source_features)
            k_tt = self.linear_kernel(target_features, target_features)
            k_st = self.linear_kernel(source_features, target_features)
        elif self.kernel_type == "poly":
            k_ss = self.poly_kernel(source_features, source_features)
            k_tt = self.poly_kernel(target_features, target_features)
            k_st = self.poly_kernel(source_features, target_features)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Compute MMD
        n_s = source_features.size(0)
        n_t = target_features.size(0)
        
        # Remove diagonal terms
        k_ss = k_ss - torch.diag(torch.diag(k_ss))
        k_tt = k_tt - torch.diag(torch.diag(k_tt))
        
        mmd = (k_ss.sum() / (n_s * (n_s - 1)) + 
               k_tt.sum() / (n_t * (n_t - 1)) - 
               2 * k_st.sum() / (n_s * n_t))
        
        return mmd


class DomainAdaptationLoss(nn.Module):
    """Combined domain adaptation loss."""
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        coral_weight: float = 0.0,
        mmd_weight: float = 0.0,
        mmd_kernel: str = "rbf",
        mmd_gamma: Optional[float] = None,
    ) -> None:
        """Initialize domain adaptation loss.
        
        Args:
            classification_weight: Weight for classification loss.
            adversarial_weight: Weight for adversarial loss.
            coral_weight: Weight for CORAL loss.
            mmd_weight: Weight for MMD loss.
            mmd_kernel: MMD kernel type.
            mmd_gamma: MMD kernel parameter.
        """
        super().__init__()
        
        self.classification_weight = classification_weight
        self.adversarial_weight = adversarial_weight
        self.coral_weight = coral_weight
        self.mmd_weight = mmd_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.adversarial_loss = AdversarialLoss()
        
        if coral_weight > 0:
            self.coral_loss = CORALLoss()
        
        if mmd_weight > 0:
            self.mmd_loss = MMDLoss(kernel_type=mmd_kernel, gamma=mmd_gamma)
    
    def forward(
        self,
        label_logits: torch.Tensor,
        domain_logits: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        source_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined domain adaptation loss.
        
        Args:
            label_logits: Label classification logits.
            domain_logits: Domain classification logits.
            labels: True labels.
            domain_labels: Domain labels.
            source_features: Source domain features (for CORAL/MMD).
            target_features: Target domain features (for CORAL/MMD).
            
        Returns:
            Combined loss.
        """
        # Classification loss
        cls_loss = self.classification_loss(label_logits, labels)
        
        # Adversarial loss
        adv_loss = self.adversarial_loss(domain_logits, domain_labels)
        
        total_loss = (self.classification_weight * cls_loss + 
                     self.adversarial_weight * adv_loss)
        
        # CORAL loss
        if self.coral_weight > 0 and source_features is not None and target_features is not None:
            coral_loss = self.coral_loss(source_features, target_features)
            total_loss += self.coral_weight * coral_loss
        
        # MMD loss
        if self.mmd_weight > 0 and source_features is not None and target_features is not None:
            mmd_loss = self.mmd_loss(source_features, target_features)
            total_loss += self.mmd_weight * mmd_loss
        
        return total_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for domain adaptation."""
    
    def __init__(self, margin: float = 1.0) -> None:
        """Initialize contrastive loss.
        
        Args:
            margin: Margin for contrastive loss.
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            features1: First set of features.
            features2: Second set of features.
            labels1: Labels for first set.
            labels2: Labels for second set.
            
        Returns:
            Contrastive loss.
        """
        # Compute pairwise distances
        distances = torch.cdist(features1, features2, p=2)
        
        # Create similarity matrix
        similarity = (labels1.unsqueeze(1) == labels2.unsqueeze(0)).float()
        
        # Contrastive loss
        positive_loss = similarity * distances.pow(2)
        negative_loss = (1 - similarity) * F.relu(self.margin - distances).pow(2)
        
        return (positive_loss + negative_loss).mean()


class TripletLoss(nn.Module):
    """Triplet loss for domain adaptation."""
    
    def __init__(self, margin: float = 1.0) -> None:
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss.
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor features.
            positive: Positive features.
            negative: Negative features.
            
        Returns:
            Triplet loss.
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
