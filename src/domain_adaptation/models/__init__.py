"""Model architectures for domain adaptation."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Feature extractor for domain adaptation."""
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        """Initialize feature extractor.
        
        Args:
            input_channels: Number of input channels.
            feature_dim: Dimension of output features.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened size
        self.flattened_size = 128 * 3 * 3  # For 28x28 input
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Feature tensor.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class LabelClassifier(nn.Module):
    """Label classifier for domain adaptation."""
    
    def __init__(
        self,
        feature_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        """Initialize label classifier.
        
        Args:
            feature_dim: Dimension of input features.
            num_classes: Number of classes.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Input features.
            
        Returns:
            Classification logits.
        """
        return self.classifier(features)


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial training."""
    
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        gradient_reversal: bool = True,
    ) -> None:
        """Initialize domain classifier.
        
        Args:
            feature_dim: Dimension of input features.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
            gradient_reversal: Whether to use gradient reversal layer.
        """
        super().__init__()
        
        self.gradient_reversal = gradient_reversal
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Input features.
            
        Returns:
            Domain classification logits.
        """
        return self.classifier(features)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            alpha: Reversal strength.
            
        Returns:
            Input tensor unchanged.
        """
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass with gradient reversal.
        
        Args:
            grad_output: Gradient from next layer.
            
        Returns:
            Reversed gradient and None for alpha.
        """
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer module."""
    
    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize gradient reversal layer.
        
        Args:
            alpha: Reversal strength.
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor with reversed gradients.
        """
        return GradientReversalFunction.apply(x, self.alpha)


class AdversarialDomainAdaptationModel(nn.Module):
    """Complete adversarial domain adaptation model."""
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.5,
        gradient_reversal_alpha: float = 1.0,
    ) -> None:
        """Initialize adversarial domain adaptation model.
        
        Args:
            input_channels: Number of input channels.
            feature_dim: Dimension of features.
            num_classes: Number of classes.
            dropout: Dropout rate.
            gradient_reversal_alpha: Gradient reversal strength.
        """
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        
        self.label_classifier = LabelClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.domain_classifier = DomainClassifier(
            feature_dim=feature_dim,
            dropout=dropout,
        )
        
        self.gradient_reversal = GradientReversalLayer(gradient_reversal_alpha)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor.
            return_features: Whether to return features.
            
        Returns:
            Tuple of (label_logits, domain_logits) or (label_logits, domain_logits, features).
        """
        features = self.feature_extractor(x)
        
        label_logits = self.label_classifier(features)
        
        # Apply gradient reversal for domain classification
        reversed_features = self.gradient_reversal(features)
        domain_logits = self.domain_classifier(reversed_features)
        
        if return_features:
            return label_logits, domain_logits, features
        
        return label_logits, domain_logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input.
        
        Args:
            x: Input tensor.
            
        Returns:
            Extracted features.
        """
        return self.feature_extractor(x)


class SourceOnlyModel(nn.Module):
    """Source-only baseline model."""
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        """Initialize source-only model.
        
        Args:
            input_channels: Number of input channels.
            feature_dim: Dimension of features.
            num_classes: Number of classes.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        
        self.classifier = LabelClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Classification logits.
        """
        features = self.feature_extractor(x)
        return self.classifier(features)


class TargetOnlyModel(nn.Module):
    """Target-only baseline model."""
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        """Initialize target-only model.
        
        Args:
            input_channels: Number of input channels.
            feature_dim: Dimension of features.
            num_classes: Number of classes.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        
        self.classifier = LabelClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Classification logits.
        """
        features = self.feature_extractor(x)
        return self.classifier(features)


class CORALModel(nn.Module):
    """CORAL (CORrelation ALignment) domain adaptation model."""
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.5,
    ) -> None:
        """Initialize CORAL model.
        
        Args:
            input_channels: Number of input channels.
            feature_dim: Dimension of features.
            num_classes: Number of classes.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        
        self.classifier = LabelClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Classification logits.
        """
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def coral_loss(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
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
