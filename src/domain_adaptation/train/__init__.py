"""Training utilities for domain adaptation models."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import DomainAdaptationLoss
from ..metrics import DomainAdaptationMetrics, compute_all_metrics
from ..utils import EarlyStopping, get_device, log_metrics


class DomainAdaptationTrainer:
    """Trainer for domain adaptation models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_type: str = "step",
        scheduler_params: Optional[Dict] = None,
        early_stopping: Optional[EarlyStopping] = None,
        log_interval: int = 100,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train.
            device: Device to train on.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            scheduler_type: Type of learning rate scheduler.
            scheduler_params: Parameters for scheduler.
            early_stopping: Early stopping utility.
            log_interval: Logging interval.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        if scheduler_type == "step":
            scheduler_params = scheduler_params or {"step_size": 30, "gamma": 0.1}
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **scheduler_params
            )
        elif scheduler_type == "cosine":
            scheduler_params = scheduler_params or {"T_max": 100}
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **scheduler_params
            )
        else:
            self.scheduler = None
        
        self.early_stopping = early_stopping
        self.log_interval = log_interval
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_metrics: List[Dict[str, float]] = []
        self.val_metrics: List[Dict[str, float]] = []
    
    def train_epoch(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        loss_fn: nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            source_loader: Source domain data loader.
            target_loader: Target domain data loader.
            loss_fn: Loss function.
            epoch: Current epoch.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Create iterators
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Get minimum length
        min_len = min(len(source_loader), len(target_loader))
        
        progress_bar = tqdm(range(min_len), desc=f"Epoch {epoch}")
        
        for batch_idx in progress_bar:
            try:
                # Get batches
                source_data, source_targets, _ = next(source_iter)
                target_data, target_targets, _ = next(target_iter)
                
                source_data = source_data.to(self.device)
                source_targets = source_targets.to(self.device)
                target_data = target_data.to(self.device)
                target_targets = target_targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if hasattr(self.model, 'forward') and len(self.model.forward(source_data)) == 2:
                    # Adversarial model
                    source_logits, source_domain_logits = self.model(source_data)
                    target_logits, target_domain_logits = self.model(target_data)
                    
                    # Create domain labels
                    source_domain_labels = torch.zeros(source_data.size(0), 1).to(self.device)
                    target_domain_labels = torch.ones(target_data.size(0), 1).to(self.device)
                    
                    # Compute loss
                    loss = loss_fn(
                        source_logits, source_domain_logits,
                        source_targets, source_domain_labels,
                        self.model.extract_features(source_data),
                        self.model.extract_features(target_data),
                    )
                else:
                    # Simple model - train on source only
                    source_logits = self.model(source_data)
                    loss = loss_fn(source_logits, source_targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Log metrics
                if batch_idx % self.log_interval == 0:
                    log_metrics(
                        {"train_loss": loss.item()},
                        epoch * min_len + batch_idx,
                        "train",
                    )
                
            except StopIteration:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {"train_loss": avg_loss}
    
    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            loss_fn: Loss function.
            epoch: Current epoch.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        metrics = DomainAdaptationMetrics()
        
        with torch.no_grad():
            for data, targets, _ in val_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                if hasattr(self.model, 'forward') and len(self.model.forward(data)) == 2:
                    # Adversarial model
                    logits, domain_logits = self.model(data)
                    loss = loss_fn(logits, targets)
                else:
                    # Simple model
                    logits = self.model(data)
                    loss = loss_fn(logits, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update metrics
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                metrics.update(predictions, targets, probabilities)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_metrics = metrics.compute()
        val_metrics["val_loss"] = avg_loss
        
        return val_metrics
    
    def train(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        loss_fn: Optional[nn.Module] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            source_loader: Source domain data loader.
            target_loader: Target domain data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            loss_fn: Loss function.
            save_path: Path to save the model.
            
        Returns:
            Dictionary of training history.
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(source_loader, target_loader, loss_fn, epoch)
            self.train_losses.append(train_metrics["train_loss"])
            self.train_metrics.append(train_metrics)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, loss_fn, epoch)
                self.val_losses.append(val_metrics["val_loss"])
                self.val_metrics.append(val_metrics)
                
                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_metrics["accuracy"], self.model):
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch metrics
            log_metrics(train_metrics, epoch, "epoch")
            if val_loader is not None:
                log_metrics(val_metrics, epoch, "epoch")
        
        # Save model
        if save_path:
            torch.save(self.model.state_dict(), save_path)
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
        }


class AdversarialTrainer(DomainAdaptationTrainer):
    """Specialized trainer for adversarial domain adaptation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        adversarial_weight: float = 1.0,
        **kwargs,
    ) -> None:
        """Initialize adversarial trainer.
        
        Args:
            model: Adversarial model to train.
            device: Device to train on.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            adversarial_weight: Weight for adversarial loss.
            **kwargs: Additional arguments for base trainer.
        """
        super().__init__(model, device, learning_rate, weight_decay, **kwargs)
        
        self.adversarial_weight = adversarial_weight
        
        # Separate optimizers for feature extractor and domain classifier
        self.feature_optimizer = optim.Adam(
            self.model.feature_extractor.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.domain_optimizer = optim.Adam(
            self.model.domain_classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    
    def train_epoch(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        loss_fn: nn.Module,
        epoch: int,
    ) -> Dict[str, float]:
        """Train adversarial model for one epoch.
        
        Args:
            source_loader: Source domain data loader.
            target_loader: Target domain data loader.
            loss_fn: Loss function.
            epoch: Current epoch.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_adv_loss = 0.0
        num_batches = 0
        
        # Create iterators
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Get minimum length
        min_len = min(len(source_loader), len(target_loader))
        
        progress_bar = tqdm(range(min_len), desc=f"Epoch {epoch}")
        
        for batch_idx in progress_bar:
            try:
                # Get batches
                source_data, source_targets, _ = next(source_iter)
                target_data, target_targets, _ = next(target_iter)
                
                source_data = source_data.to(self.device)
                source_targets = source_targets.to(self.device)
                target_data = target_data.to(self.device)
                target_targets = target_targets.to(self.device)
                
                # Create domain labels
                source_domain_labels = torch.zeros(source_data.size(0), 1).to(self.device)
                target_domain_labels = torch.ones(target_data.size(0), 1).to(self.device)
                
                # Forward pass
                source_logits, source_domain_logits = self.model(source_data)
                target_logits, target_domain_logits = self.model(target_data)
                
                # Classification loss (source only)
                cls_loss = loss_fn.classification_loss(source_logits, source_targets)
                
                # Adversarial loss
                adv_loss = loss_fn.adversarial_loss(
                    torch.cat([source_domain_logits, target_domain_logits]),
                    torch.cat([source_domain_labels, target_domain_labels]),
                )
                
                # Train domain classifier
                self.domain_optimizer.zero_grad()
                adv_loss.backward(retain_graph=True)
                self.domain_optimizer.step()
                
                # Train feature extractor (minimize classification loss, maximize adversarial loss)
                self.feature_optimizer.zero_grad()
                total_loss_batch = cls_loss - self.adversarial_weight * adv_loss
                total_loss_batch.backward()
                self.feature_optimizer.step()
                
                total_loss += total_loss_batch.item()
                total_cls_loss += cls_loss.item()
                total_adv_loss += adv_loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{total_loss_batch.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                    "adv": f"{adv_loss.item():.4f}",
                })
                
            except StopIteration:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0.0
        avg_adv_loss = total_adv_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "train_cls_loss": avg_cls_loss,
            "train_adv_loss": avg_adv_loss,
        }


def train_source_only(
    model: nn.Module,
    source_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Train source-only baseline.
    
    Args:
        model: Model to train.
        source_loader: Source domain data loader.
        val_loader: Validation data loader.
        num_epochs: Number of epochs.
        learning_rate: Learning rate.
        device: Device to train on.
        
    Returns:
        Training history.
    """
    trainer = DomainAdaptationTrainer(
        model,
        device=device,
        learning_rate=learning_rate,
    )
    
    return trainer.train(
        source_loader=source_loader,
        target_loader=source_loader,  # Use source as target for source-only
        val_loader=val_loader,
        num_epochs=num_epochs,
        loss_fn=nn.CrossEntropyLoss(),
    )


def train_target_only(
    model: nn.Module,
    target_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Train target-only baseline.
    
    Args:
        model: Model to train.
        target_loader: Target domain data loader.
        val_loader: Validation data loader.
        num_epochs: Number of epochs.
        learning_rate: Learning rate.
        device: Device to train on.
        
    Returns:
        Training history.
    """
    trainer = DomainAdaptationTrainer(
        model,
        device=device,
        learning_rate=learning_rate,
    )
    
    return trainer.train(
        source_loader=target_loader,  # Use target as source for target-only
        target_loader=target_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        loss_fn=nn.CrossEntropyLoss(),
    )
