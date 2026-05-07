"""Main training script for domain adaptation experiments."""

import argparse
import os
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from domain_adaptation.data import (
    create_domain_loaders,
    load_mnist_data,
    load_usps_data,
    split_dataset,
)
from domain_adaptation.eval import DomainAdaptationEvaluator
from domain_adaptation.losses import DomainAdaptationLoss
from domain_adaptation.models import (
    AdversarialDomainAdaptationModel,
    CORALModel,
    SourceOnlyModel,
    TargetOnlyModel,
)
from domain_adaptation.train import (
    AdversarialTrainer,
    DomainAdaptationTrainer,
    train_source_only,
    train_target_only,
)
from domain_adaptation.utils import (
    create_experiment_dir,
    get_device,
    set_seed,
)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from file."""
    return OmegaConf.load(config_path)


def create_models(config: DictConfig) -> Dict[str, torch.nn.Module]:
    """Create all models for comparison."""
    models = {}
    
    # Source-only baseline
    models["source_only"] = SourceOnlyModel(
        input_channels=config.data.input_channels,
        feature_dim=config.model.feature_dim,
        num_classes=config.data.num_classes,
        dropout=config.model.dropout,
    )
    
    # Target-only baseline
    models["target_only"] = TargetOnlyModel(
        input_channels=config.data.input_channels,
        feature_dim=config.model.feature_dim,
        num_classes=config.data.num_classes,
        dropout=config.model.dropout,
    )
    
    # Adversarial domain adaptation
    models["adversarial"] = AdversarialDomainAdaptationModel(
        input_channels=config.data.input_channels,
        feature_dim=config.model.feature_dim,
        num_classes=config.data.num_classes,
        dropout=config.model.dropout,
        gradient_reversal_alpha=config.model.gradient_reversal_alpha,
    )
    
    # CORAL domain adaptation
    models["coral"] = CORALModel(
        input_channels=config.data.input_channels,
        feature_dim=config.model.feature_dim,
        num_classes=config.data.num_classes,
        dropout=config.model.dropout,
    )
    
    return models


def prepare_data(config: DictConfig):
    """Prepare data loaders."""
    print("Loading datasets...")
    
    # Load source and target data
    source_data, source_targets = load_mnist_data(
        root=config.data.root,
        train=True,
    )
    
    target_data, target_targets = load_usps_data(
        root=config.data.root,
        train=True,
    )
    
    # Split datasets
    source_train, source_val, source_test = split_dataset(
        torch.utils.data.TensorDataset(source_data, source_targets),
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.seed,
    )
    
    target_train, target_val, target_test = split_dataset(
        torch.utils.data.TensorDataset(target_data, target_targets),
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.seed,
    )
    
    # Create data loaders
    source_train_loader = torch.utils.data.DataLoader(
        source_train,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    
    source_val_loader = torch.utils.data.DataLoader(
        source_val,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    
    source_test_loader = torch.utils.data.DataLoader(
        source_test,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    
    target_train_loader = torch.utils.data.DataLoader(
        target_train,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    
    target_val_loader = torch.utils.data.DataLoader(
        target_val,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    
    target_test_loader = torch.utils.data.DataLoader(
        target_test,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    
    return {
        "source_train": source_train_loader,
        "source_val": source_val_loader,
        "source_test": source_test_loader,
        "target_train": target_train_loader,
        "target_val": target_val_loader,
        "target_test": target_test_loader,
    }


def train_models(
    models: Dict[str, torch.nn.Module],
    data_loaders: Dict[str, torch.utils.data.DataLoader],
    config: DictConfig,
    device: torch.device,
    exp_dir: str,
) -> Dict[str, torch.nn.Module]:
    """Train all models."""
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Create model-specific directory
        model_dir = os.path.join(exp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        if model_name == "source_only":
            # Train source-only baseline
            train_source_only(
                model=model,
                source_loader=data_loaders["source_train"],
                val_loader=data_loaders["source_val"],
                num_epochs=config.training.num_epochs,
                learning_rate=config.training.learning_rate,
                device=device,
            )
            
        elif model_name == "target_only":
            # Train target-only baseline
            train_target_only(
                model=model,
                target_loader=data_loaders["target_train"],
                val_loader=data_loaders["target_val"],
                num_epochs=config.training.num_epochs,
                learning_rate=config.training.learning_rate,
                device=device,
            )
            
        elif model_name == "adversarial":
            # Train adversarial model
            loss_fn = DomainAdaptationLoss(
                classification_weight=config.loss.classification_weight,
                adversarial_weight=config.loss.adversarial_weight,
            )
            
            trainer = AdversarialTrainer(
                model=model,
                device=device,
                learning_rate=config.training.learning_rate,
                adversarial_weight=config.loss.adversarial_weight,
            )
            
            trainer.train(
                source_loader=data_loaders["source_train"],
                target_loader=data_loaders["target_train"],
                val_loader=data_loaders["target_val"],
                num_epochs=config.training.num_epochs,
                loss_fn=loss_fn,
                save_path=os.path.join(model_dir, "best_model.pth"),
            )
            
        elif model_name == "coral":
            # Train CORAL model
            trainer = DomainAdaptationTrainer(
                model=model,
                device=device,
                learning_rate=config.training.learning_rate,
            )
            
            # Custom training loop for CORAL
            optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
            criterion = torch.nn.CrossEntropyLoss()
            
            for epoch in range(config.training.num_epochs):
                model.train()
                
                total_loss = 0.0
                num_batches = 0
                
                source_iter = iter(data_loaders["source_train"])
                target_iter = iter(data_loaders["target_train"])
                
                min_len = min(len(data_loaders["source_train"]), len(data_loaders["target_train"]))
                
                for batch_idx in range(min_len):
                    try:
                        source_data, source_targets, _ = next(source_iter)
                        target_data, target_targets, _ = next(target_iter)
                        
                        source_data = source_data.to(device)
                        source_targets = source_targets.to(device)
                        target_data = target_data.to(device)
                        
                        optimizer.zero_grad()
                        
                        # Forward pass
                        source_logits = model(source_data)
                        target_logits = model(target_data)
                        
                        # Classification loss
                        cls_loss = criterion(source_logits, source_targets)
                        
                        # CORAL loss
                        source_features = model.feature_extractor(source_data)
                        target_features = model.feature_extractor(target_data)
                        coral_loss = model.coral_loss(source_features, target_features)
                        
                        # Total loss
                        total_loss_batch = cls_loss + config.loss.coral_weight * coral_loss
                        
                        total_loss_batch.backward()
                        optimizer.step()
                        
                        total_loss += total_loss_batch.item()
                        num_batches += 1
                        
                    except StopIteration:
                        break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {total_loss / num_batches:.4f}")
        
        # Save trained model
        torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pth"))
        trained_models[model_name] = model
    
    return trained_models


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train domain adaptation models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--exp-name", type=str, default="domain_adaptation_exp",
                       help="Experiment name")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed:
        config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = create_experiment_dir(config.experiment.base_dir, args.exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Save configuration
    OmegaConf.save(config, os.path.join(exp_dir, "config.yaml"))
    
    # Prepare data
    data_loaders = prepare_data(config)
    
    # Create models
    models = create_models(config)
    
    # Train models
    trained_models = train_models(models, data_loaders, config, device, exp_dir)
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluator = DomainAdaptationEvaluator(
        model=trained_models["source_only"],  # Dummy model
        device=device,
        save_dir=exp_dir,
    )
    
    results = evaluator.compare_models(
        models=trained_models,
        source_loader=data_loaders["source_test"],
        target_loader=data_loaders["target_test"],
    )
    
    # Create leaderboard
    evaluator.create_leaderboard(results, metric="target_accuracy")
    
    # Visualize results
    evaluator.visualize_results(results, save_plots=True)
    
    print(f"\nExperiment completed! Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
