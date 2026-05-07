"""Evaluation utilities for domain adaptation models."""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..metrics import compute_all_metrics
from ..utils import get_device


class DomainAdaptationEvaluator:
    """Evaluator for domain adaptation models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to evaluate on.
            save_dir: Directory to save results.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.save_dir = save_dir
        
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
    
    def evaluate(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        model_name: str = "model",
    ) -> Dict[str, float]:
        """Evaluate model on source and target domains.
        
        Args:
            source_loader: Source domain data loader.
            target_loader: Target domain data loader.
            model_name: Name of the model for saving.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        
        # Compute comprehensive metrics
        metrics = compute_all_metrics(
            self.model,
            source_loader,
            target_loader,
            self.device,
            extract_features=True,
        )
        
        # Save metrics
        if self.save_dir:
            self._save_metrics(metrics, model_name)
        
        return metrics
    
    def _save_metrics(self, metrics: Dict[str, float], model_name: str) -> None:
        """Save metrics to file.
        
        Args:
            metrics: Metrics to save.
            model_name: Name of the model.
        """
        import json
        
        metrics_file = os.path.join(self.save_dir, f"{model_name}_metrics.json")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        source_loader: DataLoader,
        target_loader: DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models.
        
        Args:
            models: Dictionary of model names to models.
            source_loader: Source domain data loader.
            target_loader: Target domain data loader.
            
        Returns:
            Dictionary of metrics for each model.
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            # Set model for evaluation
            self.model = model
            self.model.to(self.device)
            
            # Evaluate
            metrics = self.evaluate(source_loader, target_loader, model_name)
            results[model_name] = metrics
        
        # Create comparison table
        if self.save_dir:
            self._save_comparison_table(results)
        
        return results
    
    def _save_comparison_table(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save comparison table.
        
        Args:
            results: Results from model comparison.
        """
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame(results).T
        
        # Select key metrics
        key_metrics = [
            "source_accuracy", "target_accuracy",
            "source_f1_macro", "target_f1_macro",
            "mmd_distance", "coral_distance",
        ]
        
        available_metrics = [m for m in key_metrics if m in df.columns]
        df_subset = df[available_metrics]
        
        # Save to CSV
        csv_file = os.path.join(self.save_dir, "model_comparison.csv")
        df_subset.to_csv(csv_file)
        
        # Save formatted table
        table_file = os.path.join(self.save_dir, "model_comparison.txt")
        with open(table_file, 'w') as f:
            f.write("Domain Adaptation Model Comparison\n")
            f.write("=" * 50 + "\n\n")
            f.write(df_subset.round(4).to_string())
            f.write("\n\n")
            
            # Add best performing models
            f.write("Best Performing Models:\n")
            f.write("-" * 30 + "\n")
            
            for metric in available_metrics:
                if "accuracy" in metric or "f1" in metric:
                    best_model = df_subset[metric].idxmax()
                    f.write(f"{metric}: {best_model} ({df_subset.loc[best_model, metric]:.4f})\n")
                elif "distance" in metric:
                    best_model = df_subset[metric].idxmin()
                    f.write(f"{metric}: {best_model} ({df_subset.loc[best_model, metric]:.4f})\n")
    
    def visualize_results(
        self,
        results: Dict[str, Dict[str, float]],
        save_plots: bool = True,
    ) -> None:
        """Visualize evaluation results.
        
        Args:
            results: Results from model evaluation.
            save_plots: Whether to save plots.
        """
        if not results:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Domain Adaptation Model Comparison", fontsize=16)
        
        models = list(results.keys())
        
        # Accuracy comparison
        source_acc = [results[model].get("source_accuracy", 0) for model in models]
        target_acc = [results[model].get("target_accuracy", 0) for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, source_acc, width, label='Source', alpha=0.8)
        axes[0, 0].bar(x + width/2, target_acc, width, label='Target', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score comparison
        source_f1 = [results[model].get("source_f1_macro", 0) for model in models]
        target_f1 = [results[model].get("target_f1_macro", 0) for model in models]
        
        axes[0, 1].bar(x - width/2, source_f1, width, label='Source', alpha=0.8)
        axes[0, 1].bar(x + width/2, target_f1, width, label='Target', alpha=0.8)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('F1 Score (Macro)')
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Domain shift metrics
        mmd_distances = [results[model].get("mmd_distance", 0) for model in models]
        coral_distances = [results[model].get("coral_distance", 0) for model in models]
        
        axes[1, 0].bar(x - width/2, mmd_distances, width, label='MMD', alpha=0.8)
        axes[1, 0].bar(x + width/2, coral_distances, width, label='CORAL', alpha=0.8)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Distance')
        axes[1, 0].set_title('Domain Shift Metrics')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Target accuracy vs domain shift
        axes[1, 1].scatter(mmd_distances, target_acc, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (mmd_distances[i], target_acc[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('MMD Distance')
        axes[1, 1].set_ylabel('Target Accuracy')
        axes[1, 1].set_title('Target Accuracy vs Domain Shift')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots and self.save_dir:
            plot_file = os.path.join(self.save_dir, "model_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "target_accuracy",
        ascending: bool = False,
    ) -> None:
        """Create a leaderboard of models.
        
        Args:
            results: Results from model evaluation.
            metric: Metric to rank by.
            ascending: Whether to sort in ascending order.
        """
        if not results:
            return
        
        # Create leaderboard
        leaderboard = []
        for model_name, metrics in results.items():
            if metric in metrics:
                leaderboard.append((model_name, metrics[metric]))
        
        # Sort by metric
        leaderboard.sort(key=lambda x: x[1], reverse=not ascending)
        
        # Print leaderboard
        print(f"\nLeaderboard (ranked by {metric}):")
        print("=" * 50)
        for i, (model_name, score) in enumerate(leaderboard, 1):
            print(f"{i:2d}. {model_name:<20} {score:.4f}")
        
        # Save leaderboard
        if self.save_dir:
            leaderboard_file = os.path.join(self.save_dir, "leaderboard.txt")
            with open(leaderboard_file, 'w') as f:
                f.write(f"Leaderboard (ranked by {metric}):\n")
                f.write("=" * 50 + "\n")
                for i, (model_name, score) in enumerate(leaderboard, 1):
                    f.write(f"{i:2d}. {model_name:<20} {score:.4f}\n")


def evaluate_model(
    model: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Convenience function to evaluate a single model.
    
    Args:
        model: Model to evaluate.
        source_loader: Source domain data loader.
        target_loader: Target domain data loader.
        device: Device to evaluate on.
        save_dir: Directory to save results.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    evaluator = DomainAdaptationEvaluator(model, device, save_dir)
    return evaluator.evaluate(source_loader, target_loader)
