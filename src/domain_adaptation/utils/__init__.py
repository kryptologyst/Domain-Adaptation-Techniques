"""Domain Adaptation Techniques - Core utilities and configuration."""

import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        DictConfig: Loaded configuration.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        save_path: Path where to save the configuration.
    """
    OmegaConf.save(config, save_path)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory structure.
    
    Args:
        base_dir: Base directory for experiments.
        experiment_name: Name of the experiment.
        
    Returns:
        str: Path to the created experiment directory.
    """
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "outputs"), exist_ok=True)
    return exp_dir


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> None:
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            model: Model to potentially save weights from.
            
        Returns:
            bool: True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
        """
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
    logger: Optional[Any] = None,
) -> None:
    """Log metrics to console and optional logger.
    
    Args:
        metrics: Dictionary of metrics to log.
        step: Current step/epoch.
        prefix: Prefix for metric names.
        logger: Optional logger (e.g., wandb, tensorboard).
    """
    log_str = f"Step {step}: "
    for key, value in metrics.items():
        metric_name = f"{prefix}_{key}" if prefix else key
        log_str += f"{metric_name}={value:.4f} "
        
        if logger:
            if hasattr(logger, "log"):
                logger.log({metric_name: value}, step=step)
            elif hasattr(logger, "add_scalar"):
                logger.add_scalar(metric_name, value, step)
    
    print(log_str)
