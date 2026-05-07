"""Domain Adaptation Techniques Package."""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"
__github__ = "https://github.com/kryptologyst"

# Core imports
from . import data
from . import models
from . import losses
from . import metrics
from . import train
from . import eval
from . import utils
from . import safety

# Main classes
from .models import (
    AdversarialDomainAdaptationModel,
    CORALModel,
    SourceOnlyModel,
    TargetOnlyModel,
)

from .losses import (
    DomainAdaptationLoss,
    CORALLoss,
    MMDLoss,
    AdversarialLoss,
)

from .train import (
    DomainAdaptationTrainer,
    AdversarialTrainer,
)

from .eval import (
    DomainAdaptationEvaluator,
)

from .safety import (
    SafetyChecker,
    EthicsChecker,
    ComplianceManager,
)

__all__ = [
    # Package info
    "__version__",
    "__author__",
    "__email__",
    "__github__",
    
    # Modules
    "data",
    "models", 
    "losses",
    "metrics",
    "train",
    "eval",
    "utils",
    "safety",
    
    # Models
    "AdversarialDomainAdaptationModel",
    "CORALModel", 
    "SourceOnlyModel",
    "TargetOnlyModel",
    
    # Losses
    "DomainAdaptationLoss",
    "CORALLoss",
    "MMDLoss",
    "AdversarialLoss",
    
    # Training
    "DomainAdaptationTrainer",
    "AdversarialTrainer",
    
    # Evaluation
    "DomainAdaptationEvaluator",
    
    # Safety
    "SafetyChecker",
    "EthicsChecker", 
    "ComplianceManager",
]
