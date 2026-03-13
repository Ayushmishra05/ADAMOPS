"""
AdamOps - A comprehensive MLOps library for end-to-end machine learning workflows.

AdamOps provides tools for:
- Data loading, validation, cleaning, and feature engineering
- Model training, registry, and ensemble methods
- AutoML with hyperparameter tuning
- Model evaluation and explainability
- Deployment to various platforms
- Monitoring and drift detection
- Pipeline orchestration
- Visual pipeline builder (Studio)

Author: AdamOps Team
Version: 0.1.1
"""

__version__ = "0.1.1"
__author__ = "AdamOps Team"

# Import core modules for easy access
from .data import loaders, validators, preprocessors, splitters
from .models import modelops, registry, ensembles, automl
from .evaluation import metrics
from .utils import config, logging as adamops_logging, helpers

__all__ = [
    "loaders",
    "validators", 
    "preprocessors",
    "splitters",
    "modelops",
    "registry",
    "ensembles",
    "automl",
    "metrics",
    "config",
    "adamops_logging",
    "helpers",
    "__version__",
]
