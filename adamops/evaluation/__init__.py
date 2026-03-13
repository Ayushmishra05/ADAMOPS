"""
AdamOps Evaluation Module

Provides model evaluation capabilities:
- metrics: Compute classification, regression, and clustering metrics
- visualization: Plot confusion matrices, ROC curves, feature importance
- explainability: SHAP and LIME explanations
- comparison: Compare multiple models
- reports: Generate HTML/PDF reports
"""

from . import metrics
from . import reports
from . import visualization
from . import explainability
from . import comparison

__all__ = [
    "metrics",
    "visualization",
    "explainability",
    "comparison",
    "reports",
]
