"""
AdamOps Evaluation Module

Provides model evaluation capabilities:
- metrics: Compute classification, regression, and clustering metrics
- visualization: Plot confusion matrices, ROC curves, feature importance
- explainability: SHAP and LIME explanations
- comparison: Compare multiple models
- reports: Generate HTML/PDF reports
"""

from adamops.evaluation import metrics
from adamops.evaluation import visualization
from adamops.evaluation import explainability
from adamops.evaluation import comparison
from adamops.evaluation import reports

__all__ = [
    "metrics",
    "visualization",
    "explainability",
    "comparison",
    "reports",
]
