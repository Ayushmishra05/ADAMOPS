"""
AdamOps Explainability Module

Provides SHAP and LIME model explanations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from adamops.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ShapExplainer:
    """SHAP-based model explainer."""
    
    def __init__(self, model: Any, X: Union[pd.DataFrame, np.ndarray],
                 feature_names: Optional[List[str]] = None):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP required. Install with: pip install shap")
        
        self.model = model
        self.X = X
        self.feature_names = feature_names or (
            X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        )
        
        # Auto-select explainer type
        model_type = type(model).__name__.lower()
        if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgb' in model_type:
            self.explainer = shap.TreeExplainer(model)
        else:
            background = shap.sample(X, min(100, len(X)))
            self.explainer = shap.KernelExplainer(model.predict, background)
    
    def explain(self, X: Optional[np.ndarray] = None) -> Any:
        """Generate SHAP values."""
        X = X if X is not None else self.X[:100]
        return self.explainer(X)
    
    def plot_summary(self, X: Optional[np.ndarray] = None, max_display: int = 20):
        """Plot SHAP summary."""
        shap_values = self.explain(X)
        shap.summary_plot(shap_values, feature_names=self.feature_names, max_display=max_display)
    
    def plot_waterfall(self, idx: int = 0, X: Optional[np.ndarray] = None):
        """Plot waterfall for single prediction."""
        shap_values = self.explain(X)
        shap.waterfall_plot(shap_values[idx])
    
    def plot_force(self, idx: int = 0, X: Optional[np.ndarray] = None):
        """Plot force plot for single prediction."""
        shap_values = self.explain(X)
        shap.force_plot(shap_values[idx])
    
    def get_feature_importance(self, X: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Get feature importance from SHAP values."""
        shap_values = self.explain(X)
        importance = np.abs(shap_values.values).mean(axis=0)
        
        return pd.DataFrame({
            'feature': self.feature_names or [f'f{i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False)


class LimeExplainer:
    """LIME-based model explainer."""
    
    def __init__(self, model: Any, X_train: Union[pd.DataFrame, np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 mode: str = "classification"):
        if not LIME_AVAILABLE:
            raise ImportError("LIME required. Install with: pip install lime")
        
        self.model = model
        self.mode = mode
        
        if isinstance(X_train, pd.DataFrame):
            feature_names = feature_names or X_train.columns.tolist()
            X_train = X_train.values
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train, feature_names=feature_names, mode=mode, discretize_continuous=True
        )
    
    def explain(self, instance: np.ndarray, num_features: int = 10):
        """Explain a single prediction."""
        if self.mode == "classification":
            return self.explainer.explain_instance(
                instance, self.model.predict_proba, num_features=num_features
            )
        else:
            return self.explainer.explain_instance(
                instance, self.model.predict, num_features=num_features
            )
    
    def explain_multiple(self, X: np.ndarray, num_features: int = 10) -> List:
        """Explain multiple instances."""
        return [self.explain(x, num_features) for x in X]


def explain_shap(model: Any, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> ShapExplainer:
    """Create SHAP explainer."""
    return ShapExplainer(model, X, **kwargs)


def explain_lime(model: Any, X_train: Union[pd.DataFrame, np.ndarray], **kwargs) -> LimeExplainer:
    """Create LIME explainer."""
    return LimeExplainer(model, X_train, **kwargs)


def get_feature_importance(model: Any, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Get feature importance from model."""
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    
    if importance is None:
        raise ValueError("Model does not have feature_importances_ or coef_")
    
    names = feature_names or [f'feature_{i}' for i in range(len(importance))]
    
    return pd.DataFrame({
        'feature': names, 'importance': importance
    }).sort_values('importance', ascending=False)
