"""
AdamOps Visualization Module

Provides plotting for model evaluation: confusion matrices, ROC curves, etc.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

from sklearn import metrics as sklearn_metrics
from adamops.utils.logging import get_logger

logger = get_logger(__name__)


def _check_plt():
    if not PLT_AVAILABLE:
        raise ImportError("matplotlib and seaborn required. Install with: pip install matplotlib seaborn")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    labels: Optional[List[str]] = None, normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6), cmap: str = "Blues",
    title: str = "Confusion Matrix", save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix."""
    _check_plt()
    
    cm = sklearn_metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    figsize: Tuple[int, int] = (8, 6), title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot ROC curve."""
    _check_plt()
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    
    fpr, tpr, _ = sklearn_metrics.roc_curve(y_true, y_prob)
    auc = sklearn_metrics.roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    figsize: Tuple[int, int] = (8, 6), save_path: Optional[str] = None
) -> plt.Figure:
    """Plot precision-recall curve."""
    _check_plt()
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    
    precision, recall, _ = sklearn_metrics.precision_recall_curve(y_true, y_prob)
    ap = sklearn_metrics.average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, 'b-', label=f'PR (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance: np.ndarray, feature_names: List[str],
    top_n: int = 20, figsize: Tuple[int, int] = (10, 8),
    title: str = "Feature Importance", save_path: Optional[str] = None
) -> plt.Figure:
    """Plot feature importance."""
    _check_plt()
    
    indices = np.argsort(importance)[-top_n:]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(indices)), importance[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray,
    figsize: Tuple[int, int] = (12, 5), save_path: Optional[str] = None
) -> plt.Figure:
    """Plot residuals for regression."""
    _check_plt()
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    
    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray,
    figsize: Tuple[int, int] = (8, 8), save_path: Optional[str] = None
) -> plt.Figure:
    """Plot actual vs predicted for regression."""
    _check_plt()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_learning_curve(
    train_sizes: np.ndarray, train_scores: np.ndarray, val_scores: np.ndarray,
    figsize: Tuple[int, int] = (8, 6), save_path: Optional[str] = None
) -> plt.Figure:
    """Plot learning curve."""
    _check_plt()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    ax.plot(train_sizes, train_mean, 'o-', label='Training')
    ax.plot(train_sizes, val_mean, 'o-', label='Validation')
    
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results: pd.DataFrame, metric: str = "cv_mean",
    figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None
) -> plt.Figure:
    """Plot model comparison bar chart."""
    _check_plt()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    results_sorted = results.sort_values(metric, ascending=True)
    ax.barh(results_sorted['algorithm'], results_sorted[metric], color='steelblue')
    ax.set_xlabel(metric)
    ax.set_title('Model Comparison')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
