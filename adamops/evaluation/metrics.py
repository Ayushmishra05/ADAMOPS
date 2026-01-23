"""
AdamOps Evaluation Metrics Module

Provides classification, regression, and clustering metrics.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics

from adamops.utils.logging import get_logger
from adamops.utils.helpers import infer_task_type

logger = get_logger(__name__)


# =============================================================================
# Classification Metrics
# =============================================================================

def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, 
    y_prob: Optional[np.ndarray] = None, average: str = "weighted"
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Probability predictions (for ROC-AUC).
        average: Averaging method for multiclass.
    
    Returns:
        Dict with accuracy, precision, recall, f1, etc.
    """
    results = {
        "accuracy": sklearn_metrics.accuracy_score(y_true, y_pred),
        "precision": sklearn_metrics.precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": sklearn_metrics.recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": sklearn_metrics.f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Binary classification specific
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        results["mcc"] = sklearn_metrics.matthews_corrcoef(y_true, y_pred)
        
        if y_prob is not None:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            results["roc_auc"] = sklearn_metrics.roc_auc_score(y_true, y_prob)
            results["pr_auc"] = sklearn_metrics.average_precision_score(y_true, y_prob)
    
    # Multiclass with probabilities
    elif y_prob is not None and len(unique_classes) > 2:
        try:
            results["roc_auc_ovr"] = sklearn_metrics.roc_auc_score(
                y_true, y_prob, multi_class="ovr", average=average
            )
        except ValueError:
            pass
    
    return results


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None
) -> np.ndarray:
    """Compute confusion matrix."""
    return sklearn_metrics.confusion_matrix(y_true, y_pred, labels=labels)


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, 
    target_names: Optional[List[str]] = None, output_dict: bool = True
) -> Union[str, Dict]:
    """Generate classification report."""
    return sklearn_metrics.classification_report(
        y_true, y_pred, target_names=target_names, output_dict=output_dict
    )


# =============================================================================
# Regression Metrics
# =============================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Returns:
        Dict with mse, rmse, mae, r2, mape, etc.
    """
    mse = sklearn_metrics.mean_squared_error(y_true, y_pred)
    
    results = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": sklearn_metrics.mean_absolute_error(y_true, y_pred),
        "r2": sklearn_metrics.r2_score(y_true, y_pred),
        "explained_variance": sklearn_metrics.explained_variance_score(y_true, y_pred),
    }
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        results["mape"] = mape
    
    # SMAPE
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if mask.any():
        smape = np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100
        results["smape"] = smape
    
    # Adjusted R2
    n = len(y_true)
    p = 1  # Assumes 1 feature, should be passed in for accuracy
    if n > p + 1:
        results["adjusted_r2"] = 1 - (1 - results["r2"]) * (n - 1) / (n - p - 1)
    
    return results


# =============================================================================
# Clustering Metrics
# =============================================================================

def clustering_metrics(
    X: np.ndarray, labels: np.ndarray, 
    y_true: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute clustering metrics.
    
    Args:
        X: Feature data.
        labels: Cluster labels.
        y_true: True labels (if available).
    
    Returns:
        Dict with silhouette, davies_bouldin, calinski_harabasz, etc.
    """
    results = {}
    
    # Internal metrics (don't need ground truth)
    n_labels = len(np.unique(labels))
    if n_labels > 1 and n_labels < len(X):
        results["silhouette"] = sklearn_metrics.silhouette_score(X, labels)
        results["davies_bouldin"] = sklearn_metrics.davies_bouldin_score(X, labels)
        results["calinski_harabasz"] = sklearn_metrics.calinski_harabasz_score(X, labels)
    
    # External metrics (need ground truth)
    if y_true is not None:
        results["adjusted_rand"] = sklearn_metrics.adjusted_rand_score(y_true, labels)
        results["normalized_mutual_info"] = sklearn_metrics.normalized_mutual_info_score(y_true, labels)
        results["homogeneity"] = sklearn_metrics.homogeneity_score(y_true, labels)
        results["completeness"] = sklearn_metrics.completeness_score(y_true, labels)
        results["v_measure"] = sklearn_metrics.v_measure_score(y_true, labels)
    
    return results


# =============================================================================
# Unified Evaluation Interface
# =============================================================================

def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray,
    task: str = "auto", y_prob: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Unified evaluation function.
    
    Args:
        y_true: True values/labels.
        y_pred: Predicted values/labels.
        task: 'classification', 'regression', 'clustering', or 'auto'.
        y_prob: Probability predictions.
        X: Feature data (for clustering).
    
    Returns:
        Dict with relevant metrics.
    """
    if task == "auto":
        task = infer_task_type(y_true)
    
    if task in ["classification", "multiclass"]:
        return classification_metrics(y_true, y_pred, y_prob)
    elif task == "regression":
        return regression_metrics(y_true, y_pred)
    elif task == "clustering":
        if X is None:
            raise ValueError("X required for clustering metrics")
        return clustering_metrics(X, y_pred, y_true)
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: np.ndarray,
    task: str = "auto"
) -> Dict[str, float]:
    """Evaluate a model on test data."""
    y_pred = model.predict(X_test)
    y_prob = None
    
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except:
            pass
    
    return evaluate(y_test, y_pred, task, y_prob)


def results_to_dataframe(results: Dict[str, float]) -> pd.DataFrame:
    """Convert results dict to DataFrame."""
    return pd.DataFrame([results]).T.reset_index().rename(
        columns={"index": "metric", 0: "value"}
    )


def compare_results(
    results_list: List[Dict[str, float]], names: List[str]
) -> pd.DataFrame:
    """Compare multiple evaluation results."""
    data = {name: results for name, results in zip(names, results_list)}
    return pd.DataFrame(data).T
