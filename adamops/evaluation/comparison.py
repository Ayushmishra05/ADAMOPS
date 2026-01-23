"""
AdamOps Model Comparison Module

Provides tools to compare multiple models.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from adamops.utils.logging import get_logger
from adamops.evaluation.metrics import evaluate

logger = get_logger(__name__)


def compare_models(
    models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
    task: str = "classification"
) -> pd.DataFrame:
    """
    Compare multiple models on test data.
    
    Args:
        models: Dict mapping model names to fitted models.
        X_test: Test features.
        y_test: Test labels.
        task: 'classification' or 'regression'.
    
    Returns:
        DataFrame with comparison results.
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = None
        
        if hasattr(model, 'predict_proba') and task == "classification":
            try:
                y_prob = model.predict_proba(X_test)
            except:
                pass
        
        metrics = evaluate(y_test, y_pred, task, y_prob)
        metrics['model'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    cols = ['model'] + [c for c in df.columns if c != 'model']
    return df[cols]


def compare_cv(
    models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
    cv: int = 5, scoring: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare models using cross-validation.
    
    Returns:
        DataFrame with CV scores for each model.
    """
    results = []
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results.append({
            'model': name,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max(),
        })
    
    return pd.DataFrame(results).sort_values('cv_mean', ascending=False)


def rank_models(comparison_df: pd.DataFrame, metrics: List[str], 
                ascending: Optional[List[bool]] = None) -> pd.DataFrame:
    """
    Rank models by multiple metrics.
    
    Args:
        comparison_df: Comparison DataFrame.
        metrics: Metrics to rank by.
        ascending: Whether each metric should be ascending (lower is better).
    
    Returns:
        DataFrame with rankings.
    """
    df = comparison_df.copy()
    
    if ascending is None:
        ascending = [False] * len(metrics)
    
    for metric, asc in zip(metrics, ascending):
        df[f'{metric}_rank'] = df[metric].rank(ascending=asc)
    
    rank_cols = [f'{m}_rank' for m in metrics]
    df['avg_rank'] = df[rank_cols].mean(axis=1)
    
    return df.sort_values('avg_rank')


def statistical_test(
    scores_a: np.ndarray, scores_b: np.ndarray, 
    test: str = "wilcoxon"
) -> Dict[str, float]:
    """
    Perform statistical test between two sets of scores.
    
    Args:
        scores_a: Scores for model A.
        scores_b: Scores for model B.
        test: 'wilcoxon', 'ttest', or 'mannwhitney'.
    
    Returns:
        Dict with statistic and p-value.
    """
    from scipy import stats
    
    if test == "wilcoxon":
        stat, pval = stats.wilcoxon(scores_a, scores_b)
    elif test == "ttest":
        stat, pval = stats.ttest_rel(scores_a, scores_b)
    elif test == "mannwhitney":
        stat, pval = stats.mannwhitneyu(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {"statistic": stat, "p_value": pval, "significant": pval < 0.05}
