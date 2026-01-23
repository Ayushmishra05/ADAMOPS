"""
AdamOps Data Splitters Module

Provides data splitting: train/test, train/val/test, time-series, K-Fold, stratified.
"""

from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
)
from adamops.utils.logging import get_logger

logger = get_logger(__name__)


def split_train_test(
    X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None,
    test_size: float = 0.2, random_state: int = 42, stratify: bool = False, shuffle: bool = True
) -> Tuple:
    """
    Split data into train and test sets.
    
    Args:
        X: Features.
        y: Target (optional).
        test_size: Test set proportion.
        random_state: Random seed.
        stratify: Stratify by target.
        shuffle: Shuffle before splitting.
    
    Returns:
        (X_train, X_test) or (X_train, X_test, y_train, y_test)
    """
    stratify_col = y if stratify and y is not None else None
    
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_col, shuffle=shuffle
        )
        logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test


def split_train_val_test(
    X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None,
    train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15,
    random_state: int = 42, stratify: bool = False
) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Returns:
        (X_train, X_val, X_test) or (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Normalize sizes
    total = train_size + val_size + test_size
    train_size, val_size, test_size = train_size/total, val_size/total, test_size/total
    
    stratify_col = y if stratify and y is not None else None
    
    if y is not None:
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_col
        )
        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)
        stratify_temp = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state,
            stratify=stratify_temp
        )
        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_temp, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val = train_test_split(X_temp, test_size=val_ratio, random_state=random_state)
        return X_train, X_val, X_test


def split_timeseries(
    X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None,
    n_splits: int = 5, test_size: Optional[int] = None, gap: int = 0
) -> Iterator[Tuple]:
    """
    Time series split for temporal data.
    
    Args:
        X: Features.
        y: Target.
        n_splits: Number of splits.
        test_size: Test set size per split.
        gap: Gap between train and test.
    
    Yields:
        (train_idx, test_idx) tuples.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    logger.info(f"Time series split with {n_splits} folds")
    
    for train_idx, test_idx in tscv.split(X):
        yield train_idx, test_idx


def split_kfold(
    X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None,
    n_splits: int = 5, shuffle: bool = True, random_state: int = 42
) -> Iterator[Tuple]:
    """
    K-Fold cross-validation split.
    
    Yields:
        (train_idx, test_idx) tuples.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    logger.info(f"K-Fold split with {n_splits} folds")
    
    for train_idx, test_idx in kf.split(X):
        yield train_idx, test_idx


def split_stratified_kfold(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
    n_splits: int = 5, shuffle: bool = True, random_state: int = 42
) -> Iterator[Tuple]:
    """
    Stratified K-Fold cross-validation split.
    
    Preserves class distribution in each fold.
    
    Yields:
        (train_idx, test_idx) tuples.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    logger.info(f"Stratified K-Fold split with {n_splits} folds")
    
    for train_idx, test_idx in skf.split(X, y):
        yield train_idx, test_idx


def split_group_kfold(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
    groups: Union[pd.Series, np.ndarray], n_splits: int = 5
) -> Iterator[Tuple]:
    """
    Group K-Fold split. Ensures groups are not split across train/test.
    
    Yields:
        (train_idx, test_idx) tuples.
    """
    gkf = GroupKFold(n_splits=n_splits)
    logger.info(f"Group K-Fold split with {n_splits} folds")
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        yield train_idx, test_idx


def get_fold_data(
    X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]],
    train_idx: np.ndarray, test_idx: np.ndarray
) -> Tuple:
    """Get train/test data for a fold."""
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    else:
        X_train, X_test = X[train_idx], X[test_idx]
    
    if y is not None:
        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]
        return X_train, X_test, y_train, y_test
    
    return X_train, X_test


def create_cv_splits(
    X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None,
    method: str = "kfold", n_splits: int = 5, **kwargs
) -> List[Tuple]:
    """
    Create cross-validation splits.
    
    Args:
        X: Features.
        y: Target.
        method: 'kfold', 'stratified', 'timeseries', 'group'
        n_splits: Number of folds.
    
    Returns:
        List of (train_idx, test_idx) tuples.
    """
    if method == "kfold":
        return list(split_kfold(X, y, n_splits, **kwargs))
    elif method == "stratified":
        if y is None:
            raise ValueError("y is required for stratified split")
        return list(split_stratified_kfold(X, y, n_splits, **kwargs))
    elif method == "timeseries":
        return list(split_timeseries(X, y, n_splits, **kwargs))
    elif method == "group":
        if "groups" not in kwargs:
            raise ValueError("groups is required for group split")
        return list(split_group_kfold(X, y, kwargs["groups"], n_splits))
    else:
        raise ValueError(f"Unknown split method: {method}")
