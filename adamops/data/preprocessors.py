"""
AdamOps Data Preprocessors Module

Provides data cleaning capabilities: missing values, outliers, duplicates, type conversion.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from adamops.utils.logging import get_logger

logger = get_logger(__name__)


# Missing Value Handling
def handle_missing(
    df: pd.DataFrame, strategy: str = "mean", columns: Optional[List[str]] = None,
    fill_value: Optional[any] = None, n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Handle missing values.
    
    Args:
        df: DataFrame to process.
        strategy: 'drop', 'mean', 'median', 'mode', 'constant', 'ffill', 'bfill', 'knn', 'iterative'
        columns: Columns to process (None for all).
        fill_value: Value for 'constant' strategy.
        n_neighbors: Neighbors for KNN.
    
    Returns:
        Processed DataFrame.
    """
    df = df.copy()
    cols = columns or df.columns.tolist()
    logger.info(f"Handling missing values with strategy: {strategy}")

    if strategy == "drop":
        return df.dropna(subset=cols)
    elif strategy == "ffill":
        df[cols] = df[cols].ffill()
    elif strategy == "bfill":
        df[cols] = df[cols].bfill()
    elif strategy == "constant":
        df[cols] = df[cols].fillna(fill_value)
    elif strategy in ["mean", "median", "most_frequent"]:
        strat = "most_frequent" if strategy == "mode" else strategy
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            imputer = SimpleImputer(strategy=strat)
            df[num_cols] = imputer.fit_transform(df[num_cols])
    elif strategy == "mode":
        for col in cols:
            if df[col].isna().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
                df[col] = df[col].fillna(mode_val)
    elif strategy == "knn":
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[num_cols] = imputer.fit_transform(df[num_cols])
    elif strategy == "iterative":
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            imputer = IterativeImputer(random_state=42)
            df[num_cols] = imputer.fit_transform(df[num_cols])
    
    return df


# Outlier Handling
def handle_outliers(
    df: pd.DataFrame, method: str = "iqr", columns: Optional[List[str]] = None,
    threshold: float = 1.5, action: str = "clip", contamination: float = 0.1
) -> pd.DataFrame:
    """
    Handle outliers.
    
    Args:
        df: DataFrame to process.
        method: 'iqr', 'zscore', 'isolation_forest'
        columns: Columns to process (None for numeric).
        threshold: IQR multiplier or Z-score threshold.
        action: 'clip', 'drop', 'nan'
        contamination: For isolation forest.
    
    Returns:
        Processed DataFrame.
    """
    df = df.copy()
    num_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Handling outliers with method: {method}, action: {action}")

    if method == "iqr":
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
            mask = (df[col] < lower) | (df[col] > upper)
            if action == "clip":
                df[col] = df[col].clip(lower, upper)
            elif action == "drop":
                df = df[~mask]
            elif action == "nan":
                df.loc[mask, col] = np.nan
                
    elif method == "zscore":
        for col in num_cols:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z > threshold
            if action == "clip":
                mean, std = df[col].mean(), df[col].std()
                lower, upper = mean - threshold * std, mean + threshold * std
                df[col] = df[col].clip(lower, upper)
            elif action == "drop":
                df = df[~mask]
            elif action == "nan":
                df.loc[mask, col] = np.nan
                
    elif method == "isolation_forest":
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(df[num_cols])
        mask = preds == -1
        if action == "drop":
            df = df[~mask]
        elif action == "nan":
            df.loc[mask, num_cols] = np.nan

    return df


# Duplicate Handling
def handle_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first"
) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    logger.info(f"Removed {before - len(df)} duplicates")
    return df


# Type Conversion
def convert_types(
    df: pd.DataFrame, type_mapping: Optional[Dict[str, str]] = None,
    auto_convert: bool = True, datetime_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert column types.
    
    Args:
        df: DataFrame to process.
        type_mapping: {column: target_type}
        auto_convert: Auto-detect and convert types.
        datetime_columns: Columns to parse as datetime.
    """
    df = df.copy()
    
    if type_mapping:
        for col, dtype in type_mapping.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {e}")

    if datetime_columns:
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    if auto_convert:
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
    
    return df


# Text Cleaning
def clean_text(
    df: pd.DataFrame, columns: Optional[List[str]] = None,
    lowercase: bool = True, strip: bool = True, remove_special: bool = False
) -> pd.DataFrame:
    """Clean text columns."""
    df = df.copy()
    str_cols = columns or df.select_dtypes(include=['object']).columns.tolist()
    
    for col in str_cols:
        if strip:
            df[col] = df[col].str.strip()
        if lowercase:
            df[col] = df[col].str.lower()
        if remove_special:
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
    
    return df


# Full Pipeline
def preprocess(
    df: pd.DataFrame, missing_strategy: str = "mean", outlier_method: Optional[str] = None,
    remove_duplicates: bool = True, convert_types_auto: bool = True
) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    logger.info("Starting preprocessing pipeline")
    
    if remove_duplicates:
        df = handle_duplicates(df)
    
    if missing_strategy:
        df = handle_missing(df, strategy=missing_strategy)
    
    if outlier_method:
        df = handle_outliers(df, method=outlier_method)
    
    if convert_types_auto:
        df = convert_types(df, auto_convert=True)
    
    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df
