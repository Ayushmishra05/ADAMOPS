"""
AdamOps Feature Engineering Module

Provides encoding, scaling, feature selection, and auto feature generation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler, 
    MinMaxScaler, RobustScaler, MaxAbsScaler, PolynomialFeatures
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif, mutual_info_regression, RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from adamops.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Encoding
# =============================================================================

def encode_onehot(
    df: pd.DataFrame, columns: List[str], drop_first: bool = False,
    handle_unknown: str = "ignore"
) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    df = df.copy()
    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    logger.info(f"One-hot encoded {len(columns)} columns")
    return df


def encode_label(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Label encode categorical columns. Returns df and encoders dict."""
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    logger.info(f"Label encoded {len(columns)} columns")
    return df, encoders


def encode_ordinal(
    df: pd.DataFrame, columns: List[str], 
    categories: Optional[Dict[str, List]] = None
) -> pd.DataFrame:
    """Ordinal encode columns with optional category order."""
    df = df.copy()
    for col in columns:
        if categories and col in categories:
            cat_map = {v: i for i, v in enumerate(categories[col])}
            df[col] = df[col].map(cat_map)
        else:
            df[col] = pd.Categorical(df[col]).codes
    return df


def encode_target(
    df: pd.DataFrame, columns: List[str], target: str, smoothing: float = 1.0
) -> pd.DataFrame:
    """Target encode categorical columns."""
    df = df.copy()
    global_mean = df[target].mean()
    
    for col in columns:
        agg = df.groupby(col)[target].agg(['mean', 'count'])
        smooth = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
        df[col + '_target'] = df[col].map(smooth)
        df = df.drop(col, axis=1)
    
    return df


def encode(
    df: pd.DataFrame, columns: List[str], method: str = "onehot", **kwargs
) -> pd.DataFrame:
    """Encode categorical columns with specified method."""
    if method == "onehot":
        return encode_onehot(df, columns, **kwargs)
    elif method == "label":
        return encode_label(df, columns, **kwargs)[0]
    elif method == "ordinal":
        return encode_ordinal(df, columns, **kwargs)
    elif method == "target" and "target" in kwargs:
        return encode_target(df, columns, kwargs["target"])
    else:
        raise ValueError(f"Unknown encoding method: {method}")


# =============================================================================
# Scaling
# =============================================================================

def scale_standard(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Standardize features (zero mean, unit variance)."""
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def scale_minmax(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Scale features to [0, 1] range."""
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def scale_robust(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Scale with median and IQR (robust to outliers)."""
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def scale(
    df: pd.DataFrame, method: str = "standard", columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Scale numeric columns with specified method."""
    if method == "standard":
        return scale_standard(df, columns)
    elif method == "minmax":
        return scale_minmax(df, columns)
    elif method == "robust":
        return scale_robust(df, columns)
    elif method == "maxabs":
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        df[cols] = MaxAbsScaler().fit_transform(df[cols])
        return df
    else:
        raise ValueError(f"Unknown scaling method: {method}")


# =============================================================================
# Feature Selection
# =============================================================================

def select_by_variance(
    df: pd.DataFrame, threshold: float = 0.0, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Remove low variance features."""
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    selector = VarianceThreshold(threshold=threshold)
    selected = selector.fit_transform(df[cols])
    selected_cols = [cols[i] for i in selector.get_support(indices=True)]
    df_result = df.drop(cols, axis=1)
    df_result[selected_cols] = selected
    logger.info(f"Selected {len(selected_cols)}/{len(cols)} features by variance")
    return df_result


def select_by_correlation(
    df: pd.DataFrame, threshold: float = 0.9, target: Optional[str] = None
) -> pd.DataFrame:
    """Remove highly correlated features."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in num_cols:
        num_cols.remove(target)
    
    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    
    logger.info(f"Dropping {len(to_drop)} highly correlated features")
    return df.drop(to_drop, axis=1)


def select_by_importance(
    df: pd.DataFrame, target: str, n_features: int = 10, task: str = "classification"
) -> pd.DataFrame:
    """Select features by tree-based importance."""
    X = df.drop(target, axis=1).select_dtypes(include=[np.number])
    y = df[target]
    
    model = RandomForestClassifier(n_estimators=50, random_state=42) if task == "classification" \
        else RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importance.head(n_features).index.tolist()
    
    logger.info(f"Selected top {n_features} features by importance")
    return df[[target] + top_features]


def select_features(
    df: pd.DataFrame, target: str, method: str = "importance", n_features: int = 10, **kwargs
) -> pd.DataFrame:
    """Select features using specified method."""
    if method == "variance":
        return select_by_variance(df, **kwargs)
    elif method == "correlation":
        return select_by_correlation(df, target=target, **kwargs)
    elif method == "importance":
        return select_by_importance(df, target, n_features, **kwargs)
    else:
        raise ValueError(f"Unknown selection method: {method}")


# =============================================================================
# Feature Generation
# =============================================================================

def generate_polynomial(
    df: pd.DataFrame, columns: List[str], degree: int = 2, include_bias: bool = False
) -> pd.DataFrame:
    """Generate polynomial features."""
    df = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_features = poly.fit_transform(df[columns])
    poly_names = poly.get_feature_names_out(columns)
    df_poly = pd.DataFrame(poly_features, columns=poly_names, index=df.index)
    return pd.concat([df.drop(columns, axis=1), df_poly], axis=1)


def generate_interactions(
    df: pd.DataFrame, columns: List[str], operations: List[str] = ["multiply"]
) -> pd.DataFrame:
    """Generate interaction features between columns."""
    df = df.copy()
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            if "multiply" in operations:
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            if "add" in operations:
                df[f"{col1}_+_{col2}"] = df[col1] + df[col2]
            if "divide" in operations:
                df[f"{col1}_/_{col2}"] = df[col1] / (df[col2] + 1e-8)
    return df


def generate_datetime_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Extract datetime features from a column."""
    df = df.copy()
    dt = pd.to_datetime(df[column])
    prefix = column
    df[f"{prefix}_year"] = dt.dt.year
    df[f"{prefix}_month"] = dt.dt.month
    df[f"{prefix}_day"] = dt.dt.day
    df[f"{prefix}_dayofweek"] = dt.dt.dayofweek
    df[f"{prefix}_hour"] = dt.dt.hour
    df[f"{prefix}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    return df


def auto_feature_engineering(
    df: pd.DataFrame, target: Optional[str] = None,
    polynomial: bool = False, interactions: bool = False, datetime_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Automatic feature engineering pipeline."""
    logger.info("Running auto feature engineering")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in num_cols:
        num_cols.remove(target)
    
    if datetime_cols:
        for col in datetime_cols:
            df = generate_datetime_features(df, col)
    
    if polynomial and len(num_cols) <= 5:
        df = generate_polynomial(df, num_cols[:5], degree=2)
    
    if interactions and len(num_cols) >= 2:
        df = generate_interactions(df, num_cols[:4])
    
    logger.info(f"Feature engineering complete. New shape: {df.shape}")
    return df
