"""
AdamOps Helpers Module

Provides common utility functions used across the library.
"""

import os
import json
import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from datetime import datetime
from functools import wraps
import time

import numpy as np
import pandas as pd

T = TypeVar("T")


# =============================================================================
# Type Checking and Validation
# =============================================================================

def is_numeric(value: Any) -> bool:
    """
    Check if a value is numeric.
    
    Args:
        value: Value to check.
    
    Returns:
        bool: True if value is numeric.
    
    Example:
        >>> is_numeric(42)
        True
        >>> is_numeric("hello")
        False
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def is_categorical(series: pd.Series, threshold: float = 0.05) -> bool:
    """
    Check if a pandas Series is likely categorical.
    
    Args:
        series: Pandas Series to check.
        threshold: Ratio of unique values to total values.
    
    Returns:
        bool: True if series is likely categorical.
    
    Example:
        >>> df = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
        >>> is_categorical(df["cat"])
        True
    """
    if series.dtype in ["object", "category", "bool"]:
        return True
    
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < threshold
    
    return False


def infer_task_type(y: Union[np.ndarray, pd.Series]) -> str:
    """
    Infer the task type from the target variable.
    
    Args:
        y: Target variable.
    
    Returns:
        str: Task type ("classification", "regression", or "multiclass").
    
    Example:
        >>> y = np.array([0, 1, 0, 1])
        >>> infer_task_type(y)
        'classification'
    """
    if isinstance(y, pd.Series):
        y = y.values
    
    unique_values = np.unique(y)
    n_unique = len(unique_values)
    
    # Check if it's a classification problem
    if y.dtype in [np.object_, np.bool_] or n_unique <= 10:
        if n_unique == 2:
            return "classification"
        else:
            return "multiclass"
    
    # Check if values are continuous
    if np.issubdtype(y.dtype, np.floating):
        return "regression"
    
    # Integer with many unique values -> regression
    if np.issubdtype(y.dtype, np.integer) and n_unique > 10:
        return "regression"
    
    return "classification"


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    Validate a pandas DataFrame.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of required column names.
    
    Returns:
        bool: True if valid.
    
    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(df).__name__}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return True


# =============================================================================
# Data Conversion
# =============================================================================

def to_numpy(data: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> np.ndarray:
    """
    Convert data to numpy array.
    
    Args:
        data: Data to convert.
    
    Returns:
        np.ndarray: Numpy array.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise ValueError(f"Cannot convert {type(data).__name__} to numpy array")


def to_dataframe(data: Union[np.ndarray, pd.DataFrame, pd.Series, dict, list], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert data to pandas DataFrame.
    
    Args:
        data: Data to convert.
        columns: Optional column names.
    
    Returns:
        pd.DataFrame: Pandas DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, pd.Series):
        return data.to_frame()
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data, columns=columns)
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    elif isinstance(data, list):
        return pd.DataFrame(data, columns=columns)
    else:
        raise ValueError(f"Cannot convert {type(data).__name__} to DataFrame")


def safe_cast(value: Any, target_type: Type[T], default: Optional[T] = None) -> Optional[T]:
    """
    Safely cast a value to a target type.
    
    Args:
        value: Value to cast.
        target_type: Target type.
        default: Default value if casting fails.
    
    Returns:
        Cast value or default.
    
    Example:
        >>> safe_cast("42", int)
        42
        >>> safe_cast("hello", int, default=0)
        0
    """
    try:
        return target_type(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# File Operations
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory.
    
    Returns:
        Path: Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(filepath: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.
    
    Args:
        filepath: Path to file.
        algorithm: Hash algorithm (md5, sha1, sha256).
    
    Returns:
        str: Hex digest of file hash.
    """
    hash_func = getattr(hashlib, algorithm)()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def save_object(obj: Any, filepath: Union[str, Path], format: str = "pickle") -> None:
    """
    Save an object to file.
    
    Args:
        obj: Object to save.
        filepath: Path to save to.
        format: Save format (pickle, json, joblib).
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    if format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    elif format == "json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
    elif format == "joblib":
        import joblib
        joblib.dump(obj, filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_object(filepath: Union[str, Path], format: str = "pickle") -> Any:
    """
    Load an object from file.
    
    Args:
        filepath: Path to load from.
        format: Load format (pickle, json, joblib).
    
    Returns:
        Loaded object.
    """
    filepath = Path(filepath)
    
    if format == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif format == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    elif format == "joblib":
        import joblib
        return joblib.load(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


# =============================================================================
# String Operations
# =============================================================================

def slugify(text: str) -> str:
    """
    Convert text to URL-friendly slug.
    
    Args:
        text: Text to convert.
    
    Returns:
        str: Slugified text.
    
    Example:
        >>> slugify("Hello World!")
        'hello-world'
    """
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = text.strip("-")
    return text


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate.
        max_length: Maximum length.
        suffix: Suffix to add if truncated.
    
    Returns:
        str: Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# =============================================================================
# Timing and Performance
# =============================================================================

def timeit(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time.
    
    Returns:
        Wrapped function.
    
    Example:
        >>> @timeit
        ... def slow_function():
        ...     time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.4f}s")
        return result
    return wrapper


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum retry attempts.
        delay: Initial delay between retries.
        backoff: Multiplier for delay after each retry.
        exceptions: Exceptions to catch and retry.
    
    Returns:
        Decorator function.
    
    Example:
        >>> @retry(max_attempts=3, delay=1.0)
        ... def unstable_function():
        ...     # May fail sometimes
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# Memory and Performance
# =============================================================================

def get_memory_usage(obj: Any) -> int:
    """
    Get memory usage of an object in bytes.
    
    Args:
        obj: Object to measure.
    
    Returns:
        int: Memory usage in bytes.
    """
    import sys
    return sys.getsizeof(obj)


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize.
        verbose: Whether to print memory savings.
    
    Returns:
        pd.DataFrame: Optimized DataFrame.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df


# =============================================================================
# Timestamp Utilities
# =============================================================================

def now_str(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format: Datetime format string.
    
    Returns:
        str: Formatted timestamp.
    """
    return datetime.now().strftime(format)


def parse_timestamp(timestamp: Union[str, int, float, datetime]) -> datetime:
    """
    Parse various timestamp formats to datetime.
    
    Args:
        timestamp: Timestamp to parse.
    
    Returns:
        datetime: Parsed datetime object.
    """
    if isinstance(timestamp, datetime):
        return timestamp
    elif isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, str):
        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y%m%d",
            "%Y%m%d_%H%M%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse timestamp: {timestamp}")
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")


# =============================================================================
# Validation Decorators
# =============================================================================

def validate_args(**validators: Callable[[Any], bool]) -> Callable:
    """
    Decorator to validate function arguments.
    
    Args:
        **validators: Mapping of argument names to validation functions.
    
    Returns:
        Decorator function.
    
    Example:
        >>> @validate_args(x=lambda x: x > 0)
        ... def process(x):
        ...     return x * 2
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for argument '{arg_name}': {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def deprecated(message: str = "", version: str = "") -> Callable:
    """
    Decorator to mark a function as deprecated.
    
    Args:
        message: Deprecation message.
        version: Version when the function will be removed.
    
    Returns:
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            warn_msg = f"{func.__name__} is deprecated"
            if version:
                warn_msg += f" and will be removed in version {version}"
            if message:
                warn_msg += f". {message}"
            warnings.warn(warn_msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Random Seeds
# =============================================================================

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set other seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# =============================================================================
# Progress Tracking
# =============================================================================

class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    
    Example:
        >>> tracker = ProgressTracker(total=100)
        >>> for i in range(100):
        ...     tracker.update()
        >>> tracker.finish()
    """
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        pct = 100 * self.current / self.total
        print(f"\r{self.description}: {self.current}/{self.total} ({pct:.1f}%) | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="")
    
    def finish(self) -> None:
        """Mark progress as complete."""
        elapsed = time.time() - self.start_time
        print(f"\n{self.description} completed in {elapsed:.2f}s")
