"""
AdamOps Data Module

Provides comprehensive data handling capabilities:
- loaders: Load data from various sources (CSV, Excel, JSON, SQL, API, compressed files)
- validators: Validate data types, missing values, duplicates, shapes, and statistics
- preprocessors: Clean data (handle missing values, outliers, duplicates, type conversion)
- feature_engineering: Encode, scale, and generate features
- splitters: Split data for training and evaluation
"""

from . import loaders
from . import validators
from . import preprocessors
from . import feature_engineering
from . import splitters

__all__ = [
    "loaders",
    "validators",
    "preprocessors",
    "feature_engineering",
    "splitters",
]
