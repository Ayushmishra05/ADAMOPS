"""
AdamOps Data Validators Module

Provides data validation: type validation, missing value checks, 
duplicate detection, shape validation, and statistical checks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from adamops.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: str  # 'error', 'warning', 'info'
    category: str
    column: Optional[str]
    message: str
    details: Optional[Dict] = None


@dataclass
class ColumnStats:
    """Statistics for a column."""
    name: str
    dtype: str
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    shape: Tuple[int, int]
    memory_usage: float
    issues: List[ValidationIssue] = field(default_factory=list)
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)
    duplicate_rows: int = 0
    passed: bool = True

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 50, "VALIDATION REPORT", "=" * 50,
            f"Shape: {self.shape[0]} rows x {self.shape[1]} columns",
            f"Memory: {self.memory_usage:.2f} MB",
            f"Duplicates: {self.duplicate_rows}",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Issues: {len(self.issues)}", "=" * 50
        ]
        for issue in self.issues:
            col = f"[{issue.column}] " if issue.column else ""
            lines.append(f"[{issue.severity.upper()}] {col}{issue.message}")
        return "\n".join(lines)


class DataValidator:
    """Data validator for DataFrames."""
    
    def __init__(self, missing_threshold: float = 0.5, unique_threshold: float = 0.95):
        self.missing_threshold = missing_threshold
        self.unique_threshold = unique_threshold

    def validate(self, df: pd.DataFrame, schema: Optional[Dict] = None,
                 required_columns: Optional[List[str]] = None) -> ValidationReport:
        """Validate a DataFrame."""
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            shape=df.shape,
            memory_usage=df.memory_usage(deep=True).sum() / 1024**2,
        )
        
        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            for col in missing:
                report.issues.append(ValidationIssue("error", "schema", col, f"Missing: {col}"))

        # Check duplicates
        dups = df.duplicated().sum()
        report.duplicate_rows = dups
        if dups > 0:
            report.issues.append(ValidationIssue("warning", "duplicate", None, f"{dups} duplicates"))

        # Column stats
        for col in df.columns:
            series = df[col]
            missing = series.isna().sum()
            stats = ColumnStats(
                name=col, dtype=str(series.dtype), count=len(series),
                missing_count=missing, missing_pct=100*missing/len(series),
                unique_count=series.nunique(), unique_pct=100*series.nunique()/len(series),
            )
            if pd.api.types.is_numeric_dtype(series):
                stats.mean, stats.std = series.mean(), series.std()
                stats.min, stats.max = series.min(), series.max()
            report.column_stats[col] = stats
            
            if stats.missing_pct > self.missing_threshold * 100:
                report.issues.append(ValidationIssue("warning", "missing", col, 
                    f"High missing: {stats.missing_pct:.1f}%"))

        report.passed = not any(i.severity == "error" for i in report.issues)
        return report


def validate(df: pd.DataFrame, **kwargs) -> ValidationReport:
    """Validate a DataFrame."""
    return DataValidator().validate(df, **kwargs)

def check_missing(df: pd.DataFrame) -> Dict[str, Dict]:
    """Check missing values."""
    return {col: {"count": int(df[col].isna().sum()), "pct": 100*df[col].isna().mean()} 
            for col in df.columns if df[col].isna().any()}

def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Get duplicate rows."""
    return df[df.duplicated(subset=subset, keep=False)]

def check_types(df: pd.DataFrame) -> Dict[str, str]:
    """Get column types."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generate data description."""
    stats = []
    for col in df.columns:
        s = df[col]
        row = {"column": col, "dtype": str(s.dtype), "missing": s.isna().sum(), 
               "unique": s.nunique()}
        if pd.api.types.is_numeric_dtype(s):
            row.update({"mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max()})
        stats.append(row)
    return pd.DataFrame(stats)
