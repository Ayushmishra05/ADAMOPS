"""
AdamOps Drift Detection Module

Detect data drift and concept drift in production.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from adamops.utils.logging import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """Detect data and concept drift."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference/training data.
            threshold: P-value threshold for drift detection.
        """
        self.reference = reference_data
        self.threshold = threshold
        self._compute_reference_stats()
    
    def _compute_reference_stats(self):
        """Compute statistics for reference data."""
        self.ref_stats = {}
        
        for col in self.reference.columns:
            if pd.api.types.is_numeric_dtype(self.reference[col]):
                self.ref_stats[col] = {
                    "type": "numeric",
                    "mean": self.reference[col].mean(),
                    "std": self.reference[col].std(),
                    "min": self.reference[col].min(),
                    "max": self.reference[col].max(),
                    "values": self.reference[col].dropna().values,
                }
            else:
                value_counts = self.reference[col].value_counts(normalize=True)
                self.ref_stats[col] = {
                    "type": "categorical",
                    "distribution": value_counts.to_dict(),
                    "values": self.reference[col].dropna().values,
                }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Returns:
            Dict with drift detection results.
        """
        results = {"drift_detected": False, "columns": {}, "summary": {}}
        drift_count = 0
        
        for col in self.reference.columns:
            if col not in current_data.columns:
                continue
            
            col_result = self._detect_column_drift(col, current_data[col])
            results["columns"][col] = col_result
            
            if col_result["drift_detected"]:
                drift_count += 1
        
        results["drift_detected"] = drift_count > 0
        results["summary"] = {
            "total_columns": len(self.reference.columns),
            "drifted_columns": drift_count,
            "drift_ratio": drift_count / len(self.reference.columns),
        }
        
        if results["drift_detected"]:
            logger.warning(f"Drift detected in {drift_count} columns")
        
        return results
    
    def _detect_column_drift(self, col: str, current: pd.Series) -> Dict:
        """Detect drift for a single column."""
        ref_stats = self.ref_stats.get(col)
        if ref_stats is None:
            return {"drift_detected": False, "reason": "unknown_column"}
        
        current_values = current.dropna().values
        
        if ref_stats["type"] == "numeric":
            # Kolmogorov-Smirnov test
            stat, pvalue = stats.ks_2samp(ref_stats["values"], current_values)
            drift_detected = pvalue < self.threshold
            
            return {
                "drift_detected": drift_detected,
                "test": "ks_test",
                "statistic": float(stat),
                "p_value": float(pvalue),
                "ref_mean": ref_stats["mean"],
                "current_mean": float(current.mean()),
            }
        else:
            # Chi-square test for categorical
            current_dist = current.value_counts(normalize=True)
            
            # Align distributions
            all_categories = set(ref_stats["distribution"].keys()) | set(current_dist.index)
            ref_freq = [ref_stats["distribution"].get(c, 0.001) for c in all_categories]
            cur_freq = [current_dist.get(c, 0.001) for c in all_categories]
            
            # Normalize
            ref_freq = np.array(ref_freq) / sum(ref_freq)
            cur_freq = np.array(cur_freq) / sum(cur_freq)
            
            # Chi-square
            stat, pvalue = stats.chisquare(cur_freq, ref_freq)
            drift_detected = pvalue < self.threshold
            
            return {
                "drift_detected": drift_detected,
                "test": "chi_square",
                "statistic": float(stat),
                "p_value": float(pvalue),
            }
    
    def get_drift_report(self, current_data: pd.DataFrame) -> str:
        """Generate human-readable drift report."""
        results = self.detect_drift(current_data)
        
        lines = [
            "=" * 50,
            "DRIFT DETECTION REPORT",
            "=" * 50,
            f"Status: {'DRIFT DETECTED' if results['drift_detected'] else 'NO DRIFT'}",
            f"Columns with drift: {results['summary']['drifted_columns']}/{results['summary']['total_columns']}",
            "",
        ]
        
        if results["drift_detected"]:
            lines.append("Drifted Columns:")
            for col, info in results["columns"].items():
                if info["drift_detected"]:
                    lines.append(f"  - {col}: p-value={info['p_value']:.4f} ({info['test']})")
        
        return "\n".join(lines)


class PSI:
    """Population Stability Index calculator."""
    
    @staticmethod
    def calculate(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate PSI between reference and current distributions.
        
        PSI < 0.1: No significant change
        0.1 < PSI < 0.2: Moderate change
        PSI > 0.2: Significant change
        """
        # Create bins from reference
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Get distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)
        
        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    @staticmethod
    def interpret(psi: float) -> str:
        """Interpret PSI value."""
        if psi < 0.1:
            return "No significant change"
        elif psi < 0.2:
            return "Moderate change - monitor closely"
        else:
            return "Significant change - investigate"


def detect_drift(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.05
) -> Dict:
    """Detect drift between reference and current data."""
    detector = DriftDetector(reference, threshold)
    return detector.detect_drift(current)


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> Dict:
    """Calculate PSI with interpretation."""
    psi = PSI.calculate(reference, current, bins)
    return {
        "psi": psi,
        "interpretation": PSI.interpret(psi),
        "significant": psi > 0.2,
    }
