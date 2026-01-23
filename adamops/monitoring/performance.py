"""
AdamOps Performance Monitoring Module

Track model performance over time.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adamops.utils.logging import get_logger
from adamops.utils.helpers import ensure_dir
from adamops.evaluation.metrics import evaluate

logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(self, model_name: str, storage_path: Optional[str] = None):
        """
        Initialize performance monitor.
        
        Args:
            model_name: Name of the model being monitored.
            storage_path: Path to store performance logs.
        """
        self.model_name = model_name
        self.storage_path = Path(storage_path or f".adamops_monitor/{model_name}")
        ensure_dir(self.storage_path)
        
        self.metrics_file = self.storage_path / "metrics.json"
        self.predictions_file = self.storage_path / "predictions.json"
        
        self._load_history()
    
    def _load_history(self):
        """Load existing metrics history."""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                self.metrics_history = json.load(f)
        else:
            self.metrics_history = []
    
    def _save_history(self):
        """Save metrics history."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def log_metrics(
        self, metrics: Dict[str, float], 
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log performance metrics.
        
        Args:
            metrics: Dict of metric name to value.
            timestamp: Timestamp (uses current if None).
            metadata: Additional metadata.
        """
        entry = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "metrics": metrics,
            "metadata": metadata or {},
        }
        
        self.metrics_history.append(entry)
        self._save_history()
        
        logger.info(f"Logged metrics for {self.model_name}: {metrics}")
    
    def log_prediction(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        task: str = "classification", y_prob: Optional[np.ndarray] = None
    ):
        """Log prediction and compute metrics."""
        metrics = evaluate(y_true, y_pred, task, y_prob)
        self.log_metrics(metrics, metadata={"task": task, "n_samples": len(y_true)})
        return metrics
    
    def get_history(self, n_latest: Optional[int] = None) -> List[Dict]:
        """Get metrics history."""
        if n_latest:
            return self.metrics_history[-n_latest:]
        return self.metrics_history
    
    def get_metric_trend(self, metric: str) -> pd.DataFrame:
        """Get trend for a specific metric."""
        data = []
        for entry in self.metrics_history:
            if metric in entry["metrics"]:
                data.append({
                    "timestamp": entry["timestamp"],
                    "value": entry["metrics"][metric],
                })
        
        return pd.DataFrame(data)
    
    def detect_degradation(
        self, metric: str, threshold: float = 0.1, window: int = 5
    ) -> Dict:
        """
        Detect performance degradation.
        
        Args:
            metric: Metric to monitor.
            threshold: Relative change threshold.
            window: Number of recent entries to compare.
        
        Returns:
            Dict with degradation info.
        """
        trend = self.get_metric_trend(metric)
        
        if len(trend) < window + 1:
            return {"degraded": False, "reason": "insufficient_data"}
        
        baseline = trend["value"].iloc[:-window].mean()
        recent = trend["value"].iloc[-window:].mean()
        
        change = (baseline - recent) / baseline if baseline != 0 else 0
        degraded = change > threshold
        
        result = {
            "degraded": degraded,
            "metric": metric,
            "baseline": baseline,
            "recent": recent,
            "change_pct": change * 100,
            "threshold_pct": threshold * 100,
        }
        
        if degraded:
            logger.warning(f"Performance degradation detected for {metric}: {change*100:.1f}% drop")
        
        return result
    
    def summary(self) -> Dict:
        """Get monitoring summary."""
        if not self.metrics_history:
            return {"model": self.model_name, "entries": 0}
        
        latest = self.metrics_history[-1]
        
        return {
            "model": self.model_name,
            "entries": len(self.metrics_history),
            "latest_timestamp": latest["timestamp"],
            "latest_metrics": latest["metrics"],
        }


class LatencyMonitor:
    """Monitor prediction latency."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.latencies = []
    
    def record(self, latency_ms: float):
        """Record a latency measurement."""
        self.latencies.append({
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
        })
    
    def get_stats(self) -> Dict:
        """Get latency statistics."""
        if not self.latencies:
            return {}
        
        values = [l["latency_ms"] for l in self.latencies]
        
        return {
            "count": len(values),
            "mean_ms": np.mean(values),
            "std_ms": np.std(values),
            "p50_ms": np.percentile(values, 50),
            "p95_ms": np.percentile(values, 95),
            "p99_ms": np.percentile(values, 99),
            "min_ms": np.min(values),
            "max_ms": np.max(values),
        }


def create_monitor(model_name: str, storage_path: Optional[str] = None) -> PerformanceMonitor:
    """Create a performance monitor."""
    return PerformanceMonitor(model_name, storage_path)
