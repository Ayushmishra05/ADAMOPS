"""Tests for evaluation module."""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression

from adamops.evaluation.metrics import (
    classification_metrics, regression_metrics, clustering_metrics, evaluate
)
from adamops.evaluation.comparison import compare_models, rank_models


@pytest.fixture
def classification_predictions():
    """Create classification predictions."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.9, 0.6, 0.2])
    return y_true, y_pred, y_prob


@pytest.fixture
def regression_predictions():
    """Create regression predictions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    return y_true, y_pred


class TestClassificationMetrics:
    def test_basic_metrics(self, classification_predictions):
        y_true, y_pred, y_prob = classification_predictions
        
        metrics = classification_metrics(y_true, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1


class TestRegressionMetrics:
    def test_basic_metrics(self, regression_predictions):
        y_true, y_pred = regression_predictions
        
        metrics = regression_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0


class TestEvaluate:
    def test_auto_detection(self, classification_predictions):
        y_true, y_pred, _ = classification_predictions
        
        metrics = evaluate(y_true, y_pred, task='auto')
        
        assert 'accuracy' in metrics or 'mse' in metrics
    
    def test_explicit_task(self, regression_predictions):
        y_true, y_pred = regression_predictions
        
        metrics = evaluate(y_true, y_pred, task='regression')
        
        assert 'mse' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
