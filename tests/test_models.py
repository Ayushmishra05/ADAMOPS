"""Tests for models module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from adamops.models.modelops import train, train_regression, train_classification, get_available_models
from adamops.models.registry import ModelRegistry
from adamops.models.ensembles import VotingEnsemble, create_voting_ensemble
from adamops.models.automl import run as run_automl


@pytest.fixture
def classification_data():
    """Create classification dataset."""
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(10)]), pd.Series(y)


@pytest.fixture
def regression_data():
    """Create regression dataset."""
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(10)]), pd.Series(y)


class TestModelOps:
    def test_train_classification(self, classification_data):
        X, y = classification_data
        model = train(X, y, task='classification', algorithm='random_forest')
        
        assert model.is_fitted
        assert model.task == 'classification'
        assert model.algorithm == 'random_forest'
        
        preds = model.predict(X)
        assert len(preds) == len(y)
    
    def test_train_regression(self, regression_data):
        X, y = regression_data
        model = train(X, y, task='regression', algorithm='ridge')
        
        assert model.is_fitted
        assert model.task == 'regression'
        
        preds = model.predict(X)
        assert len(preds) == len(y)
    
    def test_train_auto_task(self, classification_data):
        X, y = classification_data
        model = train(X, y, task='auto', algorithm='random_forest')
        
        assert model.task in ['classification', 'multiclass']
    
    def test_get_available_models(self):
        classification_models = get_available_models('classification')
        regression_models = get_available_models('regression')
        
        assert 'random_forest' in classification_models
        assert 'ridge' in regression_models


class TestRegistry:
    def test_register_and_load(self, classification_data, tmp_path):
        X, y = classification_data
        model = train(X, y, task='classification', algorithm='random_forest')
        
        registry = ModelRegistry(str(tmp_path / 'registry'))
        
        # Register
        version = registry.register('test_model', model, metadata={'accuracy': 0.9})
        assert version.version == 'v1'
        
        # Load
        loaded = registry.load('test_model', 'v1')
        assert loaded is not None
    
    def test_list_versions(self, classification_data, tmp_path):
        X, y = classification_data
        model = train(X, y, task='classification', algorithm='random_forest')
        
        registry = ModelRegistry(str(tmp_path / 'registry'))
        
        registry.register('test_model', model)
        registry.register('test_model', model)
        
        versions = registry.list_versions('test_model')
        assert len(versions) == 2


class TestEnsembles:
    def test_voting_ensemble(self, classification_data):
        X, y = classification_data
        
        ensemble = create_voting_ensemble(
            ['random_forest', 'logistic'],
            task='classification'
        )
        ensemble.fit(X, y)
        
        preds = ensemble.predict(X)
        assert len(preds) == len(y)


class TestAutoML:
    def test_automl_quick(self, classification_data):
        X, y = classification_data
        
        result = run_automl(X, y, task='classification', tuning='none', n_trials=5)
        
        assert result.best_model is not None
        assert result.best_score > 0
        assert len(result.leaderboard) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
