"""
Tests for AdamOps Trainer Module.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from adamops.trainer import Trainer

@pytest.fixture
def dummy_data():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_trainer_init_sklearn():
    trainer = Trainer("random_forest", task="classification")
    assert trainer.backend == "sklearn"
    assert trainer.algorithm == "random_forest"
    # Auto-strategy for sklearn local is joblib or single initially
    assert trainer.strategy in ["single", "joblib", "ray"]

def test_trainer_init_pytorch():
    try:
        import torch
        model = torch.nn.Linear(10, 2)
        trainer = Trainer(model, task="regression")
        assert trainer.backend == "pytorch"
        assert trainer.raw_model is not None
        assert trainer.strategy in ["single", "ddp", "ray"]
    except ImportError:
        pass

@patch("adamops.trainer.sklearn_train")
def test_trainer_fit_sklearn(mock_train, dummy_data):
    X, y = dummy_data
    mock_train.return_value = MagicMock()
    
    trainer = Trainer("random_forest")
    trainer.fit(X, y)
    
    mock_train.assert_called_once()
    assert trainer.trained_model is not None

@patch("adamops.trainer.DeviceManager.estimate_model_vram", return_value=500.0)
@patch("adamops.trainer.DeviceManager.vram_preflight_check", return_value=False)
def test_trainer_fit_oom_protection(mock_vram_check, mock_est, dummy_data):
    try:
        import torch
        X, y = dummy_data
        model = torch.nn.Linear(10, 2)
        trainer = Trainer(model)
        
        # Should raise MemoryError because vram preflight returns False
        with pytest.raises(MemoryError):
            trainer.fit(X, y)
    except ImportError:
        pass

def test_trainer_predict_sklearn(dummy_data):
    X, y = dummy_data
    trainer = Trainer("random_forest")
    # mock trained model internally
    trainer.trained_model = MagicMock()
    trainer.trained_model.predict.return_value = np.zeros(100)
    
    preds = trainer.predict(X)
    assert len(preds) == 100
    trainer.trained_model.predict.assert_called_once()
