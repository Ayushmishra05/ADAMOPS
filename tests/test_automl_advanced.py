import pytest
import numpy as np
import pandas as pd
from adamops.models.automl import run
import os

pytestmark = pytest.mark.filterwarnings("ignore")

@pytest.fixture
def make_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_local_optuna_automl(make_data):
    X, y = make_data
    result = run(X, y, backend='local', n_trials=2, cv=2)
    assert result.best_model is not None
    assert isinstance(result.leaderboard, pd.DataFrame)
    assert 'score' in result.leaderboard.columns

def test_distributed_ray_automl(make_data):
    X, y = make_data
    import ray
    # Start ray in minimal local mode avoiding dashboard or multiprocess overhead
    os.environ["RAY_DEDUP_LOGS"] = "0"
    if not ray.is_initialized():
        ray.init(local_mode=True, include_dashboard=False)
        
    try:
        result = run(X, y, backend='distributed', n_trials=2, cv=2)
        assert result.best_model is not None
        assert isinstance(result.leaderboard, pd.DataFrame)
        assert 'score' in result.leaderboard.columns
    finally:
        ray.shutdown()
