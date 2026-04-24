"""Tests for data module."""

import pytest
import pandas as pd
import numpy as np
from io import StringIO

from adamops.data.loaders import load_csv, load_auto
from adamops.data.validators import validate, check_missing, check_duplicates
from adamops.data.preprocessors import handle_missing, handle_outliers, handle_duplicates
from adamops.data.feature_engineering import encode_onehot, encode_label, scale_standard
from adamops.data.splitters import split_train_test, split_train_val_test


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)  # Deterministic tests
    return pd.DataFrame({
        'id': range(100),
        'numeric': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100),
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4, 5],
        'b': [1.0, np.nan, 3.0, np.nan, 5.0],
        'c': ['x', 'y', None, 'y', 'x'],
    })
    return df


class TestValidators:
    def test_validate_basic(self, sample_df):
        report = validate(sample_df)
        assert report.shape == sample_df.shape
        assert report.passed
    
    def test_check_missing(self, df_with_missing):
        missing = check_missing(df_with_missing)
        assert 'a' in missing
        assert 'b' in missing
    
    def test_check_duplicates(self, sample_df):
        df = pd.concat([sample_df, sample_df.iloc[:5]])
        dups = check_duplicates(df)
        assert len(dups) == 10  # 5 original + 5 duplicates


class TestPreprocessors:
    def test_handle_missing_mean(self, df_with_missing):
        result = handle_missing(df_with_missing, strategy='mean', columns=['a', 'b'])
        assert result['a'].isna().sum() == 0
        assert result['b'].isna().sum() == 0
    
    def test_handle_missing_drop(self, df_with_missing):
        result = handle_missing(df_with_missing, strategy='drop')
        assert len(result) < len(df_with_missing)
    
    def test_handle_outliers_iqr(self, sample_df):
        result = handle_outliers(sample_df, method='iqr', columns=['numeric'])
        assert len(result) <= len(sample_df)
    
    def test_handle_duplicates(self, sample_df):
        df = pd.concat([sample_df, sample_df.iloc[:5]])
        result = handle_duplicates(df)
        assert len(result) == len(sample_df)


class TestFeatureEngineering:
    def test_encode_onehot(self, sample_df):
        result = encode_onehot(sample_df, columns=['category'])
        assert 'category_A' in result.columns or 'category_B' in result.columns
        assert 'category' not in result.columns
    
    def test_encode_label(self, sample_df):
        result, encoders = encode_label(sample_df.copy(), columns=['category'])
        assert result['category'].dtype in [np.int32, np.int64]
        assert 'category' in encoders
    
    def test_scale_standard(self, sample_df):
        result = scale_standard(sample_df.copy(), columns=['numeric'])
        assert abs(result['numeric'].mean()) < 0.1
        assert abs(result['numeric'].std() - 1) < 0.1


class TestSplitters:
    def test_split_train_test(self, sample_df):
        X = sample_df.drop('target', axis=1)
        y = sample_df['target']
        
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_split_train_val_test(self, sample_df):
        X = sample_df.drop('target', axis=1)
        y = sample_df['target']
        
        result = split_train_val_test(X, y, train_size=0.7, val_size=0.15, test_size=0.15)
        
        assert len(result) == 6  # X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
