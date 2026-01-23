# AdamOps Examples

This directory contains example notebooks and scripts demonstrating AdamOps usage.

## Notebooks

1. **01_data_loading.ipynb** - Load data from various sources
2. **02_data_validation.ipynb** - Validate data quality
3. **03_data_cleaning.ipynb** - Clean and preprocess data
4. **04_feature_engineering.ipynb** - Feature encoding, scaling, and generation
5. **05_data_splitting.ipynb** - Split data for training and evaluation
6. **06_model_training.ipynb** - Train ML models
7. **07_model_registry.ipynb** - Version and manage models
8. **08_ensemble_models.ipynb** - Create ensemble models
9. **09_automl.ipynb** - Automated machine learning
10. **10_evaluation_metrics.ipynb** - Evaluate model performance

## Quick Start

```python
# Load and preprocess data
from adamops.data import loaders, preprocessors, feature_engineering, splitters

df = loaders.load_csv("data.csv")
df = preprocessors.handle_missing(df, strategy="mean")
df = feature_engineering.encode_onehot(df, columns=["category"])
X_train, X_test, y_train, y_test = splitters.split_train_test(X, y)

# Train model
from adamops.models import modelops

model = modelops.train(X_train, y_train, task="classification", algorithm="xgboost")

# Evaluate
from adamops.evaluation import metrics

results = metrics.evaluate(y_test, model.predict(X_test))
print(results)
```
