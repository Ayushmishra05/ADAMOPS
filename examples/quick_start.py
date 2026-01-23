"""
Quick Start Example - AdamOps

This script demonstrates a complete ML workflow using AdamOps.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate sample data
print("Generating sample data...")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['target'] = y

# Save to CSV for demonstration
df.to_csv('sample_data.csv', index=False)
print(f"Created sample_data.csv with {len(df)} rows")

# =============================================================================
# Step 1: Load and Validate Data
# =============================================================================
print("\n" + "="*60)
print("Step 1: Load and Validate Data")
print("="*60)

from adamops.data.loaders import load_csv
from adamops.data.validators import validate

df = load_csv('sample_data.csv')
report = validate(df)
print(report.summary())

# =============================================================================
# Step 2: Preprocess Data
# =============================================================================
print("\n" + "="*60)
print("Step 2: Preprocess Data")
print("="*60)

from adamops.data.preprocessors import handle_missing, handle_outliers
from adamops.data.feature_engineering import scale_standard

# Handle any missing values
df = handle_missing(df, strategy='mean')

# Handle outliers
df = handle_outliers(df, method='iqr', action='clip')

# Scale features
feature_cols = [c for c in df.columns if c != 'target']
df = scale_standard(df, columns=feature_cols)

print(f"Preprocessed data shape: {df.shape}")

# =============================================================================
# Step 3: Split Data
# =============================================================================
print("\n" + "="*60)
print("Step 3: Split Data")
print("="*60)

from adamops.data.splitters import split_train_test

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, stratify=True)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# =============================================================================
# Step 4: Train Model
# =============================================================================
print("\n" + "="*60)
print("Step 4: Train Model")
print("="*60)

from adamops.models.modelops import train, compare_models

# Compare multiple models
print("Comparing models...")
comparison = compare_models(X_train, y_train, task='classification', cv=5)
print(comparison)

# Train best model
print("\nTraining Random Forest...")
model = train(X_train, y_train, task='classification', algorithm='random_forest')
print(f"Model trained: {model.algorithm}")

# =============================================================================
# Step 5: Evaluate Model
# =============================================================================
print("\n" + "="*60)
print("Step 5: Evaluate Model")
print("="*60)

from adamops.evaluation.metrics import evaluate, classification_report

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

metrics = evaluate(y_test, y_pred, task='classification', y_prob=y_prob)

print("\nEvaluation Results:")
for name, value in metrics.items():
    if isinstance(value, float):
        print(f"  {name}: {value:.4f}")

# =============================================================================
# Step 6: Save Model
# =============================================================================
print("\n" + "="*60)
print("Step 6: Save Model")
print("="*60)

from adamops.deployment.exporters import export_joblib

export_joblib(model, 'trained_model.joblib')
print("Model saved to trained_model.joblib")

# =============================================================================
# Step 7: Register Model
# =============================================================================
print("\n" + "="*60)
print("Step 7: Register Model")
print("="*60)

from adamops.models.registry import ModelRegistry

registry = ModelRegistry()
version = registry.register('quickstart_model', model, metadata={'metrics': metrics})
print(f"Registered as: {version.name} {version.version}")

print("\n" + "="*60)
print("Quick Start Complete!")
print("="*60)

# Cleanup
import os
os.remove('sample_data.csv')
os.remove('trained_model.joblib')
