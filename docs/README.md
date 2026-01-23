# AdamOps Documentation

## Overview

AdamOps is a comprehensive MLOps library for end-to-end machine learning workflows.

## Modules

### Data Module (`adamops.data`)
- `loaders` - Load data from CSV, Excel, JSON, SQL, API, compressed files
- `validators` - Validate data types, missing values, duplicates
- `preprocessors` - Handle missing values, outliers, duplicates
- `feature_engineering` - Encode, scale, and generate features
- `splitters` - Split data for training and evaluation

### Models Module (`adamops.models`)
- `modelops` - Train regression, classification, and clustering models
- `registry` - Version and track models
- `ensembles` - Create ensemble models
- `automl` - Automated model selection and tuning

### Evaluation Module (`adamops.evaluation`)
- `metrics` - Classification, regression, and clustering metrics
- `visualization` - Plots and charts
- `explainability` - SHAP and LIME explanations
- `comparison` - Compare multiple models
- `reports` - Generate HTML/PDF reports

### Deployment Module (`adamops.deployment`)
- `exporters` - Export to ONNX, PMML, pickle, joblib
- `api` - Create FastAPI/Flask APIs
- `containerize` - Docker and Kubernetes deployment
- `cloud` - AWS, GCP, Azure deployment

### Monitoring Module (`adamops.monitoring`)
- `drift` - Detect data and concept drift
- `performance` - Track model performance
- `alerts` - Set up alerting
- `dashboard` - Monitoring dashboards

### Pipelines Module (`adamops.pipelines`)
- `workflows` - Define ML workflows as DAGs
- `orchestrators` - Schedule and run pipelines

## Installation

```bash
pip install adamops
```

## Quick Start

```python
from adamops.data import loaders, preprocessors
from adamops.models import modelops
from adamops.evaluation import metrics

# Load data
df = loaders.load_csv("data.csv")

# Train model
model = modelops.train(X, y, task="classification")

# Evaluate
results = metrics.evaluate(y_test, model.predict(X_test))
```

## CLI

```bash
adamops train --data data.csv --target y
adamops evaluate --model model.pkl --data test.csv
adamops deploy --model model.pkl --type api
```
