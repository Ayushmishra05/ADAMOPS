# 🚀 AdamOps

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.1-orange.svg)](https://github.com/adamops/adamops)



**AdamOps** is a comprehensive MLOps library for end-to-end machine learning workflows. It provides a unified interface for data processing, model training, evaluation, deployment, and monitoring.

## ✨ Features

### 📊 Data Module (DataOps)
- **Loaders**: CSV, Excel, JSON, SQL, API, compressed files with auto-encoding detection
- **Validators**: Type validation, missing values, duplicates, shape, statistical checks
- **Preprocessors**: Missing value imputation, outlier handling, text cleaning
- **Feature Engineering**: Encoding, scaling, feature selection, auto feature generation
- **Splitters**: Train/test, time-series, K-Fold, stratified splitting

### 🤖 Model Module (ModelOps)
- **Regression**: Ridge, Lasso, ElasticNet, XGBoost, LightGBM
- **Classification**: Decision Tree, Gradient Boosting, XGBoost, LightGBM, Naive Bayes, KNN
- **Clustering**: K-Means, DBSCAN, Hierarchical, GMM
- **Ensembles**: Voting, Stacking, Blending, Weighted averaging
- **AutoML**: Model selection, hyperparameter tuning (Grid, Random, Bayesian)

### 📈 Evaluation Module
- **Metrics**: Classification, regression, and clustering metrics
- **Visualization**: Confusion matrices, ROC curves, feature importance plots
- **Explainability**: SHAP and LIME explanations
- **Reports**: HTML/PDF report generation

### 🚀 Deployment Module
- **Exporters**: ONNX, PMML, TFLite, CoreML
- **APIs**: FastAPI, Flask, Streamlit
- **Containerization**: Docker, Kubernetes
- **Cloud**: AWS, GCP, Azure

### 📡 Monitoring Module
- **Drift Detection**: Data and concept drift
- **Performance Tracking**: Model metrics over time
- **Alerts**: Performance degradation notifications
- **Dashboards**: Real-time monitoring dashboards

### 🔄 Pipelines & Backend
- **Workflows**: End-to-end ML workflows as DAGs
- **Backend Engine**: FastAPI-based REST API with real-time WebSocket event streaming and SQLite persistence.
- **Visual Studio & Compiler**: Design pipelines visually and compile DAGs natively into executable Python code using Jinja2 templates.

### 🌐 Colab Bridge & Distributed Training
- **Remote GPU Execution**: Instantly delegate local `.py` and `.ipynb` execution to high-performance Google Colab runtimes via Jupyter kernel gateway.
- **Zero-Config Hardware Abstraction**: Our unified `Trainer` class automatically detects CUDA, MPS, and CPU, handling VRAM pre-flight checks to prevent OOM kernel crashes.
- **Elastic Distributed Training**: Native PyTorch DDP for Multi-GPU, Ray Train for Multi-Node clusters, and parallel Joblib for async sklearn bagging.
- **Ephemeral Storage Protection**: Auto-routes deep learning checkpoints to Google Drive when running in Colab.

## 🛠️ Installation

### Basic Installation
```bash
pip install adamops
```

### Development Installation
```bash
git clone https://github.com/adamops/adamops.git
cd adamops
pip install -e ".[dev]"
```

### Full Installation (all extras)
```bash
pip install adamops[all]
```

## 🚀 Quick Start

### Data Loading
```python
from adamops.data import loaders

# Load CSV with auto-encoding detection
df = loaders.load_csv("data.csv")

# Load from SQL database
df = loaders.load_sql("SELECT * FROM table", "sqlite:///database.db")
```

### Data Validation
```python
from adamops.data import validators

# Create validation report
report = validators.validate(df)
print(report.summary())
```

### Data Preprocessing
```python
from adamops.data import preprocessors

# Handle missing values
df = preprocessors.handle_missing(df, strategy="knn")

# Handle outliers
df = preprocessors.handle_outliers(df, method="iqr")
```

### Feature Engineering
```python
from adamops.data import feature_engineering

# Encode categorical variables
df = feature_engineering.encode(df, method="onehot", columns=["category"])

# Scale numerical features
df = feature_engineering.scale(df, method="standard", columns=["value"])
```

### Model Training
```python
from adamops.models import modelops

# Train a model
model = modelops.train(
    X_train, y_train,
    task="classification",
    algorithm="xgboost"
)

# Predict
predictions = model.predict(X_test)
```

### AutoML
```python
from adamops.models import automl

# Run AutoML
best_model = automl.run(
    X_train, y_train,
    task="classification",
    tuning="bayesian",
    time_limit=3600
)
```

### Evaluation
```python
from adamops.evaluation import metrics

# Compute metrics
results = metrics.evaluate(y_true, y_pred, task="classification")
print(results)
```

### Distributed & Colab GPU Training
```python
from adamops.trainer import Trainer

# Instantly auto-discovers CUDA, validates Colab VRAM, 
# and wraps in PyTorch DDP / Ray Train if requested.
trainer = Trainer(pytorch_model, strategy="auto", task="classification")
trainer.fit(X_train, y_train, epochs=20)

# Transparently routes outputs back to CPU memory
predictions = trainer.predict(X_test)

# Saves securely to Google Drive if in a Colab environment
trainer.save()
```

### CLI Usage
```bash
# Train a model
adamops train --data data.csv --target y --algorithm xgboost

# Evaluate a model
adamops evaluate --model model.pkl --data test.csv

# Deploy as API
adamops deploy --model model.pkl --type api --port 8000
```

## 📁 Project Structure

```
adamops/
├── adamops/
│   ├── __init__.py
│   ├── cli.py
│   ├── hardware.py
│   ├── distributed.py
│   ├── trainer.py
│   ├── backend/
│   │   ├── app.py
│   │   ├── database.py
│   │   └── runner.py
│   ├── colab/
│   │   ├── bridge.py
│   │   └── setup_snippet.py
│   ├── studio/
│   │   ├── compiler.py
│   │   └── server.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── validators.py
│   │   ├── preprocessors.py
│   │   ├── feature_engineering.py
│   │   └── splitters.py
│   ├── models/
│   │   ├── modelops.py
│   │   ├── registry.py
│   │   ├── ensembles.py
│   │   └── automl.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   ├── explainability.py
│   │   ├── comparison.py
│   │   └── reports.py
│   ├── deployment/
│   │   ├── exporters.py
│   │   ├── api.py
│   │   ├── containerize.py
│   │   └── cloud.py
│   ├── monitoring/
│   │   ├── drift.py
│   │   ├── performance.py
│   │   ├── alerts.py
│   │   └── dashboard.py
│   ├── pipelines/
│   │   ├── workflows.py
│   │   └── orchestrators.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── helpers.py
├── tests/
├── examples/
├── docs/
├── setup.py
├── requirements.txt
└── README.md
```

## 📚 Documentation

Full documentation is available at [https://adamops.readthedocs.io](https://adamops.readthedocs.io)

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn team for their excellent ML library
- XGBoost and LightGBM teams for gradient boosting implementations
- SHAP and LIME teams for explainability tools
- The entire open-source ML community

---

**Made with ❤️ by the AdamOps Team ,(Deepesh Jha , Abhishek ,Ayush Mishra,Mohd Mustafa)**
