Welcome to AdamOps!
===================

AdamOps is a comprehensive MLOps library designed to streamline end-to-end machine learning workflows. It provides a modular, extensible, and user-friendly interface for everything from data preprocessing to model deployment and monitoring.

Whether you prefer writing code or using a visual interface, AdamOps gives you the tools you need to build, test, and ship models faster.

Key Features
------------

* **Data Processing**: Robust loaders, splitters, validators, and feature engineering.
* **Model Training & AutoML**: Unified interface for Scikit-learn, XGBoost, LightGBM, plus Optuna integration.
* **Evaluation**: Detailed metrics, comparison reports, and explainability natively built-in (SHAP/LIME).
* **Model Playground**: Instantly launch an interactive Streamlit dashboard (`adamops.deployment.playground`) to test your models and explore data.
* **Visual Studio**: A powerful, browser-based drag-and-drop pipeline builder (`adamops.studio`).
* **Deployment & Monitoring**: Export to ONNX, build FastAPI/Flask APIs, containerize with Docker, and track drift.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
