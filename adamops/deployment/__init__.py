"""
AdamOps Deployment Module

Provides model deployment capabilities:
- exporters: Export models to ONNX, PMML, TFLite, CoreML
- api: Create FastAPI/Flask/Streamlit APIs
- containerize: Docker and Kubernetes deployment
- cloud: AWS, GCP, Azure deployment
- playground: Interactive Streamlit UI for model testing
"""

import importlib

def __getattr__(name):
    if name in ["exporters", "api", "containerize", "cloud", "playground"]:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return ["exporters", "api", "containerize", "cloud", "playground"]

__all__ = [
    "exporters",
    "api",
    "containerize",
    "cloud",
    "playground",
]
