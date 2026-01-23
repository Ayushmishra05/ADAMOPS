"""
AdamOps Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="adamops",
    version="0.1.0",
    author="AdamOps Team",
    author_email="adamops@example.com",
    description="A comprehensive MLOps library for end-to-end machine learning workflows",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/adamops/adamops",
    project_urls={
        "Bug Tracker": "https://github.com/adamops/adamops/issues",
        "Documentation": "https://adamops.readthedocs.io",
        "Source Code": "https://github.com/adamops/adamops",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "joblib>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "xgboost>=1.7.0",
        "lightgbm>=3.3.0",
        "pyyaml>=6.0.0",
        "click>=8.1.0",
        "rich>=13.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "flask>=2.3.0",
            "streamlit>=1.25.0",
        ],
        "automl": [
            "optuna>=3.0.0",
        ],
        "explainability": [
            "shap>=0.41.0",
            "lime>=0.2.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "skl2onnx>=1.15.0",
        ],
        "tracking": [
            "mlflow>=2.5.0",
        ],
        "all": [
            "optuna>=3.0.0",
            "shap>=0.41.0",
            "lime>=0.2.0",
            "onnx>=1.14.0",
            "skl2onnx>=1.15.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "flask>=2.3.0",
            "streamlit>=1.25.0",
            "mlflow>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adamops=adamops.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "mlops",
        "machine-learning",
        "data-science",
        "automl",
        "model-deployment",
        "model-monitoring",
        "feature-engineering",
    ],
)
