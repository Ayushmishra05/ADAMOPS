"""
AdamOps Studio — Node Registry

Defines all available node types that map to AdamOps functions.
Each node has typed input/output ports and configurable parameters.
"""

from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import numpy as np


class Port:
    """Defines an input or output port on a node."""
    def __init__(self, name: str, dtype: str, label: str = ""):
        self.name = name
        self.dtype = dtype  # "dataframe", "series", "model", "metrics", "splits"
        self.label = label or name


class Param:
    """Defines a configurable parameter on a node."""
    def __init__(self, name: str, dtype: str, default: Any = None,
                 options: Optional[List] = None, label: str = "", required: bool = False):
        self.name = name
        self.dtype = dtype  # "string", "number", "select", "boolean", "file"
        self.default = default
        self.options = options
        self.label = label or name
        self.required = required

    def to_dict(self):
        d = {
            "name": self.name,
            "dtype": self.dtype,
            "default": self.default,
            "label": self.label,
            "required": self.required,
        }
        if self.options:
            d["options"] = self.options
        return d


class NodeType:
    """Defines a type of node available in the studio."""
    def __init__(self, id: str, label: str, category: str,
                 inputs: List[Port], outputs: List[Port],
                 params: List[Param], execute_fn: Callable,
                 description: str = ""):
        self.id = id
        self.label = label
        self.category = category
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.execute_fn = execute_fn
        self.description = description

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "inputs": [{"name": p.name, "dtype": p.dtype, "label": p.label} for p in self.inputs],
            "outputs": [{"name": p.name, "dtype": p.dtype, "label": p.label} for p in self.outputs],
            "params": [p.to_dict() for p in self.params],
        }


# =============================================================================
# Node execution functions
# =============================================================================

def _exec_load_csv(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.loaders import load_csv
    filepath = params.get("filepath", "")
    if not filepath:
        raise ValueError("File path is required")
    df = load_csv(filepath)
    return {"dataframe": df}


def _exec_load_excel(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.loaders import load_excel
    filepath = params.get("filepath", "")
    sheet = params.get("sheet_name", 0)
    if not filepath:
        raise ValueError("File path is required")
    df = load_excel(filepath, sheet_name=sheet)
    return {"dataframe": df}


def _exec_load_json(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.loaders import load_json
    filepath = params.get("filepath", "")
    if not filepath:
        raise ValueError("File path is required")
    df = load_json(filepath)
    return {"dataframe": df}


def _exec_handle_missing(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.preprocessors import handle_missing
    df = inputs["dataframe"].copy()
    strategy = params.get("strategy", "mean")
    df = handle_missing(df, strategy=strategy)
    return {"dataframe": df}


def _exec_handle_outliers(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.preprocessors import handle_outliers
    df = inputs["dataframe"].copy()
    method = params.get("method", "iqr")
    threshold = float(params.get("threshold", 1.5))
    df = handle_outliers(df, method=method, threshold=threshold)
    return {"dataframe": df}


def _exec_handle_duplicates(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.preprocessors import handle_duplicates
    df = inputs["dataframe"].copy()
    df = handle_duplicates(df)
    return {"dataframe": df}


def _exec_encode(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.feature_engineering import encode
    df = inputs["dataframe"].copy()
    method = params.get("method", "onehot")
    columns_str = params.get("columns", "")
    columns = [c.strip() for c in columns_str.split(",") if c.strip()] if columns_str else None
    if columns:
        df = encode(df, columns=columns, method=method)
    else:
        # Auto-detect categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            df = encode(df, columns=cat_cols, method=method)
    return {"dataframe": df}


def _exec_scale(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.feature_engineering import scale
    df = inputs["dataframe"].copy()
    method = params.get("method", "standard")
    columns_str = params.get("columns", "")
    columns = [c.strip() for c in columns_str.split(",") if c.strip()] if columns_str else None
    df = scale(df, method=method, columns=columns)
    return {"dataframe": df}


def _exec_select_features(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.feature_engineering import select_features
    df = inputs["dataframe"].copy()
    target = params.get("target", "")
    method = params.get("method", "importance")
    n_features = int(params.get("n_features", 10))
    if not target:
        raise ValueError("Target column is required")
    df = select_features(df, target=target, method=method, n_features=n_features)
    return {"dataframe": df}


def _exec_train_test_split(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.splitters import split_train_test
    df = inputs["dataframe"]
    target = params.get("target", "")
    test_size = float(params.get("test_size", 0.2))
    if not target:
        raise ValueError("Target column is required")
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test
    }


def _exec_train_val_test_split(inputs: Dict, params: Dict) -> Dict:
    from adamops.data.splitters import split_train_val_test
    df = inputs["dataframe"]
    target = params.get("target", "")
    train_size = float(params.get("train_size", 0.7))
    val_size = float(params.get("val_size", 0.15))
    test_size = float(params.get("test_size", 0.15))
    if not target:
        raise ValueError("Target column is required")
    X = df.drop(columns=[target])
    y = df[target]
    result = split_train_val_test(X, y, train_size=train_size, val_size=val_size, test_size=test_size)
    return {
        "X_train": result[0], "X_test": result[2],
        "y_train": result[3], "y_test": result[5]
    }


def _exec_train_classification(inputs: Dict, params: Dict) -> Dict:
    from adamops.models.modelops import train
    X_train = inputs["X_train"]
    y_train = inputs["y_train"]
    algorithm = params.get("algorithm", "random_forest")
    model = train(X_train, y_train, task="classification", algorithm=algorithm)
    return {"model": model}


def _exec_train_regression(inputs: Dict, params: Dict) -> Dict:
    from adamops.models.modelops import train
    X_train = inputs["X_train"]
    y_train = inputs["y_train"]
    algorithm = params.get("algorithm", "ridge")
    model = train(X_train, y_train, task="regression", algorithm=algorithm)
    return {"model": model}


def _exec_automl(inputs: Dict, params: Dict) -> Dict:
    from adamops.models.automl import run as run_automl
    X_train = inputs["X_train"]
    y_train = inputs["y_train"]
    task = params.get("task", "classification")
    tuning = params.get("tuning", "none")
    result = run_automl(X_train, y_train, task=task, tuning=tuning, n_trials=10)
    return {"model": result.best_model}


def _exec_evaluate(inputs: Dict, params: Dict) -> Dict:
    from adamops.evaluation.metrics import evaluate
    model = inputs["model"]
    X_test = inputs["X_test"]
    y_test = inputs["y_test"]
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    return {"metrics": metrics}


def _exec_cross_validate(inputs: Dict, params: Dict) -> Dict:
    from adamops.models.modelops import cross_validate
    X_train = inputs["X_train"]
    y_train = inputs["y_train"]
    task = params.get("task", "classification")
    algorithm = params.get("algorithm", "random_forest")
    cv = int(params.get("cv", 5))
    result = cross_validate(X_train, y_train, task=task, algorithm=algorithm, cv=cv)
    return {"metrics": result}


# =============================================================================
# Node Type Registry
# =============================================================================

NODE_TYPES: Dict[str, NodeType] = {}


def _register(node_type: NodeType):
    NODE_TYPES[node_type.id] = node_type


# --- Data Nodes ---

_register(NodeType(
    id="load_csv", label="Load CSV", category="data",
    description="Load data from a CSV file",
    inputs=[],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[Param("filepath", "file", label="File Path", required=True)],
    execute_fn=_exec_load_csv,
))

_register(NodeType(
    id="load_excel", label="Load Excel", category="data",
    description="Load data from an Excel file",
    inputs=[],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[
        Param("filepath", "file", label="File Path", required=True),
        Param("sheet_name", "string", default="0", label="Sheet Name"),
    ],
    execute_fn=_exec_load_excel,
))

_register(NodeType(
    id="load_json", label="Load JSON", category="data",
    description="Load data from a JSON file",
    inputs=[],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[Param("filepath", "file", label="File Path", required=True)],
    execute_fn=_exec_load_json,
))

# --- Preprocessing Nodes ---

_register(NodeType(
    id="handle_missing", label="Handle Missing", category="preprocessing",
    description="Handle missing values in the dataset",
    inputs=[Port("dataframe", "dataframe", "DataFrame")],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[Param("strategy", "select", default="mean", label="Strategy",
                  options=["drop", "mean", "median", "mode", "ffill", "bfill", "knn"])],
    execute_fn=_exec_handle_missing,
))

_register(NodeType(
    id="handle_outliers", label="Handle Outliers", category="preprocessing",
    description="Detect and handle outliers",
    inputs=[Port("dataframe", "dataframe", "DataFrame")],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[
        Param("method", "select", default="iqr", label="Method", options=["iqr", "zscore", "isolation_forest"]),
        Param("threshold", "number", default=1.5, label="Threshold"),
    ],
    execute_fn=_exec_handle_outliers,
))

_register(NodeType(
    id="handle_duplicates", label="Handle Duplicates", category="preprocessing",
    description="Remove duplicate rows",
    inputs=[Port("dataframe", "dataframe", "DataFrame")],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[],
    execute_fn=_exec_handle_duplicates,
))

# --- Feature Engineering Nodes ---

_register(NodeType(
    id="encode", label="Encode Features", category="feature_engineering",
    description="Encode categorical columns",
    inputs=[Port("dataframe", "dataframe", "DataFrame")],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[
        Param("method", "select", default="onehot", label="Method", options=["onehot", "label", "ordinal"]),
        Param("columns", "string", default="", label="Columns (comma-sep, empty=auto)"),
    ],
    execute_fn=_exec_encode,
))

_register(NodeType(
    id="scale", label="Scale Features", category="feature_engineering",
    description="Scale numerical columns",
    inputs=[Port("dataframe", "dataframe", "DataFrame")],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[
        Param("method", "select", default="standard", label="Method", options=["standard", "minmax", "robust"]),
        Param("columns", "string", default="", label="Columns (comma-sep, empty=all numeric)"),
    ],
    execute_fn=_exec_scale,
))

_register(NodeType(
    id="select_features", label="Select Features", category="feature_engineering",
    description="Select top features by importance",
    inputs=[Port("dataframe", "dataframe", "DataFrame")],
    outputs=[Port("dataframe", "dataframe", "DataFrame")],
    params=[
        Param("target", "string", label="Target Column", required=True),
        Param("method", "select", default="importance", label="Method", options=["importance", "variance", "correlation"]),
        Param("n_features", "number", default=10, label="N Features"),
    ],
    execute_fn=_exec_select_features,
))

# --- Splitting Nodes ---

_register(NodeType(
    id="train_test_split", label="Train/Test Split", category="splitting",
    description="Split data into train and test sets",
    inputs=[Port("dataframe", "dataframe", "DataFrame")],
    outputs=[
        Port("X_train", "dataframe", "X Train"),
        Port("X_test", "dataframe", "X Test"),
        Port("y_train", "series", "y Train"),
        Port("y_test", "series", "y Test"),
    ],
    params=[
        Param("target", "string", label="Target Column", required=True),
        Param("test_size", "number", default=0.2, label="Test Size"),
    ],
    execute_fn=_exec_train_test_split,
))

_register(NodeType(
    id="train_val_test_split", label="Train/Val/Test Split", category="splitting",
    description="Split data into train, validation, and test sets",
    inputs=[Port("dataframe", "dataframe", "DataFrame")],
    outputs=[
        Port("X_train", "dataframe", "X Train"),
        Port("X_test", "dataframe", "X Test"),
        Port("y_train", "series", "y Train"),
        Port("y_test", "series", "y Test"),
    ],
    params=[
        Param("target", "string", label="Target Column", required=True),
        Param("train_size", "number", default=0.7, label="Train Size"),
        Param("val_size", "number", default=0.15, label="Val Size"),
        Param("test_size", "number", default=0.15, label="Test Size"),
    ],
    execute_fn=_exec_train_val_test_split,
))

# --- Model Nodes ---

_register(NodeType(
    id="train_classification", label="Train Classifier", category="models",
    description="Train a classification model",
    inputs=[
        Port("X_train", "dataframe", "X Train"),
        Port("y_train", "series", "y Train"),
    ],
    outputs=[Port("model", "model", "Model")],
    params=[Param("algorithm", "select", default="random_forest", label="Algorithm",
                  options=["random_forest", "logistic", "decision_tree", "gradient_boosting",
                           "xgboost", "lightgbm", "knn", "naive_bayes"])],
    execute_fn=_exec_train_classification,
))

_register(NodeType(
    id="train_regression", label="Train Regressor", category="models",
    description="Train a regression model",
    inputs=[
        Port("X_train", "dataframe", "X Train"),
        Port("y_train", "series", "y Train"),
    ],
    outputs=[Port("model", "model", "Model")],
    params=[Param("algorithm", "select", default="ridge", label="Algorithm",
                  options=["ridge", "lasso", "elasticnet", "decision_tree",
                           "random_forest", "gradient_boosting", "xgboost", "lightgbm", "knn"])],
    execute_fn=_exec_train_regression,
))

_register(NodeType(
    id="automl", label="AutoML", category="models",
    description="Automatic model selection and tuning",
    inputs=[
        Port("X_train", "dataframe", "X Train"),
        Port("y_train", "series", "y Train"),
    ],
    outputs=[Port("model", "model", "Best Model")],
    params=[
        Param("task", "select", default="classification", label="Task", options=["classification", "regression"]),
        Param("tuning", "select", default="none", label="Tuning", options=["none", "grid", "random"]),
    ],
    execute_fn=_exec_automl,
))

# --- Evaluation Nodes ---

_register(NodeType(
    id="evaluate", label="Evaluate Model", category="evaluation",
    description="Compute evaluation metrics",
    inputs=[
        Port("model", "model", "Model"),
        Port("X_test", "dataframe", "X Test"),
        Port("y_test", "series", "y Test"),
    ],
    outputs=[Port("metrics", "metrics", "Metrics")],
    params=[],
    execute_fn=_exec_evaluate,
))

_register(NodeType(
    id="cross_validate", label="Cross Validate", category="evaluation",
    description="Run cross-validation",
    inputs=[
        Port("X_train", "dataframe", "X Train"),
        Port("y_train", "series", "y Train"),
    ],
    outputs=[Port("metrics", "metrics", "CV Results")],
    params=[
        Param("task", "select", default="classification", label="Task", options=["classification", "regression"]),
        Param("algorithm", "select", default="random_forest", label="Algorithm",
              options=["random_forest", "logistic", "ridge", "decision_tree", "gradient_boosting"]),
        Param("cv", "number", default=5, label="Folds"),
    ],
    execute_fn=_exec_cross_validate,
))


def get_all_node_types() -> List[Dict]:
    """Return all node types as dicts for the frontend."""
    return [nt.to_dict() for nt in NODE_TYPES.values()]


def get_node_type(node_id: str) -> Optional[NodeType]:
    """Get a node type by ID."""
    return NODE_TYPES.get(node_id)
