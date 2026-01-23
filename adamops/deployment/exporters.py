"""
AdamOps Model Exporters Module

Export models to ONNX, PMML, and other formats.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import joblib
import pickle

from adamops.utils.logging import get_logger
from adamops.utils.helpers import ensure_dir

logger = get_logger(__name__)


def export_pickle(model: Any, filepath: str) -> str:
    """Export model as pickle file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Exported model to {filepath}")
    return str(filepath)


def export_joblib(model: Any, filepath: str, compress: int = 3) -> str:
    """Export model using joblib (better for large arrays)."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    joblib.dump(model, filepath, compress=compress)
    logger.info(f"Exported model to {filepath}")
    return str(filepath)


def export_onnx(
    model: Any, filepath: str, 
    initial_types: Optional[List[Tuple[str, Any]]] = None,
    n_features: Optional[int] = None
) -> str:
    """
    Export sklearn model to ONNX format.
    
    Args:
        model: Sklearn model.
        filepath: Output path.
        initial_types: Input type specification.
        n_features: Number of input features.
    """
    try:
        import onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError("onnx and skl2onnx required. Install with: pip install onnx skl2onnx")
    
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    if initial_types is None:
        if n_features is None:
            raise ValueError("Either initial_types or n_features must be provided")
        initial_types = [('input', FloatTensorType([None, n_features]))]
    
    onnx_model = convert_sklearn(model, initial_types=initial_types)
    
    with open(filepath, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    logger.info(f"Exported ONNX model to {filepath}")
    return str(filepath)


def export_pmml(model: Any, filepath: str, feature_names: Optional[List[str]] = None) -> str:
    """Export model to PMML format."""
    try:
        from sklearn2pmml import sklearn2pmml
        from sklearn2pmml.pipeline import PMMLPipeline
    except ImportError:
        raise ImportError("sklearn2pmml required. Install with: pip install sklearn2pmml")
    
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    pipeline = PMMLPipeline([("model", model)])
    sklearn2pmml(pipeline, str(filepath))
    
    logger.info(f"Exported PMML model to {filepath}")
    return str(filepath)


def load_model(filepath: str, format: str = "auto") -> Any:
    """
    Load a saved model.
    
    Args:
        filepath: Model file path.
        format: 'pickle', 'joblib', 'onnx', or 'auto'.
    """
    filepath = Path(filepath)
    
    if format == "auto":
        suffix = filepath.suffix.lower()
        if suffix in ['.pkl', '.pickle']:
            format = "pickle"
        elif suffix in ['.joblib']:
            format = "joblib"
        elif suffix == '.onnx':
            format = "onnx"
        else:
            format = "joblib"  # Default
    
    if format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == "joblib":
        return joblib.load(filepath)
    elif format == "onnx":
        import onnxruntime as ort
        return ort.InferenceSession(str(filepath))
    else:
        raise ValueError(f"Unknown format: {format}")


def export(model: Any, filepath: str, format: str = "joblib", **kwargs) -> str:
    """
    Export model to specified format.
    
    Args:
        model: Model to export.
        filepath: Output path.
        format: 'pickle', 'joblib', 'onnx', 'pmml'.
    """
    exporters = {
        "pickle": export_pickle,
        "joblib": export_joblib,
        "onnx": export_onnx,
        "pmml": export_pmml,
    }
    
    if format not in exporters:
        raise ValueError(f"Unknown format: {format}. Available: {list(exporters.keys())}")
    
    return exporters[format](model, filepath, **kwargs)
