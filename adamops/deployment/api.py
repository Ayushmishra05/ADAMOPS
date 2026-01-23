"""
AdamOps API Module

Create REST APIs for model serving with FastAPI/Flask.
"""

from typing import Any, Callable, Dict, List, Optional
import json
from pathlib import Path

from adamops.utils.logging import get_logger
from adamops.deployment.exporters import load_model

logger = get_logger(__name__)


def create_fastapi_app(
    model: Any, model_name: str = "model",
    input_schema: Optional[Dict] = None,
    preprocess_fn: Optional[Callable] = None,
    postprocess_fn: Optional[Callable] = None
):
    """
    Create FastAPI application for model serving.
    
    Args:
        model: Trained model.
        model_name: Name for the model.
        input_schema: Pydantic model or dict for input validation.
        preprocess_fn: Function to preprocess input.
        postprocess_fn: Function to postprocess output.
    
    Returns:
        FastAPI application.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, create_model
    except ImportError:
        raise ImportError("FastAPI required. Install with: pip install fastapi uvicorn")
    
    import numpy as np
    
    app = FastAPI(title=f"{model_name} API", version="1.0.0")
    
    # Create input model
    class PredictionInput(BaseModel):
        features: List[List[float]]
    
    class PredictionOutput(BaseModel):
        predictions: List
        model_name: str
    
    @app.get("/")
    def root():
        return {"message": f"Welcome to {model_name} API", "status": "healthy"}
    
    @app.get("/health")
    def health():
        return {"status": "healthy", "model": model_name}
    
    @app.post("/predict", response_model=PredictionOutput)
    def predict(input_data: PredictionInput):
        try:
            features = np.array(input_data.features)
            
            if preprocess_fn:
                features = preprocess_fn(features)
            
            predictions = model.predict(features).tolist()
            
            if postprocess_fn:
                predictions = postprocess_fn(predictions)
            
            return PredictionOutput(predictions=predictions, model_name=model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict_proba")
    def predict_proba(input_data: PredictionInput):
        if not hasattr(model, 'predict_proba'):
            raise HTTPException(status_code=400, detail="Model does not support probability predictions")
        
        try:
            features = np.array(input_data.features)
            probas = model.predict_proba(features).tolist()
            return {"probabilities": probas, "model_name": model_name}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def create_flask_app(
    model: Any, model_name: str = "model",
    preprocess_fn: Optional[Callable] = None,
    postprocess_fn: Optional[Callable] = None
):
    """Create Flask application for model serving."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError("Flask required. Install with: pip install flask")
    
    import numpy as np
    
    app = Flask(model_name)
    
    @app.route("/")
    def root():
        return jsonify({"message": f"Welcome to {model_name} API", "status": "healthy"})
    
    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "model": model_name})
    
    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json()
            features = np.array(data["features"])
            
            if preprocess_fn:
                features = preprocess_fn(features)
            
            predictions = model.predict(features).tolist()
            
            if postprocess_fn:
                predictions = postprocess_fn(predictions)
            
            return jsonify({"predictions": predictions, "model_name": model_name})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app


def run_api(
    model: Any, framework: str = "fastapi",
    host: str = "0.0.0.0", port: int = 8000, **kwargs
):
    """
    Run model serving API.
    
    Args:
        model: Trained model.
        framework: 'fastapi' or 'flask'.
        host: Host address.
        port: Port number.
    """
    if framework == "fastapi":
        import uvicorn
        app = create_fastapi_app(model, **kwargs)
        uvicorn.run(app, host=host, port=port)
    elif framework == "flask":
        app = create_flask_app(model, **kwargs)
        app.run(host=host, port=port)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def generate_api_code(
    model_path: str, output_path: str, framework: str = "fastapi", model_name: str = "model"
) -> str:
    """
    Generate standalone API code.
    
    Args:
        model_path: Path to saved model.
        output_path: Output file path.
        framework: 'fastapi' or 'flask'.
        model_name: Name for the model.
    
    Returns:
        Path to generated file.
    """
    if framework == "fastapi":
        code = f'''"""Auto-generated FastAPI model serving code."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

app = FastAPI(title="{model_name} API")
model = joblib.load("{model_path}")

class PredictionInput(BaseModel):
    features: List[List[float]]

@app.get("/")
def root():
    return {{"message": "Welcome to {model_name} API"}}

@app.get("/health")
def health():
    return {{"status": "healthy"}}

@app.post("/predict")
def predict(input_data: PredictionInput):
    features = np.array(input_data.features)
    predictions = model.predict(features).tolist()
    return {{"predictions": predictions}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    else:
        code = f'''"""Auto-generated Flask model serving code."""
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("{model_path}")

@app.route("/")
def root():
    return jsonify({{"message": "Welcome to {model_name} API"}})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"])
    predictions = model.predict(features).tolist()
    return jsonify({{"predictions": predictions}})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
'''
    
    with open(output_path, 'w') as f:
        f.write(code)
    
    logger.info(f"Generated API code at {output_path}")
    return output_path
