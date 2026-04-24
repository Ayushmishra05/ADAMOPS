"""
AdamOps Trainer Module

Provides a unified, high-level wrapper to train models intelligently across
varying architectures and hardware setups (Single CPU, Single GPU, Multi-GPU DDP, Ray, etc).
Silently auto-discovers optimal devices, validates VRAM constraints, handles Colab
telemetry securely, and maps PyTorch/Scikit-learn executions together.
"""

import time
import os
import joblib
from typing import Any, Dict, Optional, Union
import numpy as np

try:
    import pandas as pd
except ImportError:
    pass

from adamops.utils.logging import get_logger, Timer
from adamops.hardware import DeviceManager
from adamops.distributed import DistributedManager
from adamops.models.modelops import train as sklearn_train, AdamOpsModel
from adamops.evaluation.metrics import evaluate

try:
    from tqdm.notebook import tqdm as tqdm_colab
    from tqdm import tqdm as tqdm_std
except ImportError:
    tqdm_colab = None
    tqdm_std = None

logger = get_logger("adamops.trainer")


class Trainer:
    """
    Unified training orchestrator for AdamOps.
    
    Abstracts hardware selection.
    Automatically guards against Colab OOM issues and output flooding.
    """
    
    def __init__(self, model_or_algorithm: Any, task: str = "classification", strategy: str = "auto", backend: str = "auto"):
        """
        Initializes the trainer.
        
        Args:
            model_or_algorithm: Either a string (e.g., 'random_forest') for sklearn, or a raw PyTorch nn.Module.
            task: 'classification' or 'regression'.
            strategy: 'auto', 'single', 'ddp', or 'ray'.
            backend: 'auto', 'sklearn', or 'pytorch'.
        """
        self.task = task
        
        # 1. Hardware Detection
        self.device = DeviceManager.get_optimal_device()
        self.is_colab = DeviceManager.is_colab_environment()
        
        # 2. Determine backend type automatically
        if isinstance(model_or_algorithm, str):
            self.backend = "sklearn"
            self.algorithm = model_or_algorithm
            self.raw_model = None
        else:
            self.backend = "pytorch"
            self.raw_model = model_or_algorithm
            self.algorithm = "custom_nn"
            
        if backend != "auto":
            self.backend = backend
            
        # 3. Strategy Selection
        if strategy == "auto":
            self.strategy = DistributedManager.auto_strategy(self.backend)
        else:
            self.strategy = strategy

        logger.info(f"Initialized AdamOps Trainer: backend={self.backend}, strategy={self.strategy}, device={self.device}, colab={self.is_colab}")
        
        self.trained_model = None
        self.training_history = []

    def _get_progress_bar(self, iterable, desc="Training"):
        """Returns the appropriate progress bar preventing Colab browser freezes."""
        if self.is_colab and tqdm_colab:
            return tqdm_colab(iterable, desc=desc)
        elif self.is_colab:
            # Fallback if ipywidgets is missing but we're in colab, print sparsely
            logger.info("tqdm.notebook unavailable. Proceeding with silent Colab logging to prevent browser freezing.")
            return iterable
        elif tqdm_std:
            return tqdm_std(iterable, desc=desc)
        return iterable

    def fit(self, X: Union['pd.DataFrame', np.ndarray, Any], y: Union['pd.Series', np.ndarray, Any], epochs: int = 10, batch_size: int = 32) -> 'Trainer':
        """
        Trains the managed model by silently formatting and moving the data and
        model weights into the safest compatible format based on underlying architecture.
        """
        with Timer(f"Trainer.fit() [Backend: {self.backend}, Strategy: {self.strategy}]", logger):
            if self.backend == "sklearn":
                return self._fit_sklearn(X, y)
            elif self.backend == "pytorch":
                return self._fit_pytorch(X, y, epochs, batch_size)
            else:
                raise ValueError(f"Unknown backend type: {self.backend}")
                
    def _fit_sklearn(self, X, y):
        """Standard CPU-bound execution. Could dispatch via joblib if strategy=joblib."""
        logger.info(f"Executing Scikit-Learn training sequence for {self.algorithm}")
        # Underlying modelops already wraps the internal data correctly.
        self.trained_model = sklearn_train(X, y, task=self.task, algorithm=self.algorithm)
        return self

    def _fit_pytorch(self, X, y, epochs, batch_size):
        """Silently maps torch operations safely using pre-flight checks."""
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            logger.error("PyTorch must be installed: `pip install torch`")
            return self

        # Pre-flight hardware bounds check to prevent silent notebook crashes in Colab
        model_vram_req_mb = DeviceManager.estimate_model_vram(self.raw_model)
        
        # If the model is absurdly huge (1000MB assumes batch 32 scaling generically, a safe threshold is total * 1.5)
        safe_boundary_mb = model_vram_req_mb * 1.5
        
        if not DeviceManager.vram_preflight_check(safe_boundary_mb):
            logger.error("Aborting PyTorch fit to preserve kernel state.")
            raise MemoryError("AdamOps VRAM pre-flight hook intercepted a fatal model allocation size.")

        # Tensorize
        if not isinstance(X, torch.Tensor):
            if hasattr(X, 'values'):
                X = X.values
            X_t = torch.tensor(X, dtype=torch.float32)
        else:
            X_t = X
            
        if not isinstance(y, torch.Tensor):
            if hasattr(y, 'values'):
                y = y.values
            if self.task == "classification":
                y_t = torch.tensor(y, dtype=torch.long)
            else:
                y_t = torch.tensor(y, dtype=torch.float32)
        else:
            y_t = y

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Move Model silently
        model = self.raw_model.to(self.device)
        
        # If DDP requested, wrap
        if self.strategy == "ddp":
            model = DistributedManager.wrap_ddp(model, self.device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss() if self.task == "classification" else torch.nn.MSELoss()
        
        model.train()
        
        epoch_iterator = self._get_progress_bar(range(epochs), desc=f"Training via {self.strategy.upper()}")
        
        for epoch in epoch_iterator:
            epoch_loss = 0.0
            
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                
                # Squeeze regression targets
                if self.task == "regression":
                    outputs = outputs.view(-1)
                    batch_y = batch_y.view(-1)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            self.training_history.append(avg_loss)
            
            # Rate limit stdout on tqdm specifically for colab environments manually if needed
            # Actually tqdm.notebook handles this implicitly so we trust the wrapper.
            
        if self.strategy == "ddp":
            self.trained_model = model.module
        else:
            self.trained_model = model
            
        return self

    def predict(self, X: Any) -> Any:
        """Runs predictions securely routing hardware outputs back to CPU gracefully."""
        if self.backend == "sklearn":
            return self.trained_model.predict(X)
        
        try:
            import torch
        except ImportError:
            pass
            
        self.trained_model.eval()
        if not isinstance(X, torch.Tensor):
            if hasattr(X, 'values'):
                X = X.values
            X_t = torch.tensor(X, dtype=torch.float32)
        else:
            X_t = X

        X_t = X_t.to(self.device)
        with torch.no_grad():
            preds = self.trained_model(X_t)
            
        if self.task == "classification":
            preds = torch.argmax(preds, dim=1)
            
        return preds.cpu().numpy()
        
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Convenience method combining generation and scoring natively."""
        preds = self.predict(X_test)
        if hasattr(y_test, "values"):
            y_test = y_test.values
        return evaluate(y_test, preds)

    def save(self, run_name: str = "run_001"):
        """
        Saves the model using dynamic pathing aware of ephemeral environments.
        Colab runtimes will automatically fallback to Drive paths if mounted 
        guaranteeing safe checkpointing before disconnecting.
        """
        base_dir = DeviceManager.get_checkpoint_dir()
        path = os.path.join(base_dir, f"{run_name}_{self.algorithm}.pt")
        
        if self.backend == "sklearn":
            path = path.replace(".pt", ".joblib")
            joblib.dump(self.trained_model, path)
        else:
            try:
                import torch
                torch.save(self.trained_model.state_dict(), path)
            except ImportError:
                pass
                
        logger.info(f"Model checkpoint routed successfully to: {path}")
        return path
