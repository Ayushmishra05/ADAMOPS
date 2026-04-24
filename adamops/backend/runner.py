"""
AdamOps Backend — Background Job Runner

Wraps adamops training and AutoML inside a background thread,
emitting progress events for WebSocket fan-out and persisting
results to the StateDB.
"""

import threading
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from adamops.backend.database import StateDB
from adamops.utils.logging import get_logger

logger = get_logger(__name__)


class RunEvent:
    """Lightweight event emitted during a run."""

    __slots__ = ("run_id", "event_type", "payload")

    def __init__(self, run_id: int, event_type: str, payload: Optional[Dict] = None):
        self.run_id = run_id
        self.event_type = event_type
        self.payload = payload or {}


class EventBus:
    """
    Simple in-process pub/sub for run events.
    WebSocket handlers subscribe; the runner publishes.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # run_id -> list of callback(RunEvent)
        self._subscribers: Dict[int, List[Callable]] = {}

    def subscribe(self, run_id: int, callback: Callable):
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(callback)

    def unsubscribe(self, run_id: int, callback: Callable):
        with self._lock:
            subs = self._subscribers.get(run_id, [])
            if callback in subs:
                subs.remove(callback)
            if not subs:
                self._subscribers.pop(run_id, None)

    def publish(self, event: RunEvent):
        with self._lock:
            subs = list(self._subscribers.get(event.run_id, []))
        for cb in subs:
            try:
                cb(event)
            except Exception:
                logger.debug(f"Subscriber error for run {event.run_id}", exc_info=True)


# Global singleton event bus
event_bus = EventBus()


def _emit(db: StateDB, run_id: int, event_type: str, payload: Dict = None):
    """Persist event to DB and publish to subscribers."""
    payload = payload or {}
    db.log_event(run_id, event_type, payload)
    event_bus.publish(RunEvent(run_id, event_type, payload))


def _execute_run(
    db: StateDB,
    run_id: int,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    algorithm: str,
    backend: str,
    params: Dict,
):
    """
    The core function executed in a background thread.
    Updates run status, trains model, and emits events.
    """
    try:
        # Mark running
        db.update_run_status(run_id, "running")
        _emit(db, run_id, "status_change", {"status": "running"})
        _emit(db, run_id, "log", {"message": f"Run {run_id} started: task={task}, algo={algorithm}, backend={backend}"})

        if algorithm == "automl":
            # Use the AutoML engine
            _emit(db, run_id, "log", {"message": "Starting AutoML search..."})
            from adamops.models.automl import run as run_automl

            result = run_automl(
                X, y,
                task=task,
                backend=backend,
                n_trials=params.get("n_trials", 10),
                cv=params.get("cv", 3),
            )

            metrics = {
                "best_score": float(result.best_score),
                "best_params": result.best_params,
                "leaderboard_size": len(result.leaderboard),
            }
            _emit(db, run_id, "metrics", metrics)
            db.update_run_metrics(run_id, metrics)

        else:
            # Single algorithm training
            _emit(db, run_id, "log", {"message": f"Training {algorithm}..."})
            from adamops.models.modelops import train

            model = train(X, y, task=task, algorithm=algorithm)

            # Compute a basic score
            preds = model.predict(X)
            if task == "classification":
                from sklearn.metrics import accuracy_score
                score = float(accuracy_score(y, preds))
                metrics = {"accuracy": score}
            else:
                from sklearn.metrics import r2_score
                score = float(r2_score(y, preds))
                metrics = {"r2_score": score}

            _emit(db, run_id, "metrics", metrics)
            db.update_run_metrics(run_id, metrics)

        # Mark completed
        db.update_run_status(run_id, "completed")
        _emit(db, run_id, "status_change", {"status": "completed"})
        _emit(db, run_id, "log", {"message": f"Run {run_id} completed successfully."})

    except Exception as exc:
        err_msg = traceback.format_exc()
        logger.error(f"Run {run_id} failed: {err_msg}")
        db.update_run_status(run_id, "failed", error=str(exc))
        _emit(db, run_id, "status_change", {"status": "failed", "error": str(exc)})
        _emit(db, run_id, "log", {"message": f"Run {run_id} failed: {exc}"})


def submit_run(
    db: StateDB,
    run_id: int,
    X: np.ndarray,
    y: np.ndarray,
    task: str = "auto",
    algorithm: str = "random_forest",
    backend: str = "local",
    params: Optional[Dict] = None,
) -> threading.Thread:
    """
    Submit a training run to a background thread.

    Returns the Thread object (caller can join if needed).
    """
    t = threading.Thread(
        target=_execute_run,
        args=(db, run_id, X, y, task, algorithm, backend, params or {}),
        daemon=True,
        name=f"adamops-run-{run_id}",
    )
    t.start()
    logger.info(f"Submitted run {run_id} to background thread")
    return t
