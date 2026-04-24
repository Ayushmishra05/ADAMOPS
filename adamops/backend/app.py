"""
AdamOps Backend — FastAPI Application

REST endpoints for experiment/run/model management and a
WebSocket endpoint for real-time run progress streaming.
"""

import asyncio
import json
from typing import Dict, List, Optional

from adamops.backend.database import StateDB
from adamops.backend.runner import EventBus, RunEvent, event_bus, submit_run
from adamops.utils.logging import get_logger

logger = get_logger(__name__)

# Shared state — set by launcher or tests
_db: Optional[StateDB] = None


def get_db() -> StateDB:
    """Return the global StateDB instance."""
    global _db
    if _db is None:
        _db = StateDB()
    return _db


def set_db(db: StateDB):
    """Override the global StateDB (used by tests and launcher)."""
    global _db
    _db = db


def create_app(db: Optional[StateDB] = None):
    """
    Create and return the FastAPI application.

    Args:
        db: Optional StateDB instance. If None, uses the global default.
    """
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required for the backend engine. "
            "Install with: pip install adamops[backend]"
        )

    if db is not None:
        set_db(db)

    app = FastAPI(
        title="AdamOps Backend Engine",
        version="0.1.1",
        description="Experiment tracking, model management, and real-time run monitoring.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Pydantic schemas ─────────────────────────────────────────────────

    class ExperimentCreate(BaseModel):
        name: str
        description: str = ""

    class RunCreate(BaseModel):
        experiment_id: int
        algorithm: str = "random_forest"
        task: str = "auto"
        backend: str = "local"
        params: dict = {}
        # Inline data for demo/testing — real usage would reference datasets
        features: Optional[List[List[float]]] = None
        target: Optional[List[float]] = None

    class ModelRegister(BaseModel):
        run_id: Optional[int] = None
        name: str
        version: str
        artifact_path: str = ""
        metadata: dict = {}

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/")
    def root():
        return {
            "service": "AdamOps Backend Engine",
            "version": "0.1.1",
            "status": "healthy",
        }

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    # ── Experiments ───────────────────────────────────────────────────────

    @app.get("/api/experiments")
    def list_experiments():
        return {"experiments": get_db().list_experiments()}

    @app.post("/api/experiments", status_code=201)
    def create_experiment(body: ExperimentCreate):
        exp = get_db().create_experiment(body.name, body.description)
        return {"experiment": exp}

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: int):
        exp = get_db().get_experiment(experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        runs = get_db().list_runs(experiment_id)
        return {"experiment": exp, "runs": runs}

    @app.delete("/api/experiments/{experiment_id}")
    def delete_experiment(experiment_id: int):
        exp = get_db().get_experiment(experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        get_db().delete_experiment(experiment_id)
        return {"deleted": True}

    # ── Runs ──────────────────────────────────────────────────────────────

    @app.post("/api/runs", status_code=201)
    def create_run(body: RunCreate):
        import numpy as np

        db = get_db()
        exp = db.get_experiment(body.experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Experiment not found")

        run = db.create_run(
            experiment_id=body.experiment_id,
            algorithm=body.algorithm,
            task=body.task,
            params=body.params,
            backend=body.backend,
        )

        # If inline data was provided, launch the run immediately
        if body.features is not None and body.target is not None:
            X = np.array(body.features)
            y = np.array(body.target)
            submit_run(
                db=db,
                run_id=run["id"],
                X=X,
                y=y,
                task=body.task,
                algorithm=body.algorithm,
                backend=body.backend,
                params=body.params,
            )
            run["status"] = "running"

        return {"run": run}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int):
        run = get_db().get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run": run}

    @app.get("/api/runs/{run_id}/events")
    def get_run_events(run_id: int):
        run = get_db().get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        events = get_db().get_events(run_id)
        return {"events": events}

    # ── Models ────────────────────────────────────────────────────────────

    @app.get("/api/models")
    def list_models():
        return {"models": get_db().list_models()}

    @app.post("/api/models", status_code=201)
    def register_model(body: ModelRegister):
        model = get_db().register_model(
            name=body.name,
            version=body.version,
            run_id=body.run_id,
            artifact_path=body.artifact_path,
            metadata=body.metadata,
        )
        return {"model": model}

    # ── WebSocket ─────────────────────────────────────────────────────────

    @app.websocket("/ws/runs/{run_id}")
    async def ws_run_stream(websocket: WebSocket, run_id: int):
        """
        Real-time event stream for a specific run.

        Sends JSON messages of the form:
            {"event_type": "...", "payload": {...}}
        """
        await websocket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _on_event(event: RunEvent):
            """Thread-safe callback: schedule queue put on the event loop."""
            loop.call_soon_threadsafe(queue.put_nowait, event)

        event_bus.subscribe(run_id, _on_event)

        try:
            # First send any historical events
            historical = get_db().get_events(run_id)
            for evt in historical:
                await websocket.send_json({
                    "event_type": evt["event_type"],
                    "payload": evt["payload"],
                    "timestamp": evt["timestamp"],
                    "replay": True,
                })

            # Then stream live events
            while True:
                event = await queue.get()
                await websocket.send_json({
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "replay": False,
                })
                # If the run finished, close cleanly
                if event.event_type == "status_change" and event.payload.get("status") in (
                    "completed",
                    "failed",
                ):
                    await websocket.close()
                    break
        except WebSocketDisconnect:
            logger.debug(f"WebSocket client disconnected for run {run_id}")
        except Exception as exc:
            logger.error(f"WebSocket error for run {run_id}: {exc}")
        finally:
            event_bus.unsubscribe(run_id, _on_event)

    return app
