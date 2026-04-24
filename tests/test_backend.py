"""
Tests for the AdamOps Backend Engine.

Covers:
    1. StateDB CRUD operations (in-memory SQLite)
    2. FastAPI REST endpoints (TestClient)
    3. WebSocket real-time streaming
    4. Full run lifecycle (submit → poll → completed)
"""

import time
import threading
import pytest
import numpy as np
from sklearn.datasets import make_classification

# ── Database Tests ─────────────────────────────────────────────────────────

from adamops.backend.database import StateDB


@pytest.fixture
def db():
    """Fresh in-memory database for each test."""
    return StateDB(":memory:")


class TestStateDB:
    def test_create_and_get_experiment(self, db):
        exp = db.create_experiment("test-exp", "A test experiment")
        assert exp["name"] == "test-exp"
        assert exp["description"] == "A test experiment"
        assert exp["id"] is not None

        fetched = db.get_experiment(exp["id"])
        assert fetched["name"] == "test-exp"

    def test_list_experiments(self, db):
        db.create_experiment("exp-1")
        db.create_experiment("exp-2")
        exps = db.list_experiments()
        assert len(exps) == 2

    def test_delete_experiment_cascades(self, db):
        exp = db.create_experiment("doomed")
        run = db.create_run(exp["id"], algorithm="rf")
        db.log_event(run["id"], "log", {"msg": "hello"})
        db.register_model("m1", "v1", run_id=run["id"])

        db.delete_experiment(exp["id"])
        assert db.get_experiment(exp["id"]) is None
        assert db.get_run(run["id"]) is None
        assert db.get_events(run["id"]) == []

    def test_create_and_get_run(self, db):
        exp = db.create_experiment("exp")
        run = db.create_run(exp["id"], algorithm="xgboost", task="classification")
        assert run["algorithm"] == "xgboost"
        assert run["status"] == "pending"

    def test_update_run_status_and_metrics(self, db):
        exp = db.create_experiment("exp")
        run = db.create_run(exp["id"])
        db.update_run_status(run["id"], "running")
        assert db.get_run(run["id"])["status"] == "running"

        db.update_run_metrics(run["id"], {"accuracy": 0.95})
        assert db.get_run(run["id"])["metrics"]["accuracy"] == 0.95

        db.update_run_status(run["id"], "completed")
        completed = db.get_run(run["id"])
        assert completed["status"] == "completed"
        assert completed["finished_at"] is not None

    def test_list_runs(self, db):
        exp = db.create_experiment("exp")
        db.create_run(exp["id"], algorithm="rf")
        db.create_run(exp["id"], algorithm="lr")
        runs = db.list_runs(exp["id"])
        assert len(runs) == 2

    def test_register_and_list_models(self, db):
        model = db.register_model("my-model", "v1", metadata={"acc": 0.9})
        assert model["name"] == "my-model"
        assert model["version"] == "v1"
        assert model["metadata"]["acc"] == 0.9

        models = db.list_models()
        assert len(models) == 1

    def test_log_and_get_events(self, db):
        exp = db.create_experiment("exp")
        run = db.create_run(exp["id"])
        db.log_event(run["id"], "log", {"message": "starting"})
        db.log_event(run["id"], "metrics", {"accuracy": 0.8})

        events = db.get_events(run["id"])
        assert len(events) == 2
        assert events[0]["event_type"] == "log"
        assert events[1]["payload"]["accuracy"] == 0.8


# ── FastAPI REST Tests ─────────────────────────────────────────────────────

from adamops.backend.app import create_app


@pytest.fixture
def client(db):
    """TestClient wired to an in-memory DB."""
    from fastapi.testclient import TestClient

    app = create_app(db)
    return TestClient(app)


class TestRESTEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_experiment_crud(self, client):
        # Create
        r = client.post("/api/experiments", json={"name": "e1", "description": "test"})
        assert r.status_code == 201
        exp_id = r.json()["experiment"]["id"]

        # List
        r = client.get("/api/experiments")
        assert len(r.json()["experiments"]) == 1

        # Get with runs
        r = client.get(f"/api/experiments/{exp_id}")
        assert r.json()["experiment"]["name"] == "e1"
        assert r.json()["runs"] == []

        # Delete
        r = client.delete(f"/api/experiments/{exp_id}")
        assert r.json()["deleted"] is True

    def test_experiment_not_found(self, client):
        r = client.get("/api/experiments/9999")
        assert r.status_code == 404

    def test_run_crud(self, client):
        # Setup experiment
        r = client.post("/api/experiments", json={"name": "e1"})
        exp_id = r.json()["experiment"]["id"]

        # Create run (no inline data — stays pending)
        r = client.post("/api/runs", json={
            "experiment_id": exp_id,
            "algorithm": "random_forest",
            "task": "classification",
        })
        assert r.status_code == 201
        run_id = r.json()["run"]["id"]
        assert r.json()["run"]["status"] == "pending"

        # Get run
        r = client.get(f"/api/runs/{run_id}")
        assert r.json()["run"]["algorithm"] == "random_forest"

        # Events (empty)
        r = client.get(f"/api/runs/{run_id}/events")
        assert r.json()["events"] == []

    def test_run_not_found(self, client):
        r = client.get("/api/runs/9999")
        assert r.status_code == 404

    def test_model_crud(self, client):
        r = client.post("/api/models", json={
            "name": "best-model",
            "version": "v1",
            "metadata": {"accuracy": 0.95},
        })
        assert r.status_code == 201
        assert r.json()["model"]["name"] == "best-model"

        r = client.get("/api/models")
        assert len(r.json()["models"]) == 1

    def test_run_with_inline_data(self, client):
        """Submit a run with inline data — should trigger background execution."""
        r = client.post("/api/experiments", json={"name": "e1"})
        exp_id = r.json()["experiment"]["id"]

        X, y = make_classification(n_samples=50, n_features=5, random_state=42)

        r = client.post("/api/runs", json={
            "experiment_id": exp_id,
            "algorithm": "random_forest",
            "task": "classification",
            "features": X.tolist(),
            "target": y.tolist(),
        })
        assert r.status_code == 201
        run_id = r.json()["run"]["id"]

        # Wait for the background thread to finish
        time.sleep(3)

        r = client.get(f"/api/runs/{run_id}")
        run = r.json()["run"]
        assert run["status"] == "completed"
        assert "accuracy" in run["metrics"]

        # Events should have been logged
        r = client.get(f"/api/runs/{run_id}/events")
        assert len(r.json()["events"]) > 0


# ── WebSocket Test ─────────────────────────────────────────────────────────

class TestWebSocket:
    def test_ws_streams_events(self, client, db):
        """Connect to WebSocket for a run, trigger events, verify they stream."""
        from adamops.backend.runner import event_bus, RunEvent

        exp = db.create_experiment("ws-test")
        run = db.create_run(exp["id"])

        # Log an event before connecting (should come as replay)
        db.log_event(run["id"], "log", {"message": "pre-connect"})

        def _publish_completion():
            """Publish a completion event so the WS handler closes cleanly."""
            import time
            time.sleep(0.5)
            event_bus.publish(RunEvent(
                run["id"], "status_change", {"status": "completed"}
            ))

        t = threading.Thread(target=_publish_completion, daemon=True)
        t.start()

        with client.websocket_connect(f"/ws/runs/{run['id']}") as ws:
            # Should receive the historical event as replay
            msg = ws.receive_json()
            assert msg["replay"] is True
            assert msg["event_type"] == "log"
            assert msg["payload"]["message"] == "pre-connect"

            # Should receive the live completion event
            msg2 = ws.receive_json()
            assert msg2["replay"] is False
            assert msg2["event_type"] == "status_change"
            assert msg2["payload"]["status"] == "completed"

        t.join(timeout=2)

