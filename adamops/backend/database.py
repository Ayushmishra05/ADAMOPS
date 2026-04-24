"""
AdamOps Backend — SQLite State Database

Provides persistent storage for experiments, runs, models, and events
using Python's built-in sqlite3 module (zero external dependencies).
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class StateDB:
    """
    Thread-safe SQLite state database for ADAMOPS backend.

    Manages experiments, runs, registered models, and event logs.
    Uses WAL journal mode for concurrent read/write performance.
    """

    def __init__(self, db_path: str = "adamops_state.db"):
        """
        Args:
            db_path: Path to SQLite file. Use ':memory:' for in-memory DB.
        """
        self.db_path = db_path
        self._is_memory = db_path == ":memory:"
        self._local = threading.local()
        self._shared_conn = None  # used only for :memory:
        self._lock = threading.Lock()
        self._init_schema()

    # ── Connection management ─────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """
        Get a database connection.

        For :memory: databases, return a single shared connection
        (in-memory DBs are per-connection, so thread-local would create
        separate databases). For file-based DBs, use thread-local connections.
        """
        if self._is_memory:
            if self._shared_conn is None:
                self._shared_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False
                )
                self._shared_conn.row_factory = sqlite3.Row
                self._shared_conn.execute("PRAGMA foreign_keys=ON")
            return self._shared_conn

        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn


    def _init_schema(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                description TEXT    DEFAULT '',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id   INTEGER NOT NULL,
                algorithm       TEXT    DEFAULT '',
                task            TEXT    DEFAULT 'auto',
                params          TEXT    DEFAULT '{}',
                metrics         TEXT    DEFAULT '{}',
                status          TEXT    DEFAULT 'pending',
                backend         TEXT    DEFAULT 'local',
                started_at      TEXT,
                finished_at     TEXT,
                error           TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE TABLE IF NOT EXISTS models (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id        INTEGER,
                name          TEXT NOT NULL,
                version       TEXT NOT NULL,
                artifact_path TEXT DEFAULT '',
                metadata      TEXT DEFAULT '{}',
                created_at    TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     INTEGER NOT NULL,
                event_type TEXT    NOT NULL,
                payload    TEXT    DEFAULT '{}',
                timestamp  TEXT    NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_runs_experiment  ON runs(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs(status);
            CREATE INDEX IF NOT EXISTS idx_events_run       ON events(run_id);
            CREATE INDEX IF NOT EXISTS idx_models_run       ON models(run_id);
        """)
        conn.commit()

    def close(self):
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # ── Experiments ───────────────────────────────────────────────────────

    def create_experiment(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new experiment and return its record."""
        now = _now_iso()
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO experiments (name, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, description, now, now),
        )
        conn.commit()
        return self.get_experiment(cur.lastrowid)

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get a single experiment by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments, most recent first."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment and its associated runs/events."""
        conn = self._get_conn()
        # Cascade delete events and models linked to runs
        run_ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM runs WHERE experiment_id = ?", (experiment_id,)
            ).fetchall()
        ]
        for rid in run_ids:
            conn.execute("DELETE FROM events WHERE run_id = ?", (rid,))
            conn.execute("DELETE FROM models WHERE run_id = ?", (rid,))
        conn.execute("DELETE FROM runs WHERE experiment_id = ?", (experiment_id,))
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()
        return True

    # ── Runs ──────────────────────────────────────────────────────────────

    def create_run(
        self,
        experiment_id: int,
        algorithm: str = "",
        task: str = "auto",
        params: Optional[Dict] = None,
        backend: str = "local",
    ) -> Dict[str, Any]:
        """Create a new run record in 'pending' status."""
        now = _now_iso()
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO runs
               (experiment_id, algorithm, task, params, status, backend, started_at)
               VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
            (experiment_id, algorithm, task, json.dumps(params or {}), backend, now),
        )
        conn.commit()
        return self.get_run(cur.lastrowid)

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get a single run by ID, with parsed JSON fields."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d.get("params", "{}"))
        d["metrics"] = json.loads(d.get("metrics", "{}"))
        return d

    def list_runs(self, experiment_id: int) -> List[Dict[str, Any]]:
        """List all runs for an experiment."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY started_at DESC",
            (experiment_id,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["params"] = json.loads(d.get("params", "{}"))
            d["metrics"] = json.loads(d.get("metrics", "{}"))
            results.append(d)
        return results

    def update_run_status(self, run_id: int, status: str, error: str = None):
        """Update a run's status (pending → running → completed / failed)."""
        conn = self._get_conn()
        if status in ("completed", "failed"):
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
                (status, _now_iso(), error, run_id),
            )
        else:
            conn.execute(
                "UPDATE runs SET status = ? WHERE id = ?", (status, run_id)
            )
        conn.commit()

    def update_run_metrics(self, run_id: int, metrics: Dict[str, Any]):
        """Persist final metrics for a run."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE runs SET metrics = ? WHERE id = ?",
            (json.dumps(metrics), run_id),
        )
        conn.commit()

    # ── Models ────────────────────────────────────────────────────────────

    def register_model(
        self,
        name: str,
        version: str,
        run_id: int = None,
        artifact_path: str = "",
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Register a model artifact."""
        now = _now_iso()
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO models (run_id, name, version, artifact_path, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, name, version, artifact_path, json.dumps(metadata or {}), now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM models WHERE id = ?", (cur.lastrowid,)).fetchone()
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        return d

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM models ORDER BY created_at DESC"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["metadata"] = json.loads(d.get("metadata", "{}"))
            results.append(d)
        return results

    # ── Events ────────────────────────────────────────────────────────────

    def log_event(self, run_id: int, event_type: str, payload: Optional[Dict] = None):
        """Log an event for a run (used for WebSocket fan-out replay)."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO events (run_id, event_type, payload, timestamp) VALUES (?, ?, ?, ?)",
            (run_id, event_type, json.dumps(payload or {}), _now_iso()),
        )
        conn.commit()

    def get_events(self, run_id: int) -> List[Dict[str, Any]]:
        """Get all events for a run, ordered chronologically."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY timestamp ASC",
            (run_id,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["payload"] = json.loads(d.get("payload", "{}"))
            results.append(d)
        return results
