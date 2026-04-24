"""
AdamOps Backend — Launcher

Convenience function to initialise the database, create the FastAPI app,
and start the uvicorn server.
"""

from typing import Optional

from adamops.utils.logging import get_logger

logger = get_logger(__name__)


def launch(
    host: str = "127.0.0.1",
    port: int = 8000,
    db_path: str = "adamops_state.db",
    reload: bool = False,
):
    """
    Start the AdamOps backend server.

    Args:
        host: Bind address.
        port: Bind port.
        db_path: Path to SQLite state database file.
        reload: Enable auto-reload for development.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the backend server. "
            "Install with: pip install adamops[backend]"
        )

    from adamops.backend.database import StateDB
    from adamops.backend.app import create_app

    db = StateDB(db_path)
    app = create_app(db)

    banner = f"""
╔══════════════════════════════════════════════════╗
║        ⚡ AdamOps Backend Engine                 ║
║                                                  ║
║  REST API:   http://{host}:{port}              ║
║  WebSocket:  ws://{host}:{port}/ws/runs/{{id}}  ║
║  Database:   {db_path:<35s}║
║                                                  ║
║  Press Ctrl+C to stop                            ║
╚══════════════════════════════════════════════════╝
"""
    print(banner)

    uvicorn.run(app, host=host, port=port, log_level="info")
