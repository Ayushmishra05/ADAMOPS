"""
AdamOps Backend Engine

Production-grade FastAPI backend with WebSocket support
and SQLite state persistence for experiment tracking.
"""

import importlib


def __getattr__(name):
    if name in ["create_app", "launch", "StateDB"]:
        if name == "StateDB":
            mod = importlib.import_module(".database", __name__)
            return getattr(mod, name)
        elif name == "create_app":
            mod = importlib.import_module(".app", __name__)
            return getattr(mod, name)
        elif name == "launch":
            mod = importlib.import_module(".launcher", __name__)
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return ["create_app", "launch", "StateDB"]


__all__ = ["create_app", "launch", "StateDB"]
