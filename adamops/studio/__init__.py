"""
AdamOps Studio — Visual Drag-and-Drop Pipeline Builder

Launch a browser-based UI to build ML pipelines visually.

Usage:
    from adamops.studio import launch
    results = launch()
"""

import importlib


def __getattr__(name):
    if name in ["launch", "execute_pipeline", "compile_pipeline"]:
        if name == "launch":
            mod = importlib.import_module(".launcher", __name__)
            return getattr(mod, name)
        elif name == "execute_pipeline":
            mod = importlib.import_module(".engine", __name__)
            return getattr(mod, name)
        elif name == "compile_pipeline":
            mod = importlib.import_module(".compiler", __name__)
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return ["launch", "execute_pipeline", "compile_pipeline"]


__all__ = ["launch", "execute_pipeline", "compile_pipeline"]
