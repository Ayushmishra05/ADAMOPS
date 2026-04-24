"""
AdamOps Colab Bridge — Remote GPU Execution

Connect to a Google Colab runtime and execute .py / .ipynb files
on remote GPU hardware using the Jupyter kernel gateway protocol.

Usage:
    from adamops.colab import ColabBridge
    bridge = ColabBridge("https://colab-gateway-url", token="abc123")
    result = bridge.run_script("train.py")
"""

import importlib


def __getattr__(name):
    if name == "ColabBridge":
        mod = importlib.import_module(".bridge", __name__)
        return getattr(mod, name)
    elif name == "setup_colab":
        mod = importlib.import_module(".setup_snippet", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return ["ColabBridge", "setup_colab"]


__all__ = ["ColabBridge", "setup_colab"]
