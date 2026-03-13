"""
AdamOps Studio — Visual Drag-and-Drop Pipeline Builder

Launch a browser-based UI to build ML pipelines visually.

Usage:
    from adamops.studio import launch
    results = launch()
"""

if "launch" not in locals():
    from . import launch

from .launcher import launch

__all__ = ["launch"]
