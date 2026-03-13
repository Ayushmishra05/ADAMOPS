"""
AdamOps Studio — Visual Drag-and-Drop Pipeline Builder

Launch a browser-based UI to build ML pipelines visually.

Usage:
    from adamops.studio import launch
    results = launch()
"""

from adamops.studio.launcher import launch

__all__ = ["launch"]
