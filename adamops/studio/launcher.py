"""
AdamOps Studio — Launcher

Starts the Studio server and opens the browser.
"""

import threading
import webbrowser
import time
from typing import Optional, Dict

from adamops.utils.logging import get_logger

logger = get_logger(__name__)


def launch(host: str = "127.0.0.1", port: int = 5555,
           open_browser: bool = True, debug: bool = False) -> Optional[Dict]:
    """
    Launch the AdamOps Studio visual pipeline builder.

    Opens a browser-based drag-and-drop interface for building ML pipelines.
    The function blocks until you press Ctrl+C or close the terminal.

    Args:
        host: Server host address.
        port: Server port number.
        open_browser: Whether to automatically open the browser.
        debug: Enable Flask debug mode.

    Returns:
        The last execution result dict, or None if no pipeline was run.

    Example:
        >>> from adamops.studio import launch
        >>> result = launch()
        >>> if result and result.get("final_metrics"):
        ...     print(result["final_metrics"])
    """
    from adamops.studio.server import create_app, get_last_result

    app = create_app()
    url = f"http://{host}:{port}"

    print(f"\n{'='*60}")
    print(f"  🚀 AdamOps Studio")
    print(f"  Open in browser: {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\nShutting down AdamOps Studio...")

    return get_last_result()
