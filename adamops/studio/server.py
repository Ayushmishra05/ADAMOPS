"""
AdamOps Studio — Flask Server

Serves the visual pipeline builder UI and API endpoints.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

from adamops.utils.logging import get_logger

logger = get_logger(__name__)

# Upload directory for data files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "adamops_studio_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Last execution result (shared between server and launcher)
_last_result: Optional[Dict] = None


def create_app():
    """Create the Flask application for AdamOps Studio."""
    try:
        from flask import Flask, request, jsonify, send_from_directory
    except ImportError:
        raise ImportError("Flask required for Studio. Install with: pip install flask")

    from adamops.studio.nodes import get_all_node_types
    from adamops.studio.engine import execute_pipeline

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app = Flask(__name__, static_folder=static_dir)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.route("/static/<path:filename>")
    def serve_static(filename):
        return send_from_directory(static_dir, filename)

    @app.route("/api/nodes", methods=["GET"])
    def list_nodes():
        """Return all available node types."""
        return jsonify({"nodes": get_all_node_types()})

    @app.route("/api/execute", methods=["POST"])
    def execute():
        """Execute a pipeline."""
        global _last_result
        try:
            pipeline_data = request.get_json()
            if not pipeline_data:
                return jsonify({"error": "No pipeline data provided"}), 400

            result = execute_pipeline(pipeline_data)
            _last_result = result.to_dict()
            return jsonify(_last_result)

        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            return jsonify({"error": str(e), "success": False}), 500

    @app.route("/api/upload", methods=["POST"])
    def upload_file():
        """Upload a data file."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filename = file.filename
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        logger.info(f"File uploaded: {filepath}")

        return jsonify({
            "filename": filename,
            "filepath": filepath,
            "size": os.path.getsize(filepath),
        })

    @app.route("/api/result", methods=["GET"])
    def get_result():
        """Get the last execution result."""
        if _last_result:
            return jsonify(_last_result)
        return jsonify({"error": "No results available"}), 404

    return app


def get_last_result() -> Optional[Dict]:
    """Get the last execution result (used by launcher)."""
    return _last_result
