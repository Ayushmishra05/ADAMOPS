"""
Tests for the AdamOps Colab Bridge.

Uses a mock WebSocket + REST server to test ColabBridge without
a real Colab runtime.
"""

import json
import os
import tempfile
import threading
import time
import pytest

from adamops.colab.bridge import ColabBridge, ColabResult, _make_execute_msg
from adamops.colab.setup_snippet import setup_colab, get_setup_snippet, COLAB_SETUP_SNIPPET


# ---------------------------------------------------------------------------
# Mock kernel gateway server (Flask + simple WebSocket)
# ---------------------------------------------------------------------------

_MOCK_KERNEL_ID = "mock-kernel-deadbeef"


def _create_mock_gateway(port: int):
    """Create a Flask app that mimics the Jupyter kernel gateway REST API."""
    try:
        from flask import Flask, jsonify, request
    except ImportError:
        pytest.skip("flask required for colab tests")

    app = Flask(__name__)

    @app.route("/api/kernels", methods=["GET"])
    def list_kernels():
        return jsonify([{"id": _MOCK_KERNEL_ID, "name": "python3"}])

    @app.route("/api/kernels", methods=["POST"])
    def start_kernel():
        return jsonify({"id": _MOCK_KERNEL_ID, "name": "python3"})

    return app


# ---------------------------------------------------------------------------
# Unit tests (no server needed)
# ---------------------------------------------------------------------------


class TestColabResult:
    def test_result_defaults(self):
        r = ColabResult()
        assert r.success is True
        assert r.returncode == 0
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.elapsed == 0.0
        assert r.outputs == []

    def test_result_repr_success(self):
        r = ColabResult(success=True, elapsed=1.5, stdout="hello world")
        assert "✅ OK" in repr(r)
        assert "1.50s" in repr(r)

    def test_result_repr_error(self):
        r = ColabResult(success=False, error_name="ValueError", error_value="bad input")
        assert "❌ ValueError" in repr(r)


class TestMessageProtocol:
    def test_execute_msg_structure(self):
        msg = _make_execute_msg("print('hi')", "test-session")
        assert msg["header"]["msg_type"] == "execute_request"
        assert msg["content"]["code"] == "print('hi')"
        assert msg["header"]["session"] == "test-session"
        assert msg["header"]["username"] == "adamops"
        assert msg["channel"] == "shell"
        assert msg["content"]["silent"] is False
        assert msg["content"]["allow_stdin"] is False


class TestSetupSnippet:
    def test_snippet_content(self):
        snippet = get_setup_snippet()
        assert "jupyter_kernel_gateway" in snippet
        assert "ColabBridge" in snippet
        assert "TOKEN" in snippet
        assert "secrets.token_hex" in snippet

    def test_snippet_is_string(self):
        assert isinstance(COLAB_SETUP_SNIPPET, str)
        assert len(COLAB_SETUP_SNIPPET) > 100

    def test_setup_colab_prints(self, capsys):
        setup_colab()
        captured = capsys.readouterr()
        assert "Copy and paste" in captured.out
        assert "ColabBridge" in captured.out


class TestColabBridgeInit:
    def test_init_stores_url(self):
        bridge = ColabBridge("http://localhost:8888", token="abc123")
        assert bridge._base_url == "http://localhost:8888"
        assert bridge._token == "abc123"
        assert bridge._headers == {"Authorization": "token abc123"}

    def test_init_strips_trailing_slash(self):
        bridge = ColabBridge("http://localhost:8888/", token="")
        assert bridge._base_url == "http://localhost:8888"

    def test_init_no_token(self):
        bridge = ColabBridge("http://localhost:8888")
        assert bridge._headers == {}

    def test_ws_url_http(self):
        bridge = ColabBridge("http://localhost:8888", token="tok")
        bridge._kernel_id = "kern-123"
        ws_url = bridge._ws_url()
        assert ws_url == "ws://localhost:8888/api/kernels/kern-123/channels?token=tok"

    def test_ws_url_https(self):
        bridge = ColabBridge("https://colab.example.com", token="tok")
        bridge._kernel_id = "kern-456"
        ws_url = bridge._ws_url()
        assert ws_url == "wss://colab.example.com/api/kernels/kern-456/channels?token=tok"


class TestColabBridgeWithMockServer:
    """Tests that use a real Flask mock server for REST endpoints."""

    @pytest.fixture(autouse=True)
    def _start_mock_server(self):
        """Start a mock gateway server on a random port."""
        try:
            from flask import Flask
        except ImportError:
            pytest.skip("flask required")

        self.port = 19411
        self.app = _create_mock_gateway(self.port)

        self.server_thread = threading.Thread(
            target=lambda: self.app.run(
                host="127.0.0.1", port=self.port, use_reloader=False
            ),
            daemon=True,
        )
        self.server_thread.start()
        time.sleep(0.5)  # wait for startup

        self.bridge = ColabBridge(f"http://127.0.0.1:{self.port}", token="test-token")
        yield

    def test_ensure_kernel_reuses_existing(self):
        self.bridge._ensure_kernel()
        assert self.bridge._kernel_id == _MOCK_KERNEL_ID

    def test_api_list_kernels(self):
        kernels = self.bridge._api("GET", "/api/kernels")
        assert len(kernels) == 1
        assert kernels[0]["id"] == _MOCK_KERNEL_ID


class TestRunScript:
    def test_run_script_file_not_found(self):
        bridge = ColabBridge("http://localhost:8888")
        with pytest.raises(FileNotFoundError):
            bridge.run_script("/nonexistent/path.py")


class TestRunNotebook:
    def test_run_notebook_file_not_found(self):
        bridge = ColabBridge("http://localhost:8888")
        with pytest.raises(FileNotFoundError):
            bridge.run_notebook("/nonexistent/notebook.ipynb")


class TestUpload:
    def test_upload_file_not_found(self):
        bridge = ColabBridge("http://localhost:8888")
        with pytest.raises(FileNotFoundError):
            bridge.upload("/nonexistent/data.csv")


class TestCLIIntegration:
    def test_colab_setup_cli(self):
        """Test that the colab-setup CLI command is registered."""
        from click.testing import CliRunner
        from adamops.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["colab-setup"])
        assert result.exit_code == 0
        assert "ColabBridge" in result.output
        assert "Copy and paste" in result.output
