"""
AdamOps Colab Bridge — Kernel Gateway Client

Connects to a remote Colab (or any Jupyter) kernel via the
kernel gateway REST + WebSocket protocol and executes code remotely.

Supports:
  - Raw code execution (.py strings)
  - Script file execution (.py files)
  - Notebook execution (.ipynb files, cell-by-cell)
  - File upload / download via base64 encoding
  - Studio pipeline compilation + remote execution
"""

import json
import time
import uuid
import base64
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adamops.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ColabResult:
    """Result from a remote code execution."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    elapsed: float = 0.0
    success: bool = True
    error_name: str = ""
    error_value: str = ""
    traceback: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "✅ OK" if self.success else f"❌ {self.error_name}: {self.error_value}"
        return f"ColabResult({status}, {self.elapsed:.2f}s, stdout={len(self.stdout)} chars)"


# ---------------------------------------------------------------------------
# Kernel messaging helpers
# ---------------------------------------------------------------------------


def _make_msg(msg_type: str, content: Dict, session: str) -> Dict:
    """Create a Jupyter kernel protocol message."""
    return {
        "header": {
            "msg_id": uuid.uuid4().hex,
            "msg_type": msg_type,
            "username": "adamops",
            "session": session,
            "date": "",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": content,
        "buffers": [],
        "channel": "shell",
    }


def _make_execute_msg(code: str, session: str) -> Dict:
    """Create an execute_request message."""
    return _make_msg(
        "execute_request",
        {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        session,
    )


# ---------------------------------------------------------------------------
# ColabBridge
# ---------------------------------------------------------------------------


class ColabBridge:
    """
    Client for executing code on a remote Colab (Jupyter) kernel.

    Usage:
        bridge = ColabBridge("https://gateway-url", token="abc123")
        print(bridge.status())
        result = bridge.execute("print('hello from GPU!')")
        result = bridge.run_script("train.py")
        result = bridge.run_notebook("pipeline.ipynb")
    """

    def __init__(self, gateway_url: str, token: str = ""):
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for ColabBridge. "
                "Install with: pip install requests"
            )

        self._base_url = gateway_url.rstrip("/")
        self._token = token
        self._session = uuid.uuid4().hex
        self._kernel_id: Optional[str] = None
        self._headers = {"Authorization": f"token {token}"} if token else {}
        self._requests = requests

        logger.info(f"ColabBridge connecting to {self._base_url}")

    # -- connection helpers -------------------------------------------------

    def _api(self, method: str, path: str, **kwargs) -> Any:
        """Make an authenticated REST request to the gateway."""
        url = f"{self._base_url}{path}"
        resp = self._requests.request(method, url, headers=self._headers, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else None

    def _ensure_kernel(self):
        """Start or reuse a kernel on the gateway."""
        if self._kernel_id:
            return
        # List existing kernels
        kernels = self._api("GET", "/api/kernels")
        if kernels:
            self._kernel_id = kernels[0]["id"]
            logger.info(f"Reusing existing kernel: {self._kernel_id}")
        else:
            # Start a new kernel
            result = self._api("POST", "/api/kernels", json={"name": "python3"})
            self._kernel_id = result["id"]
            logger.info(f"Started new kernel: {self._kernel_id}")

    def _ws_url(self) -> str:
        """Build the WebSocket URL for the kernel channels."""
        base = self._base_url.replace("https://", "wss://").replace("http://", "ws://")
        url = f"{base}/api/kernels/{self._kernel_id}/channels"
        if self._token:
            url += f"?token={self._token}"
        return url

    # -- public API ---------------------------------------------------------

    def status(self) -> Dict:
        """Get status of the remote kernel (GPU info, Python version)."""
        self._ensure_kernel()

        # Execute a small introspection script
        info_code = """
import sys, os, json
info = {
    "python_version": sys.version,
    "platform": sys.platform,
    "cwd": os.getcwd(),
}
try:
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
except ImportError:
    info["cuda_available"] = False
print("__ADAMOPS_STATUS__" + json.dumps(info))
"""
        result = self.execute(info_code)
        for line in result.stdout.split("\n"):
            if line.startswith("__ADAMOPS_STATUS__"):
                return json.loads(line[len("__ADAMOPS_STATUS__"):])

        return {"status": "connected", "kernel_id": self._kernel_id}

    def execute(self, code: str, timeout: int = 300) -> ColabResult:
        """
        Execute raw Python code on the remote kernel.

        Returns a ColabResult with stdout, stderr, and status.
        """
        try:
            import websocket
        except ImportError:
            raise ImportError(
                "websocket-client is required for ColabBridge. "
                "Install with: pip install websocket-client"
            )

        self._ensure_kernel()

        start_time = time.time()
        result = ColabResult()

        ws_url = self._ws_url()
        ws = websocket.create_connection(ws_url, timeout=timeout)

        try:
            # Send execute request
            msg = _make_execute_msg(code, self._session)
            ws.send(json.dumps(msg))
            parent_msg_id = msg["header"]["msg_id"]

            # Collect responses
            while True:
                raw = ws.recv()
                reply = json.loads(raw)
                msg_type = reply.get("msg_type") or reply.get("header", {}).get("msg_type", "")
                parent_id = (
                    reply.get("parent_header", {}).get("msg_id", "")
                )

                # Only process messages for our request
                if parent_id != parent_msg_id:
                    continue

                if msg_type == "stream":
                    stream_name = reply["content"].get("name", "stdout")
                    text = reply["content"].get("text", "")
                    if stream_name == "stdout":
                        result.stdout += text
                    else:
                        result.stderr += text

                elif msg_type == "execute_result":
                    data = reply["content"].get("data", {})
                    result.outputs.append(data)

                elif msg_type == "display_data":
                    data = reply["content"].get("data", {})
                    result.outputs.append(data)

                elif msg_type == "error":
                    result.success = False
                    result.returncode = 1
                    result.error_name = reply["content"].get("ename", "Error")
                    result.error_value = reply["content"].get("evalue", "")
                    result.traceback = reply["content"].get("traceback", [])
                    result.stderr += f"{result.error_name}: {result.error_value}\n"

                elif msg_type == "execute_reply":
                    status = reply["content"].get("status", "")
                    if status == "error":
                        result.success = False
                        result.returncode = 1
                        if not result.error_name:
                            result.error_name = reply["content"].get("ename", "Error")
                            result.error_value = reply["content"].get("evalue", "")
                    break  # execute_reply signals completion

                elif msg_type == "status":
                    exec_state = reply["content"].get("execution_state", "")
                    if exec_state == "idle" and result.stdout:
                        # Sometimes idle comes before execute_reply
                        pass

        finally:
            ws.close()

        result.elapsed = time.time() - start_time
        return result

    def run_script(self, script_path: str, timeout: int = 300) -> ColabResult:
        """
        Read a local .py file and execute its contents on the remote kernel.
        """
        path = os.path.abspath(script_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Script not found: {path}")

        with open(path, "r") as f:
            code = f.read()

        logger.info(f"Sending {os.path.basename(path)} ({len(code)} chars) to Colab")
        return self.execute(code, timeout=timeout)

    def run_notebook(
        self, notebook_path: str, timeout: int = 300, save: bool = True
    ) -> List[ColabResult]:
        """
        Parse a .ipynb file and execute each code cell on the remote kernel.

        If save=True, writes the executed notebook with outputs back to disk.
        Returns a list of ColabResult (one per code cell).
        """
        path = os.path.abspath(notebook_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Notebook not found: {path}")

        with open(path, "r") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        results = []

        for i, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            if not source.strip():
                continue

            logger.info(f"Executing cell {i + 1}/{len(cells)}")
            result = self.execute(source, timeout=timeout)
            results.append(result)

            # Inject outputs back into the notebook structure
            if save:
                cell_outputs = []
                if result.stdout:
                    cell_outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": result.stdout.split("\n"),
                    })
                if result.stderr and result.success:
                    cell_outputs.append({
                        "output_type": "stream",
                        "name": "stderr",
                        "text": result.stderr.split("\n"),
                    })
                if not result.success:
                    cell_outputs.append({
                        "output_type": "error",
                        "ename": result.error_name,
                        "evalue": result.error_value,
                        "traceback": result.traceback,
                    })
                for data in result.outputs:
                    cell_outputs.append({
                        "output_type": "execute_result",
                        "data": data,
                        "metadata": {},
                        "execution_count": None,
                    })
                cell["outputs"] = cell_outputs

            if not result.success:
                logger.warning(f"Cell {i + 1} failed: {result.error_name}")
                break

        # Save executed notebook
        if save:
            out_path = path.replace(".ipynb", "_executed.ipynb")
            with open(out_path, "w") as f:
                json.dump(nb, f, indent=2)
            logger.info(f"Executed notebook saved to {out_path}")

        return results

    def upload(self, local_path: str, remote_path: Optional[str] = None) -> ColabResult:
        """
        Upload a local file to the remote kernel's filesystem
        using base64 encoding over code execution.
        """
        local_path = os.path.abspath(local_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        if remote_path is None:
            remote_path = f"/content/{os.path.basename(local_path)}"

        with open(local_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")

        # Chunk the base64 string to avoid message-size limits
        chunk_size = 1_000_000  # 1MB chunks
        chunks = [data[i: i + chunk_size] for i in range(0, len(data), chunk_size)]

        code = f"import base64, os\n"
        code += f"_data = ''\n"
        for chunk in chunks:
            code += f"_data += '{chunk}'\n"
        code += f"os.makedirs(os.path.dirname('{remote_path}') or '.', exist_ok=True)\n"
        code += f"with open('{remote_path}', 'wb') as _f:\n"
        code += f"    _f.write(base64.b64decode(_data))\n"
        code += f"print(f'Uploaded {{len(base64.b64decode(_data))}} bytes to {remote_path}')\n"

        logger.info(f"Uploading {local_path} → {remote_path} ({len(data)} b64 chars)")
        return self.execute(code)

    def download(self, remote_path: str, local_path: Optional[str] = None) -> ColabResult:
        """
        Download a file from the remote kernel's filesystem
        using base64 encoding over code execution.
        """
        if local_path is None:
            local_path = os.path.basename(remote_path)

        code = f"""
import base64
with open('{remote_path}', 'rb') as _f:
    _encoded = base64.b64encode(_f.read()).decode('ascii')
print("__ADAMOPS_FILE__" + _encoded)
"""
        result = self.execute(code)

        if result.success:
            for line in result.stdout.split("\n"):
                if line.startswith("__ADAMOPS_FILE__"):
                    b64_data = line[len("__ADAMOPS_FILE__"):]
                    raw = base64.b64decode(b64_data)
                    os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(raw)
                    logger.info(f"Downloaded {remote_path} → {local_path} ({len(raw)} bytes)")
                    break

        return result

    def run_pipeline(self, pipeline_data: Dict, timeout: int = 300) -> ColabResult:
        """
        Compile a Studio DAG pipeline to Python code and execute
        it on the remote Colab kernel.
        """
        from adamops.studio.compiler import compile_pipeline

        code = compile_pipeline(pipeline_data)
        logger.info(f"Compiled pipeline ({len(code)} chars), sending to Colab")
        return self.execute(code, timeout=timeout)
