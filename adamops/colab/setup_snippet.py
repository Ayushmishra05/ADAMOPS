"""
AdamOps Colab Bridge — Setup Snippet

Provides the code snippet that users paste into a Google Colab cell
to start a Jupyter kernel gateway, enabling remote execution from
their local machine.
"""


COLAB_SETUP_SNIPPET = r'''
# ====================================================================
#   AdamOps Colab Bridge — Paste this into a Colab cell and run it
# ====================================================================
#
# This starts a Jupyter kernel gateway on the Colab runtime so your
# local ADAMOPS installation can send .py / .ipynb files to this GPU.
#
# After running, copy the printed URL and token into your local code:
#
#   from adamops.colab import ColabBridge
#   bridge = ColabBridge("<URL>", token="<TOKEN>")
#   result = bridge.run_script("train.py")
# ====================================================================

import subprocess, sys, os, secrets, json

# 1. Install kernel gateway
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
                       "jupyter_kernel_gateway"])

# 2. Generate a security token
TOKEN = secrets.token_hex(24)

# 3. Detect environment
IN_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

# 4. Start the gateway in background
import threading, time

def _start_gateway():
    subprocess.run([
        sys.executable, "-m", "jupyter", "kernelgateway",
        "--KernelGatewayApp.ip=0.0.0.0",
        "--KernelGatewayApp.port=8888",
        f"--KernelGatewayApp.auth_token={TOKEN}",
        "--KernelGatewayApp.allow_origin=*",
    ])

gw_thread = threading.Thread(target=_start_gateway, daemon=True)
gw_thread.start()
time.sleep(3)  # wait for startup

# 5. Get public URL
if IN_COLAB:
    try:
        from google.colab.output import eval_js
        public_url = eval_js("google.colab.kernel.proxyPort(8888)")
        if not public_url:
            raise RuntimeError("proxy failed")
    except Exception:
        # Fallback to pyngrok
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "pyngrok"])
        from pyngrok import ngrok
        public_url = ngrok.connect(8888, "http").public_url
else:
    public_url = "http://localhost:8888"

# 6. Print connection info
print("\n" + "=" * 60)
print("  AdamOps Colab Bridge — READY")
print("=" * 60)
print(f"  Gateway URL : {public_url}")
print(f"  Token       : {TOKEN}")
print()
print("  Connect from your local machine:")
print(f'    from adamops.colab import ColabBridge')
print(f'    bridge = ColabBridge("{public_url}", token="{TOKEN}")')
print(f'    print(bridge.status())')
print("=" * 60 + "\n")

# 7. GPU info
try:
    import torch
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM: {mem:.1f} GB")
except ImportError:
    pass
'''.strip()


def setup_colab():
    """Print the Colab setup snippet to the console for easy copy-paste."""
    print("Copy and paste the following into a Google Colab cell:\n")
    print("-" * 60)
    print(COLAB_SETUP_SNIPPET)
    print("-" * 60)
    print("\nAfter running it in Colab, use the printed URL and token to connect.")


def get_setup_snippet() -> str:
    """Return the raw setup snippet string."""
    return COLAB_SETUP_SNIPPET
