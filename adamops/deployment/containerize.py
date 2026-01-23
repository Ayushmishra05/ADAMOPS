"""
AdamOps Containerization Module

Docker and Kubernetes deployment support.
"""

from typing import Dict, List, Optional
from pathlib import Path

from adamops.utils.logging import get_logger
from adamops.utils.helpers import ensure_dir

logger = get_logger(__name__)


DOCKERFILE_TEMPLATE = '''# Auto-generated Dockerfile for AdamOps model
FROM python:{python_version}-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE {port}

# Run application
CMD ["python", "{entrypoint}"]
'''

REQUIREMENTS_TEMPLATE = '''# Auto-generated requirements for model serving
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
joblib>=1.2.0
{framework_deps}
'''

K8S_DEPLOYMENT_TEMPLATE = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}-deployment
  labels:
    app: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: {image}
        ports:
        - containerPort: {port}
        resources:
          requests:
            memory: "{memory}"
            cpu: "{cpu}"
          limits:
            memory: "{memory_limit}"
            cpu: "{cpu_limit}"
---
apiVersion: v1
kind: Service
metadata:
  name: {name}-service
spec:
  selector:
    app: {name}
  ports:
  - port: {port}
    targetPort: {port}
  type: {service_type}
'''


def generate_dockerfile(
    output_dir: str, entrypoint: str = "app.py",
    python_version: str = "3.10", port: int = 8000,
    framework: str = "fastapi"
) -> str:
    """
    Generate Dockerfile for model serving.
    
    Args:
        output_dir: Output directory.
        entrypoint: Python entrypoint file.
        python_version: Python version.
        port: Exposed port.
        framework: 'fastapi' or 'flask'.
    
    Returns:
        Path to Dockerfile.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
        python_version=python_version,
        port=port,
        entrypoint=entrypoint
    )
    
    dockerfile_path = output_dir / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    
    # Generate requirements
    framework_deps = "fastapi>=0.100.0\nuvicorn>=0.22.0" if framework == "fastapi" else "flask>=2.3.0"
    
    requirements_content = REQUIREMENTS_TEMPLATE.format(framework_deps=framework_deps)
    
    requirements_path = output_dir / "requirements.txt"
    with open(requirements_path, "w") as f:
        f.write(requirements_content)
    
    logger.info(f"Generated Dockerfile at {dockerfile_path}")
    return str(dockerfile_path)


def generate_docker_compose(
    output_dir: str, service_name: str = "model-api",
    port: int = 8000, image: Optional[str] = None
) -> str:
    """Generate docker-compose.yml."""
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    content = f'''version: "3.8"

services:
  {service_name}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - PORT={port}
    restart: unless-stopped
'''
    
    if image:
        content = content.replace("build: .", f"image: {image}")
    
    filepath = output_dir / "docker-compose.yml"
    with open(filepath, "w") as f:
        f.write(content)
    
    logger.info(f"Generated docker-compose.yml at {filepath}")
    return str(filepath)


def generate_k8s_manifests(
    output_dir: str, name: str = "model-api",
    image: str = "model-api:latest", port: int = 8000,
    replicas: int = 2, memory: str = "512Mi", cpu: str = "250m",
    service_type: str = "LoadBalancer"
) -> str:
    """
    Generate Kubernetes deployment manifests.
    
    Args:
        output_dir: Output directory.
        name: Deployment name.
        image: Docker image.
        port: Container port.
        replicas: Number of replicas.
        memory: Memory request.
        cpu: CPU request.
        service_type: K8s service type.
    
    Returns:
        Path to manifest file.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    content = K8S_DEPLOYMENT_TEMPLATE.format(
        name=name, image=image, port=port, replicas=replicas,
        memory=memory, cpu=cpu, memory_limit=memory, cpu_limit=cpu,
        service_type=service_type
    )
    
    filepath = output_dir / "k8s-deployment.yaml"
    with open(filepath, "w") as f:
        f.write(content)
    
    logger.info(f"Generated K8s manifests at {filepath}")
    return str(filepath)


def build_docker_image(
    context_dir: str, image_name: str, tag: str = "latest"
) -> bool:
    """Build Docker image (requires Docker CLI)."""
    import subprocess
    
    full_tag = f"{image_name}:{tag}"
    cmd = ["docker", "build", "-t", full_tag, context_dir]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Built Docker image: {full_tag}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker build failed: {e.stderr}")
        return False


def containerize(
    model_path: str, output_dir: str, name: str = "model-api",
    framework: str = "fastapi", port: int = 8000, build: bool = False
) -> Dict[str, str]:
    """
    Create complete containerization package.
    
    Returns:
        Dict with paths to generated files.
    """
    from adamops.deployment.api import generate_api_code
    
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    # Generate API code
    api_path = generate_api_code(model_path, output_dir / "app.py", framework, name)
    
    # Copy model
    import shutil
    model_dest = output_dir / Path(model_path).name
    shutil.copy(model_path, model_dest)
    
    # Generate Docker files
    dockerfile = generate_dockerfile(output_dir, "app.py", port=port, framework=framework)
    compose = generate_docker_compose(output_dir, name, port)
    k8s = generate_k8s_manifests(output_dir, name, f"{name}:latest", port)
    
    result = {
        "api": api_path,
        "dockerfile": dockerfile,
        "docker_compose": compose,
        "k8s": k8s,
        "model": str(model_dest),
    }
    
    if build:
        build_docker_image(str(output_dir), name)
    
    return result
