"""
AdamOps CLI Module

Command-line interface for AdamOps.
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import click
    from rich.console import Console
    from rich.table import Table
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

if CLICK_AVAILABLE:
    console = Console()
    
    @click.group()
    @click.version_option(version="0.1.1", prog_name="adamops")
    def main():
        """AdamOps - MLOps made simple."""
        pass
    
    @main.command()
    @click.option("--data", "-d", required=True, help="Path to data file")
    @click.option("--target", "-t", required=True, help="Target column name")
    @click.option("--algorithm", "-a", default="auto", help="Algorithm to use")
    @click.option("--task", default="auto", help="Task type: classification, regression, auto")
    @click.option("--output", "-o", default="model.joblib", help="Output model path")
    def train(data: str, target: str, algorithm: str, task: str, output: str):
        """Train a model."""
        console.print(f"[bold blue]Loading data from {data}...[/]")
        
        from adamops.data.loaders import load_auto
        from adamops.models.automl import quick_run
        from adamops.models.modelops import train as train_model
        from adamops.deployment.exporters import export_joblib
        
        df = load_auto(data)
        X = df.drop(columns=[target])
        y = df[target]
        
        console.print(f"[bold blue]Training {algorithm} model...[/]")
        
        if algorithm == "auto":
            model = quick_run(X, y, task)
        else:
            model = train_model(X, y, task, algorithm)
        
        export_joblib(model, output)
        console.print(f"[bold green]Model saved to {output}[/]")
    
    @main.command()
    @click.option("--model", "-m", required=True, help="Path to model file")
    @click.option("--data", "-d", required=True, help="Path to test data")
    @click.option("--target", "-t", required=True, help="Target column name")
    def evaluate(model: str, data: str, target: str):
        """Evaluate a model."""
        from adamops.deployment.exporters import load_model
        from adamops.data.loaders import load_auto
        from adamops.evaluation.metrics import evaluate as eval_metrics
        
        console.print(f"[bold blue]Loading model and data...[/]")
        
        model_obj = load_model(model)
        df = load_auto(data)
        X = df.drop(columns=[target])
        y = df[target]
        
        y_pred = model_obj.predict(X)
        metrics = eval_metrics(y, y_pred)
        
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for name, value in metrics.items():
            if isinstance(value, float):
                table.add_row(name, f"{value:.4f}")
            else:
                table.add_row(name, str(value))
        
        console.print(table)
    
    @main.command()
    @click.option("--model", "-m", required=True, help="Path to model file")
    @click.option("--type", "deploy_type", default="api", help="Deployment type: api, docker")
    @click.option("--port", "-p", default=8000, help="API port")
    @click.option("--output", "-o", default="./deploy", help="Output directory for docker")
    def deploy(model: str, deploy_type: str, port: int, output: str):
        """Deploy a model."""
        from adamops.deployment.exporters import load_model
        
        model_obj = load_model(model)
        
        if deploy_type == "api":
            from adamops.deployment.api import run_api
            console.print(f"[bold blue]Starting API on port {port}...[/]")
            run_api(model_obj, port=port)
        
        elif deploy_type == "docker":
            from adamops.deployment.containerize import containerize
            console.print(f"[bold blue]Creating Docker deployment...[/]")
            result = containerize(model, output)
            console.print(f"[bold green]Files created in {output}[/]")
            for name, path in result.items():
                console.print(f"  - {name}: {path}")
    
    @main.command()
    @click.option("--data", "-d", required=True, help="Path to data file")
    def validate(data: str):
        """Validate a data file."""
        from adamops.data.loaders import load_auto
        from adamops.data.validators import validate as validate_data
        
        console.print(f"[bold blue]Validating {data}...[/]")
        
        df = load_auto(data)
        report = validate_data(df)
        
        console.print(report.summary())
    
    @main.command()
    @click.argument("workflow_name")
    def run_workflow(workflow_name: str):
        """Run a predefined workflow."""
        from adamops.pipelines.workflows import create_ml_pipeline
        
        console.print(f"[bold blue]Running workflow: {workflow_name}[/]")
        
        workflow = create_ml_pipeline(workflow_name)
        result = workflow.run()
        
        console.print(f"[bold green]Workflow completed![/]")
        console.print(workflow.get_status())
    
    @main.command()
    def info():
        """Show AdamOps information."""
        from adamops import __version__
        
        console.print("[bold blue]AdamOps - MLOps Made Simple[/]")
        console.print(f"Version: {__version__}")
        console.print()
        console.print("Available commands:")
        console.print("  train        - Train a model")
        console.print("  evaluate     - Evaluate a model")
        console.print("  deploy       - Deploy a model")
        console.print("  validate     - Validate data")
        console.print("  run-workflow - Run a workflow")
        console.print("  studio       - Launch visual pipeline builder")

    @main.command()
    @click.option("--host", default="127.0.0.1", help="Server host")
    @click.option("--port", "-p", default=5555, type=int, help="Server port")
    @click.option("--no-browser", is_flag=True, help="Don't open browser")
    def studio(host, port, no_browser):
        """Launch the visual pipeline builder."""
        from adamops.studio import launch
        console.print("[bold magenta]Launching AdamOps Studio...[/]")
        launch(host=host, port=port, open_browser=not no_browser)

    @main.command()
    @click.option("--host", default="127.0.0.1", help="Server host")
    @click.option("--port", "-p", default=8000, type=int, help="Server port")
    @click.option("--db", default="adamops_state.db", help="Path to SQLite state database")
    def server(host, port, db):
        """Start the AdamOps backend engine (FastAPI + WebSocket)."""
        from adamops.backend.launcher import launch as launch_server
        console.print("[bold cyan]Starting AdamOps Backend Engine...[/]")
        launch_server(host=host, port=port, db_path=db)

    @main.command(name="colab-setup")
    def colab_setup():
        """Print the Colab setup snippet to paste into a Colab cell."""
        from adamops.colab.setup_snippet import setup_colab
        setup_colab()

    @main.command(name="colab-run")
    @click.argument("file_path")
    @click.option("--url", required=True, help="Colab kernel gateway URL")
    @click.option("--token", default="", help="Gateway authentication token")
    @click.option("--timeout", default=300, type=int, help="Execution timeout in seconds")
    def colab_run(file_path, url, token, timeout):
        """Execute a .py or .ipynb file on a remote Colab GPU."""
        from adamops.colab.bridge import ColabBridge
        bridge = ColabBridge(url, token=token)

        console.print(f"[bold cyan]Connecting to Colab at {url}...[/]")
        status = bridge.status()
        gpu = status.get("gpu_name", "CPU")
        console.print(f"[green]Connected! Runtime: {gpu}[/]")

        if file_path.endswith(".ipynb"):
            console.print(f"[bold blue]Running notebook: {file_path}[/]")
            results = bridge.run_notebook(file_path, timeout=timeout)
            passed = sum(1 for r in results if r.success)
            console.print(f"[bold green]Done: {passed}/{len(results)} cells passed[/]")
            for i, r in enumerate(results):
                if r.stdout.strip():
                    console.print(f"[dim]Cell {i+1} stdout:[/] {r.stdout.strip()[:200]}")
                if not r.success:
                    console.print(f"[bold red]Cell {i+1} error: {r.error_name}: {r.error_value}[/]")
        else:
            console.print(f"[bold blue]Running script: {file_path}[/]")
            result = bridge.run_script(file_path, timeout=timeout)
            if result.success:
                console.print(f"[bold green]✅ Completed in {result.elapsed:.2f}s[/]")
            else:
                console.print(f"[bold red]❌ {result.error_name}: {result.error_value}[/]")
            if result.stdout.strip():
                console.print(result.stdout)

else:
    def main():
        print("CLI requires click and rich. Install with: pip install click rich")
        sys.exit(1)


if __name__ == "__main__":
    main()
