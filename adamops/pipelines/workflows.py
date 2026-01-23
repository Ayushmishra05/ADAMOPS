"""
AdamOps Workflows Module

Define ML workflows as DAGs.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import traceback

from adamops.utils.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task:
    """Represents a single task in a workflow."""
    
    def __init__(self, name: str, func: Callable, dependencies: Optional[List[str]] = None,
                 retry: int = 0, timeout: Optional[int] = None):
        self.name = name
        self.func = func
        self.dependencies = dependencies or []
        self.retry = retry
        self.timeout = timeout
        self.status = TaskStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def run(self, context: Dict) -> Any:
        """Execute the task."""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
        
        attempts = 0
        while attempts <= self.retry:
            try:
                self.result = self.func(context)
                self.status = TaskStatus.COMPLETED
                self.end_time = datetime.now()
                logger.info(f"Task '{self.name}' completed in {self.duration:.2f}s")
                return self.result
            except Exception as e:
                attempts += 1
                if attempts > self.retry:
                    self.status = TaskStatus.FAILED
                    self.error = str(e)
                    self.end_time = datetime.now()
                    logger.error(f"Task '{self.name}' failed: {e}")
                    raise
                logger.warning(f"Task '{self.name}' failed, retrying ({attempts}/{self.retry})")
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "duration": self.duration,
            "error": self.error,
        }


class Workflow:
    """DAG-based workflow for ML pipelines."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tasks: Dict[str, Task] = {}
        self.context: Dict = {}
        self.status = TaskStatus.PENDING
    
    def add_task(self, name: str, func: Callable, dependencies: Optional[List[str]] = None,
                 **kwargs) -> "Workflow":
        """Add a task to the workflow."""
        task = Task(name, func, dependencies, **kwargs)
        self.tasks[name] = task
        return self
    
    def task(self, name: str = None, dependencies: Optional[List[str]] = None, **kwargs):
        """Decorator to add a task."""
        def decorator(func):
            task_name = name or func.__name__
            self.add_task(task_name, func, dependencies, **kwargs)
            return func
        return decorator
    
    def _get_execution_order(self) -> List[str]:
        """Topological sort for task execution order."""
        visited = set()
        order = []
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            task = self.tasks[name]
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"Unknown dependency: {dep}")
                visit(dep)
            
            order.append(name)
        
        for name in self.tasks:
            visit(name)
        
        return order
    
    def run(self, initial_context: Optional[Dict] = None) -> Dict:
        """Execute the workflow."""
        self.context = initial_context or {}
        self.status = TaskStatus.RUNNING
        
        logger.info(f"Starting workflow: {self.name}")
        start_time = datetime.now()
        
        try:
            execution_order = self._get_execution_order()
            
            for task_name in execution_order:
                task = self.tasks[task_name]
                
                # Check dependencies
                deps_ok = all(
                    self.tasks[dep].status == TaskStatus.COMPLETED 
                    for dep in task.dependencies
                )
                
                if not deps_ok:
                    task.status = TaskStatus.SKIPPED
                    logger.warning(f"Skipping '{task_name}' due to failed dependencies")
                    continue
                
                # Run task
                result = task.run(self.context)
                self.context[task_name] = result
            
            self.status = TaskStatus.COMPLETED
            logger.info(f"Workflow '{self.name}' completed in {(datetime.now() - start_time).total_seconds():.2f}s")
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.error(f"Workflow '{self.name}' failed: {e}")
            raise
        
        return self.context
    
    def get_status(self) -> Dict:
        """Get workflow status."""
        return {
            "name": self.name,
            "status": self.status.value,
            "tasks": {name: task.to_dict() for name, task in self.tasks.items()},
        }
    
    def visualize(self) -> str:
        """Generate ASCII visualization of workflow."""
        lines = [f"Workflow: {self.name}", "=" * 40]
        
        for name in self._get_execution_order():
            task = self.tasks[name]
            deps = ", ".join(task.dependencies) if task.dependencies else "None"
            status = task.status.value.upper()
            lines.append(f"  [{status}] {name} <- {deps}")
        
        return "\n".join(lines)


def create_ml_pipeline(name: str = "ml_pipeline") -> Workflow:
    """Create a standard ML pipeline workflow."""
    workflow = Workflow(name, "Standard ML Training Pipeline")
    
    @workflow.task("load_data")
    def load_data(ctx):
        logger.info("Loading data...")
        return ctx.get("data_path")
    
    @workflow.task("preprocess", dependencies=["load_data"])
    def preprocess(ctx):
        logger.info("Preprocessing data...")
        return {"preprocessed": True}
    
    @workflow.task("train", dependencies=["preprocess"])
    def train(ctx):
        logger.info("Training model...")
        return {"model": "trained"}
    
    @workflow.task("evaluate", dependencies=["train"])
    def evaluate(ctx):
        logger.info("Evaluating model...")
        return {"metrics": {"accuracy": 0.95}}
    
    return workflow
