"""
AdamOps Orchestrators Module

Schedule and run pipelines.
"""

from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
import threading
import time
import json
from pathlib import Path

from adamops.utils.logging import get_logger
from adamops.utils.helpers import ensure_dir
from adamops.pipelines.workflows import Workflow, TaskStatus

logger = get_logger(__name__)


class PipelineRun:
    """Represents a single pipeline run."""
    
    def __init__(self, workflow: Workflow, run_id: str):
        self.workflow = workflow
        self.run_id = run_id
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = TaskStatus.PENDING
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "workflow": self.workflow.name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "error": self.error,
        }


class Orchestrator:
    """Orchestrate and schedule pipeline runs."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or ".adamops_runs")
        ensure_dir(self.storage_path)
        
        self.workflows: Dict[str, Workflow] = {}
        self.runs: Dict[str, PipelineRun] = {}
        self.schedules: Dict[str, Dict] = {}
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
    
    def register(self, workflow: Workflow):
        """Register a workflow."""
        self.workflows[workflow.name] = workflow
        logger.info(f"Registered workflow: {workflow.name}")
    
    def run(self, workflow_name: str, context: Optional[Dict] = None) -> PipelineRun:
        """Run a workflow immediately."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow = self.workflows[workflow_name]
        run_id = f"{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = PipelineRun(workflow, run_id)
        self.runs[run_id] = run
        
        run.start_time = datetime.now()
        run.status = TaskStatus.RUNNING
        
        try:
            run.result = workflow.run(context or {})
            run.status = TaskStatus.COMPLETED
        except Exception as e:
            run.status = TaskStatus.FAILED
            run.error = str(e)
        finally:
            run.end_time = datetime.now()
            self._save_run(run)
        
        return run
    
    def run_async(self, workflow_name: str, context: Optional[Dict] = None) -> str:
        """Run a workflow asynchronously."""
        run_id = f"{workflow_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        def _run():
            self.run(workflow_name, context)
        
        thread = threading.Thread(target=_run, name=run_id)
        thread.start()
        
        return run_id
    
    def schedule(self, workflow_name: str, interval_seconds: int,
                 context: Optional[Dict] = None):
        """Schedule a workflow to run at intervals."""
        self.schedules[workflow_name] = {
            "interval": interval_seconds,
            "context": context or {},
            "last_run": None,
            "next_run": datetime.now(),
        }
        logger.info(f"Scheduled {workflow_name} every {interval_seconds}s")
    
    def start_scheduler(self):
        """Start the background scheduler."""
        if self._running:
            return
        
        self._running = True
        
        def _scheduler_loop():
            while self._running:
                now = datetime.now()
                
                for name, schedule in self.schedules.items():
                    if now >= schedule["next_run"]:
                        try:
                            self.run(name, schedule["context"])
                        except Exception as e:
                            logger.error(f"Scheduled run failed: {e}")
                        
                        schedule["last_run"] = now
                        schedule["next_run"] = now + timedelta(seconds=schedule["interval"])
                
                time.sleep(1)
        
        self._scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Started scheduler")
    
    def stop_scheduler(self):
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("Stopped scheduler")
    
    def _save_run(self, run: PipelineRun):
        """Save run to storage."""
        run_file = self.storage_path / f"{run.run_id}.json"
        with open(run_file, 'w') as f:
            json.dump(run.to_dict(), f, indent=2)
    
    def get_runs(self, workflow_name: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get recent runs."""
        runs = list(self.runs.values())
        
        if workflow_name:
            runs = [r for r in runs if r.workflow.name == workflow_name]
        
        runs.sort(key=lambda r: r.start_time or datetime.min, reverse=True)
        return [r.to_dict() for r in runs[:limit]]
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get a specific run."""
        if run_id in self.runs:
            return self.runs[run_id].to_dict()
        return None


# Global orchestrator
_orchestrator: Optional[Orchestrator] = None

def get_orchestrator() -> Orchestrator:
    """Get global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
