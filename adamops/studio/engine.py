"""
AdamOps Studio — Pipeline Execution Engine

Compiles a node graph (from the UI) into an executable pipeline,
runs it using AdamOps functions, and returns results.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import traceback

from adamops.studio.nodes import get_node_type
from adamops.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineError(Exception):
    """Raised when pipeline execution fails."""
    pass


class ExecutionResult:
    """Result from executing a pipeline."""
    def __init__(self):
        self.node_results: Dict[str, Dict] = {}
        self.node_status: Dict[str, str] = {}  # "success", "error", "pending"
        self.node_errors: Dict[str, str] = {}
        self.node_times: Dict[str, float] = {}
        self.logs: List[str] = []
        self.final_metrics: Optional[Dict] = None
        self.success: bool = False
        self.total_time: float = 0.0

    def to_dict(self) -> Dict:
        serialized_results = {}
        for node_id, outputs in self.node_results.items():
            serialized_results[node_id] = {}
            for key, val in outputs.items():
                if hasattr(val, 'shape'):
                    serialized_results[node_id][key] = {
                        "type": type(val).__name__,
                        "shape": list(val.shape) if hasattr(val.shape, '__iter__') else [val.shape],
                    }
                    # Add preview for dataframes
                    if hasattr(val, 'columns'):
                        serialized_results[node_id][key]["columns"] = list(val.columns[:20])
                        serialized_results[node_id][key]["head"] = val.head(5).to_dict(orient="records")
                elif isinstance(val, dict):
                    # For metrics dicts, serialize the values
                    serialized_results[node_id][key] = {
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in val.items()
                    }
                else:
                    serialized_results[node_id][key] = {
                        "type": type(val).__name__,
                        "repr": str(val)[:200],
                    }

        return {
            "success": self.success,
            "total_time": round(self.total_time, 3),
            "node_status": self.node_status,
            "node_errors": self.node_errors,
            "node_times": {k: round(v, 3) for k, v in self.node_times.items()},
            "node_results": serialized_results,
            "final_metrics": self.final_metrics,
            "logs": self.logs,
        }


def _topological_sort(nodes: List[Dict], connections: List[Dict]) -> List[str]:
    """
    Sort nodes by dependency order.
    
    nodes: [{"id": "node_1", "type": "load_csv", ...}, ...]
    connections: [{"from_node": "node_1", "from_port": "dataframe",
                   "to_node": "node_2", "to_port": "dataframe"}, ...]
    """
    # Build adjacency: node_id -> list of node_ids that depend on it
    dependents: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    dep_count: Dict[str, int] = {n["id"]: 0 for n in nodes}
    
    for conn in connections:
        from_id = conn["from_node"]
        to_id = conn["to_node"]
        if from_id in dependents:
            dependents[from_id].append(to_id)
        if to_id in dep_count:
            dep_count[to_id] += 1
    
    # Kahn's algorithm
    queue = [nid for nid, count in dep_count.items() if count == 0]
    order = []
    
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for dep in dependents.get(nid, []):
            dep_count[dep] -= 1
            if dep_count[dep] == 0:
                queue.append(dep)
    
    if len(order) != len(nodes):
        raise PipelineError("Pipeline has circular dependencies")
    
    return order


def _resolve_inputs(node_id: str, node_type_id: str, connections: List[Dict],
                    all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Gather input values for a node by following incoming connections.
    """
    from adamops.studio.nodes import get_node_type
    
    inputs = {}
    for conn in connections:
        if conn["to_node"] == node_id:
            from_node = conn["from_node"]
            from_port = conn["from_port"]
            to_port = conn["to_port"]
            
            if from_node in all_results and from_port in all_results[from_node]:
                inputs[to_port] = all_results[from_node][from_port]
    
    return inputs


def execute_pipeline(pipeline_data: Dict) -> ExecutionResult:
    """
    Execute a pipeline from the UI.
    
    pipeline_data: {
        "nodes": [{"id": "node_1", "type": "load_csv", "params": {...}, "x": 100, "y": 200}, ...],
        "connections": [{"from_node": "...", "from_port": "...", "to_node": "...", "to_port": "..."}, ...]
    }
    """
    result = ExecutionResult()
    start_time = datetime.now()
    
    nodes = pipeline_data.get("nodes", [])
    connections = pipeline_data.get("connections", [])
    
    if not nodes:
        result.logs.append("No nodes in pipeline")
        return result
    
    # Sort by dependencies
    try:
        execution_order = _topological_sort(nodes, connections)
    except PipelineError as e:
        result.logs.append(f"Error: {e}")
        return result
    
    # Build lookup
    node_map = {n["id"]: n for n in nodes}
    all_outputs: Dict[str, Dict] = {}
    
    result.logs.append(f"Executing {len(nodes)} nodes...")
    
    for node_id in execution_order:
        node_data = node_map[node_id]
        node_type_id = node_data["type"]
        node_params = node_data.get("params", {})
        
        node_type = get_node_type(node_type_id)
        if node_type is None:
            result.node_status[node_id] = "error"
            result.node_errors[node_id] = f"Unknown node type: {node_type_id}"
            result.logs.append(f"✗ {node_type_id} — unknown type")
            continue
        
        # Resolve inputs from connections
        inputs = _resolve_inputs(node_id, node_type_id, connections, all_outputs)
        
        # Execute
        node_start = datetime.now()
        try:
            result.logs.append(f"▶ Running: {node_type.label}")
            outputs = node_type.execute_fn(inputs, node_params)
            
            all_outputs[node_id] = outputs
            result.node_results[node_id] = outputs
            result.node_status[node_id] = "success"
            
            elapsed = (datetime.now() - node_start).total_seconds()
            result.node_times[node_id] = elapsed
            result.logs.append(f"✓ {node_type.label} ({elapsed:.2f}s)")
            
            # If this node outputs metrics, save as final metrics
            if "metrics" in outputs and isinstance(outputs["metrics"], dict):
                result.final_metrics = outputs["metrics"]
                
        except Exception as e:
            elapsed = (datetime.now() - node_start).total_seconds()
            result.node_status[node_id] = "error"
            result.node_errors[node_id] = str(e)
            result.node_times[node_id] = elapsed
            result.logs.append(f"✗ {node_type.label} — {e}")
            logger.error(f"Node {node_id} ({node_type_id}) failed: {traceback.format_exc()}")
            # Stop execution on error
            break
    
    result.total_time = (datetime.now() - start_time).total_seconds()
    result.success = all(s == "success" for s in result.node_status.values())
    
    if result.success:
        result.logs.append(f"Pipeline completed successfully in {result.total_time:.2f}s")
    else:
        result.logs.append(f"Pipeline failed after {result.total_time:.2f}s")
    
    return result
