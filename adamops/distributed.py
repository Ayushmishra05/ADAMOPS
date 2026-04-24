"""
AdamOps Distributed Training Layer

Orchestrates distributed execution depending on available hardware.
Wraps PyTorch DistributedDataParallel (DDP) for Multi-GPU single-node training,
Ray Train for multi-node elastic clusters, and Joblib for async sklearn bagging.
"""

import os
from typing import Callable, Any, Dict, List
from adamops.utils.logging import get_logger
from adamops.hardware import DeviceManager

logger = get_logger("adamops.distributed")

class DistributedManager:
    """Manages parallelization logic for varying hardware types."""

    @staticmethod
    def auto_strategy(backend: str = "pytorch") -> str:
        """
        Calculates the best execution strategy dynamically based on present resources.
        Returns 'single', 'ddp' (Multi-GPU), or 'ray' (Multi-node).
        """
        # If running in Colab, multi-node Ray makes little sense without paid clusters,
        # but single or DDP makes sense. We stick to single/ddp for auto-select generally
        # unless ray is explicitly provisioned in the backend.
        
        num_gpus = DeviceManager.get_num_gpus()
        if num_gpus > 1 and backend == "pytorch":
            logger.info(f"Auto-strategy selected 'ddp' (DistributedDataParallel) leveraging {num_gpus} GPUs.")
            return "ddp"
        
        try:
            import ray
            if ray.is_initialized():
                logger.info("Auto-strategy selected 'ray' due to active Ray cluster.")
                return "ray"
        except ImportError:
            pass
        
        if backend == "sklearn":
            logger.info("Auto-strategy selected 'joblib' local multithreading for sklearn backend.")
            return "joblib"

        logger.info("Auto-strategy selected 'single' (Standard Single Worker execution).")
        return "single"


    # --- Sklearn Parallel Processing ---

    @staticmethod
    def parallel_sklearn_train(train_funcs: List[Callable], n_jobs: int = -1) -> List[Any]:
        """
        Executes a list of generic training functions using joblib.
        Useful for grid searches, cross validation, or Ensembles (like bagging).
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.error("Joblib not installed. Cannot run parallel sklearn training.")
            return [f() for f in train_funcs]

        logger.info(f"Dispatching {len(train_funcs)} scikit-learn tasks to {n_jobs if n_jobs > 0 else 'all'} CPU threads.")
        
        # Depending on environment, we might use multiprocessing or threading. 
        # For Colab, loky (multiprocessing backend) is preferred for CPU tasks.
        return Parallel(n_jobs=n_jobs, backend="loky")(delayed(f)() for f in train_funcs)


    # --- PyTorch DDP Execution ---

    @staticmethod
    def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
        """
        Initializes PyTorch Distributed process groups.
        Expects rank, environment variables mapped, and world size constraints.
        backend should be 'nccl' for GPUs, 'gloo' for CPUs.
        """
        try:
            import torch
            import torch.distributed as dist
        except ImportError:
            return

        if sys.platform == "win32" and backend == "nccl":
            logger.warning("NCCL Backend not supported on Windows. Falling back to gloo.")
            backend = "gloo"

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    @staticmethod
    def wrap_ddp(model: Any, device: str) -> Any:
        """
        Wraps a PyTorch module intelligently using the correct DDP arguments 
        based on the environment and targeted device mappings.
        """
        try:
            import torch
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            return model
            
        if not isinstance(model, torch.nn.Module):
            return model
            
        logger.info(f"Wrapping model with DistributedDataParallel targeting device {device}.")
        if device == "cuda":
            # Assigning specific GPU IDs
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return DDP(model, device_ids=[local_rank])
        else:
            return DDP(model)

    @staticmethod
    def cleanup_ddp():
        """Cleans up the PyTorch DDP execution process."""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except ImportError:
            pass


    # --- Ray Train (Multi-Node / Multi-GPU PyTorch) ---

    @staticmethod
    def ray_distributed_train(train_fn: Callable, config: Dict, num_workers: int = 2, use_gpu: bool = False):
        """
        Wraps a training function natively into a Ray Train cluster.
        Scales the loop across multiple machines if a ray cluster is active.
        """
        try:
            import ray
            from ray import train as ray_train
            from ray.train import ScalingConfig
            from ray.train.torch import TorchTrainer
        except ImportError:
            logger.error("Ray[train] not installed. Use 'pip install adamops[distributed]' for Multi-Node Ray execution.")
            return None

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        logger.info(f"Dispatching Ray TorchTrainer: Workers={num_workers}, GPU_Enabled={use_gpu}.")
        
        trainer = TorchTrainer(
            train_loop_per_worker=train_fn,
            train_loop_config=config,
            scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        )
        
        result = trainer.fit()
        return result
