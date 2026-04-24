"""
AdamOps Hardware Abstraction Layer

Isolates device logic to prevent CUDA pollution in core training loops.
Handles VRAM pre-flight checks, Colab environment detection, and
ephemeral storage protection.
"""

import sys
import os
import gc
from typing import Optional, Tuple, Dict
from adamops.utils.logging import get_logger

logger = get_logger("adamops.hardware")

class DeviceManager:
    """Manages optimal hardware placement and safety validations."""

    @staticmethod
    def is_colab_environment() -> bool:
        """Detects if adamops is running inside a Google Colab notebook."""
        return "google.colab" in sys.modules or "COLAB_GPU" in os.environ

    @staticmethod
    def get_optimal_device() -> str:
        """
        Auto-detect the best available compute device.
        Returns 'cuda', 'mps', or 'cpu'.
        """
        try:
            import torch
        except ImportError:
            return "cpu"

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Hardware provisioned: GPU ({gpu_name})")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Hardware provisioned: Apple Silicon (MPS)")
            return "mps"
        
        if DeviceManager.is_colab_environment():
            logger.warning("Hardware provisioned: CPU. Training will be severely bottlenecked! Check Colab Runtime > Change runtime type > Hardware accelerator.")
        else:
            logger.info("Hardware provisioned: CPU")
        
        return "cpu"

    @staticmethod
    def get_num_gpus() -> int:
        """Returns the number of available CUDA GPUs for multi-GPU training."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 0

    @staticmethod
    def vram_preflight_check(required_mb: float) -> bool:
        """
        Checks if the required VRAM is available to prevent CUDA OutOfMemoryError.
        Always returns True if on CPU or MPS.
        """
        try:
            import torch
        except ImportError:
            return True

        if not torch.cuda.is_available():
            return True

        try:
            # Clear cache to get accurate free memory
            torch.cuda.empty_cache()
            gc.collect()

            # free, total
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mb = free_mem / (1024 ** 2)

            if required_mb > free_mb:
                logger.error(
                    f"VRAM Pre-flight Failed! Required: {required_mb:.1f} MB, "
                    f"Free: {free_mb:.1f} MB. Requesting this memory would crash the Colab kernel."
                )
                return False
            
            # Safe boundary check (within 10%)
            if required_mb > (free_mb * 0.9):
                logger.warning(
                    f"VRAM is tight. Required: {required_mb:.1f} MB, Free: {free_mb:.1f} MB. "
                    f"You may experience fragmentation failures."
                )
            
            return True

        except Exception as e:
            logger.warning(f"Unable to perform VRAM check: {e}")
            return True

    @staticmethod
    def estimate_model_vram(model, input_shape=None) -> float:
        """
        Estimates the memory required by a model in Megabytes.
        Accounts for parameters, gradients, and optimizer states.
        """
        try:
            import torch
        except ImportError:
            return 0.0

        if not isinstance(model, torch.nn.Module):
            # Not a PyTorch model, typically sklearn or tree ensembles which use system RAM
            return 0.0

        # Estimate params (float32 = 4 bytes per param)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

        total_model_bytes = param_bytes + buffer_bytes
        
        # Training rule of thumb: 
        # Weights (1x) + Gradients (1x) + Optimizer state (Adam is 2x) = ~4x model size
        training_bytes = total_model_bytes * 4

        # Activations (roughly similar to model size depending on batch size, fallback generic 1x)
        # Using a very conservative estimate if we don't know the exact batch size graph layout.
        activation_bytes = total_model_bytes

        total_mb = (training_bytes + activation_bytes) / (1024 ** 2)
        return total_mb

    @staticmethod
    def get_checkpoint_dir(project_name: str = "adamops_runs") -> str:
        """
        Determines the safest persistence directory for model artifacts.
        In Colab, this auto-detects Google Drive explicitly to prevent ephemeral data loss.
        """
        if DeviceManager.is_colab_environment():
            drive_path = "/content/drive/MyDrive"
            if os.path.exists(drive_path):
                colab_path = os.path.join(drive_path, project_name)
                os.makedirs(colab_path, exist_ok=True)
                return colab_path
            else:
                logger.warning(
                    "Google Drive is NOT mounted. Checkpoints will be mapped to ephemeral "
                    "Colab disk space and destroyed when the session disconnects. "
                    "Run `from google.colab import drive; drive.mount('/content/drive')` to persist."
                )
                
        # Default local fallback
        local_path = os.path.abspath(f"./checkpoints/{project_name}")
        os.makedirs(local_path, exist_ok=True)
        return local_path
