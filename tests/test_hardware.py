"""
Tests for AdamOps Hardware Abstraction Layer.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock

from adamops.hardware import DeviceManager


def test_is_colab_environment():
    # Test not in colab initially ideally
    with patch.dict("sys.modules", {"google.colab": MagicMock()}):
        assert DeviceManager.is_colab_environment() is True

    with patch.dict("os.environ", {"COLAB_GPU": "1"}):
        assert DeviceManager.is_colab_environment() is True


@patch.dict("sys.modules", {"torch": MagicMock()})
def test_get_optimal_device():
    import sys
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "Tesla T4"
    assert DeviceManager.get_optimal_device() == "cuda"
    
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    assert DeviceManager.get_optimal_device() == "mps"
    
    mock_torch.backends.mps.is_available.return_value = False
    assert DeviceManager.get_optimal_device() == "cpu"


@patch.dict("sys.modules", {"torch": MagicMock()})
def test_get_num_gpus():
    import sys
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4
    assert DeviceManager.get_num_gpus() == 4
    
    mock_torch.cuda.is_available.return_value = False
    assert DeviceManager.get_num_gpus() == 0


@patch("adamops.hardware.DeviceManager.is_colab_environment", return_value=True)
@patch("os.path.exists", return_value=True)
@patch("os.makedirs")
def test_get_checkpoint_dir_colab(mock_makedirs, mock_exists, mock_is_colab):
    path = DeviceManager.get_checkpoint_dir("test_proj")
    assert "/content/drive/MyDrive/test_proj" in path


@patch("adamops.hardware.DeviceManager.is_colab_environment", return_value=False)
@patch("os.makedirs")
def test_get_checkpoint_dir_local(mock_makedirs, mock_is_colab):
    path = DeviceManager.get_checkpoint_dir("test_proj")
    assert "checkpoints" in path
    assert "test_proj" in path


def test_estimate_model_vram():
    # Provide a dummy raw class that fails torch.nn.Module validation
    class DummyModel:
        pass
        
    mb = DeviceManager.estimate_model_vram(DummyModel())
    assert mb == 0.0

    try:
        import torch
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(100, 100)
                
        model = TinyModel()
        mb = DeviceManager.estimate_model_vram(model)
        assert mb > 0.0
    except ImportError:
        pass
