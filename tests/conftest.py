import os
import pytest 
from unittest.mock import MagicMock, patch
import logging
import sys
import torch
import pandas as pd

# Configure logging for tests - use a basic configuration to avoid mocking issues
logging_handlers = []
for handler in logging.getLogger().handlers:
    logging_handlers.append(handler)
    logging.getLogger().removeHandler(handler)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def reset_logging_after_test():
    """Reset logging configuration after each test to avoid handler conflicts."""
    yield
    # Remove all handlers after test
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Restore original handlers if any
    for handler in logging_handlers:
        logging.getLogger().addHandler(handler)

@pytest.fixture
def mock_sam(monkeypatch, tmp_path):
    """Setup complete SAM model mocking."""
    # Detailed logging of import paths and modules
    logger.debug("Current sys.path:")
    for path in sys.path:
        logger.debug(path)
    
    logger.debug("Loaded modules before mocking:")
    for name, module in sys.modules.items():
        if 'sam_annotator' in name or 'segment_anything' in name:
            logger.debug(f"Module: {name}")
    
    # Create mock SAM
    class MockSAM:
        def __init__(self, *args, **kwargs):
            logger.debug(f"MockSAM initialized with args: {args}, kwargs: {kwargs}")
            pass

        def to(self, *args, **kwargs):
            logger.debug(f"MockSAM.to called with args: {args}, kwargs: {kwargs}")
            return self

        def eval(self):
            logger.debug("MockSAM.eval called")
            return self

        def load_state_dict(self, state_dict, strict=True):
            logger.debug(f"MockSAM.load_state_dict called with state_dict size: {type(state_dict)}, strict: {strict}")
            # Return silently without errors, even if state_dict is empty
            # This prevents 'Error(s) in loading state_dict for Sam' in tests
            pass

        # Add these methods to prevent pickling issues
        def __getstate__(self):
            return {}

        def __setstate__(self, state):
            pass

    # Mock predictor methods
    class MockPredictor:
        def __init__(self, *args, **kwargs):
            logger.debug(f"MockPredictor initialized with args: {args}, kwargs: {kwargs}")
            self.model = MockSAM()

        def initialize(self, checkpoint_path):
            logger.debug(f"MockPredictor.initialize called with path: {checkpoint_path}")
            self.model.load_state_dict({})  # Pass empty dict instead of None

        def set_image(self, *args, **kwargs):
            logger.debug(f"MockPredictor.set_image called with args: {args}, kwargs: {kwargs}")
            pass

        def predict(self, *args, **kwargs):
            logger.debug(f"MockPredictor.predict called with args: {args}, kwargs: {kwargs}")
            # Return mock tensors that match the expected return format
            return {
                "masks": torch.ones((1, 256, 256), dtype=torch.bool),
                "scores": torch.tensor([0.9]),
                "stability_scores": torch.tensor([0.8]),
                "iou_predictions": torch.tensor([0.7])
            }

        # Add these methods to prevent pickling issues
        def __getstate__(self):
            return {}

        def __setstate__(self, state):
            pass

    # Create the checkpoint directory and file
    mock_target_path = tmp_path / "checkpoints"
    mock_target_path.mkdir(parents=True, exist_ok=True)
    mock_checkpoint_file = mock_target_path / "sam_vit_h_4b8939.pth"
    
    # Create a valid checkpoint file format
    with open(str(mock_checkpoint_file), 'wb') as f:
        # Just write some bytes - torch.save won't be used
        f.write(b'\x80\x03}q\x00.')

    # Mock torch.load to return a dictionary instead of raising an error
    def mock_torch_load(*args, **kwargs):
        logger.debug(f"mock_torch_load called with args: {args}")
        return {}  # Return empty state dict

    monkeypatch.setattr(torch, "load", mock_torch_load)

    # Mock SAM model registry
    def mock_sam_model(*args, **kwargs):
        logger.debug(f"mock_sam_model called with args: {args}, kwargs: {kwargs}")
        return MockSAM()

    # Create mock registry dict
    mock_registry = {
        "vit_h": mock_sam_model,
        "vit_b": mock_sam_model,
        "vit_l": mock_sam_model,
        "tiny": mock_sam_model,
        "small": mock_sam_model,
        "base": mock_sam_model,
        "large": mock_sam_model,
        "small_v2": mock_sam_model,
        "base_v2": mock_sam_model,
        "large_v2": mock_sam_model,
        "tiny_v2": mock_sam_model
    }

    # Apply patches
    monkeypatch.setattr("segment_anything.sam_model_registry", mock_registry)
    
    # Create patch for SAM1Predictor and SAM2Predictor in sam_annotator
    monkeypatch.setattr("sam_annotator.core.predictor.SAM1Predictor", MockPredictor)
    monkeypatch.setattr("sam_annotator.core.predictor.SAM2Predictor", MockPredictor)
    
    logger.debug("Mock SAM fixture completed setup")
    return str(mock_checkpoint_file)

@pytest.fixture
def mock_gpu_memory_manager(monkeypatch):
    """Create a mock GPU memory manager for testing."""
    # Create real class with necessary methods but no external dependencies
    class MockGPUMemoryManager:
        def __init__(self):
            self.device = torch.device('cpu')
            self.warning_threshold = 0.8
            self.critical_threshold = 0.95
            self.logger = logging.getLogger("mock_memory_manager")
            
        def get_gpu_memory_info(self):
            return {
                'used': 1000, 
                'total': 10000, 
                'free': 9000,  # Added free key
                'utilization': 0.1,
                'formatted': 'Mock memory: 1000/10000 (10.0%)'
            }
            
        def check_memory_status(self):
            return True, None
            
        def optimize_memory(self, force=False):
            pass
            
        def safe_get_memory_info(self):
            return self.get_gpu_memory_info()
            
        def _get_env_float(self, name, default):
            return default
            
        def _get_env_bool(self, name, default):
            return default
            
    # Apply monkeypatch
    monkeypatch.setattr("sam_annotator.core.memory_manager.GPUMemoryManager", MockGPUMemoryManager)
    
    return MockGPUMemoryManager()

@pytest.fixture
def sample_config():
    """Return a sample configuration for testing."""
    return {
        "last_category_path": "/test/path",
        "last_classes_csv": "/test/classes.csv",
        "last_sam_version": "sam1",
        "last_model_type": "vit_h"
    }

@pytest.fixture
def test_data(tmp_path):
    """Create sample test data for tests."""
    # Create test directories
    data_dir = tmp_path / "test_data"
    images_dir = data_dir / "images"
    
    for directory in [data_dir, images_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create a simple class CSV file
    classes_path = data_dir / "classes.csv"
    df = pd.DataFrame({"class_name": ["background", "person", "car"]})
    df.to_csv(classes_path, index=False)
    
    # Create a simple image file
    image_path = images_dir / "test.jpg"
    with open(image_path, 'wb') as f:
        # Just write some bytes to simulate an image
        f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00')
    
    return {
        "data_dir": str(data_dir),
        "images_dir": str(images_dir),
        "classes_path": str(classes_path)
    }

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock()