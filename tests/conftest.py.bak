import os
import pytest 
from unittest.mock import MagicMock
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_sam(monkeypatch, tmp_path):
    """Setup complete SAM model mocking."""
    # Detailed logging of import paths and modules
    logger.debug("Current sys.path:")
    for path in sys.path:
        logger.debug(path)
    
    logger.debug("Loaded modules before mocking:")
    for name, module in sys.modules.items():
        if 'src' in name or 'segment_anything' in name:
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
            logger.debug(f"MockSAM.load_state_dict called with state_dict: {state_dict}, strict: {strict}")
            pass

        # Add these methods to prevent pickling issues
        def __getstate__(self):
            return {}

        def __setstate__(self, state):
            pass

    # Mock Predictor
    class MockPredictor:
        def __init__(self, *args, **kwargs):
            logger.debug(f"MockPredictor initialized with args: {args}, kwargs: {kwargs}")
            self.model = MockSAM()

        def initialize(self, checkpoint_path):
            logger.debug(f"MockPredictor.initialize called with path: {checkpoint_path}")
            if checkpoint_path != "sam_vit_h_4b8939.pth":
                raise ValueError(f"Unexpected checkpoint path: {checkpoint_path}")
            self.model.load_state_dict(None)

        def set_image(self, *args, **kwargs):
            logger.debug(f"MockPredictor.set_image called with args: {args}, kwargs: {kwargs}")
            pass

        # Add these methods to prevent pickling issues
        def __getstate__(self):
            return {}

        def __setstate__(self, state):
            pass

    # Verbose mocking
    def mock_build_sam(*args, **kwargs):
        logger.debug(f"mock_build_sam called with args: {args}, kwargs: {kwargs}")
        return MockSAM()

    # Mock SAM functions
    monkeypatch.setattr("segment_anything.build_sam", mock_build_sam)
    monkeypatch.setattr("segment_anything.build_sam_vit_h", mock_build_sam)

    # Mock Registry
    mock_registry = {
        "vit_h": lambda checkpoint: (
            logger.debug(f"Mock registry called with checkpoint: {checkpoint}"),
            MockSAM()
        )[1]
    }
    monkeypatch.setattr("segment_anything.sam_model_registry", mock_registry)

    # Mock File Operations
    mock_target_path = tmp_path / "checkpoints"
    mock_target_path.mkdir(parents=True, exist_ok=True)
    mock_checkpoint_file = mock_target_path / "sam_vit_h_4b8939.pth"
    mock_checkpoint_file.write_text("mock checkpoint content")

    # Verbose mock method for checkpoint path
    def mock_get_checkpoint_path(self, checkpoint_path):
        logger.debug(f"get_checkpoint_path called with: {checkpoint_path}")
        return str(mock_checkpoint_file)

    # Update for SAMWeightManager
    monkeypatch.setattr(
        "src.core.weight_manager.SAMWeightManager.get_checkpoint_path",
        mock_get_checkpoint_path,
    )

    logger.debug("Mock SAM fixture completed setup")
    return str(mock_checkpoint_file)