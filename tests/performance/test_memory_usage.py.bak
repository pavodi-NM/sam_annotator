import pytest
import sys 
import segment_anything
# print(dir(segment_anything))
# sys.exit()
@pytest.fixture
def mock_sam(monkeypatch):
    """Setup complete SAM model mocking."""

    # Create mock SAM
    class MockSAM:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=True):
            # Simulate successful loading of state_dict
            pass

    # Create mock predictor
    class MockPredictor:
        def __init__(self, *args, **kwargs):
            self.model = MockSAM()

        def initialize(self, checkpoint_path):
            # Simulate using the checkpoint path
            if checkpoint_path != "sam_vit_h_4b8939.pth":
                raise ValueError(f"Unexpected checkpoint path: {checkpoint_path}")
            self.model.load_state_dict(None)  # Simulate loading weights

        def set_image(self, *args, **kwargs):
            # Simulate setting an image
            pass

    # Mock `build_sam` to always return a MockSAM instance
    monkeypatch.setattr("segment_anything.build_sam", lambda *args, **kwargs: MockSAM())

    # Mock the SAM model registry
    mock_registry = {'vit_h': lambda checkpoint: MockSAM()}
    monkeypatch.setattr("segment_anything.sam_model_registry", mock_registry)

    # Mock `build_sam_vit_h` if it exists directly under `segment_anything`
    monkeypatch.setattr("segment_anything.build_sam_vit_h", lambda *args, **kwargs: MockSAM())

    return "sam_vit_h_4b8939.pth"  # Return the actual weight filename to simulate usage
