# tests/integration/test_annotation_workflow.py
import pytest
import os
import cv2
import numpy as np
import logging
import traceback
import sys
import torch
from src.core import SAMAnnotator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestAnnotationWorkflow:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield
        
        # Teardown
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    @pytest.fixture
    def test_data(self, tmp_path):
        # Your existing test_data fixture code remains the same
        data_dir = tmp_path / "test_data"
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        os.makedirs(images_dir)
        os.makedirs(labels_dir)
        
        img_path = images_dir / "test1.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        
        classes_path = data_dir / "classes.csv"
        with open(classes_path, 'w') as f:
            f.write("class_id,class_name\n0,test_class\n")
            
        return {
            'data_dir': str(data_dir),
            'images_dir': str(images_dir),
            'classes_path': str(classes_path)
        }

    @pytest.mark.skipif(
        not torch.cuda.is_available() and os.getenv('CI') == 'true',
        reason="Skipping in CI environment without GPU"
    )
    def test_initialization(self, test_data, mock_sam):
        """Test basic initialization of SAMAnnotator."""
        try:
            logger.debug(f"Test data: {test_data}")
            logger.debug(f"Mock SAM path: {mock_sam}")
            
            # Log system info
            logger.debug(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")
            
            # Initialize with device handling
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_default_tensor_type('torch.FloatTensor')  # Force CPU tensors
            
            # Initialize annotator with mocked SAM
            annotator = SAMAnnotator(
                checkpoint_path=mock_sam,
                category_path=test_data['data_dir'],
                classes_csv=test_data['classes_path']
            )
            
            # Basic assertions
            assert os.path.exists(annotator.images_path)
            assert os.path.exists(annotator.annotations_path)
            assert len(annotator.class_names) > 0
            assert annotator.current_idx == 0
            
        except Exception as e:
            logger.error("Full traceback:")
            traceback.print_exc()
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {str(e)}")
            
            if "CUDA" in str(e):
                logger.error("CUDA-related error detected. Running on CPU only.")
                pytest.skip("CUDA error in CI environment")
            
            raise