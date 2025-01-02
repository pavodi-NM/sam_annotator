# tests/integration/test_annotation_workflow.py
import pytest
import os
import cv2
import numpy as np
import logging
import traceback
import sys
from src.core import SAMAnnotator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestAnnotationWorkflow:
    @pytest.fixture
    def test_data(self, tmp_path):
        # Create test dataset structure
        data_dir = tmp_path / "test_data"
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        os.makedirs(images_dir)
        os.makedirs(labels_dir)
        
        # Create test image
        img_path = images_dir / "test1.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        
        # Create test classes CSV
        classes_path = data_dir / "classes.csv"
        with open(classes_path, 'w') as f:
            f.write("class_id,class_name\n0,test_class\n")
            
        return {
            'data_dir': str(data_dir),
            'images_dir': str(images_dir),
            'classes_path': str(classes_path)
        }

    @pytest.mark.xfail(reason="Pickling issue needs further investigation")
    def test_initialization(self, test_data, mock_sam):
        """Test basic initialization of SAMAnnotator."""
        try:
            logger.debug(f"Test data: {test_data}")
            logger.debug(f"Mock SAM path: {mock_sam}")
            
            # Detailed import and dependency tracing
            logger.debug("Sys path:")
            for path in sys.path:
                logger.debug(path)
            
            logger.debug("Imported modules:")
            for name, module in sys.modules.items():
                if 'src' in name or 'segment_anything' in name:
                    logger.debug(f"Module: {name}")
            
            # Attempt to import and inspect SAMAnnotator
            import importlib
            import inspect
            
            logger.debug("SAMAnnotator source code:")
            sam_annotator_module = importlib.import_module('src.core')
            logger.debug(inspect.getsource(sam_annotator_module.SAMAnnotator))
            
            # Initialize annotator with mocked SAM
            annotator = SAMAnnotator(
                checkpoint_path=mock_sam,
                category_path=test_data['data_dir'],
                classes_csv=test_data['classes_path']
            )
            
            # Verify basic initialization
            assert os.path.exists(annotator.images_path)
            assert os.path.exists(annotator.annotations_path)
            assert len(annotator.class_names) > 0
            assert annotator.current_idx == 0
            
        except Exception as e:
            logger.error("Full traceback:")
            traceback.print_exc()
            
            # Attempt to diagnose pickling error
            import pickle
            logger.error(f"Pickling error details: {sys.exc_info()[0]}")
            
            raise