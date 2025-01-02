import pytest
import os
import numpy as np
from weakref import WeakValueDictionary  # Add this import
from src.data.dataset_manager import DatasetManager, LazyImageLoader

class TestDatasetManager:
    @pytest.fixture
    def dataset_path(self, tmp_path):
        # Create test dataset structure
        dataset_dir = tmp_path / "test_dataset"
        os.makedirs(dataset_dir / "images")
        os.makedirs(dataset_dir / "labels")
        return str(dataset_dir)

    @pytest.fixture
    def manager(self, dataset_path):
        # Create and return manager fixture
        return DatasetManager(dataset_path)

    def test_initialization(self, manager, dataset_path):
        assert manager.dataset_path == dataset_path
        assert isinstance(manager.image_cache, WeakValueDictionary)  # Fixed type check
        assert isinstance(manager.annotation_cache, dict)

class TestLazyImageLoader:
    @pytest.fixture
    def image_path(self, tmp_path):
        # Create a test image
        import cv2
        img_path = tmp_path / "test_image.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        return str(img_path)

    @pytest.fixture
    def loader(self, image_path):  # Add loader fixture
        return LazyImageLoader(image_path)

    def test_initialization(self, loader, image_path):  # Now using correct fixtures
        assert loader.image_path == image_path
        assert loader._image is None
        assert loader._metadata is None