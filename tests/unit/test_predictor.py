import pytest
import numpy as np
import torch
from unittest.mock import patch
from sam_annotator.core.predictor import SAM1Predictor, SAM2Predictor, LRUCache
from sam_annotator.core.memory_manager import GPUMemoryManager

class TestGPUMemoryManager:
    """Tests for the GPUMemoryManager."""
    
    def test_initialization(self, mock_gpu_memory_manager):
        """Test that GPUMemoryManager initializes with expected values."""
        # Use the mock manager provided by the fixture
        assert mock_gpu_memory_manager.warning_threshold == 0.8
        assert mock_gpu_memory_manager.critical_threshold == 0.95
        assert hasattr(mock_gpu_memory_manager, "device")

    def test_get_gpu_memory_info(self, mock_gpu_memory_manager):
        """Test that get_gpu_memory_info returns expected dictionary."""
        memory_info = mock_gpu_memory_manager.get_gpu_memory_info()
        assert isinstance(memory_info, dict)
        assert 'total' in memory_info
        assert 'used' in memory_info
        assert 'free' in memory_info
        assert 'utilization' in memory_info

    def test_check_memory_status(self, mock_gpu_memory_manager):
        """Test that check_memory_status returns expected values."""
        # Test with normal memory usage
        status, message = mock_gpu_memory_manager.check_memory_status()
        assert status is True
        assert message is None

class TestLRUCache:
    """Tests for the LRUCache implementation."""
    
    def test_initialization(self):
        """Test that LRUCache initializes with expected values."""
        cache = LRUCache(max_size=10)
        assert len(cache) == 0
        assert cache.max_size == 10
        
    def test_set_and_get(self):
        """Test setting and getting values in the cache."""
        cache = LRUCache(max_size=2)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        
        assert cache["key1"] == "value1"
        assert cache["key2"] == "value2"
        assert len(cache) == 2
        
    def test_lru_eviction(self):
        """Test that least recently used items are evicted when cache is full."""
        cache = LRUCache(max_size=2)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        # This should evict key1
        cache["key3"] = "value3"
        
        assert "key1" not in cache
        assert "key2" in cache
        assert "key3" in cache
        assert len(cache) == 2
        
    def test_update_usage_order(self):
        """Test that accessing an item updates its usage order."""
        cache = LRUCache(max_size=2)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        
        # Access key1 to make it most recently used
        _ = cache["key1"]
        
        # Add a new item, should evict key2 instead of key1
        cache["key3"] = "value3"
        
        assert "key1" in cache
        assert "key2" not in cache
        assert "key3" in cache
        
    def test_clear(self):
        """Test clearing the cache."""
        cache = LRUCache(max_size=2)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        
        cache.clear()
        assert len(cache) == 0

class TestSAM1Predictor:
    """Tests for SAM1Predictor."""
    
    @pytest.fixture
    def predictor(self, mock_gpu_memory_manager):
        """Create a SAM1Predictor with mocked components."""
        # Patch GPUMemoryManager to use our mock
        with patch('sam_annotator.core.predictor.GPUMemoryManager', 
                   return_value=mock_gpu_memory_manager):
            pred = SAM1Predictor(model_type="vit_h")
            return pred

    def test_initialization(self, predictor):
        """Test that predictor initializes with expected values."""
        assert predictor.model_type == "vit_h"
        assert predictor.model is None
        assert predictor.predictor is None
        assert predictor.sam_version == "sam1"
        assert isinstance(predictor.device, torch.device)

    def test_cache_initialization(self, predictor):
        """Test that cache is initialized correctly."""
        assert predictor.current_image is None
        assert predictor.current_image_hash is None
        assert isinstance(predictor.prediction_cache, LRUCache)
        assert predictor.prediction_cache.max_size == 50

    def test_generate_cache_key(self, predictor):
        """Test that cache key generation works as expected."""
        # Set a dummy image hash
        predictor.current_image_hash = "test_hash"
        
        # Create test inputs
        point_coords = np.array([[0, 0]])
        point_labels = np.array([1])
        box = np.array([0, 0, 1, 1])
        
        # Generate and verify the cache key
        key = predictor._generate_cache_key(point_coords, point_labels, box)
        assert isinstance(key, str)
        assert len(key) > 0
        
        # Generate another key with different inputs
        key2 = predictor._generate_cache_key(None, None, box)
        assert key != key2

class TestSAM2Predictor:
    """Tests for SAM2Predictor."""
    
    @pytest.fixture
    def predictor(self, mock_gpu_memory_manager):
        """Create a SAM2Predictor with mocked components."""
        # Patch GPUMemoryManager to use our mock
        with patch('sam_annotator.core.predictor.GPUMemoryManager', 
                   return_value=mock_gpu_memory_manager):
            pred = SAM2Predictor(model_type="base")
            return pred

    def test_initialization(self, predictor):
        """Test that predictor initializes with expected values."""
        assert predictor.model_type == "base"
        assert predictor.model is None
        assert predictor.sam_version == "sam2"
        assert isinstance(predictor.device, torch.device)

    def test_cache_initialization(self, predictor):
        """Test that cache is initialized correctly."""
        assert predictor.current_image is None
        assert predictor.current_image_hash is None
        assert isinstance(predictor.prediction_cache, LRUCache)
        assert predictor.prediction_cache.max_size == 20

    def test_generate_cache_key(self, predictor):
        """Test that cache key generation works as expected."""
        # Set a dummy image hash
        predictor.current_image_hash = "test_hash"
        
        # Create test inputs
        point_coords = np.array([[0, 0]])
        point_labels = np.array([1])
        box = np.array([0, 0, 1, 1])
        
        # Generate and verify the cache key
        key = predictor._generate_cache_key(point_coords, point_labels, box)
        assert isinstance(key, str)
        assert len(key) > 0
        
        # Generate another key with different inputs
        key2 = predictor._generate_cache_key(None, None, box)
        assert key != key2