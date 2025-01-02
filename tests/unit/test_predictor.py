import pytest
import numpy as np
import torch
from src.core.predictor import SAMPredictor, GPUMemoryManager

class TestGPUMemoryManager:
    @pytest.fixture
    def memory_manager(self):
        return GPUMemoryManager()

    def test_initialization(self, memory_manager):
        assert memory_manager.warning_threshold == 0.85
        assert memory_manager.critical_threshold == 0.95
        assert memory_manager.warning_count == 0

    def test_get_gpu_memory_info(self, memory_manager):
        memory_info = memory_manager.get_gpu_memory_info()
        assert isinstance(memory_info, dict)
        assert all(key in memory_info for key in ['used', 'total', 'utilization'])

    def test_check_memory_status(self, memory_manager):
        status, message = memory_manager.check_memory_status()
        assert isinstance(status, bool)
        assert isinstance(message, str)

class TestSAMPredictor:
    @pytest.fixture
    def predictor(self):
        return SAMPredictor()

    def test_cache_initialization(self, predictor):
        assert predictor.current_image_embedding is None
        assert predictor.current_image_hash is None
        assert isinstance(predictor.prediction_cache, dict)
        assert predictor.max_cache_size == 50

    def test_generate_cache_key(self, predictor):
        predictor.current_image_hash = "test_hash"
        coords = np.array([[0, 0]])
        labels = np.array([1])
        box = np.array([0, 0, 1, 1])
        
        key = predictor._generate_cache_key(coords, labels, box)
        assert isinstance(key, str)
        assert key.startswith("test_hash")

    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="Skip if CUDA not available")
    def test_memory_optimization(self, predictor):
        predictor.memory_manager.optimize_memory()
        memory_info = predictor.get_memory_usage()
        assert isinstance(memory_info, float)
        assert 0 <= memory_info <= 1