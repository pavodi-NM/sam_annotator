#!/usr/bin/env python3
"""Tests specifically for our memory management improvements"""

import pytest
import torch
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sam_annotator.core.memory_manager import GPUMemoryManager, MemoryProfiler
from sam_annotator.core.predictor import MemoryAwareLRUCache


class TestMemoryAwareLRUCache:
    """Test our new memory-aware cache implementation"""
    
    def test_cache_creation(self):
        """Test cache can be created with different parameters"""
        cache = MemoryAwareLRUCache(max_size=10, max_memory_mb=100)
        assert len(cache) == 0
        assert cache.max_size == 10
        assert cache.max_memory_bytes == 100 * 1024 * 1024

    def test_cache_basic_operations(self):
        """Test basic cache operations work correctly"""
        cache = MemoryAwareLRUCache(max_size=5, max_memory_mb=1)
        
        # Test setting and getting
        cache['key1'] = (1, 2, 3)
        cache['key2'] = (4, 5, 6)
        
        assert len(cache) == 2
        assert 'key1' in cache
        assert 'key2' in cache
        assert cache['key1'] == (1, 2, 3)
        
    def test_cache_update_method(self):
        """Test the update method works correctly"""
        cache = MemoryAwareLRUCache(max_size=5, max_memory_mb=1)
        
        cache.update({
            'test1': (1, 2, 3),
            'test2': (4, 5, 6),
            'test3': (7, 8, 9)
        })
        
        assert len(cache) == 3
        assert 'test1' in cache
        assert 'test3' in cache

    def test_cache_memory_usage_tracking(self):
        """Test memory usage tracking"""
        cache = MemoryAwareLRUCache(max_size=5, max_memory_mb=1)
        cache['test'] = (1, 2, 3)
        
        usage = cache.get_memory_usage()
        assert 'current_memory_mb' in usage
        assert 'max_memory_mb' in usage
        assert 'utilization' in usage
        assert 'items' in usage
        assert usage['items'] == 1

    def test_cache_clear(self):
        """Test cache clearing"""
        cache = MemoryAwareLRUCache(max_size=5, max_memory_mb=1)
        cache['test'] = (1, 2, 3)
        assert len(cache) == 1
        
        cache.clear()
        assert len(cache) == 0
        assert cache.current_memory == 0


class TestMemoryManagerSingleton:
    """Test singleton behavior of memory manager"""
    
    def test_singleton_behavior(self):
        """Test that multiple instances return the same object"""
        manager1 = GPUMemoryManager()
        manager2 = GPUMemoryManager()
        manager3 = GPUMemoryManager()
        
        assert manager1 is manager2
        assert manager2 is manager3
        assert manager1 is manager3

    def test_singleton_state_persistence(self):
        """Test that singleton maintains state across instances"""
        manager1 = GPUMemoryManager()
        
        # Update statistics
        manager1.update_stats('test_stat', 5)
        
        # Get new instance
        manager2 = GPUMemoryManager()
        
        # Should have the same state
        assert manager1.memory_stats == manager2.memory_stats


class TestMemoryAnalytics:
    """Test our new memory analytics features"""
    
    def test_memory_analytics_structure(self):
        """Test analytics return proper structure"""
        manager = GPUMemoryManager()
        analytics = manager.get_memory_analytics()
        
        required_keys = ['gpu_memory', 'system_memory', 'statistics', 'configuration', 'recommendations']
        for key in required_keys:
            assert key in analytics, f"Missing key: {key}"

    def test_memory_statistics_tracking(self):
        """Test memory statistics are tracked correctly"""
        manager = GPUMemoryManager()
        
        # Initial state
        initial_count = manager.memory_stats.get('optimization_count', 0)
        
        # Update statistics
        manager.update_stats('optimization_count', 3)
        
        # Check updated
        assert manager.memory_stats['optimization_count'] == initial_count + 3

    def test_recommendations_generation(self):
        """Test that recommendations are generated"""
        manager = GPUMemoryManager()
        analytics = manager.get_memory_analytics()
        
        assert 'recommendations' in analytics
        assert isinstance(analytics['recommendations'], list)


class TestMemoryProfiler:
    """Test memory operation profiling"""
    
    def test_memory_profiler_context_manager(self):
        """Test memory profiler works as context manager"""
        manager = GPUMemoryManager()
        
        with manager.profile_memory_operation("test_operation") as profiler:
            assert profiler.operation_name == "test_operation"
            assert profiler.memory_manager is manager

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_memory_profiler_with_gpu_operation(self):
        """Test profiler with actual GPU operation"""
        manager = GPUMemoryManager()
        
        with manager.profile_memory_operation("tensor_allocation"):
            # Small tensor allocation
            tensor = torch.zeros(1000, device='cuda')
            del tensor


@pytest.mark.integration
class TestMemoryManagerIntegration:
    """Integration tests for memory manager"""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initializes without errors"""
        manager = GPUMemoryManager()
        
        # Basic functionality should work
        memory_info = manager.get_gpu_memory_info()
        assert 'utilization' in memory_info
        
        # Analytics should work
        analytics = manager.get_memory_analytics()
        assert len(analytics) > 0

    def test_environment_configuration(self):
        """Test environment variable configuration"""
        # Save current environment values
        old_fraction = os.environ.get('SAM_GPU_MEMORY_FRACTION')
        old_warning = os.environ.get('SAM_MEMORY_WARNING_THRESHOLD')
        
        # Set test environment variables before creating any manager
        os.environ['SAM_GPU_MEMORY_FRACTION'] = '0.85'
        os.environ['SAM_MEMORY_WARNING_THRESHOLD'] = '0.75'
        
        # Clear the singleton to force re-initialization with new env vars
        if hasattr(GPUMemoryManager, '_instance'):
            GPUMemoryManager._instance = None
        
        # Create new manager (will read environment)
        manager = GPUMemoryManager()
        
        # Check configuration was applied
        analytics = manager.get_memory_analytics()
        config = analytics['configuration']
        
        assert config['memory_fraction'] == 0.85
        assert config['warning_threshold'] == 0.75
        
        # Cleanup - restore original values
        if old_fraction is not None:
            os.environ['SAM_GPU_MEMORY_FRACTION'] = old_fraction
        else:
            del os.environ['SAM_GPU_MEMORY_FRACTION']
            
        if old_warning is not None:
            os.environ['SAM_MEMORY_WARNING_THRESHOLD'] = old_warning
        else:
            del os.environ['SAM_MEMORY_WARNING_THRESHOLD']


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 