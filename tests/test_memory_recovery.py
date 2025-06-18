#!/usr/bin/env python3
"""Test suite for memory recovery system."""

import pytest
import torch
import time
import logging
from unittest.mock import Mock, patch, MagicMock

# Set up path for importing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam_annotator.core.memory_recovery import (
    MemoryRecoveryManager, 
    MemoryRecoveryResult, 
    RecoveryStage
)
from sam_annotator.core.memory_manager import GPUMemoryManager


class TestMemoryRecoveryResult:
    """Test MemoryRecoveryResult class."""
    
    def test_recovery_result_creation(self):
        """Test creating a recovery result."""
        result = MemoryRecoveryResult(
            success=True,
            stage=RecoveryStage.CLEAR_CACHE,
            message="Cache cleared",
            recovered_memory=100.0,
            execution_time=0.5
        )
        
        assert result.success is True
        assert result.stage == RecoveryStage.CLEAR_CACHE
        assert result.message == "Cache cleared"
        assert result.recovered_memory == 100.0
        assert result.execution_time == 0.5

    def test_recovery_result_string_representation(self):
        """Test string representation of recovery result."""
        success_result = MemoryRecoveryResult(
            success=True,
            stage=RecoveryStage.CLEAR_CACHE,
            message="Success"
        )
        
        failure_result = MemoryRecoveryResult(
            success=False,
            stage=RecoveryStage.FORCE_GC,
            message="Failed"
        )
        
        assert "✅" in str(success_result)
        assert "❌" in str(failure_result)
        assert "clear_cache" in str(success_result)
        assert "force_gc" in str(failure_result)


class TestMemoryRecoveryManager:
    """Test MemoryRecoveryManager class."""
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        mock_manager = Mock()
        mock_manager.get_gpu_memory_info.return_value = {
            'used': 1000000000,  # 1GB
            'total': 4000000000,  # 4GB
            'utilization': 0.25
        }
        return mock_manager

    @pytest.fixture
    def recovery_manager(self, mock_memory_manager):
        """Create a recovery manager with mock memory manager."""
        return MemoryRecoveryManager(mock_memory_manager)

    def test_recovery_manager_initialization(self, recovery_manager):
        """Test recovery manager initialization."""
        assert recovery_manager.memory_manager is not None
        assert 'total_attempts' in recovery_manager.recovery_stats
        assert len(recovery_manager.recovery_strategies) == 5
        assert RecoveryStage.CLEAR_CACHE in recovery_manager.recovery_strategies

    def test_stage1_clear_cache_success(self, recovery_manager):
        """Test stage 1 cache clearing success."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                result = recovery_manager._stage1_clear_cache("test context")
                
                assert result.success is True
                assert result.stage == RecoveryStage.CLEAR_CACHE
                mock_empty_cache.assert_called_once()

    def test_stage1_clear_cache_no_gpu(self, recovery_manager):
        """Test stage 1 cache clearing without GPU."""
        with patch('torch.cuda.is_available', return_value=False):
            result = recovery_manager._stage1_clear_cache("test context")
            
            # Should still succeed (no GPU is not an error)
            assert result.success is True
            assert result.stage == RecoveryStage.CLEAR_CACHE

    def test_stage2_force_gc_success(self, recovery_manager):
        """Test stage 2 garbage collection success."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('gc.collect') as mock_gc:
                with patch('torch.cuda.empty_cache') as mock_empty_cache:
                    with patch('torch.cuda.synchronize') as mock_sync:
                        result = recovery_manager._stage2_force_gc("test context")
                        
                        assert result.success is True
                        assert result.stage == RecoveryStage.FORCE_GC
                        assert mock_gc.call_count == 3  # Called 3 times in loop
                        mock_empty_cache.assert_called_once()
                        mock_sync.assert_called_once()

    def test_stage3_reduce_precision_not_implemented(self, recovery_manager):
        """Test stage 3 precision reduction (not yet implemented)."""
        result = recovery_manager._stage3_reduce_precision("test context")
        
        assert result.success is False
        assert result.stage == RecoveryStage.REDUCE_PRECISION
        assert "not yet implemented" in result.message

    def test_stage4_cpu_fallback_not_implemented(self, recovery_manager):
        """Test stage 4 CPU fallback (not yet implemented)."""
        result = recovery_manager._stage4_cpu_fallback("test context")
        
        assert result.success is False
        assert result.stage == RecoveryStage.CPU_FALLBACK
        assert "not yet implemented" in result.message

    def test_stage5_graceful_failure_success(self, recovery_manager):
        """Test stage 5 graceful failure."""
        result = recovery_manager._stage5_graceful_failure("test context")
        
        assert result.success is True  # Success in failing gracefully
        assert result.stage == RecoveryStage.GRACEFUL_FAILURE
        assert "Suggestions:" in result.message

    def test_user_suggestions_generation(self, recovery_manager):
        """Test user suggestions generation."""
        suggestions = recovery_manager._generate_user_suggestions()
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("resolution" in s for s in suggestions)
        assert any("applications" in s for s in suggestions)

    def test_user_suggestions_with_critical_memory(self, recovery_manager):
        """Test user suggestions when memory is critical."""
        recovery_manager.memory_manager.get_gpu_memory_info.return_value = {
            'utilization': 0.95  # Critical threshold
        }
        
        suggestions = recovery_manager._generate_user_suggestions()
        
        assert any("critical" in s for s in suggestions)

    def test_recover_from_oom_single_stage_success(self, recovery_manager):
        """Test successful recovery in first stage."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache'):
                result = recovery_manager.recover_from_oom("test operation")
                
                assert result.success is True
                assert result.stage == RecoveryStage.CLEAR_CACHE
                assert recovery_manager.recovery_stats['total_attempts'] == 1
                assert recovery_manager.recovery_stats['successful_recoveries'] == 1

    def test_recover_from_oom_with_operation_callback(self, recovery_manager):
        """Test recovery with operation callback."""
        mock_callback = Mock(return_value="success")
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache'):
                result = recovery_manager.recover_from_oom(
                    "test operation",
                    operation_callback=mock_callback,
                    test_arg="test_value"
                )
                
                assert result.success is True
                assert "Operation retry succeeded" in result.message
                mock_callback.assert_called_with(test_arg="test_value")

    def test_recover_from_oom_callback_fails_continues_stages(self, recovery_manager):
        """Test recovery continues to next stage if callback fails."""
        mock_callback = Mock(side_effect=Exception("callback failed"))
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache'):
                with patch('gc.collect'):
                    result = recovery_manager.recover_from_oom(
                        "test operation",
                        operation_callback=mock_callback
                    )
                    
                    # Should succeed in later stage
                    assert result.success is True
                    assert mock_callback.call_count >= 1

    def test_recovery_statistics_tracking(self, recovery_manager):
        """Test recovery statistics tracking."""
        # Simulate multiple recovery attempts
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache'):
                # Successful recovery
                recovery_manager.recover_from_oom("test 1")
                
                # Another successful recovery
                recovery_manager.recover_from_oom("test 2")
        
        stats = recovery_manager.get_recovery_statistics()
        
        assert stats['total_attempts'] == 2
        assert stats['successful_recoveries'] == 2
        assert stats['success_rate_percent'] == 100.0
        assert stats['stage_usage']['clear_cache'] == 2

    def test_recovery_statistics_with_failures(self, recovery_manager):
        """Test recovery statistics with some failures."""
        # Mock the recovery strategies dictionary directly
        mock_strategies = {
            RecoveryStage.CLEAR_CACHE: Mock(return_value=MemoryRecoveryResult(False, RecoveryStage.CLEAR_CACHE, "failed")),
            RecoveryStage.FORCE_GC: Mock(return_value=MemoryRecoveryResult(False, RecoveryStage.FORCE_GC, "failed")),
            RecoveryStage.REDUCE_PRECISION: Mock(return_value=MemoryRecoveryResult(False, RecoveryStage.REDUCE_PRECISION, "failed")),
            RecoveryStage.CPU_FALLBACK: Mock(return_value=MemoryRecoveryResult(False, RecoveryStage.CPU_FALLBACK, "failed")),
            RecoveryStage.GRACEFUL_FAILURE: Mock(return_value=MemoryRecoveryResult(True, RecoveryStage.GRACEFUL_FAILURE, "graceful failure"))
        }
        
        with patch.object(recovery_manager, 'recovery_strategies', mock_strategies):
            result = recovery_manager.recover_from_oom("failing test")
            
            assert result.stage == RecoveryStage.GRACEFUL_FAILURE
            assert result.success is True
            stats = recovery_manager.get_recovery_statistics()
            assert stats['total_attempts'] == 1

    def test_get_current_memory_usage(self, recovery_manager):
        """Test getting current memory usage."""
        memory_usage = recovery_manager._get_current_memory_usage()
        
        # Should return the mocked value converted to MB
        expected_mb = 1000000000 / (1024 * 1024)  # Convert bytes to MB
        assert memory_usage == expected_mb

    def test_get_current_memory_usage_error_handling(self, recovery_manager):
        """Test memory usage error handling."""
        recovery_manager.memory_manager.get_gpu_memory_info.side_effect = Exception("error")
        
        memory_usage = recovery_manager._get_current_memory_usage()
        assert memory_usage == 0.0

    def test_avg_recovery_time_calculation(self, recovery_manager):
        """Test average recovery time calculation."""
        # Test with proper successful recovery count
        recovery_manager.recovery_stats['successful_recoveries'] = 1
        recovery_manager._update_avg_recovery_time(1.0)
        assert recovery_manager.recovery_stats['avg_recovery_time'] == 1.0
        
        recovery_manager.recovery_stats['successful_recoveries'] = 2
        recovery_manager._update_avg_recovery_time(2.0)
        assert recovery_manager.recovery_stats['avg_recovery_time'] == 1.5
        
        recovery_manager.recovery_stats['successful_recoveries'] = 3
        recovery_manager._update_avg_recovery_time(3.0)
        assert recovery_manager.recovery_stats['avg_recovery_time'] == 2.0
        
        # Test edge case with zero successful recoveries
        recovery_manager.recovery_stats['successful_recoveries'] = 0
        original_avg = recovery_manager.recovery_stats['avg_recovery_time']
        recovery_manager._update_avg_recovery_time(5.0)
        # Should remain unchanged
        assert recovery_manager.recovery_stats['avg_recovery_time'] == original_avg

    def test_reset_statistics(self, recovery_manager):
        """Test resetting recovery statistics."""
        # Set some statistics
        recovery_manager.recovery_stats['total_attempts'] = 5
        recovery_manager.recovery_stats['successful_recoveries'] = 3
        recovery_manager.recovery_stats['stage_usage']['clear_cache'] = 2
        
        recovery_manager.reset_statistics()
        
        assert recovery_manager.recovery_stats['total_attempts'] == 0
        assert recovery_manager.recovery_stats['successful_recoveries'] == 0
        assert recovery_manager.recovery_stats['stage_usage']['clear_cache'] == 0


class TestMemoryManagerIntegration:
    """Test integration with GPUMemoryManager."""
    
    @pytest.fixture
    def memory_manager(self):
        """Get memory manager instance."""
        return GPUMemoryManager()

    def test_memory_manager_has_recovery_manager(self, memory_manager):
        """Test that memory manager has recovery manager."""
        assert hasattr(memory_manager, 'recovery_manager')
        assert memory_manager.recovery_manager is not None

    def test_execute_with_recovery_success(self, memory_manager):
        """Test successful operation without recovery needed."""
        def successful_operation():
            return "success"
        
        success, result, recovery_result = memory_manager.execute_with_recovery(
            successful_operation, "test operation"
        )
        
        assert success is True
        assert result == "success"
        assert recovery_result is None

    def test_execute_with_recovery_memory_error(self, memory_manager):
        """Test memory error triggers recovery."""
        def failing_operation():
            raise RuntimeError("CUDA out of memory")
        
        with patch.object(memory_manager.recovery_manager, 'recover_from_oom') as mock_recover:
            mock_recover.return_value = MemoryRecoveryResult(
                True, RecoveryStage.CLEAR_CACHE, "recovered"
            )
            
            success, result, recovery_result = memory_manager.execute_with_recovery(
                failing_operation, "test operation"
            )
            
            assert success is False  # Operation failed but recovery attempted
            assert recovery_result is not None
            mock_recover.assert_called_once()

    def test_execute_with_recovery_non_memory_error(self, memory_manager):
        """Test non-memory errors are re-raised."""
        def failing_operation():
            raise ValueError("Not a memory error")
        
        with pytest.raises(ValueError):
            memory_manager.execute_with_recovery(
                failing_operation, "test operation"
            )

    def test_get_recovery_statistics_integration(self, memory_manager):
        """Test recovery statistics integration."""
        stats = memory_manager.get_recovery_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_attempts' in stats
        assert 'successful_recoveries' in stats

    def test_memory_analytics_includes_recovery_stats(self, memory_manager):
        """Test that memory analytics includes recovery statistics."""
        analytics = memory_manager.get_memory_analytics()
        
        assert 'recovery_statistics' in analytics
        assert isinstance(analytics['recovery_statistics'], dict)


# Performance and stress tests
class TestRecoveryPerformance:
    """Test recovery system performance."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create recovery manager for performance tests."""
        mock_manager = Mock()
        mock_manager.get_gpu_memory_info.return_value = {
            'used': 1000000000, 'total': 4000000000, 'utilization': 0.25
        }
        return MemoryRecoveryManager(mock_manager)

    def test_recovery_speed(self, recovery_manager):
        """Test recovery execution speed."""
        start_time = time.time()
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache'):
                result = recovery_manager.recover_from_oom("speed test")
        
        execution_time = time.time() - start_time
        
        assert result.success is True
        assert execution_time < 1.0  # Should complete within 1 second
        assert result.execution_time > 0

    def test_multiple_concurrent_recoveries(self, recovery_manager):
        """Test handling multiple recovery attempts."""
        results = []
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache'):
                for i in range(5):
                    result = recovery_manager.recover_from_oom(f"concurrent test {i}")
                    results.append(result)
        
        assert len(results) == 5
        assert all(r.success for r in results)
        assert recovery_manager.recovery_stats['total_attempts'] == 5


if __name__ == "__main__":
    # Enable logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v"]) 