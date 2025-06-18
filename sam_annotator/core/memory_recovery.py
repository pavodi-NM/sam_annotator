#!/usr/bin/env python3
"""Enhanced memory recovery system for SAM Annotator"""

import logging
import torch
import gc
import time
from typing import Optional, Callable, Any, Dict, List
from enum import Enum

# Import status reporting
from .recovery_status import get_recovery_reporter


class RecoveryStage(Enum):
    """Recovery stages in order of aggressiveness"""
    CLEAR_CACHE = "clear_cache"
    FORCE_GC = "force_gc"  
    REDUCE_PRECISION = "reduce_precision"
    CPU_FALLBACK = "cpu_fallback"
    GRACEFUL_FAILURE = "graceful_failure"


class MemoryRecoveryResult:
    """Result of a memory recovery attempt"""
    
    def __init__(self, success: bool, stage: RecoveryStage, message: str, 
                 recovered_memory: float = 0.0, execution_time: float = 0.0):
        self.success = success
        self.stage = stage
        self.message = message
        self.recovered_memory = recovered_memory  # MB
        self.execution_time = execution_time  # seconds
        
    def __str__(self):
        return f"Recovery({self.stage.value}): {'✅' if self.success else '❌'} {self.message}"


class MemoryRecoveryManager:
    """Multi-stage memory recovery system"""
    
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Get status reporter
        self.status_reporter = get_recovery_reporter()
        
        # Recovery statistics
        self.recovery_stats = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'stage_usage': {stage.value: 0 for stage in RecoveryStage},
            'avg_recovery_time': 0.0,
            'total_memory_recovered': 0.0
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            RecoveryStage.CLEAR_CACHE: self._stage1_clear_cache,
            RecoveryStage.FORCE_GC: self._stage2_force_gc,
            RecoveryStage.REDUCE_PRECISION: self._stage3_reduce_precision,
            RecoveryStage.CPU_FALLBACK: self._stage4_cpu_fallback,
            RecoveryStage.GRACEFUL_FAILURE: self._stage5_graceful_failure
        }

    def recover_from_oom(self, error_context: str, 
                        operation_callback: Optional[Callable] = None,
                        **operation_kwargs) -> MemoryRecoveryResult:
        """
        Progressive recovery from out-of-memory errors.
        
        Args:
            error_context: Description of the operation that failed
            operation_callback: Optional callback to retry after recovery
            **operation_kwargs: Arguments for the operation callback
            
        Returns:
            MemoryRecoveryResult indicating success/failure and details
        """
        self.logger.warning(f"Starting memory recovery for: {error_context}")
        self.recovery_stats['total_attempts'] += 1
        
        start_time = time.time()
        initial_memory = self._get_current_memory_usage()
        
        # Try each recovery stage in order
        for stage in RecoveryStage:
            try:
                recovery_start = time.time()
                self.logger.info(f"Attempting recovery stage: {stage.value}")
                
                # Execute recovery strategy
                stage_result = self.recovery_strategies[stage](error_context, **operation_kwargs)
                
                # Update statistics
                recovery_time = time.time() - recovery_start
                self.recovery_stats['stage_usage'][stage.value] += 1
                
                if stage_result.success:
                    # Calculate recovered memory
                    current_memory = self._get_current_memory_usage()
                    recovered_memory = max(0, initial_memory - current_memory)
                    
                    # Update recovery result
                    stage_result.recovered_memory = recovered_memory
                    stage_result.execution_time = recovery_time
                    
                    # Update global statistics
                    self.recovery_stats['successful_recoveries'] += 1
                    self.recovery_stats['total_memory_recovered'] += recovered_memory
                    self._update_avg_recovery_time(time.time() - start_time)
                    
                    self.logger.info(f"Recovery successful: {stage_result}")
                    
                    # Report to status system
                    self.status_reporter.record_recovery_event(
                        operation=error_context,
                        stage=stage.value,
                        success=True,
                        recovery_time=recovery_time,
                        memory_recovered=recovered_memory,
                        message=stage_result.message
                    )
                    
                    # Try to retry the operation if callback provided
                    if operation_callback and stage != RecoveryStage.GRACEFUL_FAILURE:
                        try:
                            result = operation_callback(**operation_kwargs)
                            stage_result.message += f" | Operation retry succeeded"
                            return stage_result
                        except Exception as retry_error:
                            self.logger.warning(f"Operation retry failed: {retry_error}")
                            # Continue to next stage
                            continue
                    
                    return stage_result
                    
                else:
                    self.logger.warning(f"Recovery stage {stage.value} failed: {stage_result.message}")
                    
            except Exception as e:
                self.logger.error(f"Error in recovery stage {stage.value}: {e}")
                continue
        
        # All recovery attempts failed
        self.recovery_stats['failed_recoveries'] += 1
        total_time = time.time() - start_time
        
        # Report failure to status system
        self.status_reporter.record_recovery_event(
            operation=error_context,
            stage="all_stages_failed",
            success=False,
            recovery_time=total_time,
            memory_recovered=0.0,
            message=f"All recovery attempts failed for: {error_context}"
        )
        
        return MemoryRecoveryResult(
            success=False,
            stage=RecoveryStage.GRACEFUL_FAILURE,
            message=f"All recovery attempts failed for: {error_context}",
            execution_time=total_time
        )

    def _stage1_clear_cache(self, error_context: str, **kwargs) -> MemoryRecoveryResult:
        """Stage 1: Clear prediction caches"""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear any application-level caches if accessible
            # This would be extended based on the specific caches in use
            
            return MemoryRecoveryResult(
                success=True,
                stage=RecoveryStage.CLEAR_CACHE,
                message="Cache cleared successfully"
            )
            
        except Exception as e:
            return MemoryRecoveryResult(
                success=False,
                stage=RecoveryStage.CLEAR_CACHE,
                message=f"Cache clearing failed: {e}"
            )

    def _stage2_force_gc(self, error_context: str, **kwargs) -> MemoryRecoveryResult:
        """Stage 2: Force garbage collection"""
        try:
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
            
            # Clear PyTorch cache again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            return MemoryRecoveryResult(
                success=True,
                stage=RecoveryStage.FORCE_GC,
                message="Aggressive garbage collection completed"
            )
            
        except Exception as e:
            return MemoryRecoveryResult(
                success=False,
                stage=RecoveryStage.FORCE_GC,
                message=f"Garbage collection failed: {e}"
            )

    def _stage3_reduce_precision(self, error_context: str, **kwargs) -> MemoryRecoveryResult:
        """Stage 3: Reduce model precision (FP16/mixed precision)"""
        try:
            # This would be implemented based on the specific model being used
            # For now, we'll return a placeholder that indicates this stage is available
            
            self.logger.info("Precision reduction strategy available but not implemented yet")
            
            return MemoryRecoveryResult(
                success=False,  # Set to False until implemented
                stage=RecoveryStage.REDUCE_PRECISION,
                message="Precision reduction not yet implemented"
            )
            
        except Exception as e:
            return MemoryRecoveryResult(
                success=False,
                stage=RecoveryStage.REDUCE_PRECISION,
                message=f"Precision reduction failed: {e}"
            )

    def _stage4_cpu_fallback(self, error_context: str, **kwargs) -> MemoryRecoveryResult:
        """Stage 4: Switch to CPU inference"""
        try:
            # This would switch the model to CPU for the current operation
            # Implementation depends on the specific model architecture
            
            self.logger.info("CPU fallback strategy available but not implemented yet")
            
            return MemoryRecoveryResult(
                success=False,  # Set to False until implemented
                stage=RecoveryStage.CPU_FALLBACK,
                message="CPU fallback not yet implemented"
            )
            
        except Exception as e:
            return MemoryRecoveryResult(
                success=False,
                stage=RecoveryStage.CPU_FALLBACK,
                message=f"CPU fallback failed: {e}"
            )

    def _stage5_graceful_failure(self, error_context: str, **kwargs) -> MemoryRecoveryResult:
        """Stage 5: Graceful failure with user notification"""
        try:
            # Log the failure and provide user guidance
            self.logger.error(f"Memory recovery failed for: {error_context}")
            
            # Generate user-friendly error message with suggestions
            suggestions = self._generate_user_suggestions()
            
            return MemoryRecoveryResult(
                success=True,  # Success in failing gracefully
                stage=RecoveryStage.GRACEFUL_FAILURE,
                message=f"Graceful failure executed. Suggestions: {suggestions}"
            )
            
        except Exception as e:
            return MemoryRecoveryResult(
                success=False,
                stage=RecoveryStage.GRACEFUL_FAILURE,
                message=f"Even graceful failure failed: {e}"
            )

    def _generate_user_suggestions(self) -> List[str]:
        """Generate user-friendly suggestions for memory issues"""
        suggestions = [
            "Try reducing image resolution",
            "Close other GPU-intensive applications", 
            "Consider using CPU mode for this operation",
            "Restart the application to clear memory leaks"
        ]
        
        # Add specific suggestions based on current memory state
        memory_info = self.memory_manager.get_gpu_memory_info()
        if memory_info.get('utilization', 0) > 0.9:
            suggestions.insert(0, "GPU memory usage is critical (>90%)")
        
        return suggestions

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.memory_manager.get_gpu_memory_info()
            return memory_info.get('used', 0) / (1024 * 1024)  # Convert to MB
        except:
            return 0.0

    def _update_avg_recovery_time(self, recovery_time: float):
        """Update average recovery time"""
        current_avg = self.recovery_stats['avg_recovery_time']
        successful_count = self.recovery_stats['successful_recoveries']
        
        if successful_count <= 0:
            # Handle edge case where successful_count is 0 or negative
            return
        elif successful_count == 1:
            self.recovery_stats['avg_recovery_time'] = recovery_time
        else:
            # Incremental average calculation
            self.recovery_stats['avg_recovery_time'] = (
                (current_avg * (successful_count - 1) + recovery_time) / successful_count
            )

    def get_recovery_statistics(self) -> Dict:
        """Get comprehensive recovery statistics"""
        total_attempts = self.recovery_stats['total_attempts']
        success_rate = (
            self.recovery_stats['successful_recoveries'] / total_attempts * 100
            if total_attempts > 0 else 0
        )
        
        return {
            **self.recovery_stats,
            'success_rate_percent': success_rate,
            'most_used_stage': max(
                self.recovery_stats['stage_usage'].items(),
                key=lambda x: x[1]
            )[0] if any(self.recovery_stats['stage_usage'].values()) else None
        }

    def reset_statistics(self):
        """Reset recovery statistics"""
        for key in self.recovery_stats:
            if isinstance(self.recovery_stats[key], dict):
                for subkey in self.recovery_stats[key]:
                    self.recovery_stats[key][subkey] = 0
            else:
                self.recovery_stats[key] = 0 