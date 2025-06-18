# src/core/memory_manager.py

import os
import torch
import logging
from typing import Dict, Tuple, Optional
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import the recovery system
from .memory_recovery import MemoryRecoveryManager, MemoryRecoveryResult
from .recovery_status import get_recovery_reporter

class GPUMemoryManager:
    """Enhanced GPU memory manager with fallback options - Singleton pattern."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GPUMemoryManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Prevent multiple initialization
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from environment variables with defaults
        self.memory_fraction = self._get_env_float('SAM_GPU_MEMORY_FRACTION', 0.9)
        self.warning_threshold = self._get_env_float('SAM_MEMORY_WARNING_THRESHOLD', 0.8)
        self.critical_threshold = self._get_env_float('SAM_MEMORY_CRITICAL_THRESHOLD', 0.95)
        self.enable_memory_growth = self._get_env_bool('SAM_ENABLE_MEMORY_GROWTH', True)
        
        # Memory statistics tracking
        self.memory_stats = {
            'peak_usage': 0,
            'optimization_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_warnings': 0
        }
        
        # Try to initialize NVIDIA SMI, but don't fail if it's not available
        self.nvml_initialized = False
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            self.nvidia_smi = nvidia_smi
            self.nvml_initialized = True
            self.logger.info("NVIDIA SMI initialized successfully")
        except ImportError:
            self.logger.warning("nvidia-smi module not available, using torch memory management")
        except Exception as e:
            self.logger.warning(f"Could not initialize NVIDIA SMI: {e}")
            
        # Setup GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self._setup_gpu()
        else:
            self.logger.warning("Running on CPU - memory management will be limited")
        
        # Initialize recovery system
        self.recovery_manager = MemoryRecoveryManager(self)
            
        self._initialized = True

    def _get_env_float(self, name: str, default: float) -> float:
        """Get float value from environment variable with validation."""
        try:
            value = float(os.getenv(name, default))
            if 0.0 <= value <= 1.0:
                return value
            self.logger.warning(f"Invalid value for {name}: {value}. Using default: {default}")
            return default
        except ValueError:
            self.logger.warning(f"Could not parse {name}. Using default: {default}")
            return default

    def _get_env_bool(self, name: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        return os.getenv(name, str(default)).lower() in ('true', '1', 'yes')

    def _setup_gpu(self) -> None:
        """Configure GPU memory management."""
        try:
            if self.enable_memory_growth:
                # Enable memory growth
                for device in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(self.memory_fraction, device)
            
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.logger.info(f"GPU memory management configured with {self.memory_fraction*100}% limit")
            
        except Exception as e:
            self.logger.error(f"Error setting up GPU memory management: {e}")
            raise
        
        
    def _format_memory_info(self, info: dict) -> str:
        """Format memory info into human readable string."""
        used = self._format_bytes(info['used'])
        total = self._format_bytes(info['total'])
        percentage = info['utilization'] * 100
        return f"Used: {used} / Total: {total} ({percentage:.1f}%)"
    
    @staticmethod
    def _format_bytes(bytes_num: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_num < 1024:
                return f"{bytes_num:.2f} {unit}"
            bytes_num /= 1024
        return f"{bytes_num:.2f} PB"

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get detailed GPU memory information."""
        try:
            if self.device.type != 'cuda':
                # For CPU, return a dictionary with default values AND the formatted key
                memory_info = {'used': 0, 'total': 0, 'utilization': 0}
                memory_info['formatted'] = "Running on CPU - memory stats not available"
                return memory_info
                
            if self.nvml_initialized:
                # Use NVIDIA SMI if available
                handle = self.nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                info = self.nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                
                memory_info = {
                    'used': info.used,
                    'total': info.total,
                    'utilization': info.used / info.total
                }
            else:
                # Fallback to torch memory management
                used = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                memory_info = {
                    'used': used,
                    'total': total,
                    'utilization': used / total
                }
            
            # Add formatted string to the dictionary
            memory_info['formatted'] = self._format_memory_info(memory_info)
            return memory_info
                
        except Exception as e:
            self.logger.error(f"Error getting GPU memory info: {e}")
            # Even in case of exception, ensure formatted key is present
            return {'used': 0, 'total': 0, 'utilization': 0, 'formatted': 'Error getting memory info'}

    def check_memory_status(self) -> Tuple[bool, Optional[str]]:
        """Check memory status and return warning if needed."""
        if self.device.type != 'cuda':
            return True, None
            
        try:
            memory_info = self.get_gpu_memory_info()
            utilization = memory_info['utilization'] 
            
            if utilization >= self.critical_threshold:
                return False, f"Critical GPU memory usage: {utilization*100:.1f}%"
            elif utilization >= self.warning_threshold:
                return True, f"High GPU memory usage: {utilization*100:.1f}%"
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error checking memory status: {e}")
            return False, "Could not check memory status"

    def optimize_memory(self, force: bool = False) -> None:
        """Optimize GPU memory usage."""
        if self.device.type != 'cuda':
            return
            
        try:
            memory_info = self.get_gpu_memory_info()
            if force or memory_info['utilization'] >= self.warning_threshold:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Update statistics
                self.update_stats('optimization_count')
                
                self.logger.info("Memory optimization performed")
                
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
            
            
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system RAM usage information."""
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, returning default system memory info")
            return {
                'total': 0,
                'available': 0, 
                'used': 0,
                'utilization': 0
            }
            
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'utilization': memory.percent / 100.0
            }
        except Exception as e:
            self.logger.error(f"Error getting system memory info: {str(e)}")
            return {'total': 0, 'available': 0, 'used': 0, 'utilization': 0}
            
    def should_cache(self) -> bool:
        """Determine if it's safe to cache based on current memory usage."""
        gpu_memory = self.get_gpu_memory_info()
        system_memory = self.get_system_memory_info()
        
        return (gpu_memory['utilization'] < self.warning_threshold and 
                system_memory['utilization'] < 0.90)  # 90% RAM threshold

    def safe_get_memory_info(self) -> Dict[str, float]:
        """
        Safely get memory information with guaranteed 'formatted' key.
        This method is designed to never fail and always return a valid dictionary.
        """
        try:
            memory_info = self.get_gpu_memory_info()
            # Double-check that formatted key exists
            if 'formatted' not in memory_info:
                memory_info['formatted'] = "Memory stats not available"
            return memory_info
        except Exception as e:
            # Return a safe default in case of any error
            self.logger.error(f"Error in safe_get_memory_info: {e}")
            return {'used': 0, 'total': 0, 'utilization': 0, 'formatted': 'Memory stats not available (error)'}

    def get_memory_analytics(self) -> Dict[str, any]:
        """Get comprehensive memory analytics."""
        gpu_info = self.get_gpu_memory_info()
        system_info = self.get_system_memory_info()
        
        # Get recovery status
        status_reporter = get_recovery_reporter()
        
        return {
            'gpu_memory': gpu_info,
            'system_memory': system_info,
            'statistics': self.memory_stats.copy(),
            'recovery_statistics': self.get_recovery_statistics(),
            'recovery_status': status_reporter.get_status_summary(),
            'configuration': {
                'memory_fraction': self.memory_fraction,
                'warning_threshold': self.warning_threshold,
                'critical_threshold': self.critical_threshold,
                'enable_memory_growth': self.enable_memory_growth
            },
            'recommendations': self._generate_recommendations(gpu_info, system_info)
        }
        
    def _generate_recommendations(self, gpu_info: dict, system_info: dict) -> list:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # GPU memory recommendations
        if gpu_info['utilization'] > 0.9:
            recommendations.append("Consider reducing batch size or model resolution")
        elif gpu_info['utilization'] > 0.8:
            recommendations.append("Monitor memory usage closely - approaching limits")
            
        # System memory recommendations
        if system_info['utilization'] > 0.9:
            recommendations.append("System RAM usage is high - consider closing other applications")
            
        # Cache recommendations
        if self.memory_stats['cache_hits'] > 0:
            hit_rate = self.memory_stats['cache_hits'] / (self.memory_stats['cache_hits'] + self.memory_stats['cache_misses'])
            if hit_rate < 0.5:
                recommendations.append("Low cache hit rate - consider adjusting cache size")
                
        # Optimization frequency
        if self.memory_stats['optimization_count'] > 10:
            recommendations.append("Frequent memory optimizations - consider increasing memory limits")
            
        return recommendations
        
    def profile_memory_operation(self, operation_name: str):
        """Context manager for profiling memory operations."""
        return MemoryProfiler(self, operation_name)
        
    def update_stats(self, stat_name: str, increment: int = 1):
        """Update memory statistics."""
        if stat_name in self.memory_stats:
            self.memory_stats[stat_name] += increment
            
        # Track peak usage
        if stat_name == 'peak_usage':
            current_usage = self.get_gpu_memory_info()['utilization']
            self.memory_stats['peak_usage'] = max(self.memory_stats['peak_usage'], current_usage)

    def execute_with_recovery(self, operation_callback, error_context: str, **operation_kwargs) -> tuple:
        """
        Execute an operation with automatic memory recovery on failure.
        
        Args:
            operation_callback: Function to execute
            error_context: Description of the operation for logging
            **operation_kwargs: Arguments to pass to the operation
            
        Returns:
            tuple: (success: bool, result: any, recovery_result: MemoryRecoveryResult or None)
        """
        try:
            # Try the operation normally first
            result = operation_callback(**operation_kwargs)
            return True, result, None
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                self.logger.warning(f"Memory error detected: {e}")
                
                # Attempt recovery
                recovery_result = self.recovery_manager.recover_from_oom(
                    error_context=error_context,
                    operation_callback=operation_callback,
                    **operation_kwargs
                )
                
                if recovery_result.success and "Operation retry succeeded" in recovery_result.message:
                    # Recovery was successful and operation completed
                    return True, None, recovery_result  # Result would be captured in recovery
                else:
                    # Recovery failed or operation wasn't retried
                    return False, None, recovery_result
            else:
                # Not a memory error, re-raise
                raise
                
        except Exception as e:
            # Non-memory related error
            self.logger.error(f"Non-memory error in {error_context}: {e}")
            raise

    def get_recovery_statistics(self) -> Dict:
        """Get memory recovery statistics."""
        return self.recovery_manager.get_recovery_statistics()

    def __del__(self):
        """Cleanup NVIDIA SMI"""
        if self.nvml_initialized:
            try:
                self.nvidia_smi.nvmlShutdown()
            except:
                pass

class MemoryProfiler:
    """Context manager for profiling memory operations."""
    
    def __init__(self, memory_manager: GPUMemoryManager, operation_name: str):
        self.memory_manager = memory_manager
        self.operation_name = operation_name
        self.start_memory = None
        self.end_memory = None
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        self.start_memory = self.memory_manager.get_gpu_memory_info()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = self.memory_manager.get_gpu_memory_info()
        
        memory_delta = self.end_memory['used'] - self.start_memory['used']
        memory_delta_mb = memory_delta / (1024 * 1024)
        
        self.logger.info(f"Memory profile [{self.operation_name}]: "
                        f"Delta: {memory_delta_mb:.2f}MB, "
                        f"Final: {self.end_memory['utilization']:.1%}")
        
        # Update peak usage
        self.memory_manager.update_stats('peak_usage')