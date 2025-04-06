# src/core/memory_manager.py

import os
import torch
import logging
from typing import Dict, Tuple, Optional

class GPUMemoryManager:
    """Enhanced GPU memory manager with fallback options."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from environment variables with defaults
        self.memory_fraction = self._get_env_float('SAM_GPU_MEMORY_FRACTION', 0.9)
        self.warning_threshold = self._get_env_float('SAM_MEMORY_WARNING_THRESHOLD', 0.8)
        self.critical_threshold = self._get_env_float('SAM_MEMORY_CRITICAL_THRESHOLD', 0.95)
        self.enable_memory_growth = self._get_env_bool('SAM_ENABLE_MEMORY_GROWTH', True)
        
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
                
                self.logger.info("Memory optimization performed")
                
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {e}")
            
            
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system RAM usage information."""
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

    def __del__(self):
        """Cleanup NVIDIA SMI"""
        if self.nvml_initialized:
            try:
                self.nvidia_smi.nvmlShutdown()
            except:
                pass