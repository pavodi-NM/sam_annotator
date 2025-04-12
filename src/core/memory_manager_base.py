import torch
import logging
import gc
import psutil
from typing import Dict, Tuple

class GPUMemoryManager:
    """Manages GPU memory allocation and optimization."""
    
    def __init__(self, warning_threshold: float = 0.85, critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
        
        # Track consecutive warnings
        self.warning_count = 0
        self.max_warnings = 3
        
        # Initialize CUDA optimizations if available
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get detailed GPU memory information."""
        if not torch.cuda.is_available():
            return {'used': 0, 'total': 0, 'reserved': 0, 'utilization': 0}
            
        try:
            gpu_memory = torch.cuda.memory_stats()
            allocated = gpu_memory.get('allocated_bytes.all.current', 0)
            reserved = gpu_memory.get('reserved_bytes.all.current', 0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                'used': allocated,
                'reserved': reserved,
                'total': total,
                'utilization': allocated / total
            }
        except Exception as e:
            self.logger.error(f"Error getting GPU memory info: {str(e)}")
            return {'used': 0, 'total': 0, 'reserved': 0, 'utilization': 0}
            
    def check_memory_status(self) -> Tuple[bool, str]:
        """Check memory status and return status with message."""
        memory_info = self.get_gpu_memory_info()
        utilization = memory_info['utilization']
        
        if utilization > self.critical_threshold:
            self.warning_count += 1
            if self.warning_count >= self.max_warnings:
                return False, "CRITICAL: GPU memory usage exceeded safe limits. Stopping operation."
            return False, f"WARNING: Very high GPU memory usage ({utilization:.2%})"
            
        if utilization > self.warning_threshold:
            self.warning_count += 1
            return True, f"WARNING: High GPU memory usage ({utilization:.2%})"
            
        # Reset warning count if memory usage is normal
        self.warning_count = 0
        return True, "OK"
        
    def optimize_memory(self, force: bool = False) -> None:
        """Optimize GPU memory usage."""
        if not torch.cuda.is_available():
            return
            
        memory_info = self.get_gpu_memory_info()
        
        if force or memory_info['utilization'] > self.warning_threshold:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Run garbage collection
            gc.collect()
            
            # Reset warning count after optimization
            self.warning_count = 0
            
            self.logger.info("Memory optimization performed")
    
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