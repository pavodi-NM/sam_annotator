from typing import Optional, Tuple, List, Dict
import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import logging
import hashlib
from functools import lru_cache
import gc
import psutil

class GPUMemoryManager:
    """Manages GPU memory allocation and optimization."""
    
    def __init__(self, warning_threshold: float = 0.85, critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = logging.getLogger(__name__)
        
        # Track consecutive warnings
        self.warning_count = 0
        self.max_warnings = 3
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get detailed GPU memory information."""
        if not torch.cuda.is_available():
            return {'used': 0, 'total': 0, 'utilization': 0}
            
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
            return {'used': 0, 'total': 0, 'utilization': 0}
            
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
            
class SAMPredictor:
    """Enhanced wrapper for SAM prediction with advanced memory management."""
    
    def __init__(self, model_type: str = "vit_h"):
        self.model_type = model_type
        self.predictor = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory manager
        self.memory_manager = GPUMemoryManager()
        
        # Cache settings
        self.current_image_embedding = None
        self.current_image_hash = None
        self.prediction_cache = {}
        self.max_cache_size = 50
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            self.logger.warning("Running on CPU. Performance may be limited.")
            
    def initialize(self, checkpoint_path: str) -> None:
        """Initialize the SAM model with memory optimizations."""
        try:
            # Check available GPU memory before loading
            if self.device.type == 'cuda':
                memory_info = self.memory_manager.get_gpu_memory_info()
                available_memory = memory_info['total'] - memory_info['used']
                
                # Estimate if we have enough memory (SAM typically needs ~10GB)
                if available_memory < 10 * (1024 ** 3):  # 10GB in bytes
                    self.logger.warning("Limited GPU memory available. Performance may be affected.")
            
            # Load model with memory optimizations
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            
            # Move model to device and optimize
            sam.to(device=self.device)
            
            if self.device.type == 'cuda':
                # Enable memory optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            
            # Initialize predictor
            self.predictor = SamPredictor(sam)
            self.logger.info(f"Initialized SAM model on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing SAM model: {str(e)}")
            raise
        
        
    def set_image(self, image: np.ndarray) -> None:
        """Set image with embedding caching."""
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
            
        try:
            # Calculate image hash
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            
            # Only compute new embedding if image changed
            if image_hash != self.current_image_hash:
                with torch.no_grad():
                    self.predictor.set_image(image)
                    self.current_image_hash = image_hash
                    
                # Clear prediction cache for new image
                self.prediction_cache.clear()
                
                # Check memory status after setting image
                status_ok, message = self.memory_manager.check_memory_status()
                if not status_ok:
                    self.logger.warning(message)
                    self.memory_manager.optimize_memory()
                    
        except Exception as e:
            self.logger.error(f"Error setting image: {str(e)}")
            raise
            
    def predict(self,
               point_coords: Optional[np.ndarray] = None,
               point_labels: Optional[np.ndarray] = None,
               box: Optional[np.ndarray] = None,
               multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks with memory management."""
        if self.predictor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")
            
        try:
            # Check memory status before prediction
            status_ok, message = self.memory_manager.check_memory_status()
            if not status_ok:
                self.logger.error(message)
                raise RuntimeError(message)
                
            # Generate cache key and check cache
            cache_key = self._generate_cache_key(point_coords, point_labels, box)
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            # Run prediction with optimizations
            with torch.no_grad():
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=multimask_output
                )
                
                # Cache results if memory allows
                memory_info = self.memory_manager.get_gpu_memory_info()
                if memory_info['utilization'] < self.memory_manager.warning_threshold:
                    self.prediction_cache[cache_key] = (masks, scores, logits)
                    
                    # Manage cache size
                    if len(self.prediction_cache) > self.max_cache_size:
                        self.clear_cache(keep_current=True)
                
                return masks, scores, logits
                
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            # Try to recover memory
            self.memory_manager.optimize_memory(force=True)
            raise
            
    def clear_cache(self, keep_current: bool = False) -> None:
        """Clear prediction cache with option to keep current image."""
        if keep_current and self.current_image_hash:
            current_predictions = {k: v for k, v in self.prediction_cache.items()
                                if k.startswith(str(self.current_image_hash))}
            self.prediction_cache.clear()
            self.prediction_cache.update(current_predictions)
        else:
            self.prediction_cache.clear()
            
        # Force memory optimization
        self.memory_manager.optimize_memory(force=True)
        
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage ratio."""
        return self.memory_manager.get_gpu_memory_info()['utilization']
        
    def _generate_cache_key(self,
                          point_coords: Optional[np.ndarray],
                          point_labels: Optional[np.ndarray],
                          box: Optional[np.ndarray]) -> str:
        """Generate cache key for prediction inputs."""
        key_parts = [str(self.current_image_hash)]
        
        if point_coords is not None:
            key_parts.append(hashlib.md5(point_coords.tobytes()).hexdigest())
        if point_labels is not None:
            key_parts.append(hashlib.md5(point_labels.tobytes()).hexdigest())
        if box is not None:
            key_parts.append(hashlib.md5(box.tobytes()).hexdigest())
            
        return "_".join(key_parts)