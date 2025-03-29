from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Dict
import torch
from .memory_manager import GPUMemoryManager
import logging 

class BaseSAMPredictor(ABC):
    """Abstract base class defining interface for SAM predictors with memory management."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_manager = GPUMemoryManager()
        
        # Cache settings
        self.current_image_hash = None
        self.prediction_cache = {}
        self.max_cache_size = 50
    
    @abstractmethod
    def initialize(self, checkpoint_path: str) -> None:
        """Initialize the model with given weights."""
        pass
        
    @abstractmethod
    def set_image(self, image: np.ndarray) -> None:
        """Set the image for subsequent predictions."""
        pass
        
    @abstractmethod
    def predict(self, 
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                multimask_output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mask prediction based on prompts."""
        pass
        
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
    
    def optimize_memory(self, force: bool = False) -> None:
        """Optimize memory usage."""
        self.memory_manager.optimize_memory(force)