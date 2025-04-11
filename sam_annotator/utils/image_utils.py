# Standard library imports
from typing import Tuple, Dict, Optional, Union, List
import logging
from functools import lru_cache
import hashlib
from weakref import WeakValueDictionary 

# Third-party imports
import cv2
import numpy as np

# Local imports
from .scaling import ScalingManager, InterpolationMethod

class ImageProcessor:
    """Handles image preprocessing and size management for annotation with caching."""
    
    def __init__(self, target_size: int = 1024, min_size: int = 600):
        """Initialize image processor with ScalingManager and caching."""
        # Initialize the existing ScalingManager
        self.scaling_manager = ScalingManager(
            target_size=target_size,
            min_size=min_size,
            maintain_aspect_ratio=True
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches
        self._processed_cache = WeakValueDictionary()  # Cache processed images
        self._metadata_cache = {}  # Cache metadata
        self.max_cache_size = 10
        
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process image for annotation display using ScalingManager with caching."""
        try:
            # Generate cache key
            image_hash = hashlib.md5(image.tobytes()).hexdigest()
            
            # Check cache first
            if image_hash in self._processed_cache:
                return self._processed_cache[image_hash], self._metadata_cache[image_hash]
            
            # Process image using existing ScalingManager
            processed_image, metadata = self.scaling_manager.process_image(
                image, 
                interpolation=InterpolationMethod.AREA
            )
            
            # Cache the results
            self._processed_cache[image_hash] = processed_image
            self._metadata_cache[image_hash] = metadata
            
            # Manage cache size for metadata
            if len(self._metadata_cache) > self.max_cache_size:
                # Remove oldest items
                oldest_key = next(iter(self._metadata_cache))
                del self._metadata_cache[oldest_key]
            
            self.logger.info(f"Processed image from {metadata['original_size']} "
                           f"to {metadata['display_size']}")
            
            return processed_image, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise
    
    def scale_to_original(self, 
                         coords: Union[np.ndarray, List[Tuple[int, int]], Tuple[int, int]],
                         coord_type: str = 'point',
                         scale_factors: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Scale coordinates to original space using ScalingManager."""
        try:
            # Convert tuple to numpy array if needed
            if isinstance(coords, tuple):
                coords = np.array([coords])
            
            return self.scaling_manager.to_original_space(coords, coord_type)
        except Exception as e:
            self.logger.error(f"Error scaling to original: {str(e)}")
            raise
    
    def scale_to_display(self,
                        coords: Union[np.ndarray, List[Tuple[int, int]], Tuple[int, int]],
                        coord_type: str = 'point',
                        scale_factors: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Scale coordinates to display space using ScalingManager."""
        try:
            # Convert tuple to numpy array if needed
            if isinstance(coords, tuple):
                coords = np.array([coords])
            
            return self.scaling_manager.to_display_space(coords, coord_type)
        except Exception as e:
            self.logger.error(f"Error scaling to display: {str(e)}")
            raise
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._processed_cache.clear()
        self._metadata_cache.clear()
    
    def get_memory_usage(self) -> int:
        """Get approximate memory usage of cached images in bytes."""
        total_bytes = 0
        for img in self._processed_cache.values():
            if isinstance(img, np.ndarray):
                total_bytes += img.nbytes
        return total_bytes