from dataclasses import dataclass
import numpy as np
import cv2
from typing import Tuple, Dict, Optional, Union, List
import logging
from enum import Enum

class InterpolationMethod(Enum):
    """Supported interpolation methods for resizing."""
    NEAREST = cv2.INTER_NEAREST  # Best for masks - preserves hard edges
    BILINEAR = cv2.INTER_LINEAR  # Good for natural images
    AREA = cv2.INTER_AREA       # Best for downscaling
    CUBIC = cv2.INTER_CUBIC     # Good for upscaling natural images

@dataclass
class ImageDimensions:
    """Store image dimensions and scaling information."""
    width: int
    height: int
    scale_factor_x: float = 1.0
    scale_factor_y: float = 1.0
    aspect_ratio: float = 1.0

    def __post_init__(self):
        self.aspect_ratio = self.width / self.height if self.height != 0 else 1.0

class ScalingManager:
    """Manages consistent scaling operations for images, masks, and coordinates."""
    
    def __init__(self, 
                 target_size: int = 1024,
                 min_size: int = 600,
                 maintain_aspect_ratio: bool = True,
                 max_size: Optional[int] = None):
        """Initialize scaling manager with desired parameters.
        
        Args:
            target_size: Target size for the longer edge
            min_size: Minimum size for the shorter edge
            maintain_aspect_ratio: Whether to maintain aspect ratio during resizing
            max_size: Maximum allowed dimension (optional)
        """
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size or target_size * 2
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.original_dims: Optional[ImageDimensions] = None
        self.display_dims: Optional[ImageDimensions] = None
        self.logger = logging.getLogger(__name__)

    def set_image_dimensions(self, image: np.ndarray) -> Tuple[int, int]:
        """Calculate and store dimensions for both original and display spaces.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (display_width, display_height)
        """
        orig_h, orig_w = image.shape[:2]
        self.original_dims = ImageDimensions(orig_w, orig_h)
        
        if self.maintain_aspect_ratio:
            # Scale based on the longer edge while maintaining aspect ratio
            scale = min(self.target_size / max(orig_w, orig_h),
                       self.min_size / min(orig_w, orig_h))
            
            # Ensure we don't exceed max_size
            if max(orig_w * scale, orig_h * scale) > self.max_size:
                scale = self.max_size / max(orig_w, orig_h)
                
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
        else:
            new_w = self.target_size
            new_h = self.target_size
            
        self.display_dims = ImageDimensions(
            width=new_w,
            height=new_h,
            scale_factor_x=new_w / orig_w,
            scale_factor_y=new_h / orig_h,
            aspect_ratio=new_w / new_h
        )
        
        return new_w, new_h

    def to_original_space(self, 
                         coords: Union[np.ndarray, List[Tuple[int, int]]],
                         coord_type: str = 'point') -> np.ndarray:
        """Convert display coordinates to original image space.
        
        Args:
            coords: Coordinates to convert
            coord_type: Type of coordinates ('point', 'box', 'mask', or 'contour')
            
        Returns:
            Converted coordinates in original space
        """
        if self.original_dims is None or self.display_dims is None:
            raise ValueError("Dimensions not initialized. Call set_image_dimensions first.")
        
        coords = np.array(coords)
        if coord_type == 'point':
            return coords / [self.display_dims.scale_factor_x, 
                           self.display_dims.scale_factor_y]
        elif coord_type == 'box':
            # For boxes: [x1, y1, x2, y2] or [x, y, w, h]
            return coords / [self.display_dims.scale_factor_x,
                           self.display_dims.scale_factor_y,
                           self.display_dims.scale_factor_x,
                           self.display_dims.scale_factor_y]
        elif coord_type == 'mask':
            return cv2.resize(coords.astype(np.uint8),
                            (self.original_dims.width, self.original_dims.height),
                            interpolation=InterpolationMethod.NEAREST.value)
        elif coord_type == 'contour':
            # For contours: array of points
            scale_matrix = np.array([1/self.display_dims.scale_factor_x,
                                   1/self.display_dims.scale_factor_y])
            return coords * scale_matrix
        else:
            raise ValueError(f"Unknown coordinate type: {coord_type}")

    def to_display_space(self,
                        coords: Union[np.ndarray, List[Tuple[int, int]]],
                        coord_type: str = 'point') -> np.ndarray:
        """Convert original coordinates to display space.
        
        Args:
            coords: Coordinates to convert
            coord_type: Type of coordinates ('point', 'box', 'mask', or 'contour')
            
        Returns:
            Converted coordinates in display space
        """
        if self.original_dims is None or self.display_dims is None:
            raise ValueError("Dimensions not initialized. Call set_image_dimensions first.")
        
        coords = np.array(coords)
        if coord_type == 'point':
            return coords * [self.display_dims.scale_factor_x,
                           self.display_dims.scale_factor_y]
        elif coord_type == 'box':
            return coords * [self.display_dims.scale_factor_x,
                           self.display_dims.scale_factor_y,
                           self.display_dims.scale_factor_x,
                           self.display_dims.scale_factor_y]
        elif coord_type == 'mask':
            return cv2.resize(coords.astype(np.uint8),
                            (self.display_dims.width, self.display_dims.height),
                            interpolation=InterpolationMethod.NEAREST.value)
        elif coord_type == 'contour':
            scale_matrix = np.array([self.display_dims.scale_factor_x,
                                   self.display_dims.scale_factor_y])
            return coords * scale_matrix
        else:
            raise ValueError(f"Unknown coordinate type: {coord_type}")

    def process_image(self, 
                     image: np.ndarray,
                     interpolation: InterpolationMethod = InterpolationMethod.AREA
                     ) -> Tuple[np.ndarray, Dict]:
        """Process image for display while maintaining quality.
        
        Args:
            image: Input image
            interpolation: Interpolation method for resizing
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        new_w, new_h = self.set_image_dimensions(image)
        
        if new_w == image.shape[1] and new_h == image.shape[0]:
            return image, self._create_metadata()
            
        processed = cv2.resize(image, (new_w, new_h),
                             interpolation=interpolation.value)
        
        return processed, self._create_metadata()

    def _create_metadata(self) -> Dict:
        """Create metadata dictionary with scaling information."""
        if self.original_dims is None or self.display_dims is None:
            raise ValueError("Dimensions not initialized")
            
        return {
            'original_size': (self.original_dims.width, self.original_dims.height),
            'display_size': (self.display_dims.width, self.display_dims.height),
            'scale_factors': {
                'x': self.display_dims.scale_factor_x,
                'y': self.display_dims.scale_factor_y
            },
            'aspect_ratio': {
                'original': self.original_dims.aspect_ratio,
                'display': self.display_dims.aspect_ratio
            }
        }

    def validate_dimensions(self, width: int, height: int) -> bool:
        """Validate image dimensions against size constraints."""
        return (self.min_size <= min(width, height) and 
                max(width, height) <= self.max_size)

    def get_optimal_display_size(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate optimal display size maintaining constraints."""
        if width > height:
            scale = min(self.target_size / width, self.min_size / height)
        else:
            scale = min(self.target_size / height, self.min_size / width)
            
        new_w = int(width * scale)
        new_h = int(height * scale)
        
        return new_w, new_h