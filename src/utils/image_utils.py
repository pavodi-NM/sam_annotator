# Standard library imports
from typing import Tuple, Dict, Optional, Union, List
import logging

# Third-party imports
import cv2
import numpy as np

# Local imports
from .scaling import ScalingManager, InterpolationMethod  # Import from scaling.py


# class ImageProcessor:
#     """Handles image preprocessing and size management for annotation."""
    
#     def __init__(self, target_size: int = 1024, min_size: int = 600):
#         """
#         Initialize image processor.
        
#         Args:
#             target_size: Maximum dimension for the longer side
#             min_size: Minimum dimension for the shorter side
#         """
#         self.target_size = target_size
#         self.min_size = min_size
#         self.logger = logging.getLogger(__name__)
        
#         # Track original and display sizes
#         self.original_size: Optional[Tuple[int, int]] = None
#         self.display_size: Optional[Tuple[int, int]] = None
#         self.scale_factor: float = 1.0
        
#     def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
#         """
#         Process image for annotation display. 
        
#         Args:
#             image: Input image array, Handle images
            
#         Returns:
#             Tuple of (processed_image, metadata)
#         """
#         try:
#             # Store original size
#             height, width = image.shape[:2]
#             self.original_size = (width, height)
            
#             # Calculate target size maintaining aspect ratio
#             if width > height:
#                 if width > self.target_size:
#                     new_width = self.target_size
#                     new_height = int(height * (self.target_size / width))
#                 else:
#                     return image, self._create_metadata(1.0, (width, height))
#             else:
#                 if height > self.target_size:
#                     new_height = self.target_size
#                     new_width = int(width * (self.target_size / height))
#                 else:
#                     return image, self._create_metadata(1.0, (width, height))
            
#             # Ensure minimum size
#             if new_width < self.min_size:
#                 scale = self.min_size / new_width
#                 new_width = self.min_size
#                 new_height = int(new_height * scale)
#             elif new_height < self.min_size:
#                 scale = self.min_size / new_height
#                 new_height = self.min_size
#                 new_width = int(new_width * scale)
            
#             # Calculate scale factor
#             self.scale_factor = new_width / width
#             self.display_size = (new_width, new_height)
            
#             # Resize image
#             processed_image = cv2.resize(image, (new_width, new_height), 
#                                       interpolation=cv2.INTER_AREA)
            
#             metadata = self._create_metadata(self.scale_factor, (new_width, new_height))
            
#             self.logger.info(f"Processed image from {(width, height)} to {(new_width, new_height)}")
#             self.logger.debug(f"Scale factor: {self.scale_factor}")
            
#             return processed_image, metadata
            
#         except Exception as e:
#             self.logger.error(f"Error processing image: {str(e)}")
#             raise
            
#     def _create_metadata(self, scale: float, size: Tuple[int, int]) -> Dict:
#         """Create metadata dictionary."""
#         return {
#             'original_size': self.original_size,
#             'display_size': size,
#             'scale_factor': scale
#         }
    
#     def scale_coordinates_to_original(self, x: float, y: float) -> Tuple[float, float]:
#         """Convert display coordinates to original image coordinates."""
#         if not self.scale_factor:
#             return x, y
#         return x / self.scale_factor, y / self.scale_factor
    
#     def scale_coordinates_to_display(self, x: float, y: float) -> Tuple[float, float]:
#         """Convert original image coordinates to display coordinates."""
#         return x * self.scale_factor, y * self.scale_factor
    
#     def scale_box_to_original(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         """Convert display box coordinates to original image coordinates."""
#         x1, y1, x2, y2 = box
#         x1_orig, y1_orig = self.scale_coordinates_to_original(x1, y1)
#         x2_orig, y2_orig = self.scale_coordinates_to_original(x2, y2)
#         return (int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig))
    
#     def scale_box_to_display(self, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
#         """Convert original box coordinates to display coordinates."""
#         x1, y1, x2, y2 = box
#         x1_disp, y1_disp = self.scale_coordinates_to_display(x1, y1)
#         x2_disp, y2_disp = self.scale_coordinates_to_display(x2, y2)
#         return (int(x1_disp), int(y1_disp), int(x2_disp), int(y2_disp))
    
#     def scale_contour_to_original(self, contour: np.ndarray) -> np.ndarray:
#         """Convert display contour points to original image coordinates."""
#         return contour / self.scale_factor
    
#     def scale_contour_to_display(self, contour: np.ndarray) -> np.ndarray:


class ImageProcessor:
    """Handles image preprocessing and size management for annotation."""
    
    def __init__(self, target_size: int = 1024, min_size: int = 600):
        """Initialize image processor with ScalingManager."""
        self.scaling_manager = ScalingManager(
            target_size=target_size,
            min_size=min_size,
            maintain_aspect_ratio=True
        )
        self.logger = logging.getLogger(__name__)
        
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process image for annotation display using ScalingManager.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        try:
            # Process image using ScalingManager
            processed_image, metadata = self.scaling_manager.process_image(
                image, 
                interpolation=InterpolationMethod.AREA
            )
            
            self.logger.info(f"Processed image from {metadata['original_size']} "
                           f"to {metadata['display_size']}")
            self.logger.debug(f"Scale factors: {metadata['scale_factors']}")
            
            return processed_image, metadata
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise
    
    def scale_to_original(self, 
                         coords: Union[np.ndarray, List[Tuple[int, int]]],
                         coord_type: str = 'point') -> np.ndarray:
        """Wrapper for scaling to original space."""
        return self.scaling_manager.to_original_space(coords, coord_type)
    
    def scale_to_display(self,
                        coords: Union[np.ndarray, List[Tuple[int, int]]],
                        coord_type: str = 'point') -> np.ndarray:
        """Wrapper for scaling to display space."""
        return self.scaling_manager.to_display_space(coords, coord_type)
        """Convert original contour points to display coordinates."""
        return contour * self.scale_factor