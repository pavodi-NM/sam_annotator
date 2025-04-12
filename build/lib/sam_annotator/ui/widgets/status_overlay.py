from typing import Optional, Dict
import cv2
import numpy as np

class StatusOverlay:
    """Widget for displaying status information overlay on the main window."""
    
    def __init__(self, font_scale: float = 0.6, padding: int = 10):
        """
        Initialize the status overlay widget.
        
        Args:
            font_scale (float): Scale factor for font size
            padding (int): Padding from window edges
        """
        self.font_scale = font_scale
        self.padding = padding
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bg_color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        
    def _add_text_with_background(self, 
                                image: np.ndarray,
                                text: str,
                                position: tuple,  
                                alpha: float = 0.5) -> np.ndarray:
        """Add text with semi-transparent background."""
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, 1)
        
        # Create background rectangle
        padding = 5
        p1 = (position[0], position[1] - text_height - padding)
        p2 = (position[0] + text_width + padding * 2, position[1] + padding)
        
        # Create overlay
        overlay = image.copy()
        cv2.rectangle(overlay, p1, p2, self.bg_color, -1)
        
        # Add text
        cv2.putText(overlay, text, 
                    (position[0] + padding, position[1]), 
                    self.font, self.font_scale, self.text_color, 1)
        
        # Blend with original
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    def render(self,
              image: np.ndarray,
              status: str,
              current_class: str,
              current_class_id: int,
              current_image_path: str,
              current_idx: int,
              total_images: int,
              num_annotations: int,
              annotation_mode: str = "box") -> np.ndarray:
        """
        Render the status overlay on the image.
        
        Args:
            image: Image to render overlay on
            status: Current status message
            current_class: Name of currently selected class
            current_class_id: ID of currently selected class
            current_image_path: Path of current image
            current_idx: Current image index
            total_images: Total number of images
            num_annotations: Number of annotations on current image
            annotation_mode: Current annotation mode ("box" or "point")
            
        Returns:
            Image with rendered overlay
        """
        height, width = image.shape[:2]
        result = image.copy()
        
        # Annotation mode
        result = self._add_text_with_background(
            result,
            f"Mode: {annotation_mode.capitalize()}",
            (self.padding, 30)
        )
        
        # Status line
        if status:
            result = self._add_text_with_background(
                result,
                f"Status: {status}",
                (self.padding, height - 60)
            )
        
        # Class info
        result = self._add_text_with_background(
            result,
            f"Class: {current_class} (ID: {current_class_id})",
            (self.padding, height - 30)
        )
        
        # Image info
        image_info = f"Image {current_idx + 1}/{total_images}"
        result = self._add_text_with_background(
            result,
            image_info,
            (width - 150, height - 30)
        )
        
        # Annotation count
        result = self._add_text_with_background(
            result,
            f"Annotations: {num_annotations}",
            (width - 150, height - 60)
        )
        
        return result